import asyncio
from collections.abc import MutableMapping
import enum
from functools import partial
import gzip
import inspect
import os
from pathlib import Path
import pickle
import socket
from sys import stderr
import sys
import time
import traceback
from typing import Any, Callable, Optional, TypeVar, Union
from typing_extensions import ParamSpec
import setproctitle
import os
from tensorpc.autossh.constants import (TENSORPC_ASYNCSSH_ENV_INIT_INDICATE,
                                        TENSORPC_ASYNCSSH_INIT_SUCCESS, TENSORPC_ASYNCSSH_TASK_HTTP_PORT, TENSORPC_ASYNCSSH_TASK_KEY,
                                        TENSORPC_ASYNCSSH_TASK_PORT,
                                        TENSORPC_SSH_TASK_NAME_PREFIX)
from tensorpc.autossh.core import (LOGGER, CommandEvent, CommandEventType,
                                   EofEvent, Event, EventType, ExceptionEvent, ExternalEvent,
                                   LineEvent, RawEvent, SSHClient, SSHRequest,
                                   SSHRequestType, SubprocessRpcClient)
from tensorpc.autossh.coretypes import TaskWrapperArgs, TaskWrapperWorkerState
from tensorpc.core import marker, prim
from tensorpc.core.asyncclient import simple_remote_call_async
from tensorpc.core.moduleid import get_module_id_of_type, get_object_type_from_module_id, import_dynamic_func
from tensorpc.core.serviceunit import ServiceEventType, run_callback, run_callback_noexcept
import json

from tensorpc.utils.wait_tools import PeriodicTask, get_all_ip_addresses, get_primary_ip
from tensorpc.autossh.serv_names import serv_names
from tensorpc.utils.pyspyutil import get_all_subprocess_traceback_by_pyspy


class SSHEventLogger:

    def __init__(self, outdir: Path, open_mode: str = "w"):
        self.outdir = outdir
        self.jsonl_writer = None
        self._open_mode = open_mode

    def open(self):
        if self.jsonl_writer is None:
            self.jsonl_writer = open(self.outdir, self._open_mode)

    def log(self, metrics: dict[str, Any], compact: bool = False):
        if compact:
            print(json.dumps(metrics, separators=(',', ':')),
                  file=self.jsonl_writer,
                  flush=True)
        else:
            print(json.dumps(metrics), file=self.jsonl_writer, flush=True)

    def close(self):
        if self.jsonl_writer is not None:
            self.jsonl_writer.close()
            self.jsonl_writer = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()


class ExtendSSHEventType(enum.Enum):
    Command = EventType.Command.value
    Raw = EventType.Raw.value
    Line = EventType.Line.value
    Eof = EventType.Eof.value
    Exception = EventType.Exception.value
    External = EventType.External.value

    Init = "I"
    InitInfo = "M"
    TraceBack = "TB"


class TaskWrapper:

    def __init__(self,
                 config: Optional[Union[dict, TaskWrapperArgs]] = None,
                 msg_callback: Optional[Callable[[Event], Any]] = None,
                 default_url: str = "localhost:22",
                 exit_callback: Optional[Callable[[], Any]] = None,
                 key: Optional[str] = None) -> None:
        self.enabled = False
        self._cur_inp_queue = None
        self._is_standalone = key is None
        if not self._is_standalone:
            assert exit_callback is not None, "task managed by TaskManager must have exit_callback"
        self._key_from_manager = key
        if config is not None:
            if isinstance(config, TaskWrapperArgs):
                self.config = config
            else:
                self.config = TaskWrapperArgs(**config)
            self.enabled = True
        else:
            self.config = TaskWrapperArgs.empty()
        self.client = SSHClient(default_url, self.config.username,
                                self.config.password)
        self._ssh_task: Optional[asyncio.Task] = None
        self._cur_inp_queue = asyncio.Queue()
        self._shutdown_ev = asyncio.Event()
        self.init_terminal_size: tuple[int, int] = (80, 40)
        self._ssh_inited = False
        self._ssh_current_cmd = ""
        self._all_events: list[Event] = []
        self._msg_func_task: Optional[asyncio.Task] = None
        self._error_msg_task: Optional[asyncio.Task] = None
        self._msg_handler_func: Optional[Callable[[list[Event], str],
                                                  Any]] = None
        self.event_logger = None
        self._state = TaskWrapperWorkerState("idle", "")
        self._all_worker_addrs: dict[str, TaskWrapperWorkerState] = {}
        self._is_dist_inited: bool = False
        self._prevent_env_log: bool = False
        self._init_info: Any = None
        self._pyspy_period_task: Optional[PeriodicTask] = None

        self._msg_callback = msg_callback
        self._exit_callback = exit_callback
        self._request_pty = self.config.request_pty

    @marker.mark_server_event(event_type=ServiceEventType.Init)
    async def init_async(self):
        if not self.enabled:
            return
        if self.config.log_path is not None:
            # create log dir if not exists
            Path(self.config.log_path).parent.mkdir(parents=True,
                                                    exist_ok=True,
                                                    mode=0o755)
            self.event_logger = SSHEventLogger(Path(self.config.log_path),
                                               open_mode="a")
            self.event_logger.open()
        if self.config.msg_handler is not None:
            self._msg_handler_func = import_dynamic_func(
                self.config.msg_handler, True)
        if self.config.log_path is not None and self.config.pyspy_period is not None:
            if self.config.pyspy_period > 0:
                self._pyspy_period_task = PeriodicTask(
                    self.config.pyspy_period,
                    self._log_pyspy_status,
                    align_ts=True)

        self._cur_inp_queue = asyncio.Queue()
        port = prim.get_server_grpc_port()
        http_port = prim.get_server_http_port()
        await self._cur_inp_queue.put(
            f" export \"{TENSORPC_ASYNCSSH_TASK_PORT}\"={port}\n")
        if http_port > 0:
            await self._cur_inp_queue.put(
                f" export \"{TENSORPC_ASYNCSSH_TASK_HTTP_PORT}\"={http_port}\n")
        if self._key_from_manager is not None:
            await self._cur_inp_queue.put(
                f" export \"{TENSORPC_ASYNCSSH_TASK_KEY}\"={self._key_from_manager}\n")

        await self._cur_inp_queue.put(f" cd {os.getcwd()}\n")
        await self._cur_inp_queue.put(
            f" echo \"{TENSORPC_ASYNCSSH_INIT_SUCCESS}|PID=$TENSORPC_SSH_CURRENT_PID\"\n"
        )
        sd_task = asyncio.create_task(self._shutdown_ev.wait())
        # we still enable raw event here to get last timestamp if event contains progress bar like output
        # that won't produce line event to let us know it's hanging
        os_envs = os.environ
        new_envs: MutableMapping = {}
        for k, v in os_envs.items():
            # remove all vscode envs because we use vscode-style bash injection
            if not k.startswith("VSCODE"):
                new_envs[k] = v
        self._ssh_task = asyncio.create_task(
            self.client.connect_queue(self._cur_inp_queue,
                                      self._handle_ssh_queue,
                                      shutdown_task=sd_task,
                                      request_pty=self._request_pty,
                                      enable_raw_event=True,
                                      env=new_envs))
        setproctitle.setproctitle(f"{TENSORPC_SSH_TASK_NAME_PREFIX}-{port}")
        self_ip = get_primary_ip()
        self_addr = f"{self_ip}:{port}"
        self._state.addr = self_addr
        # init distributed
        if self.config.master_url is not None:
            ip_addr = self.config.master_url.split(":")[0]
            # determine master is self
            found = False
            for (_, addr) in get_all_ip_addresses(socket.AF_INET):
                if ip_addr == addr:
                    found = True
                    self._state.is_master = True
                    self._all_worker_addrs[self._state.addr] = self._state
                    break
            if not found:
                await simple_remote_call_async(
                    self.config.master_url,
                    serv_names.TASK_WRAPPER_DISTRIBUTED_INIT_ADDR,
                    self_addr,
                    rpc_timeout=self.config.init_timeout)
        if self.event_logger is not None:
            self._state.init_timestamp = time.time_ns()
            self.event_logger.log({
                "ts":
                self._ns_to_us(self._state.init_timestamp),
                "type":
                ExtendSSHEventType.Init.value,
                "d": {
                    "url": self_addr,
                    "is_master": self._state.is_master,
                }
            })
        if self.config.init_info_getter is not None and self.event_logger is not None:
            try:
                init_meta_log_func = import_dynamic_func(
                    self.config.init_info_getter, True)
                res = init_meta_log_func()
                if inspect.iscoroutine(res):
                    res = await res
                # check is json dumppable
                json.dumps(res)
                self._init_info = res
                self.event_logger.log({
                    "d":
                    res,
                    "ts":
                    self._get_relative_timestamp_us(),
                    "type":
                    ExtendSSHEventType.InitInfo.value,
                })
            except:
                traceback.print_exc(file=sys.stderr)

    async def change_terminal_size(self, width: int, height: int):
        if not self.enabled:
            return
        if self._cur_inp_queue is not None:
            await self._cur_inp_queue.put(
                SSHRequest(SSHRequestType.ChangeSize, (width, height)))

    def _ns_to_us(self, ns: int):
        return ns // 1000

    def _get_relative_timestamp_us(self):
        return self._ns_to_us(time.time_ns() - self._state.init_timestamp)

    def _get_timestamp_us(self):
        return time.time_ns() // 1000

    @marker.mark_server_event(event_type=ServiceEventType.Exit)
    async def exit_async(self):
        if not self.enabled:
            return
        self._shutdown_ev.set()
        if self.event_logger is not None:
            self.event_logger.close()
        if self._ssh_task is not None:
            await self._ssh_task
            self._ssh_task = None
        if self._pyspy_period_task is not None:
            await self._pyspy_period_task.close()

    async def distributed_set_all_worker_addr(self, all_worker_addrs):
        assert not self._state.is_master, "can only be called in worker"
        self._all_worker_addrs = all_worker_addrs
        if not all_worker_addrs:
            self._is_dist_inited = False

    async def init_set_worker_addr(self, addr: str):
        self._all_worker_addrs[addr] = TaskWrapperWorkerState("idle", addr)
        if len(self._all_worker_addrs) == self.config.num_workers:
            self._is_dist_inited = True
            cancel_tasks = []
            for addr in self._all_worker_addrs:
                if addr == self._state.addr:
                    continue
                cancel_tasks.append(
                    asyncio.create_task(
                        simple_remote_call_async(
                            addr, serv_names.
                            _TASK_WRAPPER_DISTRIBUTED_INIT_WORKER_ADDRS,
                            self._all_worker_addrs)))
            await asyncio.gather(*cancel_tasks)

        elif len(self._all_worker_addrs) > self.config.num_workers:
            # wrong number of workers
            LOGGER.error(
                f"Wrong number of workers: {len(self._all_worker_addrs)}, distributed methods disabled."
            )
            self._all_worker_addrs.clear()
            self._is_dist_inited = False

    async def _run_msg_func_task_func(self, func: Callable, *args, **kwargs):
        try:
            res = func(*args, **kwargs)
            if inspect.iscoroutine(res):
                await res
            await asyncio.sleep(self.config.msg_throttle)
        finally:
            self._msg_func_task = None

    def run_dynamic_func_rate_limited(self, func_id: str, *args,
                                      **kwargs) -> bool:
        if self._msg_func_task is not None:
            return False
        func = get_object_type_from_module_id(func_id)
        assert func is not None, f"func {func_id} not found"
        self._msg_func_task = asyncio.create_task(
            self._run_msg_func_task_func(func, *args, **kwargs))
        return True

    def get_last_active_timestamp(self) -> int:
        return self._state.last_timestamp

    def get_last_active_duration(self) -> int:
        return time.time_ns() - self._state.last_timestamp

    def get_events_pickled(self,
                           lastn: int = -1,
                           do_gzip: bool = True,
                           ignore_line_events: bool = False) -> bytes:
        events = self._all_events
        if lastn > 0:
            events = events[-lastn:]
        if ignore_line_events:
            events = [e for e in events if not isinstance(e, LineEvent)]
        res = pickle.dumps(events)
        if do_gzip:
            return gzip.compress(res)
        return res

    async def log_event(self, event_dict: Any):
        # log event from external
        if self._msg_callback is not None:
            ev = ExternalEvent(time.time_ns(), event_dict)
            _, exc = await run_callback_noexcept(self._msg_callback, ev)
            if exc is not None:
                LOGGER.error(f"Error in msg_callback: {exc[0]}")
                LOGGER.error(exc[1])
        if self.event_logger is not None:
            metric = {}
            metric.update({
                "d": event_dict,
                "ts": self._get_relative_timestamp_us(),
                "type": ExtendSSHEventType.External.value,
            })
            self.event_logger.log(metric)

    async def distributed_cancel_all(self):
        # only master can run this method.
        if not self._is_dist_inited:
            return False
        if self._state.is_master:
            await self._send_ctrl_c()
            cancel_tasks = []
            for addr in self._all_worker_addrs:
                cancel_tasks.append(
                    asyncio.create_task(
                        simple_remote_call_async(
                            addr,
                            serv_names._TASK_WRAPPER_DISTRIBUTED_CANCEL)))
            await asyncio.gather(*cancel_tasks)
            return True
        return False

    async def cancel_ctrl_c(self):
        # only master can run this method.
        if not self._is_dist_inited:
            return False
        if not self._state.is_master:
            await self._send_ctrl_c()
            return True
        return False

    async def run_dynamic_func(self, func_id_or_code: str, is_func_id: bool,
                               *args, **kwargs):
        func = import_dynamic_func(func_id_or_code, is_func_id)
        return run_callback(func, *args, **kwargs)

    async def _send_ctrl_c(self):
        assert self._cur_inp_queue is not None
        # https://github.com/ronf/asyncssh/issues/112#issuecomment-343318916
        return await self._cur_inp_queue.put('\x03')

    async def _send_error_msg_rate_limited(self, return_code: int,
                                           error_lines: list[str]):
        if self.config.error_handler is not None:
            t = time.time()
            try:
                error_handler = import_dynamic_func(self.config.error_handler,
                                                    True)
                res = error_handler(f"Abnormal Exit with code {return_code}",
                                    "".join(error_lines))
                if inspect.iscoroutine(res):
                    await res
            except:
                traceback.print_exc(file=sys.stderr)

            duration = time.time() - t
            sleep_duration = max(0,
                                 self.config.error_handle_throttle - duration)
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)

            self._error_msg_task = None

    async def worker_msg_command_end(self,
                                     addr: str,
                                     return_code: Optional[int] = None,
                                     error_lines: Optional[list[str]] = None):
        if return_code is not None and return_code != 0:
            if error_lines is None:
                error_lines = []
            if self._error_msg_task is None and self.config.error_handler is not None:
                self._error_msg_task = asyncio.create_task(
                    self._send_error_msg_rate_limited(return_code,
                                                      error_lines))
            print(f"Worker {addr} failed with return code: {return_code}.",
                  file=stderr)
            print("".join(error_lines), file=stderr)
            if self.event_logger is not None:
                metric = {
                    "ts": self._get_relative_timestamp_us(),
                    "type": ExtendSSHEventType.Exception.value,
                    "d": {
                        "return_code": return_code,
                        "error_lines": error_lines,
                    }
                }
                self.event_logger.log(metric)

        if return_code is None or return_code == 0:
            print(f"Worker {addr} completed successfully.", file=stderr)

    def _get_last_stderr_lines(self,
                               max_lines: int = 200,
                               max_line_length: int = 100,
                               max_duration_ms: int = 10000) -> list[str]:
        lines = []
        cur_time_us = time.time_ns() // 1000
        for event in self._all_events[::-1]:
            if isinstance(
                    event,
                    LineEvent) and event.is_stderr and not event.is_command:
                # event timestamp: us
                if len(
                        lines
                ) >= max_lines or cur_time_us - event.timestamp > max_duration_ms * 1000:
                    break
                lines.append(event.get_line_str()[:max_line_length])
        return lines[::-1]

    async def _handle_ssh_queue(self, event: Event):
        assert self._cur_inp_queue is not None
        self._state.last_timestamp = event.timestamp
        if self._msg_callback is not None:
            res, exc = await run_callback_noexcept(self._msg_callback, event)
            if exc is not None:
                LOGGER.error(f"Error in msg_callback: {exc[0]}")
                LOGGER.error(exc[1])

        if not isinstance(event, RawEvent):
            if self.event_logger is not None:
                metric = {}
                if isinstance(event, LineEvent):
                    metric["d"] = event.get_line_str()
                elif isinstance(event, CommandEvent):
                    arg_str = event.get_arg_str()
                    # don't log env to file because it may contains sensitive information
                    # such as secrets, user can log them by themselves
                    if not self._prevent_env_log and arg_str is not None:
                        if TENSORPC_ASYNCSSH_ENV_INIT_INDICATE in arg_str:
                            self._prevent_env_log = True
                            arg_str = None
                    metric["d"] = arg_str
                    metric["ctype"] = event.type.value
                metric.update({
                    "ts":
                    self._ns_to_us(event.timestamp -
                                   self._state.init_timestamp),
                    "type":
                    event.name,
                })
                if isinstance(event, LineEvent) and event.is_stderr:
                    metric.update({"stderr": event.is_stderr})
                if self.event_logger.jsonl_writer is not None:
                    self.event_logger.log(metric)
        if isinstance(event, LineEvent):
            self._all_events.append(event)
            if not self._ssh_inited:
                text = event.line.decode("utf-8").strip()
                if text.startswith(f"{TENSORPC_ASYNCSSH_INIT_SUCCESS}|PID="):
                    self._ssh_inited = True
                    pid = int(text.split("=")[1])
                    self._state.pid = pid
                    # await self._cur_inp_queue.put(
                    #     SSHRequest(SSHRequestType.ChangeSize, self.init_terminal_size))
                    await self._cur_inp_queue.put(self.config.cmd + "\n")
            if not event.is_command:
                # print("LINE", len(event.line), event.line)
                if event.is_stderr:
                    print(event.get_line_str(), end="", file=stderr)
                else:
                    print(event.get_line_str(), end="")
                # if self._state.pid is not None:
                #     await _log_subprocess_status_by_pyspy_debug(self._state.pid)
                if self._msg_handler_func is not None:
                    res, exc = await run_callback_noexcept(
                        self._msg_handler_func, self._all_events,
                        self._ssh_current_cmd)
                    if exc is not None:
                        LOGGER.error(f"Error in msg_handler_func: {exc[0]}")
                        LOGGER.error(exc[1])

        elif isinstance(event, (CommandEvent)):
            self._all_events.append(event)
            if event.type == CommandEventType.CURRENT_COMMAND:
                if event.arg is not None:
                    parts = event.arg.decode("utf-8").split(";")
                    self._ssh_current_cmd = ";".join(parts[:-1])
                    (LOGGER.warning if self._ssh_inited else
                     LOGGER.info)(f"CurrentCmd: {self._ssh_current_cmd}")

            if event.type == CommandEventType.COMMAND_OUTPUT_START:
                if event.arg is not None:
                    current_cmd = event.arg.decode("utf-8")
                    LOGGER.warning(f"Executing command: {current_cmd}")
            if event.type == CommandEventType.COMMAND_COMPLETE:
                return_code = None
                if event.arg is not None:
                    return_code = int(event.arg)
                    # only log after init
                    log_func = (LOGGER.warning
                                if self._ssh_inited else LOGGER.info)
                    log_func(f"Cmd completed with return code: {return_code}.")
                    log_func(f"CompletedCmd: {self._ssh_current_cmd}")
                if self._ssh_inited and TENSORPC_ASYNCSSH_INIT_SUCCESS not in self._ssh_current_cmd:
                    error_lines: Optional[list[str]] = None
                    if return_code is not None and return_code != 0:
                        error_lines = self._get_last_stderr_lines()
                    if self._is_dist_inited:
                        # send end message to master, let master determine should we stop task
                        await simple_remote_call_async(
                            self.config.master_url,
                            serv_names._TASK_WRAPPER_DISTRIBUTED_WORKER_CLOSE,
                            self._state.addr, return_code, error_lines)
                    else:
                        await self.worker_msg_command_end(
                            self._state.addr, return_code, error_lines)
                    if self.config.max_retries <= 1:
                        if self._is_standalone:
                            prim.get_async_shutdown_event().set()
                        else:
                            assert self._exit_callback is not None
                        if self._exit_callback is not None:
                            res, exc = await run_callback_noexcept(
                                self._exit_callback)
                            if exc is not None:
                                LOGGER.error(
                                    f"Error in exit callback: {exc[0]}")
                                LOGGER.error(exc[1])
                        LOGGER.warning(
                            f"CmdComplete. Trying to shutdown the server")

        elif isinstance(event, (EofEvent, ExceptionEvent)):
            # for eof event, always close
            self._all_events.append(event)
            if isinstance(event, ExceptionEvent):
                LOGGER.error(event.traceback_str)
            else:
                LOGGER.warning(event)
            if self._is_standalone:
                prim.get_async_shutdown_event().set()
            else:
                assert self._exit_callback is not None
            if self._exit_callback is not None:
                res, exc = await run_callback_noexcept(self._exit_callback)
                if exc is not None:
                    LOGGER.error(f"Error in exit callback: {exc[0]}")
                    LOGGER.error(exc[1])
            if self._is_dist_inited:
                await simple_remote_call_async(
                    self.config.master_url,
                    serv_names._TASK_WRAPPER_DISTRIBUTED_WORKER_CLOSE,
                    self._state.addr, 0, None)
            LOGGER.warning(f"SSH Eof/Exception. Trying to shutdown the server")

    async def _log_pyspy_status(self):
        if self.event_logger is None or self._state.pid is None:
            return
        ts = self._get_relative_timestamp_us()
        self.event_logger.log(
            {
                "d": await get_all_subprocess_traceback_by_pyspy(
                    self._state.pid),
                "ts": ts,
                "type": ExtendSSHEventType.TraceBack.value,
            },
            compact=True)

_T_ret = TypeVar("_T_ret")
P = ParamSpec("P")

class TaskManager:
    """this is used to launch autossh task by current process, e.g. launch shell task from a app"""

    def __init__(self):
        self._running_sshs: dict[str, TaskWrapper] = {}

    async def _run_ssh_rpc_base(self, key: str, user: str, password: str, 
            args, kwargs, func_id, is_code: bool, need_result: bool = True, addi_cmd: Optional[str] = None,
            default_url: str = "localhost:22", msg_callback: Optional[Callable[[Event],
                                                          Any]] = None):
        if key in self._running_sshs:
            raise ValueError(f"Task {key} already running.")
        port = prim.get_server_grpc_port()
        arg_event_id, ret_event_id, ev, result = SubprocessRpcClient.set_args_to_argserver(args, kwargs, func_id, is_code)
        run_cmd = f"python -m tensorpc.cli.cmd_rpc_call {arg_event_id} {ret_event_id} {port} {need_result}"
        final_cmds = [
            run_cmd,
        ]
        if addi_cmd is not None:
            final_cmds.insert(0, addi_cmd)
        task_wrapper_args = TaskWrapperArgs(" && ".join(final_cmds), password, user)
        try:
            await self.launch_task(key, task_wrapper_args, default_url=default_url, msg_callback=msg_callback, exit_callback=lambda: ev.set())
            await ev.wait()
            if not result:
                raise ValueError(f"Task {key} failed outside user code")
            else:
                if "exception" in result:
                    raise ValueError(result["exception"])
                else:
                    return result["ret"]
        finally:
            SubprocessRpcClient.remove_args_to_argserver(arg_event_id, ret_event_id, False)

    async def call_local_ssh(self, key: str, user: str, password: str, func_id: Union[str, Callable[P, _T_ret]], *args: P.args, **kwargs: P.kwargs) -> _T_ret:
        if not isinstance(func_id, str):
            func_id = get_module_id_of_type(func_id)
        return await self._run_ssh_rpc_base(key, user, password, args, kwargs, func_id, False)

    async def call_local_ssh_with_code(self, key: str, user: str, password: str, func_code: str, *args, **kwargs):
        return await self._run_ssh_rpc_base(key, user, password, args, kwargs, func_code, True)

    async def launch_task(self,
                          key: str,
                          config: TaskWrapperArgs,
                          msg_callback: Optional[Callable[[Event],
                                                          Any]] = None,
                          exit_callback: Optional[Callable[[], Any]] = None,
                          default_url: str = "localhost:22") -> bool:
        if key in self._running_sshs:
            return False
        assert config.max_retries == 1 and config.master_url is None and config.num_workers == 1, "TaskManager only support non-dist one-shot task"
        task = TaskWrapper(config,
                           key=key,
                           msg_callback=msg_callback,
                           default_url=default_url,
                           exit_callback=partial(self.stop_task, key=key, user_exit_callback=exit_callback))
        await task.init_async()
        self._running_sshs[key] = task
        return True

    async def stop_task(self, key: str, user_exit_callback: Optional[Callable[[], Any]] = None):
        if key in self._running_sshs:
            task = self._running_sshs.pop(key)
            await task.exit_async()
            if user_exit_callback is not None:
                await run_callback(user_exit_callback)

    async def change_task_terminal_size(self, key: str, width: int,
                                        height: int):
        if key in self._running_sshs:
            await self._running_sshs[key].change_terminal_size(width, height)

    async def log_event(self, key: str, event_dict: Any):
        if key in self._running_sshs:
            await self._running_sshs[key].log_event(event_dict)