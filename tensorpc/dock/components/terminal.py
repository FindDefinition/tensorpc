import asyncio
from collections import deque
import enum
from functools import partial
import inspect
import time
import traceback
from typing import Any, Awaitable, Callable, Optional, Union
from typing_extensions import Literal
from tensorpc.autossh.constants import TENSORPC_ASYNCSSH_INIT_SUCCESS
from tensorpc.autossh.core import CommandEvent, CommandEventType, EofEvent, ExceptionEvent, LineEvent, LineEventType, LineRawEvent, RawEvent, SSHClient, SSHConnDesc, SSHRequest, SSHRequestType, LOGGER, remove_trivial_r_lines
from tensorpc.autossh.core import Event as SSHEvent
import dataclasses as dataclasses_plain
import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.dock.core.common import handle_standard_event
from tensorpc.dock.core.component import EventSlotEmitter, FrontendEventType, UIType
from tensorpc.dock.jsonlike import Undefined, undefined
from tensorpc.autossh.core import LOGGER as SSH_LOGGER
from .mui import (MUIBasicProps, MUIComponentBase, FlexBoxProps, NumberType,
                  Event)
import bisect 

@dataclasses.dataclass
class TerminalProps(MUIBasicProps):
    initData: Union[str, bytes, Undefined] = undefined
    terminalId: Union[str, Undefined] = undefined
    boxProps: Union[FlexBoxProps, Undefined] = undefined
    theme: Union[Literal["light", "dark"], Undefined] = undefined

    allowProposedApi: Union[bool, Undefined] = undefined
    allowTransparency: Union[bool, Undefined] = undefined
    altClickMovesCursor: Union[bool, Undefined] = undefined
    convertEol: Union[bool, Undefined] = undefined
    cursorBlink: Union[bool, Undefined] = undefined
    cursorStyle: Union[Literal["block", "underline", "bar"],
                       Undefined] = undefined

    cursorWidth: Union[NumberType, Undefined] = undefined
    cursorInactiveStyle: Union[Literal["block", "underline", "bar"],
                               Undefined] = undefined
    customGlyphs: Union[bool, Undefined] = undefined
    disableStdin: Union[bool, Undefined] = undefined
    drawBoldTextInBrightColors: Union[bool, Undefined] = undefined
    fastScrollModifier: Union[Literal["alt", "ctrl", "shift", "none"],
                              Undefined] = undefined
    fastScrollSensitivity: Union[NumberType, Undefined] = undefined
    fontSize: Union[NumberType, Undefined] = undefined
    fontFamily: Union[str, Undefined] = undefined
    fontWeight: Union[Literal["normal", "bold"], NumberType,
                      Undefined] = undefined
    fontWeightBold: Union[Literal["normal", "bold"], NumberType,
                          Undefined] = undefined
    ignoreBracketedPasteMode: Union[bool, Undefined] = undefined
    letterSpacing: Union[NumberType, Undefined] = undefined
    lineHeight: Union[NumberType, Undefined] = undefined
    macOptionIsMeta: Union[bool, Undefined] = undefined
    macOptionClickForcesSelection: Union[bool, Undefined] = undefined
    minimumContrastRatio: Union[NumberType, Undefined] = undefined
    rescaleOverlappingGlyphs: Union[bool, Undefined] = undefined
    rightClickSelectsWord: Union[bool, Undefined] = undefined
    screenReaderMode: Union[bool, Undefined] = undefined
    scrollback: Union[NumberType, Undefined] = undefined
    scrollOnUserInput: Union[bool, Undefined] = undefined
    scrollSensitivity: Union[NumberType, Undefined] = undefined
    smoothScrollDuration: Union[NumberType, Undefined] = undefined
    tabStopWidth: Union[NumberType, Undefined] = undefined
    wordSeparator: Union[str, Undefined] = undefined
    overviewRulerWidth: Union[NumberType, Undefined] = undefined


class TerminalEventType(enum.IntEnum):
    Raw = 0
    Eof = 1
    ClearAndWrite = 2

@dataclasses.dataclass
class TerminalResizeEvent:
    width: int
    height: int

class TerminalBuffer:
    def __init__(self, maxlen: int = 10000, min_buffer_len: int = 256):
        self._raw_buffers_with_ts: deque[tuple[bytes, int]] = deque(maxlen=maxlen)
        self._min_buffer_len = min_buffer_len

    def get_total_size(self):
        return sum(len(x[0]) for x in self._raw_buffers_with_ts)

    def save_state(self, state: bytes, state_ts: int):
        assert state_ts >= 0
        if not self._raw_buffers_with_ts:
            self._raw_buffers_with_ts.append((state, state_ts))
            return
        # clean up old data with ts belongs to [0, state_ts]
        old_ts_idx = bisect.bisect_right(self._raw_buffers_with_ts, state_ts, key=lambda x: x[1])
        # print("old_ts_idx", old_ts_idx, len(self._raw_buffers_with_ts), state_ts)
        new_dq = list(self._raw_buffers_with_ts)[old_ts_idx:]
        # assert new_dq[0][1] > state_ts
        self._raw_buffers_with_ts = deque([(state, state_ts)] + new_dq, maxlen=self._raw_buffers_with_ts.maxlen)
    
    def load_state(self):
        # merge all buffers to one
        if not self._raw_buffers_with_ts:
            return b"", 0
        last_ts = self._raw_buffers_with_ts[-1][1]
        buffers = [x[0] for x in self._raw_buffers_with_ts]
        buffer = b"".join(buffers)
        self._raw_buffers_with_ts.clear()
        self._raw_buffers_with_ts.append((buffer, last_ts))
        return buffer, last_ts

    def load_state_backend_only(self):
        # load state without merge buffer
        last_ts = self._raw_buffers_with_ts[-1][1]
        buffers = [x[0] for x in self._raw_buffers_with_ts]
        buffer = b"".join(buffers)
        return buffer, last_ts

    def append_buffer(self, buffer: bytes, ts: int):
        if self._raw_buffers_with_ts:
            last_buf = self._raw_buffers_with_ts[-1]
            assert ts >= last_buf[1], f"ts {ts} is less than last ts {last_buf[1]}"
            if len(buffer) + len(last_buf[0]) < self._min_buffer_len:
                # merge buffer with last buffer
                self._raw_buffers_with_ts[-1] = (last_buf[0] + buffer, ts)
            else:
                # add new buffer
                self._raw_buffers_with_ts.append((buffer, ts))
        else:
            self._raw_buffers_with_ts.append((buffer, ts))

class Terminal(MUIComponentBase[TerminalProps]):

    def __init__(self, init_data: Optional[Union[bytes, str]] = None, callback: Optional[Callable[[Union[str, bytes]], Any]] = None, 
            terminalId: Optional[str] = None, 
            state_buffers: Optional[dict[str, TerminalBuffer]] = None,
            dont_use_frontend_state: bool = True) -> None:
        super().__init__(UIType.Terminal,
                         TerminalProps,
                         allowed_events=[
                             FrontendEventType.TerminalInput.value,
                             FrontendEventType.TerminalResize.value,
                             FrontendEventType.TerminalFrontendUnmount.value,
                             FrontendEventType.TerminalFrontendMount.value,
                         ])
        if init_data is not None:
            self.prop(initData=init_data)
        if terminalId is not None:
            self.prop(terminalId=terminalId)
        if state_buffers is not None:
            assert terminalId is not None, "state_buffers must be None when terminalId is None."
            assert terminalId in state_buffers, f"terminalId {terminalId} not in state_buffers."
        self._state_buffers = state_buffers
        self._dont_use_frontend_state = dont_use_frontend_state
        self.event_terminal_input = self._create_event_slot(
            FrontendEventType.TerminalInput)
        self.event_terminal_resize = self._create_event_slot(
            FrontendEventType.TerminalResize, lambda x: TerminalResizeEvent(**x))
        self.event_terminal_frontend_unmount = self._create_event_slot(
            FrontendEventType.TerminalFrontendUnmount)
        self.event_terminal_frontend_mount = self._create_event_slot(
            FrontendEventType.TerminalFrontendMount)

        # self._terminal_state = b""
        # self._is_terminal_frontend_unmounted = False
        # if self._state_buffers is not None:
        self.event_terminal_frontend_unmount.on(self._terminal_state_unmount)
        self.event_terminal_frontend_mount.on(self._terminal_state_mount)
        if callback is not None:
            self.event_terminal_input.on(callback)

    def set_state_buffers(self, state_buffers: dict[str, TerminalBuffer]):
        assert not isinstance(self.props.terminalId, Undefined), "terminalId must be set when state_buffers is not None."
        assert self.props.terminalId in state_buffers, f"terminalId {self.props.terminalId} not in state_buffers."
        self._state_buffers = state_buffers

    def reset_state_buffers(self):
        self._state_buffers = None

    async def _terminal_state_unmount(self, evdata):
        state = evdata["state"]
        term_id = evdata["terminalId"]
        ts = evdata["ts"]
        # print(ts, state, evdata["terminalId"])
        if self._state_buffers is not None and not self._dont_use_frontend_state:
            if term_id in self._state_buffers:
                buffer = self._state_buffers[term_id]
                buffer.save_state(state, ts)
                # print("terminal state unmount", len(state), buffer.get_total_size(), "|", len(buffer._raw_buffers_with_ts), self.props.terminalId)
                # print(state)
        # self._terminal_state = state
        # self._is_terminal_frontend_unmounted = True

    async def _terminal_state_mount(self, term_id):
        if self._state_buffers is not None:
            if term_id is not None and term_id in self._state_buffers:
                buffer = self._state_buffers[term_id]
                if self._dont_use_frontend_state:
                    state_to_send, ts = buffer.load_state_backend_only()
                else:
                    state_to_send, ts = buffer.load_state()
                # print("terminal state mount", len(state_to_send), self.props.terminalId)
                # print(state_to_send)
                await self.clear_and_write(state_to_send, ts)

        # state = self._state_buffers[self.props.terminalId].load_state()
        # state_to_send = self._terminal_state
        # self._terminal_state = b""
        # self._is_terminal_frontend_unmounted = False
        # print("terminal state mount", state_to_send, self.props.terminalId)
        # await self.send_raw(state_to_send)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync=is_sync)

    async def clear(self):
        await self.clear_and_write(b"")

    async def clear_and_write(self, content: Union[str, bytes], ts: Optional[int] = None):
        await self.put_app_event(
            self.create_comp_raw_event([TerminalEventType.ClearAndWrite.value, content, ts]))

    def append_buffer(self, data: Union[bytes, str], ts: int = 0):
        if self._state_buffers is not None:
            assert not isinstance(self.props.terminalId, Undefined), "terminalId must be set when state_buffers is not None."
            if self.props.terminalId in self._state_buffers:
                buffer = self._state_buffers[self.props.terminalId]
                if isinstance(data, str):
                    data = data.encode("utf-8")
                buffer.append_buffer(data, ts)

    async def send_raw(self, data: Union[bytes, str], ts: int = 0, append_buffer: bool = True):
        # if self._is_terminal_frontend_unmounted:
        #     if isinstance(data, str):
        #         data = data.encode("utf-8")
        #     self._terminal_state += data
        #     return
        # print("SEND RAW", data, self.props.terminalId)
        if append_buffer:
            self.append_buffer(data, ts)
        await self.put_app_event(
            self.create_comp_raw_event([TerminalEventType.Raw.value, data, ts]))
    
    async def send_eof(self):
        await self.put_app_event(
            self.create_comp_raw_event([TerminalEventType.Eof.value, None]))


@dataclasses_plain.dataclass
class TerminalLineEvent:
    ts: int 
    d: bytes
@dataclasses_plain.dataclass
class TerminalCmdCompleteEvent:
    cmd: str 
    outputs: list[TerminalLineEvent]
    return_code: Optional[int] = None

    def get_output(self):
        return b"".join([x.d for x in self.outputs])

@dataclasses_plain.dataclass
class _AsyncSSHTerminalState:
    inp_queue: asyncio.Queue
    task: asyncio.Task 
    inited: bool 
    pid: int
    size: TerminalResizeEvent
    base_ts: int
    current_cmd: str = ""
    current_cmd_buffer: Optional[deque[TerminalLineEvent]] = None
    current_cmd_rpc_future: Optional[asyncio.Future[TerminalCmdCompleteEvent]] = None

class AsyncSSHTerminal(Terminal):

    def __init__(self,
                 desc: Optional[SSHConnDesc] = None,
                 init_data: Optional[Union[bytes, str]] = None,
                 manual_connect: bool = True,
                 manual_disconnect: bool = False,
                 line_raw_ev_max_length: int = 10000,
                 init_size: Optional[tuple[int, int]] = (80, 24),
                 terminalId: Optional[str] = None) -> None:
        super().__init__(init_data, terminalId=terminalId)
        if desc is None:
            assert manual_connect, "Cannot auto connect/disconnect when mount without url_with_port(ip:port), username, and password."
        self._shutdown_ev = asyncio.Event()
        if desc is not None:
            self._client = SSHClient(desc.url_with_port, desc.username, desc.password)
        else:
            self._client = SSHClient("", "", "")
        self.event_after_mount.on(self._on_mount)
        self.event_after_unmount.on(self._on_unmount)
        self._manual_connect = manual_connect
        self._manual_disconnect = manual_disconnect
        self.event_terminal_input.on(self._on_input)
        self.event_terminal_resize.on(self._on_resize)

        self._backend_ssh_conn_inited_event_key = "__backend_ssh_conn_inited"
        self._backend_ssh_conn_close_event_key = "__backend_ssh_conn_close"
        self._backend_ssh_cmd_complete_event_key = "__backend_ssh_cmd_complete"

        self.event_ssh_conn_close = self._create_emitter_event_slot_noarg(
            self._backend_ssh_conn_close_event_key)
        self.event_ssh_conn_inited = self._create_emitter_event_slot_noarg(
            self._backend_ssh_conn_inited_event_key)
        self.event_ssh_cmd_complete: EventSlotEmitter[TerminalCmdCompleteEvent] = self._create_emitter_event_slot(
            self._backend_ssh_cmd_complete_event_key)

        self._ssh_state: Optional[_AsyncSSHTerminalState] = None
        self._init_event = asyncio.Event()
        self._init_size: Optional[TerminalResizeEvent] = None
        if init_size is not None:
            self._init_size = TerminalResizeEvent(init_size[0], init_size[1])
        self._raw_data_buffer: Optional[bytes] = None
        self._raw_data_ts: int = 0
        self._line_raw_ev_max_length = line_raw_ev_max_length
        self._line_raw_event_buffer: deque[bytes] = deque(maxlen=line_raw_ev_max_length)

    async def connect(self,
                      event_callback: Optional[Callable[[SSHEvent],
                                                        None]] = None):
        assert self._client.url and self._client.username and self._client.password, "Cannot connect without url, username, and password."
        desc = SSHConnDesc(self._client.url,
                                         self._client.username,
                                         self._client.password)
        await self.connect_with_new_desc(desc, event_callback)

    async def _on_exit(self):
        if self._ssh_state is not None:
            if self._ssh_state.current_cmd_rpc_future is not None:
                self._ssh_state.current_cmd_rpc_future.set_exception(
                    Exception("SSH Unknown error."))
                self._ssh_state.current_cmd_rpc_future = None
            # we can't await task here because it will cause deadlock
            self._ssh_state = None
        SSH_LOGGER.warning("SSH Exit.")
        self._init_event.set()

        await self.flow_event_emitter.emit_async(
            self._backend_ssh_conn_close_event_key,
            Event(self._backend_ssh_conn_close_event_key, None))

    async def connect_with_new_desc(
            self,
            desc: SSHConnDesc,
            event_callback: Optional[Callable[[SSHEvent], None]] = None,
            init_cmds: Optional[list[str]] = None,
            term_line_event_callback: Optional[Callable[[TerminalLineEvent], None]] = None,
            exit_event: Optional[asyncio.Event] = None):
        if init_cmds:
            for cmd in init_cmds:
                assert cmd.endswith("\n"), "All command must end with \\n"
        assert self._ssh_state is None, "Cannot connect with new info while the current connection is still active."
        self._client = SSHClient(desc.url_with_port, desc.username, desc.password)
        if self.is_mounted():
            await self.clear()
        self._shutdown_ev.clear()
        self._init_event.clear()
        sd_task = asyncio.create_task(self._shutdown_ev.wait())
        cur_inp_queue = asyncio.Queue()
        size_state = TerminalResizeEvent(-1, -1)
        if self._init_size is not None:
            size_state = self._init_size
            await cur_inp_queue.put(
                SSHRequest(
                    SSHRequestType.ChangeSize,
                    [self._init_size.width, self._init_size.height]))
        if init_cmds:
            for cmd in init_cmds:
                await cur_inp_queue.put(cmd)
        await cur_inp_queue.put(
            f" echo \"{TENSORPC_ASYNCSSH_INIT_SUCCESS}|PID=$TENSORPC_SSH_CURRENT_PID\"\n"
        )
        base_ts = time.time_ns()
        ssh_task = asyncio.create_task(
            self._client.connect_queue(cur_inp_queue,
                                       partial(self._handle_ssh_queue,
                                               user_callback=event_callback),
                                       shutdown_task=sd_task,
                                       request_pty=True,
                                       exit_callback=self._on_exit,
                                       term_type="xterm-256color",
                                       enable_raw_event=True,
                                       exit_event=exit_event,
                                       line_raw_callback=partial(
                                           self._handle_line_raw_event,
                                           term_line_event_callback=term_line_event_callback),))
        self._ssh_state = _AsyncSSHTerminalState(cur_inp_queue, ssh_task, False, -1, size_state, base_ts)
        # TODO change init event to future
        await self._init_event.wait()
        if self._ssh_state is None:
            raise RuntimeError("SSH connect failed.")
        return self._ssh_state.inp_queue

    async def disconnect(self):
        self._shutdown_ev.set()
        if self._ssh_state is not None:
            await self._ssh_state.task
            self._ssh_state = None

    async def _on_mount(self):
        if self._line_raw_event_buffer:
            self._terminal_state = b"".join(self._line_raw_event_buffer)
            self._line_raw_event_buffer.clear()
        if not self._manual_connect:
            await self.connect()
        # else:
        #     await self.clear_and_write("disconnected.")

    async def _on_unmount(self):
        if not self._manual_disconnect:
            await self.disconnect()
            await self.clear_and_write("disconnected.")

    async def _on_input(self, data: Union[str, bytes]):
        if self._ssh_state is not None:
            assert self._ssh_state.inp_queue is not None 
            await self._ssh_state.inp_queue.put(data)

    async def _on_resize(self, data: TerminalResizeEvent):
        self._size_state = data
        if self._ssh_state is not None:
            assert self._ssh_state.inp_queue is not None 
            await self._ssh_state.inp_queue.put(
                SSHRequest(SSHRequestType.ChangeSize,
                           [data.width, data.height]))

    async def _handle_line_raw_event(self,
                                event: LineRawEvent,
                                term_line_event_callback: Optional[Callable[[TerminalLineEvent], None]] = None):
        if event.line_ev_type != LineEventType.EXCEPTION:
            if event.line_ev_type == LineEventType.RBUF_OVERFLOW or event.line_ev_type == LineEventType.INCOMPLETE_START:
                if self._raw_data_buffer is not None:
                    rbuf = self._raw_data_buffer
                    rbuf += event.data
                else:
                    rbuf = event.data
                self._raw_data_buffer = remove_trivial_r_lines(rbuf)
                self._raw_data_ts = event.ts
            else:
                if self._raw_data_buffer is not None:
                    self._line_raw_event_buffer.append(self._raw_data_buffer)
                    if self._ssh_state is not None:
                        if self._ssh_state.current_cmd_buffer is not None:
                            self._ssh_state.current_cmd_buffer.append(TerminalLineEvent(self._raw_data_ts, self._raw_data_buffer))
                        if term_line_event_callback is not None:
                            term_line_event_callback(TerminalLineEvent(self._raw_data_ts, self._raw_data_buffer))

                    self._raw_data_buffer = None
                    self._raw_data_ts = 0
                data = event.data
                self._line_raw_event_buffer.append(data)
                if self._ssh_state is not None:
                    if self._ssh_state.current_cmd_buffer is not None:
                        self._ssh_state.current_cmd_buffer.append(TerminalLineEvent(event.ts, data))
                    if term_line_event_callback is not None:
                        term_line_event_callback(TerminalLineEvent(event.ts, data))


    async def _handle_ssh_queue(self,
                                event: SSHEvent,
                                user_callback: Optional[Callable[[SSHEvent],
                                                                 Any]] = None):
        assert self._ssh_state is not None
        try:
            if not isinstance(event, RawEvent):
                if isinstance(event, LineEvent):
                    if not self._ssh_state.inited:
                        text = event.line.decode("utf-8").strip()
                        if text.startswith(f"{TENSORPC_ASYNCSSH_INIT_SUCCESS}|PID="):
                            pid = int(text.split("=")[1])
                            self._ssh_state.pid = pid
                            # self._ssh_state.inited = True
                            # self._init_event.set()
                            # self._ssh_state.current_cmd_buffer = None
                            # # self._ssh_state.current_cmd = ""
                            # self._ssh_state.current_cmd_rpc_future = None
                            # SSH_LOGGER.warning("SSH init success, pid: %d", pid)
                            # await self.flow_event_emitter.emit_async(
                            #     self._backend_ssh_conn_inited_event_key,
                            #     Event(self._backend_ssh_conn_inited_event_key, None))
                elif isinstance(event, (CommandEvent)):
                    if event.type == CommandEventType.CURRENT_COMMAND:
                        if event.arg is not None:
                            parts = event.arg.decode("utf-8").split(";")
                            self._ssh_state.current_cmd = ";".join(parts[:-1])
                    elif event.type == CommandEventType.COMMAND_OUTPUT_START:
                        self._ssh_state.current_cmd_buffer = deque(maxlen=self._line_raw_ev_max_length)
                    elif event.type == CommandEventType.COMMAND_COMPLETE:
                        return_code = 0
                        if event.arg is not None:
                            return_code = int(event.arg)
                        msg_buf = self._ssh_state.current_cmd_buffer
                        cur_cmd = self._ssh_state.current_cmd
                        future = self._ssh_state.current_cmd_rpc_future
                        self._ssh_state.current_cmd_buffer = None
                        self._ssh_state.current_cmd = ""
                        self._ssh_state.current_cmd_rpc_future = None
                        cmd_outputs: list[TerminalLineEvent] = []
                        if msg_buf is not None:
                            # line raw callback is called before event handler, so we remove last line.
                            cmd_outputs = list(msg_buf)[:-1]
                            # remove last lines starts with \x1b
                            last_idx = len(cmd_outputs)
                            while last_idx > 0:
                                if cmd_outputs[last_idx - 1].d.startswith(b"\x1b"):
                                    last_idx -= 1
                                else:
                                    break
                            cmd_outputs = cmd_outputs[:last_idx]
                        # print(cur_cmd, TENSORPC_ASYNCSSH_INIT_SUCCESS in cur_cmd, self._ssh_state.inited)
                        if TENSORPC_ASYNCSSH_INIT_SUCCESS in cur_cmd and not self._ssh_state.inited:
                            self._ssh_state.inited = True
                            self._init_event.set()
                            SSH_LOGGER.warning("SSH init success, pid: %d", self._ssh_state.pid)
                            await self.flow_event_emitter.emit_async(
                                self._backend_ssh_conn_inited_event_key,
                                Event(self._backend_ssh_conn_inited_event_key, None))

                        if self._ssh_state.inited and TENSORPC_ASYNCSSH_INIT_SUCCESS not in cur_cmd:
                            ev = TerminalCmdCompleteEvent(cur_cmd, cmd_outputs, return_code)
                            SSH_LOGGER.warning("Command \"%s\" succeed, retcode: %d", cur_cmd, return_code)
                            if future is not None:
                                future.set_result(ev)
                            await self.flow_event_emitter.emit_async(
                                self._backend_ssh_cmd_complete_event_key,
                                Event(self._backend_ssh_cmd_complete_event_key, ev))

                if user_callback is not None:
                    try:
                        res = user_callback(event)
                        if inspect.iscoroutine(res):
                            await res
                    except:
                        traceback.print_exc()
            if isinstance(event, RawEvent):
                # print(event.raw)
                if self.is_mounted():
                    await self.send_raw(event.raw, event.timestamp - self._ssh_state.base_ts)
                else:
                    self.append_buffer(event.raw, event.timestamp - self._ssh_state.base_ts)
            elif isinstance(event, (EofEvent, ExceptionEvent)):
                if self._ssh_state.current_cmd_rpc_future is not None:
                    self._ssh_state.current_cmd_rpc_future.set_exception(
                        Exception("SSH connection closed."))
                    self._ssh_state.current_cmd_rpc_future = None
                if isinstance(event, ExceptionEvent):
                    LOGGER.error(event.traceback_str)
                else:
                    LOGGER.warning(event)
                if self.is_mounted():
                    await self.send_eof()
        except:
            if self._ssh_state.current_cmd_rpc_future is not None:
                self._ssh_state.current_cmd_rpc_future.set_exception(
                    Exception("SSH Unknown error."))
                self._ssh_state.current_cmd_rpc_future = None

            traceback.print_exc()
            raise

    def get_current_state(self):
        return self._ssh_state

    async def ssh_command_rpc_future(self, cmd: str) -> asyncio.Future[TerminalCmdCompleteEvent]:
        """Run a bash command in ssh target, return a event with return code and stdout/stderr data.
        
        WARNING: this function can't be called parallelly, e.g. asyncio.gather
        """
        assert cmd.endswith("\n")
        assert self._ssh_state is not None and self._ssh_state.current_cmd == "", f"current command is not empty: {self._ssh_state}."
        assert self._ssh_state.current_cmd_rpc_future is None, "previous command is not finished yet."
        future: asyncio.Future[TerminalCmdCompleteEvent] = asyncio.get_running_loop().create_future()
        self._ssh_state.current_cmd_rpc_future = future
        await self._ssh_state.inp_queue.put(cmd)
        return future

    async def ssh_command_rpc(self, cmd: str) -> TerminalCmdCompleteEvent:

        """Run a bash command in ssh target, return a event with return code and stdout/stderr data.
        
        WARNING: this function can't be called parallelly, e.g. asyncio.gather
        """
        future = await self.ssh_command_rpc_future(cmd)
        await future
        return future.result()

    def is_connected(self):
        return self._ssh_state is not None and self._ssh_state.inited