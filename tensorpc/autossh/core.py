import abc
import ast
import asyncio
import bisect
from collections.abc import Mapping, MutableMapping
import contextlib
import dataclasses
import enum
from functools import partial
import io
import os
from pathlib import Path
import re
import sys
import threading
import uuid
import async_timeout
from typing_extensions import Literal
import warnings
import time
import traceback
from asyncio.tasks import FIRST_COMPLETED
from contextlib import suppress
from typing import (TYPE_CHECKING, Any, AnyStr, Awaitable, Callable, Coroutine, Deque,
                    Dict, Iterable, List, Optional, ParamSpec, Set, Tuple, Type, TypeVar, Union,
                    cast)

import asyncssh
from asyncssh import stream as asyncsshss
from asyncssh.misc import SoftEOFReceived
from asyncssh.scp import scp as asyncsshscp

import tensorpc
from tensorpc.autossh.constants import TENSORPC_ASYNCSSH_ENV_INIT_INDICATE, TENSORPC_ASYNCSSH_PROXY
from tensorpc.autossh.coretypes import SSHTarget
from tensorpc.compat import InWindows
from tensorpc.constants import PACKAGE_ROOT, TENSORPC_READUNTIL
from tensorpc.core import prim
from tensorpc.core.moduleid import get_module_id_of_type, get_object_type_from_module_id, import_dynamic_func
from tensorpc.core.rprint_dispatch import rprint

from tensorpc.utils.rich_logging import get_logger
from tensorpc.core.bgserver import BACKGROUND_SERVER
from tensorpc.core.prim import is_in_server_context, check_is_service_available, get_server_grpc_port
from tensorpc.core.client import RemoteManager, simple_remote_call
LOGGER = get_logger("ssh")

_SSH_ARGSERVER_SERV_NAME = "tensorpc.services.collection::ArgServer"

BASH_HOOKS_FILE_NAME = "hooks-bash.sh"

@dataclasses.dataclass
class ShellInfo:
    type: Literal["bash", "zsh", "fish", "powershell", "cmd", "cygwin"]
    os_type: Literal["linux", "macos", "windows"]

    def multiple_cmd(self, cmds: List[str]):
        if self.type == "powershell":
            return "; ".join(cmds)
        elif self.type == "bash" or self.type == "zsh" or self.type == "fish" or self.type == "cygwin":
            return " && ".join(cmds)
        elif self.type == "cmd":
            return " && ".join(cmds)
        raise NotImplementedError

    def single_cmd_shell_wrapper(self, cmd: str):
        if self.type == "powershell":
            return f"powershell -c {cmd}"
        elif self.type == "bash" or self.type == "zsh" or self.type == "fish" or self.type == "cygwin":
            return f"{self.type} -ic \"{cmd}\""
        elif self.type == "cmd":
            return f"cmd /c {cmd}"
        raise NotImplementedError

async def terminal_shell_type_detector(cmd_runner: Callable[[str, bool], Coroutine[None, None, Optional[str]]]):
    # TODO pwsh in linux
    # supported shell types: bash, zsh, fish, powershell
    # if not found, return None
    # if found, return shell type
    # 1. check if powershell is available
    res = await cmd_runner("$PSVersionTable", False)
    if res is not None and res.strip():
        return ShellInfo("powershell", "windows")
    # 2. use ver to check is windows cmd (default bash type for windows)
    res = await cmd_runner("ver", False)
    if res is not None and res.strip().startswith("Microsoft Windows"):
        return ShellInfo("cmd", "windows")
    # now we are in linux or macos or cygwin (windows)
    # we can use uname to check os types 
    res = await cmd_runner("uname -s", False)
    if res is None:
        return None
    res = res.strip()
    os_type: Literal["linux", "macos", "windows"] = "linux"
    if res.startswith("CYGWIN"):
        os_type = "windows"
    elif res == "Darwin":
        os_type = "macos"
    # now we can check shell type
    # we use a cmd that shoule be unknown in bash/zsh/fish
    res = await cmd_runner("tensorpcisverygood", True)
    if res is None:
        return None
    res = res.strip()
    parts = res.split(":")
    shell_type = parts[0]
    if shell_type == "bash" or shell_type == "zsh" or shell_type == "fish":
        return ShellInfo(shell_type, os_type)
    return None

def determine_hook_path_by_shell_info(shell_info: ShellInfo) -> Path:
    if shell_info.os_type == "windows":
        return PACKAGE_ROOT / "autossh" / "media" / "hooks-ps1.ps1"
    if shell_info.type == "bash":
        return PACKAGE_ROOT / "autossh" / "media" / BASH_HOOKS_FILE_NAME
    elif shell_info.type == "zsh":
        return PACKAGE_ROOT / "autossh" / "media" / ".tensorpc_hooks-zsh/.zshrc"
    # don't support fish
    raise NotImplementedError

class CommandEventType(enum.Enum):
    PROMPT_START = "A"
    PROMPT_END = "B"
    COMMAND_OUTPUT_START = "C"
    COMMAND_COMPLETE = "D"
    CURRENT_COMMAND = "E"

    UPDATE_CWD = "P"
    CONTINUATION_START = "F"
    CONTINUATION_END = "G"


class CommandEventParseState(enum.IntEnum):
    VscPromptStart = 0  # reached when we encounter \033
    # VscCmdIdReached = 1 # reached when we encounter \]784;
    VscCmdCodeABCFG = 2  # reached when we encounter A/B/C/F/G
    VscCmdCodeD = 3  # reached when we encounter D
    VscCmdCodeE = 4  # reached when we encounter E
    VscCmdCodeP = 5  # reached when we encounter P
    VscPromptEnd = 100  # reached when we encounter \007, idle state


class CommandParseSpecialCharactors:
    Start = b"\033"
    StartAll = b"\033]784;"

    End = b"\007"


_DEFAULT_SEPARATORS = rb"(?:\r\n)|(?:\n)|(?:\r)|(?:\033\]784;[ABPCEFGD](?:;(.*?))?\007)"
# _DEFAULT_SEPARATORS = "\n"

class OutData:

    def __init__(self) -> None:
        pass


class Event:
    name = "Event"

    def __init__(self, timestamp: int, is_stderr: bool, uid: str = ""):
        self.timestamp = timestamp
        self.is_stderr = is_stderr
        self.uid = uid

    def __repr__(self):
        return "{}({})".format(self.name, self.timestamp)

    def to_dict(self):
        return {
            "type": self.name,
            "ts": self.timestamp,
            "uid": self.uid,
            "is_stderr": self.is_stderr,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        assert cls.name == data["type"]
        return cls(data["ts"], data["is_stderr"], data["uid"])

    def __lt__(self, other: Union["Event", int]):
        if isinstance(other, Event):
            other = other.timestamp
        return self.timestamp < other

    def __le__(self, other: Union["Event", int]):
        if isinstance(other, Event):
            other = other.timestamp
        return self.timestamp <= other

    def __gt__(self, other: Union["Event", int]):
        if isinstance(other, Event):
            other = other.timestamp
        return self.timestamp > other

    def __ge__(self, other: Union["Event", int]):
        if isinstance(other, Event):
            other = other.timestamp
        return self.timestamp >= other

    def __eq__(self, other: Any):
        if isinstance(other, Event):
            return self.timestamp == other.timestamp
        elif isinstance(other, int):
            return self.timestamp == other
        raise NotImplementedError

    def __ne__(self, other: Any):
        if isinstance(other, Event):
            return self.timestamp != other.timestamp
        elif isinstance(other, int):
            return self.timestamp != other
        raise NotImplementedError

class EventType(enum.Enum):
    Line = "L"
    Eof = "Eof"
    Exception = "Exc"
    Raw = "R"
    Command = "C"
    External = "Ex"

class EofEvent(Event):
    name = EventType.Eof.value

    def __init__(self,
                 timestamp: int,
                 status: int = 0,
                 is_stderr=False,
                 uid: str = ""):
        super().__init__(timestamp, is_stderr, uid)
        self.status = status

    def __bool__(self):
        return self.status == 0

    def __repr__(self):
        return "{}({}|{})".format(self.name, self.status, self.timestamp)

    def to_dict(self):
        res = super().to_dict()
        res["status"] = self.status
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        assert cls.name == data["type"]
        return cls(data["ts"], data["status"], data["is_stderr"], data["uid"])

class ExternalEvent(Event):
    name = EventType.External.value

    def __init__(self,
                 timestamp: int,
                 data: Any,
                 is_stderr=False,
                 uid: str = ""):
        super().__init__(timestamp, is_stderr, uid)
        self._data = data

    def to_dict(self):
        res = super().to_dict()
        res["data"] = self._data
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        assert cls.name == data["type"]
        return cls(data["ts"], data["data"], data["is_stderr"], data["uid"])


class LineEvent(Event):
    name = EventType.Line.value

    def __init__(self,
                 timestamp: int,
                 line: bytes,
                 is_stderr=False,
                 uid: str = "",
                 is_command: bool = False):
        super().__init__(timestamp, is_stderr, uid)
        self.line = line
        self.is_command = is_command

        self._line_str_cache = None

    def get_line_str(self):
        if self._line_str_cache is not None:
            return self._line_str_cache
        self._line_str_cache = self.line.decode("utf-8")
        return self._line_str_cache

    def __repr__(self):
        return "{}({}|{}|line={})".format(self.name, self.is_stderr,
                                          self.timestamp, self.line)

    def to_dict(self):
        res = super().to_dict()
        res["line"] = self.line
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        assert cls.name == data["type"]
        return cls(data["ts"], data["line"], data["is_stderr"], data["uid"])


class RawEvent(Event):
    name = EventType.Raw.value

    def __init__(self,
                 timestamp: int,
                 raw: bytes,
                 is_stderr=False,
                 uid: str = ""):
        super().__init__(timestamp, is_stderr, uid)
        self.raw = raw

    def __repr__(self):
        r = self.raw
        return "{}({}|{}|raw={})".format(self.name, self.is_stderr,
                                         self.timestamp, r)

    def to_dict(self):
        res = super().to_dict()
        res["raw"] = self.raw
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        assert cls.name == data["type"]
        return cls(data["ts"], data["line"], data["is_stderr"], data["uid"])


class ExceptionEvent(Event):
    name = EventType.Exception.value

    def __init__(self,
                 timestamp: int,
                 data: Any,
                 is_stderr=False,
                 uid: str = "",
                 traceback_str: str = ""):
        super().__init__(timestamp, is_stderr, uid)
        self.data = data
        self.traceback_str = traceback_str

    def to_dict(self):
        res = super().to_dict()
        res["traceback_str"] = self.traceback_str
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        assert cls.name == data["type"]
        return cls(data["ts"], None, data["is_stderr"], data["uid"],
                   data["traceback_str"])


class CommandEvent(Event):
    name = EventType.Command.value

    def __init__(self,
                 timestamp: int,
                 type: str,
                 arg: Optional[bytes],
                 is_stderr=False,
                 uid: str = ""):
        super().__init__(timestamp, is_stderr, uid)
        self.type = CommandEventType(type)
        self.arg = arg

        self._line_str_cache = None
        
    def get_arg_str(self):
        if self.arg is None or self._line_str_cache is not None:
            return self._line_str_cache
        self._line_str_cache = self.arg.decode("utf-8")
        return self._line_str_cache

    def __repr__(self):
        return "{}({}|{}|type={}|arg={})".format(self.name, self.is_stderr,
                                                 self.timestamp, self.type,
                                                 self.arg)

    def to_dict(self):
        res = super().to_dict()
        res["cmdtype"] = self.type.value
        if self.arg is not None:
            res["arg"] = self.arg
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        assert cls.name == data["type"]
        return cls(data["ts"], data["cmdtype"], data.get("arg", None),
                   data["is_stderr"], data["uid"])


_ALL_EVENT_TYPES: List[Type[Event]] = [
    LineEvent, CommandEvent, EofEvent, ExceptionEvent, ExternalEvent, RawEvent
]


def event_from_dict(data: Dict[str, Any]):
    for t in _ALL_EVENT_TYPES:
        if data["type"] == t.name:
            return t.from_dict(data)
    raise NotImplementedError


async def _cancel(task):
    # more info: https://stackoverflow.com/a/43810272/1113207
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


class ReadResult:

    def __init__(self,
                 data: Any,
                 is_eof: bool,
                 is_exc: bool,
                 traceback_str: str = "",
                 should_exit: bool = True) -> None:
        self.data = data
        self.is_eof = is_eof
        self.is_exc = is_exc
        self.traceback_str = traceback_str
        self.should_exit = should_exit


def _warp_exception_to_event(exc: Exception, uid: str):
    tb_str = io.StringIO()
    traceback.print_exc(file=tb_str)
    ts = time.time_ns()
    return ExceptionEvent(ts, exc, uid=uid, traceback_str=tb_str.getvalue())


_ENCODE = "utf-8"
# _ENCODE = "latin-1"

class PeerSSHClient:
    def __init__(self,
                 stdin: asyncssh.stream.SSHWriter,
                 stdout: asyncssh.stream.SSHReader,
                 stderr: asyncssh.stream.SSHReader,
                 separators: bytes = _DEFAULT_SEPARATORS,
                 uid: str = "",
                 encoding: Optional[str] = None):
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        # stdout/err history
        # create read tasks. they should exists during peer open.
        self._vsc_re = re.compile(rb"\033\]784;([ABPCEFGD])(?:;(.*?))?\007")

        self.uid = uid

    async def send(self, content: str):
        self.stdin.write(content)

    async def send_ctrl_c(self):
        # https://github.com/ronf/asyncssh/issues/112#issuecomment-343318916
        return await self.send('\x03')

    async def _readuntil(self, reader: asyncssh.stream.SSHReader):
        try:
            res = await reader.readuntil(self._vsc_re)
            is_eof = reader.at_eof()
            return ReadResult(res, is_eof, False)
        except asyncio.IncompleteReadError as exc:
            tb_str = io.StringIO()
            traceback.print_exc(file=tb_str)
            is_eof = reader.at_eof()
            print("IncompleteReadError")
            if is_eof:
                return ReadResult(exc.partial, True, False, should_exit=True)
            else:
                print(tb_str.getvalue())
                return ReadResult(exc.partial,
                                  False,
                                  False,
                                  tb_str.getvalue(),
                                  should_exit=False)
            # return ReadResult(exc.partial, True, False)
        except Exception as exc:
            tb_str = io.StringIO()
            traceback.print_exc(file=tb_str)
            return ReadResult(exc, False, True, tb_str.getvalue())

    async def _handle_result(self, res: ReadResult,
                             reader: asyncssh.stream.SSHReader, ts: int,
                             callback: Callable[[Event], Awaitable[None]],
                             is_stderr: bool):
        if res.is_eof:
            await callback(LineEvent(ts, res.data, uid=self.uid))
            retcode: int = -1
            if isinstance(reader.channel, asyncssh.SSHClientChannel):
                retcode_maynone = reader.channel.get_returncode()
                if retcode_maynone is not None:
                    retcode = retcode_maynone
            await callback(EofEvent(ts, retcode, uid=self.uid))
            return True
        elif res.is_exc:
            await callback(
                ExceptionEvent(ts,
                               res.data,
                               uid=self.uid,
                               traceback_str=res.traceback_str))
            # if exception, exit loop
            return res.should_exit
        else:
            match = self._vsc_re.search(res.data)
            data = res.data
            if match:
                cmd_type = match.group(1)
                additional = match.group(2)
                data_line = data[:match.start()]
                cmd_type_s = cmd_type
                if isinstance(cmd_type_s, bytes):
                    cmd_type_s = cmd_type_s.decode("utf-8")
                ce = CommandEvent(ts,
                                  cmd_type_s,
                                  additional,
                                  is_stderr,
                                  uid=self.uid)
                if ce.type == CommandEventType.PROMPT_END:
                    ce.arg = data[:match.start()]
                else:
                    if data_line:
                        await callback(
                            LineEvent(ts,
                                      data[:match.start()],
                                      is_stderr=is_stderr,
                                      uid=self.uid,
                                      is_command=True))
                await callback(ce)
            else:
                await callback(
                    LineEvent(ts, data, is_stderr=is_stderr, uid=self.uid))
        return False

    async def wait_loop_queue(self, callback: Callable[[Event],
                                                       Awaitable[None]],
                              shutdown_task: asyncio.Task):
        """events: stdout/err line, eof, error
        """
        shut_task = shutdown_task
        read_line_task = asyncio.create_task(self._readuntil(self.stdout))
        read_err_line_task = asyncio.create_task(self._readuntil(self.stderr))
        wait_tasks: List[asyncio.Task] = [
            shut_task, read_line_task, read_err_line_task
        ]
        while True:
            (done,
             pending) = await asyncio.wait(wait_tasks,
                                           return_when=asyncio.FIRST_COMPLETED)
            ts = time.time_ns()
            if shutdown_task in done:
                for task in pending:
                    await _cancel(task)
                break
            # if read_line_task in done or read_err_line_task in done:
            if read_line_task in done:
                res = read_line_task.result()
                try:
                    if await self._handle_result(res, self.stdout, ts, callback,
                                                False):
                        break
                except:
                    traceback.print_exc()
                    break
                read_line_task = asyncio.create_task(
                    self._readuntil(self.stdout))
            if read_err_line_task in done:
                res = read_err_line_task.result()
                try:
                    if await self._handle_result(res, self.stderr, ts, callback,
                                                True):
                        break
                except:
                    traceback.print_exc()
                    break

                read_err_line_task = asyncio.create_task(
                    self._readuntil(self.stderr))

            wait_tasks = [shut_task, read_line_task, read_err_line_task]

class SSHRequestType(enum.Enum):
    ChangeSize = 0


class SSHRequest:

    def __init__(self, type: SSHRequestType, data: Any) -> None:
        self.type = type
        self.data = data

class VscodeStyleSSHClientStreamSession(asyncssh.stream.SSHClientStreamSession
                                        ):

    def __init__(self) -> None:
        super().__init__()
        self.callback: Optional[Callable[[Event], Awaitable[None]]] = None
        self.uid = ""

        self.state = CommandEventParseState.VscPromptEnd  # idle
        self._rbuf_max_length = 4000

    def data_received(self, data: bytes, datatype) -> None:
        res = super().data_received(data, datatype)
        if self.callback is not None:
            ts = time.time_ns()
            res_str = data
            loop = asyncio.get_running_loop()
            asyncio.run_coroutine_threadsafe(
                self.callback(RawEvent(ts, res_str, False, self.uid)), loop)
        return res

    async def readuntil(self,
                        separator: object,
                        datatype: asyncssh.DataType,
                        max_separator_len: int = 0) -> AnyStr:
        """Read data from the channel until a separator is seen"""

        if not separator:
            raise ValueError('Separator cannot be empty')

        buf = cast(AnyStr, '' if self._encoding else b'')
        recv_buf = self._recv_buf[datatype]
        is_re = False
        if isinstance(separator, re.Pattern):
            seplen = len(separator.pattern)
            is_re = True
            pat = separator
        else:
            if separator is asyncsshss._NEWLINE:
                seplen = 1
                separators = cast(AnyStr, '\n' if self._encoding else b'\n')
            elif isinstance(separator, (bytes, str)):
                seplen = len(separator)
                separators = re.escape(cast(AnyStr, separator))
            else:
                bar = cast(AnyStr, '|' if self._encoding else b'|')
                seplist = list(cast(Iterable[AnyStr], separator))
                seplen = max(len(sep) for sep in seplist)
                separators = bar.join(re.escape(sep) for sep in seplist)

            pat = re.compile(separators)
        curbuf = 0
        buflen = 0
        async with self._read_locks[datatype]:
            while True:
                while curbuf < len(recv_buf):
                    if isinstance(recv_buf[curbuf], Exception):
                        if buf:
                            recv_buf[:curbuf] = []
                            self._recv_buf_len -= buflen
                            raise asyncio.IncompleteReadError(
                                cast(bytes, buf), None)
                        else:
                            exc = recv_buf.pop(0)

                            if isinstance(exc, SoftEOFReceived):
                                return buf
                            else:
                                raise cast(Exception, exc)

                    newbuf = cast(AnyStr, recv_buf[curbuf])
                    buf += newbuf
                    start = 0
                    # use regex to detect both shell events and \n is very slow if output contains many \r lines.
                    # so we use following code to handle both shell events and `\n`.
                    # if we find the first charactor of shell event, we will wait until the end exists in buffer.
                    # we also limit the maximum buffer length to avoid stuck in the \r (e.g. progress bar).
                    idx_start_all = buf.find(
                        CommandParseSpecialCharactors.StartAll)
                    idx_start = buf.find(CommandParseSpecialCharactors.Start)
                    # ensure if buf start is partial, we should wait for all possible string available.
                    if idx_start != -1:
                        if len(buf) - start >= len(
                                CommandParseSpecialCharactors.StartAll):
                            if idx_start_all == -1:
                                idx_start = -1
                    idx_end = buf.find(CommandParseSpecialCharactors.End)
                    if idx_start_all != -1 and idx_end != -1:
                        if idx_start_all < idx_end:
                            match = pat.search(buf, start)
                            if match:
                                idx = match.end()
                                recv_buf[:curbuf] = []
                                recv_buf[0] = buf[idx:]
                                buf = buf[:idx]
                                self._recv_buf_len -= idx

                                if not recv_buf[0]:
                                    recv_buf.pop(0)
                                self._maybe_resume_reading()
                                return buf
                        else:
                            idx = idx_start_all
                            recv_buf[:curbuf] = []
                            recv_buf[0] = buf[idx:]
                            buf = buf[:idx]
                            self._recv_buf_len -= idx
                            if not recv_buf[0]:
                                recv_buf.pop(0)
                            self._maybe_resume_reading()
                            return buf
                    elif idx_start_all == -1 and idx_end != -1:
                        idx = idx_end + 1
                        recv_buf[:curbuf] = []
                        recv_buf[0] = buf[idx:]
                        buf = buf[:idx]
                        self._recv_buf_len -= idx
                        if not recv_buf[0]:
                            recv_buf.pop(0)
                        self._maybe_resume_reading()
                        return buf
                    elif idx_start_all != -1 and idx_end == -1:
                        if idx_start_all != 0:
                            idx = idx_start_all
                            recv_buf[:curbuf] = []
                            recv_buf[0] = buf[idx:]
                            buf = buf[:idx]
                            self._recv_buf_len -= idx
                            if not recv_buf[0]:
                                recv_buf.pop(0)
                            self._maybe_resume_reading()
                            return buf
                    else:
                        idx = buf.find(b"\n")
                        idx_r_buf = buf.rfind(b"\r")
                        if idx != -1:
                            idx += 1
                            recv_buf[:curbuf] = []
                            recv_buf[0] = buf[idx:]
                            buf = buf[:idx]
                            self._recv_buf_len -= idx
                            if not recv_buf[0]:
                                recv_buf.pop(0)
                            self._maybe_resume_reading()
                            return buf
                        if idx_r_buf >= self._rbuf_max_length:
                            idx = idx_r_buf + 1
                            recv_buf[:curbuf] = []
                            recv_buf[0] = buf[idx:]
                            buf = buf[:idx]
                            self._recv_buf_len -= idx
                            if not recv_buf[0]:
                                recv_buf.pop(0)
                            self._maybe_resume_reading()
                            return buf

                    buflen += len(newbuf)
                    curbuf += 1

                if self._read_paused or self._eof_received:
                    recv_buf[:curbuf] = []
                    self._recv_buf_len -= buflen
                    self._maybe_resume_reading()
                    raise asyncio.IncompleteReadError(cast(bytes, buf), None)

                await self._block_read(datatype)

_T_ret = TypeVar("_T_ret")
P = ParamSpec("P")

async def _get_args_and_params(args, kwargs, code: str, is_func_id: bool):
    return args, kwargs, code, is_func_id

async def _set_ret(ret, exception_msg: Optional[str] = None, ret_ev: Optional[asyncio.Event] = None, ret_container: Optional[dict] = None):
    if ret_container is None or ret_ev is None:
        return # shouldn't happen
    if exception_msg is not None:
        ret_container["exception"] = exception_msg
    else:
        ret_container["ret"] = ret
    ret_ev.set()

class SubprocessRpcClient(abc.ABC):
    def __init__(self, port: int, timeout: int = 10, bkgd_loop: Optional[asyncio.AbstractEventLoop] = None):
        self._port = port
        self._bkgd_loop = bkgd_loop
        self._is_bkgd = self._bkgd_loop is not None
        self.timeout = timeout

    @abc.abstractmethod
    async def run_command(self, cmds: list[str]) -> Optional[str]: ...

    @staticmethod 
    def set_args_to_argserver(args, kwargs, func_id, is_code: bool, bkgd_loop: Optional[asyncio.AbstractEventLoop] = None):
        arg_event_id = f"arg-{uuid.uuid4().hex}"
        ret_event_id = f"ret-{uuid.uuid4().hex}"
        ev = asyncio.Event()
        result = {}
        once_serv_key = f"{_SSH_ARGSERVER_SERV_NAME}.once"
        if bkgd_loop is not None:
            BACKGROUND_SERVER.execute_service(once_serv_key, arg_event_id, partial(_get_args_and_params, args, kwargs, func_id, is_code), loop=bkgd_loop)
            BACKGROUND_SERVER.execute_service(once_serv_key, ret_event_id, partial(_set_ret, ret_ev=ev, ret_container=result), loop=bkgd_loop)
        else:
            prim.get_service(once_serv_key)(arg_event_id, partial(_get_args_and_params, args, kwargs, func_id, is_code))
            prim.get_service(once_serv_key)(ret_event_id, partial(_set_ret, ret_ev=ev, ret_container=result))
        return arg_event_id, ret_event_id, ev, result
        
    @staticmethod 
    def remove_args_to_argserver(arg_event_id: str, ret_event_id: str, is_bkgd: bool):
        off_serv_key = f"{_SSH_ARGSERVER_SERV_NAME}.off"
        if is_bkgd:
            BACKGROUND_SERVER.execute_service(off_serv_key, arg_event_id)
            BACKGROUND_SERVER.execute_service(off_serv_key, ret_event_id)
        else:
            prim.get_service(off_serv_key)(arg_event_id)
            prim.get_service(off_serv_key)(ret_event_id)


    async def _call_base(self, args, kwargs, func_id, is_code: bool, need_result: bool = True):
        arg_event_id, ret_event_id, ev, result = self.set_args_to_argserver(args, kwargs, func_id, is_code, self._is_bkgd)

        run_cmd = f"python -m tensorpc.cli.cmd_rpc_call {arg_event_id} {ret_event_id} {self._port} {need_result}"
        final_cmds = [
            run_cmd,
        ]
        try:
            error_msg = await self.run_command(final_cmds)
            async with async_timeout.timeout(self.timeout):
                await ev.wait()
            if "exception" in result:
                if error_msg:
                    LOGGER.error("Command Message: %s", error_msg)
                raise Exception(result["exception"])
            else:
                return result["ret"]
        finally:
            self.remove_args_to_argserver(arg_event_id, ret_event_id, self._is_bkgd)

        
    async def call(self, func_id: Union[str, Callable[P, _T_ret]], *args: P.args, **kwargs: P.kwargs) -> _T_ret:
        if not isinstance(func_id, str):
            func_id = get_module_id_of_type(func_id)
        return await self._call_base(args, kwargs, func_id, False)

    async def call_with_code(self, func_code: str, *args, **kwargs):
        return await self._call_base(args, kwargs, func_code, True)

    async def call_without_result(self, func_id: Union[str, Callable[P, _T_ret]], *args: P.args, **kwargs: P.kwargs) -> _T_ret:
        if not isinstance(func_id, str):
            func_id = get_module_id_of_type(func_id)
        return await self._call_base(args, kwargs, func_id, False, False)

class SSHRpcClient(SubprocessRpcClient):
    def __init__(self, conn: asyncssh.SSHClientConnection, port: int, shell_init_cmd: str, 
            bkgd_loop: Optional[asyncio.AbstractEventLoop] = None, 
            remote_fwd_listeners: Optional[List[asyncssh.SSHListener]] = None,
            manual_close: bool = False,
            user_init_cmd: str = "",
            host_url: str = ""):
        super().__init__(port, 10, bkgd_loop)
        self._conn = conn

        self._shell_init_cmd = shell_init_cmd
        self._cmd: str = user_init_cmd
        self._remote_fwd_listeners = remote_fwd_listeners
        self._manual_close = manual_close

        self.host_url = host_url

    async def close_and_wait(self):
        if self._manual_close:
            self._conn.close()
            if self._remote_fwd_listeners is not None:
                for listener in self._remote_fwd_listeners:
                    listener.close()
            await self._conn.wait_closed()
        else:
            raise RuntimeError("closed by context manager")

    def set_init_cmd(self, cmd: str):
        self._cmd = cmd

    async def run_command(self, cmds: List[str]) -> Optional[str]:
        final_cmds = cmds.copy()
        if self._cmd != "":
            final_cmds.insert(0, self._cmd)
        final_cmd = " && ".join(final_cmds)
        proc_res = await self._conn.run(f"{self._shell_init_cmd} \"{final_cmd}\"")
        if proc_res.exit_status != 0 and proc_res.exit_status is not None:
            print("proc status", proc_res.exit_status, proc_res.stdout, proc_res.stderr)
            return None
        return None

class SSHClient:

    def __init__(self,
                 url: str,
                 username: str,
                 password: str,
                 known_hosts: Any = None,
                 uid: str = "",
                 encoding: Optional[str] = None,
                 enable_vscode_cmd_util: bool = True) -> None:
        url_parts = url.split(":")
        if len(url_parts) == 1:
            self.url_no_port = url
            self.port = 22
        else:
            self.url_no_port = url_parts[0]
            self.port = int(url_parts[1])
        self.url = url
        self._enable_vscode_cmd_util = enable_vscode_cmd_util
        self.username = username
        self.password = password
        self.known_hosts = known_hosts
        self.uid = uid

        self.bash_file_inited: bool = False
        self.encoding = encoding
        self.tunnel = None

    @classmethod
    def from_ssh_target(cls, target: SSHTarget):
        url = f"{target.hostname}:{target.port}"
        return cls(url,
                   target.username,
                   target.password,
                   target.known_hosts,
                   uid=target.uid)

    async def determine_shell_type_by_conn(self, conn: asyncssh.SSHClientConnection):
        async def _cmd_runner(cmd: str, skip_check: bool = False):
            try:
                result = await conn.run(cmd, check=not skip_check)
                # print(result.stderr, result.stdout)
                if result.stderr:
                    stdout_content = result.stderr
                    if isinstance(stdout_content, (bytes, bytearray)):
                        stdout_content = stdout_content.decode(_ENCODE)
                    elif isinstance(stdout_content, memoryview):
                        stdout_content = stdout_content.tobytes().decode(_ENCODE)
                    
                    return stdout_content
                elif result.stdout:
                    stdout_content = result.stdout
                    if isinstance(stdout_content, (bytes, bytearray)):
                        stdout_content = stdout_content.decode(_ENCODE)
                    elif isinstance(stdout_content, memoryview):
                        stdout_content = stdout_content.tobytes().decode(_ENCODE)
                    return stdout_content
                else:
                    return ""
            except:
                return None 
        res = await terminal_shell_type_detector(_cmd_runner)
        if res is None:
            return ShellInfo("bash", "linux")
        return res 

    @contextlib.asynccontextmanager
    async def simple_connect(self, init_bash: bool = True):
        conn_task = asyncssh.connection.connect(self.url_no_port,
                                                self.port,
                                                username=self.username,
                                                password=self.password,
                                                keepalive_interval=15,
                                                login_timeout=10,
                                                known_hosts=None,
                                                tunnel=self.tunnel)
        conn_ctx = await asyncio.wait_for(conn_task, timeout=10)
        async with conn_ctx as conn:
            assert isinstance(conn, asyncssh.SSHClientConnection)
            
            shell_info = await self.determine_shell_type_by_conn(conn)
            if (not self.bash_file_inited) and init_bash:
                p = determine_hook_path_by_shell_info(shell_info)
                if shell_info.os_type == "windows":
                    # remove CRLF
                    with open(p, "r") as f:
                        content = f.readlines()
                    await conn.run(f'cat > ~/.tensorpc_hooks-bash.sh',
                                   input="\n".join(content))
                else:
                    await asyncsshscp(str(p),
                                      (conn, '~/.tensorpc_hooks-bash.sh'))
                self.bash_file_inited = True
            yield conn

    async def simple_connect_with_rpc(self, user_init_cmd: str = "", init_bash: bool = False):
        conn_task = asyncssh.connection.connect(self.url_no_port,
                                                self.port,
                                                username=self.username,
                                                password=self.password,
                                                keepalive_interval=15,
                                                login_timeout=10,
                                                known_hosts=self.known_hosts,
                                                tunnel=self.tunnel)
        
        conn = await asyncio.wait_for(conn_task, timeout=10)
        rfwd_listeners: List[asyncssh.SSHListener] = []
        try:
            assert isinstance(conn, asyncssh.SSHClientConnection)
            shell_info = await self.determine_shell_type_by_conn(conn)
            if (not self.bash_file_inited) and init_bash:
                await self._sync_sh_init_file(conn, shell_info)
                self.bash_file_inited = True
            loop: Optional[asyncio.AbstractEventLoop] = None
            if is_in_server_context() and check_is_service_available(_SSH_ARGSERVER_SERV_NAME):
                # if some service (such as App) is running, we use the service port
                serv_port = get_server_grpc_port()
            elif BACKGROUND_SERVER.is_started and BACKGROUND_SERVER.service_core.service_units.has_service_unit(_SSH_ARGSERVER_SERV_NAME):
                serv_port = BACKGROUND_SERVER.port
                # when we run main coroutines in background server, we need to use thread-safe version and provide loop.
                loop = asyncio.get_running_loop()
            else:
                raise NotImplementedError("you must run bg server or app to use RPC based ssh")
            # remote forward port that client inside ssh process can connect to server in current process.
            fwd_ports, rfwd_ports, fwd_listeners, rfwd_listeners = await self._handle_forward_ports(conn, None, [serv_port])
            # client inside ssh can use this port to connect to server
            forwarded_port = rfwd_ports[0]
            rpc_client = SSHRpcClient(conn, forwarded_port, 
                self._get_shell_single_cmd(shell_info), bkgd_loop=loop,
                remote_fwd_listeners=rfwd_listeners,
                manual_close=True,
                user_init_cmd=user_init_cmd,
                host_url=self.url_no_port)
            return rpc_client
        except:
            conn.close()
            await conn.wait_closed()
            for listener in rfwd_listeners:
                await listener.wait_closed()
            raise

    @contextlib.asynccontextmanager
    async def simple_connect_with_rpc_ctx(self, user_init_cmd: str = "", init_bash: bool = False):
        client = await self.simple_connect_with_rpc(user_init_cmd, init_bash)
        try:
            yield client
        finally:
            await client.close_and_wait()

    async def create_local_tunnel(self, port_pairs: List[Tuple[int, int]],
                                  shutdown_task: asyncio.Task):
        conn_task = asyncssh.connection.connect(self.url_no_port,
                                                self.port,
                                                username=self.username,
                                                password=self.password,
                                                keepalive_interval=10,
                                                login_timeout=10,
                                                known_hosts=None,
                                                tunnel=self.tunnel)
        conn_ctx = await asyncio.wait_for(conn_task, timeout=10)
        async with conn_ctx as conn:
            wait_tasks = [
                shutdown_task,
            ]
            for p_local, p_remote in port_pairs:
                listener = await conn.forward_local_port(
                    '', p_local, 'localhost', p_remote)
                wait_tasks.append(asyncio.create_task(listener.wait_closed()))
            done, pending = await asyncio.wait(
                wait_tasks, return_when=asyncio.FIRST_COMPLETED)
            return

    async def _sync_sh_init_file(self, conn: asyncssh.SSHClientConnection, shell_type: ShellInfo):
        bash_file_path = determine_hook_path_by_shell_info(shell_type)
        if InWindows:
            # remove CRLF
            with open(bash_file_path, "r") as f:
                content = f.readlines()
            await conn.run(f'cat > ~/.tensorpc_hooks-bash{bash_file_path.suffix}',
                            input="\n".join(content))
        else:
            if shell_type.type == "zsh":
                await asyncsshscp(str(bash_file_path.parent),
                                (conn, f'~/'), recurse=True)
            else:
                await asyncsshscp(str(bash_file_path),
                                (conn, f'~/.tensorpc_hooks-bash{bash_file_path.suffix}'))

    def _get_shell_init_cmd(self, shell_type: ShellInfo, bash_file_path: Optional[Path] = None):
        if bash_file_path is not None:
            init_cmd = f"bash --init-file ~/.tensorpc_hooks-bash{bash_file_path.suffix}"
            init_cmd_2 = ""
            if shell_type.os_type == "windows":
                pwsh_win_cmds = ['-l', '-noexit', '-command', 'try { . ~/.tensorpc_hooks-bash{bash_file_path.suffix} } catch {}{}']
                init_cmd = f"powershell {' '.join(pwsh_win_cmds)}"
                init_cmd_2 = f". ~/.tensorpc_hooks-bash{bash_file_path.suffix}"
                init_cmd_2 = ""                
            elif shell_type.type != "bash" and shell_type.type != "zsh":
                init_cmd =shell_type.type
                init_cmd_2 = f"source ~/.tensorpc_hooks-bash{bash_file_path.suffix}"
            elif shell_type.type == "zsh":
                init_cmd_2 = ""
                user_zdotdir = os.getenv("ZDOTDIR", "$HOME")
                init_cmd = f"export ZDOTDIR=~/.tensorpc_hooks-zsh && export USER_ZDOTDIR={user_zdotdir} && zsh -il"
        else:
            init_cmd = f"bash"
            init_cmd_2 = ""
            if shell_type.os_type == "windows":
                pwsh_win_cmds = ['-l', '-noexit']
                init_cmd = f"powershell {' '.join(pwsh_win_cmds)}"
                init_cmd_2 = ""                
            elif shell_type.type != "bash" and shell_type.type != "zsh":
                init_cmd = shell_type.type
                init_cmd_2 = f""
            elif shell_type.type == "zsh":
                init_cmd_2 = ""
                user_zdotdir = os.getenv("ZDOTDIR", "$HOME")
                init_cmd = f"zsh -il"
        return init_cmd, init_cmd_2

    def _get_shell_single_cmd(self, shell_type: ShellInfo):
        if shell_type.os_type == "windows":
            return f"powershell -c"         
        else:
            return f"{shell_type.type} -ic"

    async def _create_controlled_session(self, conn: asyncssh.SSHClientConnection, 
            shell_type: ShellInfo, init_bash_file: bool = True, 
            request_pty: Union[Literal["force", "auto"], bool] = "force",
            rbuf_max_length: int = 4000,
            term_type: Optional[str] = None,
            env: Optional[MutableMapping[str, str]] = None):
        session: VscodeStyleSSHClientStreamSession
        bash_file_path = determine_hook_path_by_shell_info(shell_type)
        if init_bash_file:
            await self._sync_sh_init_file(conn, shell_type)
        init_cmd, init_cmd_2 = self._get_shell_init_cmd(shell_type, bash_file_path if self._enable_vscode_cmd_util else None)

        chan, session = await conn.create_session(
            VscodeStyleSSHClientStreamSession,
            init_cmd,
            request_pty=request_pty,
            # we don't use this env here
            env=list(env.items()) if env is not None else None,
            term_type=term_type,
            encoding=self.encoding) # type: ignore
        session._rbuf_max_length = rbuf_max_length
        stdin, stdout, stderr = (
            asyncssh.stream.SSHWriter(session, chan),
            asyncssh.stream.SSHReader(session, chan),
            asyncssh.stream.SSHReader(
                session, chan,
                asyncssh.constants.EXTENDED_DATA_STDERR))
        if init_cmd_2:
            stdin.write((init_cmd_2 + "\n").encode("utf-8"))
        # set ignore space in history
        if shell_type.os_type == "windows":
            stdin.write(b"Set-PSReadlineOption -AddToHistoryHandler {\n"
                         b"param([string]$line)\n"
                         b"return $line.Length -gt 3 -and $line[0] -ne ' ' -and $line[0] -ne ';'\n"
                         b"}\n")
        else:
            if shell_type.type == "zsh":
                stdin.write(b"setopt HIST_IGNORE_SPACE\n")
            elif shell_type.type == "bash":
                stdin.write(b"export HISTCONTROL=ignorespace\n")
        return chan, session, stdin, stdout, stderr

    async def connect_queue(
            self,
            inp_queue: asyncio.Queue,
            callback: Callable[[Event], Awaitable[None]],
            shutdown_task: asyncio.Task,
            env: Optional[MutableMapping[str, str]] = None,
            forward_ports: Optional[List[int]] = None,
            r_forward_ports: Optional[List[Union[Tuple[int, int],
                                                 int]]] = None,
            env_port_modifier: Optional[Callable[
                [List[int], List[int], MutableMapping[str, str]], None]] = None,
            exit_callback: Optional[Callable[[], Awaitable[None]]] = None,
            client_ip_callback: Optional[Callable[[str], None]] = None,
            init_event: Optional[asyncio.Event] = None,
            exit_event: Optional[asyncio.Event] = None,
            term_type: Optional[str] = "xterm-256color",
            request_pty: Union[Literal["force", "auto"], bool] = "force",
            rbuf_max_length: int = 4000,
            enable_raw_event: bool = True):
        if env is None:
            env = {}
        # TODO better keepalive
        session: VscodeStyleSSHClientStreamSession
        try:
            conn_task = asyncssh.connection.connect(self.url_no_port,
                                                    self.port,
                                                    username=self.username,
                                                    password=self.password,
                                                    keepalive_interval=10,
                                                    login_timeout=10,
                                                    known_hosts=None,
                                                    tunnel=self.tunnel)
            conn_ctx = await asyncio.wait_for(conn_task, timeout=10)
            async with conn_ctx as conn:
                assert isinstance(conn, asyncssh.SSHClientConnection)
                shell_type = await self.determine_shell_type_by_conn(conn)
                
                if client_ip_callback is not None and shell_type.os_type != "windows":
                    # TODO if fail?
                    result = await conn.run(
                        " echo $SSH_CLIENT | awk '{ print $1}'", check=True)
                    if result.stdout is not None:
                        stdout_content = result.stdout
                        if isinstance(stdout_content, (bytes, bytearray)):
                            stdout_content = stdout_content.decode(_ENCODE)
                        elif isinstance(stdout_content, memoryview):
                            stdout_content = stdout_content.tobytes().decode(
                                _ENCODE)
                        if stdout_content.strip() == "::1":
                            stdout_content = "localhost"
                        client_ip_callback(stdout_content)
                if not self.bash_file_inited:
                    init_bash_file = True
                    self.bash_file_inited = True
                else:
                    init_bash_file = False
                chan, session, stdin, stdout, stderr = await self._create_controlled_session(conn, 
                    shell_type, init_bash_file=init_bash_file, request_pty=request_pty,
                    rbuf_max_length=rbuf_max_length, term_type=term_type)
                session.uid = self.uid
                if enable_raw_event:
                    session.callback = callback
                peer_client = PeerSSHClient(stdin,
                                            stdout,
                                            stderr,
                                            uid=self.uid)
                loop_task = asyncio.create_task(
                    peer_client.wait_loop_queue(callback, shutdown_task))
                wait_tasks = [
                    asyncio.create_task(inp_queue.get()), shutdown_task,
                    loop_task
                ]
                fwd_ports, rfwd_ports, fwd_listeners, rfwd_listeners = await self._handle_forward_ports(conn, forward_ports, r_forward_ports)
                for listener in rfwd_listeners:
                    wait_tasks.append(asyncio.create_task(listener.wait_closed()))
                for listener in fwd_listeners:
                    wait_tasks.append(asyncio.create_task(listener.wait_closed()))
                if env_port_modifier is not None and (rfwd_ports or fwd_ports):
                    env_port_modifier(fwd_ports, rfwd_ports, env)
                if init_event is not None:
                    init_event.set()
                if env:
                    if self.encoding is None:
                        cmds2: List[bytes] = []
                        cmds2.append(f"echo \"{TENSORPC_ASYNCSSH_ENV_INIT_INDICATE}\"".encode("utf-8"))
                        for k, v in env.items():
                            if shell_type.os_type == "windows":
                                cmds2.append(f"$Env:{k} = '{v}'".encode("utf-8"))
                            else:
                                cmds2.append(f"export {k}=\"{v}\"".encode("utf-8"))
                        if shell_type.os_type == "windows":
                            stdin.write(b" " + b"; ".join(cmds2) + b"\n")
                        else:
                            stdin.write(b" " + b" && ".join(cmds2) + b"\n")
                    else:
                        cmds: List[str] = []
                        cmds.append(f"echo \"{TENSORPC_ASYNCSSH_ENV_INIT_INDICATE}\"")
                        for k, v in env.items():
                            if shell_type.os_type == "windows":
                                cmds.append(f"$Env:{k} = '{v}'")
                            else:
                                cmds.append(f"export {k}=\"{v}\"")
                        if shell_type.os_type == "windows":
                            stdin.write(" " + "; ".join(cmds) + "\n")
                        else:
                            stdin.write(" " + " && ".join(cmds) + "\n")
                while True:
                    done, pending = await asyncio.wait(
                        wait_tasks, return_when=asyncio.FIRST_COMPLETED)
                    if shutdown_task in done:
                        for task in pending:
                            await _cancel(task)
                        await callback(EofEvent(time.time_ns(), uid=self.uid))
                        break
                    if loop_task in done:
                        break
                    text = wait_tasks[0].result()
                    if isinstance(text, SSHRequest):
                        if text.type == SSHRequestType.ChangeSize:
                            chan.change_terminal_size(text.data[0],
                                                      text.data[1])
                    else:
                        if isinstance(text, bytes):
                            if self.encoding is not None:
                                stdin.write(text.decode(self.encoding))
                            else:
                                stdin.write(text)
                        else:
                            if self.encoding is None:
                                stdin.write(text.encode("utf-8"))
                            else:
                                stdin.write(text)
                    wait_tasks = [
                        asyncio.create_task(inp_queue.get()), shutdown_task, loop_task
                    ]
                await loop_task
        except Exception as exc:
            await callback(_warp_exception_to_event(exc, self.uid))
        finally:
            if init_event:
                init_event.set()
            if exit_event is not None:
                exit_event.set()
            if exit_callback is not None:
                await exit_callback()

    async def _handle_forward_ports(self, conn: asyncssh.SSHClientConnection, forward_ports: Optional[List[int]], r_forward_ports: Optional[List[Union[Tuple[int, int], int]]]):
        rfwd_ports: List[int] = []
        fwd_ports: List[int] = []
        rfwd_listeners: List[asyncssh.SSHListener] = []
        fwd_listeners: List[asyncssh.SSHListener] = []
        if r_forward_ports is not None:
            for p in r_forward_ports:
                if isinstance(p, (tuple, list)):
                    listener = await conn.forward_remote_port(
                        '', p[0], 'localhost', p[1])
                else:
                    listener = await conn.forward_remote_port(
                        '', 0, 'localhost', p)

                rfwd_ports.append(listener.get_port())
                LOGGER.warning(
                    f'Listening on Remote port {p} <- {listener.get_port()}...'
                )
                rfwd_listeners.append(listener)
        if forward_ports is not None:
            for p in forward_ports:
                listener = await conn.forward_local_port(
                    '', 0, 'localhost', p)
                fwd_ports.append(listener.get_port())
                LOGGER.warning(
                    f'Listening on Local port {listener.get_port()} -> {p}...'
                )
                fwd_listeners.append(listener)
        return fwd_ports, rfwd_ports, fwd_listeners, rfwd_listeners

def run_ssh_rpc_call(arg_event_id: str, ret_event_id: str, rf_port: int, need_result: bool = True):
    # should be run inside remote ssh 
    with RemoteManager(f"localhost:{rf_port}") as robj:
        args, kwargs, code, is_func_code = robj.remote_call(f"{_SSH_ARGSERVER_SERV_NAME}.call_event", arg_event_id)
        try:
            func = import_dynamic_func(code, not is_func_code)
            res = func(*args, **kwargs)
            if need_result:
                robj.remote_call(f"{_SSH_ARGSERVER_SERV_NAME}.call_event", ret_event_id, res)
        except Exception as exc:
            exc_msg_ss = io.StringIO()
            traceback.print_exc(file=exc_msg_ss)
            exc_msg = exc_msg_ss.getvalue()
            robj.remote_call(f"{_SSH_ARGSERVER_SERV_NAME}.call_event", ret_event_id, None, exc_msg)

