import asyncio
import bisect

import abc
import asyncio
import bisect
import contextlib
import enum
import io
import re
import sys
import time
from asyncio.tasks import FIRST_COMPLETED
from collections import deque
from dataclasses import dataclass
import traceback
from typing import Any, AnyStr, Awaitable, Callable, Deque, Dict, List, Optional, Set, Tuple, Type, Union
from contextlib import suppress
import asyncssh
import tensorpc
from tensorpc.apps.flow.constants import TENSORPC_READUNTIL
from tensorpc.constants import PACKAGE_ROOT
import getpass
from asyncssh.scp import scp as asyncsshscp
# 7-bit C1 ANSI sequences
ANSI_ESCAPE_REGEX = re.compile(
    br'''
    (?: # either 7-bit C1, two bytes, ESC Fe (omitting CSI)
        \x1B
        [@-Z\\-_]
    |   # or a single 8-bit byte Fe (omitting CSI)
        [\x80-\x9A\x9C-\x9F]
    |   # or CSI + control codes
        (?: # 7-bit CSI, ESC [ 
            \x1B\[
        |   # 8-bit CSI, 9B
            \x9B
        )
        [0-?]*  # Parameter bytes
        [ -/]*  # Intermediate bytes
        [@-~]   # Final byte
    )
''', re.VERBOSE)
ANSI_ESCAPE_REGEX_8BIT = re.compile(
    br'''
    (?: # either 7-bit C1, two bytes, ESC Fe (omitting CSI)
        \x1B
        [@-Z\\-_]
    |   # or a single 8-bit byte Fe (omitting CSI)
        [\x80-\x9A\x9C-\x9F]
    |   # or CSI + control codes
        (?: # 7-bit CSI, ESC [ 
            \x1B\[
        |   # 8-bit CSI, 9B
            \x9B
        )
        [0-?]*  # Parameter bytes
        [ -/]*  # Intermediate bytes
        [@-~]   # Final byte
    )
''', re.VERBOSE)


class CommandEventType(enum.Enum):
    PROMPT_START = "A"
    PROMPT_END = "B"
    COMMAND_OUTPUT_START = "C"
    COMMAND_COMPLETE = "D"
    CURRENT_COMMAND = "E"

    UPDATE_CWD = "P"
    CONTINUATION_START = "F"
    CONTINUATION_END = "G"


_DEFAULT_SEPARATORS = rb"(?:\r\n)|(?:\n)|(?:\r)|(?:\033\]784;[ABPCFGD](?:;(.*?))?\007)"
# _DEFAULT_SEPARATORS = "\n"


def remove_ansi_seq(string: Union[str, bytes]):
    # https://stackoverflow.com/questions/14693701/how-can-i-remove-the-ansi-escape-sequences-from-a-string-in-python
    if isinstance(string, str):
        return ANSI_ESCAPE_REGEX.sub(b'',
                                     string.encode("utf-8")).decode("utf-8")
    else:
        return ANSI_ESCAPE_REGEX.sub(b'', string).decode("utf-8")


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

    def __eq__(self, other: Union["Event", int]):
        if isinstance(other, Event):
            other = other.timestamp
        return self.timestamp == other

    def __ne__(self, other: Union["Event", int]):
        if isinstance(other, Event):
            other = other.timestamp
        return self.timestamp != other


class EofEvent(Event):
    name = "EofEvent"

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


class LineEvent(Event):
    name = "LineEvent"

    def __init__(self,
                 timestamp: int,
                 line: str,
                 is_stderr=False,
                 uid: str = ""):
        super().__init__(timestamp, is_stderr, uid)
        self.line = line

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
    name = "RawEvent"

    def __init__(self,
                 timestamp: int,
                 raw: Union[str, bytes],
                 is_stderr=False,
                 uid: str = ""):
        super().__init__(timestamp, is_stderr, uid)
        self.raw = raw

    def __repr__(self):
        r = self.raw
        if not isinstance(r, bytes):
            r = r.encode("utf-8")
        return "{}({}|{}|raw={})".format(self.name, self.is_stderr,
                                         self.timestamp,
                                         r)

    def to_dict(self):
        res = super().to_dict()
        res["raw"] = self.raw
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        assert cls.name == data["type"]
        return cls(data["ts"], data["line"], data["is_stderr"], data["uid"])


class ExceptionEvent(Event):
    name = "ExceptionEvent"

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
    name = "CommandEvent"

    def __init__(self,
                 timestamp: int,
                 type: str,
                 arg: Optional[Union[str, bytes]],
                 is_stderr=False,
                 uid: str = ""):
        super().__init__(timestamp, is_stderr, uid)
        self.type = CommandEventType(type)
        self.arg = arg

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
    LineEvent, CommandEvent, EofEvent, ExceptionEvent
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
                 traceback_str: str = "") -> None:
        self.data = data
        self.is_eof = is_eof
        self.is_exc = is_exc
        self.traceback_str = traceback_str


def _warp_exception_to_event(exc: Exception, uid: str):
    tb_str = io.StringIO()
    traceback.print_exc(file=tb_str)
    ts = time.time_ns()
    return ExceptionEvent(ts, exc, uid=uid, traceback_str=tb_str.getvalue())

_ENCODE = "utf-8"
# _ENCODE = "latin-1"


class PeerSSHClient:
    """
    during handle stdout/err message, client will 
    1. identifier extraction
    2. code path detection
    """

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
        if encoding is None:
            self.separators = separators
            self._vsc_re = re.compile(rb"\033\]784;([ABPCFGD])(?:;(.*))?\007")
        else:
            self.separators = separators.decode("utf-8")
            self._vsc_re = re.compile(r"\033\]784;([ABPCFGD])(?:;(.*))?\007")

        self.uid = uid


    async def send(self, content: str):
        self.stdin.write(content)

    async def send_ctrl_c(self):
        # https://github.com/ronf/asyncssh/issues/112#issuecomment-343318916
        return await self.send('\x03')

    async def _readuntil(self, reader: asyncssh.stream.SSHReader):
        try:
            # print(separators)
            res = await reader.readuntil(self.separators)
            is_eof = reader.at_eof()
            return ReadResult(res, is_eof, False)
        except asyncio.IncompleteReadError as exc:
            # print("WTFWTF")
            return ReadResult(exc.partial, True, False)
        except Exception as exc:
            tb_str = io.StringIO()
            traceback.print_exc(file=tb_str)
            return ReadResult(exc, False, True, tb_str.getvalue())

    # def _parse_line(self, data: str):

    async def _handle_result(self, res: ReadResult,
                             reader: asyncssh.stream.SSHReader, ts: int,
                             callback: Callable[[Event], Awaitable[None]],
                             is_stderr: bool):
        if res.is_eof:
            await callback(LineEvent(ts, res.data, uid=self.uid))
            await callback(
                EofEvent(ts, reader.channel.get_returncode(), uid=self.uid))
            return True
        elif res.is_exc:
            await callback(
                ExceptionEvent(ts,
                               res.data,
                               uid=self.uid,
                               traceback_str=res.traceback_str))
            # if exception, exit loop
            return True
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
                                      uid=self.uid))
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
                if await self._handle_result(res, self.stdout, ts, callback,
                                             False):
                    break
                read_line_task = asyncio.create_task(
                    self._readuntil(self.stdout))
            if read_err_line_task in done:
                res = read_err_line_task.result()
                if await self._handle_result(res, self.stderr, ts, callback,
                                             True):
                    break
                read_err_line_task = asyncio.create_task(
                    self._readuntil(self.stderr))

            wait_tasks = [shut_task, read_line_task, read_err_line_task]


async def wait_queue_until_event(handler: Callable[[Any], None],
                                 q: asyncio.Queue, ev: asyncio.Event):
    q_get_task = asyncio.create_task(q.get())
    shut_task = asyncio.create_task(ev.wait())
    wait_tasks: List[asyncio.Task] = [q_get_task, shut_task]
    while True:
        (done,
         pending) = await asyncio.wait(wait_tasks,
                                       return_when=asyncio.FIRST_COMPLETED)
        if ev.is_set():
            for task in pending:
                await _cancel(task)
            break
        if q_get_task in done:
            handler(q_get_task.result())
            q_get_task = asyncio.create_task(q.get())
        wait_tasks = [q_get_task, shut_task]


class SSHRequestType(enum.Enum):
    ChangeSize = 0


class SSHRequest:

    def __init__(self, type: SSHRequestType, data: Any) -> None:
        self.type = type
        self.data = data


class MySSHClientStreamSession(asyncssh.stream.SSHClientStreamSession):

    def __init__(self) -> None:
        super().__init__()
        self.callback: Optional[Callable[[Event], Awaitable[None]]] = None
        self.uid = ""
        self.encoding: Optional[str] = None

    def data_received(self, data: AnyStr, datatype) -> None:
        res = super().data_received(data, datatype)
        if self.callback is not None:
            ts = time.time_ns()
            if self.encoding is not None:
                if isinstance(data, bytes):
                    res_str = data.decode(_ENCODE)
                else:
                    res_str = data
            loop = asyncio.get_running_loop()
            asyncio.run_coroutine_threadsafe(
                self.callback(RawEvent(ts, res_str, False, self.uid)), loop)
        return res


class SSHClient:

    def __init__(self,
                 url: str,
                 username: str,
                 password: str,
                 known_hosts,
                 uid: str = "",
                 encoding: Optional[str] = None) -> None:
        url_parts = url.split(":")
        if len(url_parts) == 1:
            self.url_no_port = url
            self.port = 22
        else:
            self.url_no_port = url_parts[0]
            self.port = int(url_parts[1])
        self.username = username
        self.password = password
        self.known_hosts = known_hosts
        self.uid = uid

        self.bash_file_inited: bool = False
        self.encoding = encoding

    @contextlib.asynccontextmanager
    async def simple_connect(self):
        async with asyncssh.connection.connect(self.url_no_port,
                                    self.port,
                                    username=self.username,
                                    password=self.password,
                                    keepalive_interval=15,
                                    known_hosts=None) as conn:
            if not self.bash_file_inited:
                p = PACKAGE_ROOT / "autossh" / "media" / "hooks-bash.sh"
                await asyncsshscp(str(p), (conn, '~/.tensorpc_hooks-bash.sh'))
                self.bash_file_inited = True
            yield conn

    async def simple_run_command(self, cmd: str):
        async with self.simple_connect() as conn:
            stdin, stdout, stderr = await conn.open_session(
                "bash --init-file ~/.tensorpc_hooks-bash.sh",
                request_pty="force")
            stdin.write(cmd + "\n")
            line = await stdout.readuntil(TENSORPC_READUNTIL)
            return line

    async def connect_queue(
            self,
            inp_queue: asyncio.Queue,
            callback: Callable[[Event], Awaitable[None]],
            shutdown_task: asyncio.Task,
            env: Optional[Dict[str, str]] = None,
            forward_ports: Optional[List[int]] = None,
            r_forward_ports: Optional[List[Union[Tuple[int, int], int]]] = None,
            env_port_modifier: Optional[Callable[
                [List[int], List[int], Dict[str, str]], None]] = None,
            exit_callback: Optional[Callable[[], Awaitable[None]]] = None,
            client_ip_callback: Optional[Callable[[str], None]] = None,
            init_event: Optional[asyncio.Event] = None,
            exit_event: Optional[asyncio.Event] = None):
        if env is None:
            env = {}
        # TODO better keepalive
        try:
            async with asyncssh.connection.connect(self.url_no_port,
                                        self.port,
                                        username=self.username,
                                        password=self.password,
                                        keepalive_interval=10,
                                        known_hosts=None) as conn:
                if not self.bash_file_inited:
                    p = PACKAGE_ROOT / "autossh" / "media" / "hooks-bash.sh"
                    await asyncsshscp(str(p),
                                       (conn, '~/.tensorpc_hooks-bash.sh'))
                    self.bash_file_inited = True
                if client_ip_callback is not None:
                    # TODO if fail?
                    result = await conn.run(
                        "echo $SSH_CLIENT | awk '{ print $1}'", check=True)
                    if result.stdout is not None:
                        stdout_content = result.stdout
                        if isinstance(stdout_content, bytes):
                            stdout_content = stdout_content.decode(_ENCODE)
                        client_ip_callback(stdout_content)

                chan, session = await conn.create_session(
                    MySSHClientStreamSession,
                    "bash --init-file ~/.tensorpc_hooks-bash.sh",
                    request_pty="force",
                    encoding=self.encoding)  # type: ignore
                # chan, session = await conn.create_session(
                #             MySSHClientStreamSession, request_pty="force") # type: ignore

                session: MySSHClientStreamSession
                session.uid = self.uid
                session.callback = callback
                # stdin, stdout, stderr = await conn.open_session(
                #     "bash --init-file ~/.tensorpc_hooks-bash.sh",
                #     request_pty="force")
                stdin, stdout, stderr = (
                    asyncssh.stream.SSHWriter(session, chan),
                    asyncssh.stream.SSHReader(session, chan),
                    asyncssh.stream.SSHReader(
                        session, chan,
                        asyncssh.constants.EXTENDED_DATA_STDERR))

                peer_client = PeerSSHClient(stdin,
                                            stdout,
                                            stderr,
                                            uid=self.uid,
                                            encoding=self.encoding)
                loop_task = asyncio.create_task(
                    peer_client.wait_loop_queue(callback, shutdown_task))
                wait_tasks = [
                    asyncio.create_task(inp_queue.get()), shutdown_task,
                    loop_task
                ]
                rfwd_ports: List[int] = []
                fwd_ports: List[int] = []

                if r_forward_ports is not None:
                    for p in r_forward_ports:
                        if isinstance(p, (tuple, list)):
                            listener = await conn.forward_remote_port(
                                '', p[0], 'localhost', p[1])
                        else:
                            listener = await conn.forward_remote_port(
                                '', 0, 'localhost', p)

                        rfwd_ports.append(listener.get_port())
                        print(f'Listening on Remote port {p} <- {listener.get_port()}...')
                        wait_tasks.append(
                            asyncio.create_task(listener.wait_closed()))
                if forward_ports is not None:
                    for p in forward_ports:
                        listener = await conn.forward_local_port(
                            '', 0, 'localhost', p)
                        fwd_ports.append(listener.get_port())
                        print(f'Listening on Local port {listener.get_port()} -> {p}...')
                        wait_tasks.append(
                            asyncio.create_task(listener.wait_closed()))
                # await listener.wait_closed()
                if env_port_modifier is not None and (rfwd_ports or fwd_ports):
                    env_port_modifier(fwd_ports, rfwd_ports, env)
                if init_event is not None:
                    init_event.set()
                if env:
                    if self.encoding is None:
                        cmds2: List[bytes] = []
                        for k, v in env.items():
                            cmds2.append(f"export {k}={v}".encode("utf-8"))
                        stdin.write(b" && ".join(cmds2) + b"\n")
                    else:
                        cmds: List[str] = []
                        for k, v in env.items():
                            cmds.append(f"export {k}={v}")
                        stdin.write(" && ".join(cmds) + "\n")
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
                            # print("CHANGE SIZE", text.data)
                            chan.change_terminal_size(text.data[0],
                                                      text.data[1])
                    else:
                        # print("INPUTWTF", text.encode("utf-8"))
                        if self.encoding is None:
                            stdin.write(text.encode("utf-8"))
                        else:
                            stdin.write(text)

                    wait_tasks = [
                        asyncio.create_task(inp_queue.get()), shutdown_task
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


async def main2():
    from prompt_toolkit.shortcuts.prompt import PromptSession
    prompt_session = PromptSession(">")

    # with tensorpc.RemoteManager("localhost:51051") as robj:
    def handler(ev: Event):
        # print(ev)
        if isinstance(ev, CommandEvent):
            if ev.type == CommandEventType.PROMPT_END:
                print(ev.arg, end="", flush=True)
            # tensorpc.simple_remote_call("localhost:51051", "tensorpc.services.collection:FileOps.print_in_server", str(ev).encode("utf-8"))
            # robj.remote_call("tensorpc.services.collection:FileOps.print_in_server", str(ev))
            # print(ev)
        if isinstance(ev, LineEvent):
            # line = remove_ansi_seq(ev.line)
            # print("\033]633;A\007" in ev.line)
            # tensorpc.simple_remote_call("localhost:51051", "tensorpc.services.collection:FileOps.print_in_server", str(ev).encode("utf-8"))

            print(ev.line, end="")

    username = input("username:")
    password = getpass.getpass("password:")
    async with asyncssh.connection.connect('localhost',
                                username=username,
                                password=password,
                                known_hosts=None) as conn:
        p = PACKAGE_ROOT / "autossh" / "media" / "hooks-bash.sh"
        await asyncsshscp(str(p), (conn, '~/.tensorpc_hooks-bash.sh'))
        stdin, stdout, stderr = await conn.open_session(
            "bash --init-file ~/.tensorpc_hooks-bash.sh", request_pty="force")
        print(stdin, stdout, stderr)
        peer_client = PeerSSHClient(stdin, stdout, stderr)

        q = asyncio.Queue()

        async def callback(ev: Event):
            await q.put(ev)

        shutdown_ev = asyncio.Event()
        shutdown_task = asyncio.create_task(shutdown_ev.wait())
        task = asyncio.create_task(
            peer_client.wait_loop_queue(callback, shutdown_task))
        task2 = asyncio.create_task(
            wait_queue_until_event(handler, q, shutdown_ev))

        while True:
            try:
                text = await prompt_session.prompt_async("")
                text = text.strip()
                if text == "exit":
                    shutdown_ev.set()
                    break
                stdin.write(text + "\n")
            except KeyboardInterrupt:
                shutdown_ev.set()
                break
        await asyncio.gather(task, task2)


async def main3():
    from prompt_toolkit.shortcuts.prompt import PromptSession
    prompt_session = PromptSession(">")
    shutdown_ev = asyncio.Event()

    async def handler(ev: Event):
        if isinstance(ev, CommandEvent):
            if ev.type == CommandEventType.PROMPT_END:
                print(ev.arg, end="", flush=True)
        if isinstance(ev, LineEvent):
            # print(ev.line.encode("utf-8"))
            print(ev.line, end="")
        if isinstance(ev, ExceptionEvent):
            print("ERROR", ev.data)
            shutdown_ev.set()

    username = input("username:")
    password = getpass.getpass("password:")
    client = SSHClient('localhost',
                       username=username,
                       password=password,
                       known_hosts=None)
    q = asyncio.Queue()
    shutdown_task = asyncio.create_task(shutdown_ev.wait())
    task = asyncio.create_task(
        client.connect_queue(q,
                             handler,
                             shutdown_task,
                             r_forward_ports=[51051]))
    while True:
        try:
            text = await prompt_session.prompt_async("")
            text = text.strip()
            if text == "exit":
                shutdown_ev.set()
                break
            await q.put(text + "\n")
            # stdin.write(text + "\n")
        except KeyboardInterrupt:
            shutdown_ev.set()
            break
    await task


if __name__ == "__main__":
    asyncio.run(main3())
