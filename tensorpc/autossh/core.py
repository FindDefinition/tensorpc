import asyncio
import bisect

import abc
import asyncio
import bisect
import enum
import re
import sys
import time
from asyncio.tasks import FIRST_COMPLETED
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional, Set, Tuple, Type, Union
from contextlib import suppress
import asyncssh
import tensorpc
from tensorpc.constants import PACKAGE_ROOT
import getpass
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


class CommandEventType(enum.Enum):
    PROMPT_START = "A"
    PROMPT_END = "B"
    COMMAND_OUTPUT_START = "C"
    COMMAND_COMPLETE = "D"
    UPDATE_CWD = "P"
    CONTINUATION_START = "F"
    CONTINUATION_END = "G"


_DEFAULT_SEPARATORS = r"(?:\r\n)|(?:\n)|(?:\r)|(?:\033\]784;[ABPCFGD](?:;(.*?))?\007)"
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
        }


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


class ExceptionEvent(Event):
    name = "ExceptionEvent"

    def __init__(self,
                 timestamp: int,
                 data: Any,
                 is_stderr=False,
                 uid: str = ""):
        super().__init__(timestamp, is_stderr, uid)
        self.data = data


class CommandEvent(Event):
    name = "CommandEvent"

    def __init__(self,
                 timestamp: int,
                 type: str,
                 arg: Optional[str],
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


async def _cancel(task):
    # more info: https://stackoverflow.com/a/43810272/1113207
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


class ReadResult:
    def __init__(self, data: Any, is_eof: bool, is_exc: bool) -> None:
        self.data = data
        self.is_eof = is_eof
        self.is_exc = is_exc


class PeerSSHClient:
    """
    during handle stdout/err message, client will 
    1. identifier extraction
    2. code path detection
    """
    def __init__(self,
                 stdin: asyncssh.SSHWriter,
                 stdout: asyncssh.SSHReader,
                 stderr: asyncssh.SSHReader,
                 separators: str = _DEFAULT_SEPARATORS,
                 uid: str = ""):
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        # stdout/err history
        # create read tasks. they should exists during peer open.
        self.separators = separators
        self.uid = uid

        self._vsc_re = re.compile(r"\033\]784;([ABPCFGD])(?:;(.*))?\007")

    async def send(self, content: str):
        self.stdin.write(content)

    async def send_ctrl_c(self):
        # https://github.com/ronf/asyncssh/issues/112#issuecomment-343318916
        return await self.send('\x03')

    async def _readuntil(self, reader: asyncssh.SSHReader):
        if isinstance(self.separators, str):
            separators = self.separators
            if reader._session._encoding:
                separators = separators
            else:
                separators = separators.encode("utf-8")
        else:
            separators = []
            for separator in self.separators:
                if reader._session._encoding:
                    separator = separator
                else:
                    separator = separator.encode("utf-8")
                separators.append(separator)
        try:
            # print(separators)
            res = await reader.readuntil(separators)
            is_eof = reader.at_eof()
            # print("ISEOF", is_eof)
            return ReadResult(res, is_eof, False)
        except asyncio.IncompleteReadError as exc:
            # print("WTFWTF")
            return ReadResult(exc.partial, True, False)
        except Exception as exc:
            return ReadResult(exc, False, True)

    # def _parse_line(self, data: str):

    async def _handle_result(self, res: ReadResult, reader: asyncssh.SSHReader,
                             ts: int, callback: Callable[[Event],
                                                         Awaitable[None]],
                             is_stderr: bool):
        if res.is_eof:
            await callback(LineEvent(ts, res.data, uid=self.uid))
            await callback(
                EofEvent(ts, reader.channel.get_returncode(), uid=self.uid))
            return True
        elif res.is_exc:
            await callback(ExceptionEvent(ts, res.data, uid=self.uid))
            # if exception, exit loop
            return True
        else:
            match = self._vsc_re.search(res.data)
            data = res.data
            if match:
                cmd_type = match.group(1)
                additional = match.group(2)
                data_line = data[:match.start()]
                ce = CommandEvent(ts, cmd_type, additional, is_stderr, uid=self.uid)
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


class SSHClient:
    def __init__(self,
                 url: str,
                 username: str,
                 password: str,
                 known_hosts,
                 uid: str = "") -> None:
        url_parts = url.split(":")
        if len(url_parts) == 1:

            self.url = url
            self.port = 22
        else:
            self.url = url_parts[0]
            self.port = int(url_parts[1])
        self.username = username
        self.password = password
        self.known_hosts = known_hosts
        self.uid = uid

    async def connect_queue(self, inp_queue: asyncio.Queue,
                            callback: Callable[[Event], Awaitable[None]],
                            shutdown_task: asyncio.Task,
                            env: Optional[Dict[str, str]] = None,
                            r_forward_ports: Optional[List[int]] = None,
                            env_port_modifier: Optional[Callable[[List[int], Dict[str, str]], None]] = None):
        if env is None:
            env = {}
        async with asyncssh.connect(self.url, self.port,
                                    username=self.username,
                                    password=self.password,
                                    known_hosts=None) as conn:
            p = PACKAGE_ROOT / "autossh" / "media" / "hooks-bash.sh"
            await asyncssh.scp(str(p), (conn, '~/.tensorpc_hooks-bash.sh'))
            stdin, stdout, stderr = await conn.open_session(
                "bash --init-file ~/.tensorpc_hooks-bash.sh",
                request_pty="force")
            # print("WTF")

            peer_client = PeerSSHClient(stdin, stdout, stderr, uid=self.uid)
            loop_task = asyncio.create_task(
                peer_client.wait_loop_queue(callback, shutdown_task))
            wait_tasks = [asyncio.create_task(inp_queue.get()), shutdown_task]
            if r_forward_ports is not None:
                for p in r_forward_ports:
                    listener = await conn.forward_remote_port('', 0, 'localhost', p)
                    print('Listening on port %s...' % listener.get_port())
                    wait_tasks.append(asyncio.create_task(listener.wait_closed()))
                # await listener.wait_closed()
                if env_port_modifier is not None:
                    env_port_modifier(r_forward_ports, env)
            if env:
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
                    break
                text = wait_tasks[0].result()
                # text = text.strip()
                # print("INPUT", text)
                stdin.write(text)
                wait_tasks = [
                    asyncio.create_task(inp_queue.get()), shutdown_task
                ]
            await loop_task


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
    async with asyncssh.connect('localhost',
                                username=username,
                                password=password,
                                known_hosts=None) as conn:
        p = PACKAGE_ROOT / "autossh" / "media" / "hooks-bash.sh"
        await asyncssh.scp(str(p), (conn, '~/.tensorpc_hooks-bash.sh'))
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
    task = asyncio.create_task(client.connect_queue(q, handler, shutdown_task,
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
