"""autossh design
Client: 
1. run command with handler, wait eof/stdout/stderr/external control message
2. process every line in stdout/stderr, extract paths, identifiers, words and symbols

History:
save history lines of stdout/stderr.
we need to save metadata of a line:
1. identifiers with "." and "-"
2. paths

Log Iterator:
used for handler.
when handler find a keyword, we need log iter to:
1. find previous path
2. find some previous keywords

Handler:
requires a well-designed history and log iterator.


"""

import abc
import asyncio
import bisect
import re
import sys
import time
from asyncio.tasks import FIRST_COMPLETED
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Set, Tuple, Type

import asyncssh

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


def remove_ansi_seq(string: str):
    # https://stackoverflow.com/questions/14693701/how-can-i-remove-the-ansi-escape-sequences-from-a-string-in-python
    print(type(string))
    return ANSI_ESCAPE_REGEX.sub(b'', string.encode("utf-8")).decode("utf-8")


class Event:
    name = "Event"

    def __init__(self, timestamp: float):
        self.timestamp = timestamp

    def __repr__(self):
        return "{}({})".format(self.name, self.timestamp)


class TimeoutEvent(Event):
    name = "TimeoutEvent"


class ExitEvent(Event):
    name = "ExitEventEvent"

    def __init__(self, timestamp: float, status: int = 0):
        super().__init__(timestamp)
        self.status = status

    def __bool__(self):
        return self.status == 0

    def __repr__(self):
        return "{}({}|{})".format(self.name, self.status, self.timestamp)


class Handler:
    def handle_stderr_line(self, timestamp: float, line: str) -> List[Event]:
        return []

    def handle_stdout_line(self, timestamp: float, line: str) -> List[Event]:
        return []

    def handle_stderr(self, line: str):
        return []

    def handle_stdout(self, line: str):
        return []


class PrintHandler(Handler):
    def handle_stderr_line(self, timestamp: float, line: str):
        print(line, end='', file=sys.stderr)
        return []

    def handle_stdout_line(self, timestamp: float, line: str):
        print(line, end='')
        return []


class LineEvent(Event):
    name = "LineEvent"

    def __init__(self, timestamp: float, line: str, is_stderr=False):
        super().__init__(timestamp)
        self.line = line


class LineHandler(Handler):
    def handle_stderr_line(self, timestamp: float, line: str):
        return [LineEvent(timestamp, line, True)]

    def handle_stdout_line(self, timestamp: float, line: str):
        return [LineEvent(timestamp, line)]


class RegexMatchEvent(Event):
    name = "RegexMatchEvent"

    def __init__(self, timestamp: float, matched: str, is_stderr=False):
        super().__init__(timestamp)
        self.matched = matched
        self.is_stderr = is_stderr


class RegexHandler(Handler):
    """trigger event by regex search
    """
    def __init__(self, pattern: str, err_pattern: str):
        self.regex = re.compile(pattern)
        self.err_regex = re.compile(err_pattern)

    def handle_stdout_line(self, timestamp: float, line: str):
        if self.err_regex.search(line) is not None:
            return [RegexMatchEvent(timestamp, line, False)]
        return []

    def handle_stderr_line(self, timestamp: float, line: str):
        if self.err_regex.search(line) is not None:
            return [RegexMatchEvent(timestamp, line, True)]
        return []


_DEFAULT_SEPARATORS = ('\n', '\r')


class SSHClient:
    """
    problems:
    Q1. I can't see any output when run a long-time script
    A1. request_pty="force" to enable regular terminal behavior for human.
    Q2. I can't see progress bar
    A2. set 'separators' to ('\r', '\r\n')
    design
    stdout/stderr observer: produce events
    Ctrl+C: https://github.com/ronf/asyncssh/issues/112#issuecomment-343318916
    """
    def __init__(self, conn: asyncssh.SSHClientConnection, historysize=2000):
        self.historysize = historysize
        self.conn = conn

    async def peer_command(self,
                           cmd,
                           watchers: Optional[List[Handler]] = None,
                           history_size=None,
                           separators=_DEFAULT_SEPARATORS):
        stdin, stdout, stderr = await self.conn.open_session(
            cmd, request_pty="force")
        return PeerSSHClient(stdin, stdout, stderr, watchers, history_size,
                             separators)

    async def command_noprint(self,
                              cmd,
                              watchers: Optional[List[Handler]] = None,
                              history_size=None,
                              timeout: Optional[float] = None,
                              separators=_DEFAULT_SEPARATORS):
        client = await self.peer_command(cmd, watchers, history_size,
                                         separators)
        evs = await client.wait_exit(timeout)
        return evs, client

    async def command(self,
                      cmd,
                      watchers: Optional[List[Handler]] = None,
                      history_size=None,
                      timeout: Optional[float] = None,
                      separators=_DEFAULT_SEPARATORS):
        if watchers is None:
            watchers = []
        return await self.command_noprint(
            cmd, [*watchers, PrintHandler()], history_size, timeout,
            separators)


class History:
    def __init__(self, history_size=None, max_display_length=10):
        self.ts_history: Deque[float] = deque([], maxlen=history_size)
        self.data_history: Deque[str] = deque([], maxlen=history_size)
        self.max_display_length = max_display_length

    def append(self, timestamp: float, data: str):
        self.ts_history.append(timestamp)
        self.data_history.append(data)

    def __len__(self):
        return len(self.ts_history)

    def __getitem__(self, idx):
        ts, data = self.get_raw_item(idx)
        return (ts, remove_ansi_seq(data))

    def get_raw_item(self, idx):
        data = self.data_history[idx]
        ts = self.ts_history[idx]
        return (ts, data)

    def find_index_left(self, timestamp: float):
        hi = len(self.ts_history)
        left = bisect.bisect_left(self.ts_history, timestamp, 0, hi)
        return left

    def find_index_right(self, timestamp: float):
        hi = len(self.ts_history)
        right = bisect.bisect_right(self.ts_history, timestamp, 0, hi)
        return right

    def __repr__(self):
        max_length = self.max_display_length
        res = ""
        if len(self) > max_length * 2:
            for i in range(max_length):
                date_ts = time.strftime('%Y-%m-%d %H:%M:%S',
                                        time.localtime(self.ts_history[i]))
                res += "[{}]{}\n".format(date_ts,
                                         self.data_history[i].rstrip())
            res += "...{} more lines...".format(len(self) - max_length * 2)
            for i in range(len(self) - max_length - 1, len(self)):
                date_ts = time.strftime('%Y-%m-%d %H:%M:%S',
                                        time.localtime(self.ts_history[i]))
                res += "[{}]{}\n".format(date_ts,
                                         self.data_history[i].rstrip())
        else:
            for i in range(len(self)):
                date_ts = time.strftime('%Y-%m-%d %H:%M:%S',
                                        time.localtime(self.ts_history[i]))
                res += "[{}]{}\n".format(date_ts,
                                         self.data_history[i].rstrip())
        return res


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
                 watchers: Optional[List[Handler]] = None,
                 history_size=None,
                 separators=_DEFAULT_SEPARATORS):
        if watchers is None:
            watchers = []
        self.watchers = watchers
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        # stdout/err history
        self.stdout_history: History = History(history_size)
        self.stderr_history: History = History(history_size)
        # create read tasks. they should exists during peer open.
        self.tasks: List[asyncio.Task] = list()
        self.separators = separators
        self.read_line_task = asyncio.create_task(self._readuntil(self.stdout))
        self.read_err_line_task = asyncio.create_task(
            self._readuntil(self.stderr))
        self.tasks.append(self.read_line_task)
        self.tasks.append(self.read_err_line_task)

    async def send(self, content: str):
        self.stdin.write(content)

    async def send_ctrl_c(self):
        # https://github.com/ronf/asyncssh/issues/112#issuecomment-343318916
        return await self.send('\x03')

    async def _readuntil(self, reader: asyncssh.SSHReader):
        separators = []
        for separator in self.separators:
            if reader._session._encoding:
                separator = separator
            else:
                separator = separator.encode("utf-8")
            separators.append(separator)
        try:
            return await reader.readuntil(separators)
        except asyncio.IncompleteReadError as exc:
            return exc.partial

    async def _wait_loop(self,
                         until_events: List[Type[Event]],
                         timeout_ev: Optional[asyncio.Event],
                         return_next_event: bool = False,
                         until_event_any: bool = True):
        tasks: List[asyncio.Task] = [
            self.read_line_task, self.read_err_line_task
        ]
        timeout_task: Optional[asyncio.Task] = None
        if timeout_ev is not None:
            timeout_task = asyncio.create_task(timeout_ev.wait())
            tasks.append(timeout_task)
        until_event_names: Set[str] = set([cl.name for cl in until_events])
        evs: List[Event] = []
        stdout_is_empty = False
        stderr_is_empty = False
        while True:
            try:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED)
            except asyncio.CancelledError:
                return evs
            timestamp = time.time()
            if self.read_line_task in done:
                # update read line task
                # if not self.stdout.at_eof():
                res = self.read_line_task.result()
                print("RT", res)

                if not res:
                    stdout_is_empty = True
                else:
                    self.stdout_history.append(timestamp, res)
                    for watcher in self.watchers:
                        watch_evs = watcher.handle_stdout_line(timestamp, res)
                        if watch_evs:
                            evs.extend(watch_evs)
                    self.read_line_task = asyncio.create_task(
                        self._readuntil(self.stdout))
                    tasks[0] = self.read_line_task
            if self.read_err_line_task in done:
                # update read line task
                # if not self.stderr.at_eof():
                res = self.read_err_line_task.result()
                if not res:
                    stderr_is_empty = True
                else:
                    self.stderr_history.append(timestamp, res)
                    for watcher in self.watchers:
                        watch_evs = watcher.handle_stderr_line(timestamp, res)
                        if watch_evs:
                            evs.extend(watch_evs)
                    self.read_err_line_task = asyncio.create_task(
                        self._readuntil(self.stderr))
                    tasks[1] = self.read_err_line_task
            if self.stdout.at_eof() and self.stderr.at_eof():
                ex_status = self.stdout.channel.get_exit_status()
                evs.append(ExitEvent(timestamp, ex_status))
                break
            if stderr_is_empty and stdout_is_empty:
                ex_status = self.stdout.channel.get_exit_status()
                evs.append(ExitEvent(timestamp, ex_status))
                break
            if evs and return_next_event:
                break
            time_to_exit = False
            for ev in evs:
                if until_event_any:
                    if ev.name in until_event_names:
                        time_to_exit = True
                        break
                else:
                    if ev.name in until_event_names:
                        until_event_names.pop(ev.name)
                if until_events and not until_event_names:
                    time_to_exit = True
                    break
            if time_to_exit:
                break
            if timeout_task is not None:
                if timeout_task in done:
                    # we don't cancel read tasks here. they should
                    # be canceled when peer exit.
                    # we dont raise error here.
                    evs.append(TimeoutEvent(timestamp))
                    break
        if timeout_ev is not None:
            timeout_ev.set()
            await timeout_task

        return evs

    async def _timeout_timer(self, timeout, timeout_ev: asyncio.Event):
        try:
            await asyncio.sleep(timeout)
            timeout_ev.set()
        except asyncio.CancelledError:
            pass
        return

    async def _wait_loop_timeout(self,
                                 until_events: List[Type[Event]],
                                 timeout: Optional[float] = None,
                                 return_next_event: bool = False,
                                 until_event_any: bool = True) -> List[Event]:
        timeout_ev = None
        if timeout is not None:
            timeout_ev = asyncio.Event()
        wait_loop = self._wait_loop(until_events, timeout_ev,
                                    return_next_event, until_event_any)
        if timeout is not None:
            timer_task = asyncio.create_task(
                self._timeout_timer(timeout, timeout_ev))
            wait_loop_task = asyncio.create_task(wait_loop)
            done, pending = await asyncio.wait([wait_loop_task, timer_task],
                                               return_when=FIRST_COMPLETED)
            if wait_loop_task in done:
                timer_task.cancel()
                await timer_task
                return wait_loop_task.result()
            else:
                await wait_loop_task
                return wait_loop_task.result()
        return await wait_loop

    async def wait_for_all(self,
                           until_events: List[Type[Event]],
                           timeout: Optional[float] = None) -> List[Event]:
        return await self._wait_loop_timeout(until_events, timeout, False,
                                             False)

    async def wait_for_any(self,
                           until_events: List[Type[Event]],
                           timeout: Optional[float] = None) -> List[Event]:
        return await self._wait_loop_timeout(until_events, timeout, False,
                                             True)

    async def wait_next(self, timeout: Optional[float] = None) -> List[Event]:
        return await self._wait_loop_timeout([], timeout, True, False)

    async def wait_exit(self, timeout: Optional[float] = None) -> List[Event]:
        return await self._wait_loop_timeout([], timeout, False, False)

    async def shutdown(self, timeout: Optional[float] = None):
        if self.stdout.at_eof() or self.stderr.at_eof():
            return
        await self.send_ctrl_c()
        if timeout is not None:
            evs = await self.wait_exit(timeout)
            if evs[-1].name == TimeoutEvent.name:
                raise TimeoutError()
        else:
            await self.wait_exit(timeout)

    def close(self):
        self.read_err_line_task.cancel()
        self.read_line_task.cancel()



async def main():
    async with asyncssh.connect('10.131.64.199',
                                username='root',
                                password='root',
                                known_hosts=None) as conn:
        client = SSHClient(conn)
        pc = await client.peer_command(None, [LineHandler()])
        await pc.send("nvidia-smi\n")
        time.sleep(1)
        evs = await pc.wait_next()
        evs = await pc.wait_next()
        evs = await pc.wait_next()
        evs = await pc.wait_next()
        evs = await pc.wait_next()

        print(len(evs))
        print(pc.stdout_history)
        await pc.send("exit")
        print(2)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())