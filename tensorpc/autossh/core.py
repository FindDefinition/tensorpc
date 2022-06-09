import asyncio
import bisect

import abc
import asyncio
import bisect
import re
import sys
import time
from asyncio.tasks import FIRST_COMPLETED
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Set, Tuple, Type, Union

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

_DEFAULT_SEPARATORS = ('\n', '\r')

def remove_ansi_seq(string: Union[str, bytes]):
    # https://stackoverflow.com/questions/14693701/how-can-i-remove-the-ansi-escape-sequences-from-a-string-in-python
    if isinstance(string, str):
        return ANSI_ESCAPE_REGEX.sub(b'', string.encode("utf-8")).decode("utf-8")
    else:
        return ANSI_ESCAPE_REGEX.sub(b'', string).decode("utf-8")

class OutData:
    def __init__(self) -> None:
        pass

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
                 separators=_DEFAULT_SEPARATORS):
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        # stdout/err history
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
            print("?")
            return exc.partial


    async def wait_loop_queue(self, q: asyncio.Queue):
        """events: stdout/err line, eof
        """
        pass