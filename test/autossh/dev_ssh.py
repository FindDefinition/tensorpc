from tensorpc.autossh.core import EofEvent, Event, RawEvent, SSHClient, CommandEvent, CommandEventType, LineEvent, ExceptionEvent, SSHRequest, SSHRequestType
import getpass 
import asyncio 
import shutil
from prompt_toolkit.input import create_input
import sys
import asyncio
from prompt_toolkit.application.application import attach_winch_signal_handler

class TerminalResizer:
    def __init__(self, init_size: tuple[int, int], q: asyncio.Queue):
        self._q = q
        self.size = init_size

    def on_resize(self):
        new_size = shutil.get_terminal_size((80, 20))
        if new_size != self.size:
            self.size = new_size
            asyncio.create_task(self._q.put(SSHRequest(SSHRequestType.ChangeSize, new_size)))

async def main():
    shutdown_ev = asyncio.Event()
    async def handler(ev: Event):
        if isinstance(ev, RawEvent):
            sys.stdout.buffer.write(ev.raw)
            sys.stdout.flush()
        elif isinstance(ev, ExceptionEvent):
            print("ERROR", ev)
            shutdown_ev.set()
        elif isinstance(ev, EofEvent):
            print("Eof", ev)
            shutdown_ev.set()
    username = input("username:")
    password = getpass.getpass("password:")
    client = SSHClient('localhost',
                       username=username,
                       password=password,
                       known_hosts=None,
                       enable_vscode_cmd_util=True)
    q = asyncio.Queue()
    shutdown_task = asyncio.create_task(shutdown_ev.wait())
    terminal_size = shutil.get_terminal_size((80, 20))
    terminal_resizer = TerminalResizer(terminal_size, q)
    await q.put(
        SSHRequest(SSHRequestType.ChangeSize, terminal_size))
    task = asyncio.create_task(
        client.connect_queue(q,
                             handler,
                             shutdown_task,
                             request_pty=True))
    # stdin don't work for fake terminal, because it only produce data when newline is inputed.
    # we need to use some low-level way.
    # fortunately, prompt_toolkit provide a way to do this.
    ptyinput = create_input()
    def keys_ready():
        res = []
        for key_press in ptyinput.read_keys():
            res.append(key_press.data)
        asyncio.create_task(q.put("".join(res)))
    with attach_winch_signal_handler(terminal_resizer.on_resize):
        with ptyinput.raw_mode():
            with ptyinput.attach(keys_ready):
                await task

if __name__ == "__main__":
    asyncio.run(main())
