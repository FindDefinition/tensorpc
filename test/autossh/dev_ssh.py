from tensorpc.autossh.constants import TENSORPC_ASYNCSSH_INIT_SUCCESS
from tensorpc.autossh.core import Event, SSHClient, CommandEvent, CommandEventType, LineEvent, ExceptionEvent, SSHRequest, SSHRequestType
import getpass 
import asyncio 
from tensorpc.core.bgserver import BACKGROUND_SERVER
import shutil

async def main3():
    from prompt_toolkit.shortcuts.prompt import PromptSession
    prompt_session = PromptSession(">")
    shutdown_ev = asyncio.Event()
    async def handler(ev: Event):
        # print(ev)
        if isinstance(ev, CommandEvent):
            if ev.type == CommandEventType.PROMPT_END:
                # print(ev.arg, end="", flush=True)
                print(ev.arg.decode("utf-8"), end="", )
        if isinstance(ev, LineEvent):
            print(ev.line.decode("utf-8"), end="", )
            # print("LINE", ev.line, end="")
        if isinstance(ev, ExceptionEvent):
            print("ERROR", ev.data)
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
    # await q.put(f"echo \"{TENSORPC_ASYNCSSH_INIT_SUCCESS}\"\n")
    terminal_size = shutil.get_terminal_size((80, 20))
    await q.put(
        SSHRequest(SSHRequestType.ChangeSize, terminal_size))
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

async def _main_ssh_rpc():
    BACKGROUND_SERVER.start_async()
    client = SSHClient('localhost', "root", "1")
    async with client.simple_connect_with_rpc(True) as rpc_client:
        rpc_client.set_init_cmd("conda activate torchdev")
        for j in range(5):
            res = await rpc_client.call_with_code(f"""
def add(a, b):
    return a + b
            """, 1, 2)
            print(res)

if __name__ == "__main__":
    asyncio.run(main3())
