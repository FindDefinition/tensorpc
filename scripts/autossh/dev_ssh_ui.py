from tensorpc.autossh.core import SSHConnDesc
from tensorpc.dock.components.terminal import AsyncSSHTerminal
import asyncio 

async def _main():
    import rich 
    term = AsyncSSHTerminal()
    desc = SSHConnDesc("localhost:22", "root", "1")
    await term.connect_with_new_desc(desc)
    res = await term.ssh_command_rpc("echo $HOME\n")
    rich.print(res)
    await term.disconnect()
    pass 

if __name__ == "__main__":
    asyncio.run(_main())