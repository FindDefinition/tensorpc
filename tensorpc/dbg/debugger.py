from typing import Optional
from . import bkpt

class Debugger:
    def __init__(self, proc_name: str, port: int = -1):
        """
        Args:
            proc_name: the process name of the background server, only valid before init
            port: the port of the background server, only valid before init
        """
        self._proc_name = proc_name
        self._port = port

    def breakpoint(self, name: Optional[str] = None, timeout: Optional[float] = None):
        return bkpt.breakpoint(name, timeout, self._port, self._proc_name)