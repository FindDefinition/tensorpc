from typing import Any, Optional, IO

try:
    import rich 
    __rprint = rich.print 
except ImportError:
    __rprint = print


def rprint(
    *objects: Any,
    sep: str = " ",
    end: str = "\n",
    file: Optional[IO[str]] = None,
    flush: bool = False):
    return __rprint(*objects, sep=sep, end=end, file=file, flush=flush)
