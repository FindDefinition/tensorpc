from rich.logging import RichHandler
from tensorpc.constants import TENSORPC_ENABLE_RICH_LOG
import logging 


TENSORPC_LOGGING_OVERRIDED_PATH_LINENO_KEY = "__tensorpc_overrided_path_lineno_key"

# logging.basicConfig(
#     level="WARNING",
#     format="[%(name)s]%(message)s",
#     datefmt="[%X]",
# )

class ModifiedRichHandler(RichHandler):
    """A custom rich handler that add ability to override the path and 
    lineno of the log record.
    """
    def emit(self, record):
        if hasattr(record, TENSORPC_LOGGING_OVERRIDED_PATH_LINENO_KEY):
            path_lineno = getattr(record, TENSORPC_LOGGING_OVERRIDED_PATH_LINENO_KEY)
            if isinstance(path_lineno, tuple) and len(path_lineno) == 2:
                if isinstance(path_lineno[0], str) and isinstance(path_lineno[1], int):
                    record.pathname = path_lineno[0]
                    record.lineno = path_lineno[1]
        super().emit(record)

def get_logger(name: str, level: str = "WARNING", show_path: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        for handler in logger.handlers:
            if isinstance(handler, ModifiedRichHandler):
                # already initialized
                return logger
    logger.setLevel(level)
    formatter = logging.Formatter("%(name)s|%(message)s")
    if TENSORPC_ENABLE_RICH_LOG:
        rh = ModifiedRichHandler(rich_tracebacks=True, show_path=show_path)
        rh.setFormatter(formatter)
        logger.addHandler(rh)
    else:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    logger.propagate = False
    return logger

