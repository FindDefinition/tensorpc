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
    def emit(self, record):
        if hasattr(record, TENSORPC_LOGGING_OVERRIDED_PATH_LINENO_KEY):
            path_lineno = getattr(record, TENSORPC_LOGGING_OVERRIDED_PATH_LINENO_KEY)
            if isinstance(path_lineno, tuple) and len(path_lineno) == 2:
                if isinstance(path_lineno[0], str) and isinstance(path_lineno[1], int):
                    record.pathname = path_lineno[0]
                    record.lineno = path_lineno[1]
        super().emit(record)

def get_logger(name: str, level: str = "WARNING") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter("%(name)s|%(message)s")
    # if logger.handlers:
    #     logger.handlers.clear()
    if TENSORPC_ENABLE_RICH_LOG:
        rh = ModifiedRichHandler(rich_tracebacks=True)
        rh.setFormatter(formatter)
        logger.addHandler(rh)
    else:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    logger.propagate = False
    return logger

