from pathlib import Path 

PACKAGE_ROOT = Path(__file__).parent.resolve()


TENSORPC_FUNC_META_KEY = "__distflow_func_meta"
TENSORPC_CLASS_META_KEY = "__distflow_class_meta"

TENSORPC_WEBSOCKET_MSG_SIZE = (4 << 20) - 128

TENSORPC_SPLIT = "::"