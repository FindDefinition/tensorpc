

from base64 import b64encode
import enum
from pathlib import Path
from types import FrameType
from tensorpc.utils.loader import get_frame_module_meta
from tensorpc.dbg.constants import TENSORPC_DBG_FRAME_STORAGE_PREFIX

class VariableMetaType(enum.Enum):
    Layout = "layout"

def get_frame_uid(frame: FrameType):
    frame_mod_meta = get_frame_module_meta(frame)
    if frame_mod_meta.is_path:
        path_no_suffix = Path(frame_mod_meta.module).with_suffix("")
        path_enc = b64encode(path_no_suffix.as_posix().encode()).decode()
        path_enc = path_enc.replace("=", "a")
        uid = f"{frame_mod_meta.qualname}-{path_enc}"
    else:
        uid = f"{frame_mod_meta.qualname}-{frame_mod_meta.module}"

    return uid, frame_mod_meta

def get_storage_frame_path(frame_uid: str):
    return f"{TENSORPC_DBG_FRAME_STORAGE_PREFIX}/{frame_uid}/"
