
import json
from typing import Any, Dict, Optional
from tensorpc.core.dataclass_dispatch import dataclass
from pydantic_core import PydanticCustomError, core_schema
from pydantic import (
    GetCoreSchemaHandler,
)
from ... import mui, three

UNKNOWN_VIS_KEY = "unknown_vis"
UNKNOWN_KEY_SPLIT = "!!%"
RESERVED_NAMES = set([UNKNOWN_VIS_KEY, "reserved"])

def is_reserved_name(name: str):
    parts = name.split(".")
    return parts[0] in RESERVED_NAMES

class CanvasItemProxy:
    def __init__(self) -> None:
        super().__init__()
        self._detail_layout: Optional[mui.FlexBox] = None

        self._tdata: Optional[Dict[str, Any]] = None

    def update_event(self, comp: three.Component):
        pass 

    def detail_layout(self, layout: mui.FlexBox):
        self._detail_layout = layout
        return self

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any, _handler: GetCoreSchemaHandler):
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.any_schema(),
        )

    @classmethod
    def validate(cls, v):
        if not isinstance(v, CanvasItemProxy):
            raise TypeError('CanvasItemProxy required')
        return v

    def tdata(self, data: Dict[str, Any]):
        # make sure data is json serializable
        json.dumps(data)
        self._tdata = data
        return self

@dataclass
class CanvasItemCfg:
    lock: bool = False
    visible: bool = True
    proxy: Optional[CanvasItemProxy] = None 
    is_vapi: bool = False
    # if exists, will use it as detail layout
    detail_layout: Optional[mui.Component] = None

def get_canvas_item_cfg(comp: three.Component) -> Optional[CanvasItemCfg]:
    if comp._flow_user_data is not None and isinstance(comp._flow_user_data, CanvasItemCfg):
        return comp._flow_user_data
    return None 

def get_or_create_canvas_item_cfg(comp: three.Component, is_vapi: Optional[bool] = None):
    if comp._flow_user_data is None:
        comp._flow_user_data = CanvasItemCfg()
    assert isinstance(comp._flow_user_data, CanvasItemCfg)
    if is_vapi is not None:
        comp._flow_user_data.is_vapi = is_vapi
    return comp._flow_user_data
