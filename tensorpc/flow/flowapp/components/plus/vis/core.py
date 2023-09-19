
from typing import Any, Optional
from tensorpc.core.dataclass_dispatch import dataclass
from pydantic_core import PydanticCustomError, core_schema
from pydantic import (
    GetCoreSchemaHandler,
)
from ... import mui, three


class CanvasItemProxy:
    def __init__(self) -> None:
        super().__init__()
        self._detail_layout: Optional[mui.FlexBox] = None
    
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

@dataclass
class CanvasItemCfg:
    lock: bool = False
    visible: bool = True
    proxy: Optional[CanvasItemProxy] = None 
    is_vapi: bool = False
    # if exists, will use it as detail layout
    detail_layout: Optional[mui.Component] = None

def get_canvas_item_cfg(comp: three.Component) -> Optional[CanvasItemCfg]:
    if comp._flow_user_data is not None and not isinstance(comp._flow_user_data, CanvasItemCfg):
        return comp._flow_user_data
    return None 

def get_or_create_canvas_item_cfg(comp: three.Component, is_vapi: Optional[bool] = None):
    if comp._flow_user_data is None:
        comp._flow_user_data = CanvasItemCfg()
    assert isinstance(comp._flow_user_data, CanvasItemCfg)
    if is_vapi is not None:
        comp._flow_user_data.is_vapi = is_vapi
    return comp._flow_user_data
