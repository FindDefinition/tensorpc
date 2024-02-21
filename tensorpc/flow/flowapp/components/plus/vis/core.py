
import json
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
from tensorpc.core.dataclass_dispatch import dataclass, field
from pydantic_core import PydanticCustomError, core_schema
from pydantic import (
    GetCoreSchemaHandler,
)
from tensorpc.core.tree_id import UniqueTreeId, UniqueTreeIdForTree
from tensorpc.flow.jsonlike import JsonLikeNode
from ... import mui, three
if TYPE_CHECKING:
    from .canvas import ComplexCanvas
from tensorpc.utils.uniquename import UniqueNamePool

UNKNOWN_VIS_KEY = "unknown_vis"
UNKNOWN_KEY_SPLIT = "%"
RESERVED_NAMES = set([UNKNOWN_VIS_KEY, "reserved"])

def is_reserved_name(name: str):
    parts = name.split(".")
    return parts[0] in RESERVED_NAMES

def is_reserved_uid(uid: UniqueTreeId):
    parts = uid.parts
    return parts[0] in RESERVED_NAMES

class CanvasItemProxy:
    def __init__(self) -> None:
        super().__init__()
        self._detail_layout: Optional[mui.FlexBox] = None

        self._tdata: Optional[Dict[str, Any]] = None

    # def update_event(self, comp: three.Component):
    #     pass 

    # def detail_layout(self, layout: mui.FlexBox):
    #     self._detail_layout = layout
    #     return self

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any, _handler: GetCoreSchemaHandler):
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.any_schema(),
        )

    @classmethod
    def validate(cls, v):
        if not isinstance(v, CanvasItemProxy):
            raise ValueError('CanvasItemProxy required')
        return v

    # def tdata(self, data: Dict[str, Any]):
    #     # make sure data is json serializable
    #     json.dumps(data)
    #     self._tdata = data
    #     return self

@dataclass
class CanvasItemCfg:
    lock: bool = False
    visible: bool = True
    proxy: Optional[CanvasItemProxy] = None 
    is_vapi: bool = False
    # if exists, will use it as detail layout
    detail_layout: Optional[mui.Component] = None
    type_str_override: Optional[str] = None
    alias: Optional[str] = None
    node: Optional[JsonLikeNode] = None
    tdata: Optional[Dict[str, Any]] = None

def get_canvas_item_cfg(comp: three.Component) -> Optional[CanvasItemCfg]:
    res = comp.find_user_meta_by_type(CanvasItemCfg)
    if res is not None:
        return res
    return None 

def get_or_create_canvas_item_cfg(comp: three.Component, is_vapi: Optional[bool] = None):
    res = comp.find_user_meta_by_type(CanvasItemCfg)
    if res is None:
        res = CanvasItemCfg()
        comp._flow_user_datas.append(res)
    if is_vapi is not None:
        res.is_vapi = is_vapi
    return res


class VContext:
    def __init__(self,
                 canvas: "ComplexCanvas",
                 root: Optional[three.ContainerBase] = None):
        self.stack = []
        self.canvas = canvas
        self.name_stack: List[str] = []
        self.exist_name_stack: List[str] = []
        if root is None:
            root = canvas._item_root
        self.root = root
        self._name_to_group: Dict[str, three.ContainerBase] = {"": root}
        self._group_assigns: Dict[three.ContainerBase, Tuple[three.Component,
                                                             str]] = {}

    @property
    def current_namespace(self):
        return UniqueTreeIdForTree.from_parts(self.name_stack)

    @property
    def current_container(self):
        if not self.name_stack:
            return self.root
        else:
            return self._name_to_group[self.current_namespace.uid_encoded]

    def clear(self):
        self.stack.clear()
        self.name_stack.clear()
        self.exist_name_stack.clear()
        self._name_to_group.clear()
        self._name_to_group[""] = self.root
        self._group_assigns.clear()


    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any, _handler: GetCoreSchemaHandler):
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.any_schema(),
        )

    @classmethod
    def validate(cls, v):
        if not isinstance(v, VContext):
            raise ValueError('VContext required')
        return v



class ContainerProxy(CanvasItemProxy):
    pass

class GroupProxy(ContainerProxy):
    def __init__(self, uid: str) -> None:
        super().__init__()
        self.uid = uid

        self.childs: Dict[str, three.Component] = {}

        self.namepool = UniqueNamePool()

    def __repr__(self) -> str:
        return f"<GroupProxy {self.uid}>"

    def clear(self):
        self.childs.clear()
        self.namepool.clear()


class UserTreeItemCard(mui.FlexBox):
    def __init__(self, name: str, type_str: str, callback: Callable[[bool], Any], init_width: mui.ValueType = "240px"):
        self.preview_layout = mui.VBox([])
        self._is_selected = mui.Checkbox(callback=callback).prop(checked=True)
        self._print_blocks = mui.DataFlexBox(mui.Markdown().set_override_props(value="data")).prop(flexFlow="column nowrap", overflowY="auto", height="100%")
        layout: mui.LayoutType = [
            mui.HBox([
                mui.Typography(f"{name}@{type_str}").prop(variant="caption", flex=1),
                self._is_selected.prop(size="small"),
            ]),
            mui.HDivider(),
            self.preview_layout,
            self._print_blocks,
        ]
        super().__init__(layout)
        self.prop(flexFlow="column nowrap", height="100%", width=init_width)
        self.name = name
        self.type_str = type_str

    @property
    def is_selected(self):
        return self._is_selected.checked
    
    def print_blocks_event(self, data: List[str]):
        data_list = [{"id": i, "data": d} for i, d in enumerate(data)]
        return self._print_blocks.update_event(dataList=data_list)


@dataclass
class CanvasUserTreeItem:
    key: Tuple[str, ...]
    # TODO fix circular dependency between canvas and this
    vctx: VContext
    card: UserTreeItemCard
    # childs will be filled from vctx when top vctx exit
    childs: Dict[str, three.ThreeComponentBase] = field(default_factory=dict)

    md_prints: List[str] = field(default_factory=list)


class _VapiObjects:
    def prepare_vapi_props(self) -> None:
        raise NotImplementedError
