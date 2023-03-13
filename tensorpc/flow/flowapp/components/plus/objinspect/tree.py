import dataclasses
import enum
import inspect
import traceback
import types
from functools import partial
from typing import (Any, Callable, Dict, Hashable, Iterable, List, Optional,
                    Set, Tuple, Type, Union)

import numpy as np

from tensorpc.core.inspecttools import get_members
from tensorpc.core.serviceunit import ReloadableDynamicClass
from tensorpc.flow.flowapp.components import mui, three
from tensorpc.flow.flowapp.core import FlowSpecialMethods, FrontendEventType
from tensorpc.core.moduleid import get_qualname_of_type
from tensorpc.flow.flowapp.components.plus.monitor import ComputeResourceMonitor
from tensorpc.flow.flowapp.reload import reload_object_methods
from tensorpc.flow.flowapp.components.plus.objinspect.core import ALL_OBJECT_PREVIEW_HANDLERS, ALL_OBJECT_LAYOUT_HANDLERS
from tensorpc.flow.flowapp.components.plus.canvas import SimpleCanvas
from tensorpc.flow.flowapp.coretypes import TreeDragTarget
from tensorpc.flow.flowapp.components.plus.common import CommonQualNames

_DEFAULT_OBJ_NAME = "default"

_DEFAULT_BUILTINS_NAME = "builtins"

_ROOT = "root"

BASE_OBJ_TO_TYPE = {
    int: mui.JsonLikeType.Int,
    float: mui.JsonLikeType.Float,
    complex: mui.JsonLikeType.Complex,
    bool: mui.JsonLikeType.Bool,
    str: mui.JsonLikeType.String,
}

_GLOBAL_SPLIT = "::"

CONTAINER_TYPES = {mui.JsonLikeType.List.value, mui.JsonLikeType.Dict.value}
FOLDER_TYPES = {
    mui.JsonLikeType.ListFolder.value, mui.JsonLikeType.DictFolder.value
}

STRING_LENGTH_LIMIT = 2000

SET_CONTAINER_LIMIT_SIZE = 50


class ButtonType(enum.Enum):
    Reload = "reload"


def parse_obj_item(obj,
                   name: str,
                   id: str,
                   checker: Callable[[Type], bool],
                   obj_meta_cache=None):
    obj_type = type(obj)
    if obj is None or obj is Ellipsis:
        return mui.JsonLikeNode(id,
                                name,
                                mui.JsonLikeType.Constant.value,
                                value=str(obj))
    elif isinstance(obj, enum.Enum):
        return mui.JsonLikeNode(id,
                                name,
                                mui.JsonLikeType.Enum.value,
                                "enum",
                                value=str(obj))
    elif isinstance(obj, (bool)):
        # bool is inherit from int, so we must check bool first.
        return mui.JsonLikeNode(id,
                                name,
                                mui.JsonLikeType.Bool.value,
                                value=str(obj))
    elif isinstance(obj, (int)):
        return mui.JsonLikeNode(id,
                                name,
                                mui.JsonLikeType.Int.value,
                                value=str(obj))
    elif isinstance(obj, (float)):
        return mui.JsonLikeNode(id,
                                name,
                                mui.JsonLikeType.Float.value,
                                value=str(obj))
    elif isinstance(obj, (complex)):
        return mui.JsonLikeNode(id,
                                name,
                                mui.JsonLikeType.Complex.value,
                                value=str(obj))
    elif isinstance(obj, str):
        if len(obj) > STRING_LENGTH_LIMIT:
            value = obj[:STRING_LENGTH_LIMIT] + "..."
        else:
            value = obj
        return mui.JsonLikeNode(id,
                                name,
                                mui.JsonLikeType.String.value,
                                value=value)

    elif isinstance(obj, (list, dict, tuple, set)):
        t = mui.JsonLikeType.List
        if isinstance(obj, list):
            t = mui.JsonLikeType.List
        elif isinstance(obj, dict):
            t = mui.JsonLikeType.Dict
        elif isinstance(obj, tuple):
            t = mui.JsonLikeType.Tuple
        elif isinstance(obj, set):
            t = mui.JsonLikeType.Set
        else:
            raise NotImplementedError
        # TODO suppert nested view
        return mui.JsonLikeNode(id, name, t.value, cnt=len(obj), drag=False)
    elif isinstance(obj, np.ndarray):
        t = mui.JsonLikeType.Tensor
        return mui.JsonLikeNode(id,
                                name,
                                t.value,
                                typeStr="np.ndarray",
                                value=f"{obj.shape}|{obj.dtype}",
                                drag=True)
    elif get_qualname_of_type(obj_type) == CommonQualNames.TorchTensor:
        t = mui.JsonLikeType.Tensor
        return mui.JsonLikeNode(id,
                                name,
                                t.value,
                                typeStr="torch.Tensor",
                                value=f"{list(obj.shape)}|{obj.dtype}",
                                drag=True)
    elif get_qualname_of_type(obj_type) == CommonQualNames.TVTensor:
        t = mui.JsonLikeType.Tensor
        return mui.JsonLikeNode(id,
                                name,
                                t.value,
                                typeStr="tv.Tensor",
                                value=f"{obj.shape}|{obj.dtype}",
                                drag=True)
    else:
        t = mui.JsonLikeType.Object
        obj_type = type(obj)
        # obj_dict = _get_obj_dict(obj, checker)
        if obj_meta_cache is None:
            obj_meta_cache = {}
        if obj_type in obj_meta_cache:
            is_layout = obj_meta_cache[obj_type]
        else:
            if obj_type in ALL_OBJECT_LAYOUT_HANDLERS:
                is_layout = True
            else:
                try:
                    metas = ReloadableDynamicClass.get_metas_of_regular_methods(
                        obj_type, False, no_code=True)
                    special_methods = FlowSpecialMethods(metas)
                    is_layout = special_methods.create_layout is not None
                except:
                    is_layout = False 
                    traceback.print_exc()
                    print("ERROR", obj_type)
            obj_meta_cache[obj_type] = is_layout
        is_draggable = is_layout
        if isinstance(obj, mui.Component):
            is_layout = True
            is_draggable = obj._flow_reference_count == 0
        is_draggable = True
        if is_layout:
            t = mui.JsonLikeType.Layout
        return mui.JsonLikeNode(id,
                                name,
                                t.value,
                                typeStr=obj_type.__qualname__,
                                cnt=1,
                                drag=is_draggable,
                                iconBtns=[(ButtonType.Reload.value,
                                           mui.IconType.Refresh.value)])


def parse_obj_dict(obj_dict: Dict[str, Any],
                   ns: str,
                   checker: Callable[[Type], bool],
                   obj_meta_cache=None):
    res_node: List[mui.JsonLikeNode] = []
    for k, v in obj_dict.items():

        node = parse_obj_item(v,
                              k,
                              f"{ns}{_GLOBAL_SPLIT}{k}",
                              checker,
                              obj_meta_cache=obj_meta_cache)
        res_node.append(node)
    return res_node


def _check_is_valid(obj_type, cared_types: Set[Type],
                    ignored_types: Set[Type]):
    valid = True
    if len(cared_types) != 0:
        valid &= obj_type in cared_types
    if len(ignored_types) != 0:
        valid &= obj_type not in ignored_types
    return valid


def _get_obj_dict(obj,
                  checker: Callable[[Type], bool],
                  check_obj: bool = True):
    res: Dict[str, Any] = {}
    # TODO size limit node
    if isinstance(obj, (list, tuple, set)):
        if isinstance(obj, set):
            # set size is limited since it don't support nested view.
            obj_list = list(obj)[:SET_CONTAINER_LIMIT_SIZE]
        else:
            obj_list = obj
        return {str(i): obj_list[i] for i in range(len(obj))}
    elif isinstance(obj, dict):
        return {str(k): v for k, v in obj.items()}
    if inspect.isbuiltin(obj):
        return {}
    if not checker(obj) and check_obj:
        return {}
    if isinstance(obj, types.ModuleType):
        return {}
    # if isinstance(obj, mui.Component):
    #     return {}
    # members = get_members(obj, no_parent=False)
    # member_keys = set([m[0] for m in members])
    for k in dir(obj):
        if k.startswith("__"):
            continue
        # if k in member_keys:
        #     continue
        try:
            v = getattr(obj, k)
        except:
            continue
        if not (checker(v)):
            continue
        if isinstance(v, types.ModuleType):
            continue
        if inspect.isfunction(v) or inspect.ismethod(v) or inspect.isbuiltin(
                v):
            continue
        res[k] = v
    return res


def _get_obj_single_attr(obj,
                         key: str,
                         checker: Callable[[Type], bool],
                         check_obj: bool = True) -> Union[mui.Undefined, Any]:
    # if isinstance(obj, (list, tuple, set)):
    #     try:
    #         key_int = int(key)
    #     except:
    #         return mui.undefined
    #     if key_int < 0 or key_int >= len(obj):
    #         return mui.undefined
    #     obj_list = list(obj)
    #     return obj_list[key_int]
    # elif isinstance(obj, dict):
    #     if key not in obj:
    #         return mui.undefined
    #     return obj[key]
    if inspect.isbuiltin(obj):
        return mui.undefined
    if not checker(obj) and check_obj:
        return mui.undefined
    if isinstance(obj, types.ModuleType):
        return mui.undefined
    # if isinstance(obj, mui.Component):
    #     return {}
    # members = get_members(obj, no_parent=False)
    # member_keys = set([m[0] for m in members])
    obj_keys = dir(obj)
    if key in obj_keys:
        try:
            v = getattr(obj, key)
        except:
            return mui.undefined
        if not (checker(v)):
            return mui.undefined
        if isinstance(v, types.ModuleType):
            return mui.undefined
        if inspect.isfunction(v) or inspect.ismethod(v) or inspect.isbuiltin(
                v):
            return mui.undefined
        return v
    return mui.undefined


def _get_obj_by_uid(obj, uid: str,
                    checker: Callable[[Type], bool]) -> Tuple[Any, bool]:
    parts = uid.split(_GLOBAL_SPLIT)
    if len(parts) == 1:
        return obj, True
    # uid contains root, remove it at first.
    return _get_obj_by_uid_resursive(obj, parts[1:], checker)


def _get_obj_by_uid_resursive(
        obj, parts: List[str], checker: Callable[[Type],
                                                 bool]) -> Tuple[Any, bool]:
    key = parts[0]
    if isinstance(obj, (list, tuple, set)):
        if isinstance(obj, set):
            obj_list = list(obj)
        else:
            obj_list = obj
        try:
            key_index = int(key)
        except:
            return obj, False
        if key_index < 0 or key_index >= len(obj_list):
            return obj, False
        child_obj = obj_list[key_index]
    elif isinstance(obj, dict):
        obj_dict = obj
        if key not in obj_dict:
            return obj, False
        child_obj = obj_dict[key]
    else:
        child_obj = _get_obj_single_attr(obj, key, checker)
        if isinstance(obj, mui.Undefined):
            return obj, False
    if len(parts) == 1:
        return child_obj, True
    else:
        return _get_obj_by_uid_resursive(child_obj, parts[1:], checker)


def _get_root_tree(obj,
                   checker: Callable[[Type], bool],
                   key: str,
                   obj_meta_cache=None):
    obj_dict = _get_obj_dict(obj, checker)
    root_node = parse_obj_item(obj, key, key, checker, obj_meta_cache)
    root_node.children = parse_obj_dict(obj_dict, key, checker, obj_meta_cache)
    for (k, o), c in zip(obj_dict.items(), root_node.children):
        obj_child_dict = _get_obj_dict(o, checker)
        c.drag = False
        c.children = parse_obj_dict(obj_child_dict, c.id, checker,
                                    obj_meta_cache)
    root_node.cnt = len(obj_dict)
    return root_node


class ObjectTree(mui.FlexBox):

    def __init__(self,
                 init: Optional[Any] = None,
                 cared_types: Optional[Set[Type]] = None,
                 ignored_types: Optional[Set[Type]] = None,
                 limit: int = 50) -> None:
        self.tree = mui.JsonLikeTree()
        super().__init__([
            self.tree.prop(ignore_root=True, use_fast_tree=False),
        ])
        self.prop(overflow="auto")
        self._uid_to_node: Dict[str, mui.JsonLikeNode] = {}
        if cared_types is None:
            cared_types = set()
        if ignored_types is None:
            ignored_types = set()
        self._cared_types = cared_types
        self._ignored_types = ignored_types
        self._obj_meta_cache = {}
        self.limit = limit
        default_builtins = {
            _DEFAULT_BUILTINS_NAME: {
                "monitor": ComputeResourceMonitor(),
                "appTerminal": mui.AppTerminal(),
                "simpleCanvas": SimpleCanvas(),
            }
        }
        if init is None:
            self.root = {}
            # self.tree.props.tree = mui.JsonLikeNode(
            #     _ROOT, _ROOT, mui.JsonLikeType.Dict.value)
        else:
            self.root = {_DEFAULT_OBJ_NAME: init}
        self.root.update(default_builtins)
        self.tree.props.tree = _get_root_tree(self.root, self._valid_checker,
                                              _ROOT, self._obj_meta_cache)
        # print(self.tree.props.tree)
        # inspect.isbuiltin()
        self.tree.register_event_handler(
            FrontendEventType.TreeLazyExpand.value, self._on_expand)
        self.tree.register_event_handler(
            FrontendEventType.TreeItemButton.value, self._on_custom_button)

        self.tree.register_event_handler(FrontendEventType.DragCollect.value,
                                         self._on_drag_collect,
                                         backend_only=True)

    def _checker(self, obj):
        return _check_is_valid(type(obj), self._cared_types,
                               self._ignored_types)

    def _valid_checker(self, obj):
        return True

    @property
    def _objinspect_root(self):
        return self.tree.props.tree

    def _get_obj_by_uid(self, uid: str):
        return _get_obj_by_uid(self.root, uid, self._valid_checker)

    async def _on_drag_collect(self, data):
        uid = data["id"]
        obj, found = _get_obj_by_uid(self.root, uid, self._valid_checker)
        if not found:
            return None
        tab_id = ""
        if "complexLayoutTabNodeId" in data:
            # for complex layout UI: FlexLayout
            tab_id = data["complexLayoutTabNodeId"]
        # if isinstance(obj, mui.FlexBox):
        #     wrapped_obj = obj
        # else:
        #     wrapped_obj = mui.flex_wrapper(obj)
        return TreeDragTarget(obj, uid, tab_id)

    async def _on_expand(self, uid: str):
        node = self._objinspect_root._get_node_by_uid(uid)
        if node.type in FOLDER_TYPES:
            assert not isinstance(node.start, mui.Undefined)
            assert not isinstance(node.realId, mui.Undefined)
            if node._is_divisible(self.limit):
                node.children = node._get_divided_tree(self.limit, node.start)
                upd = self.tree.update_event(tree=self._objinspect_root)
                return await self.tree.send_and_wait(upd)
            # real_node = self._objinspect_root._get_node_by_uid(node.realId)
            real_obj, found = _get_obj_by_uid(self.root, node.realId,
                                              self._valid_checker)
            if node.type == mui.JsonLikeType.ListFolder.value:
                data = real_obj[node.start:node.start + node.cnt]
            else:
                assert not isinstance(node.keys, mui.Undefined)
                data = {k: real_obj[k] for k in node.keys.data}
            obj_dict = _get_obj_dict(data, self._checker)
            tree = parse_obj_dict(obj_dict, node.id, self._checker,
                                  self._obj_meta_cache)
            node.children = tree
            upd = self.tree.update_event(tree=self._objinspect_root)
            return await self.tree.send_and_wait(upd)
        obj, found = _get_obj_by_uid(self.root, uid, self._valid_checker)
        if node.type in CONTAINER_TYPES and node._is_divisible(self.limit):
            if node.type == mui.JsonLikeType.Dict.value:
                node.keys = mui.BackendOnlyProp(list(obj.keys()))
            node.children = node._get_divided_tree(self.limit, 0)
            upd = self.tree.update_event(tree=self._objinspect_root)
            return await self.tree.send_and_wait(upd)
        # if not found, we expand (update) the deepest valid object.

        obj_dict = _get_obj_dict(obj, self._checker)
        tree = parse_obj_dict(obj_dict, node.id, self._checker,
                              self._obj_meta_cache)
        node.children = tree
        upd = self.tree.update_event(tree=self._objinspect_root)
        return await self.tree.send_and_wait(upd)

    async def _on_custom_button(self, uid_btn: Tuple[str, str]):
        uid = uid_btn[0]
        obj, found = _get_obj_by_uid(self.root, uid, self._valid_checker)
        if not found:
            return
        btn = ButtonType(uid_btn[1])
        if btn == ButtonType.Reload:
            metas = reload_object_methods(
                obj, reload_mgr=self.flow_app_comp_core.reload_mgr)
            if metas is not None:
                special_methods = FlowSpecialMethods(metas)
            else:
                print("reload failed.")

    async def set_object(self, obj, key: str = _DEFAULT_OBJ_NAME):
        self.root[key] = obj
        self.tree.props.tree = _get_root_tree(self.root, self._valid_checker,
                                              _ROOT, self._obj_meta_cache)
        await self.tree.send_and_wait(
            self.tree.update_event(tree=self.tree.props.tree))

    async def update_tree(self):
        self.tree.props.tree = _get_root_tree(self.root, self._valid_checker,
                                              _ROOT, self._obj_meta_cache)
        await self.tree.send_and_wait(
            self.tree.update_event(tree=self.tree.props.tree))

    async def remove_object(self, key: str):
        assert key in self.root
        self.tree.props.tree = _get_root_tree(self.root, self._valid_checker,
                                              _ROOT, self._obj_meta_cache)
        await self.tree.send_and_wait(
            self.tree.update_event(tree=self.tree.props.tree))
