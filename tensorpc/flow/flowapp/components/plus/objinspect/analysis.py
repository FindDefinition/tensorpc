import dataclasses
import enum
import inspect
import traceback
import types
from pathlib import Path, PurePath
from typing import (Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional,
                    Set, Tuple, Type, Union)

from tensorpc.core import inspecttools
from tensorpc.core.serviceunit import ReloadableDynamicClass
from tensorpc.flow.flowapp.components import mui
from tensorpc.flow.flowapp.components.plus.objinspect.core import (
    ALL_OBJECT_LAYOUT_HANDLERS, USER_OBJ_TREE_TYPES, ObjectLayoutCreator)
from tensorpc.flow.flowapp.core import FlowSpecialMethods
from tensorpc.flow.jsonlike import (IconButtonData, TreeItem,
                                    parse_obj_to_jsonlike)

GLOBAL_SPLIT = "::"
STRING_LENGTH_LIMIT = 500
_IGNORE_ATTR_NAMES = set(["_abc_impl", "__abstractmethods__"])

SET_CONTAINER_LIMIT_SIZE = 50

class ButtonType(enum.Enum):
    Reload = "reload"
    Delete = "delete"
    Watch = "watch"
    Record = "record"

def parse_obj_item(obj,
                   name: str,
                   id: str,
                   checker: Callable[[Type], bool],
                   obj_meta_cache=None):
    obj_type = type(obj)
    try:
        isinst = isinstance(obj, TreeItem)
    except:
        print("???", type(obj))
        raise 
    if isinst:
        node_candidate = obj.get_json_like_node(id)
        if node_candidate is not None:
            return node_candidate

    node = parse_obj_to_jsonlike(obj, name, id)
    if isinstance(obj, mui.JsonLikeNode):
        return node
    if node.type == mui.JsonLikeType.Object.value:
        t = mui.JsonLikeType.Object
        value = mui.undefined
        count = 1  # make object expandable
        if isinstance(obj, PurePath):
            count = 0
            value = str(obj)
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
                        obj_type, True, no_code=True)
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
        if isinstance(obj, ObjectLayoutCreator):
            is_draggable = True
            is_layout = True
        is_draggable = True
        if is_layout:
            t = mui.JsonLikeType.Layout
        return mui.JsonLikeNode(id,
                                name,
                                t.value,
                                value=value,
                                typeStr=obj_type.__qualname__,
                                cnt=count,
                                drag=is_draggable,
                                iconBtns=[
                                    IconButtonData(ButtonType.Reload.value,
                                                   mui.IconType.Refresh.value,
                                                   "Reload Object")
                                ])
    return node


def parse_obj_dict(obj_dict: Mapping[Any, Any],
                   ns: str,
                   checker: Callable[[Type], bool],
                   obj_meta_cache=None):
    res_node: List[mui.JsonLikeNode] = []
    for k, v in obj_dict.items():
        str_k = str(k)

        node = parse_obj_item(v,
                              str_k,
                              f"{ns}{GLOBAL_SPLIT}{str_k}",
                              checker,
                              obj_meta_cache=obj_meta_cache)
        if not isinstance(k, str):
            node.dictKey = mui.BackendOnlyProp(k)
        res_node.append(node)
    return res_node



def get_obj_dict(obj,
                  checker: Callable[[Type], bool],
                  check_obj: bool = True) -> Dict[Any, Any]:
    res: Dict[Any, Any] = {}
    # TODO size limit node
    if isinstance(obj, (list, tuple, set)):
        if isinstance(obj, set):
            # set size is limited since it don't support nested view.
            obj_list = list(obj)[:SET_CONTAINER_LIMIT_SIZE]
        else:
            obj_list = obj
        return {str(i): obj_list[i] for i in range(len(obj))}
    elif isinstance(obj, dict):
        # return {k: v for k, v in obj.items() if not _is_obj_builtin_or_module(v)}
        return obj
    if not checker(obj) and check_obj:
        return {}
    # if isinstance(obj, types.ModuleType):
    #     return {}
    if inspecttools.is_obj_builtin_or_module(obj):
        return {}
    if isinstance(obj, tuple(USER_OBJ_TREE_TYPES)):
        return obj.get_childs()
    # if isinstance(obj, mui.Component):
    #     return {}
    # members = get_members(obj, no_parent=False)
    # member_keys = set([m[0] for m in members])
    user_defined_prop_keys = inspecttools.get_obj_userdefined_properties(obj)
    # print(_chehck_obj_is_pybind(obj), user_defined_prop_keys)
    for k in dir(obj):
        if k.startswith("__"):
            continue
        if k in _IGNORE_ATTR_NAMES:
            continue
        # if k in member_keys:
        #     continue
        # TODO here we ignore all properies, we should add a lazy evaluation
        if k in user_defined_prop_keys:
            continue
        try:
            v = getattr(obj, k)
        except:
            continue
        if not (checker(v)):
            continue
        if inspecttools.is_obj_builtin_or_module(v):
            continue
        try:
            isinstance(v, TreeItem)
        except:
            continue
        res[k] = v
    return res


def get_obj_single_attr(obj,
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


async def get_obj_by_uid(
    obj,
    uid: str,
    checker: Callable[[Type], bool],
    real_keys: Optional[List[Union[mui.Undefined, Hashable]]] = None
) -> Tuple[Any, bool]:
    parts = uid.split(GLOBAL_SPLIT)
    if real_keys is None:
        real_keys = [mui.undefined for _ in range(len(parts))]
    if len(parts) == 1:
        return obj, True
    # uid contains root, remove it at first.
    return await get_obj_by_uid_resursive(obj, parts[1:], real_keys[1:],
                                           checker)


async def get_obj_by_uid_resursive(
        obj, parts: List[str], real_keys: List[Union[mui.Undefined, Hashable]],
        checker: Callable[[Type], bool]) -> Tuple[Any, bool]:
    key = parts[0]
    real_key = real_keys[0]
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
        if not isinstance(real_key, mui.Undefined):
            key = real_key
        if key not in obj_dict:
            return obj, False
        child_obj = obj_dict[key]
    elif isinstance(obj, TreeItem):
        child_obj = await obj.get_child(key)
    elif isinstance(obj, tuple(USER_OBJ_TREE_TYPES)):
        childs = obj.get_childs()
        child_obj = childs[key]
    else:
        child_obj = get_obj_single_attr(obj, key, checker)
        if isinstance(obj, mui.Undefined):
            return obj, False
    if len(parts) == 1:
        return child_obj, True
    else:
        return await get_obj_by_uid_resursive(child_obj, parts[1:],
                                               real_keys[1:], checker)


async def get_obj_by_uid_trace(
    obj,
    uid: str,
    checker: Callable[[Type], bool],
    real_keys: Optional[List[Union[mui.Undefined, Hashable]]] = None
) -> Tuple[Any, bool]:
    parts = uid.split(GLOBAL_SPLIT)
    if real_keys is None:
        real_keys = [mui.undefined for _ in range(len(parts))]
    if len(parts) == 1:
        return [obj], True
    # uid contains root, remove it at first.
    trace, found = await get_obj_by_uid_trace_resursive(
        obj, parts[1:], real_keys[1:], checker)
    return [obj] + trace, found


async def get_obj_by_uid_trace_resursive(
        obj, parts: List[str], real_keys: List[Union[mui.Undefined, Hashable]],
        checker: Callable[[Type], bool]) -> Tuple[List[Any], bool]:
    key = parts[0]
    real_key = real_keys[0]
    if isinstance(obj, (list, tuple, set)):
        if isinstance(obj, set):
            obj_list = list(obj)
        else:
            obj_list = obj
        try:
            key_index = int(key)
        except:
            return [obj], False
        if key_index < 0 or key_index >= len(obj_list):
            return [obj], False
        child_obj = obj_list[key_index]
    elif isinstance(obj, dict):
        obj_dict = obj
        if not isinstance(real_key, mui.Undefined):
            key = real_key
        if key not in obj_dict:
            return [obj], False
        child_obj = obj_dict[key]
    elif isinstance(obj, TreeItem):
        child_obj = await obj.get_child(key)
    elif isinstance(obj, tuple(USER_OBJ_TREE_TYPES)):
        childs = obj.get_childs()
        child_obj = childs[key]
    else:
        child_obj = get_obj_single_attr(obj, key, checker)
        if isinstance(obj, mui.Undefined):
            return [obj], False
    if len(parts) == 1:
        return [child_obj], True
    else:
        trace, found = await get_obj_by_uid_trace_resursive(
            child_obj, parts[1:], real_keys[1:], checker)
        return [child_obj] + trace, found
