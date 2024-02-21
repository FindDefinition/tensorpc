import contextlib
import dataclasses
import enum
import inspect
from re import T
import traceback
import types
from pathlib import Path, PurePath
from typing import (Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional,
                    Set, Tuple, Type, TypeVar, Union)
import contextvars
from tensorpc.core import inspecttools
from tensorpc.core.serviceunit import ReloadableDynamicClass
from tensorpc.core.tree_id import UniqueTreeIdForTree
from tensorpc.flow.flowapp.components import mui
from tensorpc.flow.flowapp.components.plus.core import (
    ALL_OBJECT_LAYOUT_HANDLERS, USER_OBJ_TREE_TYPES, CustomTreeItemHandler, ObjectLayoutCreator)
from tensorpc.flow.flowapp.core import FlowSpecialMethods
from tensorpc.flow.jsonlike import (IconButtonData, TreeItem,
                                    parse_obj_to_jsonlike)
# GLOBAL_SPLIT = "::"
STRING_LENGTH_LIMIT = 500
_IGNORE_ATTR_NAMES = set(["_abc_impl", "__abstractmethods__"])

SET_CONTAINER_LIMIT_SIZE = 50

class ButtonType(enum.Enum):
    Reload = "reload"
    Delete = "delete"
    Watch = "watch"
    Record = "record"

_SHOULD_EXPAND_TYPES = {mui.JsonLikeType.List.value, mui.JsonLikeType.Tuple.value,
                        mui.JsonLikeType.Dict.value, mui.JsonLikeType.Object.value,
                            mui.JsonLikeType.ListFolder.value, mui.JsonLikeType.DictFolder.value,
                            mui.JsonLikeType.Layout.value}


class ObjectTreeParser:
    """expandable: determine a object can be expand
    parseable: determine a object can be parsed to JsonLikeNode
    attr_parseable: determine a attribute of a object can be parsed to JsonLikeNode
    """
    def __init__(self, 
                 cared_types: Optional[Set[Type]] = None,
                 ignored_types: Optional[Set[Type]] = None,
                 custom_type_expanders: Optional[Dict[Type, Callable[[Any], dict]]] = None,
                 custom_tree_item_handler: Optional[CustomTreeItemHandler] = None):
        if cared_types is None:
            cared_types = set()
        if ignored_types is None:
            ignored_types = set()
        self._cared_types = cared_types
        self._ignored_types = ignored_types
        self._obj_meta_cache = {}
        self._cached_lazy_expand_uids: Set[UniqueTreeIdForTree] = set()
        if custom_type_expanders is None:
            custom_type_expanders = {}
        self._custom_type_expanders = custom_type_expanders
        self.custom_tree_item_handler = custom_tree_item_handler

    def parseable(self, obj, check_obj: bool = True):
        if not self._check_is_valid(obj) and check_obj:
            return False
        if inspecttools.is_obj_builtin_or_module(obj):
            return False
        return True 

    def attr_parseable(self, obj, attr_name: str, user_defined_prop_keys: Set[str], check_obj: bool = True):
        res = self.parseable(obj, check_obj)
        if not res:
            return False, None
        if attr_name.startswith("__"):
            return False, None
        if attr_name in _IGNORE_ATTR_NAMES:
            return False, None
        if attr_name in user_defined_prop_keys:
            return False, None
        try:
            v = getattr(obj, attr_name)
            isinstance(v, TreeItem)
        except:
            return False, None
        return True, v

    def _check_is_valid(self, obj_type):
        valid = True
        if len(self._cared_types) != 0:
            valid &= obj_type in self._cared_types
        if len(self._ignored_types) != 0:
            valid &= obj_type not in self._ignored_types
        return valid

    def _valid_check(self, obj_type):
        return True 

    async def expand_object(self, obj, check_obj: bool = True):
        if not self.parseable(obj, check_obj):
            return {}
        res: Dict[Any, Any] = {}
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
        elif isinstance(obj, tuple(USER_OBJ_TREE_TYPES)):
            return obj.get_childs()
        elif isinstance(obj, TreeItem):
            # this is very special, we need to lazy access the child of a treeitem.
            return await obj.get_child_desps() 
        if self.custom_tree_item_handler is not None:
            res_tmp = await self.custom_tree_item_handler.get_childs(obj)
            if res_tmp is not None:
                return res_tmp
        for k, v in self._custom_type_expanders.items():
            if isinstance(obj, k):
                return v(obj)
        user_defined_prop_keys = inspecttools.get_obj_userdefined_properties(obj)
        for k in dir(obj):
            valid, attr = self.attr_parseable(obj, k, user_defined_prop_keys, check_obj)
            if not valid:
                continue
            res[k] = attr
        return res

    async def parse_obj_to_tree_node(self, obj, name: str, obj_meta_cache=None):
        obj_type = type(obj)
        try:
            isinst = isinstance(obj, TreeItem)
        except:
            print("???", type(obj))
            raise 
        if isinst:
            node_candidate = obj.get_json_like_node()
            if node_candidate is not None:
                return node_candidate
        uid_name = UniqueTreeIdForTree.from_parts([name])
        node = parse_obj_to_jsonlike(obj, name, uid_name)
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
            return mui.JsonLikeNode(uid_name,
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

    async def parse_obj_dict_to_nodes(self, obj_dict: Mapping[Any, Any],
                    ns: UniqueTreeIdForTree, obj_meta_cache=None):
        res_node: List[mui.JsonLikeNode] = []
        for k, v in obj_dict.items():
            str_k = str(k)
            node = await self.parse_obj_to_tree_node(v,
                                str_k, obj_meta_cache)
            if self.custom_tree_item_handler is not None:
                node_tmp = self.custom_tree_item_handler.patch_node(v, node)
                if node_tmp is not None:
                    node = node_tmp
            # node.id = f"{ns}{GLOBAL_SPLIT}{str_k}"
            node.id = ns.append_part(str_k)
            if not isinstance(k, str):
                node.dictKey = mui.BackendOnlyProp(k)
            res_node.append(node)
        return res_node

    async def parse_obj_to_tree(self, obj, node: mui.JsonLikeNode, total_expand_level: int = 0):
        """parse object to json like tree.
        """
        if not self._should_expand_node(obj, node, total_expand_level):
            return 
        if isinstance(obj, TreeItem):
            obj_dict = await obj.get_child_desps()
            for k, v in obj_dict.items():
                # v.id = f"{node.id}{GLOBAL_SPLIT}{v.id}"
                v.id = node.id + v.id
            tree_children = list(obj_dict.values())
        else:
            obj_dict = await self.expand_object(obj)
            tree_children = await self.parse_obj_dict_to_nodes(obj_dict, node.id, self._obj_meta_cache)
        node.children = tree_children
        node.cnt = len(obj_dict)
        for (k, v), child_node in zip(obj_dict.items(), node.children):
            # should_expand = child_node.id in self._cached_lazy_expand_uids or total_expand_level > 0
            # if isinstance(v, TreeItem) and v.default_expand():
            #     should_expand = True
            # if should_expand:
            await self.parse_obj_to_tree(v, child_node,
                            total_expand_level - 1)
    
    def _should_expand_node(self, obj, node: mui.JsonLikeNode, total_expand_level: int):
        if node.type not in _SHOULD_EXPAND_TYPES:
            return False
        should_expand = node.id in self._cached_lazy_expand_uids or total_expand_level > 0
        if isinstance(obj, TreeItem) and obj.default_expand():
            should_expand = True
        elif isinstance(obj, tuple(USER_OBJ_TREE_TYPES)) and obj.default_expand():
            should_expand = True
        return should_expand


    def update_lazy_expand_uids(self, new_uid: UniqueTreeIdForTree):
        # if we lazy-expand a node, we should remove all its children from cached_lazy_expand_uids
        new_lazy_expand_uids: List[UniqueTreeIdForTree] = list(
            filter(lambda n: not n.startswith(new_uid),
                    self._cached_lazy_expand_uids))
        new_lazy_expand_uids.append(new_uid)
        self._cached_lazy_expand_uids = set(new_lazy_expand_uids)

    def get_obj_single_attr(self, obj,
                            key: str,
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
        if not self._check_is_valid(obj) and check_obj:
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
            if not (self._check_is_valid(v)):
                return mui.undefined
            if isinstance(v, types.ModuleType):
                return mui.undefined
            if inspect.isfunction(v) or inspect.ismethod(v) or inspect.isbuiltin(
                    v):
                return mui.undefined
            return v
        return mui.undefined


    async def get_obj_by_uid(
        self,
        obj,
        uid: UniqueTreeIdForTree,
        real_keys: Optional[List[Union[mui.Undefined, Hashable]]] = None
    ) -> Tuple[Any, bool]:
        parts = uid.parts
        if real_keys is None:
            real_keys = [mui.undefined for _ in range(len(parts))]
        if len(parts) == 1:
            return obj, True
        # uid contains root, remove it at first.
        return await self.get_obj_by_uid_resursive(obj, parts[1:], real_keys[1:])


    async def get_obj_by_uid_resursive(
            self, obj, parts: List[str], real_keys: List[Union[mui.Undefined, Hashable]]) -> Tuple[Any, bool]:
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
            child_obj = mui.undefined
            if self.custom_tree_item_handler is not None:
                childs = await self.custom_tree_item_handler.get_childs(obj)
                if childs is not None:
                    child_obj = childs[key]
            if isinstance(child_obj, mui.Undefined):
                child_obj = self.get_obj_single_attr(obj, key, check_obj=False)
                if isinstance(obj, mui.Undefined):
                    return obj, False
        if len(parts) == 1:
            return child_obj, True
        else:
            return await self.get_obj_by_uid_resursive(child_obj, parts[1:],
                                                real_keys[1:])


    async def get_obj_by_uid_trace(
        self, 
        obj,
        uid: UniqueTreeIdForTree,
        real_keys: Optional[List[Union[mui.Undefined, Hashable]]] = None
    ) -> Tuple[Any, bool]:
        parts = uid.parts
        if real_keys is None:
            real_keys = [mui.undefined for _ in range(len(parts))]
        if len(parts) == 1:
            return [obj], True
        # uid contains root, remove it at first.
        trace, found = await self.get_obj_by_uid_trace_resursive(
            obj, parts[1:], real_keys[1:])
        return [obj] + trace, found


    async def get_obj_by_uid_trace_resursive(
            self, obj, parts: List[str], real_keys: List[Union[mui.Undefined, Hashable]] ) -> Tuple[List[Any], bool]:
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
            child_obj = mui.undefined
            if self.custom_tree_item_handler is not None:
                childs = await self.custom_tree_item_handler.get_childs(obj)
                if childs is not None:
                    child_obj = childs[key]
            if isinstance(child_obj, mui.Undefined):
                child_obj = self.get_obj_single_attr(obj, key, check_obj=False)
                if isinstance(obj, mui.Undefined):
                    return [obj], False
        if len(parts) == 1:
            return [child_obj], True
        else:
            trace, found = await self.get_obj_by_uid_trace_resursive(
                child_obj, parts[1:], real_keys[1:])
            return [child_obj] + trace, found


    async def get_root_tree(self, obj_root, root_name: str, expand_level: int, ns: UniqueTreeIdForTree = UniqueTreeIdForTree("")):
        root_node = await self.parse_obj_to_tree_node(obj_root, root_name, self._obj_meta_cache)
        if not ns.empty():
            # root_node.id = f"{ns}{GLOBAL_SPLIT}{root_node.id}"
            root_node.id = ns + root_node.id
        await self.parse_obj_to_tree(obj_root, root_node, expand_level)
        return root_node

T = TypeVar("T")

class TreeContext:
    def __init__(self, parser: ObjectTreeParser, tree: Union[mui.JsonLikeTree, mui.TanstackJsonLikeTree], tree_instance: Any) -> None:
        self.parser = parser
        self.tree = tree
        self._tree_instance = tree_instance

    def get_tree_instance(self, tree_type: Type[T]) -> T:
        assert isinstance(self._tree_instance, tree_type)
        return self._tree_instance

TREE_CONTEXT_VAR: contextvars.ContextVar[
    Optional[TreeContext]] = contextvars.ContextVar("treectx",
                                                   default=None)


def get_tree_context() -> Optional[TreeContext]:
    return TREE_CONTEXT_VAR.get()

def get_tree_context_noexcept() -> TreeContext:
    res = TREE_CONTEXT_VAR.get()
    assert res is not None 
    return res 


@contextlib.contextmanager
def enter_tree_conetxt(ctx: TreeContext):
    """expose tree apis for user defined tree items.
    """
    token = TREE_CONTEXT_VAR.set(ctx)
    try:
        yield ctx
    finally:
        TREE_CONTEXT_VAR.reset(token)
