import contextlib
import dataclasses
import enum
import inspect
import traceback
import types
from pathlib import Path, PurePath
from typing import (Any, Callable, Dict, Hashable, Iterable, List, Mapping,
                    Optional, Set, Tuple, Type, TypeVar, Union)
import contextvars
from tensorpc.core import inspecttools
from tensorpc.core.serviceunit import ReloadableDynamicClass
from tensorpc.core.tree_id import UniqueTreeIdForTree
from tensorpc.dock.components import mui
from tensorpc.dock.components.plus.core import (
    ALL_OBJECT_LAYOUT_HANDLERS, USER_OBJ_TREE_TYPES, CustomTreeItemHandler,
    ObjectLayoutCreator, ContextMenuType)
from tensorpc.dock.components.plus.objinspect.treefilter import TreeExpandFilter, TreeExpandFilterDesc
from tensorpc.dock.components.three import is_three_component
from tensorpc.dock.core.component import FlowSpecialMethods
from tensorpc.dock.jsonlike import (IconButtonData, TreeItem,
                                    parse_obj_to_jsonlike)
# GLOBAL_SPLIT = "::"
STRING_LENGTH_LIMIT = 500
_IGNORE_ATTR_NAMES = set(["_abc_impl", "__abstractmethods__"])

SET_CONTAINER_LIMIT_SIZE = 100


class ButtonType(enum.Enum):
    Reload = "reload"
    Delete = "delete"
    Watch = "watch"
    Record = "record"


_SHOULD_EXPAND_TYPES = {
    mui.JsonLikeType.List.value, mui.JsonLikeType.Tuple.value,
    mui.JsonLikeType.Dict.value, mui.JsonLikeType.Object.value,
    mui.JsonLikeType.ListFolder.value, mui.JsonLikeType.DictFolder.value,
    mui.JsonLikeType.Layout.value
}

DEFAULT_EXPAND_METHOD = ""

@dataclasses.dataclass
class TreeExpandState:
    expand_level: int = 0
    validator: Optional[Callable[[Any], bool]] = None
    expand_method: str = DEFAULT_EXPAND_METHOD
    expand_filter: Optional[TreeExpandFilter] = None
    expand_is_recursive: bool = False
    check_obj: bool = True
    auto_expand_count_limit: int = -1

    def get_decreased_expand_level(self):
        return dataclasses.replace(self, expand_level=self.expand_level - 1)

    def should_keep(self, obj):
        if self.expand_filter is not None:
            return self.expand_filter.should_keep(obj, self.expand_method)
        return True

class ObjectTreeParser:
    """expandable: determine a object can be expand
    parseable: determine a object can be parsed to JsonLikeNode
    attr_parseable: determine a attribute of a object can be parsed to JsonLikeNode
    """

    def __init__(
            self,
            cared_types: Optional[Set[Type]] = None,
            ignored_types: Optional[Set[Type]] = None,
            custom_type_expanders: Optional[Dict[Type, Callable[[Any],
                                                                dict]]] = None,
            custom_tree_item_handler: Optional[CustomTreeItemHandler] = None,
            expand_filters: Optional[list[TreeExpandFilterDesc]] = None):
        if cared_types is None:
            cared_types = set()
        if ignored_types is None:
            ignored_types = set()
        self._cared_types = cared_types
        self._ignored_types = ignored_types
        self._obj_meta_cache = {}
        self._cached_lazy_expand_uids: dict[UniqueTreeIdForTree, str] = {}
        if custom_type_expanders is None:
            custom_type_expanders = {}
        self.custom_tree_item_handler = custom_tree_item_handler
        self._auto_expand_count_limit = 100

        self._tree_expand_filters: list[TreeExpandFilterDesc] = []
        if expand_filters is not None:
            self._tree_expand_filters = expand_filters
        self._tree_expand_filter_type_cache: dict[Any, TreeExpandFilterDesc] = {}

    def _cached_get_tree_expand_filter(self, obj: Any):
        obj_type = type(obj)
        if obj_type in self._tree_expand_filter_type_cache:
            return self._tree_expand_filter_type_cache[obj_type]
        for filter_desc in self._tree_expand_filters:
            if filter_desc.filter.is_cared_type(obj):
                self._tree_expand_filter_type_cache[obj_type] = filter_desc
                return filter_desc
        return None

    def parseable(self, obj, check_obj: bool = True):
        if check_obj and not self._check_is_valid(obj):
            return False
        if inspecttools.is_obj_builtin_or_module(obj):
            return False
        return True

    def attr_parseable(self,
                       obj,
                       attr_name: str,
                       user_defined_prop_keys: Set[str],
                       check_obj: bool = True):
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
            if isinstance(v, types.ModuleType):
                return False, None
            if inspect.isfunction(v) or inspect.ismethod(
                    v) or inspect.isbuiltin(v):
                return False, None
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

    async def expand_object(self, obj, check_obj: bool = True, start_for_list: int = 0):
        if not self.parseable(obj, check_obj):
            return {}
        res: Dict[Any, Any] = {}
        if isinstance(obj, (list, tuple, set)):
            if isinstance(obj, set):
                # set size is limited since it don't support nested view.
                obj_list = list(obj)[:SET_CONTAINER_LIMIT_SIZE]
            else:
                obj_list = obj
            len_obj = len(obj_list)
            return {str(i + start_for_list): obj_list[i] for i in range(len_obj)}
        elif isinstance(obj, dict):
            # return {k: v for k, v in obj.items() if not _is_obj_builtin_or_module(v)}
            return obj
        elif isinstance(obj, tuple(USER_OBJ_TREE_TYPES)):
            return obj.get_childs()
        elif isinstance(obj, TreeItem):
            # this is very special, we need to lazy access the child of a treeitem.
            return await obj.get_child_desps(UniqueTreeIdForTree(""))
        if self.custom_tree_item_handler is not None:
            res_tmp = await self.custom_tree_item_handler.get_childs(obj)
            if res_tmp is not None:
                return res_tmp
        user_defined_prop_keys = inspecttools.get_obj_userdefined_properties(
            obj)
        for k in dir(obj):
            valid, attr = self.attr_parseable(obj, k, user_defined_prop_keys,
                                              check_obj)
            if not valid:
                continue
            res[k] = attr
        return res

    async def parse_obj_to_tree_node(self,
                                     obj,
                                     name: str,
                                     obj_meta_cache=None):
        obj_type = type(obj)
        try:
            isinst = isinstance(obj, TreeItem)
        except:
            print("???", type(obj))
            raise
        uid_name = UniqueTreeIdForTree.from_parts([name])
        if isinst:
            node_candidate = obj.get_json_like_node(uid_name)
            if node_candidate is not None:
                return node_candidate
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
            if isinstance(obj, mui.Component) and not is_three_component(obj):
                is_layout = True
                is_draggable = obj._flow_reference_count == 0
            if isinstance(obj, ObjectLayoutCreator):
                is_draggable = True
                is_layout = True
            # is_draggable = True
            if is_layout:
                t = mui.JsonLikeType.Layout
            return mui.JsonLikeNode(
                uid_name,
                name,
                t.value,
                value=value,
                typeStr=obj_type.__qualname__,
                cnt=count,
                drag=is_draggable)
        return node

    async def parse_obj_dict_to_nodes(self,
                                      obj_dict: Mapping[Any, Any],
                                      ns: UniqueTreeIdForTree,
                                      obj_meta_cache=None):
        res_node: List[mui.JsonLikeNode] = []
        for k, v in obj_dict.items():
            new_node_id = ns.append_part(str(k))
            node = await self._parse_obj_to_tree_node_v2(v, k, new_node_id)
            res_node.append(node)
        return res_node

    async def _parse_obj_to_tree_node_v2(self, v: Any, k: Any, node_id: UniqueTreeIdForTree):
        str_k = str(k)
        node = await self.parse_obj_to_tree_node(v, str_k, self._obj_meta_cache)
        if self.custom_tree_item_handler is not None:
            node_tmp = self.custom_tree_item_handler.patch_node(v, node)
            if node_tmp is not None:
                node = node_tmp
        expand_filter_desc = self._cached_get_tree_expand_filter(v)
        if expand_filter_desc is not None:
            menuitems = expand_filter_desc.filter.get_expand_menu_items().copy()
            for i in range(len(menuitems)):
                menuitems[i] = dataclasses.replace(
                    menuitems[i], userdata={
                        "type": ContextMenuType.CustomExpand.value,
                        "method": menuitems[i].id,
                    }
                )
            if isinstance(node.menus, list):
                if menuitems:
                    menuitems.append(mui.MenuItem(id="custom_expand_divider", divider=True))
                node.menus.extend(menuitems)
            else:
                node.menus = menuitems
        node.id = node_id
        if not isinstance(k, str):
            node.dictKey = mui.BackendOnlyProp(k)
        return node

    async def parse_obj_childs_to_tree(self, obj: Any, node_id: UniqueTreeIdForTree, start_for_list: int = 0):
        obj_dict = await self.expand_object(obj, start_for_list=start_for_list)
        tree_children = await self.parse_obj_dict_to_nodes(
            obj_dict, node_id, self._obj_meta_cache)
        return tree_children, len(obj_dict)

    def _get_next_expand_state(self, obj: Any, node_id: UniqueTreeIdForTree, state: TreeExpandState):
        expand_method = state.expand_method
        expand_filter = state.expand_filter
        expand_is_recursive = state.expand_is_recursive
        if not state.expand_is_recursive:
            # if not recursive, remove parent filter.
            expand_method = DEFAULT_EXPAND_METHOD
            expand_filter = None 
        if node_id in self._cached_lazy_expand_uids:
            expand_method = self._cached_lazy_expand_uids[node_id]
        if expand_method != DEFAULT_EXPAND_METHOD:
            expand_filter_desc = self._cached_get_tree_expand_filter(obj)
            if expand_filter_desc is not None:
                expand_filter = expand_filter_desc.filter
                expand_is_recursive = expand_filter_desc.is_recursive
            else:
                expand_method = DEFAULT_EXPAND_METHOD
        return dataclasses.replace(state, expand_method=expand_method,
                                   expand_filter=expand_filter,
                                   expand_is_recursive=expand_is_recursive,
                                   expand_level=state.expand_level - 1)
    
    async def parse_obj_childs_to_tree_v2(self, obj: Any, node: mui.JsonLikeNode, start_for_list: int = 0, expand_method: str = DEFAULT_EXPAND_METHOD, custom_expand_level: int = 1000) -> list[mui.JsonLikeNode]:
        state = TreeExpandState(
                expand_level=1,
                validator=None,
                check_obj=True)
        if expand_method != DEFAULT_EXPAND_METHOD:
            filter_desc = self._cached_get_tree_expand_filter(obj)
            if filter_desc is not None:
                state = TreeExpandState(
                    expand_level=custom_expand_level,
                    validator=None,
                    expand_method=expand_method,
                    expand_filter=filter_desc.filter,
                    expand_is_recursive=filter_desc.is_recursive,
                    check_obj=True)
        return await self.parse_obj_childs_to_tree_v2_recursive(
            obj, node, state, start_for_list)

    async def _get_child_nodes_dict(self, obj: Any, k: Any, node_id: UniqueTreeIdForTree, state: TreeExpandState):
        next_node = await self._parse_obj_to_tree_node_v2(obj, k, node_id)
        next_node.children = await self.parse_obj_childs_to_tree_v2_recursive(
            obj, next_node, state)
        return next_node

    async def parse_obj_childs_to_tree_v2_recursive(self, obj: Any, node: mui.JsonLikeNode, state: TreeExpandState, start_for_list: int = 0) -> list[mui.JsonLikeNode]:
        if not self.parseable(obj, state.check_obj):
            return []
        should_expand = self._should_expand_node(obj, node, state)

        if not should_expand:
            # if state.expand_method != DEFAULT_EXPAND_METHOD:
            return []
        res: list[mui.JsonLikeNode] = []
        if isinstance(obj, (list, tuple, set)):
            if isinstance(obj, set):
                # set size is limited since it don't support nested view.
                obj_list = list(obj)[:SET_CONTAINER_LIMIT_SIZE]
            else:
                obj_list = obj
            len_obj = len(obj_list)
            for i in range(len_obj):
                child_obj = obj_list[i]
                new_node_id = node.id.append_part(str(i + start_for_list))
                next_state = self._get_next_expand_state(child_obj, new_node_id, state)
                if not state.should_keep(child_obj):
                    continue 
                res.append(await self._get_child_nodes_dict(child_obj, str(i + start_for_list), new_node_id, next_state))
            return res 
        elif isinstance(obj, dict):
            for k, v in obj.items():
                new_node_id = node.id.append_part(str(k))
                next_state = self._get_next_expand_state(v, new_node_id, state)
                if not state.should_keep(v):
                    continue 
                res.append(await self._get_child_nodes_dict(v, k, new_node_id, next_state))
            return res 
        elif isinstance(obj, tuple(USER_OBJ_TREE_TYPES)):
            for k, v in obj.get_childs():
                new_node_id = node.id.append_part(str(k))
                next_state = self._get_next_expand_state(v, new_node_id, state)
                if not state.should_keep(v):
                    continue 
                res.append(await self._get_child_nodes_dict(v, k, new_node_id, next_state))
            return res 
        elif isinstance(obj, TreeItem):
            # this is very special, we need to lazy access the child of a treeitem.
            for k, v in (await obj.get_child_desps(UniqueTreeIdForTree(""))).items():
                new_node_id = node.id.append_part(str(k))
                next_state = self._get_next_expand_state(v, new_node_id, state)
                if not state.should_keep(v):
                    continue 
                res.append(await self._get_child_nodes_dict(v, k, new_node_id, next_state))
            return res 
        if self.custom_tree_item_handler is not None:
            res_tmp = await self.custom_tree_item_handler.get_childs(obj)
            if res_tmp is not None:
                for k, v in res_tmp:
                    if not state.should_keep(v):
                        continue 
                    new_node_id = node.id.append_part(str(k))
                    next_state = self._get_next_expand_state(v, new_node_id, state)
                    if not state.should_keep(v):
                        continue 
                    res.append(await self._get_child_nodes_dict(v, k, new_node_id, next_state))
                return res 
        filter_desc = self._cached_get_tree_expand_filter(obj)
        if filter_desc is not None:
            childs_may_none = filter_desc.filter.expand_dict(obj)
            if childs_may_none is not None:
                for k, v in childs_may_none.items():
                    new_node_id = node.id.append_part(str(k))
                    next_state = self._get_next_expand_state(v, new_node_id, state)
                    if not state.should_keep(v):
                        continue 
                    res.append(await self._get_child_nodes_dict(v, k, new_node_id, next_state))
                return res 
        user_defined_prop_keys = inspecttools.get_obj_userdefined_properties(
            obj)
        for k in dir(obj):
            valid, attr = self.attr_parseable(obj, k, user_defined_prop_keys,
                                              state.check_obj)
            if not valid:
                continue
            new_node_id = node.id.append_part(str(k))
            next_state = self._get_next_expand_state(attr, new_node_id, state)
            if not state.should_keep(attr):
                continue 
            res.append(await self._get_child_nodes_dict(attr, k, new_node_id, next_state))
        return res

    async def get_root_tree_v2(self,
                            obj_root,
                            root_name: str,
                            expand_level: int,
                            ns: UniqueTreeIdForTree = UniqueTreeIdForTree(""),
                            validator: Optional[Callable[[Any], bool]] = None):
        root_id = UniqueTreeIdForTree.from_parts([root_name])
        if not ns.empty():
            # root_node.id = f"{ns}{GLOBAL_SPLIT}{root_node.id}"
            root_id = ns + root_id

        root_node = await self._parse_obj_to_tree_node_v2(obj_root, root_name,
                                                      root_id)
        state = TreeExpandState(
            expand_level=expand_level,
            validator=validator,
            auto_expand_count_limit=self._auto_expand_count_limit)
        children = await self.parse_obj_childs_to_tree_v2_recursive(obj_root, root_node, state)
        root_node.children = children
        return root_node

    async def parse_obj_to_tree(self,
                                obj,
                                node: mui.JsonLikeNode,
                                total_expand_level: int = 0,
                                validator: Optional[Callable[[Any], bool]] = None):
        state = TreeExpandState(
            expand_level=total_expand_level,
            validator=validator)
        return await self._parse_obj_to_tree_recursive(obj, node, state)


    async def _parse_obj_to_tree_recursive(self,
                                obj,
                                node: mui.JsonLikeNode,
                                state: TreeExpandState):
        """parse object to json like tree.
        """
        if not self._should_expand_node(obj, node, state):
            return
        if isinstance(obj, TreeItem):
            obj_dict = await obj.get_child_desps(node.id)
            tree_children = list(obj_dict.values())
            real_length = len(obj_dict)
        else:
            obj_dict = await self.expand_object(obj)
            tree_children = await self.parse_obj_dict_to_nodes(
                obj_dict, node.id, self._obj_meta_cache)
            real_length = len(obj_dict)
        node.children = tree_children
        node.cnt = real_length
        for (k, v), child_node in zip(obj_dict.items(), node.children):
            if isinstance(obj, TreeItem):
                v = await obj.get_child(k)
            await self._parse_obj_to_tree_recursive(v, child_node, state.get_decreased_expand_level())

    def _should_expand_node(self, obj, node: mui.JsonLikeNode,
                            state: TreeExpandState) -> bool:
        if node.type not in _SHOULD_EXPAND_TYPES:
            return False
        if state.validator is not None:
            if not state.validator(obj):
                return False 
        if state.expand_filter is not None:
            should_expand = not state.expand_filter.is_leaf(obj, state.expand_method)
            assert state.expand_method != DEFAULT_EXPAND_METHOD
            # when user use custom expand method, we override 
            # the default expand behavior.
            return should_expand
        should_expand = node.id in self._cached_lazy_expand_uids or state.expand_level > 0
        if isinstance(obj, TreeItem) and obj.default_expand():
            should_expand = state.expand_level > 0
        elif isinstance(obj,
                        tuple(USER_OBJ_TREE_TYPES)) and obj.default_expand():
            should_expand = True
        elif state.auto_expand_count_limit > 0 and node.cnt > state.auto_expand_count_limit:
            should_expand = False
        return should_expand

    def update_lazy_expand_uids(self, new_uid: UniqueTreeIdForTree, expand_method: str = DEFAULT_EXPAND_METHOD):
        # if we lazy-expand a node, we should remove all its children from cached_lazy_expand_uids
        new_lazy_expand_uids: dict[UniqueTreeIdForTree, str] = {}
        for k, v in self._cached_lazy_expand_uids.items():
            if k.startswith(new_uid):
                # remove all its children
                continue
            new_lazy_expand_uids[k] = v
        new_lazy_expand_uids[new_uid] = expand_method
        self._cached_lazy_expand_uids = new_lazy_expand_uids

    def get_obj_single_attr(
            self,
            obj,
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
            if inspect.isfunction(v) or inspect.ismethod(
                    v) or inspect.isbuiltin(v):
                return mui.undefined
            return v
        return mui.undefined

    async def get_obj_by_uid_trace(
        self,
        obj,
        uid: UniqueTreeIdForTree,
        real_keys: Optional[List[Union[mui.Undefined, Hashable]]] = None
    ) -> Tuple[list[Any], bool]:
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
        self, obj, parts: List[str],
        real_keys: List[Union[mui.Undefined,
                              Hashable]]) -> Tuple[List[Any], bool]:
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
            filter_desc = self._cached_get_tree_expand_filter(obj)
            if filter_desc is not None:
                childs_may_none = filter_desc.filter.expand_dict(obj)
                if childs_may_none is not None:
                    child_obj = childs_may_none[key]
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

    async def get_root_tree(self,
                            obj_root,
                            root_name: str,
                            expand_level: int,
                            ns: UniqueTreeIdForTree = UniqueTreeIdForTree(""),
                            validator: Optional[Callable[[Any], bool]] = None):
        root_node = await self.parse_obj_to_tree_node(obj_root, root_name,
                                                      self._obj_meta_cache)
        if not ns.empty():
            # root_node.id = f"{ns}{GLOBAL_SPLIT}{root_node.id}"
            root_node.id = ns + root_node.id
        await self.parse_obj_to_tree(obj_root, root_node, expand_level, validator)
        return root_node


T = TypeVar("T")


class TreeContext:

    def __init__(self, parser: ObjectTreeParser,
                 tree: Union[mui.JsonLikeTree, mui.TanstackJsonLikeTree],
                 tree_instance: Any) -> None:
        self.parser = parser
        self.tree = tree
        self._tree_instance = tree_instance

    def get_tree_instance(self, tree_type: Type[T]) -> T:
        assert isinstance(self._tree_instance, tree_type)
        return self._tree_instance


TREE_CONTEXT_VAR: contextvars.ContextVar[
    Optional[TreeContext]] = contextvars.ContextVar("treectx", default=None)


def get_tree_context() -> Optional[TreeContext]:
    return TREE_CONTEXT_VAR.get()


def get_tree_context_noexcept() -> TreeContext:
    res = TREE_CONTEXT_VAR.get()
    assert res is not None
    return res


@contextlib.contextmanager
def enter_tree_context(ctx: TreeContext):
    """expose tree apis for user defined tree items.
    """
    token = TREE_CONTEXT_VAR.set(ctx)
    try:
        yield ctx
    finally:
        TREE_CONTEXT_VAR.reset(token)
