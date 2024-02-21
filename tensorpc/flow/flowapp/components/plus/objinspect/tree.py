import dataclasses
import enum
import inspect
import time
import traceback
import types
from functools import partial
from pathlib import Path, PurePath
from typing import (Any, Callable, Dict, Hashable, Iterable, List, Optional,
                    Set, Tuple, Type, Union)

import numpy as np
from tensorpc.core.tree_id import UniqueTreeId, UniqueTreeIdForTree

from tensorpc.flow.flowapp import appctx
from tensorpc.flow.flowapp.appcore import AppSpecialEventType
from tensorpc.flow.flowapp.components import mui, three
from tensorpc.flow.flowapp.components.plus.canvas import SimpleCanvas
from tensorpc.flow.flowapp.components.plus.collection import SimpleFileReader, ScriptExecutor
from tensorpc.flow.flowapp.components.plus.scriptmgr import ScriptManager

from tensorpc.flow.flowapp.components.plus.monitor import \
    ComputeResourceMonitor
from tensorpc.flow.flowapp.components.plus.core import (
    ALL_OBJECT_LAYOUT_HANDLERS, ALL_OBJECT_PREVIEW_HANDLERS,
    USER_OBJ_TREE_TYPES, ContextMenuType, CustomTreeItemHandler, ObjectLayoutCreator,
    ObjectLayoutHandler)
from tensorpc.flow.flowapp.core import FlowSpecialMethods, FrontendEventType
from tensorpc.flow.flowapp.coretypes import TreeDragTarget
from tensorpc.flow.flowapp.objtree import UserObjTree, UserObjTreeProtocol
from tensorpc.flow.flowapp.reload import reload_object_methods
from tensorpc.flow.jsonlike import (CommonQualNames, ContextMenuData,
                                    IconButtonData, parse_obj_to_jsonlike, TreeItem)
from tensorpc.flow.marker import mark_did_mount, mark_will_unmount
from .controllers import CallbackSlider, ThreadLocker, MarkdownViewer
from .analysis import ObjectTreeParser, TreeContext, enter_tree_conetxt

_DEFAULT_OBJ_NAME = "default"

_DEFAULT_BUILTINS_NAME = "builtins"
_DEFAULT_DATA_STORAGE_NAME = "flowStorage"
_DEFAULT_OBSERVED_FUNC_NAME = "observedFuncs"

_ROOT = "root"

BASE_OBJ_TO_TYPE = {
    int: mui.JsonLikeType.Int,
    float: mui.JsonLikeType.Float,
    complex: mui.JsonLikeType.Complex,
    bool: mui.JsonLikeType.Bool,
    str: mui.JsonLikeType.String,
}

CONTAINER_TYPES = {mui.JsonLikeType.List.value, mui.JsonLikeType.Dict.value}
FOLDER_TYPES = {
    mui.JsonLikeType.ListFolder.value, mui.JsonLikeType.DictFolder.value
}


class ButtonType(enum.Enum):
    Reload = "reload"
    Delete = "delete"
    Watch = "watch"
    Record = "record"

# _SHOULD_EXPAND_TYPES = {mui.JsonLikeType.List.value, mui.JsonLikeType.Tuple.value,
#                         mui.JsonLikeType.Dict.value, mui.JsonLikeType.Object.value,
#                             mui.JsonLikeType.ListFolder.value, mui.JsonLikeType.DictFolder.value,
#                             mui.JsonLikeType.Layout.value}

# def _check_is_valid(obj_type, cared_types: Set[Type],
#                     ignored_types: Set[Type]):
#     valid = True
#     if len(cared_types) != 0:
#         valid &= obj_type in cared_types
#     if len(ignored_types) != 0:
#         valid &= obj_type not in ignored_types
#     return valid


# def _get_root_tree(obj,
#                    checker: Callable[[Type], bool],
#                    key: str,
#                    obj_meta_cache=None):
#     obj_dict = get_obj_dict(obj, checker)
#     root_node = parse_obj_item(obj, key, key, checker, obj_meta_cache)
#     root_node.children = parse_obj_dict_to_nodes(obj_dict, key, checker, obj_meta_cache)
#     for (k, o), c in zip(obj_dict.items(), root_node.children):
#         obj_child_dict = get_obj_dict(o, checker)
#         c.drag = False
#         c.children = parse_obj_dict_to_nodes(obj_child_dict, c.id, checker,
#                                     obj_meta_cache)
#     root_node.cnt = len(obj_dict)
#     return root_node


# async def _parse_obj_to_node(obj,
#                        node: mui.JsonLikeNode,
#                        checker: Callable[[Type], bool],
#                        cached_lazy_expand_ids: Set[str],
#                        obj_meta_cache=None,
#                        total_expand_level: int = 0):
#     if node.type not in _SHOULD_EXPAND_TYPES:
#         return 
#     if isinstance(obj, TreeItem):
#         obj_dict = await obj.get_child_desps(node.id)  # type: ignore
#         tree_children = list(obj_dict.values())
#     else:
#         obj_dict = get_obj_dict(obj, checker)
#         tree_children = parse_obj_dict_to_nodes(obj_dict, node.id, checker, obj_meta_cache)
#     node.children = tree_children
#     node.cnt = len(obj_dict)
#     for (k, v), child_node in zip(obj_dict.items(), node.children):
#         should_expand = child_node.id in cached_lazy_expand_ids or total_expand_level > 0
#         if isinstance(v, TreeItem) and v.default_expand():
#             should_expand = True
#         if should_expand:
#             await _parse_obj_to_node(v, child_node, checker, cached_lazy_expand_ids,
#                                obj_meta_cache, total_expand_level - 1)


# async def _get_obj_tree(obj,
#                   checker: Callable[[Type], bool],
#                   key: str,
#                   parent_id: str,
#                   obj_meta_cache=None,
#                   cached_lazy_expand_ids: Optional[List[str]] = None,
#                   total_expand_level: int = 0):
#     if parent_id == "":
#         obj_id = key 
#     else:
#         obj_id = f"{parent_id}{GLOBAL_SPLIT}{key}"
#     assert total_expand_level >= 0
#     root_node = parse_obj_item(obj, key, obj_id, checker, obj_meta_cache)
#     if cached_lazy_expand_ids is None:
#         cached_lazy_expand_ids = []
#     cached_lazy_expand_ids_set = set(cached_lazy_expand_ids)
#     # TODO determine auto-expand limits
#     if root_node.type == mui.JsonLikeType.Object.value:

#         await _parse_obj_to_node(obj, root_node, checker, cached_lazy_expand_ids_set,
#                            obj_meta_cache)
#         # obj_dict = get_obj_dict(obj, checker)
#         # root_node.children = parse_obj_dict_to_nodes(obj_dict, obj_id, checker,
#         #                                     obj_meta_cache)
#         # root_node.cnt = len(obj_dict)
#     else:
#         should_expand = obj_id in cached_lazy_expand_ids_set or total_expand_level >= 0
#         if isinstance(obj, TreeItem) and obj.default_expand():
#             should_expand = True
#         if should_expand:
#             await _parse_obj_to_node(obj, root_node, checker,
#                                cached_lazy_expand_ids_set, obj_meta_cache, total_expand_level=total_expand_level)

#     return root_node


class DataStorageTreeItem(TreeItem):

    def __init__(self, node_id: str, readable_id: str) -> None:
        super().__init__()
        self.node_id = node_id
        self.readable_id = readable_id

    async def get_child_desps(self) -> Dict[str, mui.JsonLikeNode]:
        metas = await appctx.list_data_storage(self.node_id)
        for m in metas:
            userdata = {
                "type": ContextMenuType.DataStorageItemDelete.value,
            }
            userdata_cpycmd = {
                "type": ContextMenuType.DataStorageItemCommand.value,
            }

            m.menus = [
                ContextMenuData("Delete",
                                m.id.uid_encoded,
                                mui.IconType.Delete.value,
                                userdata=userdata),
                ContextMenuData("Copy Command",
                                m.id.uid_encoded,
                                userdata=userdata_cpycmd),
            ]
            m.edit = True
            # m.cnt = 1
            # m.iconBtns = [(ButtonType.Delete.value,
            #                         mui.IconType.Delete.value)]
        return {m.last_part(): m for m in metas}

    async def get_child(self, key: str) -> Any:
        res = await appctx.read_data_storage(key, self.node_id)
        return res

    def get_json_like_node(self) -> Optional[mui.JsonLikeNode]:
        btns = [
            IconButtonData(ButtonType.Delete.value, mui.IconType.Delete.value)
        ]
        return mui.JsonLikeNode(UniqueTreeIdForTree.from_parts([self.readable_id]),
                                self.readable_id,
                                mui.JsonLikeType.Object.value,
                                typeStr="DataStorageTreeItem",
                                cnt=1,
                                drag=False,
                                iconBtns=btns)

    async def handle_button(self, button_key: str):
        if button_key == ButtonType.Delete.value:
            # clear
            await appctx.remove_data_storage(None, self.node_id)
            return True
        return

    async def handle_child_button(self, button_key: str, child_key: str):
        if button_key == ButtonType.Delete.value:
            await appctx.remove_data_storage(child_key, self.node_id)
            return True
        return

    async def handle_child_rename(self, child_key: str, newname: str):
        await appctx.rename_data_storage_item(child_key, newname, self.node_id)
        return True

    async def handle_child_context_menu(self, child_key: str,
                                        userdata: Dict[str, Any]):
        type = ContextMenuType(userdata["type"])
        if type == ContextMenuType.DataStorageItemDelete:
            await appctx.remove_data_storage(child_key, self.node_id)
            return True  # tell outside update childs
        if type == ContextMenuType.DataStorageItemCommand:
            await appctx.get_app().copy_text_to_clipboard(f"await appctx.read_data_storage('{child_key}', '{self.node_id}')")

    async def handle_context_menu(self, userdata: Dict[str, Any]):
        return


class ObservedFunctionTree(TreeItem):

    def __init__(self) -> None:
        super().__init__()
        self._watched_funcs: Set[str] = set()

    # async def get_childs(self):

    async def get_child_desps(self) -> Dict[str, mui.JsonLikeNode]:
        metas: Dict[str, mui.JsonLikeNode] = {}
        for k, v in appctx.get_app().get_observed_func_registry().items():
            node = mui.JsonLikeNode(UniqueTreeIdForTree.from_parts([k]), v.name,
                                    mui.JsonLikeType.Function.value, alias=v.name)
            node.iconBtns = [
                IconButtonData(ButtonType.Watch.value,
                               mui.IconType.Visibility.value, "Watch"),
                IconButtonData(ButtonType.Record.value, mui.IconType.Mic.value,
                               "Record")
            ]
            value = ""
            if k in self._watched_funcs:
                value += f"👀"
            if v.enable_args_record:
                value += f"🎙️"
            if v.recorded_data is not None:
                value += f"💾"
            node.value = value
            metas[k] = node
        return metas

    async def get_child(self, key: str) -> Any:
        return appctx.get_app().get_observed_func_registry()[key]

    def get_json_like_node(self) -> Optional[mui.JsonLikeNode]:
        return mui.JsonLikeNode(UniqueTreeIdForTree.from_parts(["observedFunc"]),
                                "observedFunc",
                                mui.JsonLikeType.Object.value,
                                typeStr="ObservedFunctions",
                                cnt=1,
                                drag=False)

    async def handle_child_button(self, button_key: str, child_key: str):
        if button_key == ButtonType.Watch.value:
            if child_key in self._watched_funcs:
                self._watched_funcs.remove(child_key)
            else:
                self._watched_funcs.add(child_key)
            return True
        rg = appctx.get_app().get_observed_func_registry()
        if button_key == ButtonType.Record.value:
            if child_key in rg:
                entry = rg[child_key]
                if entry.enable_args_record:
                    entry.enable_args_record = False
                else:
                    entry.enable_args_record = True
            return True
        return


class SimpleCanvasCreator(ObjectLayoutCreator):

    def create(self):
        return SimpleCanvas()

class BasicTreeEventType(enum.IntEnum):
    SelectSingle = 0

@dataclasses.dataclass
class SelectSingleEvent:
    nodes: List[mui.JsonLikeNode]
    objs: Optional[List[Any]] = None

class BasicObjectTree(mui.FlexBox):
    """basic object tree, contains enough features
    to analysis python object.
    TODO auto expand child when you expand a node.
    """

    def __init__(self,
                 init: Optional[Any] = None,
                 cared_types: Optional[Set[Type]] = None,
                 ignored_types: Optional[Set[Type]] = None,
                 limit: int = 50,
                 use_fast_tree: bool = False,
                 auto_lazy_expand: bool = True,
                 default_expand_level: int = 2,
                 fixed_size: bool = False,
                 custom_tree_handler: Optional[CustomTreeItemHandler] = None,
                 use_init_as_root: bool = False) -> None:
        if use_fast_tree:
            self.tree = mui.TanstackJsonLikeTree()
        else:
            self.tree = mui.JsonLikeTree()
        super().__init__([
            self.tree.prop(ignoreRoot=True, fixedSize=fixed_size),
        ])
        self.prop(overflow="auto")
        self._tree_parser = ObjectTreeParser(cared_types, ignored_types, custom_tree_item_handler=custom_tree_handler)
        self._uid_to_node: Dict[str, mui.JsonLikeNode] = {}
        if cared_types is None:
            cared_types = set()
        if ignored_types is None:
            ignored_types = set()
        self._cared_types = cared_types
        self._ignored_types = ignored_types
        self._obj_meta_cache = {}
        self._auto_lazy_expand = auto_lazy_expand
        # self._cached_lazy_expand_uids: List[str] = []
        self._cared_dnd_uids: Dict[UniqueTreeIdForTree, Callable[[UniqueTreeIdForTree, Any],
                                                 mui.CORO_NONE]] = {}
        self.limit = limit
        if init is None:
            self.root = {}
            # self.tree.props.tree = mui.JsonLikeNode(
            #     _ROOT, _ROOT, mui.JsonLikeType.Dict.value)
        else:
            if use_init_as_root:
                self.root = init
            else:
                self.root = {_DEFAULT_OBJ_NAME: init}
        # self.tree.props.tree = _get_root_tree(self.root, self._valid_checker,
        #                                       _ROOT, self._obj_meta_cache)
        # print(self.tree.props.tree)
        # inspect.isbuiltin()
        self.tree.register_event_handler(
            FrontendEventType.TreeLazyExpand.value, self._on_expand)
        self.tree.register_event_handler(
            FrontendEventType.TreeItemButton.value, self._on_custom_button)
        self.tree.register_event_handler(
            FrontendEventType.ContextMenuSelect.value, self._on_contextmenu)
        self.tree.register_event_handler(
            FrontendEventType.TreeItemRename.value, self._on_rename)
        self.tree.register_event_handler(
            FrontendEventType.TreeItemSelectChange.value, self._on_select_single)

        self.tree.register_event_handler(FrontendEventType.DragCollect.value,
                                         self._on_drag_collect,
                                         backend_only=True)
        self.default_expand_level = default_expand_level

        self.event_async_select_single = self._create_emitter_event_slot(BasicTreeEventType.SelectSingle)

    @mark_did_mount
    async def _on_mount(self):
        # self.tree.props.tree = await _get_root_tree_async(
        #     self.root, self._valid_checker, _ROOT, self._obj_meta_cache)
        with enter_tree_conetxt(TreeContext(self._tree_parser, self.tree, self)):
            root_node = await self._tree_parser.get_root_tree(self.root, _ROOT, self.default_expand_level)
        # self.tree.props.tree = await _get_obj_tree(
        #     self.root, self._valid_checker, _ROOT, "", self._obj_meta_cache, total_expand_level=self.default_expand_level)
        self.tree.props.tree = root_node
        await self.tree.send_and_wait(
            self.tree.update_event(tree=self.tree.props.tree))

    # def _checker(self, obj):
    #     return _check_is_valid(type(obj), self._cared_types,
    #                            self._ignored_types)

    # def _valid_checker(self, obj):
    #     return True

    @property
    def _objinspect_root(self):
        return self.tree.props.tree

    async def _get_obj_by_uid(self, uid: UniqueTreeIdForTree,
                              tree_node_trace: List[mui.JsonLikeNode]):
        uids = uid.parts
        real_uids: List[str] = []
        real_keys: List[Union[Hashable, mui.Undefined]] = []
        for i in range(len(tree_node_trace)):
            k = tree_node_trace[i].get_dict_key()
            if not tree_node_trace[i].is_folder():
                real_keys.append(k)
                real_uids.append(uids[i])
        with enter_tree_conetxt(TreeContext(self._tree_parser, self.tree, self)):
            return await self._tree_parser.get_obj_by_uid(self.root,
                                        UniqueTreeIdForTree.from_parts(real_uids),
                                        real_keys=real_keys)

    async def _get_obj_by_uid_trace(self, uid: UniqueTreeIdForTree,
                                    tree_node_trace: List[mui.JsonLikeNode]):
        uids = uid.parts
        real_uids: List[str] = []
        real_keys: List[Union[Hashable, mui.Undefined]] = []
        for i in range(len(tree_node_trace)):
            k = tree_node_trace[i].get_dict_key()
            if not tree_node_trace[i].is_folder():
                real_keys.append(k)
                real_uids.append(uids[i])
        with enter_tree_conetxt(TreeContext(self._tree_parser, self.tree, self)):
            return await self._tree_parser.get_obj_by_uid_trace(self.root,
                                            UniqueTreeIdForTree.from_parts(real_uids),
                                            real_keys=real_keys)

    def _register_dnd_uid(self, uid: UniqueTreeIdForTree, cb: Callable[[UniqueTreeIdForTree, Any],
                                                       mui.CORO_NONE]):
        self._cared_dnd_uids[uid] = cb

    def _unregister_dnd_uid(self, uid: UniqueTreeIdForTree):
        if uid in self._cared_dnd_uids:
            self._cared_dnd_uids.pop(uid)

    def _unregister_all_dnd_uid(self):
        self._cared_dnd_uids.clear()

    async def _do_when_tree_updated(self, updated_uid: UniqueTreeIdForTree):
        # iterate all cared dnd uids
        # if updated, launch callback to perform dnd
        deleted: List[UniqueTreeIdForTree] = []
        for k, v in self._cared_dnd_uids.items():
            if k.startswith(updated_uid):
                k_obj = k
                nodes, found = self._objinspect_root._get_node_by_uid_trace_found(
                    k_obj.parts)
                if not found:
                    # remove from cared
                    deleted.append(k)
                else:
                    obj, found = await self._get_obj_by_uid(k_obj, nodes)
                    res = v(k, obj)
                    if inspect.iscoroutine(res):
                        await res
        for d in deleted:
            self._cared_dnd_uids.pop(d)

    async def _on_drag_collect(self, data):
        uid: str = data["id"]
        uid_obj = UniqueTreeIdForTree(uid)
        with enter_tree_conetxt(TreeContext(self._tree_parser, self.tree, self)):
            objs, found = await self._tree_parser.get_obj_by_uid_trace(self.root, uid_obj)
        if not found:
            return None
        tab_id = ""
        if "complexLayoutTabNodeId" in data:
            # for complex layout UI: FlexLayout
            tab_id = data["complexLayoutTabNodeId"]
        root: Optional[UserObjTreeProtocol] = None
        for obj in objs:
            if isinstance(obj, tuple(USER_OBJ_TREE_TYPES)):
                root = obj
                break
        if root is not None:
            return TreeDragTarget(objs[-1], uid, tab_id, self._flow_uid_encoded,
                                  lambda: root.enter_context(root))
        return TreeDragTarget(objs[-1], uid, tab_id, self._flow_uid_encoded)
    
    async def _on_select_single(self, uid_list: Union[List[str], str, Dict[str, bool]]):
        if isinstance(uid_list, list):
            # node id list may empty
            if not uid_list:
                return
            uid = uid_list[0]
        elif isinstance(uid_list, dict):
            if not uid_list:
                return
            uid = list(uid_list.keys())[0]
        else:
            uid = uid_list
        uid_obj = UniqueTreeIdForTree(uid)
        nodes = self._objinspect_root._get_node_by_uid_trace(uid_obj.parts)
        objs, found = await self._get_obj_by_uid_trace(uid_obj, nodes)
        return await self.flow_event_emitter.emit_async(BasicTreeEventType.SelectSingle, 
            mui.Event(BasicTreeEventType.SelectSingle, SelectSingleEvent(nodes, objs if found else None)))

    async def _on_expand(self, uid_encoded: str):
        with enter_tree_conetxt(TreeContext(self._tree_parser, self.tree, self)):
            uid = UniqueTreeIdForTree(uid_encoded)
            node = self._objinspect_root._get_node_by_uid(uid_encoded)
            nodes, node_found = self._objinspect_root._get_node_by_uid_trace_found(
                uid.parts)
            assert node_found, "can't find your node via uid"
            node = nodes[-1]
            obj_dict: Dict[Hashable, Any] = {}
            if self._auto_lazy_expand:
                self._tree_parser.update_lazy_expand_uids(uid)
            if node.type in FOLDER_TYPES:
                assert not isinstance(node.start, mui.Undefined)
                assert not isinstance(node.realId, mui.Undefined)
                if node._is_divisible(self.limit):
                    node.children = node._get_divided_tree(self.limit, node.start)
                    upd = self.tree.update_event(tree=self._objinspect_root)
                    return await self.tree.send_and_wait(upd)
                # real_node = self._objinspect_root._get_node_by_uid(node.realId)
                real_obj, found = await self._get_obj_by_uid(node.realId, nodes)
                if node.type == mui.JsonLikeType.ListFolder.value:
                    data = real_obj[node.start:node.start + node.cnt]
                else:
                    assert not isinstance(node.keys, mui.Undefined)
                    data = {k: real_obj[k] for k in node.keys.data}
                obj_dict = {**(await self._tree_parser.expand_object(data))}
                tree = await self._tree_parser.parse_obj_dict_to_nodes(obj_dict, node.id)
                node.children = tree
                upd = self.tree.update_event(tree=self._objinspect_root)
                return await self.tree.send_and_wait(upd)
            obj, found = await self._get_obj_by_uid(uid, nodes)
            if node.type in CONTAINER_TYPES and node._is_divisible(self.limit):
                if node.type == mui.JsonLikeType.Dict.value:
                    node.keys = mui.BackendOnlyProp(list(obj.keys()))
                node.children = node._get_divided_tree(self.limit, 0)
                upd = self.tree.update_event(tree=self._objinspect_root)
                return await self.tree.send_and_wait(upd)
            # if not found, we expand (update) the deepest valid object.
            # if the object is special (extend TreeItem), we use used-defined
            # function instead of analysis it.
            if isinstance(obj, TreeItem):
                obj_dict_desp = await obj.get_child_desps()  
                for k, v in obj_dict_desp.items():
                    # v.id = f"{uid}{GLOBAL_SPLIT}{v.id}"
                    v.id = uid + v.id
                obj_dict = {**obj_dict_desp}
                tree = list(obj_dict_desp.values())
            else:
                obj_dict = {**(await self._tree_parser.expand_object(obj))}
                tree = await self._tree_parser.parse_obj_dict_to_nodes(obj_dict, node.id)
            node.children = tree
            upd = self.tree.update_event(tree=self._objinspect_root)
            return await self.tree.send_and_wait(upd)

    async def _on_rename(self, uid_newname: Tuple[str, str]):
        with enter_tree_conetxt(TreeContext(self._tree_parser, self.tree, self)):

            uid = uid_newname[0]
            uid_obj = UniqueTreeIdForTree(uid)
            uid_parts = uid_obj.parts

            obj_trace, found = await self._tree_parser.get_obj_by_uid_trace(self.root, uid_obj)
            if not found:
                return
            # if object is TreeItem or parent is TreeItem,
            # the button/contextmenu event will be handled in TreeItem instead of common
            # handler.
            if len(obj_trace) >= 2:
                parent = obj_trace[-2]
                if isinstance(parent, TreeItem):
                    nodes = self._objinspect_root._get_node_by_uid_trace(uid_obj.parts)
                    parent_node = nodes[-2]
                    res = await parent.handle_child_rename(uid_parts[-1],
                                                        uid_newname[1])
                    if res == True:
                        await self._on_expand(parent_node.id.uid_encoded)
                    return

    async def _on_custom_button(self, uid_btn: Tuple[str, str]):
        with enter_tree_conetxt(TreeContext(self._tree_parser, self.tree, self)):
            uid = uid_btn[0]
            uid_obj = UniqueTreeIdForTree(uid)
            uid_parts = uid_obj.parts
            obj_trace, found = await self._tree_parser.get_obj_by_uid_trace(self.root, uid_obj)
            if not found:
                return

            obj = obj_trace[-1]
            # if object is TreeItem or parent is TreeItem,
            # the button/contextmenu event will be handled in TreeItem instead of common
            # handler.
            if isinstance(obj, TreeItem):
                res = await obj.handle_button(uid_btn[1])
                if res == True:
                    # update this node
                    await self._on_expand(uid)
                return
            nodes = self._objinspect_root._get_node_by_uid_trace(uid_obj.parts)
            if len(obj_trace) >= 2:
                parent = obj_trace[-2]
                if isinstance(parent, TreeItem):
                    parent_node = nodes[-2]
                    res = await parent.handle_child_button(uid_btn[1],
                                                        uid_parts[-1])
                    if res == True:
                        await self._on_expand(parent_node.id.uid_encoded)
                    return

            if uid_btn[1] == ButtonType.Reload.value:
                metas, is_reload = reload_object_methods(
                    obj, reload_mgr=self.flow_app_comp_core.reload_mgr)
                if metas is not None:
                    special_methods = FlowSpecialMethods(metas)
                else:
                    print("reload failed.")
            if self._tree_parser.custom_tree_item_handler is not None:
                await self._tree_parser.custom_tree_item_handler.handle_button(obj_trace, nodes, uid_btn[1])

    async def _on_contextmenu(self, uid_menuid_data: Tuple[str, str,
                                                           Optional[Any]]):
        with enter_tree_conetxt(TreeContext(self._tree_parser, self.tree, self)):

            uid = uid_menuid_data[0]
            uid_obj = UniqueTreeIdForTree(uid)
            uid_parts = uid_obj.parts
            userdata = uid_menuid_data[2]
            if userdata is not None:
                obj_trace, found = await self._tree_parser.get_obj_by_uid_trace(
                    self.root, uid_obj)
                if found:
                    obj = obj_trace[-1]

                    # handle tree item first
                    if isinstance(obj, TreeItem):
                        res = await obj.handle_context_menu(userdata)
                        if res == True:
                            # update this node
                            await self._on_expand(uid)
                        return
                    nodes = self._objinspect_root._get_node_by_uid_trace(uid_obj.parts)
                    if len(obj_trace) >= 2:
                        parent = obj_trace[-2]
                        parent_node = nodes[-2]
                        if isinstance(parent, TreeItem):
                            res = await parent.handle_child_context_menu(
                                uid_parts[-1], userdata)
                            if res == True:
                                # update this node
                                await self._on_expand(parent_node.id.uid_encoded)
                            return
                    if self._tree_parser.custom_tree_item_handler is not None:
                        await self._tree_parser.custom_tree_item_handler.handle_button(obj_trace, nodes, userdata)

    def has_object(self, key: str):
        return key in self.root

    async def set_object(self, obj, key: str = _DEFAULT_OBJ_NAME, expand_level: int = 1):
        key_in_root = key in self.root
        self.root[key] = obj
        with enter_tree_conetxt(TreeContext(self._tree_parser, self.tree, self)):
            obj_tree = await self._tree_parser.get_root_tree(obj, key, expand_level, ns=self.tree.props.tree.id)
            await self._tree_parser.parse_obj_to_tree(obj, obj_tree, expand_level)
        # obj_tree = await _get_obj_tree(obj, self._checker, key,
        #                          self.tree.props.tree.id, self._obj_meta_cache,
        #                          self._cached_lazy_expand_uids,
        #                          total_expand_level=expand_level)
        if key_in_root:
            for i, node in enumerate(self.tree.props.tree.children):
                if node.name == key:
                    self.tree.props.tree.children[i] = obj_tree
                    break
        else:
            self.tree.props.tree.children.append(obj_tree)
        # self.tree.props.tree = _get_root_tree(self.root, self._valid_checker,
        #                                       _ROOT, self._obj_meta_cache)
        await self.tree.send_and_wait(
            self.tree.update_event(tree=self.tree.props.tree))
        await self._do_when_tree_updated(obj_tree.id)

    async def update_tree(self, wait: bool = True, update_tree: bool = True, update_iff_change: bool = False):
        t = time.time()
        with enter_tree_conetxt(TreeContext(self._tree_parser, self.tree, self)):
            new_tree = await self._tree_parser.get_root_tree(self.root, _ROOT, self.default_expand_level)
        # print(0, time.time() - t)
        if update_tree:
            if update_iff_change:
                # send tree to frontend is greatly slower than compare.
                if new_tree != self.tree.props.tree:
                    await self.tree.send_and_wait(
                        self.tree.update_event(tree=new_tree), wait=wait)
            else:
                await self.tree.send_and_wait(
                    self.tree.update_event(tree=new_tree), wait=wait)
            self.tree.props.tree = new_tree
        # print(1, time.time() - t)

        await self._do_when_tree_updated(self.tree.props.tree.id)
        # print(2, time.time() - t)

    async def update_tree_event(self):
        with enter_tree_conetxt(TreeContext(self._tree_parser, self.tree, self)):
            self.tree.props.tree = await self._tree_parser.get_root_tree(self.root, _ROOT, self.default_expand_level)
        return self.tree.update_event(tree=self.tree.props.tree)

    async def remove_object(self, key: str):
        key_in_root = key in self.root
        if not key_in_root:
            return
        new_child = []
        for i, node in enumerate(self.tree.props.tree.children):
            if node.name != key:
                new_child.append(node)
        self.tree.props.tree.children = new_child
        await self.tree.send_and_wait(
            self.tree.update_event(tree=self.tree.props.tree))

    async def _get_obj_by_uid_with_folder(self, uid_may_obj: Union[str, UniqueTreeIdForTree], nodes: List[mui.JsonLikeNode]):
        if isinstance(uid_may_obj, str):
            uid_obj = UniqueTreeIdForTree(uid_may_obj)
            uid = uid_may_obj
        else:
            uid_obj = uid_may_obj
            uid = uid_may_obj.uid_encoded

        node = nodes[-1]
        if len(nodes) > 1:
            folder_node = nodes[-2]
            # print(folder_node)

            if folder_node.type in FOLDER_TYPES:
                assert isinstance(folder_node.realId, UniqueTreeIdForTree)
                assert isinstance(folder_node.start, int)
                real_nodes = self._objinspect_root._get_node_by_uid_trace(
                    folder_node.realId.parts)
                real_obj, found = await self._get_obj_by_uid(
                    folder_node.realId, real_nodes)
                obj = None
                if found:
                    found = True
                    if nodes[-2].type == mui.JsonLikeType.ListFolder.value:
                        slice_idx = int(node.name)
                        real_slice = folder_node.start + slice_idx
                        obj = real_obj[real_slice]
                    else:
                        # dict folder
                        assert isinstance(folder_node.keys,
                                          mui.BackendOnlyProp)
                        key = node.name
                        if not isinstance(node.get_dict_key(), mui.Undefined):
                            key = node.get_dict_key()
                        obj = real_obj[key]
            else:
                obj, found = await self._get_obj_by_uid(uid_obj, nodes)
        else:
            obj, found = await self._get_obj_by_uid(uid_obj, nodes)
        return obj, found

    async def _get_obj_by_uid_with_folder_trace(self, uid_may_obj: Union[str, UniqueTreeIdForTree], nodes: List[mui.JsonLikeNode]):
        # TODO merge this with _get_obj_by_uid_with_folder
        node = nodes[-1]
        if isinstance(uid_may_obj, str):
            uid_obj = UniqueTreeIdForTree(uid_may_obj)
            uid = uid_may_obj
        else:
            uid_obj = uid_may_obj
            uid = uid_may_obj.uid_encoded

        if len(nodes) > 1:
            folder_node = nodes[-2]
            # print(folder_node)

            if folder_node.type in FOLDER_TYPES:
                assert isinstance(folder_node.realId, UniqueTreeIdForTree)
                assert isinstance(folder_node.start, int)
                real_nodes = self._objinspect_root._get_node_by_uid_trace(
                    folder_node.realId.parts)
                real_objs, found = await self._get_obj_by_uid_trace(
                    folder_node.realId, real_nodes)
                real_obj = real_objs[-1]
                obj = None
                obj_trace = []
                if found:
                    obj_trace = real_objs.copy()
                    found = True
                    if nodes[-2].type == mui.JsonLikeType.ListFolder.value:
                        slice_idx = int(node.name)
                        real_slice = folder_node.start + slice_idx
                        obj = real_obj[real_slice]
                    else:
                        # dict folder
                        assert isinstance(folder_node.keys,
                                          mui.BackendOnlyProp)
                        key = node.name
                        if not isinstance(node.get_dict_key(), mui.Undefined):
                            key = node.get_dict_key()
                        obj = real_obj[key]
                    obj_trace.append(obj)
            else:
                obj_trace, found = await self._get_obj_by_uid_trace(uid_obj, nodes)
        else:
            obj_trace, found = await self._get_obj_by_uid(uid_obj, nodes)
        return obj_trace, found

    async def get_object_by_uid(self, uid_may_obj: Union[str, UniqueTreeIdForTree]):
        if isinstance(uid_may_obj, str):
            uid_obj = UniqueTreeIdForTree(uid_may_obj)
            uid = uid_may_obj
        else:
            uid_obj = uid_may_obj
            uid = uid_may_obj.uid_encoded

        nodes = self.tree.props.tree._get_node_by_uid_trace(uid_obj.parts)
        node = nodes[-1]
        if node.type in FOLDER_TYPES:
            return None
        if len(nodes) > 1:
            folder_node = nodes[-2]
            if folder_node.type in FOLDER_TYPES:
                assert isinstance(folder_node.realId, UniqueTreeIdForTree)
                assert isinstance(folder_node.start, int)
                real_nodes = self.tree.props.tree._get_node_by_uid_trace(
                    folder_node.realId.parts)
                real_obj, found = await self._get_obj_by_uid(
                    folder_node.realId, real_nodes)
                obj = None
                if found:
                    slice_idx = int(node.name)
                    real_slice = folder_node.start + slice_idx
                    found = True
                    if nodes[-2].type == mui.JsonLikeType.ListFolder.value:
                        obj = real_obj[real_slice]
                    else:
                        # dict folder
                        assert isinstance(folder_node.keys,
                                          mui.BackendOnlyProp)
                        key = node.name
                        if not isinstance(node.get_dict_key(), mui.Undefined):
                            key = node.get_dict_key()
                        obj = real_obj[key]
            else:
                obj, found = await self._get_obj_by_uid(uid_obj, nodes)
        else:
            obj, found = await self._get_obj_by_uid(uid_obj, nodes)
        if not found:
            raise ValueError(
                f"your object {uid} is invalid, may need to reflesh")
        return obj

class ObjectTree(BasicObjectTree):
    """object tree for object inspector.
    """

    def __init__(self,
                 init: Optional[Any] = None,
                 cared_types: Optional[Set[Type]] = None,
                 ignored_types: Optional[Set[Type]] = None,
                 use_fast_tree: bool = False,
                 limit: int = 50,
                 fixed_size: bool = False,
                 custom_tree_handler: Optional[CustomTreeItemHandler] = None) -> None:
        self._default_data_storage_nodes: Dict[str, DataStorageTreeItem] = {}
        self._default_obs_funcs = ObservedFunctionTree()
        default_builtins = {
            _DEFAULT_BUILTINS_NAME: {
                "script": ScriptManager(),
                "monitor": ComputeResourceMonitor(),
                "appTerminal": mui.AppTerminal(),
                "simpleCanvas": SimpleCanvasCreator(),
                "scriptRunner": ScriptExecutor(),
                "fileReader": SimpleFileReader(),
                "callbackSlider": CallbackSlider(),
                "threadLocker": ThreadLocker(),
                "markdown": MarkdownViewer(),
            },
            _DEFAULT_DATA_STORAGE_NAME: self._default_data_storage_nodes,
            _DEFAULT_OBSERVED_FUNC_NAME: self._default_obs_funcs,
        }
        self._data_storage_uid = UniqueTreeIdForTree.from_parts([_ROOT, _DEFAULT_DATA_STORAGE_NAME])
        super().__init__(init, cared_types, ignored_types, limit, use_fast_tree, fixed_size=fixed_size, custom_tree_handler=custom_tree_handler)
        self.root.update(default_builtins)

    @mark_did_mount
    async def _on_mount(self):
        all_data_nodes = await appctx.list_all_data_storage_nodes()
        userdata = {
            "type": ContextMenuType.CopyReadItemCode.value,
        }

        context_menus: List[mui.ContextMenuData] = [
            mui.ContextMenuData(f"Copy Load Code",
                                id="Copy Load Code",
                                userdata=userdata)
        ]
        for n, readable_n in all_data_nodes:
            userdata = {
                "type": ContextMenuType.DataStorageStore.value,
                "node_id": n,
            }
            context_menus.append(
                mui.ContextMenuData(f"Add To {readable_n}",
                                    id=n,
                                    icon=mui.IconType.DataObject.value,
                                    userdata=userdata))

            self._default_data_storage_nodes[readable_n] = DataStorageTreeItem(
                n, readable_n)
        # self.tree.props.tree = await _get_root_tree_async(
        #     self.root, self._valid_checker, _ROOT, self._obj_meta_cache)
        with enter_tree_conetxt(TreeContext(self._tree_parser, self.tree, self)):
            self.tree.props.tree = await self._tree_parser.get_root_tree(self.root, _ROOT, self.default_expand_level)
        if context_menus:
            await self.tree.send_and_wait(
                self.tree.update_event(tree=self.tree.props.tree,
                                       contextMenus=context_menus))
        else:
            await self.tree.send_and_wait(
                self.tree.update_event(tree=self.tree.props.tree))
        appctx.get_app().register_app_special_event_handler(
            AppSpecialEventType.ObservedFunctionChange,
            self._on_obs_func_change)

    @mark_will_unmount
    def _on_unmount(self):
        appctx.get_app().unregister_app_special_event_handler(
            AppSpecialEventType.ObservedFunctionChange,
            self._on_obs_func_change)

    async def _on_obs_func_change(self, changed_qualnames: List[str]):
        rg = appctx.get_app().get_observed_func_registry()
        for qualname in changed_qualnames:
            entry = rg[qualname]
            rg.invalid_record(entry)
            if qualname in rg and qualname in self._default_obs_funcs._watched_funcs:
                if entry.recorded_data is not None:
                    await self.run_callback(entry.run_function_with_record,
                                            sync_status_first=False,
                                            change_status=False)

    async def _sync_data_storage_node(self):
        await self._on_expand(self._data_storage_uid.uid_encoded)


    async def _on_contextmenu(self, uid_menuid_data: Tuple[str, str,
                                                           Optional[Any]]):
        uid = uid_menuid_data[0]
        uid_obj = UniqueTreeIdForTree(uid)
        uid_parts = uid_obj.parts
        userdata = uid_menuid_data[2]
        if userdata is not None:
            nodes = self._objinspect_root._get_node_by_uid_trace(uid_obj.parts)
            obj_trace, found = await self._get_obj_by_uid_with_folder_trace(
                uid, nodes)
            if found:
                obj = obj_trace[-1]
                # handle tree item first
                if isinstance(obj, TreeItem):
                    res = await obj.handle_context_menu(userdata)
                    if res == True:
                        # update this node
                        await self._on_expand(uid)
                    if res is not None: # handle this event
                        return
                if len(obj_trace) >= 2:
                    parent = obj_trace[-2]
                    nodes = self._objinspect_root._get_node_by_uid_trace(uid_obj.parts)
                    parent_node = nodes[-2]

                    if isinstance(parent, TreeItem):
                        res = await parent.handle_child_context_menu(
                            uid_parts[-1], userdata)
                        if res == True:
                            # update this node
                            await self._on_expand(parent_node.id.uid_encoded)
                        if res is not None: # handle this event
                            return
                # handle regular objects
                menu_type = ContextMenuType(userdata["type"])
                if menu_type == ContextMenuType.DataStorageStore:
                    node_id = userdata["node_id"]
                    if not found:
                        return
                    await appctx.save_data_storage(uid_parts[-1], obj, node_id)
                    await self._sync_data_storage_node()
                if menu_type == ContextMenuType.CopyReadItemCode:
                    await appctx.get_app().copy_text_to_clipboard(f"await appctx.inspector.read_item('{uid}')")

