import asyncio
import contextlib
import dataclasses
import enum
import importlib
import inspect
from pathlib import Path
import sys
import threading
import traceback
import types
from functools import partial
from typing import (Any, Callable, Dict, Hashable, Iterable, List, Optional,
                    Set, Tuple, Type, TypeVar, Union)

import numpy as np
from typing_extensions import ParamSpec

from tensorpc.core.inspecttools import get_members
from tensorpc.core.moduleid import get_qualname_of_type
from tensorpc.core.serviceunit import AppFuncType, ReloadableDynamicClass, ServFunctionMeta
from tensorpc.core.tracer import FrameResult, TraceType, Tracer
from tensorpc.flow.flowapp.appcore import Event, get_app
from tensorpc.flow.flowapp.components import mui, three
from tensorpc.flow.flowapp.components.plus.objinspect.treeitems import TraceTreeItem, parse_frame_result_to_trace_item
from tensorpc.flow.flowapp.core import FlowSpecialMethods, FrontendEventType, _get_obj_def_path
from tensorpc.flow.flowapp.objtree import UserObjTreeProtocol

from .core import (ALL_OBJECT_PREVIEW_HANDLERS, USER_OBJ_TREE_TYPES,
                   ObjectPreviewHandler, DataClassesType)
from .tree import _DEFAULT_OBJ_NAME, FOLDER_TYPES, ObjectTree
from tensorpc.core import inspecttools

_DEFAULT_LOCALS_NAME = "locals"

_MAX_STRING_IN_DETAIL = 10000
P = ParamSpec('P')

T = TypeVar('T')


def _parse_trace_modules(traced_locs: List[Union[str, Path, types.ModuleType]]):
    traced_folders: Set[str] = set()
    for m in traced_locs:
        if isinstance(m, (str, Path)):
            folder = Path(m)
            assert folder.exists(), f"{folder} must exists"
            traced_folders.add(str(folder))
        else:
            mod = m
            if mod.__file__ is not None:
                mod_file = Path(mod.__file__).parent.resolve()
                traced_folders.add(str(mod_file))
    return traced_folders


def get_exception_frame_stack() -> Dict[str, TraceTreeItem]:
    _, _, exc_traceback = sys.exc_info()
    frame_stacks: Dict[str, TraceTreeItem] = {}
    for tb_frame, tb_lineno in traceback.walk_tb(exc_traceback):
        fr = Tracer.get_frame_result(TraceType.Return, tb_frame)
        frame_stacks[fr.get_unique_id()] = TraceTreeItem(fr)
    return frame_stacks


class DefaultHandler(ObjectPreviewHandler):
    """
    TODO if the object support any-layout, add a button to enable it.
    """

    def __init__(self) -> None:
        self.tags = mui.FlexBox().prop(flexFlow="row wrap")
        self.title = mui.Typography("").prop(wordBreak="break-word")
        self.path = mui.Typography("").prop(wordBreak="break-word")

        self.data_print = mui.Typography("").prop(fontFamily="monospace",
                                                  fontSize="12px",
                                                  wordBreak="break-word")
        layout = [
            self.title.prop(fontSize="14px", fontFamily="monospace"),
            self.path.prop(fontSize="14px", fontFamily="monospace"),
            self.tags,
            mui.Divider().prop(padding="3px"),
            mui.HBox([
                mui.Button("print", self._on_print),
            ]),
            self.data_print,
        ]

        super().__init__(layout)
        self.prop(flexDirection="column")
        self.obj: Any = np.zeros([1])

    async def _on_print(self):
        string = str(self.obj)
        if len(string) > _MAX_STRING_IN_DETAIL:
            string = string[:_MAX_STRING_IN_DETAIL] + "..."
        await self.data_print.write(string)

    async def bind(self, obj: Any, uid: str):
        # bind np object, update all metadata
        self.obj = obj
        ev = self.data_print.update_event(value="")
        ev += self.title.update_event(value=get_qualname_of_type(type(obj)))
        try:
            sf = inspect.getsourcefile(type(obj))
        except TypeError:
            sf = None
        if sf is None:
            sf = ""
        ev += self.path.update_event(value=sf)
        await self.send_and_wait(ev)
        # await self.tags.set_new_layout([*tags])


class ObjectInspector(mui.FlexBox):

    def __init__(self,
                 init: Optional[Any] = None,
                 cared_types: Optional[Set[Type]] = None,
                 ignored_types: Optional[Set[Type]] = None,
                 with_detail: bool = True,
                 use_allotment: bool = True,
                 enable_exception_inspect: bool = True,
                 use_fast_tree: bool = False) -> None:
        self.detail_container = mui.HBox([]).prop(overflow="auto",
                                                  padding="3px")
        if use_allotment:
            self.detail_container.prop(height="100%")
        else:
            self.detail_container.prop(flex=1)
        self.enable_exception_inspect = enable_exception_inspect
        self.with_detail = with_detail
        self.tree = ObjectTree(init,
                               cared_types,
                               ignored_types,
                               use_fast_tree=use_fast_tree)
        if use_allotment:
            layout: mui.LayoutType = [
                self.tree.prop(
                    overflow="auto",
                    height="100%",
                )
            ]
        else:
            layout: mui.LayoutType = [self.tree.prop(flex=1)]

        if with_detail:
            if not use_allotment:
                layout.append(mui.Divider())
            layout.append(self.detail_container)
        self.default_handler = DefaultHandler()
        final_layout: mui.LayoutType = layout
        if use_allotment:
            final_layout = [
                mui.Allotment(final_layout).prop(
                    overflow="hidden",
                    defaultSizes=[1.5, 1] if with_detail else [1],
                    vertical=True)
            ]
        super().__init__(final_layout)
        self.prop(flexDirection="column",
                  flex=1,
                  overflow="hidden",
                  minHeight=0,
                  minWidth=0)

        if with_detail:
            self.tree.tree.register_event_handler(
                FrontendEventType.TreeItemSelect.value, self._on_select)
        self._type_to_handler_object: Dict[Type[Any],
                                           ObjectPreviewHandler] = {}
        self._current_preview_layout: Optional[mui.FlexBox] = None


    async def get_object_by_uid(self, uid_list: Union[List[str], str]):
        if isinstance(uid_list, list):
            # node id list may empty (TODO don't send event in frontend?)
            if not uid_list:
                return None
            uid = uid_list[0]
        else:
            uid = uid_list
        nodes = self.tree._objinspect_root._get_node_by_uid_trace(uid)
        node = nodes[-1]
        if node.type in FOLDER_TYPES:
            return None
        if len(nodes) > 1:
            folder_node = nodes[-2]
            if folder_node.type in FOLDER_TYPES:
                assert isinstance(folder_node.realId, str)
                assert isinstance(folder_node.start, int)
                real_nodes = self.tree._objinspect_root._get_node_by_uid_trace(
                    folder_node.realId)
                real_obj, found = await self.tree._get_obj_by_uid(
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
                obj, found = await self.tree._get_obj_by_uid(uid, nodes)
        else:
            obj, found = await self.tree._get_obj_by_uid(uid, nodes)
        if not found:
            raise ValueError(
                f"your object {uid} is invalid, may need to reflesh")
        return obj

    async def _on_select(self, uid_list: Union[List[str], str]):
        if isinstance(uid_list, list):
            # node id list may empty
            if not uid_list:
                return
            uid = uid_list[0]
        else:
            uid = uid_list
        nodes = self.tree._objinspect_root._get_node_by_uid_trace(uid)
        node = nodes[-1]
        if node.type in FOLDER_TYPES:
            await self.detail_container.set_new_layout([])
            return
        obj, found = await self.tree._get_obj_by_uid_with_folder(uid, nodes)
        if not found:
            raise ValueError(
                f"your object {uid} is invalid, may need to reflesh")
        obj_type: Type = type(obj)
        is_dcls = dataclasses.is_dataclass(obj)

        preview_layout: Optional[mui.FlexBox] = None

        objs, found = await self.tree._get_obj_by_uid_trace(uid, nodes)
        # determine objtree root
        # we don't require your tree is strictly nested,
        # you can have a tree with non-tree-item container,
        # e.g. treeitem-anyobject-treeitem
        assert found, f"shouldn't happen, {uid}"
        root: Optional[UserObjTreeProtocol] = None
        for obj_iter_val in objs:
            if isinstance(obj_iter_val, tuple(USER_OBJ_TREE_TYPES)):
                root = obj_iter_val
                break

        # preview layout is checked firstly, then preview handler.
        if obj_type in self._type_to_handler_object:
            handler = self._type_to_handler_object[obj_type]
        elif is_dcls and DataClassesType in self._type_to_handler_object:
            handler = self._type_to_handler_object[DataClassesType]
        else:
            metas = self.flow_app_comp_core.reload_mgr.query_type_method_meta(
                obj_type, True)
            special_methods = FlowSpecialMethods(metas)
            if special_methods.create_preview_layout is not None:
                if root is None:
                    preview_layout = mui.flex_preview_wrapper(
                        obj, metas, self.flow_app_comp_core.reload_mgr)
                else:
                    with root.enter_context(root):
                        preview_layout = mui.flex_preview_wrapper(
                            obj, metas, self.flow_app_comp_core.reload_mgr)
                handler = self.default_handler
            else:
                obj_qualname = get_qualname_of_type(type(obj))
                handler_type: Optional[Type[ObjectPreviewHandler]] = None
                modified_obj_type = obj_type
                if obj is not None:
                    # check standard type first, if not found, check datasetclass type.
                    if obj_type in ALL_OBJECT_PREVIEW_HANDLERS:
                        handler_type = ALL_OBJECT_PREVIEW_HANDLERS[obj_type]
                    elif obj_qualname in ALL_OBJECT_PREVIEW_HANDLERS:
                        handler_type = ALL_OBJECT_PREVIEW_HANDLERS[obj_qualname]
                    elif is_dcls and DataClassesType in ALL_OBJECT_PREVIEW_HANDLERS:
                        modified_obj_type = DataClassesType
                        handler_type = ALL_OBJECT_PREVIEW_HANDLERS[DataClassesType]
                if handler_type is not None:
                    handler = handler_type()
                else:
                    handler = self.default_handler
                self._type_to_handler_object[modified_obj_type] = handler
            # if preview_layout is None:
            #     self._type_to_handler_object[modified_obj_type] = handler
        if preview_layout is not None:
            if root is not None:
                preview_layout.set_flow_event_context_creator(
                    lambda: root.enter_context(root))
            # preview_layout.event_emitter.remove_listener()
            if self._current_preview_layout is None:
                get_app()._get_self_as_editable_app()._flowapp_observe(
                        preview_layout, self._on_preview_layout_reload)
            else:
                get_app()._get_self_as_editable_app()._flowapp_remove_observer(
                        self._current_preview_layout)
                get_app()._get_self_as_editable_app()._flowapp_observe(
                        preview_layout, self._on_preview_layout_reload)
            self._current_preview_layout = preview_layout
            # self.__install_preview_event_listeners(preview_layout)
            await self.detail_container.set_new_layout([preview_layout])
        else:
            childs = list(self.detail_container._child_comps.values())
            if not childs or childs[0] is not handler:
                await self.detail_container.set_new_layout([handler])
            await handler.bind(obj, uid)

    # def __install_preview_event_listeners(self, layout: mui.FlexBox):
    #     # if not layout.event_emitter.listeners(
    #     #         FrontendEventType.BeforeUnmount.name):
    #     layout.flow_event_emitter.once(
    #         FrontendEventType.BeforeUnmount.value, partial(self._on_preview_layout_unmount, layout=layout))
    #     # if not layout.event_emitter.listeners(
    #     #         FrontendEventType.BeforeMount.name):
    #     layout.flow_event_emitter.once(
    #         FrontendEventType.BeforeMount.value, partial(self._on_preview_layout_mount, layout=layout))


    # def _on_preview_layout_mount(self, event: Event, layout: mui.FlexBox):
    #     # print("preview layout mount")
    #     return get_app()._get_self_as_editable_app()._flowapp_observe(
    #                     layout, self._on_preview_layout_reload)
    
    # def _on_preview_layout_unmount(self, event: Event, layout: mui.FlexBox):
    #     # print("preview layout unmount")

    #     return get_app()._get_self_as_editable_app()._flowapp_remove_observer(
    #                     layout)

    async def _on_preview_layout_reload(self, layout: mui.FlexBox,
                                        create_layout: ServFunctionMeta):
        # print("DO PREVIEW LAYOUT RELOAD", create_layout.user_app_meta)
        if create_layout.user_app_meta is not None and create_layout.user_app_meta.type == AppFuncType.CreatePreviewLayout:
            if layout._wrapped_obj is not None:
                # print("DO PREVIEW LAYOUT RELOAD 2")

                layout_flex = create_layout.get_binded_fn()()
                assert isinstance(
                    layout_flex, mui.FlexBox
                ), f"create_layout must return a flexbox when use anylayout"
                layout_flex._flow_comp_def_path = _get_obj_def_path(
                    layout._wrapped_obj)
                layout_flex._wrapped_obj = layout._wrapped_obj
                layout_flex.set_flow_event_context_creator(layout._flow_event_context_creator)
                # self.__install_preview_event_listeners(layout_flex)
                await self.detail_container.set_new_layout([layout_flex])
            else:
                layout_flex = create_layout.get_binded_fn()()
                layout_flex.set_flow_event_context_creator(layout._flow_event_context_creator)
                # self.__install_preview_event_listeners(layout_flex)
                await layout.set_new_layout(layout_flex)
            return layout_flex

    async def set_object(self, obj, key: str = _DEFAULT_OBJ_NAME, expand_level: int = 0):
        await self.tree.set_object(obj, key, expand_level=expand_level)


    async def update_locals(self,
                            key: str = _DEFAULT_LOCALS_NAME,
                            *,
                            _frame_cnt: int = 1,
                            exclude_self: bool = False):
        cur_frame = inspect.currentframe()
        assert cur_frame is not None
        frame = cur_frame
        while _frame_cnt > 0:
            frame = cur_frame.f_back
            assert frame is not None
            cur_frame = frame
            _frame_cnt -= 1
        # del frame
        local_vars = cur_frame.f_locals.copy()
        if exclude_self:
            local_vars.pop("self", None)
        frame_name = cur_frame.f_code.co_name
        del frame
        del cur_frame
        await self.tree.set_object(inspecttools.filter_local_vars(local_vars),
                                   key + f"-{frame_name}")

    def update_locals_sync(self,
                           key: str = _DEFAULT_LOCALS_NAME,
                           *,
                           _frame_cnt: int = 1,
                           loop: Optional[asyncio.AbstractEventLoop] = None,
                           exclude_self: bool = False):
        """update locals in sync manner, usually used on non-sync code via appctx.
        """
        if loop is None:
            loop = asyncio.get_running_loop()
        cur_frame = inspect.currentframe()
        assert cur_frame is not None
        frame = cur_frame
        while _frame_cnt > 0:
            frame = cur_frame.f_back
            assert frame is not None
            cur_frame = frame
            _frame_cnt -= 1
        # del frame
        local_vars = cur_frame.f_locals.copy()
        if exclude_self:
            local_vars.pop("self", None)
        frame_name = cur_frame.f_code.co_name
        del frame
        del cur_frame
        if get_app()._flowapp_thread_id == threading.get_ident():
            task = asyncio.create_task(self.tree.set_object(inspecttools.filter_local_vars(local_vars),
                                 key + f"-{frame_name}"))
            # we can't wait fut here
            return task
        else:
            # we can wait fut here.
            fut = asyncio.run_coroutine_threadsafe(
                self.tree.set_object(inspecttools.filter_local_vars(local_vars),
                                    key + f"-{frame_name}"), loop)
            return fut.result()

    def set_object_sync(self,
                        obj,
                        key: str = _DEFAULT_OBJ_NAME,
                        loop: Optional[asyncio.AbstractEventLoop] = None,
                        expand_level: int = 0):
        """set object in sync manner, usually used on non-sync code via appctx.
        """
        if loop is None:
            loop = asyncio.get_running_loop()
        if get_app()._flowapp_thread_id == threading.get_ident():
            # we can't wait fut here
            task = asyncio.create_task(self.set_object(obj, key, expand_level))
            # we can't wait fut here
            return task

            # return fut
        else:
            # we can wait fut here.
            fut = asyncio.run_coroutine_threadsafe(self.set_object(obj, key, expand_level), loop)

            return fut.result()

    async def update_tree(self):
        await self.tree.update_tree()

    async def remove_object(self, key: str):
        await self.tree.remove_object(key)

    def run_with_exception_inspect(self, func: Callable[P, T], *args: P.args,
                                   **kwargs: P.kwargs) -> T:
        """WARNING: we shouldn't run this function in run_in_executor.
        """
        loop = asyncio.get_running_loop()
        try:
            return func(*args, **kwargs)
        except:
            asyncio.run_coroutine_threadsafe(
                self.set_object(get_exception_frame_stack(), "exception"),
                loop)
            raise

    async def run_with_exception_inspect_async(self, func: Callable[P, T],
                                               *args: P.args,
                                               **kwargs: P.kwargs) -> T:
        try:
            res = func(*args, **kwargs)
            if inspect.iscoroutine(res):
                return await res
            else:
                return res
        except:
            await self.set_object(get_exception_frame_stack(), "exception")
            raise

    async def run_in_executor_with_exception_inspect(self, func: Callable[P,
                                                                          T],
                                                     *args: P.args,
                                                     **kwargs: P.kwargs) -> T:
        """run a sync function in executor with exception inspect.

        """
        loop = asyncio.get_running_loop()
        app = get_app()
        try:
            if kwargs:
                return await loop.run_in_executor(None,
                                                  partial(func, **kwargs), app,
                                                  func, *args)
            else:
                return await loop.run_in_executor(None, func, app, func, *args)
        except:
            await self.set_object(get_exception_frame_stack(), "exception")
            raise

    @contextlib.contextmanager
    def trace_sync(self,
                   traced_locs: List[Union[str, Path, types.ModuleType]],
                   key: str = "trace",
                   traced_types: Optional[Tuple[Type]] = None,
                   traced_names: Optional[Set[str]] = None,
                   traced_folders: Optional[Set[str]] = None,
                   trace_return: bool = True,
                   depth: int = 5,
                   use_return_locals: bool = False,
                   ignored_names: Optional[Set[str]] = None,
                   *,
                   _frame_cnt=3,
                   loop: Optional[asyncio.AbstractEventLoop] = None):
        if traced_folders is None:
            traced_folders = set()
        traced_folders.update(_parse_trace_modules(traced_locs))
        trace_res: List[FrameResult] = []
        if ignored_names is None:
            ignored_names = set([
                "_call_impl",  # torch nn forward
            ])
        tracer = Tracer(lambda x: trace_res.append(x),
                        traced_types,
                        traced_names,
                        traced_folders,
                        trace_return,
                        depth,
                        ignored_names,
                        _frame_cnt=_frame_cnt)
        try:
            with tracer:
                yield
        finally:
            tree_items = parse_frame_result_to_trace_item(
                trace_res, use_return_locals)
            show_dict = {v.get_uid(): v for v in tree_items}
            self.set_object_sync(show_dict, key, loop=loop)

    def trace_sync_return(self,
                          traced_locs: List[Union[str, Path, types.ModuleType]],
                          key: str = "trace",
                          traced_types: Optional[Tuple[Type]] = None,
                          traced_names: Optional[Set[str]] = None,
                          traced_folders: Optional[Set[str]] = None,
                          trace_return: bool = True,
                          depth: int = 5,
                          ignored_names: Optional[Set[str]] = None,
                          *,
                          _frame_cnt: int = 4,
                          loop: Optional[asyncio.AbstractEventLoop] = None):
        return self.trace_sync(traced_locs,
                               key,
                               traced_types,
                               traced_names,
                               traced_folders,
                               trace_return,
                               depth,
                               ignored_names=ignored_names,
                               use_return_locals=True,
                               _frame_cnt=_frame_cnt,
                               loop=loop)

    @contextlib.asynccontextmanager
    async def trace(self,
                    traced_locs: List[Union[str, Path, types.ModuleType]],
                    key: str = "trace",
                    traced_types: Optional[Tuple[Type]] = None,
                    traced_names: Optional[Set[str]] = None,
                    traced_folders: Optional[Set[str]] = None,
                    trace_return: bool = True,
                    depth: int = 5,
                    use_return_locals: bool = False,
                    ignored_names: Optional[Set[str]] = None,
                    *,
                    _frame_cnt: int = 3):
        if traced_folders is None:
            traced_folders = set()
        traced_folders.update(_parse_trace_modules(traced_locs))
        trace_res: List[FrameResult] = []
        tracer = Tracer(lambda x: trace_res.append(x),
                        traced_types,
                        traced_names,
                        traced_folders,
                        trace_return,
                        depth,
                        ignored_names,
                        _frame_cnt=_frame_cnt)
        try:
            with tracer:

                yield
        finally:
            tree_items = parse_frame_result_to_trace_item(
                trace_res, use_return_locals)
            show_dict = {v.get_uid(): v for v in tree_items}
            await self.set_object(show_dict, key)