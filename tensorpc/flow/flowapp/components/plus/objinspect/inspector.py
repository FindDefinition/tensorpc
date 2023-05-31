import asyncio
import contextlib
import enum
import inspect
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
from tensorpc.flow.flowapp.appcore import get_app
from tensorpc.flow.flowapp.components import mui, three
from tensorpc.flow.flowapp.components.plus.objinspect.treeitems import TraceTreeItem, parse_frame_result_to_trace_item
from tensorpc.flow.flowapp.core import FlowSpecialMethods, FrontendEventType, _get_obj_def_path
from tensorpc.flow.flowapp.objtree import UserObjTreeProtocol

from .core import (ALL_OBJECT_PREVIEW_HANDLERS, USER_OBJ_TREE_TYPES,
                   ObjectPreviewHandler)
from .tree import _DEFAULT_OBJ_NAME, FOLDER_TYPES, ObjectTree
from tensorpc.core import inspecttools

_DEFAULT_LOCALS_NAME = "locals"

_MAX_STRING_IN_DETAIL = 10000
P = ParamSpec('P')

T = TypeVar('T')


class TreeTracer(Tracer):
    def __init__(
            self,
            #  tree: "ObjectInspector",
            traced_types: Optional[Tuple[Type]] = None,
            traced_names: Optional[Set[str]] = None,
            traced_folders: Optional[Set[str]] = None,
            trace_return: bool = True,
            depth: int = -1):
        super().__init__(self.callback, traced_types, traced_names,
                         traced_folders, trace_return, depth)
        # self.tree = tree

        self._record_res: List[FrameResult] = []

    def callback(self, fr: FrameResult):
        print("????????????")
        self._record_res.append(fr)

    def __enter__(self):
        self._record_res.clear()
        print("ENTERENTER")
        return super().__enter__()

    # def __exit__(self, exc_type, exc_value, exc_traceback):

    #     return super().__exit__(exc_type, exc_value, exc_traceback)


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
        self.tags = mui.FlexBox().prop(flex_flow="row wrap")
        self.title = mui.Typography("").prop(word_break="break-word")
        self.path = mui.Typography("").prop(word_break="break-word")

        self.data_print = mui.Typography("").prop(font_family="monospace",
                                                  font_size="12px",
                                                  word_break="break-word")
        layout = [
            self.title.prop(font_size="14px", font_family="monospace"),
            self.path.prop(font_size="14px", font_family="monospace"),
            self.tags,
            mui.Divider().prop(padding="3px"),
            mui.HBox([
                mui.Button("print", self._on_print),
            ]),
            self.data_print,
        ]

        super().__init__(layout)
        self.prop(flex_direction="column")
        self.obj: Any = np.zeros([1])

    async def _on_print(self):
        string = str(self.obj)
        if len(string) > _MAX_STRING_IN_DETAIL:
            string = string[:_MAX_STRING_IN_DETAIL] + "..."
        await self.data_print.write(string)

    async def bind(self, obj):
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
            final_layout = [mui.Allotment(final_layout).prop(overflow="hidden",
                      default_sizes=[1.5, 1] if with_detail else [1],
                      vertical=True)]
        super().__init__(final_layout)
        self.prop(flex_direction="column",
                    flex=1,
                    overflow="hidden",
                    min_height=0,
                    min_width=0)

        if with_detail:
            self.tree.tree.register_event_handler(
                FrontendEventType.TreeItemSelect.value, self._on_select)
        self._type_to_handler_object: Dict[Type[Any],
                                           ObjectPreviewHandler] = {}

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
        obj_type = type(obj)
        preview_layout: Optional[mui.FlexBox] = None
        if obj_type in self._type_to_handler_object:
            handler = self._type_to_handler_object[obj_type]
        else:
            obj_qualname = get_qualname_of_type(type(obj))
            handler_type: Optional[Type[ObjectPreviewHandler]] = None
            if obj is not None:
                obj_type = type(obj)
                if obj_type in ALL_OBJECT_PREVIEW_HANDLERS:
                    handler_type = ALL_OBJECT_PREVIEW_HANDLERS[obj_type]
                elif obj_qualname in ALL_OBJECT_PREVIEW_HANDLERS:
                    handler_type = ALL_OBJECT_PREVIEW_HANDLERS[obj_qualname]
            if handler_type is not None:
                handler = handler_type()
            else:
                # check obj have create_preview_layout
                metas = self.flow_app_comp_core.reload_mgr.query_type_method_meta(
                    obj_type, True)
                special_methods = FlowSpecialMethods(metas)
                if special_methods.create_preview_layout is not None:
                    preview_layout = mui.flex_preview_wrapper(
                        obj, metas, self.flow_app_comp_core.reload_mgr)

                handler = self.default_handler
            if preview_layout is None:
                self._type_to_handler_object[obj_type] = handler
        if preview_layout is not None:
            objs, found = await self.tree._get_obj_by_uid_trace(uid, nodes)
            # determine objtree root
            assert found, f"shouldn't happen, {uid}"
            root: Optional[UserObjTreeProtocol] = None
            for obj_iter_val in objs:
                if isinstance(obj_iter_val, tuple(USER_OBJ_TREE_TYPES)):
                    root = obj_iter_val
                    break
            if root is not None:
                preview_layout.set_flow_event_context_creator(
                    lambda: root.enter_context(root))
            # preview_layout.event_emitter.remove_listener()
            if not preview_layout.event_emitter.listeners(
                    FrontendEventType.BeforeUnmount.name):
                preview_layout.event_emitter.on(
                    FrontendEventType.BeforeUnmount.name,
                    lambda: get_app()._get_self_as_editable_app().
                    _flowapp_remove_observer(preview_layout))
            if not preview_layout.event_emitter.listeners(
                    FrontendEventType.BeforeMount.name):
                preview_layout.event_emitter.on(
                    FrontendEventType.BeforeMount.name, lambda: get_app().
                    _get_self_as_editable_app()._flowapp_observe(
                        preview_layout, self._on_preview_layout_reload))
            await self.detail_container.set_new_layout([preview_layout])
        else:
            childs = list(self.detail_container._child_comps.values())
            if not childs or childs[0] is not handler:
                await self.detail_container.set_new_layout([handler])
            await handler.bind(obj)

    async def _on_preview_layout_reload(self, layout: mui.FlexBox,
                                        create_layout: ServFunctionMeta):
        if create_layout.user_app_meta is not None and create_layout.user_app_meta.type == AppFuncType.CreatePreviewLayout:
            if layout._wrapped_obj is not None:
                layout_flex = create_layout.get_binded_fn()()
                assert isinstance(
                    layout_flex, mui.FlexBox
                ), f"create_layout must return a flexbox when use anylayout"
                layout_flex._flow_comp_def_path = _get_obj_def_path(
                    layout._wrapped_obj)
                layout_flex._wrapped_obj = layout._wrapped_obj
                await self.detail_container.set_new_layout([layout_flex])
            else:
                layout_flex = create_layout.get_binded_fn()()
                await layout.set_new_layout(layout_flex)
            return layout_flex

    async def set_object(self, obj, key: str = _DEFAULT_OBJ_NAME):
        await self.tree.set_object(obj, key)

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
        fut = asyncio.run_coroutine_threadsafe(
            self.tree.set_object(inspecttools.filter_local_vars(local_vars),
                                 key + f"-{frame_name}"), loop)
        if get_app()._flowapp_thread_id == threading.get_ident():
            # we can't wait fut here
            return fut
        else:
            # we can wait fut here.
            return fut.result()

    def set_object_sync(self,
                        obj,
                        key: str = _DEFAULT_OBJ_NAME,
                        loop: Optional[asyncio.AbstractEventLoop] = None):
        """set object in sync manner, usually used on non-sync code via appctx.
        """
        if loop is None:
            loop = asyncio.get_running_loop()
        fut = asyncio.run_coroutine_threadsafe(self.set_object(obj, key), loop)
        if get_app()._flowapp_thread_id == threading.get_ident():
            # we can't wait fut here
            return fut
        else:
            # we can wait fut here.
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
        trace_res: List[FrameResult] = []
        if ignored_names is None:
            ignored_names = set([
                "_call_impl", # torch nn forward
            ])
        tracer = Tracer(lambda x: trace_res.append(x),
                        traced_types,
                        traced_names,
                        traced_folders,
                        trace_return,
                        depth,
                        ignored_names,
                        _frame_cnt=_frame_cnt)
        with tracer:
            yield
        tree_items = parse_frame_result_to_trace_item(trace_res,
                                                      use_return_locals)
        show_dict = {v.get_uid(): v for v in tree_items}
        self.set_object_sync(show_dict, key, loop=loop)

    def trace_sync_return(self,
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
        return self.trace_sync(key,
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
        trace_res: List[FrameResult] = []
        tracer = Tracer(lambda x: trace_res.append(x),
                        traced_types,
                        traced_names,
                        traced_folders,
                        trace_return,
                        depth,
                        ignored_names,
                        _frame_cnt=_frame_cnt)
        with tracer:
            yield
        tree_items = parse_frame_result_to_trace_item(trace_res,
                                                      use_return_locals)
        show_dict = {v.get_uid(): v for v in tree_items}
        await self.set_object(show_dict, key)