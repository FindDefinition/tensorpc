"""
VAPI Design
===========

with V.ctx():
    with V.group("key1.key2"):
        V.points(...)
        lines = V.lines()
        lines.add(...)
        lines.polygon(...).to(...).to(...)
        V.box(...).tdata({
            "score": ...,
            "label": ...,
        })
        V.text(...)

        def vprogram(x: V.Annotated[float, V.RangedFloat(0, 10)] = 5):
            V.point(x, 0, 0)

        V.program(vprogram, ctx_creator)

# send to browser when ctx exits


"""

import asyncio
import builtins
import dataclasses
from functools import partial
import inspect
import io
import threading
import traceback
from typing import IO, Any, AnyStr, Callable, Dict, Optional, List, Tuple, Type, TypeVar, Union, get_type_hints
from typing_extensions import Annotated, Literal
import contextvars
import contextlib
from tensorpc.core.dataclass_dispatch import dataclass
from tensorpc.core.tree_id import UniqueTreeId, UniqueTreeIdForTree
from tensorpc.flow.client import is_inside_app_session
from tensorpc.flow.flowapp import appctx
from tensorpc.flow.flowapp.appcore import get_app
from tensorpc.flow.flowapp.components.plus.config import ConfigPanelV2
from tensorpc.flow.flowapp.core import AppEvent
from tensorpc.utils.typeutils import take_annotation_from
from ... import three, mui
from ...typemetas import (ColorRGB, ColorRGBA, RangedFloat, RangedInt,
                          RangedVector3, Vector3,
                          annotated_function_to_dataclass)
from .canvas import ComplexCanvas, find_component_trace_by_uid_with_not_exist_parts
from .core import CanvasItemCfg, CanvasItemProxy, is_reserved_name, VContext, _VapiObjects
import numpy as np
from .core import get_canvas_item_cfg, get_or_create_canvas_item_cfg, ContainerProxy, GroupProxy



class Points(three.Points, _VapiObjects):
    def __init__(self, limit: int) -> None:
        super().__init__(limit)
        self._points: List[three.Vector3Type] = []
        self._points_arr: List[np.ndarray] = []
        self.prop(layers=31)
    
    @property
    def _count(self):
        return len(self._points) + sum([arr.shape[0] for arr in self._points_arr])

    def p(self, x: float, y: float, z: float):
        assert self._count + 1 <= self.props.limit, f"points count exceed limit {self.props.limit}"
        self._points.append((x, y, z))
        return self

    def array(self, data: np.ndarray):
        assert data.ndim == 2 and data.shape[1] in [3, 4], "points dim must be 3 or 4 (with intensity)"
        assert self._count + data.shape[0] <= self.props.limit, f"points count exceed limit {self.props.limit}"
        if self._points_arr:
            assert data.shape[1] == self._points_arr[0].shape[1], "points dim must be same as first array"
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        self._points_arr.append(data)
        return self
    
    def prepare_vapi_props(self):
        # TODO global config
        points_nparray = np.array(self._points, dtype=np.float32).reshape(-1, 3)
        if self._points_arr:
            has_intensity = self._points_arr[0].shape[1] == 4
            if has_intensity:
                points_nparray = np.concatenate([points_nparray, np.full((points_nparray.shape[0], 1), 255.0, dtype=np.float32)], axis=-1)
            points_nparray = np.concatenate(self._points_arr +
                                            [points_nparray])
        self.prop(points=points_nparray)

class ColoredPoints(three.Points, _VapiObjects):
    def __init__(self, limit: int) -> None:
        super().__init__(limit)
        self._points: List[three.Vector3Type] = []
        self._colors: List[Tuple[int, int, int]] = []

        self._points_arr: List[np.ndarray] = []
        self._colors_arr: List[np.ndarray] = []

    @property
    def _count(self):
        return len(self._points) + sum([arr.shape[0] for arr in self._points_arr])

    def p(self, x: float, y: float, z: float, r: int, g: int, b: int):
        assert r >= 0 and r <= 255
        assert g >= 0 and g <= 255
        assert b >= 0 and b <= 255
        assert self._count + 1 <= self.props.limit, f"points count exceed limit {self.props.limit}"
        self._points.append((x, y, z))
        self._colors.append((r, g, b))
        return self

    def array(self, data: np.ndarray, color: np.ndarray):
        assert color.dtype == np.uint8, "color must be uint8 array"
        assert data.ndim == 2 and data.shape[1] == 3, "points with color dim must be 3"
        assert color.ndim == 2 and color.shape[1] == 3, "points with color dim must be 3"
        assert self._count + data.shape[0] <= self.props.limit, f"points count exceed limit {self.props.limit}"
        if self._points_arr:
            assert data.shape[1] == self._points_arr[0].shape[1], "points dim must be same as first array"
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        self._points_arr.append(data)
        self._colors_arr.append(color)
        return self
    
    def prepare_vapi_props(self):
        # TODO global config
        points_nparray = np.array(self._points, dtype=np.float32).reshape(-1, 3)
        color_nparray = np.array(self._colors, dtype=np.uint8).reshape(-1, 3)
        if self._points_arr:
            points_nparray = np.concatenate(self._points_arr +
                                            [points_nparray])
            color_nparray = np.concatenate(self._colors_arr +
                                            [color_nparray])
        self.prop(points=points_nparray, colors=color_nparray)


class _Polygon:
    def __init__(self, start: three.Vector3Type, closed: bool,
                 line_proxy: "Lines") -> None:
        self.line_proxy = line_proxy
        self.closed = closed
        self.start = start

    def to(self, x: float, y: float, z: float):
        self.line_proxy.p(self.start[0], self.start[1], self.start[2], x, y, z)
        self.start = (x, y, z)
        return self

class Lines(three.Segments, _VapiObjects):
    def __init__(self, limit: int,
                 line_width: float = 1.0,
                 color: Union[str, mui.Undefined] = mui.undefined) -> None:
        super().__init__(limit, line_width, color)
        self._point_pairs: List[Tuple[three.Vector3Type,
                                      three.Vector3Type]] = []
        self._lines_arr: List[np.ndarray] = []
        self.prop(layers=31)

    @property
    def _count(self):
        return len(self._point_pairs) + sum([arr.shape[0] for arr in self._lines_arr])

    def p(self, x1: float, y1: float, z1: float, x2: float, y2: float,
          z2: float):
        assert self._count + 1 <= self.props.limit, f"lines count exceed limit {self.props.limit}"
        self._point_pairs.append(((x1, y1, z1), (x2, y2, z2)))
        return self

    def array(self, data: np.ndarray):
        assert self._count + data.shape[0] <= self.props.limit, f"lines count exceed limit {self.props.limit}"
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        three.SegmentsProps.lines_validator(data)
        self._lines_arr.append(data)
        return self

    def prepare_vapi_props(self):
        lines_array = np.array(self._point_pairs, dtype=np.float32).reshape(-1, 2, 3)
        if self._lines_arr:
            lines_array = np.concatenate(self._lines_arr + [lines_array])
        self.prop(lines=lines_array)

    def polygon(self, x: float, y: float, z: float, closed: bool = False):
        return _Polygon((x, y, z), closed, self)

    def closed_polygon(self, x: float, y: float, z: float):
        return _Polygon((x, y, z), True, self)

class BoundingBox(three.BoundingBox, _VapiObjects):
    def prepare_vapi_props(self):
        self.prop(enableSelect=True)

class Image(three.Image, _VapiObjects):
    def __init__(self, img: np.ndarray, use_datatex: bool = False) -> None:
        super().__init__()
        self._img = img
        self._use_datatex = use_datatex
        assert img.dtype == np.uint8

    def prepare_vapi_props(self):
        # TODO currently use texture loader (webimage) in frontend cause problem,
        # so we use data texture for now.
        use_datatex = self._use_datatex
        img = self._img
        if use_datatex:
            if img.ndim == 3 and img.shape[-1] == 3:
                img = np.concatenate([img, np.full((*img.shape[:-1], 1), 255, dtype=np.uint8)], axis=-1)
            elif img.ndim == 2:
                # gray to rgba
                img = img.reshape((*img.shape, 1))
                img = np.tile(img, (1, 1, 3))
                img = np.concatenate([img, np.full((*img.shape[:-1], 1), 255, dtype=np.uint8)], axis=-1)
            self.prop(image=img, enableSelect=True)
        else:
            self.prop(image=mui.Image.encode_image_bytes(img), enableSelect=True)


V_CONTEXT_VAR: contextvars.ContextVar[
    Optional[VContext]] = contextvars.ContextVar("v_context", default=None)

GROUP_CONTEXT_VAR: contextvars.ContextVar[
    Optional[GroupProxy]] = contextvars.ContextVar("group_context",
                                                   default=None)


def get_v_context() -> Optional[VContext]:
    return V_CONTEXT_VAR.get()


@contextlib.contextmanager
def enter_v_conetxt(robj: VContext):
    token = V_CONTEXT_VAR.set(robj)
    try:
        yield robj
    finally:
        V_CONTEXT_VAR.reset(token)


async def _draw_all_in_vctx(vctx: VContext,
                            detail_update_prefix: Optional[UniqueTreeId] = None,
                            app_event: Optional[AppEvent] = None,
                            update_iff_change: bool = False):
    # import rich 
    # rich.print("?", vctx._group_assigns)
    # rich.print(vctx._name_to_group)
    vctx.canvas._tree_collect_in_vctx()
    for k, v in vctx._name_to_group.items():
        cfg = get_canvas_item_cfg(v)
        # print(k, cfg )
        if cfg is not None:
            proxy = cfg.proxy
            if proxy is not None:
                assert isinstance(proxy, GroupProxy)
                for c in proxy.childs.values():
                    c_cfg = get_canvas_item_cfg(c)
                    assert c_cfg is not None
                    # c_proxy = c_cfg.proxy
                    if isinstance(c, _VapiObjects):
                        c.prepare_vapi_props()
                    # if c_proxy is not None:
                    #     c_proxy.update_event(c)
                if cfg.is_vapi and v is not vctx.root:
                    assert not v.is_mounted(), f"{type(v)}"
                    v.init_add_layout(proxy.childs)
                    # rich.print(k, cfg.is_vapi, proxy.childs)

                else:
                    await v.update_childs(proxy.childs)
                proxy.childs.clear()
    for container, (group, name) in vctx._group_assigns.items():
        assert isinstance(group, three.Group)
        # print(group)
        if container.is_mounted():
            await container.update_childs({name: group})
    await vctx.canvas._show_visible_groups_of_objtree()

    await vctx.canvas.item_tree.update_tree(wait=True, update_iff_change=update_iff_change)
    if detail_update_prefix is not None:
        await vctx.canvas.update_detail_layout(detail_update_prefix)
    if app_event is not None:
        await vctx.canvas.send_and_wait(app_event)


# @contextlib.contextmanager
# def ctx(
#     canvas: Optional[ComplexCanvas] = None,
#     loop: Optional[asyncio.AbstractEventLoop] = None,
# ):
#     if canvas is None:
#         canvas = appctx.find_component(ComplexCanvas)
#         assert canvas is not None, "you must add complex canvas before using vapi"

#     prev_ctx = get_v_context()
#     is_first_ctx = False
#     if prev_ctx is None:
#         is_first_ctx = True
#         prev_ctx = VContext(canvas)
#     token = V_CONTEXT_VAR.set(prev_ctx)
#     try:
#         yield prev_ctx
#     finally:
#         V_CONTEXT_VAR.reset(token)
#         if is_first_ctx:
#             if loop is None:
#                 loop = asyncio.get_running_loop()
#             if get_app()._flowapp_thread_id == threading.get_ident():
#                 # we can't wait fut here
#                 task = asyncio.create_task(_draw_all_in_vctx(prev_ctx))
#                 # we can't wait fut here
#                 return task
#                 # return fut
#             else:
#                 # we can wait fut here.
#                 fut = asyncio.run_coroutine_threadsafe(
#                     _draw_all_in_vctx(prev_ctx), loop)
#                 return fut.result()


def get_group_context() -> Optional[GroupProxy]:
    return GROUP_CONTEXT_VAR.get()


@contextlib.contextmanager
def enter_group_context(robj: GroupProxy):
    token = GROUP_CONTEXT_VAR.set(robj)
    try:
        yield robj
    finally:
        GROUP_CONTEXT_VAR.reset(token)


_CARED_CONTAINERS = (three.Group, three.Fragment, three.Hud)

T_container = TypeVar("T_container", bound=three.ContainerBase)
T_container_proxy = TypeVar("T_container_proxy", bound=three.ContainerBase)


@contextlib.contextmanager
def group(name: str,
          pos: Optional[three.Vector3Type] = None,
          rot: Optional[three.Vector3Type] = None,
          canvas: Optional[ComplexCanvas] = None,
          variant: Optional[Literal["default", "faceToCamera", "relativeToCamera"]] = None,
          loop: Optional[asyncio.AbstractEventLoop] = None):
    if canvas is None:
        canvas = appctx.find_component(ComplexCanvas)
        assert canvas is not None, "you must add complex canvas before using vapi"

    name_parts = name.split(".")
    name_obj = UniqueTreeIdForTree.from_parts(name_parts)
    for p in name_parts:
        assert p, "group name can not be empty"
    # find exist group in canvas
    v_ctx = get_v_context()
    is_first_ctx = False
    is_objtree_ctx = False
    token = None
    if v_ctx is None:
        is_first_ctx = True
    # we need to increase frame cnt due to contextmanager
    # obj_self = _find_frame_self(_frame_cnt=2 + 1)
    # if obj_self is not None:
    #     obj_self_id = id(obj_self)
    #     if obj_self_id in canvas._user_obj_tree_item_to_meta:
    #         v_ctx = canvas._user_obj_tree_item_to_meta[obj_self_id].vctx
    #         v_ctx.canvas = canvas
    #         is_objtree_ctx = True
    if v_ctx is None:
        assert not is_reserved_name(name), f"{name} should not be reserved name"
        v_ctx = VContext(canvas)
    #     token = V_CONTEXT_VAR.set(v_ctx)
    # else:
    if is_first_ctx:
        token = V_CONTEXT_VAR.set(v_ctx)
    # v_ctx = get_v_context()
    # assert v_ctx is not None
    # canvas = v_ctx.canvas
    if v_ctx.name_stack:
        # uid = f"{v_ctx.current_namespace}.{name}"
        uid = v_ctx.current_namespace.append_part(name)
    else:
        uid = name_obj
    if uid.uid_encoded in v_ctx._name_to_group:
        group = v_ctx._name_to_group[uid.uid_encoded]
    else:
        trace, remain, consumed = find_component_trace_by_uid_with_not_exist_parts(
            v_ctx.root, uid, _CARED_CONTAINERS)
        # find first vapi-created group
        # print(remain, consumed, uid)
        for i, comp in enumerate(trace):
            cfg = get_canvas_item_cfg(comp)
            if cfg is not None and cfg.is_vapi:
                trace = trace[:i]
                consumed_remain = consumed[i:]
                consumed = consumed[:i]
                remain = consumed_remain + remain
                break
        # print(2, v_ctx, name, remain, consumed)

        # fill existed group to ctx
        # print(trace, remain, consumed)
        trace.insert(0, v_ctx.root)
        for i in range(len(consumed)):
            cur_name = UniqueTreeIdForTree.from_parts(consumed[:i + 1])
            # cur_name = ".".join(consumed[:i + 1])
            if cur_name not in v_ctx._name_to_group:

                if i != len(consumed) - 1:
                    v_ctx._name_to_group[cur_name.uid_encoded] = trace[i]
                else:
                    comp = trace[-1]
                    container = trace[-2]

                    if not isinstance(comp, _CARED_CONTAINERS):
                        # replace this comp by group
                        group = three.Group([])
                        item_cfg = get_or_create_canvas_item_cfg(group, True)
                        item_cfg.proxy = GroupProxy(cur_name.uid_encoded)
                        v_ctx._group_assigns[container] = (group, consumed[-1])
                        v_ctx._name_to_group[cur_name.uid_encoded] = group
                        trace[-1] = group
                    else:
                        v_ctx._name_to_group[cur_name.uid_encoded] = comp
        comsumed_name = ".".join(consumed)
        comp = v_ctx._name_to_group[comsumed_name]

        # check is remain tracked in vctx
        remain_copy = remain.copy()
        for remain_part in remain_copy:
            if comsumed_name == "":
                cur_name = remain_part
            else:
                cur_name = f"{comsumed_name}.{remain_part}"
            if cur_name in v_ctx._name_to_group:
                remain.pop(0)
                trace.append(v_ctx._name_to_group[cur_name])
        # found component, check is container first
        # handle remain
        group = trace[-1]
        if remain:
            g = three.Group([])
            _install_obj_event_handlers(g, v_ctx.canvas)

            group_to_yield = g
            v_ctx._name_to_group[uid.uid_encoded] = g
            item_cfg = get_or_create_canvas_item_cfg(g, True)
            item_cfg.proxy = GroupProxy(uid.uid_encoded)

            cur_uid = uid
            for i, remain_part in enumerate(remain[::-1]):
                if i != len(remain) - 1:
                    new_g = three.Group([])
                    _install_obj_event_handlers(new_g, v_ctx.canvas)

                    # cur_uid = cur_uid[:len(cur_uid) - len(remain_part) - 1]
                    cur_uid = cur_uid.pop()
                    # if i == 0:
                    #     assert cur_uid == uid, f"{cur_uid} != {uid}"
                    v_ctx._name_to_group[cur_uid.uid_encoded] = new_g
                    item_cfg = get_or_create_canvas_item_cfg(new_g, True)
                    item_cfg.proxy = GroupProxy(cur_uid.uid_encoded)
                    item_cfg.proxy.childs[remain_part] = new_g
                    g = new_g
                else:
                    if not group.is_mounted():
                        item_cfg = get_or_create_canvas_item_cfg(group, True)
                        assert isinstance(item_cfg.proxy, GroupProxy)
                        item_cfg.proxy.childs[remain_part] = g
                    else:
                        v_ctx._group_assigns[group] = (g, remain[0])
                    # v_ctx._name_to_group[uid] = g
            group = group_to_yield

        # group = three.Group([])
        # v_ctx._name_to_group[uid] = group
    # print(v_ctx._name_to_group)
    # print(v_ctx._group_assigns)
    ev = AppEvent("", {})
    if isinstance(group, three.Group):
        if pos is not None:
            ev += group.update_event(position=pos)
        if rot is not None:
            ev += group.update_event(rotation=rot)
        if variant is not None:
            ev += group.update_event(variant=variant)

    item_cfg = get_or_create_canvas_item_cfg(group)
    if item_cfg.proxy is None:
        item_cfg.proxy = GroupProxy(uid.uid_encoded)
    try:
        v_ctx.name_stack.extend(name_parts)
        yield item_cfg.proxy
    finally:
        for i in range(len(name_parts)):
            v_ctx.name_stack.pop()
        if is_first_ctx:
            if token is not None:
                V_CONTEXT_VAR.reset(token)
            app = get_app()

            if loop is None:
                loop = app._loop
            if loop is None:
                loop = asyncio.get_running_loop()
            if app._flowapp_thread_id == threading.get_ident():
                # we can't wait fut here
                task = asyncio.create_task(_draw_all_in_vctx(v_ctx, UniqueTreeId(""), ev))
                # we can't wait fut here
                # return task
                # return fut
            else:
                # we can wait fut here.
                fut = asyncio.run_coroutine_threadsafe(
                    _draw_all_in_vctx(v_ctx, UniqueTreeId(""), ev), loop)
                fut.result()

async def _uninstall_detail_when_unmount(ev: three.Event, obj: three.Component, canvas: ComplexCanvas):
    object_pyid = obj._flow_uid
    cur_detail_obj_pyid = canvas._cur_detail_layout_object_id
    if object_pyid == cur_detail_obj_pyid:
        await canvas._uninstall_detail_layout() 

async def _install_detail_before_mount(ev: three.Event, obj: three.Component, canvas: ComplexCanvas):
    object_pyid = obj._flow_uid
    cur_detail_obj_pyid = canvas._cur_detail_layout_object_id
    if object_pyid == cur_detail_obj_pyid:
        await canvas._install_detail_layout(obj) 

async def _uninstall_table_when_unmount(ev: three.Event, obj: three.Component, canvas: ComplexCanvas):
    object_pyid = obj._flow_uid
    cur_detail_obj_pyid = canvas._cur_table_object_id
    if object_pyid == cur_detail_obj_pyid:
        await canvas._uninstall_table_layout() 

async def _install_table_before_mount(ev: three.Event, obj: three.Component, canvas: ComplexCanvas):
    object_pyid = obj._flow_uid
    cur_detail_obj_pyid = canvas._cur_table_object_id
    if object_pyid == cur_detail_obj_pyid:
        await canvas._install_table_layout(obj) 

def _install_obj_event_handlers(obj: three.Component, canvas: ComplexCanvas):
    if isinstance(obj, three.Group):
        obj.event_before_mount.on(partial(_install_table_before_mount, obj=obj, canvas=canvas))
        obj.event_before_unmount.on(partial(_uninstall_table_when_unmount, obj=obj, canvas=canvas))
    obj.event_before_mount.on(partial(_install_detail_before_mount, obj=obj, canvas=canvas))
    obj.event_before_unmount.on(partial(_uninstall_detail_when_unmount, obj=obj, canvas=canvas))

def _find_frame_self(*, _frame_cnt: int=1):
    cur_frame = inspect.currentframe()
    assert cur_frame is not None
    frame = cur_frame
    while _frame_cnt > 0:
        frame = cur_frame.f_back
        assert frame is not None
        cur_frame = frame
        _frame_cnt -= 1
    local_vars = cur_frame.f_locals
    if "self" in local_vars:
        obj = local_vars["self"]
        return obj
    return None 
 
def _create_vapi_three_obj_pcfg(obj: three.Component, name: Optional[str], default_name_prefix: str, *, _frame_cnt: int=1):
    v_ctx = get_v_context()
    assert v_ctx is not None
    # if v_ctx.canvas._user_obj_tree_item_to_meta:
    #     # write to standalone vctx for tree item
    #     obj_self = _find_frame_self(_frame_cnt=_frame_cnt + 1)
    #     if obj_self is not None:
    #         if id(obj_self) in v_ctx.canvas._user_obj_tree_item_to_meta:
    #             v_ctx = v_ctx.canvas._user_obj_tree_item_to_meta[id(obj_self)].vctx
    cfg = get_or_create_canvas_item_cfg(v_ctx.current_container)
    proxy = cfg.proxy
    assert proxy is not None
    assert isinstance(proxy, GroupProxy)
    if name is None:
        name = proxy.namepool(default_name_prefix)
    proxy.childs[name] = obj
    _install_obj_event_handlers(obj, v_ctx.canvas)
    pcfg = get_or_create_canvas_item_cfg(obj, True)
    return pcfg

def points(name: str, limit: int):
    point = Points(limit)
    pcfg = _create_vapi_three_obj_pcfg(point, name, "points", _frame_cnt=2)
    pcfg.proxy = CanvasItemProxy()
    return point

def colored_points(name: str, limit: int):
    point = ColoredPoints(limit)
    pcfg = _create_vapi_three_obj_pcfg(point, name, "points", _frame_cnt=2)
    pcfg.proxy = CanvasItemProxy()
    return point

def lines(name: str, limit: int):
    point = Lines(limit)
    pcfg = _create_vapi_three_obj_pcfg(point, name, "lines", _frame_cnt=2)
    pcfg.proxy = CanvasItemProxy()
    return point

def bounding_box(dim: three.Vector3Type, rot: Optional[three.Vector3Type] = None, pos: Optional[three.Vector3Type] = None, name: Optional[str] = None):
    obj = BoundingBox(dim)
    if rot is not None:
        obj.prop(rotation=rot)
    if pos is not None:
        obj.prop(position=pos)
    pcfg = _create_vapi_three_obj_pcfg(obj, name, "box", _frame_cnt=2)
    pcfg.proxy = CanvasItemProxy()
    return obj

def text(text: str, rot: Optional[three.Vector3Type] = None, pos: Optional[three.Vector3Type] = None, name: Optional[str] = None):
    obj = three.Text(text)
    if rot is not None:
        obj.prop(rotation=rot)
    if pos is not None:
        obj.prop(position=pos)
    pcfg = _create_vapi_three_obj_pcfg(obj, name, "text", _frame_cnt=2)
    pcfg.proxy = CanvasItemProxy()
    return obj

def image(img: np.ndarray, rot: Optional[three.Vector3Type] = None, pos: Optional[three.Vector3Type] = None, name: Optional[str] = None, use_datatex: bool = False):
    assert img.dtype == np.uint8 and (img.ndim == 3 or img.ndim == 2)
    obj = Image(img, use_datatex)
    if rot is not None:
        obj.prop(rotation=rot)
    if pos is not None:
        obj.prop(position=pos)
    pcfg = _create_vapi_three_obj_pcfg(obj, name, "img", _frame_cnt=2)
    pcfg.proxy = CanvasItemProxy()
    return obj

def three_ui(comp: three.ThreeComponentType, name: Optional[str] = None):
    # FIXME better way to handle cast layer
    if isinstance(comp, (three.Points, three.Segments, three.BufferMesh, three.InstancedMesh, three.VoxelMesh)):
        comp.props.layers = 31
    _create_vapi_three_obj_pcfg(comp, name, "obj3d", _frame_cnt=2)
    return 

def set_tdata(obj: three.Component, tdata: Dict[str, Any]):
    cfg = get_or_create_canvas_item_cfg(obj)
    cfg.tdata = tdata

def set_detail_layout(obj: three.Component, layout: mui.FlexBox):
    cfg = get_or_create_canvas_item_cfg(obj)
    cfg.detail_layout = layout

def program(name: str, func: Callable):
    # raise NotImplementedError
    group = three.Group([])
    func_dcls = annotated_function_to_dataclass(func)
    func_dcls_obj = func_dcls()
    v_ctx = get_v_context()
    assert v_ctx is not None
    # if v_ctx.canvas._user_obj_tree_item_to_meta:
    #     # write to standalone vctx for tree item
    #     obj_self = _find_frame_self(_frame_cnt=2)
    #     if obj_self is not None:
    #         if obj_self in v_ctx.canvas._user_obj_tree_item_to_meta:
    #             v_ctx = v_ctx.canvas._user_obj_tree_item_to_meta[obj_self].vctx
    cfg = get_or_create_canvas_item_cfg(v_ctx.current_container)
    proxy = cfg.proxy
    assert proxy is not None
    assert isinstance(proxy, GroupProxy)
    proxy.childs[name] = group
    pcfg = get_or_create_canvas_item_cfg(group, True)

    async def callback(uid: str, value: Any):
        # print(uid, value)
        if "." in uid:
            return
        if group.is_mounted():
            setattr(func_dcls_obj, uid, value)
            vctx_program = VContext(v_ctx.canvas, group)
            with enter_v_conetxt(vctx_program):
                kwargs = {}
                for field in dataclasses.fields(func_dcls_obj):
                    kwargs[field.name] = getattr(func_dcls_obj, field.name)
                res = func(**kwargs)
                if inspect.iscoroutine(res):
                    await res
            # we need to update tree iff tree change because update tree is very slow.
            await _draw_all_in_vctx(vctx_program, group._flow_uid, update_iff_change=True)

    pcfg.detail_layout = ConfigPanelV2(func_dcls_obj, callback)
    pcfg.proxy = GroupProxy("")
    return
