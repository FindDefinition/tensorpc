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

import dataclasses
import inspect
from typing import Any, Callable, Dict, Optional, List, Tuple, Type, TypeVar, Union, get_type_hints
from typing_extensions import Annotated
import contextvars
import contextlib
from tensorpc.core.dataclass_dispatch import dataclass
from tensorpc.flow.flowapp import appctx
from tensorpc.flow.flowapp.components.plus.config import ConfigPanelV2
from ... import three
from ...typemetas import (ColorRGB, ColorRGBA, RangedFloat, RangedInt,
                          RangedVector3, Vector3, annotated_function_to_dataclass)
from .canvas import ComplexCanvas, find_component_trace_by_uid_with_not_exist_parts
from .core import CanvasItemCfg, CanvasItemProxy
import numpy as np 
from tensorpc.utils.uniquename import UniqueNamePool
from .core import get_canvas_item_cfg, get_or_create_canvas_item_cfg

class ContainerProxy(CanvasItemProxy):
    pass

class GroupProxy(ContainerProxy):
    def __init__(self, uid: str) -> None:
        super().__init__() 
        self.uid = uid 

        self.childs: Dict[str, three.Component] = {}

        self._namepool = UniqueNamePool()

    def __repr__(self) -> str:
        return f"<GroupProxy {self.uid}>"

class PointsProxy(CanvasItemProxy):
    def __init__(self) -> None:
        super().__init__()
        self._points: List[three.Vector3Type] = []
        self._points_arr: List[np.ndarray] = []

        self._size: three.NumberType = 3
        self._limit : Optional[int] = None

    def p(self, x: float, y: float, z: float):
        self._points.append((x, y, z))
        return self 

    def array(self, data: np.ndarray):
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        self._points_arr.append(data)
        return self

    def size(self, size: three.NumberType):
        self._size = size
        return self

    def limit(self, limit: int):
        self._limit = limit
        return self

    def update_event(self, comp: three.Points):
        # TODO global config
        points_nparray = np.array(self._points, dtype=np.float32)
        if self._points_arr:
            points_nparray = np.concatenate(self._points_arr + [points_nparray])
        if self._limit is not None:
            return comp.update_event(limit=self._limit, size=self._size, points=points_nparray)
        else:
            return comp.update_event(size=self._size, points=points_nparray)

class ColoredPointsProxy(PointsProxy):
    def __init__(self) -> None:
        super().__init__()
        self._point_colors: List[Tuple[int, int, int]] = []
        self._point_colors_arr: List[np.ndarray] = []

    def p(self, x: float, y: float, z: float, r: int, g: int, b: int):
        self._points.append((x, y, z))
        self._point_colors.append((r, g, b))
        return self 

    def array(self, points: np.ndarray, colors: np.ndarray):
        if points.dtype != np.float32:
            points = points.astype(np.float32)
        three.Points._check_colors(colors, points)
        self._points_arr.append(points)
        self._point_colors_arr.append(colors)
        return self

    def update_event(self, comp: three.Points):
        points_nparray = np.array(self._points, dtype=np.float32)
        colors_nparray = np.array(self._point_colors, dtype=np.uint8)
        if self._points_arr:
            points_nparray = np.concatenate(self._points_arr + [points_nparray])
            colors_nparray = np.concatenate(self._point_colors_arr + [colors_nparray])
        if self._limit is not None:
            return comp.update_event(limit=self._limit, size=self._size, points=points_nparray, colors=colors_nparray)
        else:
            return comp.update_event(size=self._size, points=points_nparray, colors=colors_nparray)

class _Polygon:
    def __init__(self, start: three.Vector3Type, closed: bool, line_proxy: "LinesProxy") -> None:
        self.line_proxy = line_proxy
        self.closed = closed
        self.start = start

    def to(self, x: float, y: float, z: float):
        self.line_proxy.p(self.start[0], self.start[1], self.start[2], x, y, z)
        self.start = (x, y, z)
        return self

class LinesProxy(CanvasItemProxy):
    def __init__(self) -> None:
        super().__init__()
        self._point_pairs: List[Tuple[three.Vector3Type, three.Vector3Type]] = []
        self._width: three.NumberType = 1
        self._limit : Optional[int] = None
        self._lines_arr: List[np.ndarray] = []

    def limit(self, limit: int):
        self._limit = limit
        return self

    def p(self, x1: float, y1: float, z1: float, x2: float, y2: float, z2: float):
        self._point_pairs.append(((x1, y1, z1), (x2, y2, z2)))
        return self 

    def array(self, data: np.ndarray):
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        three.SegmentsProps.lines_validator(data)
        self._lines_arr.append(data)
        return self

    def width(self, width: three.NumberType):
        self._width = width
        return self

    def update_event(self, comp: three.Segments):
        lines_array = np.array(self._point_pairs, dtype=np.float32)
        if self._lines_arr:
            lines_array = np.concatenate(self._lines_arr + [lines_array])
        if self._limit is not None:
            return comp.update_event(limit=self._limit, lineWidth=self._width, lines=lines_array)
        else:
            return comp.update_event(lineWidth=self._width, lines=lines_array)

    def polygon(self, x: float, y: float, z: float, closed: bool = False):
        return _Polygon((x, y, z), closed, self)

    def closed_polygon(self, x: float, y: float, z: float):
        return _Polygon((x, y, z), True, self)

class VContext:
    def __init__(self, canvas: ComplexCanvas, root: Optional[three.ContainerBase] = None):
        self.stack = []
        self.canvas = canvas
        self.name_stack: List[str] = []
        self.exist_name_stack: List[str] = []
        if root is None:
            root = canvas._item_root
        self.root = root
        self._name_to_group: Dict[str, three.ContainerBase] = {
            "": root
        }
        self._group_assigns: Dict[three.ContainerBase, Tuple[three.Component, str]] = {}

    @property 
    def current_namespace(self):
        return ".".join(self.name_stack)

    @property 
    def current_container(self):
        if not self.name_stack:
            return self.root
        else:
            return self._name_to_group[self.current_namespace]

    def extract_group_assigns(self):
        """group created by vapi will be recreated in each vctx.
        """
        pass 

V_CONTEXT_VAR: contextvars.ContextVar[
    Optional[VContext]] = contextvars.ContextVar("v_context",
                                                   default=None)

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

async def _draw_all_in_vctx(vctx: VContext):
    for k, v in vctx._name_to_group.items():
        cfg = get_canvas_item_cfg(v)
        if cfg is not None:
            proxy = cfg.proxy
            if proxy is not None:
                assert isinstance(proxy, GroupProxy)
                if cfg.is_vapi:
                    assert not v.is_mounted()
                    v.init_add_layout(proxy.childs)
                else:
                    await v.update_childs(proxy.childs)
    for container, (group, name) in vctx._group_assigns.items():
        await container.update_childs({
            name: group
        })
    await vctx.canvas.item_tree.update_tree()

@contextlib.contextmanager
def ctx(canvas: Optional[ComplexCanvas] = None):
    if canvas is None:
        canvas = appctx.find_component(ComplexCanvas)
        assert canvas is not None, "you must add complex canvas before using vapi"
    
    prev_ctx = get_v_context()
    is_first_ctx = False
    if prev_ctx is None:
        is_first_ctx = True
        prev_ctx = VContext(canvas)
    token = V_CONTEXT_VAR.set(prev_ctx)
    try:
        yield prev_ctx
    finally:
        if is_first_ctx:
            # construct components
            pass 
        V_CONTEXT_VAR.reset(token)

def get_group_context() -> Optional[GroupProxy]:
    return GROUP_CONTEXT_VAR.get()

@contextlib.contextmanager
def enter_group_context(robj: GroupProxy):
    token = GROUP_CONTEXT_VAR.set(robj)
    try:
        yield robj
    finally:
        GROUP_CONTEXT_VAR.reset(token)

_CARED_CONTAINERS  =  (three.Group, three.Fragment, three.Hud)

T_container = TypeVar("T_container", bound=three.ContainerBase)
T_container_proxy = TypeVar("T_container_proxy", bound=three.ContainerBase)

@contextlib.contextmanager
def group(name: str, check_override: bool = False, ):
    name_parts = name.split(".")
    assert name_parts[0] != "reserved" and name != ""
    for p in name_parts:
        assert p, "group name can not be empty"
    # find exist group in canvas
    v_ctx = get_v_context()
    assert v_ctx is not None 
    # canvas = v_ctx.canvas 
    if v_ctx.name_stack:
        uid = f"{v_ctx.current_namespace}.{name}"
    else:
        uid = name
    if uid in v_ctx._name_to_group:
        group = v_ctx._name_to_group[uid]
    else:
        trace, remain, consumed = find_component_trace_by_uid_with_not_exist_parts(v_ctx.root, uid, _CARED_CONTAINERS)
        # find first vapi-created group
        for i, comp in enumerate(trace):
            cfg = get_canvas_item_cfg(comp)
            if cfg is not None and cfg.is_vapi:
                trace = trace[:i]
                consumed_remain = consumed[i:]
                consumed = consumed[:i]
                remain = consumed_remain + remain
                break
        # fill existed group to ctx
        # print(trace, remain, consumed)
        trace.insert(0, v_ctx.root)
        for i in range(len(consumed)):
            cur_name = ".".join(consumed[:i + 1])
            if cur_name not in v_ctx._name_to_group:
                if i != len(consumed) - 1:
                    v_ctx._name_to_group[cur_name] = trace[i]
                else:
                    comp = trace[-1]
                    container = trace[-2]

                    if not isinstance(comp, _CARED_CONTAINERS):
                        # replace this comp by group
                        group = three.Group([])
                        item_cfg = get_or_create_canvas_item_cfg(group, True)
                        item_cfg.proxy = GroupProxy(cur_name)
                        v_ctx._group_assigns[container] = (group, consumed[-1])
                        v_ctx._name_to_group[cur_name] = group
                        trace[-1] = group
                    else:
                        v_ctx._name_to_group[cur_name] = comp
        comsumed_name = ".".join(consumed)
        comp = v_ctx._name_to_group[comsumed_name]

        # check is remain tracked in vctx
        remain_copy = remain.copy()
        for remain_part in remain_copy:
            cur_name = f"{comsumed_name}.{remain_part}"
            if cur_name in v_ctx._name_to_group:
                remain.pop(0)
                trace.append(v_ctx._name_to_group[cur_name])
        
        # found component, check is container first
        # handle remain
        group = trace[-1]
        if remain:
            g = three.Group([]) 
            group_to_yield = g
            v_ctx._name_to_group[uid] = g
            item_cfg = get_or_create_canvas_item_cfg(g, True)
            item_cfg.proxy = GroupProxy(uid)

            cur_uid = uid
            for i, remain_part in enumerate(remain[::-1]):
                if i != len(remain) - 1:
                    new_g = three.Group([])
                    cur_uid = cur_uid[:len(cur_uid) - len(remain_part) - 1]
                    # if i == 0:
                    #     assert cur_uid == uid, f"{cur_uid} != {uid}"
                    v_ctx._name_to_group[cur_uid] = new_g
                    item_cfg = get_or_create_canvas_item_cfg(new_g, True)
                    item_cfg.proxy = GroupProxy(cur_uid)
                    item_cfg.proxy.childs[remain_part] = new_g
                    g = new_g
                else:
                    v_ctx._group_assigns[group] = (g, remain[0])
                    # v_ctx._name_to_group[uid] = g
            group = group_to_yield
        
        # group = three.Group([])
        # v_ctx._name_to_group[uid] = group
    # print(v_ctx._name_to_group)
    # print(v_ctx._group_assigns)
    item_cfg = get_or_create_canvas_item_cfg(group)
    if item_cfg.proxy is None:
        item_cfg.proxy = GroupProxy(uid)
    try:
        v_ctx.name_stack.extend(name_parts)
        yield item_cfg.proxy
    finally:
        for i in range(len(name_parts)):
            v_ctx.name_stack.pop()


def points(name: str, limit: int):
    point = three.Points(limit)
    v_ctx = get_v_context()
    assert v_ctx is not None 
    cfg = get_or_create_canvas_item_cfg(v_ctx.current_container)
    proxy = cfg.proxy 
    assert proxy is not None 
    assert isinstance(proxy, GroupProxy)
    proxy.childs[name] = point
    pcfg = get_or_create_canvas_item_cfg(point, True)
    pcfg.proxy = PointsProxy()
    return pcfg.proxy

def lines(name: str, limit: int):
    point = three.Segments(limit)
    v_ctx = get_v_context()
    assert v_ctx is not None 
    cfg = get_or_create_canvas_item_cfg(v_ctx.current_container)
    proxy = cfg.proxy 
    assert proxy is not None 
    assert isinstance(proxy, GroupProxy)
    proxy.childs[name] = point
    pcfg = get_or_create_canvas_item_cfg(point, True)
    pcfg.proxy = LinesProxy()
    return pcfg.proxy


def program(name: str, func: Callable):
    # raise NotImplementedError
    group = three.Group([])
    func_dcls = annotated_function_to_dataclass(func)
    func_dcls_obj = func_dcls()
    v_ctx = get_v_context()
    assert v_ctx is not None 
    cfg = get_or_create_canvas_item_cfg(v_ctx.current_container)
    proxy = cfg.proxy 
    assert proxy is not None 
    assert isinstance(proxy, GroupProxy)
    proxy.childs[name] = group
    pcfg = get_or_create_canvas_item_cfg(group, True)
    async def callback(uid: str, value: Any):
        if "." in uid:
            return 
        setattr(func_dcls_obj, uid, value)
        vctx_program = VContext(v_ctx.canvas, group)
        with enter_v_conetxt(vctx_program):
            kwargs = {}
            for field in dataclasses.fields(func_dcls_obj):
                kwargs[field.name] = getattr(func_dcls_obj, field.name)
            res = func(**kwargs)
            if inspect.iscoroutine(res):
                await res 
    pcfg.detail_layout = ConfigPanelV2(func_dcls_obj, callback)
    return

