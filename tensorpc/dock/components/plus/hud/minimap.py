import asyncio
import bisect
import enum
from functools import partial
import math
import time

from tensorpc.core.datamodel.draft import DraftBase, DraftFieldMeta
from tensorpc.core.datamodel.events import DraftChangeEvent
from tensorpc.dock.components import three, mui
from tensorpc.core import dataclass_dispatch as dataclasses, pfl
from typing import Any, Optional, Union 
from typing_extensions import Annotated
import numpy as np
import tensorpc.core.datamodel as D
from tensorpc.dock.components.plus.tensorutil import TensorContainer
from tensorpc.dock.components.three.event import HudLayoutChangeEvent, KeyboardHoldEvent, PointerEvent
from tensorpc.core.pfl.backends.js import Math, MathUtil
from tensorpc.dock.core.datamodel import _compile_pfllibrary

@dataclasses.dataclass
class SimpleLayout:
    scrollFactorX: three.NumberType = 1.0
    scrollFactorY: three.NumberType = 1.0
    innerSizeX: three.NumberType = 1.0
    innerSizeY: three.NumberType = 1.0


class MinimapFitMode(enum.IntEnum):
    WIDTH = 0
    HEIGHT = 1
    AUTO = 2

class MinimapAlignMode(enum.IntEnum):
    CENTER = 0
    LEFT_TOP = 1
    RIGHT_BOTTOM = 2

@dataclasses.dataclass
class MinimapModel:
    width: float = 1.0
    height: float = 1.0
    scale: float = 1.0
    scrollValueX: float = 0.0
    scrollValueY: float = 0.0

    hover: Optional[PointerEvent] = None
    layout: SimpleLayout = dataclasses.field(default_factory=SimpleLayout)

    isMinimapDown: bool = False

    viewport_canvas_width: float = 1.0
    viewport_canvas_height: float = 1.0

    viewport_canvas_scale_w: float = 1.0
    viewport_canvas_scale_h: float = 1.0

    child_scale: float = 1.0 
    childOffsetX: float = 0.0
    childOffsetY: float = 0.0

    wheel_speed: float = 0.001
    fit_mode: int = int(MinimapFitMode.AUTO)
    align_mode: int = int(MinimapAlignMode.CENTER)

    @pfl.js.mark_js_compilable
    def _wheel_handler_pfl_v2(self, data: PointerEvent):
        if data.wheel:
            dx = -data.wheel.deltaY * self.wheel_speed * self.scale
            self._update_new_scroll_value(dx, data.pointLocal[0], data.pointLocal[1])

    @pfl.js.mark_js_compilable
    def _minimap_cllick_pfl(self, data: PointerEvent):
        w = Math.max(1 - self.layout.scrollFactorX, 1e-6)
        h = Math.max(1 - self.layout.scrollFactorY, 1e-6)
        self.scrollValueX = MathUtil.clamp((data.pointLocal[0] + 0.5 - self.layout.scrollFactorX / 2) / w, 0, 1)
        self.scrollValueY = MathUtil.clamp((-data.pointLocal[1] + 0.5 - self.layout.scrollFactorY / 2) / h, 0, 1)

    @pfl.js.mark_js_compilable
    def _minimap_downmove_pfl(self, data: PointerEvent):
        if self.isMinimapDown and data.numIntersections > 0:
            w = 1 - self.layout.scrollFactorX
            h = 1 - self.layout.scrollFactorY
            px = MathUtil.clamp(data.pointLocal[0] + 0.5, 0, 1)
            py = MathUtil.clamp(-data.pointLocal[1] + 0.5, 0, 1)
            self.scrollValueX = MathUtil.clamp((px - self.layout.scrollFactorX / 2) / w, 0, 1)
            self.scrollValueY = MathUtil.clamp((py - self.layout.scrollFactorY / 2) / h, 0, 1)

            # print("_minimap_downmove_pfl", data, px, py, self.scrollValueX, self.scrollValueY)

    @pfl.js.mark_js_compilable
    def _keyhold_handler_pfl(self, data: KeyboardHoldEvent):
        if data.code == "Space":
            # reset scales and scroll values
            self.scale = 1.0
            self.scrollValueX = 0.0
            self.scrollValueY = 0.0
            return
        is_move = data.code == "KeyW" or data.code == "KeyS" or data.code == "KeyA" or data.code == "KeyD"
        is_zoom = data.code == "KeyZ" or data.code == "KeyX"
        if is_move:
            dx = 0
            dy = 0
            if data.code == "KeyW":
                dy = -data.deltaTime
            elif data.code == "KeyS":
                dy = data.deltaTime
            elif data.code == "KeyA":
                dx = -data.deltaTime
            else:
                dx = data.deltaTime
            dx *= 0.002 / self.scale 
            dy *= 0.002 / self.scale
            self.scrollValueX = MathUtil.clamp(self.scrollValueX + dx, 0.0, 1.0)
            self.scrollValueY = MathUtil.clamp(self.scrollValueY + dy, 0.0, 1.0)
        if is_zoom:
            delta = 0
            if data.code == "KeyZ":
                delta = -data.deltaTime * 0.001
            elif data.code == "KeyX":
                delta = data.deltaTime * 0.001
            dx = delta * self.scale
            self._update_new_scroll_value(dx, 0, 0)
    
    @pfl.js.mark_js_compilable
    def _update_new_scroll_value(self, dx: float, x: float, y: float) -> None:
        prev = self.scale

        prev_scroll_value_x = self.scrollValueX
        prev_scroll_value_y = self.scrollValueY
        new_scale = MathUtil.clamp(dx + prev, 1.0, 100.0)
        self.scale = new_scale
        scale_x_prev = 1
        scale_y_prev = 1
        new_scale_x = 1
        new_scale_y = 1
        scale_x_prev = prev * self.viewport_canvas_scale_w
        scale_y_prev = prev * self.viewport_canvas_scale_h
        new_scale_x = new_scale * self.viewport_canvas_scale_w
        new_scale_y = new_scale * self.viewport_canvas_scale_h

        real_dx = new_scale_x - scale_x_prev
        real_dy = new_scale_y - scale_y_prev
        rprev_x = 1 / scale_x_prev
        rprev_y = 1 / scale_y_prev
        Px = (1 - rprev_x) * prev_scroll_value_x + (x + 0.5) * rprev_x
        Py = (1 - rprev_y) * prev_scroll_value_y + (-y + 0.5) * rprev_y
        # Px = data.pointLocal[0] + 0.5
        # Py = -data.pointLocal[1] + 0.5
        # print("PxPy", Px, Py, (data.pointLocal[0] + 0.5), (-data.pointLocal[1] + 0.5))
        self.scrollValueX = MathUtil.clamp((Px - prev_scroll_value_x) * real_dx / Math.max(new_scale_x - 1.0, 1e-6) + prev_scroll_value_x, 0.0, 1.0)
        self.scrollValueY = MathUtil.clamp((Py - prev_scroll_value_y) * real_dy / Math.max(new_scale_y - 1.0, 1e-6) + prev_scroll_value_y, 0.0, 1.0)
        self._do_layout()

    @pfl.js.mark_js_compilable
    def _handle_layout_event(self, ev: HudLayoutChangeEvent):
        self.layout.scrollFactorX = ev.scrollFactorX
        self.layout.scrollFactorY = ev.scrollFactorY
        self.layout.innerSizeX = ev.innerSizeX
        self.layout.innerSizeY = ev.innerSizeY
        self._do_layout()

    @pfl.js.mark_js_compilable
    def _do_layout(self):
        w_scale_rate = self.layout.innerSizeX / self.width
        h_scale_rate = self.layout.innerSizeY / self.height
        self.viewport_canvas_scale_w = 1.0
        self.viewport_canvas_scale_h = 1.0
        is_auto_w = self.fit_mode == MinimapFitMode.AUTO and ((self.layout.innerSizeY / self.layout.innerSizeX) > (self.height / self.width))
        if self.fit_mode == MinimapFitMode.WIDTH or is_auto_w:
            self.viewport_canvas_scale_h = Math.max(1.0, self.height * w_scale_rate / self.layout.innerSizeY)
            self.viewport_canvas_width = self.scale * self.layout.innerSizeX
            self.viewport_canvas_height = self.scale * self.layout.innerSizeY * self.viewport_canvas_scale_h
            self.child_scale = self.scale * w_scale_rate
            self.childOffsetX = 0.0
            if self.align_mode == MinimapAlignMode.LEFT_TOP:
                self.childOffsetY = (self.viewport_canvas_height - self.height * self.child_scale) / 2
            elif self.align_mode == MinimapAlignMode.RIGHT_BOTTOM:
                self.childOffsetY = -(self.viewport_canvas_height - self.height * self.child_scale) / 2

        elif self.fit_mode == MinimapFitMode.HEIGHT or (not is_auto_w):
            self.viewport_canvas_scale_w = Math.max(1.0, self.width * h_scale_rate / self.layout.innerSizeX)
            self.viewport_canvas_width = self.scale * self.layout.innerSizeX * self.viewport_canvas_scale_w
            self.viewport_canvas_height = self.scale * self.layout.innerSizeY
            self.child_scale = self.scale * h_scale_rate
            self.childOffsetY = 0.0
            if self.align_mode == MinimapAlignMode.LEFT_TOP:
                self.childOffsetX = -(self.viewport_canvas_width - self.width * self.child_scale) / 2
            elif self.align_mode == MinimapAlignMode.RIGHT_BOTTOM:
                self.childOffsetX = (self.viewport_canvas_width - self.width * self.child_scale) / 2


class MinimapLayer(enum.IntEnum):
    BKGD_LAYER = -100
    CHILD_LAYER = -4
    MINIMAP_EVENT_LAYER = -3
    MINIMAP_LAYER = -2

class MiniMap(three.Group):
    def __init__(self, draft: MinimapModel, childs: three.ThreeLayoutType, 
            mini_childs: Optional[three.ThreeLayoutType] = None, minimap_event_key: str = ""):
        self._wheel_speed = 0.001
        # ten can be torch or numpy.
        assert isinstance(draft, DraftBase), "draft must be a DraftBase instance"
        base_event_plane = three.Mesh([
            three.PlaneGeometry(1.0, 1.0),
            three.MeshBasicMaterial().prop(transparent=True),
        ]).prop(position=(0, 0, int(MinimapLayer.BKGD_LAYER)))
        self.event_plane = base_event_plane
        child_group = three.Group(childs).prop(position=(0, 0, int(MinimapLayer.CHILD_LAYER)))
        self._child_group = child_group
        viewport_group = three.HudGroup([
            child_group,
            # base_event_plane

        ])
        viewport_group.prop(top=0, left=0, padding=1, width="100%", height="100%", alignContent=False, alwaysPortal=False)
        self.viewport_group = viewport_group
        line_minimap = three.Group([
            three.Line([(-0.0, 0.0, 0.0), ]).prop(color="blue", lineWidth=2, variant="aabb", aabbSizes=(1, 1, 1))
        ])
        scrollbar_plane_group = three.HudGroup([
            base_event_plane
        ]).prop(top=0, left=0, width="100%", height="100%", childWidth=1, childHeight=1, alignContent="stretch")

        minimap_event_plane = three.Mesh([
            three.PlaneGeometry(1.0, 1),
            three.MeshBasicMaterial().prop(transparent=True, opacity=0.0),
        ]).prop(position=(0, 0, -1))

        minimap_group = three.HudGroup([
            line_minimap,
            minimap_event_plane,
        ]).prop(position=(0, 0, int(MinimapLayer.MINIMAP_LAYER)), bottom=5, right=5, padding=0, 
                width="20%", height="20%", alignContent="stretch", alwaysPortal=False, borderWidth=1, 
                borderColor="red", childWidth=1, childHeight=1)
        if mini_childs is not None:
            minimap_group.init_add_layout(mini_childs)
        dm = mui.DataModel.get_datamodel_from_draft(draft)
        dm.install_model_update_callback(f"_minimap_do_layout_{minimap_event_key}", MinimapModel._do_layout,   
            submodel_draft=draft)
        # viewport_group.event_hud_layout_change.add_frontend_draft_change(draft, "layout", r"{innerSizeX: innerSizeX, innerSizeY: innerSizeY, scrollFactorX: scrollFactorX, scrollFactorY: scrollFactorY}")
        viewport_group.event_hud_layout_change.add_frontend_handler(
            dm, 
            MinimapModel._handle_layout_event, targetPath=str(draft))
        # image.event_move.add_frontend_draft_change(draft, "hover")
        # image.event_leave.add_frontend_draft_set_none(draft, "hover")
        base_event_plane.event_wheel.add_frontend_handler(dm, MinimapModel._wheel_handler_pfl_v2, targetPath=str(draft))
        viewport_group.bind_fields(
            childWidth=draft.viewport_canvas_width, 
            childHeight=draft.viewport_canvas_height, 
            scrollValueY=draft.scrollValueY, scrollValueX=draft.scrollValueX)
        # child_group.bind_fields(scale=draft.child_scale)
        child_group.bind_fields_unchecked_dict({
            "scale-x": f"{draft.child_scale}",
            "scale-y": f"{draft.child_scale}",
            "position-x": f"{draft.childOffsetX}",
            "position-y": f"{draft.childOffsetY}",
        })

        line_minimap.bind_fields_unchecked_dict({
            "position-x": f"({draft.scrollValueX} - 0.5) * (1 - {draft.layout.scrollFactorX})",
            "position-y": f"-({draft.scrollValueY} - 0.5) * (1 - {draft.layout.scrollFactorY})",
            "scale-x": draft.layout.scrollFactorX,
            "scale-y": draft.layout.scrollFactorY,
        })

        minimap_event_plane.event_move.add_frontend_handler(dm, MinimapModel._minimap_downmove_pfl, targetPath=str(draft))
        
        
        minimap_event_plane.event_leave.add_frontend_draft_change(draft, "isMinimapDown", "False")
        minimap_event_plane.event_down.add_frontend_draft_change(draft, "isMinimapDown", "True")
        minimap_event_plane.event_up.add_frontend_draft_change(draft, "isMinimapDown", "Fanse")
        minimap_event_plane.event_down.configure(set_pointer_capture=True)
        minimap_event_plane.event_up.configure(release_pointer_capture=True)

        minimap_event_plane.event_click.add_frontend_handler(dm, MinimapModel._minimap_cllick_pfl, targetPath=str(draft))

        super().__init__([
            viewport_group,  
            minimap_group,
            scrollbar_plane_group,
        ])


    def install_canvas_events(self, draft: Any, canvas: Union[three.Canvas, three.View]):
        dm = mui.DataModel.get_datamodel_from_draft(draft)
        canvas.event_keyboard_hold.configure(key_codes=["KeyW", "KeyS", "KeyA", "KeyD", "KeyZ", "KeyX", "Space"])
        canvas.event_keyboard_hold.add_frontend_handler(dm, MinimapModel._keyhold_handler_pfl, targetPath=str(draft))

    async def set_new_childs(self, childs: three.ThreeLayoutType):
        await self._child_group.set_new_layout(childs) # type: ignore

if __name__ == "__main__":
    _compile_pfllibrary(MinimapModel)