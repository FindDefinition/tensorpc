import asyncio
import bisect
from functools import partial
import math
import time

from tensorpc.core.datamodel.draft import DraftFieldMeta
from tensorpc.core.datamodel.events import DraftChangeEvent
from tensorpc.dock import mui, three, plus, appctx, mark_create_layout
import dataclasses
from typing import Any, Optional 
from typing_extensions import Annotated
import numpy as np
import tensorpc.core.datamodel as D
from tensorpc.dock.components.plus.tensorutil import TensorContainer
from tensorpc.dock.components.three.event import PointerEvent
from tensorpc.core.pfl.pfl_std import Math, MathUtil

@dataclasses.dataclass
class SimpleLayout:
    scrollFactorX: three.NumberType = 1.0
    scrollFactorY: three.NumberType = 1.0
    innerSizeX: three.NumberType = 1.0
    innerSizeY: three.NumberType = 1.0


@dataclasses.dataclass
class Model:
    image: np.ndarray
    scale: float # image will keep aspect ratio, so we only need one scale.
    scrollValueX: float 
    scrollValueY: float
    linePosX: Optional[float]
    linePosY: Optional[float]

    hover: Optional[PointerEvent] = None
    layout: SimpleLayout = dataclasses.field(default_factory=SimpleLayout)

    isMinimapDown: bool = False

def _wheel_handler_pfl(root: Model, data: PointerEvent):
    if data.wheel:
        prev = root.scale
        prev_scroll_value_x = root.scrollValueX
        prev_scroll_value_y = root.scrollValueY
        dx = -data.wheel.deltaY * 0.001 * root.scale
        new_scale = MathUtil.clamp(dx + prev, 1.0, 100.0)
        real_dx = new_scale - prev
        root.scale = new_scale
        root.scrollValueX = MathUtil.clamp((data.pointLocal[0] + 0.5 - prev_scroll_value_x) * real_dx / Math.max(new_scale - 1.0, 1e-6) + prev_scroll_value_x, 0.0, 1.0)
        root.scrollValueY = MathUtil.clamp((-data.pointLocal[1] + 0.5 - prev_scroll_value_y) * real_dx / Math.max(new_scale - 1.0, 1e-6) + prev_scroll_value_y, 0.0, 1.0)

def _minimap_cllick_pfl(root: Model, data: PointerEvent):
    w = Math.max(1 - root.layout.scrollFactorX, 1e-6)
    h = Math.max(1 - root.layout.scrollFactorY, 1e-6)
    root.scrollValueX = MathUtil.clamp((data.pointLocal[0] + 0.5 - root.layout.scrollFactorX / 2) / w, 0, 1)
    root.scrollValueY = MathUtil.clamp((-data.pointLocal[1] + 0.5 - root.layout.scrollFactorY / 2) / h, 0, 1)

def _minimap_downmove_pfl(root: Model, data: PointerEvent):
    if root.isMinimapDown:
        w = 1 - root.layout.scrollFactorX
        h = 1 - root.layout.scrollFactorY

        root.scrollValueX = MathUtil.clamp((data.pointLocal[0] + 0.5 - root.layout.scrollFactorX / 2) / w, 0, 1)
        root.scrollValueY = MathUtil.clamp((-data.pointLocal[1] + 0.5 - root.layout.scrollFactorY / 2) / h, 0, 1)


def _img_move_pfl(root: Model, data: PointerEvent):
    pixel_x = Math.floor((data.pointLocal[0] + 0.5) * root.image.shape[1])
    pixel_y = Math.floor((-data.pointLocal[1] + 0.5) * root.image.shape[0])
    x = (pixel_x + 0.5) - root.image.shape[1] / 2
    y = (-(pixel_y + 0.5)) + root.image.shape[0] / 2
    root.linePosX = x
    root.linePosY = y


class TensorPanel(mui.FlexBox):
    def __init__(self):
        # ten can be torch or numpy.
        image = three.Image()

        # line = three.Line([(0, 0, 0), (1, 1, 1)]).prop(color="red", lineWidth=2, variant="aabb")
        line = three.Group([
            three.Line([(0.0, 0.0, 0.0), ]).prop(color="blue", lineWidth=2, variant="aabb", aabbSizes=(1, 1, 1))
        ]).prop(position=(0, 0, 0.1))
        line_cond = mui.MatchCase.binary_selection(True, line)

        img_group = three.Group([
            image,
            line_cond,
        ]).prop(position=(0, 0, -1.2))

        self._cam_ctrl = three.CameraControl().prop(makeDefault=True)

        viewport_group = three.HudGroup([
            img_group
        ]).prop(top=0, left=0, padding=5, width="100%", height="100%", alignContent=False, alwaysPortal=False, borderWidth=2, borderColor="aqua")

        line_minimap = three.Group([
            three.Line([(-0.0, 0.0, 0.0), ]).prop(color="blue", lineWidth=2, variant="aabb", aabbSizes=(1, 1, 1))
        ])
        minimap_event_plane = three.Mesh([
            three.PlaneGeometry(1.0, 1000.0),
            three.MeshBasicMaterial().prop(transparent=True, opacity=0.0),
        ]).prop(position=(0, 0, -0.2))

        minimap_group = three.HudGroup([
            line_minimap,
            minimap_event_plane,
        ]).prop(bottom=5, right=5, padding=0, width="20%", height="20%", alignContent="stretch", alwaysPortal=False, borderWidth=1, borderColor="red", childWidth=1, childHeight=1)

        cam = three.OrthographicCamera(near=0.1, far=1000, children=[
            viewport_group,  
            minimap_group,
        ]).prop(position=(0, 0, 10))
        # cam = three.PerspectiveCamera(fov=75, near=0.1, far=1000, children=[
        #     # viewport_group,  
        #     # # boxmeshX,
        #     # scrollbar_group,
        #     # scrollbar_bottom_group,
        # ]).prop(position=(0, 0, 10))

        canvas = three.Canvas([
            # self._cam_ctrl,
            cam.prop(makeDefault=True),
            # three.InfiniteGridHelper(5, 50, "gray"),
            # image,
        ]).prop(enablePerf=False, allowKeyboardEvent=True)
        empty_model = self._create_empty_vis_model()
        dm = mui.DataModel(empty_model, [])
        draft = dm.get_draft()
        viewport_group.event_hud_layout_change.add_frontend_draft_change(draft, "layout", r"{innerSizeX: innerSizeX, innerSizeY: innerSizeY, scrollFactorX: scrollFactorX, scrollFactorY: scrollFactorY}")
        # image.event_move.add_frontend_draft_change(draft, "hover")
        # image.event_leave.add_frontend_draft_set_none(draft, "hover")
        image.event_wheel.add_frontend_handler(_wheel_handler_pfl)
        image.event_move.add_frontend_handler(_img_move_pfl)
        image.event_leave.add_frontend_draft_set_none(draft, "linePosX")

        image.bind_fields(image="image", scale="image.shape[0]")

        viewport_group.bind_fields(childWidthScale="scale", childHeightScale=f"scale", scrollValueY="scrollValueY", scrollValueX="scrollValueX")
        img_group.bind_fields_unchecked_dict({
            "scale-x": "scale * layout.innerSizeX / (where(image.shape[1] == `0`, layout.innerSizeX, image.shape[1]))",
            "scale-y": "scale * layout.innerSizeY / (where(image.shape[0] == `0`, layout.innerSizeY, image.shape[0]))",
        })
        line_minimap.bind_fields_unchecked_dict({
            "position-x": "(scrollValueX - `0.5`) * (`1` - layout.scrollFactorX)",
            "position-y": "-(scrollValueY - `0.5`) * (`1` - layout.scrollFactorY)",
            "scale-x": "layout.scrollFactorX",
            "scale-y": "layout.scrollFactorY",
        })
        line.bind_fields_unchecked_dict({
            "position-x": "linePosX",
            "position-y": "linePosY",
        })

        minimap_event_plane.event_move.add_frontend_handler(_minimap_downmove_pfl)
        minimap_event_plane.event_leave.add_frontend_draft_change(draft, "isMinimapDown", "`false`")
        minimap_event_plane.event_down.add_frontend_draft_change(draft, "isMinimapDown", "`true`")
        minimap_event_plane.event_up.add_frontend_draft_change(draft, "isMinimapDown", "`false`")
        minimap_event_plane.event_click.add_frontend_handler(_minimap_cllick_pfl)
        line_cond.bind_fields(condition="$.linePosX != `null`")

        dm.init_add_layout([
            canvas.prop(flex=1),
        ])
        self.dm = dm
        super().__init__([dm])
        self.prop(minHeight=0,
                minWidth=0,
                flexFlow="row nowrap",
                width="100%",
                height="100%",
                overflow="hidden")

    def _create_empty_vis_model(self) -> Model:
        return Model(
            (np.random.rand(240, 320, 4) * 255).astype(np.uint8),
            # img,
            1.0,
            0.0,
            0.0,
            None,
            None)

    async def set_new_tensor(self, ten: np.ndarray):
        async with self.dm.draft_update() as draft:
            draft.image = ten
            draft.scale = 1.0
            draft.scrollValueX = 0.0
            draft.scrollValueY = 0.0
            draft.linePosX = None
            draft.linePosY = None

            # draft.layout = SimpleLayout()

