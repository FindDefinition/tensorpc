import types
from tensorpc.core import pfl
from tensorpc.dock import mui, three, plus, appctx, mark_create_layout
import dataclasses
import numpy as np
from tensorpc.dock import mui, three, plus, appctx, mark_create_layout
import dataclasses
from typing import Any, Optional 
import numpy as np
from tensorpc.dock.components.three.event import KeyboardHoldEvent, PointerEvent
from tensorpc.core.pfl.backends.js import Math, MathUtil

@dataclasses.dataclass
class Model:
    image: np.ndarray
    minimap: plus.hud.MinimapModel
    linePosX: Optional[float]
    linePosY: Optional[float]
    hover: Optional[PointerEvent] = None

    @pfl.mark_pfl_compilable
    def _img_move_pfl(self, data: PointerEvent):
        pixel_x = Math.floor((data.pointLocal[0] + 0.5) * self.image.shape[1])
        pixel_y = Math.floor((-data.pointLocal[1] + 0.5) * self.image.shape[0])
        x = (pixel_x + 0.5) - self.image.shape[1] / 2
        y = (-(pixel_y + 0.5)) + self.image.shape[0] / 2
        self.linePosX = x
        self.linePosY = y


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
            three.Text("IMAGE").prop(fontSize=20, position=(0, 0, 0.1), color="blue"),
            image,
            line_cond,
        ]).prop(position=(0, 0, 0))
        empty_model = self._create_empty_vis_model()

        dm = mui.DataModel(empty_model, [])
        draft = dm.get_draft()

        minimap = plus.hud.MiniMap(draft.minimap, [
            img_group
        ])

        cam = three.OrthographicCamera(near=0.1, far=1000, children=[
            minimap,
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
        image.event_move.add_frontend_handler(dm, Model._img_move_pfl)
        image.event_leave.add_frontend_draft_set_none(draft, "linePosX")
        image.event_leave.add_frontend_draft_set_none(draft, "linePosY")

        image.bind_fields(image="image", scale="minimap.height")

        line.bind_fields_unchecked_dict({
            "position-x": "linePosX",
            "position-y": "linePosY",
        })
        line_cond.bind_fields(condition="linePosX is not None")

        dm.init_add_layout([
            canvas.prop(flex=1),
        ])
        minimap.install_canvas_events(draft.minimap, canvas)
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
            (np.random.rand(320, 320, 4) * 255).astype(np.uint8),
            # img,
            plus.hud.MinimapModel(320, 320 + 0, fit_mode=int(plus.hud.MinimapFitMode.AUTO)),
            0.0,
            0.0,
            None)

    async def set_new_tensor(self, ten: np.ndarray):
        async with self.dm.draft_update() as draft:
            draft.image = ten
            draft.minimap.width = ten.shape[1]
            draft.minimap.height = ten.shape[0] #  + 20
            draft.minimap.scale = 1.0
            draft.minimap.scrollValueX = 0.0
            draft.minimap.scrollValueY = 0.0
            draft.hover = None
            draft.linePosX = None
            draft.linePosY = None




class App:
    @mark_create_layout
    def my_layout(self):
        self.monitor = TensorPanel()
        # self.monitor2 = mui.HBox([mui.Markdown("## PerfMonitor"),])
        # self.monitor2 = PerfMonitor(use_view=True)

        return mui.VBox([
            mui.Button("Load Trace", self._set_data),
            self.monitor.prop(flex=1),

            # three.ViewCanvas([
            #     self.monitor.prop(flex=1),
            #     # self.monitor2.prop(flex=1),

            # ]).prop(display="flex",
            #     flexDirection="column", width="100%", height="100%", overflow="hidden"),
            
        ]).prop(minHeight=0,
                minWidth=0,
                width="100%",
                height="100%",
                overflow="hidden")

    async def _set_data(self):
        img = (np.random.rand(480, 640, 4) * 255).astype(np.uint8)
        print(img.shape, img.dtype)
        await self.monitor.set_new_tensor(img)
        # await self.monitor2.append_perf_data(0, [trace_events], [None], max_depth=4)