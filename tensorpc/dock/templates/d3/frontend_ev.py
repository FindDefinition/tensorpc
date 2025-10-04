from tensorpc.dock import mui, three, plus, appctx, mark_create_layout
import dataclasses
from typing import Any 
import numpy as np 

@dataclasses.dataclass
class DataModel:
    data: Any
    arr: np.ndarray
    label: Any = None

class App:
    @mark_create_layout
    def my_layout(self):
        cam = three.PerspectiveCamera(fov=75, near=0.1, far=1000).prop(position=(0, 0, 5))
        boxmesh = three.Mesh([
                three.BoxGeometry(),
                three.MeshStandardMaterial().prop(color="orange"),
            ]).prop(castShadow=True, position=(0, 0, 0), rotation=(0.3, 0.3, 0.3))
        canvas = three.Canvas([
            cam,
            three.CameraControl().prop(makeDefault=True),
            three.AmbientLight(intensity=3.14),
            three.PointLight().prop(position=(13, 3, 5),
                                    castShadow=True,
                                    color=0xffffff,
                                    intensity=500),
            three.Mesh([
                three.PlaneGeometry(1000, 1000),
                three.MeshStandardMaterial().prop(color="#f0f0f0"),
            ]).prop(receiveShadow=True, position=(0.0, 0.0, -2)),

            boxmesh,
        ])
        dm = mui.DataModel(DataModel(None, np.random.uniform(0, 1, size=[100])), [])
        draft = dm.get_draft()
        jv = mui.JsonViewer()
        boxmesh.event_move.add_frontend_draft_change(draft, "data")
        boxmesh.event_leave.add_frontend_draft_set_none(draft, "data")
        label_box = mui.VBox([
            mui.JsonViewer().bind_fields(data="data.pointer"),
            # mui.Markdown("### Hello!")
        ]).prop(width="300px", position="absolute", backgroundColor="rgba(255, 255, 255, 0.5)", pointerEvents="none")
        label_box.bind_fields(top="not_null(data.offset[1], 0) + 5", left="not_null(data.offset[0], 0) + 5")
        label = mui.MatchCase.binary_selection(True, label_box)
        label.bind_fields(condition="data is not None")
        dm.init_add_layout([
            canvas.prop(flex=2, shadows=True),
            mui.HBox([
                jv
            ]).prop(flex=1, overflow="auto"),
            label,
        ])
        jv.bind_fields(data="data.offset")
        return mui.HBox([
            dm
        ]).prop(minHeight=0,
                minWidth=0,
                width="100%",
                height="100%",
                overflow="hidden",
                position="relative")
