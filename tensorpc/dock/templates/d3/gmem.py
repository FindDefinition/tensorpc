from tensorpc.apps.mls.components.global_mem import Matrix, MatrixPanel, GlobalMemContainer
from tensorpc.core.datamodel.draft import DraftBase
from tensorpc.dock import mui, three, plus, appctx, mark_create_layout
from tensorpc.core import dataclass_dispatch as dataclasses
from typing import Any 
import numpy as np

from tensorpc.dock.marker import mark_did_mount 

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class DataModel:
    data: Any
    arr: np.ndarray
    label: Any = None

@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class LocalMatrix(Matrix):
    global_indices: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)

@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class LocalMemoryModel:
    matrix: Matrix
    minimap: plus.hud.MinimapModel

class LocalMemContainer(mui.FlexBox):
    def __init__(self, draft: LocalMemoryModel, use_view: bool = False):
        assert isinstance(draft, DraftBase)
        minimap = plus.hud.MiniMap(draft.minimap, [
            MatrixPanel(draft.matrix, enable_hover_line=True)
        ])
        self.minimap = minimap
        self._draft = draft
        cam = three.OrthographicCamera(near=0.1, far=1000, children=[
            minimap,
        ]).prop(position=(0, 0, 10))
        if use_view:
            canvas = three.View([
                cam.prop(makeDefault=True),
            ]).prop(allowKeyboardEvent=True)
        else:
            canvas = three.Canvas([
                cam.prop(makeDefault=True),
            ]).prop(allowKeyboardEvent=True)
        minimap.install_canvas_events(draft.minimap, canvas)
        layout = [canvas.prop(flex=1)]
        super().__init__(layout)
        self.prop(minHeight=0,
                minWidth=0,
                flexFlow="row nowrap",
                width="100%",
                height="100%",
                overflow="hidden")

class App:
    @mark_create_layout
    def my_layout(self):
        cam = three.PerspectiveCamera(fov=75, near=0.1, far=1000).prop(position=(0, 0, 5))
        mat = LocalMatrix.from_shape("A", [16, 16])
        mat_vis_wh = mat.get_vis_wh()
        model = LocalMemoryModel(matrix=mat,
                                 minimap=plus.hud.MinimapModel(mat_vis_wh[0], mat_vis_wh[1]))
        dm = mui.DataModel(model, [])
        draft = dm.get_draft()
        global_matrix_panel = LocalMemContainer(draft)
        dm.init_add_layout([
            global_matrix_panel
        ])
        return mui.HBox([
            dm
        ]).prop(minHeight=0,
                minWidth=0,
                flexFlow="row nowrap",
                width="100%",
                height="100%",
                overflow="hidden")

class DevApp:
    @mark_create_layout
    def my_layout(self):
        self.gmem_container = GlobalMemContainer({
            "a": np.random.rand(64, 128),
            "b": np.random.rand(128, 64),
            "c": np.random.rand(128, 128),
        })
        return self.gmem_container

    # @mark_did_mount
    # async def _on_mount(self):
    #     print("ASFASFAS")
    #     await self.gmem_container.set_matrix_dict()