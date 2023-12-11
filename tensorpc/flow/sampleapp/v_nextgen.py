import asyncio
from pathlib import Path
import random
from typing import Optional, Tuple

import aiohttp
from tensorpc.flow import mui, three, plus, mark_create_layout, appctx, V, mark_create_preview_layout
import sys
from tensorpc import PACKAGE_ROOT
import numpy as np
from tensorpc.flow.flowapp.components.plus.core import ObjectGridItemConfig

from tensorpc.flow.marker import mark_did_mount
from tensorpc import prim
from tensorpc.flow.flowapp.objtree import UserObjTree
from tensorpc.flow import observe_function

class TestNodeNode0(UserObjTree):
    def __init__(self, wh: Tuple[float, float], uid: str = "0") -> None:
        super().__init__()
        self.uid = uid
        self.wh = wh

    def func(self, a, b):
        V.points("points0", 1000).p(a, b, 1).color("red").size(5)
        V.mdprint("## hello world")
        V.mdprint("## hello world")
        V.mdprint("## hello world")
        V.mdprint("## hello world")
        V.mdprint("## hello world")
        V.mdprint("## hello world")
        V.mdprint("## hello world")
        V.mdprint("## hello world")
        V.mdprint("## hello world")

    @mark_create_preview_layout
    def layout_func(self):
        res = mui.VBox([
            mui.Markdown(f"{self.uid}|`{self.wh}`")
        ])
        res.set_user_meta_by_type(ObjectGridItemConfig(self.wh[0], self.wh[1]))
        return res 


class TestNodeRoot(UserObjTree):
    def __init__(self) -> None:
        super().__init__()
        self.node0 = TestNodeNode0((0.5, 0.5), "0")
        for i in range(20):
            random_w = 1.0
            random_h = np.random.randint(1, 3) * 2 / 4

            self._childs[f"node{i}"] = TestNodeNode0((random_w, random_h), str(i))
        # self._childs["node1"] = TestNodeNode0("1")
        # self._childs["node2"] = TestNodeNode0("2")
        # self._childs["node3"] = TestNodeNode0("3")

    def func(self, a, b):
        with V.group("dev"):
            V.bounding_box((a, b, 2))
            self.node0.func(a, b)
            V.mdprint("## hello world")

    @mark_create_preview_layout
    def layout_func(self):
        return mui.VBox([
            mui.Button("ROOT"),
            mui.Markdown("## ROOT132")
        ])

class DevApp:

    @mark_create_layout
    def my_layout(self):

        # res = plus.ComplexCanvas([
        #     three.EffectComposer([
        #         three.Outline().prop(blur=True, edgeStrength=100, 
        #                                 width=1000, visibleEdgeColor=0xfff, 
        #                                 hiddenEdgeColor=0xfff, blendFunction=three.BlendFunction.ALPHA),
        #         # three.Bloom(),
        #         # three.GammaCorrection(),
        #         three.ToneMapping().prop(mode=three.ToneMapppingMode.ACES_FILMIC),
        #     ]).prop(autoClear=False),

        #     three.AmbientLight(intensity=0.314),
        #     three.PointLight().prop(position=(13, 3, 5),
        #                             castShadow=True,
        #                             color=0xffffff,
        #                             intensity=500),
        #     three.Mesh([
        #         three.BoxGeometry(),
        #         three.Edges(),
        #         three.MeshStandardMaterial().prop(color="orange", transparent=True),
        #     ]).prop(enableSelect=True, castShadow=True, position=(0, 0, 0), enableHover=True, 
        #         enablePivotControl=True,
        #         enablePivotOnSelected=True,
        #         pivotControlProps=three.PivotControlsCommonProps(depthTest=False, annotations=True, anchor=(0, 0, 0))
        #         ),
        #     three.BoundingBox((1 ,1 ,1)).prop(position=(3, 3, 3),
        #     enableSelect=True, enableSelectOutline=False),
        # ])
        root = TestNodeRoot()
        self.root = root
        res = plus.ComplexCanvas([
            three.EffectComposer([
                three.Outline().prop(blur=True, edgeStrength=100, 
                                    width=2000, visibleEdgeColor=0xfff, 
                                    hiddenEdgeColor=0xfff, blendFunction=three.BlendFunction.ALPHA),
                # three.Bloom(),
                # three.GammaCorrection(),
                three.ToneMapping().prop(mode=three.ToneMapppingMode.ACES_FILMIC),
            ]).prop(autoClear=False),
            # three.Mesh([
            #     three.PlaneGeometry(1000, 1000),
            #     three.MeshStandardMaterial().prop(color="#f0f0f0"),
            # ]).prop(receiveShadow=True, position=(0.0, 0.0, -0.1)),

            # three.AmbientLight(intensity=0.314, color=0xffffff),
            # three.PointLight().prop(position=(2, 2, 2),
            #                         castShadow=True,
            #                         color=0xffffff,
            #                         intensity=31.14,
            #                         helperSize=0.3),
            # three.Mesh([
            #     three.BoxGeometry(),
            #     three.MeshStandardMaterial().prop(color="orange", transparent=True),
            # ]).prop(enableSelect=True, castShadow=True, position=(-2, 1, 1), enableHover=True, 
            #     enablePivotControl=True,
            #     enablePivotOnSelected=True,
            #     pivotControlProps=three.PivotControlsCommonProps(depthTest=False, annotations=True, anchor=(0, 0, 0))
            #     ),
            # three.Mesh([
            #     three.BoxGeometry(),
            #     three.MeshStandardMaterial().prop(color="orange", transparent=True),
            # ]).prop(enableSelect=True, castShadow=True, position=(-2, 3, 1), enableHover=True, 
            #     enablePivotControl=True,
            #     enablePivotOnSelected=True,
            #     pivotControlProps=three.PivotControlsCommonProps(depthTest=False, annotations=True, anchor=(0, 0, 0))
            #     ),
            # three.Mesh([
            #     three.BoxGeometry(),
            #     three.MeshStandardMaterial().prop(color="orange", transparent=True),
            # ]).prop(enableSelect=True, castShadow=True, position=(-2, 5, 1), enableHover=True, 
            #     enablePivotControl=True,
            #     enablePivotOnSelected=True,
            #     pivotControlProps=three.PivotControlsCommonProps(depthTest=False, annotations=True, anchor=(0, 0, 0))
            #     ),

        ], init_tree_root=root, init_tree_child_accessor=lambda x: x.get_childs())

        res.canvas.prop(flat=True, shadows=True)
        self.canvas = res
        return mui.VBox([
            mui.HBox([
                mui.Button("Test V", self.on_click),
                mui.Button("Test Tree", self.on_test_tree),
                mui.Button("Test custom layout", self.on_custom_gv_layout),
            ]),
            res,
        ]) 
    
    async def on_test_tree(self):
        self.root.func(3, 4)

    async def on_custom_gv_layout(self):
        items = {}
        for k in range(3):
            half = random.random() > 0.5
            items[f"name{k}"] = mui.FlexBox([
                mui.Markdown(f"## hello world {k}"),
            ]).set_user_meta_by_type(ObjectGridItemConfig(1, 0.5 if half else 1))
        await self.canvas.set_new_grid_items(items, False)

    async def on_click(self):
        print("clicked")
        # with V.ctx():
        random_img = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        random_img[:, :, -1] = 255
        # await self.canvas.canvas.update_childs([
        #     three.Group([
        #         three.Image().prop(image=random_img)
        #     ])
        # ])
        with V.group("dev"):
            points = np.random.uniform(-5, 5, size=[1000, 3]).astype(np.float32)
            colors = np.random.uniform(0, 255, size=[1000]).astype(np.uint8)
            points[:, 0] -= 20

            # V.points("points0", 1000).p(1, 1, 1).p(0, 0, 0).color("red").size(5)
            # V.points("points1X", 1000).array(points).size(5)

            V.lines("lines0", 1000).p(2, 2, 2, 5, 5, 5).prop(color="blue")
            def wtfrtx(a: V.Annotated[float, V.RangedFloat(0, 10, 0.1)] = 5):
                V.points('points0', 1000).p(a, a, a).prop(colors="blue", size=5)
            V.program("wtfrtx", wtfrtx)
            random_img = np.random.randint(0, 255, (128 * 16, 128 * 16, 3), dtype=np.uint8)
            V.image(random_img, pos=(5, 5, 2), use_datatex=True)
            V.three_ui(three.BoundingBox((1, 1, 1)).prop(position=(9, 9, 3),))
            V.text("WTF1").prop(color="red")
            with V.group("box_with_table", (0, 0, 3)):
                for i in range(random.randint(3, 7)):
                    tdata = {
                        "score": random.random(),
                        "name": f"test{i}",
                    }
                    box = V.bounding_box((1, 1, 1), pos=(0, i * 1.5, 0))
                    V.set_tdata(box, tdata)

            with V.group("box0", (1, 3, 1)):
                V.bounding_box((1, 1, 1))
                V.text("WTF").prop(color="blue")

            points = np.random.uniform(-1, 1, size=[1000, 3]).astype(np.float32)

        # await self.canvas._unknown_visualization("foo.bar", points)

