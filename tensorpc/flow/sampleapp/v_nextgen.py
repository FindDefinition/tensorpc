import asyncio
from pathlib import Path
import random
from typing import Optional

import aiohttp
from tensorpc.flow import mui, three, plus, mark_create_layout, appctx, V
import sys
from tensorpc import PACKAGE_ROOT
import numpy as np

from tensorpc.flow.marker import mark_did_mount
from tensorpc import prim
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

        res = plus.ComplexCanvas([
            three.EffectComposer([
                three.Outline().prop(blur=True, edgeStrength=100, 
                                    width=2000, visibleEdgeColor=0xfff, 
                                    hiddenEdgeColor=0xfff, blendFunction=three.BlendFunction.ALPHA),
                # three.Bloom(),
                # three.GammaCorrection(),
                three.ToneMapping().prop(mode=three.ToneMapppingMode.ACES_FILMIC),
            ]).prop(autoClear=False),
            three.Mesh([
                three.PlaneGeometry(1000, 1000),
                three.MeshStandardMaterial().prop(color="#f0f0f0"),
            ]).prop(receiveShadow=True, position=(0.0, 0.0, -0.1)),

            three.AmbientLight(intensity=0.314, color=0xffffff),
            three.PointLight().prop(position=(2, 2, 2),
                                    castShadow=True,
                                    color=0xffffff,
                                    intensity=31.14,
                                    helperSize=0.3),
            three.Mesh([
                three.BoxGeometry(),
                three.MeshStandardMaterial().prop(color="orange", transparent=True),
            ]).prop(enableSelect=True, castShadow=True, position=(-2, 1, 1), enableHover=True, 
                enablePivotControl=True,
                enablePivotOnSelected=True,
                pivotControlProps=three.PivotControlsCommonProps(depthTest=False, annotations=True, anchor=(0, 0, 0))
                ),
            three.Mesh([
                three.BoxGeometry(),
                three.MeshStandardMaterial().prop(color="orange", transparent=True),
            ]).prop(enableSelect=True, castShadow=True, position=(-2, 3, 1), enableHover=True, 
                enablePivotControl=True,
                enablePivotOnSelected=True,
                pivotControlProps=three.PivotControlsCommonProps(depthTest=False, annotations=True, anchor=(0, 0, 0))
                ),
            three.Mesh([
                three.BoxGeometry(),
                three.MeshStandardMaterial().prop(color="orange", transparent=True),
            ]).prop(enableSelect=True, castShadow=True, position=(-2, 5, 1), enableHover=True, 
                enablePivotControl=True,
                enablePivotOnSelected=True,
                pivotControlProps=three.PivotControlsCommonProps(depthTest=False, annotations=True, anchor=(0, 0, 0))
                ),

        ])

        res.canvas.prop(flat=True, shadows=True)
        self.canvas = res
        return mui.VBox([
            mui.Button("Click me", self.on_click),

            res,
        ]) 

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
            V.points("points0", 1000).p(1, 1, 1).p(0, 0, 0).color("red").size(5)
            V.lines("lines0", 1000).p(2, 2, 2, 3, 4, 3)
            def wtfrtx(a: V.Annotated[float, V.RangedFloat(0, 10, 0.1)] = 5):
                V.points('points0', 1000).p(a, a, a).color("blue").size(5)
            V.program("wtfrtx", wtfrtx)
            random_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            V.image(random_img, pos=(5, 5, 2))
            with V.group("box0", (1, 3, 1)):
                V.bounding_box((1, 1, 1))
                V.text("WTF")
            with V.group("box_with_table", (0, 0, 3)):
                for i in range(random.randint(3, 7)):
                    tdata = {
                        "score": random.random(),
                        "name": f"test{i}",
                    }
                    V.bounding_box((1, 1, 1), pos=(0, i * 1.5, 0)).tdata(tdata)
            points = np.random.uniform(-1, 1, size=[1000, 3]).astype(np.float32)

        # await self.canvas._unknown_visualization("foo.bar", points)

