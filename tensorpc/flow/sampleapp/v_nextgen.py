import asyncio
from pathlib import Path
from typing import Optional

import aiohttp
from tensorpc.flow import mui, three, plus, mark_create_layout, appctx
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
            ]).prop(receiveShadow=True, position=(0.0, 0.0, 0)),

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

        res.canvas.prop(flat=False, shadows=True)
        return res 

