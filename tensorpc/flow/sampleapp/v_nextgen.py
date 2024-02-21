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
from tensorpc.flow.flowapp.objtree import UserObjTree, find
from tensorpc.flow import observe_function

class TestNodeNode0(UserObjTree):
    def __init__(self, wh: Tuple[float, float], uid: str = "0") -> None:
        super().__init__()
        self.uid = uid
        self.wh = wh

    def func(self, a, b):
        V.points("points0", 1000).p(a, b, 1).prop(colors="red", size=5)

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
            self._childs[f"node{0}"].func(a, b)

    async def on_task_loop(self):
        async for x in self.task_loop.task_loop(list(range(10))):
            await asyncio.sleep(0.1)
            print(find(TestNodeNode0))


    @mark_create_preview_layout
    def layout_func(self):
        self.task_loop = mui.TaskLoop("dev", self.on_task_loop)
        return mui.VBox([
            mui.Button("ROOT"),
            mui.Markdown("## ROOT1"),
            self.task_loop,
        ])

class DevApp:

    @mark_create_layout
    def my_layout(self):
        root = TestNodeRoot()
        self.root = root
        canvas = plus.ComplexCanvas([
        ], init_tree_root=root, init_tree_child_accessor=lambda x: x.get_childs())
        canvas.canvas.prop(flat=True, shadows=True)
        self.canvas = canvas
        self.random_img = np.random.randint(0, 255, (128 * 16, 128 * 16, 3), dtype=np.uint8)
        return mui.VBox([
            mui.HBox([
                mui.Button("Test V", self.on_click),
                mui.Button("Test Tree", self.on_test_tree),
                mui.Button("Test custom layout", self.on_custom_gv_layout),
                mui.Button("Test gv locals", self.on_gv_locals_layout),

            ]),
            plus.InspectPanel(self, canvas).prop(width="100%", height="100%"),
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

        random_img_rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # await self.canvas.canvas.update_childs([
        #     three.Group([
        #         three.Image().prop(image=random_img)
        #     ])
        # ])
        # random_img = np.random.randint(0, 255, (128 * 16, 128 * 16, 3), dtype=np.uint8)

        with V.group("debug"):
            V.image(random_img, pos=(5, 5, 2), use_datatex=True)
        with V.group("debugX"):

            V.image(random_img_rgb, pos=(0, 5, 2), use_datatex=False)

            points = np.random.uniform(-1, 1, size=[1000, 3]).astype(np.float32)
            # V.bounding_box((1, 1, 1))
            mesh = three.Mesh([
                three.BoxGeometry(1, 1, 1),
                three.MeshBasicMaterial().prop(color="red")
            ]).prop(position=(2, 0, 0), enableSelect=True, enablePivotControl=True, pivotControlProps=three.PivotControlsCommonProps(
                        depthTest=False, annotations=True, anchor=(0, 0, 0)))
            mesh.event_change.on(lambda x: print(x))

            V.three_ui(mesh)
            box = three.BoundingBox([2, 2, 2]).prop(position=(4, 0, 0), enableSelect=True, enablePivotControl=True, pivotControlProps=three.PivotControlsCommonProps(
                        depthTest=False, annotations=True, anchor=(0, 0, 0)))
            box.event_change.on(lambda x: print(x))

            V.three_ui(box)

        # await self.canvas._unknown_visualization("foo.bar", points)

    async def on_gv_locals_layout(self):
        a = np.array([1, 2, 3])
        b = np.array([1, np.nan, 3])
        c = np.zeros([1, 3, 224, 224], np.int64)
        await self.canvas.update_locals()