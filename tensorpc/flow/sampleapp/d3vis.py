# Copyright 2023 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tensorpc.flow import mui, three, plus, EditableLayoutApp, mark_create_layout
import numpy as np


class MyApp(EditableLayoutApp):

    @mark_create_layout
    def my_layout(self):
        cam = three.PerspectiveCamera(True, fov=75, near=0.1, far=1000)

        self.canvas = plus.SimpleCanvas(cam)
        self.slider = mui.Slider("Slider",
                                 0,
                                 1,
                                 1,
                                 callback=self._on_slider_select)

        return [
            mui.HBox([
                mui.Button("Change Slider Range",
                           self._on_slider_range_change),
                self.slider.prop(flex=1),
            ]),
            self.canvas.prop(flex=1),
        ]

    async def _on_slider_range_change(self):
        await self.slider.update_ranges(0, 10, 1)

    async def _on_slider_select(self, value):
        print("select slider!", value)
        # you need to specify a key for a group of point
        # you also need to specify number limit of current point
        points = np.random.uniform(-1, 1, size=[1000, 3])
        # colors can be:
        # 1. [N, 3] float, value range: [0, 1]
        # 2. [N], int8 (intensity), value range: [0, 255]
        # 3. a color string, e.g. red, green
        colors = np.random.uniform(0, 1, size=[1000, 3])

        await self.canvas.show_points("key0",
                                      points,
                                      limit=100000,
                                      colors=colors)
        # boxes: dims, locs, rots, colors (string list, don't support ndarray currently)
        dims = np.random.uniform(1, 2, size=[5, 3])
        locs = np.random.uniform(-5, 5, size=[5, 3])
        rots = np.random.uniform(-1, 1, size=[5, 3])
        rots[:, :2] = 0
        colors = ["red", "yellow", "red", "blue", "yellow"]
        await self.canvas.show_boxes(dims, locs, rots, colors)

        # lines: [N, 2, 3]
        lines = np.random.uniform(-3, 3, size=[10, 2, 3])
        await self.canvas.show_lines("key0", lines, limit=10000, color="aqua")
