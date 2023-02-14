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
from tensorpc.flow.flowapp.components import mui, three
from typing import Dict, Iterable, Optional, Union, List
import numpy as np 


class SimpleCanvas(mui.FlexBox):

    def __init__(self,
                 camera: Optional[three.PerspectiveCamera] = None,
                 with_grid: bool = True):
        if camera is None:
            camera = three.PerspectiveCamera(fov=75, near=0.1, far=1000)
        self.camera = camera
        self.ctrl = three.CameraControl()
        infgrid = three.InfiniteGridHelper(5, 50, "gray")
        self._dynamic_pcs = three.Group({})
        self._dynamic_lines = three.Group({})
        self._dynamic_images = three.Group({})
        self._dynamic_boxes = three.Group({})
        self._dynamic_custom_objs = three.Group({})

        canvas_layout = [ 
                self.ctrl,
                self.camera,
                self._dynamic_pcs,
                self._dynamic_lines,
                self._dynamic_images,
                self._dynamic_boxes,
                three.AxesHelper(20),
            ]
        if with_grid:
            canvas_layout.append(infgrid)

        self.canvas = three.ThreeCanvas(canvas_layout).prop(flex=1)
        layout: mui.LayoutType = [ 
            self.canvas, 
            mui.ToggleButton("wtf", icon=mui.IconType.SwapVert, callback=self._on_pan_to_fwd).prop(
                position="absolute",
                top=3,
                left=3,
                z_index=5,
                selected=True)
        ]
        self._point_dict: Dict[str, three.Points] = {}
        self._image_dict: Dict[str, three.Image] = {}
        self._segment_dict: Dict[str, three.Segments] = {}
        self._box_dict: Dict[str, three.BoundingBox] = {}

        super().__init__(layout)
        self.prop(min_height=0,
                min_width=0,
                flex=1,
                position="relative")


    async def _on_pan_to_fwd(self, selected):
        await self.ctrl.send_and_wait(self.ctrl.update_event(vertical_drag_to_forward=not selected))

    async def set_cam2world(self, cam2world: Union[List[float], np.ndarray],
                            distance: float):
        return await self.ctrl.set_cam2world(cam2world, distance)

    async def reset_camera(self):
        return await self.ctrl.reset_camera()

    async def show_points(self, key: str, points: np.ndarray, limit: int, colors: Optional[Union[np.ndarray, str]] = None):
        if key not in self._point_dict:
            self._point_dict[key] = three.Points(limit)
            await self._dynamic_pcs.set_new_layout({**self._point_dict})
        point_ui = self._point_dict[key]
        await point_ui.update_points(points, colors, limit=limit)

    async def clear_points(self, clear_keys: Optional[List[str]] = None):
        if clear_keys is None:
            clear_keys = list(self._point_dict.keys())
        for k in clear_keys:
            await self._point_dict[k].clear()

    async def clear_points_except(self, keep_keys: List[str]):
        for k in self._point_dict:
            if k not in keep_keys:
                await self._point_dict[k].clear()

    async def clear_all_points(self):
        await self.clear_points()

    async def show_boxes(self, dims: np.ndarray, locs: np.ndarray, rots: np.ndarray, color: Union[str, List[str]] = "green"):
        box_dict = {}
        for i in range(len(dims)):
            if isinstance(color, list):
                cur_color = color[i]
            else:
                cur_color = color
            box_dict[str(i)] = three.BoundingBox(dims[i].tolist()).prop(edge_color=cur_color, position=locs[i].tolist(), rotation=rots[i].tolist())
        await self._dynamic_boxes.set_new_layout({**box_dict})

    async def clear_all_boxes(self):
        await self._dynamic_boxes.set_new_layout({})

    async def show_lines(self, key: str, lines: np.ndarray, limit: int, color: str = "green"):
        if key not in self._segment_dict:
            self._segment_dict[key] = three.Segments(limit).prop(color=color)
            await self._dynamic_lines.set_new_layout({**self._segment_dict})
        ui = self._segment_dict[key]
        
        await ui.update_lines(lines)

    async def clear_all_lines(self):
        for v in self._segment_dict.values():
            await v.clear()

    async def show_image(self, key: str, image: np.ndarray,
                         position: three.Vector3Type,
                         rotation: three.Vector3Type, scale: float):
        if key not in self._image_dict:
            self._image_dict[key] = three.Image().prop(position=position,
                                                       rotation=rotation,
                                                       scale=(scale, scale,
                                                              scale))
            await self._dynamic_images.set_new_layout({**self._image_dict})
        ui = self._image_dict[key]
        await ui.send_and_wait(
            ui.update_event(position=position,
                            rotation=rotation,
                            scale=(scale, scale, scale)))
        await ui.show(image)

    async def clear_all_images(self):
        for v in self._image_dict.values():
            await v.clear()

    async def show_objects(self, objs: Dict[str, mui.Component]):
        await self._dynamic_custom_objs.update_childs(objs)

    async def remove_objects(self, keys: Iterable[str]):
        await self._dynamic_custom_objs.remove_childs_by_keys(list(keys))

