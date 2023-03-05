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
import dataclasses
from tensorpc.flow.flowapp.components import mui, three
from tensorpc.flow import marker

from typing import Any, Dict, Iterable, Optional, Union, List
import numpy as np
from tensorpc.flow.flowapp.components.plus.objinspect.tree import TreeDragTarget
from tensorpc.flow.flowapp.components.plus.config import ConfigPanel
from tensorpc.flow.flowapp.core import FrontendEventType


def _is_point_cloud(obj: np.ndarray):
    ndim = obj.ndim
    if ndim == 2:
        dtype = obj.dtype
        if dtype == np.float32 or dtype == np.float16 or dtype == np.float64:
            num_ft = obj.shape[1]
            if num_ft >= 3 and num_ft <= 4:
                return True
    return False


def _is_array_image(obj: np.ndarray):
    ndim = obj.ndim
    if ndim == 2:
        return obj.dtype == np.uint8
    elif ndim == 3:
        return obj.dtype == np.uint8 and obj.shape[2] == 3
    return False


@dataclasses.dataclass
class PointCfg:
    size: float = dataclasses.field(default=3,
                                             metadata=ConfigPanel.slider_meta(
                                                 1, 10))


@dataclasses.dataclass
class BoxCfg:
    edge_width: float = dataclasses.field(
        default=1, metadata=ConfigPanel.slider_meta(1, 5))
    add_cross: bool = True


@dataclasses.dataclass
class CanvasGlobalCfg:
    point: PointCfg
    box: BoxCfg


class SimpleCanvas(mui.FlexBox):

    def __init__(self,
                 camera: Optional[three.PerspectiveCamera] = None,
                 with_grid: bool = True):
        if camera is None:
            camera = three.PerspectiveCamera(fov=75, near=0.1, far=1000)
        self.camera = camera
        self.ctrl = three.CameraControl()
        infgrid = three.InfiniteGridHelper(5, 50, "gray")
        self.infgrid = infgrid
        self._dynamic_grid = three.Group([infgrid])
        self._cfg = CanvasGlobalCfg(PointCfg(), BoxCfg())
        self._cfg_panel = ConfigPanel(self._cfg, self._on_cfg_change)
        self._cfg_panel.prop(border="1px solid", border_color="gray", collapsed=True, title="configs")
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
            self._dynamic_grid,
        ]
        # if with_grid:
        #     canvas_layout.append(infgrid)

        self.canvas = three.ThreeCanvas(canvas_layout).prop(flex=1)
        self._point_dict: Dict[str, three.Points] = {}
        self._image_dict: Dict[str, three.Image] = {}
        self._segment_dict: Dict[str, three.Segments] = {}
        self._box_dict: Dict[str, three.BoundingBox] = {}
        super().__init__()
        self.init_add_layout([*self._layout_func()])

    async def _on_cfg_change(self, uid: str, value: Any):
        if uid == "point.size":
            ev = mui.AppEvent("", {})
            for v in self._point_dict.values():
                ev += v.update_event(size=value)
            await self.send_and_wait(ev)
        if uid == "box.edge_width":
            ev = mui.AppEvent("", {})
            for v in self._dynamic_boxes._child_comps.values():
                if isinstance(v, three.BoundingBox):
                    ev += v.update_event(edge_width=value)
            await self.send_and_wait(ev)
        if uid == "box.add_cross":
            ev = mui.AppEvent("", {})
            for v in self._dynamic_boxes._child_comps.values():
                if isinstance(v, three.BoundingBox):
                    ev += v.update_event(add_cross=value)
            await self.send_and_wait(ev)

    @marker.mark_create_layout
    def _layout_func(self):
        layout: mui.LayoutType = [
            self.canvas,
            mui.VBox([
                mui.ToggleButton("wtf",
                                 icon=mui.IconType.SwapVert,
                                 callback=self._on_pan_to_fwd).prop(
                                     selected=True, size="small"),
                mui.ToggleButton("enableGrid",
                                 icon=mui.IconType.Grid3x3,
                                 callback=self._on_enable_grid).prop(
                                     selected=True, size="small"),
                mui.IconButton(mui.IconType.Clear, callback=self._on_clear),
            ]).prop(position="absolute", top=3, left=3, z_index=5),
            mui.VBox([
                self._cfg_panel,
            ]).prop(position="absolute", top=3, right=3, z_index=5)
        ]
        self.register_event_handler(FrontendEventType.Drop.value,
                                    self._on_drop)
        self.prop(min_height=0,
                  min_width=0,
                  flex=1,
                  position="relative",
                  droppable=True)
        return layout

    async def _on_enable_grid(self, selected):
        if selected:
            await self._dynamic_grid.set_new_layout([self.infgrid])
        else:
            await self._dynamic_grid.set_new_layout([])

    async def _on_drop(self, data):
        if isinstance(data, TreeDragTarget):
            obj = data.obj
            if isinstance(obj, np.ndarray):
                if _is_point_cloud(obj):
                    await self.show_points(data.tree_id,
                                           obj.astype(np.float32),
                                           obj.shape[0])

            print(data)
        # print(data)

    async def _on_pan_to_fwd(self, selected):
        await self.ctrl.send_and_wait(
            self.ctrl.update_event(vertical_drag_to_forward=not selected))

    async def _on_clear(self):
        self._point_dict.clear()
        self._segment_dict.clear()
        self._image_dict.clear()
        self._box_dict.clear()

        await self._dynamic_pcs.set_new_layout({})
        await self._dynamic_lines.set_new_layout({})
        await self._dynamic_images.set_new_layout({})
        await self._dynamic_boxes.set_new_layout({})

    async def set_cam2world(self, cam2world: Union[List[float], np.ndarray],
                            distance: float):
        return await self.ctrl.set_cam2world(cam2world, distance)

    async def reset_camera(self):
        return await self.ctrl.reset_camera()

    async def show_points(self,
                          key: str,
                          points: np.ndarray,
                          limit: int,
                          colors: Optional[Union[np.ndarray, str]] = None,
                          sizes: Optional[Union[mui.Undefined,
                                                np.ndarray]] = None,
                          size_attenuation: bool = False):
        if key not in self._point_dict:
            self._point_dict[key] = three.Points(limit)
            await self._dynamic_pcs.set_new_layout({**self._point_dict})
        point_ui = self._point_dict[key]
        await point_ui.update_points(points,
                                     colors,
                                     limit=limit,
                                     size=self._cfg.point.size,
                                     sizes=sizes,
                                     size_attenuation=size_attenuation)

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

    async def show_boxes(self,
                         dims: np.ndarray,
                         locs: np.ndarray,
                         rots: np.ndarray,
                         color: Union[str, List[str]] = "green",
                         edge_width: Optional[float] = None):
        box_dict = {}
        if edge_width is None:
            edge_width = self._cfg.box.edge_width
        for i in range(len(dims)):
            if isinstance(color, list):
                cur_color = color[i]
            else:
                cur_color = color
            box_dict[str(i)] = three.BoundingBox(dims[i].tolist()).prop(
                edge_color=cur_color,
                position=locs[i].tolist(),
                rotation=rots[i].tolist(),
                edge_width=edge_width,
                add_cross=self._cfg.box.add_cross)
        await self._dynamic_boxes.set_new_layout({**box_dict})

    async def clear_all_boxes(self):
        await self._dynamic_boxes.set_new_layout({})

    async def show_lines(self,
                         key: str,
                         lines: np.ndarray,
                         limit: int,
                         color: str = "green"):
        if key not in self._segment_dict:
            self._segment_dict[key] = three.Segments(limit).prop(color=color)
            await self._dynamic_lines.set_new_layout({**self._segment_dict})
        ui = self._segment_dict[key]

        await ui.update_lines(lines)

    async def clear_all_lines(self):
        # TODO currently no way to clear lines without unmount
        self._segment_dict.clear()
        await self._dynamic_lines.set_new_layout({})

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
