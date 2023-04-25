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
import enum
import inspect
import urllib.request
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

from tensorpc.core.moduleid import get_qualname_of_type
from tensorpc.flow import marker
from tensorpc.flow.flowapp.appcore import find_component_by_uid
from tensorpc.flow.flowapp.components import mui, three
from tensorpc.flow.flowapp.components.plus.common import CommonQualNames
from tensorpc.flow.flowapp.components.plus.config import ConfigPanel
from tensorpc.flow.flowapp.core import FrontendEventType
from tensorpc.flow.flowapp.coretypes import TreeDragTarget
from tensorpc.flow.flowapp import colors
from tensorpc.flow.jsonlike import TreeItem

def _try_cast_tensor_dtype(obj: Any):
    try:
        if isinstance(obj, np.ndarray):
            return obj.dtype
        elif get_qualname_of_type(type(obj)) == CommonQualNames.TVTensor:
            from cumm.dtypes import get_npdtype_from_tvdtype
            return get_npdtype_from_tvdtype(obj.dtype)
        elif get_qualname_of_type(type(obj)) == CommonQualNames.TorchTensor:
            import torch
            _TORCH_DTYPE_TO_NP = {
                torch.float32: np.dtype(np.float32),
                torch.float64: np.dtype(np.float64),
                torch.float16: np.dtype(np.float16),
                torch.int32: np.dtype(np.int32),
                torch.int64: np.dtype(np.int64),
                torch.int8: np.dtype(np.int8),
                torch.int16: np.dtype(np.int16),
                torch.uint8: np.dtype(np.uint8),
            }
            return _TORCH_DTYPE_TO_NP[obj.dtype]
    except:
        return None


def _cast_tensor_to_np(obj: Any) -> Optional[np.ndarray]:
    if isinstance(obj, np.ndarray):
        return obj
    elif get_qualname_of_type(type(obj)) == CommonQualNames.TVTensor:
        if obj.device == 0:
            return obj.cpu().numpy()
        return obj.numpy()

    elif get_qualname_of_type(type(obj)) == CommonQualNames.TorchTensor:
        if obj.is_cuda():
            return obj.cpu().numpy()
        return obj.numpy()
    return None


def _try_cast_to_point_cloud(obj: Any):
    obj_dtype = _try_cast_tensor_dtype(obj)
    if obj_dtype is None:
        return None
    ndim = obj.ndim
    if ndim == 2:
        dtype = obj_dtype
        if dtype == np.float32 or dtype == np.float16 or dtype == np.float64:
            num_ft = obj.shape[1]
            if num_ft >= 3 and num_ft <= 4:
                return _cast_tensor_to_np(obj)
    return None


def _try_cast_to_box3d(obj: Any):
    obj_dtype = _try_cast_tensor_dtype(obj)
    if obj_dtype is None:
        return None
    ndim = obj.ndim
    if ndim == 2:
        dtype = obj_dtype
        if dtype == np.float32 or dtype == np.float16 or dtype == np.float64:
            num_ft = obj.shape[1]
            if num_ft == 7:
                return _cast_tensor_to_np(obj)
    return None


def _try_cast_to_lines(obj: Any):
    obj_dtype = _try_cast_tensor_dtype(obj)
    if obj_dtype is None:
        return None
    ndim = obj.ndim
    if ndim == 3:
        dtype = obj_dtype
        if dtype == np.float32 or dtype == np.float16 or dtype == np.float64:
            if obj.shape[1] == 2 and obj.shape[2] == 3:
                return _cast_tensor_to_np(obj)
    return None


def _try_cast_to_image(obj: Any):
    obj_dtype = _try_cast_tensor_dtype(obj)
    if obj_dtype is None:
        return None
    ndim = obj.ndim
    valid = False
    is_rgba = False
    if ndim == 2:
        valid = obj_dtype == np.uint8
    elif ndim == 3:
        valid = obj_dtype == np.uint8 and (obj.shape[2] == 3
                                           or obj.shape[2] == 4)
        is_rgba = obj.shape[2] == 4
    if valid:
        res = _cast_tensor_to_np(obj)
        if is_rgba and res is not None:
            res = res[..., :3]
        return res
    return None

class CanvasTreeItem(TreeItem):
    pass 

@dataclasses.dataclass
class PointCfg:
    size: float = dataclasses.field(default=3,
                                    metadata=ConfigPanel.slider_meta(1, 10))


@dataclasses.dataclass
class BoxCfg:
    edge_width: float = dataclasses.field(default=1,
                                          metadata=ConfigPanel.slider_meta(
                                              1, 5))
    add_cross: bool = True


@dataclasses.dataclass
class GlobalCfg:
    background: mui.ControlColorRGB


class CamCtrlKeyboardMode(enum.Enum):
    Fly = "Fly"
    Helicopter = "Helicopter"


@dataclasses.dataclass
class CameraCfg:
    keyboard_mode: CamCtrlKeyboardMode = dataclasses.field(
        default=CamCtrlKeyboardMode.Helicopter,
        metadata=ConfigPanel.base_meta(alias="Keyboard Mode"))
    move_speed: float = dataclasses.field(default=20,
                                          metadata=ConfigPanel.slider_meta(
                                              5, 40, alias="Move speed (m/s)"))
    elevate_speed: float = dataclasses.field(default=5,
                                             metadata=ConfigPanel.slider_meta(
                                                 1,
                                                 20,
                                                 alias="Elevate speed (m/s)"))


@dataclasses.dataclass
class CanvasGlobalCfg:
    point: PointCfg
    box: BoxCfg
    canvas: GlobalCfg
    camera: CameraCfg


class SimpleCanvas(mui.FlexBox):

    def __init__(
            self,
            camera: Optional[three.PerspectiveCamera] = None,
            screenshot_callback: Optional[Callable[[bytes, Any],
                                                   mui._CORO_NONE]] = None,
            transparent_canvas: bool = False):
        if camera is None:
            camera = three.PerspectiveCamera(fov=75, near=0.1, far=1000)
        self.camera = camera
        self.ctrl = three.CameraControl().prop(make_default=True)
        infgrid = three.InfiniteGridHelper(5, 50, "gray")
        self.axis_helper = three.AxesHelper(20)
        self.infgrid = infgrid
        self._is_transparent = transparent_canvas
        self._dynamic_grid = three.Group([infgrid, self.axis_helper])
        gcfg = GlobalCfg(mui.ControlColorRGB(255, 255, 255))
        self.gcfg = gcfg
        self._cfg = CanvasGlobalCfg(PointCfg(), BoxCfg(), gcfg, CameraCfg())
        self._cfg_panel = ConfigPanel(self._cfg, self._on_cfg_change)
        self._cfg_panel.prop(border="1px solid",
                             border_color="gray",
                             collapsed=True,
                             title="configs",
                             margin_left="5px")
        self._dynamic_pcs = three.Group({})
        self._dynamic_lines = three.Group({})
        self._dynamic_images = three.Group({})
        self._dynamic_boxes = three.Group({})
        self._dynamic_custom_objs = three.Group({})
        self._screen_shot = three.ScreenShot(self._on_screen_shot_finish)
        self._screen_shot_v2 = three.ScreenShotSyncReturn()
        self.background_img = mui.Image()

        self._screenshot_callback = screenshot_callback
        canvas_layout = [
            self.ctrl,
            self.camera,
            self._dynamic_pcs,
            self._dynamic_lines,
            self._dynamic_images,
            self._dynamic_boxes,
            # three.AxesHelper(20),
            self._dynamic_grid,
            # self._screen_shot,
            self._screen_shot_v2,
        ]
        # if with_grid:
        #     canvas_layout.append(infgrid)

        self.canvas = three.ThreeCanvas(canvas_layout).prop(
            flex=1, allow_keyboard_event=True)
        if not self._is_transparent:
            self.canvas.prop(three_background_color="#ffffff")
        self._point_dict: Dict[str, three.Points] = {}
        self._image_dict: Dict[str, three.Image] = {}
        self._segment_dict: Dict[str, three.Segments] = {}
        self._box_dict: Dict[str, three.BoundingBox] = {}

        self._random_colors: Dict[str, str] = {}

        self._dnd_trees: Set[str] = set()

        super().__init__()
        self.init_add_layout([*self._layout_func()])

    async def _on_screen_shot_finish(self, img_and_data: Tuple[str, Any]):
        if self._screenshot_callback:
            img = img_and_data[0]
            data = img_and_data[1]
            resp = urllib.request.urlopen(img)
            res = self._screenshot_callback(resp.read(), data)
            if inspect.iscoroutine(res):
                await res

    async def trigger_screen_shot(self, data: Optional[Any] = None):
        assert self._screenshot_callback is not None
        await self._screen_shot.trigger_screen_shot(data)

    async def get_screen_shot(self, timeout: int = 2):
        return await self._screen_shot_v2.get_screen_shot(timeout)

    async def _on_cfg_change(self, uid: str, value: Any):
        if uid == "point.size":
            ev = mui.AppEvent("", {})
            for v in self._point_dict.values():
                ev += v.update_event(size=value)
            await self.send_and_wait(ev)
        elif uid == "box.edge_width":
            ev = mui.AppEvent("", {})
            all_childs = self._dynamic_boxes._get_uid_to_comp_dict()
            for v in all_childs.values():
                if isinstance(v, three.BoundingBox):
                    ev += v.update_event(edge_width=value)
            await self.send_and_wait(ev)
        elif uid == "box.add_cross":
            ev = mui.AppEvent("", {})
            all_childs = self._dynamic_boxes._get_uid_to_comp_dict()
            for v in all_childs.values():
                if isinstance(v, three.BoundingBox):
                    ev += v.update_event(add_cross=value)
            await self.send_and_wait(ev)
        if uid == "camera.keyboard_mode":
            if value == CamCtrlKeyboardMode.Helicopter:
                await self.send_and_wait(
                    self.ctrl.update_event(keyboard_front=False))
            elif value == CamCtrlKeyboardMode.Fly:
                await self.send_and_wait(
                    self.ctrl.update_event(keyboard_front=True))
        elif uid == "canvas.background":
            if not self._is_transparent:
                color_str = f"rgb({value.r}, {value.g}, {value.b})"
                await self.canvas.send_and_wait(
                    self.canvas.update_event(three_background_color=color_str))
        elif uid == "camera.move_speed":
            await self.canvas.send_and_wait(
                self.ctrl.update_event(keyboard_move_speed=value / 1000))
        elif uid == "camera.elevate_speed":
            await self.canvas.send_and_wait(
                self.ctrl.update_event(keyboard_elevate_speed=value / 1000))

    @marker.mark_create_layout
    def _layout_func(self):
        help_string = (f"Keyboard\n"
                       f"WSAD: move camera\n"
                       f"Z: descend camera\n"
                       f"SpaceBar: ascend camera\n"
                       f"use dolly (wheel) to\n"
                       f"simulate first-persion")

        layout: mui.LayoutType = [
            self.canvas.prop(z_index=1),
            mui.HBox([
                mui.VBox([
                    mui.ToggleButton("switch",
                                     icon=mui.IconType.SwapVert,
                                     callback=self._on_pan_to_fwd).prop(
                                         selected=True,
                                         size="small",
                                         tooltip="Mouse Right Button Mode",
                                         tooltip_placement="right"),
                    mui.ToggleButton("enableGrid",
                                     icon=mui.IconType.Grid3x3,
                                     callback=self._on_enable_grid).prop(
                                         selected=True,
                                         size="small",
                                         tooltip="Enable Grid",
                                         tooltip_placement="right"),
                    mui.IconButton(mui.IconType.Clear,
                                   callback=self._on_clear).prop(
                                       tooltip="Clear",
                                       tooltip_placement="right"),
                    mui.IconButton(mui.IconType.Refresh,
                                   callback=self._on_reset_cam).prop(
                                       tooltip="Reset Camera",
                                       tooltip_placement="right"),
                ]),
                self._cfg_panel,
            ]).prop(position="absolute", top=3, left=3, z_index=5),
            mui.IconButton(mui.IconType.Help,
                           lambda: None).prop(tooltip=help_string,
                                              position="absolute",
                                              tooltip_multiline=True,
                                              top=3,
                                              right=3,
                                              z_index=5),
            self.background_img.prop(position="absolute",
                                     top=0,
                                     left=0,
                                     width="100%",
                                     height="100%")
        ]

        self.register_event_handler(FrontendEventType.Drop.value,
                                    self._on_drop)
        self.prop(
            min_height=0,
            min_width=0,
            flex=1,
            position="relative",
            droppable=True,
            width="100%",
            height="100%",
            overflow="hidden",
            border="4px solid transparent",
            sx_over_drop={"border": "4px solid green"},
        )
        return layout
    
    async def set_transparent(self, is_transparent: bool):
        if is_transparent:
            await self.canvas.send_and_wait(
                self.canvas.update_event(three_background_color=mui.undefined))
        else:
            await self.canvas.send_and_wait(
                self.canvas.update_event(three_background_color="#ffffff"))


    async def register_cam_control_event_handler(self,
                                           handler: Callable[[Any],
                                                             mui.CORO_NONE],
                                           throttle: int = 100,
                                           debounce: Optional[int] = None):
        self.ctrl.register_event_handler(self.ctrl.EvChange,
                                         handler,
                                         throttle=throttle,
                                         debounce=debounce)
        await self.ctrl.sync_used_events()

    async def clear_cam_control_event_handler(self):
        self.ctrl.remove_event_handler(self.ctrl.EvChange)
        await self.ctrl.sync_used_events()

    async def _on_enable_grid(self, selected):
        if selected:
            await self._dynamic_grid.set_new_layout(
                [self.infgrid, self.axis_helper])
        else:
            await self._dynamic_grid.set_new_layout([])

    async def _unknown_visualization(self, tree_id: str, obj: Any):
        pc_obj = _try_cast_to_point_cloud(obj)
        if pc_obj is not None:
            if tree_id in self._random_colors:
                pick = self._random_colors[tree_id]
            else:
                random_colors = colors.RANDOM_COLORS_FOR_UI
                pick = random_colors[len(self._dynamic_pcs) %
                                     len(random_colors)]
                self._random_colors[tree_id] = pick
            colors_pc: Optional[str] = None
            if pc_obj.shape[1] == 3:
                colors_pc = pick
            await self.show_points(tree_id, pc_obj.astype(np.float32),
                                   pc_obj.shape[0], colors=colors_pc)
            return True
        img_obj = _try_cast_to_image(obj)
        if img_obj is not None:
            await self.show_image(tree_id, img_obj, (0, 0, 0), (0, 0, 0), 3)
            return True
        b3d_obj = _try_cast_to_box3d(obj)
        if b3d_obj is not None:
            rots = np.array([[0, 0, float(b[-1])] for b in b3d_obj],
                            np.float32)
            if tree_id in self._random_colors:
                pick = self._random_colors[tree_id]
            else:
                random_colors = colors.RANDOM_COLORS_FOR_UI
                pick = random_colors[len(self._dynamic_boxes) %
                                     len(random_colors)]
                self._random_colors[tree_id] = pick
            await self.show_boxes(tree_id,
                                  b3d_obj[:, 3:6],
                                  b3d_obj[:, :3],
                                  rots,
                                  color=pick)
            return True
        line_obj = _try_cast_to_lines(obj)
        if line_obj is not None:
            await self.show_lines(tree_id, line_obj, line_obj.shape[0])
            return True
        return False

    async def _on_drop(self, data):
        if isinstance(data, TreeDragTarget):
            obj = data.obj
            success = await self._unknown_visualization(data.tree_id, obj)
            if success:
                # register to tree
                tree = find_component_by_uid(data.source_comp_uid)
                if tree is not None:
                    tree._register_dnd_uid(data.tree_id,
                                           self._unknown_visualization)
                    self._dnd_trees.add(data.source_comp_uid)

    async def _on_pan_to_fwd(self, selected):
        await self.ctrl.send_and_wait(
            self.ctrl.update_event(vertical_drag_to_forward=not selected))

    async def _on_reset_cam(self):
        await self.ctrl.reset_camera()

    async def _on_clear(self):
        self._point_dict.clear()
        self._segment_dict.clear()
        self._image_dict.clear()
        self._box_dict.clear()

        await self._dynamic_pcs.set_new_layout({})
        await self._dynamic_lines.set_new_layout({})
        await self._dynamic_images.set_new_layout({})
        await self._dynamic_boxes.set_new_layout({})

        for uid in self._dnd_trees:
            tree = find_component_by_uid(uid)
            if tree is not None:
                tree._unregister_all_dnd_uid()
        self._dnd_trees.clear()
        self._random_colors.clear()

    async def set_cam2world(self,
                            cam2world: Union[List[float], np.ndarray],
                            distance: float,
                            update_now: bool = False):
        return await self.ctrl.set_cam2world(cam2world,
                                             distance,
                                             update_now=update_now)

    async def reset_camera(self):
        return await self.ctrl.reset_camera()

    async def show_points(self,
                          key: str,
                          points: np.ndarray,
                          limit: int,
                          colors: Optional[Union[np.ndarray, str]] = None,
                          sizes: Optional[Union[mui.Undefined,
                                                np.ndarray]] = None,
                          size_attenuation: bool = False,
                          size: Optional[float] = None):
        if key not in self._point_dict:
            ui = three.Points(limit)
            self._point_dict[key] = ui
            await self._dynamic_pcs.update_childs({key: ui})
        point_ui = self._point_dict[key]
        await point_ui.update_points(points,
                                     colors,
                                     limit=limit,
                                     size=self._cfg.point.size if size is None else size,
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
                         key: str,
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
        if key not in self._dynamic_boxes:
            new_box = three.Group([]).prop()
            await self._dynamic_boxes.update_childs({key: new_box})
        new_box = self._dynamic_boxes[key]
        assert isinstance(new_box, three.Group)
        await new_box.set_new_layout({**box_dict})

    async def clear_all_boxes(self):
        await self._dynamic_boxes.set_new_layout({})

    async def show_lines(self,
                         key: str,
                         lines: np.ndarray,
                         limit: int,
                         color: str = "green"):
        if key not in self._segment_dict:
            ui = three.Segments(limit).prop(color=color)
            self._segment_dict[key] = ui
            await self._dynamic_lines.update_childs({key: ui})
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
            ui = three.Image().prop(position=position,
                                    rotation=rotation,
                                    scale=(scale, scale, scale))
            self._image_dict[key] = ui
            await self._dynamic_images.update_childs({key: ui})
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

    async def set_background_image(self, image: np.ndarray):
        await self.background_img.show(image)
