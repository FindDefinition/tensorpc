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
from typing import Any, Callable, Coroutine, Dict, Hashable, Iterable, List, Literal, Optional, Set, Tuple, Type, Union

import numpy as np
from tensorpc.core.tree_id import UniqueTreeIdForTree

from tensorpc.flow import marker
from tensorpc.flow.flowapp.appcore import find_component_by_uid_with_type_check
from tensorpc.flow.flowapp.components import mui, three
from tensorpc.flow.flowapp.components.plus.config import ConfigPanel, ConfigPanelV2
from tensorpc.flow.flowapp.core import FrontendEventType
from tensorpc.flow.flowapp.coretypes import TreeDragTarget
from tensorpc.flow.flowapp import colors
from tensorpc.flow.jsonlike import TreeItem
from tensorpc.utils.registry import HashableSeqRegistryKeyOnly
from tensorpc.flow.flowapp.components.core import get_tensor_container

UNKNOWN_VIS_REGISTRY: HashableSeqRegistryKeyOnly[Callable[[Any, str, "SimpleCanvas"], Coroutine[None, None, bool]]] = HashableSeqRegistryKeyOnly()


def _try_cast_to_point_cloud(obj: Any):
    tc = get_tensor_container(obj)
    if tc is None:
        return None 

    ndim = obj.ndim
    if ndim == 2:
        dtype = tc.dtype
        if dtype == np.float32 or dtype == np.float16 or dtype == np.float64:
            num_ft = obj.shape[1]
            if num_ft >= 3 and num_ft <= 4:
                return tc.numpy()
    return None


def _try_cast_to_box3d(obj: Any):
    tc = get_tensor_container(obj)
    if tc is None:
        return None 
    ndim = obj.ndim
    if ndim == 2:
        dtype = tc.dtype
        if dtype == np.float32 or dtype == np.float16 or dtype == np.float64:
            num_ft = obj.shape[1]
            if num_ft == 7:
                return tc.numpy()
    return None


def _try_cast_to_lines(obj: Any):
    tc = get_tensor_container(obj)
    if tc is None:
        return None 
    ndim = obj.ndim
    if ndim == 3:
        dtype = tc.dtype
        if dtype == np.float32 or dtype == np.float16 or dtype == np.float64:
            if obj.shape[1] == 2 and obj.shape[2] == 3:
                return tc.numpy()
    return None


def _try_cast_to_image(obj: Any):
    tc = get_tensor_container(obj)
    if tc is None:
        return None 
    ndim = obj.ndim
    valid = False
    is_rgba = False
    if ndim == 2:
        valid = tc.dtype == np.uint8
    elif ndim == 3:
        valid = tc.dtype == np.uint8 and (obj.shape[2] == 3
                                           or obj.shape[2] == 4)
        is_rgba = obj.shape[2] == 4
    if valid:
        res = tc.numpy()
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
    encode_method: Literal["none", "int16"] = "none"
    encode_scale: mui.NumberType = dataclasses.field(default=50,
                                    metadata=ConfigPanel.slider_meta(25, 100))

@dataclasses.dataclass
class BoxCfg:
    edge_width: float = dataclasses.field(default=1,
                                          metadata=ConfigPanel.slider_meta(
                                              1, 5))
    add_cross: bool = True
    opacity: float = dataclasses.field(default=0.2,
                                          metadata=ConfigPanel.slider_meta(
                                              0.0, 1.0))


@dataclasses.dataclass
class GlobalCfg:
    background: mui.ControlColorRGBA
    enable_perf: bool = dataclasses.field(
        default=False,
        metadata=ConfigPanel.base_meta(alias="Enable Perf"))


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
            transparent_canvas: bool = False,
            init_canvas_childs: Optional[List[three.ThreeComponentType]] = None,
            key: str = "canvas",
            sync_canvases: Optional[Set["SimpleCanvas"]] = None):
        if camera is None:
            camera = three.PerspectiveCamera(fov=75, near=0.1, far=1000)
        self.camera = camera
        # self.ctrl = three.FirstPersonControl().prop(makeDefault=True,
        #                                             enabled=True, activeLook=True, constrainVertical=False, 
        #                                             autoForward=False, heightCoef=1, heightMin=0, heightMax=1,
        #                                             lookVertical=True, lookSpeed=0.005, movementSpeed=1, verticalMax=np.pi, verticalMin=0,)
        # self.ctrl = three.PointerLockControl().prop(
        #                                             enabled=True, makeDefault=True)
        self.ctrl = three.CameraControl().prop(makeDefault=True)
        
        infgrid = three.InfiniteGridHelper(5, 50, "gray")
        self.axis_helper = three.AxesHelper(20)
        self.infgrid = infgrid
        self._is_transparent = transparent_canvas
        self._dynamic_grid = three.Group([infgrid, self.axis_helper])
        gcfg = GlobalCfg(mui.ControlColorRGBA(255, 255, 255, 1))
        self.gcfg = gcfg
        self.cfg = CanvasGlobalCfg(PointCfg(), BoxCfg(), gcfg, CameraCfg())
        self._dynamic_pcs = three.Group({})
        self._dynamic_lines = three.Group({})
        self._dynamic_images = three.Group({})
        self._dynamic_boxes = three.Group({})
        self._dynamic_custom_objs = three.Group({})
        self._dynamic_voxels = three.Group({})

        self._screen_shot = three.ScreenShot(self._on_screen_shot_finish)
        self._screen_shot_v2 = three.ScreenShotSyncReturn()
        self.background_img = mui.Image()

        self._screenshot_callback = screenshot_callback
        if init_canvas_childs is None:
            init_canvas_childs = []
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
            self._dynamic_voxels,
            self._dynamic_custom_objs,

            # three.GizmoHelper().prop(alignment="bottom-right", renderPriority=1),
            *init_canvas_childs,

        ]
        # if with_grid:
        #     canvas_layout.append(infgrid)
        self._ctrl_container = mui.Fragment([])
        self.canvas = three.Canvas(canvas_layout).prop(
            flex=1, allowKeyboardEvent=True)
        if not self._is_transparent:
            self.canvas.prop(threeBackgroundColor="#ffffff")
        self._point_dict: Dict[str, three.Points] = {}
        self._image_dict: Dict[str, three.Image] = {}
        self._segment_dict: Dict[str, three.Segments] = {}
        self._box_dict: Dict[str, three.BoundingBox] = {}
        self._voxels_dict: Dict[str, three.VoxelMesh] = {}

        self._random_colors: Dict[str, str] = {}

        self._dnd_trees: Set[str] = set()
        # for find_component selection
        self.key = key
        if sync_canvases is None:
            sync_canvases = set()
        self._sync_canvases: Set[SimpleCanvas] = sync_canvases
        if len(sync_canvases) > 0:
            self.ctrl.event_change.on(self._sync_camera_ctrl).configure(throttle=100)

        super().__init__()
        self.init_add_layout([*self._layout_func()])

    def __get_cfg_panel(self):
        _cfg_panel = ConfigPanelV2(self.cfg, self._on_cfg_change)
        _cfg_panel.prop(border="1px solid",
                             borderColor="gray",
                             backgroundColor="white",
                            #  collapsed=True,
                            #  title="configs",
                             marginLeft="5px",
                             width="400px",
                             height="300px")
        return _cfg_panel

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
            all_childs = self._dynamic_boxes._get_uid_encoded_to_comp_dict()
            for v in all_childs.values():
                if isinstance(v, three.BoundingBox):
                    ev += v.update_event(edgeWidth=value)
            await self.send_and_wait(ev)
        elif uid == "box.opacity":
            ev = mui.AppEvent("", {})
            all_childs = self._dynamic_boxes._get_uid_encoded_to_comp_dict()
            for v in all_childs.values():
                if isinstance(v, three.BoundingBox):
                    ev += v.update_event(opacity=value)
            await self.send_and_wait(ev)
        elif uid == "box.add_cross":
            ev = mui.AppEvent("", {})
            all_childs = self._dynamic_boxes._get_uid_encoded_to_comp_dict()
            for v in all_childs.values():
                if isinstance(v, three.BoundingBox):
                    ev += v.update_event(addCross=value)
            await self.send_and_wait(ev)
        elif uid == "camera.keyboard_mode":
            if value == CamCtrlKeyboardMode.Helicopter:
                await self.send_and_wait(
                    self.ctrl.update_event(keyboardFront=False))
            elif value == CamCtrlKeyboardMode.Fly:
                await self.send_and_wait(
                    self.ctrl.update_event(keyboardFront=True))
        elif uid == "canvas.background":
            if not self._is_transparent:
                color_str = f"rgb({value.r}, {value.g}, {value.b})"
                await self.canvas.send_and_wait(
                    self.canvas.update_event(threeBackgroundColor=color_str))
        elif uid == "canvas.enable_perf":
            await self.canvas.send_and_wait(
                self.canvas.update_event(enablePerf=value))
        elif uid == "camera.move_speed":
            await self.canvas.send_and_wait(
                self.ctrl.update_event(keyboardMoveSpeed=value / 1000))
        elif uid == "camera.elevate_speed":
            await self.canvas.send_and_wait(
                self.ctrl.update_event(keyboardElevateSpeed=value / 1000))

    @marker.mark_create_layout
    def _layout_func(self):
        help_string = (f"Keyboard\n"
                       f"WSAD: move camera\n"
                       f"Z: descend camera\n"
                       f"SpaceBar: ascend camera\n"
                       f"use dolly (wheel) to\n"
                       f"simulate first-persion")

        layout: mui.LayoutType = [
            self.canvas.prop(zIndex=1),
            mui.HBox([
                mui.VBox([
                    mui.ToggleButton("switch",
                                     icon=mui.IconType.SwapVert,
                                     callback=self._on_pan_to_fwd).prop(
                                         selected=True,
                                         size="small",
                                         tooltip="Mouse Right Button Mode",
                                         tooltipPlacement="right"),
                    mui.ToggleButton("enableGrid",
                                     icon=mui.IconType.Grid3x3,
                                     callback=self._on_enable_grid).prop(
                                         selected=True,
                                         size="small",
                                         tooltip="Enable Grid",
                                         tooltipPlacement="right"),
                    mui.ToggleButton("enableCfgPanel",
                                     icon=mui.IconType.Settings,
                                     callback=self._on_enable_cfg_panel).prop(
                                         selected=False,
                                         size="small",
                                         tooltip="Enable Config Panel",
                                         tooltipPlacement="right"),

                    mui.IconButton(mui.IconType.Clear,
                                   callback=self._on_clear).prop(
                                       tooltip="Clear",
                                       tooltipPlacement="right"),
                    mui.IconButton(mui.IconType.Refresh,
                                   callback=self._on_reset_cam).prop(
                                       tooltip="Reset Camera",
                                       tooltipPlacement="right"),
                ]),
                # self._cfg_panel,
                self._ctrl_container,
            ]).prop(position="absolute", top=3, left=3, zIndex=5, maxHeight="10%"),
            mui.IconButton(mui.IconType.Help,
                           lambda: None).prop(tooltip=help_string,
                                              position="absolute",
                                              tooltipMultiline=True,
                                              top=3,
                                              right=3,
                                              zIndex=5),
            self.background_img.prop(position="absolute",
                                     top=0,
                                     left=0,
                                     width="100%",
                                     height="100%")
        ]
        # layout: mui.LayoutType = [
        #     self._cfg_panel
        # ]
        # TODO this should be put on mount and remove on unmount
        self.event_drop.on(self._on_drop)
        self.prop(
            minHeight=0,
            minWidth=0,
            flex=1,
            position="relative",
            droppable=True,
            width="100%",
            height="100%",
            overflow="hidden",
            border="4px solid transparent",
            sxOverDrop={"border": "4px solid green"},
        )
        return layout
    
    async def set_transparent(self, is_transparent: bool):
        if is_transparent:
            await self.canvas.send_and_wait(
                self.canvas.update_event(threeBackgroundColor=mui.undefined))
        else:
            await self.canvas.send_and_wait(
                self.canvas.update_event(threeBackgroundColor="#ffffff"))

    @staticmethod 
    def register_unknown_vis_handler(key: Type):
        """register a handler for unknown vis. the handle must be a 
        function with (obj, uid) argument.
        """
        return UNKNOWN_VIS_REGISTRY.register(key)
    
    @staticmethod 
    def get_tensor_container(obj: Any):
        return get_tensor_container(obj)
        
    async def register_cam_control_event_handler(self,
                                           handler: Callable[[Any],
                                                             mui.CORO_NONE],
                                           throttle: int = 100,
                                           debounce: Optional[int] = None):
        self.ctrl.event_change.on(handler).configure(throttle=throttle,
                                         debounce=debounce)
        await self.ctrl.sync_used_events()

    async def clear_cam_control_event_handler(self):
        self.ctrl.remove_event_handlers(self.ctrl.event_change.event_type)
        await self.ctrl.sync_used_events()

    async def _on_enable_grid(self, selected):
        if selected:
            await self._dynamic_grid.set_new_layout(
                [self.infgrid, self.axis_helper])
        else:
            await self._dynamic_grid.set_new_layout([])

    async def _on_enable_cfg_panel(self, selected):
        if selected:
            await self._ctrl_container.set_new_layout(
                [self.__get_cfg_panel()])
        else:
            await self._ctrl_container.set_new_layout([])

    async def register_sync_canvases(self, *canvas: "SimpleCanvas"):
        """add camera handler for canvas, if changed, will
        set camera pose for all canvas.
        """
        for c in canvas:
            self._sync_canvases.add(c)
        if self._sync_canvases:
            await self.register_cam_control_event_handler(self._sync_camera_ctrl)

    async def _sync_camera_ctrl(self, camdata):
        # print(camdata)
        # TODO this looks so ugly
        mat = np.array(camdata["matrixWorld"]).reshape(4, 4).T
        for canvas in self._sync_canvases:
            await canvas.set_cam2world(mat, 50)

    async def _unknown_visualization(self, tree_id: str, obj: Any, ignore_registry: bool = False):
        obj_type = type(obj)
        if obj_type in UNKNOWN_VIS_REGISTRY and not ignore_registry:
            handlers = UNKNOWN_VIS_REGISTRY[obj_type]
            for handler in handlers:
                res = await handler(obj, tree_id, self)
                if res == True:
                    return True 
        # found nothing in registry. use default one.
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
            if tree_id in self._random_colors:
                pick = self._random_colors[tree_id]
            else:
                random_colors = colors.RANDOM_COLORS_FOR_UI
                pick = random_colors[len(self._dynamic_lines) %
                                     len(random_colors)]
                self._random_colors[tree_id] = pick

            await self.show_lines(tree_id, line_obj, line_obj.shape[0], color=pick)
            return True
        return False

    async def _dnd_cb(self, uid: UniqueTreeIdForTree, data: Any):
        await self._unknown_visualization(uid.uid_encoded, data)

    async def _on_drop(self, data):
        from tensorpc.flow.flowapp.components.plus import BasicObjectTree
        if isinstance(data, TreeDragTarget):
            obj = data.obj
            success = await self._unknown_visualization(data.tree_id, obj)
            if success:
                # register to tree
                tree = find_component_by_uid_with_type_check(data.source_comp_uid, BasicObjectTree)
                if tree is not None:
                    tree._register_dnd_uid(UniqueTreeIdForTree(data.tree_id),
                                           self._dnd_cb)
                    self._dnd_trees.add(data.source_comp_uid)

    async def _on_pan_to_fwd(self, selected):
        await self.ctrl.send_and_wait(
            self.ctrl.update_event(verticalDragToForward=not selected))

    async def _on_reset_cam(self):
        await self.ctrl.reset_camera()

    async def _on_clear(self):
        from tensorpc.flow.flowapp.components.plus import BasicObjectTree

        self._point_dict.clear()
        self._segment_dict.clear()
        self._image_dict.clear()
        self._box_dict.clear()
        self._voxels_dict.clear()

        await self._dynamic_pcs.set_new_layout({})
        await self._dynamic_lines.set_new_layout({})
        await self._dynamic_images.set_new_layout({})
        await self._dynamic_boxes.set_new_layout({})
        await self._dynamic_voxels.set_new_layout({})

        for uid in self._dnd_trees:
            tree = find_component_by_uid_with_type_check(uid, BasicObjectTree)
            if tree is not None:
                tree._unregister_all_dnd_uid()
        self._dnd_trees.clear()
        self._random_colors.clear()
        await self.background_img.clear()

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
                          size: Optional[float] = None,
                          encode_method: Optional[Union[Literal["none", "int16"], mui.Undefined]] = None, 
                          encode_scale: Optional[Union[mui.NumberType, mui.Undefined]] = None,
                        attrs: Optional[Union[np.ndarray,
                                                mui.Undefined]] = None,
                        attr_fields: Optional[List[str]] = None,

                          ):
        if encode_method is None:
            encode_method = self.cfg.point.encode_method
        if encode_scale is None:
            encode_scale = self.cfg.point.encode_scale
        if key not in self._point_dict:
            if encode_method is not None:
                if attrs is None:
                    ui = three.Points(limit).prop(encodeMethod=encode_method, encodeScale=encode_scale, colorMap=three.ColorMap(min=points[:, 2].min(), max=points[:, 2].max()))
                else:
                    assert attr_fields is not None 
                    ui = three.Points(limit).prop(encodeMethod=encode_method, encodeScale=encode_scale, attrs=attrs, attrFields=attr_fields, colorMap=three.ColorMap(min=points[:, 2].min(), max=points[:, 2].max()))
            else:
                ui = three.Points(limit).prop(colorMap=three.ColorMap(min=points[:, 2].min(), max=points[:, 2].max()))
            self._point_dict[key] = ui
            await self._dynamic_pcs.update_childs({key: ui})
        point_ui = self._point_dict[key]
        await point_ui.update_points(points,
                                     colors,
                                     limit=limit,
                                     size=self.cfg.point.size if size is None else size,
                                     sizes=sizes,
                                     size_attenuation=size_attenuation,
                                     encode_method=encode_method, 
                                     encode_scale=encode_scale,
                                     attrs=attrs,
                                     attr_fields=attr_fields)
        return point_ui

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
            edge_width = self.cfg.box.edge_width
        opacity = self.cfg.box.opacity
        for i in range(len(dims)):
            if isinstance(color, list):
                cur_color = color[i]
            else:
                cur_color = color
            box_dict[str(i)] = three.BoundingBox(dims[i].tolist()).prop(
                edgeColor=cur_color,
                position=locs[i].tolist(),
                rotation=rots[i].tolist(),
                edgeWidth=edge_width,
                addCross=self.cfg.box.add_cross,
                opacity=opacity)
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

        await ui.update_lines(lines, limit=limit)

    async def clear_all_lines(self):
        # TODO currently no way to clear lines without unmount
        self._segment_dict.clear()
        await self._dynamic_lines.set_new_layout({})

    async def show_voxels(self,
                         key: str,
                         centers: np.ndarray,
                         colors: Union[np.ndarray, str],
                         size: float,
                         limit: int):
        if key not in self._voxels_dict:
            # ui = three.VoxelMesh(centers, size, limit, [
            #     three.MeshStandardMaterial().prop(vertexColors=isinstance(colors, np.ndarray), color=colors if isinstance(colors, str) else mui.undefined),
            # ], colors=colors if isinstance(colors, np.ndarray) else mui.undefined)
            ui = three.InstancedMesh(centers, limit, [
                three.BoxGeometry(size, size, size),
                three.MeshBasicMaterial().prop(vertexColors=False, color=colors if isinstance(colors, str) else mui.undefined),
            ], colors=colors if isinstance(colors, np.ndarray) else mui.undefined)
            self._voxels_dict[key] = ui
            await self._dynamic_voxels.update_childs({key: ui})
            return 
        ui = self._voxels_dict[key]
        limit_prev = ui.props.limit
        assert not isinstance(limit_prev, mui.Undefined)
        if isinstance(ui, three.InstancedMesh):
            if limit <= limit_prev:
                await ui.send_and_wait(ui.update_event(size=size, colors=colors, transforms=centers))
            else:
                await ui.send_and_wait(ui.update_event(size=size, colors=colors, transforms=centers, limit=limit))
        else:
            if limit <= limit_prev:
                await ui.send_and_wait(ui.update_event(size=size, colors=colors, centers=centers))
            else:
                await ui.send_and_wait(ui.update_event(size=size, colors=colors, centers=centers, limit=limit))

    async def clear_all_voxels(self):
        # TODO currently no way to clear lines without unmount
        self._voxels_dict.clear()
        await self._dynamic_voxels.set_new_layout({})

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
