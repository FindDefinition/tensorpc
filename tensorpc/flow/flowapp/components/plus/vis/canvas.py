import dataclasses
import enum
import inspect
import urllib.request
from typing import Any, Callable, Coroutine, Dict, Hashable, Iterable, List, Literal, Optional, Set, Tuple, Type, Union

import numpy as np

from tensorpc.flow import marker
from tensorpc.flow.flowapp.appcore import find_component_by_uid_with_type_check
from tensorpc.flow.flowapp.components import mui, three
from tensorpc.flow.flowapp.components.plus.config import ConfigPanel
from tensorpc.flow.flowapp.components.plus.objinspect.tree import BasicObjectTree
from tensorpc.flow.flowapp.core import FrontendEventType
from tensorpc.flow.flowapp.coretypes import TreeDragTarget
from tensorpc.flow.flowapp import colors
from tensorpc.flow.jsonlike import TreeItem
from tensorpc.utils.registry import HashableSeqRegistryKeyOnly
from tensorpc.flow.flowapp.components.core import get_tensor_container
from tensorpc.flow.flowapp.components.plus.config import ConfigPanelV2
from .treeview import CanvasTreeItemHandler, lock_component

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


class ComplexCanvas(mui.FlexBox):
    """
    a blender-like canvas
    Design:
        * put controls to left as toggle buttons
        * put canvas object tree view to right
        * support helpers such as light, camera, etc.
        * support switch to camera view
    """
    def __init__(self, transparent_canvas: bool = False):

        super().__init__()
        self.component_tree = three.Fragment([])
        self.camera = three.PerspectiveCamera(fov=75, near=0.1, far=1000)

        self.ctrl = three.CameraControl().prop(makeDefault=True)
        self.background_img = mui.Image()
        self._infgrid = three.InfiniteGridHelper(5, 50, "gray")
        self._axis_helper = three.AxesHelper(20)
        gcfg = GlobalCfg(mui.ControlColorRGBA(255, 255, 255, 1))
        self.gcfg = gcfg
        self.cfg = CanvasGlobalCfg(PointCfg(), BoxCfg(), gcfg, CameraCfg())
        self._screen_shot_v2 = three.ScreenShotSyncReturn()

        self._dynamic_grid = three.Group([ self._infgrid, self._axis_helper])
        self._cfg_container = mui.Fragment([])
        self._is_transparent = transparent_canvas
        self._gizmo_helper = three.GizmoHelper().prop(alignment="bottom-right")

        init_layout = {
            "control": self.ctrl,
            "camera": self.camera,
            "grid": self._dynamic_grid,
            "screen_shot": self._screen_shot_v2,
            "gizmo": self._gizmo_helper,
        }
        for comp in init_layout.values():
            lock_component(comp)
        reserved_group = three.Group(init_layout)
        lock_component(self._axis_helper)
        lock_component(self._infgrid)
        lock_component(reserved_group)

        self.prop_container = mui.HBox([]).prop(overflow="auto",
                                                   padding="3px",
                                                   flex=1,
                                                   width="100%",
                                                   height="100%")

        self.canvas = three.Canvas({
            "reserved": reserved_group,
        }).prop(
            flex=1, allowKeyboardEvent=True)
        self.custom_tree_handler = CanvasTreeItemHandler()
        self.item_tree = BasicObjectTree(self.canvas, use_init_as_root=True, custom_tree_handler=self.custom_tree_handler, default_expand_level=1000)
        self.init_add_layout([*self._layout_func()])

    @marker.mark_create_layout
    def _layout_func(self):
        help_string = (f"Keyboard\n"
                       f"WSAD: move camera\n"
                       f"Z: descend camera\n"
                       f"SpaceBar: ascend camera\n"
                       f"use dolly (wheel) to\n"
                       f"simulate first-persion")

        canvas_layout = mui.HBox([
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

                    # mui.IconButton(mui.IconType.Clear,
                    #                callback=self._on_clear).prop(
                    #                    tooltip="Clear",
                    #                    tooltipPlacement="right"),
                    mui.IconButton(mui.IconType.Refresh,
                                   callback=self._on_reset_cam).prop(
                                       tooltip="Reset Camera",
                                       tooltipPlacement="right"),
                ]),
                # self._cfg_panel,
                self._cfg_container,
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
        ]).prop(minHeight=0,
            minWidth=0,
            flex=3,
            position="relative",
            droppable=True,
            width="100%",
            height="100%",
            overflow="hidden",
            border="2px solid transparent",
            sxOverDrop={"border": "2px solid green"},
        )
        tab_theme = mui.Theme(
            components={
                "MuiTab": {
                    "styleOverrides": {
                        "root": {
                            "padding": "0",
                            "minWidth": "28px",
                            "minHeight": "28px",
                        }
                    }
                }
            })
            

        detail_container = mui.HBox([
            mui.ThemeProvider([
                mui.Tabs([
                    mui.TabDef("",
                               "1",
                               self.prop_container,
                               icon=mui.IconType.Preview,
                               tooltip="Common Prop of Item",
                               tooltipPlacement="right"),
                    mui.TabDef("",
                               "2",
                               mui.AppTerminal(),
                               icon=mui.IconType.Terminal,
                               tooltip="app terminal (read only)",
                               tooltipPlacement="right"),
                ]).prop(panelProps=mui.FlexBoxProps(width="100%", padding=0),
                        orientation="vertical",
                        borderRight=1,
                        borderColor='divider')
            ], tab_theme)
        ]).prop(height="100%")
        self.prop(
            minHeight=0,
            minWidth=0,
                flex=1,

            width="100%",
            height="100%",
            overflow="hidden"
        )
        return [
            canvas_layout.prop(flex=3),
            mui.HBox([
                mui.Allotment([
                self.item_tree,
                detail_container,
            ]).prop(
            overflow="hidden",
            defaultSizes=[1.5, 1],
            vertical=True)
            ]).prop(flex=1)
            ]

    async def _on_pan_to_fwd(self, selected):
        await self.ctrl.send_and_wait(
            self.ctrl.update_event(verticalDragToForward=not selected))

    async def _on_enable_grid(self, selected):
        if selected:
            await self._dynamic_grid.set_new_layout(
                [self._infgrid, self._axis_helper])
        else:
            await self._dynamic_grid.set_new_layout([])


    async def _on_enable_cfg_panel(self, selected):
        if selected:
            await self._cfg_container.set_new_layout(
                [self.__get_cfg_panel()])
        else:
            await self._cfg_container.set_new_layout([])

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


    async def _on_cfg_change(self, uid: str, value: Any):
        if uid == "point.size":
            ev = mui.AppEvent("", {})
            # for v in self._point_dict.values():
            #     ev += v.update_event(size=value)
            # await self.send_and_wait(ev)
        elif uid == "box.edge_width":
            ev = mui.AppEvent("", {})
            # all_childs = self._dynamic_boxes._get_uid_to_comp_dict()
            # for v in all_childs.values():
            #     if isinstance(v, three.BoundingBox):
            #         ev += v.update_event(edgeWidth=value)
            # await self.send_and_wait(ev)
        elif uid == "box.opacity":
            ev = mui.AppEvent("", {})
            # all_childs = self._dynamic_boxes._get_uid_to_comp_dict()
            # for v in all_childs.values():
            #     if isinstance(v, three.BoundingBox):
            #         ev += v.update_event(opacity=value)
            # await self.send_and_wait(ev)
        elif uid == "box.add_cross":
            ev = mui.AppEvent("", {})
            # all_childs = self._dynamic_boxes._get_uid_to_comp_dict()
            # for v in all_childs.values():
            #     if isinstance(v, three.BoundingBox):
            #         ev += v.update_event(add_cross=value)
            # await self.send_and_wait(ev)
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

    async def _on_reset_cam(self):
        await self.ctrl.reset_camera()
