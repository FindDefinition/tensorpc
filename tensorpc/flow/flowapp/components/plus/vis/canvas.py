import asyncio
import dataclasses
import enum
from functools import partial
import inspect
import re
import threading
import urllib.request
from typing import Any, Callable, Coroutine, Dict, Hashable, Iterable, List, Literal, Optional, Set, Tuple, Type, TypeVar, Union
from typing_extensions import Annotated
import numpy as np
from tensorpc.core.moduleid import get_qualname_of_type

from tensorpc.flow import marker
from tensorpc.core.tree_id import UniqueTreeId, UniqueTreeIdForTree
from tensorpc.flow.flowapp.appcore import find_component_by_uid_with_type_check
from tensorpc.flow.flowapp.components import mui, three
from tensorpc.flow.flowapp.components.plus.arraycommon import can_cast_to_np_array, try_cast_to_np_array
from tensorpc.flow.flowapp.components.plus.arraygrid import NumpyArrayGridTable
from tensorpc.flow.flowapp.components.plus.config import ConfigPanel
from tensorpc.flow.flowapp.components.plus.core import ALL_OBJECT_LAYOUT_HANDLERS, ObjectGridItemConfig
from tensorpc.flow.flowapp.components.plus.grid_preview_layout import GridPreviewLayout
from tensorpc.flow.flowapp.components.plus.objinspect.tree import BasicObjectTree, SelectSingleEvent
from .core import UNKNOWN_KEY_SPLIT, UNKNOWN_VIS_KEY, UserTreeItemCard, VContext, get_canvas_item_cfg, get_or_create_canvas_item_cfg, _VapiObjects, is_reserved_uid
from tensorpc.flow.flowapp.components.typemetas import RangedFloat
from tensorpc.flow.flowapp.core import Component, ContainerBase, FrontendEventType
from tensorpc.flow.flowapp.coretypes import TreeDragTarget
from tensorpc.flow.flowapp import appctx, colors
from tensorpc.flow.jsonlike import TreeItem
from tensorpc.utils.registry import HashableSeqRegistryKeyOnly
from tensorpc.flow.flowapp.components.core import get_tensor_container
from tensorpc.flow.flowapp.components.plus.config import ConfigPanelV2
from tensorpc.flow.flowapp.components.plus.scriptmgr import ScriptManager

from .treeview import CanvasTreeItemHandler, lock_component
from tensorpc.utils.registry import HashableSeqRegistryKeyOnly
from .core import is_reserved_name, CanvasUserTreeItem, GroupProxy

UNKNOWN_VIS_REGISTRY: HashableSeqRegistryKeyOnly[
    Callable[[Any, str, "ComplexCanvas"],
             Coroutine[None, None, bool]]] = HashableSeqRegistryKeyOnly()

T = TypeVar("T")


def _count_child_type(container: three.ContainerBase,
                      obj_type: Type[three.Component]):
    res = 0
    for v in container._child_comps.values():
        if isinstance(v, obj_type):
            res += 1
    return res


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


def find_component_trace_by_uid_with_not_exist_parts(
    comp: Component,
    uid: UniqueTreeId,
    container_cls: Tuple[Type[ContainerBase], ...] = (ContainerBase, )
) -> Tuple[List[Component], List[str], List[str]]:
    # if comp._flow_uid == uid:
    #     return [comp], []
    uid_parts = uid.parts
    # if len(uid_parts) == 0:
    #     return [comp], []
    res: List[Component] = []
    cur_comp = comp
    for i, part in enumerate(uid_parts):
        # first_part = cur_comp._flow_uid.split(".")[0]
        # if first_part != part:
        #     return res, uid_parts[i:]
        if not isinstance(cur_comp, container_cls):
            return res, uid_parts[i:], uid_parts[:i]
        if part in cur_comp._child_comps:
            cur_comp = cur_comp._child_comps[part]
        else:
            return res, uid_parts[i:], uid_parts[:i]
        res.append(cur_comp)

        # if i != len(uid_parts) - 1:
    return res, [], uid_parts


@dataclasses.dataclass
class PointCfg:
    size: Annotated[float, RangedFloat(1, 10, 0.1)] = 3
    encode_method: Literal["none", "int16"] = "none"
    encode_scale: Annotated[float, RangedFloat(25, 100, 0.1)] = 50


@dataclasses.dataclass
class BoxCfg:
    edge_width: Annotated[float, RangedFloat(1, 5, 0.1)] = 1
    add_cross: bool = True
    opacity: Annotated[float, RangedFloat(0.0, 1.0, 0.01)] = 0.2


@dataclasses.dataclass
class GlobalCfg:
    background: mui.ControlColorRGBA
    enable_perf: bool = dataclasses.field(
        default=False, metadata=ConfigPanel.base_meta(alias="Enable Perf"))


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


def _extrace_all_tree_item_via_accessor(
        root: Any, accessor: Callable[[Any], Dict[str, Any]],
        root_key: str) -> Dict[Tuple[str, ...], Any]:
    res: Dict[Tuple[str, ...], Any] = {}

    def _dfs(cur_obj: Any, cur_key: Tuple[str, ...]):
        if cur_obj is None:
            return
        res[cur_key] = cur_obj
        childs = accessor(cur_obj)
        for k, v in childs.items():
            _dfs(v, cur_key + (k, ))

    _dfs(root, (root_key, ))
    return res


_EXAMPLE_SCRIPT = f"""
from tensorpc.flow import V
import numpy as np 
with V.group("example"):
    V.text("hello world!!!").prop(color="red")
    points = np.random.uniform(-5, 5, size=[1000, 3]).astype(np.float32)
    V.points("point_key", limit=5000).array(points)
    box = V.bounding_box((3, 1, 1), pos=(0, 3, 0))
"""


class ComplexCanvasView(three.View):

    def __init__(
            self,
            key: str,
            childs: Optional[three.ThreeLayoutType] = None,
            transparent_canvas: bool = False,
            custom_effect: Optional[three.EffectComposer] = None,
            camera: Optional[Union[three.PerspectiveCamera, three.OrthographicCamera]] = None,
            control: Optional[three.CameraControl] = None,
            enable_gizmo: bool = True):
        if camera is None:
            camera = three.PerspectiveCamera(fov=75, near=0.1, far=1000)
        self.camera = camera
        self.camera.prop(layers=1 | (1 << 31)) # only objects in layer 0 and 31 is visible.
        if control is None:
            control = three.CameraControl().prop(makeDefault=True)
        self.ctrl = control
        self._infgrid = three.InfiniteGridHelper(5, 50, "gray")
        self._axis_helper = three.AxesHelper(20)

        self._dynamic_grid = three.Group([self._infgrid, self._axis_helper])

        init_layout = {
            # "camera": self.camera,
            "grid": self._dynamic_grid,
            # "utree": self._user_obj_tree_group,
            "effects": custom_effect,
        }
        gizmo_helper = three.GizmoHelper().prop(alignment="bottom-right",
                                                      renderPriority=2)
        if enable_gizmo:
            init_layout["gizmo"] = gizmo_helper
        self._lock = asyncio.Lock()
        for comp in init_layout.values():
            lock_component(comp)
        reserved_group = three.Group(init_layout)
        lock_component(self._axis_helper)
        lock_component(self._infgrid)
        lock_component(reserved_group)
        layout: three.ThreeLayoutType = {
            "reserved": reserved_group,
        }
        self.reserved_group = reserved_group
        if childs is not None:
            layout["init"] = three.Group(childs)
        self._init_layout = layout
        self._item_root = three.SelectionContext(layout,
                                                 self._on_3d_object_select).prop(useOutline=True)

        super().__init__({
            "root": self._item_root,
            "camera": self.camera,
            "control": self.ctrl,
        })
        self.prop(flex=1, allowKeyboardEvent=True, menuItems=[
                    mui.MenuItem("reset", "reset"),
                    mui.MenuItem("clear", "clear"),
                ])
        self.event_context_menu.on(self._on_menu_select)

    def _set_selection_callback(self):
        pass 

    async def _on_menu_select(self, value: str):
        if value == "reset":
            await self._on_reset_cam()
        elif value == "clear":
            await self.clear()

    async def clear(self):
        await self._item_root.set_new_layout({**self._init_layout})
        await self.item_tree.update_tree()

    async def _on_reset_cam(self):
        await self.ctrl.reset_camera()

class ComplexCanvas(mui.FlexBox):
    """
    a blender-like canvas
    Design:
        * put controls to left as toggle buttons
        * put canvas object tree view to right
        * support helpers such as light, camera, etc.
        * support switch to camera view
    """

    def __init__(
            self,
            init_canvas_childs: Optional[three.ThreeLayoutType] = None,
            transparent_canvas: bool = False,
            init_tree_root: Optional[Any] = None,
            init_tree_child_accessor: Optional[Callable[[Any],
                                                        Dict[str,
                                                             Any]]] = None,
            key: str = "canvas",
            custom_effect: Optional[three.EffectComposer] = None,
            init_enable_grid: bool = True,
            camera: Optional[Union[three.PerspectiveCamera, three.OrthographicCamera]] = None):

        super().__init__()
        self.component_tree = three.Fragment([])
        if camera is None:
            camera = three.PerspectiveCamera(fov=75, near=0.1, far=1000)
        self.camera = camera
        self._init_enable_grid = init_enable_grid
        self.camera.prop(layers=1 | (1 << 31))

        self.ctrl = three.CameraControl().prop(makeDefault=True)
        self.background_img = mui.Image()
        self._infgrid = three.InfiniteGridHelper(5, 50, "gray")
        self._axis_helper = three.AxesHelper(20)
        gcfg = GlobalCfg(mui.ControlColorRGBA(255, 255, 255, 1))
        self.gcfg = gcfg
        self.cfg = CanvasGlobalCfg(PointCfg(), BoxCfg(), gcfg, CameraCfg())
        self._screen_shot_v2 = three.ScreenShotSyncReturn()

        self._dynamic_grid = three.Group([self._infgrid, self._axis_helper])
        self._cfg_container = mui.Fragment([])
        self._is_transparent = transparent_canvas
        self._gizmo_helper = three.GizmoHelper().prop(alignment="bottom-right")
        self._cur_detail_layout_uid: Optional[UniqueTreeId] = None
        self._cur_detail_layout_object_id: Optional[UniqueTreeId] = None

        self._cur_table_uid: Optional[UniqueTreeId] = None
        self._cur_table_object_id: Optional[UniqueTreeId] = None
        self._dnd_trees: Set[str] = set()

        self._user_obj_tree_item_to_meta: Dict[int, CanvasUserTreeItem] = {}

        self._random_colors: Dict[str, str] = {}
        self._user_obj_tree_group = three.Group([])
        # self.gv_tree_layout: mui.FlexBox = mui.HBox([
        #         mui.Markdown("## Unused")
        #     ]).prop(width="100%", height="100%", overflow="auto")
        self.gv_tree_layout: GridPreviewLayout = GridPreviewLayout(
            {}, None, use_typename_as_title=True).prop(flex=1,
                                                       height="100%",
                                                       width="100%",
                                                       overflow="auto")
        self.gv_locals_layout: GridPreviewLayout = GridPreviewLayout(
            {}, None).prop(flex=1,
                           height="100%",
                           width="100%",
                           overflow="auto")
        self.gv_custom_layout: GridPreviewLayout = GridPreviewLayout(
            {}, None).prop(flex=1,
                           height="100%",
                           width="100%",
                           overflow="auto")
        self.array_table = NumpyArrayGridTable()
        self.flatted_tree_nodes: Dict[Tuple[str, ...], Any] = {}
        if init_tree_root is not None and init_tree_child_accessor is not None:
            flatted_tree_nodes = _extrace_all_tree_item_via_accessor(
                init_tree_root, init_tree_child_accessor, "root")
            self.flatted_tree_nodes = flatted_tree_nodes
            groups, self._user_obj_tree_item_to_meta = self._get_tree_cards_and_groups(
                flatted_tree_nodes)
            self._user_obj_tree_group.init_add_layout({**groups})
            # self.treeitem_container.init_add_layout({
            #     UNKNOWN_KEY_SPLIT.join(v.key): v.card
            #     for k, v in self._user_obj_tree_item_to_meta.items()
            # })
            self.gv_tree_layout = GridPreviewLayout(
                {
                    ".".join(k): v
                    for k, v in flatted_tree_nodes.items()
                },
                init_tree_root,
                use_typename_as_title=True).prop(flex=1,
                                                 height="100%",
                                                 width="100%",
                                                 overflow="auto")
        if custom_effect is None:
            custom_effect = three.EffectComposer([
                three.Outline().prop(blur=True,
                                     edgeStrength=100,
                                     width=2000,
                                     visibleEdgeColor=0xfff,
                                     hiddenEdgeColor=0xfff,
                                     blendFunction=three.BlendFunction.ALPHA),
                three.ToneMapping().prop(
                    mode=three.ToneMapppingMode.ACES_FILMIC),
            ]).prop(autoClear=False)

        init_layout = {
            # "camera": self.camera,
            "grid": self._dynamic_grid,
            "screen shot": self._screen_shot_v2,
            "gizmo": self._gizmo_helper,
            "utree": self._user_obj_tree_group,
            "effects": custom_effect,
        }
        self._lock = asyncio.Lock()
        for comp in init_layout.values():
            lock_component(comp)
        reserved_group = three.Group(init_layout)
        lock_component(self._axis_helper)
        lock_component(self._infgrid)
        lock_component(reserved_group)
        layout: three.ThreeLayoutType = {
            "reserved": reserved_group,
        }
        self.reserved_group = reserved_group
        if init_canvas_childs is not None:
            layout["init"] = three.Group(init_canvas_childs)
        self._init_layout = layout
        self._item_root = three.SelectionContext(layout,
                                                 self._on_3d_object_select).prop(useOutline=False)
        # self._item_root = three.Group(layout)
        self.key = key
        self.prop_container = mui.HBox([]).prop(overflow="auto",
                                                padding="3px",
                                                flex=1,
                                                width="100%",
                                                height="100%")
        self.tdata_container = mui.HBox([]).prop(overflow="auto",
                                                 padding="3px",
                                                 flex=1,
                                                 width="100%",
                                                 height="100%")
        self.tdata_container_v2 = mui.HBox([]).prop(overflow="auto",
                                                    padding="3px",
                                                    flex=1,
                                                    width="100%",
                                                    height="100%")

        self.canvas = three.Canvas({
            "root": self._item_root,
            "camera": self.camera,
            "control": self.ctrl,
        }).prop(flex=1, allowKeyboardEvent=True, shadows=True, menuItems=[
            mui.MenuItem("reset", "reset"),
            mui.MenuItem("clear", "clear"),
        ])
        self.canvas.event_context_menu.on(self._on_menu_select)
        self.custom_tree_handler = CanvasTreeItemHandler()
        self.item_tree = BasicObjectTree(
            self._item_root,
            use_init_as_root=True,
            custom_tree_handler=self.custom_tree_handler,
            default_expand_level=1000,
            use_fast_tree=True)
        self.item_tree.event_async_select_single.on(self._on_tree_select)
        self.init_add_layout([*self._layout_func()])
        self.prop(
            droppable=True,
            border="4px solid transparent",
            sxOverDrop={"border": "4px solid green"},
        )


    @marker.mark_create_layout
    def _layout_func(self):
        help_string = (f"Keyboard\n"
                       f"WSAD: move camera\n"
                       f"Z: descend camera\n"
                       f"SpaceBar: ascend camera\n"
                       f"use dolly (wheel) to\n"
                       f"simulate first-persion")
        bottom_pane_visible = len(self.flatted_tree_nodes) > 0
        bottom_pane_visible = self._init_enable_grid
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
                    mui.ToggleButton("enableTData",
                                     icon=mui.IconType.DataArray,
                                     callback=self._on_enable_tdata_pane).prop(
                                         selected=bottom_pane_visible,
                                         size="small",
                                         tooltip="Enable Data Grid Pane",
                                         tooltipPlacement="right"),
                    mui.IconButton(mui.IconType.Clear,
                                   callback=self.clear).prop(
                                       tooltip="Clear",
                                       tooltipPlacement="right"),
                    mui.IconButton(mui.IconType.Refresh,
                                   callback=self._on_reset_cam).prop(
                                       tooltip="Reset Camera",
                                       tooltipPlacement="right"),
                ]).prop(backgroundColor="lavender", borderRadius="4px"),
                # self._cfg_panel,
                self._cfg_container,
            ]).prop(position="absolute",
                    top=3,
                    left=3,
                    zIndex=5),
            mui.HBox([
                mui.IconButton(mui.IconType.Help, lambda: None).prop(
                    tooltip=help_string,
                    tooltipMultiline=True,
                ),
            ]).prop(top=3, position="absolute", right=3, zIndex=5),
            self.background_img.prop(position="absolute",
                                     top=0,
                                     left=0,
                                     width="100%",
                                     height="100%")
        ]).prop(
            minHeight=0,
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
                               tooltip="Common Prop of Item"),
                    mui.TabDef("",
                               "2",
                               self.tdata_container,
                               icon=mui.IconType.DataObject,
                               tooltip="Data Table For Group"),
                    mui.TabDef("",
                               "3",
                               mui.AppTerminal(),
                               icon=mui.IconType.Terminal,
                               tooltip="app terminal (read only)"),
                ]).prop(panelProps=mui.FlexBoxProps(width="100%", padding=0),
                        orientation="vertical",
                        tooltipPlacement="right",
                        borderRight=1,
                        borderColor='divider')
            ], tab_theme)
        ]).prop(height="100%")

        bottom_container = mui.HBox([
            mui.ThemeProvider([
                mui.Tabs([
                    mui.TabDef("",
                               "1",
                               self.tdata_container_v2,
                               icon=mui.IconType.DataObject,
                               tooltip="Data Table For Group"),
                    mui.TabDef("",
                               "2",
                               self.gv_tree_layout,
                               icon=mui.IconType.Menu,
                               tooltip="Preview Grid of tree"),
                    mui.TabDef("",
                               "3",
                               self.gv_locals_layout,
                               icon=mui.IconType.Dashboard,
                               tooltip="Frame Locals Grid"),
                    mui.TabDef("",
                               "4",
                               self.gv_custom_layout,
                               icon=mui.IconType.DashboardCustomize,
                               tooltip="Custom Grid"),
                    mui.TabDef("",
                               "5",
                               self.array_table,
                               icon=mui.IconType.Dataset,
                               tooltip="numpy array table"),

                    mui.TabDef(
                        "",
                        "6",
                        ScriptManager(init_scripts={
                            "python": _EXAMPLE_SCRIPT
                        }),
                        icon=mui.IconType.Code,
                        tooltip="python script playground"),
                ], "2").prop(panelProps=mui.FlexBoxProps(width="100%",
                                                         padding=0),
                             orientation="vertical",
                             borderRight=1,
                             tooltipPlacement="right",
                             borderColor='divider')
            ], tab_theme)
        ]).prop(height="100%", width="100%")

        self.prop(minHeight=0,
                  minWidth=0,
                  flex=1,
                  width="100%",
                  height="100%",
                  overflow="hidden")
        # self.item_tree.event_
        self.tdata_container_v2_pane = mui.Allotment.Pane(
            bottom_container, preferredSize="40%", visible=bottom_pane_visible)
        self._canvas_spitter = mui.Allotment(
            mui.Allotment.ChildDef([
                mui.Allotment.Pane(canvas_layout, preferredSize="60%"),
                self.tdata_container_v2_pane,
            ])).prop(vertical=True, proportionalLayout=True)
        self.event_drop.on(self._on_drop)

        return [
            mui.Allotment([
                self._canvas_spitter,
                mui.HBox([
                    mui.Allotment([
                        self.item_tree.prop(width="100%", height="100%"),
                        detail_container,
                    ]).prop(overflow="hidden",
                            defaultSizes=[1.5, 1],
                            vertical=True)
                ]).prop(flex=1, width="100%", height="100%")
            ]).prop(overflow="hidden", defaultSizes=[3, 1], vertical=False)
        ]

    async def _on_menu_select(self, value: str):
        if value == "reset":
            await self._on_reset_cam()
        elif value == "clear":
            await self.clear()

    async def set_new_tree_root(self, tree_root: T,
                                tree_child_accessor: Callable[[T], Dict[str,
                                                                        Any]]):
        flatted_tree_nodes = _extrace_all_tree_item_via_accessor(
            tree_root, tree_child_accessor, "root")
        self.flatted_tree_nodes = flatted_tree_nodes
        groups, self._user_obj_tree_item_to_meta = self._get_tree_cards_and_groups(
            flatted_tree_nodes)
        await self._user_obj_tree_group.set_new_layout({**groups})
        self.gv_tree_layout.set_tree_root(tree_root)
        await self.gv_tree_layout.set_new_items({
            ".".join(k): v
            for k, v in flatted_tree_nodes.items()
        })

    async def set_new_grid_items(self, items: Dict[str, Any], is_local: bool = False):
        if is_local:
            await self.gv_locals_layout.set_new_items(items)
        else:
            await self.gv_custom_layout.set_new_items(items)

    async def update_grid_items(self, items: Dict[str, Any], is_local: bool = False):
        if is_local:
            await self.gv_locals_layout.update_items(items)
        else:
            await self.gv_custom_layout.update_items(items)

    async def delete_grid_items(self, items: List[str], is_local: bool = False):
        if is_local:
            await self.gv_locals_layout.delete_items(items)
        else:
            await self.gv_custom_layout.delete_items(items)

    async def clear_grid(self, is_local: bool = False):
        if is_local:
            await self.gv_locals_layout.clear_items()
        else:
            await self.gv_custom_layout.clear_items()

    def _tree_collect_in_vctx(self):
        for k1, meta in self._user_obj_tree_item_to_meta.items():
            group_childs = {}
            for k2, v in meta.vctx._name_to_group.items():
                cfg = get_canvas_item_cfg(v)
                # print(k, cfg )
                if cfg is not None:
                    proxy = cfg.proxy
                    if proxy is not None:
                        assert isinstance(proxy, GroupProxy)
                        for c in proxy.childs.values():
                            c_cfg = get_canvas_item_cfg(c)
                            assert c_cfg is not None
                            c_proxy = c_cfg.proxy
                            assert c_proxy is not None
                            # TODO
                            if isinstance(c, _VapiObjects):
                                c.prepare_vapi_props()
                        if v is not meta.vctx.root:
                            assert not v.is_mounted(), f"{type(v)}"
                            v.init_add_layout(proxy.childs)
                        else:
                            group_childs.update(proxy.childs)
                        proxy.childs.clear()
            for container, (group, name) in meta.vctx._group_assigns.items():
                assert container is meta.vctx.root
                group_childs[name] = group
            meta.vctx.clear()
            meta.childs = group_childs

    async def _show_visible_groups_of_objtree(self):
        app_ev = mui.AppEvent("", {})
        all_attached, all_removed = [], []
        for k1, meta in self._user_obj_tree_item_to_meta.items():
            group = meta.vctx.root
            new_ev, attached, removed = group.set_new_layout_locally(
                {**meta.childs})
            all_attached.extend(attached)
            all_removed.extend(removed)
            mdprints = meta.md_prints.copy()
            meta.md_prints.clear()
            app_ev += new_ev
            if mdprints:
                app_ev += meta.card.print_blocks_event(mdprints)
        await self.send_and_wait(app_ev)
        for deleted in all_removed:
            await deleted._cancel_task()
        await self._run_special_methods(all_attached, all_removed)

    async def _gv_cards_callback(self, checked: bool, group: three.Group):
        if group.is_mounted():
            # TODO sync tree visible state
            await group.send_and_wait(group.update_event(visible=checked))

    def _get_tree_cards_and_groups(self, flatted_tree: Dict[Tuple[str, ...],
                                                            Any]):
        # TODO implement real mui card, we use flexbox currently
        flatted_tree_items = list(flatted_tree.items())
        flatted_tree_items.sort(key=lambda x: x[0])
        # 1. get flatted group for each item
        groups: Dict[str, three.Group] = {}
        obj_to_item_meta: Dict[int, CanvasUserTreeItem] = {}
        for k, v in flatted_tree_items:
            k_str = UNKNOWN_KEY_SPLIT.join(k)
            group = three.Group([])
            groups[k_str] = group
            cfg = get_or_create_canvas_item_cfg(group)
            cfg.proxy = GroupProxy("")
            cfg.type_str_override = get_qualname_of_type(
                type(v)).split(".")[-1]
            cfg.alias = k[-1]
            cfg.is_vapi = False
            vctx = VContext(self, group)
            obj_to_item_meta[id(v)] = CanvasUserTreeItem(
                k, vctx,
                UserTreeItemCard(k[-1], cfg.type_str_override,
                                 partial(self._gv_cards_callback,
                                         group=group)))
        return groups, obj_to_item_meta

    def _get_tdata_container_pane(self, tdata_table: mui.DataGrid):
        return mui.Allotment.Pane(tdata_table, preferredSize=1)


    async def _on_pan_to_fwd(self, selected):
        await self.ctrl.send_and_wait(
            self.ctrl.update_event(verticalDragToForward=not selected))

    async def _on_enable_grid(self, selected):
        if selected:
            await self._dynamic_grid.set_new_layout(
                [self._infgrid, self._axis_helper])
        else:
            await self._dynamic_grid.set_new_layout([])

    async def _on_enable_tdata_pane(self, selected):
        self.tdata_container_v2_pane.visible = selected
        await self._canvas_spitter.update_childs_complex()

    async def _on_enable_cfg_panel(self, selected):
        if selected:
            await self._cfg_container.set_new_layout([self.__get_cfg_panel()])
        else:
            await self._cfg_container.set_new_layout([])

    def __get_cfg_panel(self):
        _cfg_panel = ConfigPanelV2(self.cfg, self._on_cfg_change)
        _cfg_panel.prop(
            border="1px solid",
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
            # all_childs = self._dynamic_boxes._get_uid_encoded_to_comp_dict()
            # for v in all_childs.values():
            #     if isinstance(v, three.BoundingBox):
            #         ev += v.update_event(edgeWidth=value)
            # await self.send_and_wait(ev)
        elif uid == "box.opacity":
            ev = mui.AppEvent("", {})
            # all_childs = self._dynamic_boxes._get_uid_encoded_to_comp_dict()
            # for v in all_childs.values():
            #     if isinstance(v, three.BoundingBox):
            #         ev += v.update_event(opacity=value)
            # await self.send_and_wait(ev)
        elif uid == "box.add_cross":
            ev = mui.AppEvent("", {})
            # all_childs = self._dynamic_boxes._get_uid_encoded_to_comp_dict()
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

    def _extract_tdata_from_group(self, group: three.ContainerBase):
        common_keys = set()
        data_items = []
        for key, obj in group._child_comps.items():
            obj_cfg = get_canvas_item_cfg(obj)
            if obj_cfg is not None and obj_cfg.proxy is not None:
                tdata = obj_cfg.proxy._tdata
                if tdata is not None:
                    common_keys.update(tdata.keys())
                    data_items.append({"id": key, **tdata})
        return common_keys, data_items

    async def _on_group_select_object(self, ev: mui.Event,
                                      group: three.ContainerBase):
        if not isinstance(ev.keys, mui.Undefined):
            obj_local_id = ev.keys[0]
            if obj_local_id in group._child_comps:
                await self.ctrl.lookat_object(
                    group._child_comps[obj_local_id]._flow_uid_encoded)

    async def clear(self):
        await self._item_root.set_new_layout({**self._init_layout})
        await self.item_tree.update_tree()

    def _extract_table_from_group(self, group: three.ContainerBase):
        common_keys, data_items = self._extract_tdata_from_group(group)
        if len(common_keys) == 0:
            return None

        btn = mui.IconButton(mui.IconType.Visibility).prop(
            size="small",
            # fontSize="14px",
            iconFontSize="13px")
        btn.event_click.on_standard(
            partial(self._on_group_select_object, group=group)).configure(True)
        column_defs: List[mui.DataGrid.ColumnDef] = [
            mui.DataGrid.ColumnDef(id="__tensorpc_vis", cell=btn, width=30)
        ]
        # key_to_typo: Dict[str, mui.Typography] = {}
        for k in common_keys:
            typo = mui.Typography().prop(
                precisionDigits=4, fontSize="14px").set_override_props(value=k)
            column_defs.append(
                mui.DataGrid.ColumnDef(k, accessorKey=k, cell=typo))

        dgrid = mui.DataGrid(column_defs, data_items).prop(idKey="id",
                                                           rowHover=True,
                                                           virtualized=True,
                                                           enableFilter=True,
                                                           size="small",
                                                           fullWidth=True)

        return dgrid

    async def _on_tree_select(self, event: mui.Event):
        data = event.data
        assert isinstance(data, SelectSingleEvent)
        if data.objs is None:
            return
        obj = data.objs[-1]
        if isinstance(obj, three.Component) and obj.is_mounted():
            # print(find_component_trace_by_uid_with_not_exist_parts(self._item_root, "reserved.grid.2"))
            obj_cfg = get_canvas_item_cfg(obj)
            if obj_cfg is not None and obj_cfg.detail_layout is not None:
                if obj._flow_uid is not None:

                    await self.prop_container.set_new_layout(
                        [obj_cfg.detail_layout])
                    self._cur_detail_layout_uid = data.nodes[-1].id
                    self._cur_detail_layout_object_id = obj._flow_uid
            else:
                if three.is_three_component(obj):
                    if obj._flow_uid is not None:
                        panel = self._get_default_detail_layout(obj).prop(
                            reactKey=data.nodes[-1].id.uid_encoded)
                        self._cur_detail_layout_uid = data.nodes[-1].id
                        self._cur_detail_layout_object_id = obj._flow_uid
                        await self.prop_container.set_new_layout([panel])

                else:
                    await self.prop_container.set_new_layout([])
                    self._cur_detail_layout_uid = None
                    self._cur_detail_layout_object_id = None

            if isinstance(obj, three.ContainerBase):
                table = self._extract_table_from_group(obj)
                if table is not None:
                    if obj._flow_uid is not None:
                        self._cur_table_uid = data.nodes[-1].id
                        self._cur_table_object_id = obj._flow_uid
                        await self.tdata_container_v2.set_new_layout([table])

    async def _uninstall_detail_layout(self):
        async with self._lock:
            self._cur_detail_layout_uid = None
            self._cur_detail_layout_object_id = None
            await self.prop_container.set_new_layout([])

    async def _uninstall_table_layout(self):
        async with self._lock:
            self._cur_table_object_id = None
            self._cur_table_uid = None
            await self.tdata_container_v2.set_new_layout([])

    def _get_default_detail_layout(self, obj: three.Component):
        panel = ConfigPanelV2(obj.props,
                              partial(self._on_cfg_panel_change, obj=obj),
                              ignored_field_names=set([
                                  "status",
                                  "pivotControlProps",
                              ]))
        return panel

    async def _install_detail_layout(self, obj: three.Component):
        if self._cur_detail_layout_uid is not None:
            obj_cfg = get_canvas_item_cfg(obj)
            if obj_cfg is not None and obj_cfg.detail_layout is not None:
                await self.prop_container.set_new_layout(
                    [obj_cfg.detail_layout])
            else:
                if three.is_three_component(obj):
                    panel = self._get_default_detail_layout(obj).prop(
                        reactKey=self._cur_detail_layout_uid.uid_encoded)
                    await self.prop_container.set_new_layout([panel])

    async def _install_table_layout(self, obj: three.Component):
        if isinstance(obj, three.ContainerBase):
            table = self._extract_table_from_group(obj)
            if table is not None:
                await self.tdata_container_v2.set_new_layout([table])
            else:
                await self.tdata_container_v2.set_new_layout([])
        else:
            await self.tdata_container_v2.set_new_layout([])

    async def update_detail_layout(self, common_uid: UniqueTreeId):
        # TODO when we upgrade tree, we must check if the current selected node is still valid.
        if self._cur_detail_layout_uid is not None:
            # reg = re.compile(regex)
            # print(prefix, self._cur_detail_layout_uid)
            local_uid = self._convert_tree_node_uid_to_local_uid(
                self._cur_detail_layout_uid)
            if local_uid.startswith(common_uid) is not None:
                container_parents, remain_keys, _ = find_component_trace_by_uid_with_not_exist_parts(
                    self._item_root, local_uid)
                if len(remain_keys) == 0:
                    obj = container_parents[-1]
                    await self._install_detail_layout(obj)
                else:
                    await self.prop_container.set_new_layout([])

    async def set_layout_in_container(self, container_key: UniqueTreeId,
                                      layout: three.ThreeLayoutType):
        if isinstance(layout, list):
            layout = {str(i): v for i, v in enumerate(layout)}

        assert container_key != "" and not is_reserved_uid(
            container_key), "you can't set layout of canvas and reserved."
        container_parents, remain_keys, _ = find_component_trace_by_uid_with_not_exist_parts(
            self._item_root, container_key)
        if len(remain_keys) == 0:
            container = container_parents[-1]
            if isinstance(container, (three.Group, mui.Fragment)):
                await container.set_new_layout({**layout})
                await self.item_tree.update_tree()

    async def update_layout_in_container(self, container_key: UniqueTreeId,
                                         layout: three.ThreeLayoutType):
        assert container_key != "" and not is_reserved_uid(
            container_key), "you can't update layout of reserved."
        if isinstance(layout, list):
            layout = {str(i): v for i, v in enumerate(layout)}
        container_parents, remain_keys, _ = find_component_trace_by_uid_with_not_exist_parts(
            self._item_root, container_key)
        if len(remain_keys) == 0:
            container = container_parents[-1]
            if isinstance(container, (three.Group, mui.Fragment)):
                await container.update_childs({**layout})
                await self.item_tree.update_tree()

    async def _on_cfg_panel_change(self, uid: str, value: Any, obj: Component):
        # TODO support nested change
        uid_parts = uid.split(".")
        if len(uid_parts) > 1:
            return
        if "visible" == uid:
            cfg = get_canvas_item_cfg(obj)
            if cfg is not None and cfg.node is not None:
                cfg.visible = not cfg.visible
                cfg.node.fixedIconBtns = self.custom_tree_handler._get_icon_button(
                    obj)
                await self.item_tree.tree.update_subtree(cfg.node)
        await obj.send_and_wait(
            obj.create_update_event({
                uid: value,
            }, validate=True))

    def _get_local_uid_of_object(self, uid: UniqueTreeId):
        assert self._item_root._flow_uid is not None 
        assert uid.startswith(self._item_root._flow_uid)
        return UniqueTreeIdForTree.from_parts(uid.parts[uid.common_prefix_index(self._item_root._flow_uid):])

    def _convert_tree_node_uid_to_local_uid(self, uid: UniqueTreeId):
        assert uid.parts[0] == "root"
        return UniqueTreeIdForTree.from_parts(uid.parts[1:])

    async def _on_3d_object_select(self, selected: list):
        if not selected:
            await self.item_tree.tree.select([])
            return
        assert len(selected) == 1
        select = selected[0]
        selected_uid = select["userData"]["uid"]
        # print(self.item_tree.tree.props.tree)
        selected_uid_local_uid = self._get_local_uid_of_object(UniqueTreeIdForTree(selected_uid))

        container_parents, remain_keys, _ = find_component_trace_by_uid_with_not_exist_parts(
            self._item_root, selected_uid_local_uid)
        if len(remain_keys) == 0:
            obj = container_parents[-1]
            # we need to convert object component uid to tree node uid.
            # tree node uid always start with "root"
            # tree_node_uid = f"root::{selected_uid_local_uid.replace('.', '::')}"
            tree_node_uid = UniqueTreeIdForTree.from_parts(["root"] + selected_uid_local_uid.parts)
            await self.item_tree.tree.select([tree_node_uid.uid_encoded])
            await self.item_tree._on_select_single(tree_node_uid.uid_encoded)
            # print(selected_uid_local_uid, obj, obj._flow_uid)

    async def _unknown_visualization(self,
                                     tree_id: str,
                                     obj: Any,
                                     ignore_registry: bool = False):
        from . import vapi_core as V
        obj_type = type(obj)
        if obj_type in UNKNOWN_VIS_REGISTRY and not ignore_registry:
            handlers = UNKNOWN_VIS_REGISTRY[obj_type]
            for handler in handlers:
                res = await handler(obj, tree_id, self)
                if res == True:
                    return True
        tree_id_replaced = tree_id.replace(".", UNKNOWN_KEY_SPLIT)
        if UNKNOWN_VIS_KEY in self._item_root._child_comps:
            unk_container = self._item_root._child_comps[UNKNOWN_VIS_KEY]
            assert isinstance(unk_container, three.Group)
        else:
            unk_container = three.Group([])
            await self._item_root.update_childs(
                {UNKNOWN_VIS_KEY: unk_container})
            cfg = get_or_create_canvas_item_cfg(unk_container)
            cfg.proxy = V.GroupProxy("")
        vctx_unk = V.VContext(self, unk_container)
        pc_obj = _try_cast_to_point_cloud(obj)
        if pc_obj is not None:
            if tree_id_replaced in self._random_colors:
                pick = self._random_colors[tree_id_replaced]
            else:
                random_colors = colors.RANDOM_COLORS_FOR_UI
                pick = random_colors[_count_child_type(
                    unk_container, three.Points) % len(random_colors)]
                self._random_colors[tree_id_replaced] = pick
            with V.enter_v_conetxt(V.VContext(self, unk_container)):
                points = V.points(tree_id_replaced, pc_obj.shape[0]).array(
                    pc_obj.astype(np.float32))
                if pc_obj.shape[1] == 3:
                    points.prop(colors=pick)

            await V._draw_all_in_vctx(vctx_unk,
                                      unk_container._flow_uid)
            return True
        img_obj = _try_cast_to_image(obj)
        if img_obj is not None:
            with V.enter_v_conetxt(V.VContext(self, unk_container)):
                V.image(img_obj, name=tree_id_replaced,
                        pos=(0, 0, 0.1)).prop(scale=(3, 3, 3))
            await V._draw_all_in_vctx(vctx_unk,
                                      unk_container._flow_uid)
            return True
        b3d_obj = _try_cast_to_box3d(obj)
        if b3d_obj is not None:
            if tree_id in self._random_colors:
                pick = self._random_colors[tree_id]
            else:
                random_colors = colors.RANDOM_COLORS_FOR_UI
                pick = random_colors[_count_child_type(
                    unk_container, three.BoundingBox) % len(random_colors)]
                self._random_colors[tree_id] = pick
            with V.enter_v_conetxt(V.VContext(self, unk_container)):

                with V.group(tree_id_replaced):
                    for box in b3d_obj:
                        V.bounding_box(box[3:6], (0, 0, box[6]),
                                       box[:3]).prop(color=pick)
            await V._draw_all_in_vctx(vctx_unk,
                                      unk_container._flow_uid)
            return True
        line_obj = _try_cast_to_lines(obj)
        if line_obj is not None:
            if tree_id in self._random_colors:
                pick = self._random_colors[tree_id]
            else:
                random_colors = colors.RANDOM_COLORS_FOR_UI
                pick = random_colors[_count_child_type(
                    unk_container, three.Segments) % len(random_colors)]
                self._random_colors[tree_id] = pick
            with V.enter_v_conetxt(V.VContext(self, unk_container)):

                points = V.lines(tree_id_replaced, line_obj.shape[0]).array(
                    line_obj.astype(np.float32))

            await V._draw_all_in_vctx(vctx_unk,
                                      unk_container._flow_uid)
            return True
        return False

    async def _on_drop(self, data):
        from tensorpc.flow.flowapp.components.plus import BasicObjectTree
        if isinstance(data, TreeDragTarget):
            obj = data.obj
            # tree_id_replaced = data.tree_id.replace("::", "__")

            success = await self._unknown_visualization(data.tree_id, obj)
            if success:
                # register to tree
                tree = find_component_by_uid_with_type_check(
                    data.source_comp_uid, BasicObjectTree)
                if tree is not None:
                    tree._register_dnd_uid(UniqueTreeIdForTree(data.tree_id), self._dnd_cb)
                    self._dnd_trees.add(data.source_comp_uid)

    async def _dnd_cb(self, uid: UniqueTreeIdForTree, data: Any):
        await self._unknown_visualization(uid.uid_encoded, data)

    async def register_cam_control_event_handler(
            self,
            handler: Callable[[Any], mui.CORO_NONE],
            throttle: int = 100,
            debounce: Optional[int] = None):
        self.ctrl.event_change.on(handler).configure(throttle=throttle,
                                                     debounce=debounce)
        await self.ctrl.sync_used_events()

    async def clear_cam_control_event_handler(self):
        self.ctrl.remove_event_handlers(self.ctrl.event_change.event_type)
        await self.ctrl.sync_used_events()

    async def set_transparent(self, is_transparent: bool):
        if is_transparent:
            await self.canvas.send_and_wait(
                self.canvas.update_event(threeBackgroundColor=mui.undefined))
        else:
            await self.canvas.send_and_wait(
                self.canvas.update_event(threeBackgroundColor="#ffffff"))

    async def update_locals(self,
                            *,
                            _frame_cnt: int = 1,
                            exclude_self: bool = False):
        cur_frame = inspect.currentframe()
        assert cur_frame is not None
        frame = cur_frame
        while _frame_cnt > 0:
            frame = cur_frame.f_back
            assert frame is not None
            cur_frame = frame
            _frame_cnt -= 1
        # del frame
        local_vars = cur_frame.f_locals.copy()
        if exclude_self:
            local_vars.pop("self", None)
        del frame
        del cur_frame
        new_local_vars = {}
        local_var_arrays: Dict[str, np.ndarray] = {}
        for k, v in local_vars.items():
            arr = try_cast_to_np_array(v)
            if arr is not None:
                local_var_arrays[k] = arr
            else:
                if self.gv_locals_layout._check_type_support_preview(type(v)):
                    new_local_vars[k] = v
        arr_table = NumpyArrayGridTable({**local_var_arrays})
        arr_table.set_user_meta_by_type(ObjectGridItemConfig(2.0, 2.0))
        new_local_vars["__tensorpc_arrays"] = arr_table
        await self.gv_locals_layout.set_new_items(new_local_vars)

    def update_locals_sync(self,
                           *,
                           _frame_cnt: int = 1,
                           loop: Optional[asyncio.AbstractEventLoop] = None,
                           exclude_self: bool = False):
        """update locals in sync manner, usually used on non-sync code via appctx.
        """
        if loop is None:
            loop = asyncio.get_running_loop()
        cur_frame = inspect.currentframe()
        assert cur_frame is not None
        frame = cur_frame
        while _frame_cnt > 0:
            frame = cur_frame.f_back
            assert frame is not None
            cur_frame = frame
            _frame_cnt -= 1
        # del frame
        local_vars = cur_frame.f_locals.copy()
        if exclude_self:
            local_vars.pop("self", None)
        del frame
        del cur_frame
        new_local_vars = {}
        local_var_arrays: Dict[str, np.ndarray] = {}
        for k, v in local_vars.items():
            arr = try_cast_to_np_array(v)
            if arr is not None:
                local_var_arrays[k] = arr
            else:
                if self.gv_locals_layout._check_type_support_preview(type(v)):
                    new_local_vars[k] = v
        arr_table = NumpyArrayGridTable({**local_var_arrays})
        arr_table.set_user_meta_by_type(ObjectGridItemConfig(2.0, 2.0))

        new_local_vars["__tensorpc_arrays"] = arr_table
        if appctx.get_app()._flowapp_thread_id == threading.get_ident():
            task = asyncio.create_task(
                self.gv_locals_layout.set_new_items(new_local_vars))
            # we can't wait fut here
            return task
        else:
            # we can wait fut here.
            fut = asyncio.run_coroutine_threadsafe(
                self.gv_locals_layout.set_new_items(new_local_vars), loop)
            return fut.result()

    async def set_background_image(self, image: np.ndarray):
        await self.background_img.show(image)
