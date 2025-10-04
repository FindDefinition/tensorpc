import asyncio
import bisect
from functools import partial
import math
import time

from tensorpc.core import pfl
from tensorpc.core.datamodel.draft import DraftFieldMeta
from tensorpc.core.datamodel.events import DraftChangeEvent
from tensorpc.dock import mui, three, plus, appctx, mark_create_layout
import dataclasses
from typing import Any, Optional, cast 
from typing_extensions import Annotated
import numpy as np
import tensorpc.core.datamodel as D
from tensorpc.core.pfl.backends.js import Math, MathUtil

from tensorpc.dock.components.three.event import HudLayoutChangeEvent, KeyboardHoldEvent, PointerEvent, ViewportChangeEvent
from tensorpc.utils.perfetto_colors import perfetto_slice_to_color 
from tensorpc.apps.dbg.components.perfutils import build_depth_from_trace_events

@dataclasses.dataclass
class PerfFieldInfo:
    name: str 
    min_ts: float 
    max_ts: float
    duration: float
    # all_real_duration / (max_ts - min_ts)
    rate: float
    left_line: list[list[float]] 
    right_line: list[list[float]] 
    cnt: int = 0
    cluster_name: str = ""


@dataclasses.dataclass
class VisInfo:
    trs: np.ndarray
    colors: np.ndarray
    scales: np.ndarray
    info_idxes: np.ndarray
    rank_ids: np.ndarray
    durations: np.ndarray
    width: float 
    height: float 
    minWidth: float

@dataclasses.dataclass
class SimpleLayout:
    scrollFactorX: three.NumberType = 1.0
    scrollFactorY: three.NumberType = 1.0
    innerSizeX: three.NumberType = 1.0
    innerSizeY: three.NumberType = 1.0


@dataclasses.dataclass
class VisModel(VisInfo):
    total_duration: float
    infos: list[PerfFieldInfo]
    hoverData: Any = None
    clickInfo: Any = None

    hoverInfoId: Optional[int] = None
    clickInstanceId: Optional[int] = None
    clickClusterPoints: Optional[Any] = None
    clickClusterAABBSizes: Optional[Any] = None
    viewport: Optional[ViewportChangeEvent] = None
    scaleYGlobal: float = 1.0
    step: int = -1
    scaleX: float = 1.0
    scaleY: float = 20.0
    layout: SimpleLayout = dataclasses.field(default_factory=SimpleLayout)
    scrollValueX: float = 0.0
    scrollValueY: float = 0.0
    debug: Any = None
    perfHover: Optional[PointerEvent] = None

    meta_datas: Annotated[list[Any], DraftFieldMeta(is_external=True)] = dataclasses.field(default_factory=list)
    all_events: Annotated[list[dict], DraftFieldMeta(is_external=True)] = dataclasses.field(default_factory=list)
    name_cnt_to_polygons: Annotated[dict[str, tuple[np.ndarray, np.ndarray]], DraftFieldMeta(is_external=True)] = dataclasses.field(default_factory=dict)
    @staticmethod
    def bind_scale_xy(comp: mui.Component):
        comp.bind_fields_unchecked_dict({
            "scale-x": "scaleX",
            "scale-y": "scaleY",
        })
        return comp

    @pfl.mark_pfl_compilable
    def _keyhold_handler_pfl(self, data: KeyboardHoldEvent):
        """This function will be compiled to javascript in frontend. so we only support limited python syntax/method.
        """
        if self.perfHover is not None:
            prev = self.scaleX
            prev_scroll_value = self.scrollValueX
            dx = 0.0
            is_scale = data.code == "KeyW" or data.code == "KeyS"
            if data.code == "KeyW":
                dx = data.deltaTime * 0.002 * self.scaleX
            elif data.code == "KeyS":
                dx = -data.deltaTime * 0.002 * self.scaleX
            elif data.code == "KeyA":
                dx = -data.deltaTime * 0.002 / self.scaleX
            elif data.code == "KeyD":
                dx = data.deltaTime * 0.002 / self.scaleX
            if is_scale:
                new_scale = MathUtil.clamp(dx + prev, 1.0, 100.0)
                real_dx = new_scale - prev
                self.scaleX = new_scale
                # scaledWidth = (innerWidth * scaleX)
                # offset_view_inner = offset / width * (innerWidth * scaleX) - scrollValueX * ((innerWidth * scaleX) - innerWidth)
                # new_offset_view_inner = offset / width * (innerWidth * (scaleX + dX)) - scrollValueX * ((innerWidth * (scaleX + dX)) - innerWidth)

                # offset_view_inner = offset / width * scaleX - scrollValueX * (scaleX - 1)
                # new_offset_view_inner = offset / width * (scaleX + dX) - newScrollValueX * (scaleX + dX - 1)


                # offset_view_inner = offset / width 
                # new_offset_view_inner = offset / width * (1 + dX) - newScrollValueX * dX

                # offset_view_inner = 0
                # new_offset_view_inner = offset / width * dX - newScrollValueX * dX

                # offset_view_inner = offset / width * (scaleX - dX) - scrollValueX * (scaleX - dX - 1)
                # new_offset_view_inner = offset / width * scaleX - newScrollValueX * (scaleX - 1)

                # offset_view_inner = offset / width
                # new_offset_view_inner = offset / width * (1 + dX) - newScrollValueX * dX

                # offset / width * (scaleX - dX) - scrollValueX * (scaleX - dX - 1) == offset / width * (scaleX) - newScrollValueX * (scaleX - 1)
                # -offset / width * dX - scrollValueX * (scaleX - dX - 1) == -newScrollValueX * (scaleX - 1)

                # newScrollValueX = (offset / width * dX + scrollValueX * (scaleX - dX - 1)) / (scaleX - 1)
                # newScrollValueX = (offset / width + scrollValueX) * dX / (scaleX - 1) + scrollValueX

                self.scrollValueX = MathUtil.clamp((self.perfHover.pointLocal[0] + 0.5 - prev_scroll_value) * real_dx / Math.max(new_scale - 1.0, 1e-6) + prev_scroll_value, 0.0, 1.0)
            else:
                self.scrollValueX = MathUtil.clamp(prev_scroll_value + dx, 0.0, 1.0)
    
    @staticmethod
    def _create_empty_vis_model():
        trs_empty = np.zeros((0, 3), dtype=np.float32)
        colors_empty = np.zeros((0, 3), dtype=np.float32)
        scales_empty = np.zeros((0, 3), dtype=np.float32)
        indexes_empty = np.zeros((0,), dtype=np.int32)
        durs_empty = np.zeros((0,), dtype=np.float32)
        return VisModel(trs_empty, colors_empty, scales_empty, indexes_empty, 
            indexes_empty, durs_empty, 0, 0, 0, 0, [])

    @pfl.mark_pfl_compilable
    def _box_event_move(self, data: three.PointerEvent):
        insatance_id = cast(int, data.instanceId if data.instanceId else 0)
        dur = pfl.js.MathUtil.getTypedArray(self.durations)
        info_idxes_arr = pfl.js.MathUtil.getTypedArray(self.info_idxes)[insatance_id]
        self.hoverData = {
            "offset": data.offset,
            "instanceId": data.instanceId,
            "dur": dur[insatance_id],
            "info": self.infos[int(info_idxes_arr)]
        }

    @pfl.mark_pfl_compilable
    def _wheel_change(self, event: three.PointerEvent):
        self.scrollValueY = MathUtil.clamp(self.scrollValueY - event.wheel.deltaY * 0.001 * self.layout.scrollFactorY, 0.0, 1.0)


    @pfl.mark_pfl_compilable
    def _scrollbar_pos_change(self, data: three.PoseChangeEvent):
        self.scrollValueY = MathUtil.clamp(-data.positionLocal[1] / (Math.max(1 - self.scrollValueY, 0.0001)) + 0.5, 0.0, 1.0)

    @pfl.mark_pfl_compilable
    def _scrollbar_bottom_pos_change(self, data: three.PoseChangeEvent):
        self.scrollValueX = MathUtil.clamp(data.positionLocal[0] / (Math.max(1 - self.scrollValueX, 0.0001)) + 0.5, 0.0, 1.0)


def _get_vis_data_from_duration_events(duration_events: list[dict], dur_scale: float, 
        min_ts: int, depth_padding: float, height: float) -> tuple[VisInfo, Any]:
    """
    Get the vis data from the duration events.
    :param duration_events: The duration events.
    :param dur_scale: The scale of the duration.
    :param depth_padding: The padding of the depth.
    :param height: The height of the boxes.
    :return: The positions, colors and scales of the boxes.
    """
    t = time.time()
    colors = []
    name_cnt_to_index: dict[str, Any] = {}
    event_ts_u64 = np.array([ev["ts"] for ev in duration_events], dtype=np.uint64)
    event_dur = np.array([ev["dur"] for ev in duration_events], dtype=np.float32)
    event_depth = np.array([ev["depth"] for ev in duration_events], dtype=np.float32)
    x_arr = ((event_ts_u64 - min_ts).astype(np.float32) + event_dur / 2) * dur_scale
    y_arr = depth_padding + (event_depth - 0.5) * (height + depth_padding) - 1.5 * depth_padding
    trs_arr = np.stack([x_arr, -y_arr, np.zeros_like(x_arr)], axis=1)
    scales_arr = np.stack([event_dur * dur_scale, height * np.ones_like(event_dur), np.ones_like(event_dur)], axis=1)
    
    width = float((x_arr + scales_arr[:, 0] / 2).max())
    height = float((y_arr + scales_arr[:, 1] / 2).max())
    block_min_width = scales_arr[:, 0].min()
    trs_arr[:, 0] -= width / 2
    trs_arr[:, 1] += height / 2
    info_idxes_arr = np.array([ev["field_idx"] for ev in duration_events], dtype=np.int32)
    perfetto_slice_cache: dict[str, Any] = {}
    # print("1.1", time.time() - t, len(duration_events))
    for i, event in enumerate(duration_events):
        name = event["name"]
        ev_cnt = event["cnt"]
        name_with_cnt = f"{event['name']}::{ev_cnt}"
        if name in perfetto_slice_cache:
            color = perfetto_slice_cache[name]
        else:
            color = perfetto_slice_to_color(name).base.rgb
            perfetto_slice_cache[name] = color
        colors.append(color)
        if name_with_cnt not in name_cnt_to_index:
            name_cnt_to_index[name_with_cnt] = []
        name_cnt_to_index[name_with_cnt].append(i)
    # print("1.2", time.time() - t)

    name_cnt_to_polygons = {}
    for cluster_name, indexes in name_cnt_to_index.items():
        indexes_arr = np.array(indexes, dtype=np.int32)
        scales_namecnt = scales_arr[indexes_arr]
        trs_namecnt = trs_arr[indexes_arr]
        name_cnt_to_polygons[cluster_name] = (trs_namecnt, scales_namecnt)
    # print("1.3", time.time() - t)
    colors_arr = np.array(colors, dtype=np.uint8)
    return VisInfo(
        trs=trs_arr,
        colors=colors_arr,
        scales=scales_arr,
        info_idxes=info_idxes_arr,
        rank_ids=np.array([event["rank"] for event in duration_events], dtype=np.int32),
        durations=np.array(event_dur / 1e9, dtype=np.float32),
        width=width,
        height=height,
        minWidth=block_min_width,
    ), name_cnt_to_polygons

class PerfMonitor(mui.FlexBox):
    def __init__(self, use_view: bool = False):
        trs_empty = np.zeros((0, 3), dtype=np.float32)

        boxmesh = three.InstancedMesh(trs_empty, 200000, [
            three.PlaneGeometry(),
            three.MeshBasicMaterial(),
        ]).prop(raycaster="2d_aabb")
        line = three.Line([(0, 0, 0), (1, 1, 1)]).prop(color="red", lineWidth=2, variant="aabb")
        line_cond = mui.MatchCase.binary_selection(True, line)

        line_start = three.Line([(0, 0, 0), (1, 1, 1)]).prop(color="gray", lineWidth=1, dashed=True, dashSize=0.5, gapSize=0.5)
        line_start_cond = mui.MatchCase.binary_selection(True, line_start)
        line_end = three.Line([(0, 0, 0), (1, 1, 1)]).prop(color="gray", lineWidth=1, dashed=True, dashSize=0.5, gapSize=0.5)
        line_end_cond = mui.MatchCase.binary_selection(True, line_end)
        line_select = three.Line([(0, 0, 0), (1, 1, 1)]).prop(color="blue", lineWidth=2, variant="aabb")
        line_select_cond = mui.MatchCase.binary_selection(True, line_select)

        line_select_samename = three.Line([(0, 0, 0), (1, 1, 1)]).prop(color="green", lineWidth=1, opacity=0.7, segments=True, variant="aabb")
        line_select_samename_cond = mui.MatchCase.binary_selection(True, line_select_samename)

        self._cam_ctrl = three.CameraControl().prop(makeDefault=True, mouseButtons=three.MouseButtonConfig(left="none"))

        perf_event_plane = three.Mesh([
            three.PlaneGeometry(1.0, 1000.0),
            three.MeshBasicMaterial().prop(transparent=True, opacity=0.0),
        ]).prop(position=(0, 0, -0.2))
        perf_group = three.Group([
            boxmesh,
            line_cond,
            line_start_cond,
            line_end_cond,
            three.Group([
                line_select_cond
            ]).prop(position=(0, 0, 0.015)),
            three.Group([
                line_select_samename_cond
            ]).prop(position=(0, 0, 0.014)),
            perf_event_plane,
        ]).prop(position=(0, 0, -1.1))
        # TODO: portal don't support view
        viewport_group = three.HudGroup([
            perf_group
        ]).prop(top=0, left=0, padding=2, width="calc(100% - 15px)", height="calc(100% - 15px)", alignContent=False, alwaysPortal=False)
        # viewport_group.event_hud_layout_change.on(lambda ev: print(ev))
        scrollbar_event_plane = three.Mesh([
            three.PlaneGeometry(1.0, 1.0),
            three.MeshBasicMaterial().prop(color="white"),
        ]).prop(position=(0, 0, -2.1))

        scrollbar = three.Mesh([
            three.PlaneGeometry(1, 1),
            three.MeshBasicMaterial().prop(color="orange"),
        ])
        # scrollbar.event_pose_change.on(lambda ev: print(ev)).configure(debounce=300)
        scrollbar_group = three.HudGroup([
            scrollbar
        ]).prop(top=0, right=0, padding=2, width="15px", height="calc(100% - 15px)", borderColor="gray", borderWidth=1, childWidth=1, childHeight=1, alignContent="stretch")
        scrollbar_bottom = three.Mesh([
            three.PlaneGeometry(1, 1),
            three.MeshBasicMaterial().prop(color="orange"),
        ])
        scrollbar_bottom_group = three.HudGroup([
            scrollbar_bottom
        ]).prop(bottom=0, left=0, padding=2, width="100%", height="15px", borderColor="gray", borderWidth=1, childWidth=1, childHeight=1, alignContent="stretch")

        scrollbar_plane_group = three.HudGroup([
            scrollbar_event_plane
        ]).prop(top=0, left=0, width="100%", height="100%", childWidth=1, childHeight=1, position=(0, 0, -2), alignContent="stretch")

        cam = three.PerspectiveCamera(fov=75, near=0.1, far=1000, children=[
            viewport_group,  
            # boxmeshX,
            scrollbar_group,
            scrollbar_bottom_group,
        ]).prop(position=(0, 0, 10))

        cam = three.OrthographicCamera(near=0.1, far=1000, children=[
            viewport_group,  
            # boxmeshX,
            scrollbar_group,
            scrollbar_plane_group,
            scrollbar_bottom_group,

        ]).prop(position=(0, 0, 10))
        if use_view:
            canvas = three.View([
                # self._cam_ctrl,
                three.InfiniteGridHelper(5, 50, "gray"),
                cam.prop(makeDefault=True),
            ]).prop(allowKeyboardEvent=True)
        else:
            canvas = three.Canvas([
                # self._cam_ctrl,
                three.InfiniteGridHelper(5, 50, "gray"),
                cam.prop(makeDefault=True),
            ]).prop(enablePerf=False, allowKeyboardEvent=True)

        canvas.prop(menuItems=[
            mui.MenuItem("reset", "reset"),
            mui.MenuItem("clear", "clear"),
        ])
        canvas.event_context_menu.on(self._on_menu_select)

        empty_model = VisModel._create_empty_vis_model()
        dm = mui.DataModel(empty_model, [])
        draft = dm.get_draft()
        dm.install_draft_change_handler(draft.clickInstanceId, self._on_click_instance_id_change)
        boxmesh.event_move.add_frontend_handler(dm, VisModel._box_event_move, use_immer=False)
        boxmesh.event_leave.add_frontend_draft_set_none(draft, "hoverData")
        boxmesh.event_click.on_standard(self._on_click)
        perf_event_plane.bind_fields_unchecked_dict({
            "scale-x": "width",
            "scale-y": "height",
        })
        perf_event_plane.event_move.add_frontend_draft_change(draft, "perfHover")
        perf_event_plane.event_leave.add_frontend_draft_set_none(draft, "perfHover")

        viewport_group.event_hud_layout_change.add_frontend_draft_change(draft, "layout")
        # scrollbar_event_plane.event_wheel.add_frontend_draft_change(draft, "scrollValueY", f"clamp(__PREV_VALUE__ + wheel.deltaY * `0.001`, `0`, `1`)")
        scrollbar_event_plane.event_wheel.add_frontend_handler(dm, VisModel._wheel_change, use_immer=False)
        # scrollbar_event_plane.event_wheel.on(lambda e: print(e)).configure(debounce=300)
        # scrollbar.event_pose_change.add_frontend_draft_change(draft, "scrollValueY", f"clamp(-positionLocal[1] / maximum(`1` - __TARGET__.layout.scrollFactorY, `0.0001`) + `0.5`, `0`, `1`)")
        # scrollbar_bottom.event_pose_change.add_frontend_draft_change(draft, "scrollValueX", f"clamp(positionLocal[0] / maximum(`1` - __TARGET__.layout.scrollFactorX, `0.0001`) + `0.5`, `0`, `1`)")
        scrollbar.event_pose_change.add_frontend_handler(dm, VisModel._scrollbar_pos_change, use_immer=False)
        scrollbar_bottom.event_pose_change.add_frontend_handler(dm, VisModel._scrollbar_bottom_pos_change, use_immer=False)
        canvas.event_viewport_change.add_frontend_draft_change(draft, "viewport")

        canvas.event_keyboard_hold.configure(key_codes=["KeyW", "KeyS", "KeyA", "KeyD"])
        canvas.event_keyboard_hold.add_frontend_handler(dm, VisModel._keyhold_handler_pfl, use_immer=False)
        boxmesh.bind_fields(transforms="trs", colors="colors", scales="scales")
        # VisModel.bind_scale_xy(perf_group)
        # devmesh.bind_fields(scale="whole_scales")
        label_box = mui.VBox([
            mui.Typography("")
                .prop(variant="caption")
                .bind_fields(value="cformat('%s[%d](dur=%.3fs, alldur=%.3fs)', hoverData.info.name, ndarrayGetItem(rank_ids, not_null(hoverData.instanceId, 0)), hoverData.dur, hoverData.info.duration)"),
            # mui.JsonViewer().bind_fields(data="getItem(infos, ndarrayGetItem(info_idxes, not_null(hoverData.instanceId, `0`)))"),
        ]).prop(width="300px", position="absolute", backgroundColor="rgba(255, 255, 255, 0.5)", pointerEvents="none", zIndex=1)
        label_box.bind_fields(top="not_null(hoverData.offset[1], 0) + 5", left="not_null(hoverData.offset[0], 0) + 5")
        label = mui.MatchCase.binary_selection(True, label_box)
        label.bind_fields(condition="hoverData is not None")
        line.bind_fields(points="[ndarrayGetItem(trs, not_null(hoverData.instanceId, 0))]", 
                         aabbSizes="ndarrayGetItem(scales, not_null(hoverData.instanceId, 0))")
        viewport_group.bind_fields(childWidthScale="scaleX", childHeight=f"height * scaleY", scrollValueY="scrollValueY", scrollValueX="scrollValueX")
        # scrollbar_group.bind_fields(childHeight=f"not_null(layout.scrollFactorY, `1`)")
        scrollbar.bind_fields_unchecked_dict({
            "position-y": "-(scrollValueY - 0.5) * (1 - layout.scrollFactorY)",
            "scale-y": "layout.scrollFactorY",
        })
        scrollbar_bottom.bind_fields_unchecked_dict({
            "position-x": "(scrollValueX - 0.5) * (1 - layout.scrollFactorX)",
            "scale-x": "layout.scrollFactorX",
        })

        perf_group.bind_fields_unchecked_dict({
            "scale-x": "scaleX * layout.innerSizeX / (layout.innerSizeX if width == 0 else width)",

            "scale-y": "scaleY",
        })
        line_cond.bind_fields(condition="hoverData is not None")

        line_select_samename.bind_fields(points="clickClusterPoints", aabbSizes="clickClusterAABBSizes")
        line_select_samename_cond.bind_fields(condition="clickClusterPoints is not None")


        line_select.bind_fields(points="[ndarrayGetItem(trs, not_null(clickInstanceId, 0))]", 
                         aabbSizes="ndarrayGetItem(scales, not_null(clickInstanceId, 0))")
        line_select_cond.bind_fields(condition="clickInstanceId is not None")

        line_start.bind_fields(points="hoverData.info.left_line")

        line_start_cond.bind_fields(condition="hoverData is not None")
        line_end.bind_fields(points="hoverData.info.right_line")

        line_end_cond.bind_fields(condition="hoverData is not None")
        header = mui.Typography().prop(variant="caption")
        self.history: list[VisModel] = []
        slider = mui.BlenderSlider(0, 0, 1, self._select_vis_model)
        slider.prop(isInteger=True, showControlButton=True, showTotal=True)
        # select = mui.Autocomplete("history", [], self._select_vis_model).prop(size="small", textFieldProps=mui.TextFieldProps(muiMargin="dense"))
        self.history_slider = slider
        self._header = header
        self._detail_viewer = mui.JsonViewer() # .bind_fields(data="perfHover")
        self._update_lock = asyncio.Lock()
        self.max_num_history = 1000
        canvas_container_with_tooltip = mui.TooltipFlexBox("", [
            canvas.prop(flex=1),
            # label,
        ]).prop(minHeight=0,
                minWidth=0,
                overflow="hidden",
                flex=1,
                followCursor=True)
        canvas_container_with_tooltip.bind_fields(title="cformat('%s[%d](dur=%.3fs, alldur=%.3fs)', hoverData.info.name, ndarrayGetItem(rank_ids, not_null(hoverData.instanceId, 0)), hoverData.dur, hoverData.info.duration)")
        dm.init_add_layout([
            mui.VBox([
                mui.HBox([
                    header.prop(flex=1),
                    mui.VDivider(),
                    slider.prop(flex=2),
                ]),
                mui.HDivider(),
                canvas_container_with_tooltip,
            ]).prop(minHeight=0,
                    minWidth=0,
                    overflow="hidden",
                    flex=3),
            mui.VDivider(),
            mui.HBox([
                self._detail_viewer,
            ]).prop(flex=1, overflow="auto", fontSize=12)
        ])
        self.dm = dm
        super().__init__([dm])
        self.prop(minHeight=0,
                minWidth=0,
                flexFlow="row nowrap",
                width="100%",
                height="100%",
                overflow="hidden")

    async def _on_menu_select(self, value: str):
        if value == "reset":
            await self._cam_ctrl.reset_camera()
        elif value == "clear":
            await self.clear()

    async def clear(self):
        self.history.clear()
        vis_model = VisModel._create_empty_vis_model()
        await self._sync_history_select()
        async with self.dm.draft_update() as draft:
            draft.trs = vis_model.trs 
            draft.colors = vis_model.colors 
            draft.scales = vis_model.scales 
            draft.infos = vis_model.infos 
            draft.info_idxes = vis_model.info_idxes 
            draft.rank_ids = vis_model.rank_ids 
            draft.durations = vis_model.durations
            draft.total_duration = vis_model.total_duration
            draft.meta_datas = vis_model.meta_datas
            draft.name_cnt_to_polygons = vis_model.name_cnt_to_polygons
            draft.clickClusterPoints = None 
            draft.clickClusterAABBSizes = None
        await self._header.write("")
        await self._detail_viewer.write(None)

    async def _on_click(self, ev: mui.Event):
        instance_id = ev.data.instanceId 
        info_idx = int(self.dm.model.info_idxes[instance_id]) 
        info = self.dm.model.infos[info_idx]
        self.dm.get_draft().clickInstanceId = instance_id 
        self.dm.get_draft().clickClusterPoints = self.dm.model.name_cnt_to_polygons[info.cluster_name][0]
        self.dm.get_draft().clickClusterAABBSizes = self.dm.model.name_cnt_to_polygons[info.cluster_name][1]


    async def _update_detail(self, instance_id: Optional[int]):
        if instance_id is not None and instance_id < self.dm.model.info_idxes.shape[0]:
            info_idx = int(self.dm.model.info_idxes[instance_id]) 
            rank = int(self.dm.model.rank_ids[instance_id])
            info = self.dm.model.infos[info_idx]
            raw_event = self.dm.model.all_events[instance_id]
            res = {
                "name": info.name, 
                "rank": rank,
                "duration": round(float(self.dm.model.durations[instance_id]), 4),
                "all_duration": round(info.duration, 4),
                "rate": round(info.rate, 4),
                "start_ts": str(raw_event["ts"]),
                "end_ts": str(raw_event["ts"] + raw_event["dur"]),
                "cnt": raw_event["cnt"],
            }
            if "args" in raw_event:
                res["args"] = raw_event["args"]
            if rank < len(self.dm.model.meta_datas):
                metadata = self.dm.model.meta_datas[rank]
                if metadata is not None:
                    res["meta"] = metadata
            await self._detail_viewer.write(res)
        else:
            await self._detail_viewer.write(None)

    async def _on_click_instance_id_change(self, ev: DraftChangeEvent):
        if ev.new_value is not None:
            instance_id = ev.new_value
            await self._update_detail(instance_id)
        else:
            await self._detail_viewer.write(None)

    async def append_perf_data(self, step: int, data_list_all_rank: list[list[dict]], meta_datas: list[Any], scale: Optional[float] = None, max_depth: int = 3):
        async with self._update_lock:
            t = time.time()
            vis_model = await asyncio.get_running_loop().run_in_executor(None, partial(self.perf_data_to_vis_model, user_scale=scale, max_depth=max_depth), data_list_all_rank)
            # vis_model = self.perf_data_to_vis_model(data_list_all_rank, user_scale=scale)
            duration = time.time() - t
            if duration > 0.5:
                print("perf_data_to_vis_model time", time.time() - t, vis_model.trs.shape)
            # insert step sorted
            # calc insert loc by bisect 
            vis_model.meta_datas = meta_datas
            vis_model.step = step
            # remove all data with step >= provided step
            loc = bisect.bisect_left(self.history, vis_model.step, key=lambda v: v.step)
            self.history = self.history[:loc]
            # insert new data
            self.history.append(vis_model)
            max_num_history = self.max_num_history
            dropped_cnt = 0
            if len(self.history) > max_num_history:
                dropped_cnt = len(self.history) - max_num_history
                self.history = self.history[-max_num_history:]
            prev_index = self.history_slider.int() - dropped_cnt

            if prev_index + dropped_cnt < loc - 1 and prev_index >= 0:
                hist_len = max(0, len(self.history) - 1)
                await self.history_slider.update_ranges(0, hist_len, value=prev_index)
            else:
                await self._sync_history_select()

        # bisect.insort(self.history, vis_model, key=lambda v: v.step)
        # await self._sync_history_select()

    async def _sync_history_select(self):
        hist_len = max(0, len(self.history) - 1)
        print("self.history", len(self.history))
        if not self.history:
            await self.history_slider.update_ranges(0, 0, 1, value=0)
        else:
            await self.history_slider.update_ranges(0, len(self.history) - 1, value=len(self.history) - 1)
            await self._select_vis_model(len(self.history) - 1)

    async def _select_vis_model(self, val: mui.ValueType):
        index = int(val)
        vis_model = self.history[index]
        dur = vis_model.total_duration / 1e9
        await self._header.write(f"Step-{vis_model.step} ({dur:.2f}s)")

        async with self.dm.draft_update() as draft:
            draft.trs = vis_model.trs 
            draft.colors = vis_model.colors 
            draft.scales = vis_model.scales 
            # draft.infos = JsonSpecialData.from_option(vis_model.infos, is_json_only=True, need_freeze=True) 
            # draft.infos = vis_model.infos

            draft.info_idxes = vis_model.info_idxes 
            draft.rank_ids = vis_model.rank_ids 
            draft.durations = vis_model.durations
            draft.total_duration = vis_model.total_duration
            draft.meta_datas = vis_model.meta_datas
            draft.all_events = vis_model.all_events
            draft.name_cnt_to_polygons = vis_model.name_cnt_to_polygons
            draft.width = vis_model.width
            draft.height = vis_model.height
            draft.scrollValueX = 0
            draft.scrollValueY = 0
            draft.scaleX = 1
            draft.scaleY = 20
        async with self.dm.draft_update(is_json_only=True, need_freeze=True) as draft:
            # infos is too large and don't need to modify
            draft.infos = vis_model.infos
        # self.dm.model.name_cnt_to_polygons = vis_model.name_cnt_to_polygons
        # self.dm.model.all_events = vis_model.all_events
        async with self.dm.draft_update() as draft:
            prev_click_instance_id = self.dm.model.clickInstanceId
            if prev_click_instance_id is not None and prev_click_instance_id < vis_model.info_idxes.shape[0]:
                await self._update_detail(prev_click_instance_id)
                info_idx = int(self.dm.model.info_idxes[prev_click_instance_id]) 
                info = self.dm.model.infos[info_idx]
                draft.clickClusterPoints = vis_model.name_cnt_to_polygons[info.cluster_name][0]
                draft.clickClusterAABBSizes = vis_model.name_cnt_to_polygons[info.cluster_name][1]
            else:
                draft.clickInstanceId = None
                draft.clickClusterPoints = None
                draft.clickClusterAABBSizes = None

    def perf_data_to_vis_model(self, data_list_all_rank: list[list[dict]], max_length: float = 35, depth_padding: float = 0.02, 
            height: float = 0.5, user_scale: Optional[float] = None, max_depth: int = 3):
        t = time.time()
        # data list: chrome trace duration events
        # use name as field
        use_sync_event_count_name = True
        name_to_events: dict[tuple[str, int], list[dict]] = {}
        min_ts_all = math.inf
        max_ts_all = 0
        depth_accum = 0
        for rank, data_list in enumerate(data_list_all_rank):
            data_list = build_depth_from_trace_events(data_list)
            # remove event with depth > 1
            data_list = [ev for ev in data_list if ev["depth"] <= max_depth]
            max_depth_cur = max(ev["depth"] for ev in data_list)
            data_list_all_rank[rank] = data_list
            name_count_local: dict[str, int] = {}
            for ev in data_list:
                # set rank as depth
                ev["depth"] = depth_accum + ev["depth"]
                min_ts_all = min(min_ts_all, ev["ts"])
                max_ts_all = max(max_ts_all, ev["ts"] + ev["dur"])
                name = ev["name"]
                if name not in name_count_local:
                    name_count_local[name] = 0
                name_count_local[name] += 1
                if use_sync_event_count_name:
                    cnt = name_count_local[name]
                else:
                    cnt = -1 
                if (name, cnt) not in name_to_events:
                    name_to_events[(name, cnt)] = []
                ev["rank"] = rank
                ev["cnt"] = cnt
                name_to_events[(name, cnt)].append(ev)
            depth_accum += max_depth_cur
        # print(3, time.time() - t)

        for rank, data_list in enumerate(data_list_all_rank):
            for ev in data_list:
                ev["ts"] -= min_ts_all
        # calc field infos
        if user_scale is None:
            time_scale = 1 / (max_ts_all - min_ts_all)
        else:
            time_scale = 1 / (user_scale * 1e9)
        dur_scale = max_length * time_scale

        field_infos: list[PerfFieldInfo] = []
        for (name, cnt), events in name_to_events.items():
            min_ts = math.inf
            max_ts = 0
            total_dur = 0
            for ev in events:
                ev["field_idx"] = len(field_infos)
                min_ts = min(min_ts, ev["ts"])
                max_ts = max(max_ts, ev["ts"] + ev["dur"])
                total_dur += ev["dur"]
            rate = total_dur / (max_ts - min_ts) / len(events)
            left_line_points = [
                [min_ts * dur_scale , -5000, 0.02],
                [min_ts * dur_scale , 5000, 0.02],
            ]
            right_line_points = [
                [max_ts * dur_scale, -5000, 0.02],
                [max_ts * dur_scale, 5000, 0.02],
            ]
            duration_second = (max_ts - min_ts) / 1e9
            field_infos.append(PerfFieldInfo(name, min_ts, max_ts, duration_second, rate, left_line_points, right_line_points, cnt, f"{name}::{cnt}"))
        # print(4, time.time() - t)
        all_events = sum(data_list_all_rank, [])
        vis_info, name_cnt_to_polygons = _get_vis_data_from_duration_events(all_events, dur_scale, 0, depth_padding, height)
        for info in field_infos:
            info.left_line = [[x - vis_info.width / 2, y, z] for x, y, z in info.left_line]
            info.right_line = [[x - vis_info.width / 2, y, z] for x, y, z in info.right_line]
        # print(5, time.time() - t)
        vis_model = VisModel(
            total_duration=max_ts_all - min_ts_all,
            trs=vis_info.trs,
            colors=vis_info.colors,
            scales=vis_info.scales,
            info_idxes=vis_info.info_idxes,
            rank_ids=vis_info.rank_ids,
            durations=vis_info.durations,
            infos=field_infos,
            all_events=all_events,
            name_cnt_to_polygons=name_cnt_to_polygons,
            width=vis_info.width,
            height=vis_info.height,
            minWidth=vis_info.minWidth,
        )
        return vis_model 


def _main():
    dm = mui.DataModel(VisModel._create_empty_vis_model(), [])
    assert dm._pfl_library is not None 
    print(dm._pfl_library.all_compiled.keys())

if __name__ == "__main__":
    _main()