import bisect
import copy
import json
import math

import yaml
from tensorpc.constants import TENSORPC_DEV_SECRET_PATH
from tensorpc.core.datamodel.draft import DraftFieldMeta
from tensorpc.core.datamodel.events import DraftChangeEvent
from tensorpc.dock import mui, three, plus, appctx, mark_create_layout
import dataclasses
from typing import Any, Optional 
from typing_extensions import Annotated
import numpy as np
import tensorpc.core.datamodel as D

from tensorpc.utils.perfetto_colors import perfetto_slice_to_color 
from tensorpc.apps.dbg.components.traceview import parse_viztracer_trace_events_to_raw_tree

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


@dataclasses.dataclass
class VisInfo:
    trs: np.ndarray
    colors: np.ndarray
    scales: np.ndarray
    polygons: np.ndarray
    info_idxes: np.ndarray
    rank_ids: np.ndarray
    durations: np.ndarray

@dataclasses.dataclass
class VisModel(VisInfo):
    total_duration: float
    infos: list[PerfFieldInfo]
    hoverData: Any = None
    clickInfo: Any = None

    hoverInfoId: Optional[int] = None
    clickInstanceId: Optional[int] = None

    step: int = -1
    whole_scales: list[float] = dataclasses.field(default_factory=lambda: [1.0, 1.0, 1.0])
    meta_datas: Annotated[list[Any], DraftFieldMeta(is_external=True)] = dataclasses.field(default_factory=list)

def _get_polygons_from_pos_and_scales(trs: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """
    Get the polygons from the positions and scales.
    :param trs: The positions of the boxes.
    :param scales: The scales of the boxes.
    :return: The polygons of the boxes.
    """
    polygons = []
    for i in range(len(trs)):
        x, y, z = trs[i]
        sx, sy, sz = scales[i]
        polygons.append([
            [x - sx / 2, y - sy / 2, z + 0.01],
            [x + sx / 2, y - sy / 2, z + 0.01],
            [x + sx / 2, y + sy / 2, z + 0.01],
            [x - sx / 2, y + sy / 2, z + 0.01],
            # close the polygon
            [x - sx / 2, y - sy / 2, z + 0.01],
        ])
    return np.array(polygons)


def _get_vis_data_from_duration_events(duration_events: list[dict], dur_scale: float, 
        min_ts: int, depth_padding: float, height: float) -> VisInfo:
    """
    Get the vis data from the duration events.
    :param duration_events: The duration events.
    :param dur_scale: The scale of the duration.
    :param depth_padding: The padding of the depth.
    :param height: The height of the boxes.
    :return: The positions, colors and scales of the boxes.
    """
    trs = []
    colors = []
    scales = []
    info_idxes = []
    durations = []
    max_y_bound = 0
    for event in duration_events:
        x = (event["ts"] - min_ts + event["dur"] / 2) * dur_scale
        y = depth_padding + (event["depth"] - 0.5) * (height + depth_padding) - 1.5 * depth_padding
        cur_y_bound = y + height
        max_y_bound = max(max_y_bound, cur_y_bound)
        z = 0
        trs.append([x, -y, z])
        info_idxes.append(event["field_idx"])
        colors.append([*perfetto_slice_to_color(event["name"]).base.rgb])
        scales.append([event["dur"] * dur_scale, height, 1])
        durations.append(event["dur"] / 1e9)
    return VisInfo(
        trs=np.array(trs, dtype=np.float32),
        colors=np.array(colors, dtype=np.float32) / 255,
        scales=np.array(scales, dtype=np.float32),
        polygons=_get_polygons_from_pos_and_scales(np.array(trs, dtype=np.float32), np.array(scales, dtype=np.float32)),
        info_idxes=np.array(info_idxes, dtype=np.int32),
        rank_ids=np.array([event["rank"] for event in duration_events], dtype=np.int32),
        durations=np.array(durations, dtype=np.float32),
    )

class PerfMonitor(mui.FlexBox):
    def __init__(self):
        trs_empty = np.zeros((0, 3), dtype=np.float32)
        boxmesh = three.InstancedMesh(trs_empty, 150000, [
            three.PlaneGeometry(),
            three.MeshBasicMaterial(),
        ]) # .prop(castShadow=True)
        line = three.Line([(0, 0, 0), (1, 1, 1)]).prop(color="red", lineWidth=2)
        line_cond = mui.MatchCase.binary_selection(True, line)

        line_start = three.Line([(0, 0, 0), (1, 1, 1)]).prop(color="gray", lineWidth=1, dashed=True, dashSize=0.5, gapSize=0.5)
        line_start_cond = mui.MatchCase.binary_selection(True, line_start)
        line_end = three.Line([(0, 0, 0), (1, 1, 1)]).prop(color="gray", lineWidth=1, dashed=True, dashSize=0.5, gapSize=0.5)
        line_end_cond = mui.MatchCase.binary_selection(True, line_end)
        line_select = three.Line([(0, 0, 0), (1, 1, 1)]).prop(color="blue", lineWidth=2)
        line_select_cond = mui.MatchCase.binary_selection(True, line_select)

        # canvas = three.Canvas([
        #     # three.OrbitControl().prop(makeDefault=True),
        #     # three.AxesHelper(10),
        #     three.uikit.Fullscreen([
        #         three.uikit.Container([
        #             three.uikit.Content([
        #                 boxmesh,
        #                 line_cond,
        #                 line_start_cond,
        #                 line_end_cond,
        #                 # boxmesh_container,
        #             ]).prop(width="98%", keepAspectRatio=True) # .prop(flexGrow=1, margin=32)
        #         ]).prop(flexGrow=1, flexShrink=1, flexBasis=0, overflow="scroll", scrollbarWidth=4, scrollbarColor="red"),
        #     ]).prop(sizeX=8, sizeY=4, flexDirection="row", )
        # ]).prop(allowKeyboardEvent=False, localClippingEnabled=True)
        self._cam_ctrl = three.CameraControl().prop(makeDefault=True, mouseButtons=three.MouseButtonConfig(left="none"))
        canvas = three.Canvas([
            # cam,
            self._cam_ctrl,
            three.InfiniteGridHelper(5, 50, "gray"),
            # three.AxesHelper(10),
            # three.AmbientLight(intensity=3.14),
            # three.PointLight().prop(position=(13, 3, 5),
            #                         castShadow=True,
            #                         color=0xffffff,
            #                         intensity=500),
            # three.Mesh([
            #     three.PlaneGeometry(1000, 1000),
            #     three.MeshStandardMaterial().prop(color="#f0f0f0"),
            # ]).prop(receiveShadow=True, position=(0.0, 0.0, -2)),
            three.Group([
                boxmesh,
                line_cond,
                line_start_cond,
                line_end_cond,
                three.Group([
                    line_select_cond
                ]).prop(position=(0, 0, 0.015)),
            ]).prop(position=(-17, 17, 1))
        ]).prop(enablePerf=False, allowKeyboardEvent=True, localClippingEnabled=True)
        canvas.prop(cameraProps=three.PerspectiveCameraProps(position=(0, 0, 25)))
        canvas.prop(menuItems=[
            mui.MenuItem("reset", "reset"),
            mui.MenuItem("clear", "clear"),
        ])
        canvas.event_context_menu.on(self._on_menu_select)

        empty_model = self._create_empty_vis_model()
        dm = mui.DataModel(empty_model, [])
        draft = dm.get_draft()
        dm.install_draft_change_handler(draft.clickInstanceId, self._on_click_instance_id_change)
        boxmesh.event_move.configure(dont_send_to_backend=True)
        boxmesh.event_move.add_frontend_draft_change(draft, "hoverData", r"{offset: offset, instanceId: instanceId, dur: ndarray_getitem(__TARGET__.durations, not_null(instanceId, `0`)), info: getitem(__TARGET__.infos, ndarray_getitem(__TARGET__.info_idxes, not_null(instanceId, `0`))) }")
        boxmesh.event_leave.configure(dont_send_to_backend=True)
        boxmesh.event_leave.add_frontend_draft_set_none(draft, "hoverData")
        boxmesh.event_click.on_standard(self._on_click)

        canvas.event_keyboard_hold.configure(dont_send_to_backend=True, key_codes=["KeyW", "KeyS"])

        canvas.event_keyboard_hold.add_frontend_draft_change(draft.whole_scales, 0, f"clamp(__PREV_VALUE__ + "
            "matchcase_varg(code, 'KeyW', deltaTime * `0.01`, 'KeyS', -deltaTime * `0.01`), "
            "`1.0`, `10.0`)")

        boxmesh.bind_fields(transforms="$.trs", colors="$.colors", scales="$.scales", scale="$.whole_scales")
        # devmesh.bind_fields(scale="$.whole_scales")
        label_box = mui.VBox([
            mui.Typography("")
                .prop(variant="caption")
                .bind_fields(value="cformat('%s[%d](dur=%.3fs, alldur=%.3fs)', hoverData.info.name, ndarray_getitem($.rank_ids, not_null($.hoverData.instanceId, `0`)), hoverData.dur, hoverData.info.duration)"),
            # mui.JsonViewer().bind_fields(data="getitem($.infos, ndarray_getitem($.info_idxes, not_null($.hoverData.instanceId, `0`)))"),
        ]).prop(width="300px", position="absolute", backgroundColor="rgba(255, 255, 255, 0.5)", pointerEvents="none")
        label_box.bind_fields(top="not_null($.hoverData.offset[1], `0`) + `5`", left="not_null($.hoverData.offset[0], `0`) + `5`")
        label = mui.MatchCase.binary_selection(True, label_box)
        label.bind_fields(condition="$.hoverData != `null`")
        line.bind_fields(points="ndarray_getitem($.polygons, not_null($.hoverData.instanceId, `0`))", scale="$.whole_scales")
        line_cond.bind_fields(condition="$.hoverData != `null`")

        line_select.bind_fields(points="ndarray_getitem($.polygons, not_null($.clickInstanceId, `0`))", scale="$.whole_scales")
        line_select_cond.bind_fields(condition="$.clickInstanceId != `null`")

        line_start.bind_fields(points="hoverData.info.left_line", scale="$.whole_scales")
        line_start_cond.bind_fields(condition="$.hoverData != `null`")
        line_end.bind_fields(points="hoverData.info.right_line", scale="$.whole_scales")
        line_end_cond.bind_fields(condition="$.hoverData != `null`")
        header = mui.Typography().prop(variant="caption")
        self.history: list[VisModel] = []
        slider = mui.BlenderSlider(0, 1, 1, self._select_vis_model)
        slider.prop(isInteger=True, showControlButton=True)
        # select = mui.Autocomplete("history", [], self._select_vis_model).prop(size="small", textFieldProps=mui.TextFieldProps(muiMargin="dense"))
        self.history_slider = slider
        self._header = header
        self._detail_viewer = mui.JsonViewer()
        dm.init_add_layout([
            mui.VBox([
                mui.HBox([
                    header.prop(flex=1),
                    mui.VDivider(),
                    slider.prop(flex=2),
                ]),
                mui.HDivider(),
                mui.HBox([
                    canvas.prop(flex=1),
                    label,
                ]).prop(minHeight=0,
                        minWidth=0,
                        overflow="hidden",
                        flex=1,
                        position="relative")
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

    def _create_empty_vis_model(self):
        trs_empty = np.zeros((0, 3), dtype=np.float32)
        colors_empty = np.zeros((0, 3), dtype=np.float32)
        scales_empty = np.zeros((0, 3), dtype=np.float32)
        polygons_empty = np.zeros((0, 5, 3), dtype=np.float32)
        indexes_empty = np.zeros((0,), dtype=np.int32)
        durs_empty = np.zeros((0,), dtype=np.float32)
        return VisModel(trs_empty, colors_empty, scales_empty, polygons_empty, indexes_empty, indexes_empty, durs_empty, 0, [])

    async def _on_menu_select(self, value: str):
        if value == "reset":
            await self._cam_ctrl.reset_camera()
        elif value == "clear":
            await self.clear()

    async def clear(self):
        self.history.clear()
        vis_model = self._create_empty_vis_model()
        await self._sync_history_select()
        async with self.dm.draft_update() as draft:
            draft.trs = vis_model.trs 
            draft.colors = vis_model.colors 
            draft.scales = vis_model.scales 
            draft.infos = vis_model.infos 
            draft.polygons = vis_model.polygons
            draft.info_idxes = vis_model.info_idxes 
            draft.rank_ids = vis_model.rank_ids 
            draft.durations = vis_model.durations
            draft.total_duration = vis_model.total_duration
            draft.meta_datas = vis_model.meta_datas
        await self._header.write("")
        await self._detail_viewer.write(None)

    async def _on_click(self, ev: mui.Event):
        instance_id = ev.data.instanceId 
        self.dm.get_draft().clickInstanceId = instance_id 

    async def _update_detail(self, instance_id: Optional[int]):
        if instance_id is not None and instance_id < self.dm.model.info_idxes.shape[0]:
            info_idx = int(self.dm.model.info_idxes[instance_id]) 
            rank = int(self.dm.model.rank_ids[instance_id])
            info = self.dm.model.infos[info_idx]
            res = {
                "name": info.name, 
                "rank": rank,
                "duration": round(float(self.dm.model.durations[instance_id]), 4),
                "all_duration": round(info.duration, 4),
                "rate": round(info.rate, 4),
            }
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

    async def append_perf_data(self, step: int, data_list_all_rank: list[list[dict]], meta_datas: list[Any], scale: Optional[float] = None):
        vis_model = self.perf_data_to_vis_model(data_list_all_rank, user_scale=scale)
        # insert step sorted
        # calc insert loc by bisect 
        vis_model.meta_datas = meta_datas
        vis_model.step = step

        bisect.insort(self.history, vis_model, key=lambda v: v.step)
        # self.history[step] = vis_model
        await self._sync_history_select()

    async def _sync_history_select(self):
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
            draft.infos = vis_model.infos 
            draft.polygons = vis_model.polygons
            draft.info_idxes = vis_model.info_idxes 
            draft.rank_ids = vis_model.rank_ids 
            draft.durations = vis_model.durations
            draft.total_duration = vis_model.total_duration
            draft.meta_datas = vis_model.meta_datas
            prev_click_instance_id = self.dm.model.clickInstanceId
            if prev_click_instance_id is not None and prev_click_instance_id < vis_model.info_idxes.shape[0]:
                await self._update_detail(prev_click_instance_id)
            else:
                draft.clickInstanceId = None

    def perf_data_to_vis_model(self, data_list_all_rank: list[list[dict]], max_length: float = 35, depth_padding: float = 0.02, 
            height: float = 0.5, user_scale: Optional[float] = None, max_depth: int = 3):
        data_list_all_rank = copy.deepcopy(data_list_all_rank)
        # import rich 
        # rich.print(data_list_all_rank)
        # data list: chrome trace duration events
        # use name as field
        name_to_events: dict[str, list[dict]] = {}
        min_ts_all = math.inf
        max_ts_all = 0
        depth_accum = 1
        for rank, data_list in enumerate(data_list_all_rank):
            _, data_list, _ = parse_viztracer_trace_events_to_raw_tree(data_list, add_depth_to_event=True, parse_viztracer_name=False)
            # remove event with depth > 1
            data_list = [ev for ev in data_list if ev["depth"] <= max_depth]
            max_depth = max(max_depth, max(ev["depth"] for ev in data_list))
            data_list_all_rank[rank] = data_list
            for ev in data_list:
                # set rank as depth
                ev["depth"] = depth_accum + ev["depth"]
                min_ts_all = min(min_ts_all, ev["ts"])
                max_ts_all = max(max_ts_all, ev["ts"] + ev["dur"])
                name = ev["name"]
                if name not in name_to_events:
                    name_to_events[name] = []
                ev["rank"] = rank
                name_to_events[name].append(ev)
            depth_accum += max_depth
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
        for name, events in name_to_events.items():
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
                [min_ts * dur_scale, -5000, 0.02],
                [min_ts * dur_scale, 5000, 0.02],
            ]
            right_line_points = [
                [max_ts * dur_scale, -5000, 0.02],
                [max_ts * dur_scale, 5000, 0.02],
            ]
            duration_second = (max_ts - min_ts) / 1e9
            name_slice = f"{name}"
            field_infos.append(PerfFieldInfo(name_slice, min_ts, max_ts, duration_second, rate, left_line_points, right_line_points))
        all_events = sum(data_list_all_rank, [])
        vis_info = _get_vis_data_from_duration_events(all_events, dur_scale, 0, depth_padding, height)
        vis_model = VisModel(
            total_duration=max_ts_all - min_ts_all,
            trs=vis_info.trs,
            colors=vis_info.colors,
            scales=vis_info.scales,
            polygons=vis_info.polygons,
            info_idxes=vis_info.info_idxes,
            rank_ids=vis_info.rank_ids,
            durations=vis_info.durations,
            infos=field_infos,
        )
        return vis_model 