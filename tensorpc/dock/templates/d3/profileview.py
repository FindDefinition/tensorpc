import json
import math

import yaml
from tensorpc.constants import TENSORPC_DEV_SECRET_PATH
from tensorpc.dock import mui, three, plus, appctx, mark_create_layout
import dataclasses
from typing import Any 
import numpy as np

from tensorpc.utils.perfetto_colors import perfetto_slice_to_color 
from tensorpc.apps.dbg.components.traceview import parse_viztracer_trace_events_to_raw_tree
@dataclasses.dataclass
class DataModel:
    data: Any
    trs: np.ndarray
    colors: np.ndarray
    scales: np.ndarray
    polygons: np.ndarray
    infos: list[Any]
    label: Any = None

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


def _get_vis_data_from_duration_events(duration_events: list[dict], dur_scale: float, min_ts: int, depth_padding: float, height: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    for event in duration_events:
        x = (event["ts"] - min_ts + event["dur"] / 2) * dur_scale
        y = depth_padding + (event["depth"]) * (height + depth_padding)
        z = 0
        trs.append([x, y, z])
        colors.append([*perfetto_slice_to_color(event["name"]).base.rgb])
        scales.append([event["dur"] * dur_scale, height, 1])
    return np.array(trs, dtype=np.float32), np.array(colors, dtype=np.float32), np.array(scales, dtype=np.float32)

class App:
    @mark_create_layout
    def my_layout(self):
        with open(TENSORPC_DEV_SECRET_PATH, "r") as f:
            path = yaml.safe_load(f)["perfetto_debug"]["trace_path"]
        with open(path, "r") as f:
            trace = json.load(f)
        trace_events = trace["traceEvents"]

        _, duration_events, _ = parse_viztracer_trace_events_to_raw_tree(trace_events, add_depth_to_event=True)
        min_ts = math.inf
        max_ts = 0
        for event in duration_events:
            min_ts = min(min_ts, event["ts"])
            max_ts = max(max_ts, event["ts"] + event["dur"])
        max_length = 200
        for ev in duration_events:
            ev["ts"] -= min_ts
        print(min_ts, max_ts)
        dur_scale = max_length / (max_ts - min_ts)
        depth_padding = 0.1
        cam = three.PerspectiveCamera(fov=75, near=0.1, far=1000).prop(position=(0, 0, 5))
        trs = np.array([
            [0, 0, 0],
            [2, 0, 0],
        ], dtype=np.float32)
        colors = np.array([
            [*perfetto_slice_to_color("hello").base.rgb],
            [*perfetto_slice_to_color("world").base.rgb],
        ], dtype=np.float32) / 255
        scales = np.array([
            [2, 1, 1],
            [1, 2, 2],
        ], dtype=np.float32)
        trs, colors, scales = _get_vis_data_from_duration_events(duration_events, dur_scale, 0, depth_padding, 0.5)
        colors /= 255
        polygons = _get_polygons_from_pos_and_scales(trs, scales)
        print(trs.shape, colors.shape, scales.shape)
        print(trs)
        print(colors)
        print(scales)
        infos = [
            {
                "name": "hello",
                "start": 0,
                "end": 1,
            },
            {
                "name": "world",
                "start": 1,
                "end": 2,
            }
        ]
        infos = duration_events
        boxmesh = three.InstancedMesh(trs, 150000, [
                three.PlaneGeometry(),
                three.MeshStandardMaterial(),
            ]).prop(castShadow=True)
        line = three.Line([(0, 0, 0), (1, 1, 1)]).prop(color="red", lineWidth=2)
        line_cond = mui.MatchCase.binary_selection(True, line)
        canvas = three.Canvas([
            cam,
            three.CameraControl().prop(makeDefault=True),
            three.AmbientLight(intensity=3.14),
            three.PointLight().prop(position=(13, 3, 5),
                                    castShadow=True,
                                    color=0xffffff,
                                    intensity=500),
            three.Mesh([
                three.PlaneGeometry(1000, 1000),
                three.MeshStandardMaterial().prop(color="#f0f0f0"),
            ]).prop(receiveShadow=True, position=(0.0, 0.0, -2)),
            boxmesh,
            # three.Bvh([boxmesh]).prop(verbose=True),
            line_cond,
        ]).prop(enablePerf=True)
        dm = mui.DataModel(DataModel(None, trs, colors, scales, polygons, infos), [])
        draft = dm.get_draft()
        jv = mui.JsonViewer()
        boxmesh.event_move.configure(dont_send_to_backend=True)
        boxmesh.event_move.add_frontend_draft_change(draft, "data")
        boxmesh.event_leave.configure(dont_send_to_backend=True)
        boxmesh.event_leave.add_frontend_draft_set_none(draft, "data")
        boxmesh.bind_fields(transforms="$.trs", colors="$.colors", scales="$.scales")
        label_box = mui.VBox([
            mui.JsonViewer().bind_fields(data="getitem($.infos, $.data.instanceId)"),
        ]).prop(width="300px", position="absolute", backgroundColor="rgba(255, 255, 255, 0.5)", pointerEvents="none")
        label_box.bind_fields(top="not_null($.data.offset[1], `0`) + `5`", left="not_null($.data.offset[0], `0`) + `5`")
        label = mui.MatchCase.binary_selection(True, label_box)
        label.bind_fields(condition="$.data != `null`")
        line.bind_fields(points="npToList(npGetSubArray($.polygons, not_null($.data.instanceId, `0`)))")
        line_cond.bind_fields(condition="$.data != `null`")

        dm.init_add_layout([
            canvas.prop(flex=2, shadows=True),
            mui.HBox([
                jv
            ]).prop(flex=1, overflow="auto"),
            label,
        ])
        jv.bind_fields(data="$.data.offset")
        return mui.HBox([
            dm
        ]).prop(minHeight=0,
                minWidth=0,
                width="100%",
                height="100%",
                overflow="hidden",
                position="relative")
