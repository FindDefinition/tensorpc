# Copyright 2022 Yan Yan
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

import asyncio
import base64
import dataclasses
import enum
import io
import random
import sys
import time
import traceback
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import faker
from tensorpc import PACKAGE_ROOT

import cv2
import imageio
import numpy as np
from faker import Faker
from typing_extensions import Annotated, Literal

import tqdm
import tensorpc
from tensorpc.autossh.scheduler.core import TaskType
from tensorpc.core import prim
from tensorpc.core.asynctools import cancel_task
from tensorpc.core.inspecttools import get_all_members_by_type
from tensorpc.flow import (App, EditableApp, EditableLayoutApp, leaflet,
                           mark_autorun, mark_create_layout, marker, mui,
                           plotly, plus, three, UserObjTree, appctx, V)
from tensorpc.flow.client import AppClient, AsyncAppClient, add_message
from tensorpc.flow.coretypes import MessageLevel, ScheduleEvent
from tensorpc.flow.flowapp.appcore import observe_autorun_function, observe_function, observe_autorun_script
from tensorpc.flow.flowapp.components.mui import (Button, HBox, ListItemButton,
                                                  ListItemText,
                                                  MUIComponentType, VBox,
                                                  VList)
from tensorpc.flow.flowapp.components import typemetas
from tensorpc.flow.flowapp.components.plus.config import ConfigPanel
from tensorpc.flow.sampleapp.sample_reload_fn import func_support_reload
from tensorpc.flow.flowapp.objtree import get_objtree_context
from tensorpc.flow.sampleapp.sample_preview import TestPreview0

class SampleApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.img_ui = mui.Image()
        self.task_loop = mui.TaskLoop("Test", self.on_task_loop)
        self.swi = mui.Switch("Switch Dynamic Layout", self.on_switch)
        self.swi_box = mui.FlexBox()
        self.root.add_layout({
            "btn":
            mui.ButtonGroup({
                "btn0":
                mui.Button("LoadImage",
                           partial(self.on_button_click, name="LoadImage")),
                "btn1":
                mui.Button("SendMessage",
                           partial(self.on_button_click, name="SendMessage")),
                "btn2":
                mui.Button("OpenCam",
                           partial(self.on_button_click, name="OpenCam")),
                "btn3":
                mui.Button("Sleep", partial(self.on_button_click,
                                            name="Sleep")),
            }),
            "btn2":
            mui.Button("Sleep", self.on_one_button_click),
            "swi":
            self.swi,
            "swi_box":
            self.swi_box,
            "inp":
            mui.Input("Image Path", callback=self.on_input_change),
            "img_ui":
            self.img_ui,
            "taskloop":
            self.task_loop,
            "select":
            mui.Select("Select", [("One", 0), ("Two", 1)],
                       self.on_select_change),
        })
        self.img_path = ""
        self.set_init_window_size([480, 640])
        self.task = None
        self.code = ""

    async def on_radio(self, name: str):
        print(name)

    # on_one_button_click
    async def on_one_button_click(self):
        await asyncio.sleep(3)
        print("SLEEP FINISHED")

    async def on_button_click(self, name: str):
        print(name)
        if name == "LoadImage":
            path = Path(self.img_path)
            print(path)
            if path.exists():
                if path.suffix == ".gif":
                    with path.open("rb") as f:
                        data = f.read()
                    await self.img_ui.show_raw(data, "gif")
                else:
                    img = cv2.imread(str(path))
                    # print(type(img))
                    # print(img.shape)
                    await self.img_ui.show(img)
        elif name == "SendMessage":
            add_message("New Message From App!!!", MessageLevel.Warning, [])
        elif name == "OpenCam":
            if self.task is None:
                loop = asyncio.get_running_loop()
                self.task = asyncio.create_task(self._video_task())
            else:
                await cancel_task(self.task)
                self.task = None
            print("?")
        elif name == "Sleep":
            await asyncio.sleep(3)
            print("SLEEP FINISHED")

    async def on_switch(self, checked: bool):
        if checked:
            await self.swi_box.set_new_layout(
                {"wtf": mui.Typography("Dynamic Layout")})
        else:
            await self.swi_box.set_new_layout({})
        print(checked)

    async def on_input_change(self, value: str):
        print(value)
        self.img_path = value

    async def on_code_change(self, value: str):
        self.code = value
        print("CODE CHANGE")

    async def on_slider_change(self, value: Union[int, float]):
        print("SLIDER", value)

    async def on_select_change(self, value: Any):
        print("SELECT", value)

    async def on_task_loop(self):
        await self.task_loop.update_label("TASK")

        print("TASK START!!!")
        async for item in self.task_loop.task_loop(range(5), total=5):
            if item == 3:
                raise ValueError("debug error")
            async for item in self.task_loop.task_loop(range(20), total=20):
                await asyncio.sleep(0.05)
        print("TASK END!!!")
        await self.task_loop.update_label("FINISHED!")

    async def _video_task(self):
        import time
        cap = cv2.VideoCapture(0)
        loop = asyncio.get_running_loop()
        t = time.time()
        fr = 0
        dura = 1
        t = time.time()
        fr = 0
        while True:
            ret, frame = cap.read()
            font = cv2.FONT_HERSHEY_SIMPLEX
            now = datetime.now()
            dt_string = now.strftime("%H:%M:%S")

            frame = cv2.putText(frame, f'{dt_string} FrameRate={1 / dura:.2f}',
                                (10, 30), font, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)
            suffix = "jpg"
            _, img_str = cv2.imencode(".{}".format(suffix), frame)

            await self.img_ui.show_raw(img_str, "jpg")
            dura = time.time() - t
            t = time.time()
            # await asyncio.sleep(0)
            # print(cnt, len(img_str), (time.time() - t) / cnt)


class SampleDictApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.vlist = VList({
            "text0": ListItemText("0"),
            "text1": ListItemText("1"),
        })
        self.vlist.prop(flex=1)
        self.cnt = 2
        self.root.add_layout({
            "btn0":
            Button("CLICK ME", lambda: print("HELLO BTN")),
            # "vlist0": VList({
            #     "btn0": ListItemButton("wtf1", lambda: print("HELLO List BTN1")),
            #     "btn1": ListItemButton("wtf2", lambda: print("HELLO List BTN2")),
            # }),
            "layout0":
            HBox({
                "btn0":
                Button("CLICK ME1", lambda: print("HELLO BTN1")).prop(flex=1),
                "btn1":
                Button("Add", self._ppend_list).prop(flex=1),
            }),
            "l0":
            HBox({
                "items": self.vlist,
                "text": mui.Typography("content").prop(flex=3),
            }).prop(height="100%"),
        })
        self.set_init_window_size([480, 640])

    async def _ppend_list(self):
        await self.vlist.update_childs(
            {f"text{self.cnt}": ListItemText(str(self.cnt))})
        self.cnt += 1


class SamplePlotMetricApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.plots = plus.HomogeneousMetricFigure(300, 300)

        self.root.add_layout({
            "plot0":
            self.plots,
            "btn":
            mui.Button("Increment", self._increment),
            "btn2":
            mui.Button("MaskFirstTrace", self._mask_first_trace)
        })
        self.set_init_window_size([640, 480])
        self.cnt = 0
        self.visible_test = True

    async def _increment(self):
        await self.plots.update_metric(
            self.cnt, "x", "green", {
                "sinx": float(np.sin(self.cnt / 10)),
                "cosx": float(np.cos(self.cnt / 10)),
            })
        await self.plots.update_metric(
            self.cnt, "y", "red", {
                "sinx": float(np.sin((self.cnt + 5) / 10)),
                "cosx": float(np.cos((self.cnt + 5) / 10)),
            })
        await self.plots.update_metric(
            self.cnt, "z", "blue", {
                "sinx": float(np.sin((self.cnt + 8) / 10)),
                "cosx": float(np.cos((self.cnt + 8) / 10)),
            })

        self.cnt += 1

    async def _mask_first_trace(self):
        await self.plots.set_trace_visible("x", not self.visible_test)
        self.visible_test = not self.visible_test


class SampleFlowApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.text = mui.Typography("")
        self.root.add_layout({
            "text": self.text,
        })
        self.set_init_window_size([480, 320])

    async def flow_run(self, ev: ScheduleEvent):
        await self.text.write(str(b"\n".join(ev.data)))
        return None


class SampleEditorApp(EditableApp):
    def __init__(self) -> None:
        super().__init__()
        self.text = mui.Typography("WTF")
        self.root.add_layout({
            "text": self.text,
            "btn": Button("runCB", self.example_cb),
            "btn2": Button("ShowTS", self.show_ts),
        })
        self.root._get_all_nested_childs()
        self.set_init_window_size([480, 320])
        self.init_enable_editor()

    def example_cb(self):
        print("dynamic loadable APP!!!")
        print("example cb 4")
        self.new_method()

    async def show_ts(self):
        await self.text.write(str(time.time_ns()))

    def new_method(self):
        print("new method")


class SampleEditorAppV2(EditableApp):
    def __init__(self) -> None:
        super().__init__(reloadable_layout=True)
        self.text = mui.Typography("WTF")
        # self.root.add_layout({
        #     "text": self.text,
        #     "btn": Button("runCB", self.example_cb),
        #     "btn2": Button("ShowTS", self.show_ts),
        # })
        # self.root._get_all_nested_childs()
        self.set_init_window_size([480, 320])
        # self.init_enable_editor()

    def app_create_layout(self) -> Dict[str, mui.MUIComponentType]:
        return {
            "text": self.text,
            "btn": Button("runCB", self.example_cb),
            "btn2": Button("ShowTS", self.show_ts),
        }

    def example_cb(self):
        print("dynamic loadable APP!!!")
        print("example cb 5")
        self.new_method()

    async def show_ts(self):
        await self.text.write(str(time.time_ns()))

    def new_method(self):
        print("new method")


class SampleThreeApp(EditableApp):
    def __init__(self) -> None:
        super().__init__(reloadable_layout=True)
        self.set_init_window_size([800, 600])
        # makesure three canvas size fit parent.
        self.root.props.minHeight = 0
        # store components here if you want to keep
        # data after reload layout.
        self.points = three.Points(2000000)
        self.lines = three.Segments(20000)

    def app_create_layout(self) -> Dict[str, MUIComponentType]:
        cam = three.PerspectiveCamera(True, fov=75, near=0.1, far=1000)
        cam.prop(position=(0, 0, 20), up=(0, 0, 1))
        # cam = three.OrthographicCamera(True, position=[0, 0, 10], up=[0, 0, 1], near=0.1, far=1000,
        #                               zoom=8.0)
        self.img = three.Image()
        # self.ctrl = three.PointerLockControl().prop(enabled=True)
        self.ctrl = three.CameraControl().prop(damping_factor=1.0)

        # ctrl = three.OrbitControl()
        infgrid = three.InfiniteGridHelper(5, 50, "gray")
        self.lines.prop(lineWidth=1, color="green")
        self.b2d = three.Boxes2D(1000).prop(color="red", alpha=0.5)
        mesh = three.MeshV1(three.BoxGeometry(), three.MeshBasicMaterial())
        mesh.set_pointer_callback(
            on_click=three.EventHandler(lambda x: print(x)))
        self.canvas = three.Canvas({
            "cam": cam,
            "points": self.points,
            "lines": self.lines,
            "ctrl": self.ctrl,
            "axes": three.AxesHelper(10),
            "infgrid": infgrid,
            "img": self.img,
            "b2d": self.b2d,
            "mesh": mesh,
            # "tc": self.scene_ctrl,
            # "box": three.BoundingBox((2, 5, 2), [0, 10, 0], [0, 0, 0.5])
        })
        btn_random_pc = Button("showRandomRPC", self.show_Random_pc)
        return {
            "d3v":
            VBox({
                "d3":
                VBox({
                    "d32": self.canvas,
                }).prop(flex=1, minHeight=0, minWidth=0),
                "btn":
                btn_random_pc,
                # "btn2":
                # Button("rpcTest", self.rpc_test),
                "btn3":
                Button("Reset Camera", self.reset_camera),
            }).prop(flex=1, minHeight=0),
        }

    async def show_Random_pc(self):
        data = np.load(
            "/home/yy/Projects/spconv-release/spconv/test/data/benchmark-pc.npz"
        )

        pc = np.ascontiguousarray(data["pc"])
        # num = 50
        # pc = np.random.uniform(-5, 5, size=[num, 3]).astype(np.float32)
        # for i in range(num):
        #     pc[i] = i
        # print(pc)
        # print(pc.shape)
        # attrs = [str(i) for i in range(num)]
        attrs = pc
        attrFields = ["x", "y", "z"]
        # print("???", pc.size * pc.itemsize)
        await self.points.update_points(pc,
                                        attrs=attrs,
                                        attrFields=attrFields)

        random_lines = np.random.uniform(-5, 5, size=[5, 2,
                                                      3]).astype(np.float32)
        await self.lines.update_lines(random_lines)
        # print("???????", random_lines)
        # with open("/home/yy/Pictures/Screenshot from 2022-02-11 15-10-06.png", "rb") as f:
        #     await self.img.show_raw(f.read(), "png")
        centers = np.array([[0, 0], [2, 2], [3, 3]], np.float32)
        dimensions = np.array([[1, 1], [1, 1], [1, 1]], np.float32)
        attrs = [str(i) for i in range(centers.shape[0])]
        await self.b2d.update_boxes(centers, dimensions)
        print("???")
        await self.b2d.update_object3d(position=(0, 0, 1))

    async def reset_camera(self):
        mat = np.eye(4)
        mat[0, 3] = 1
        mat[1, 3] = 1
        mat[2, 3] = 1

        await self.ctrl.set_cam2world(mat, 50)

    async def show_pc(self, pc):
        intensity = None
        if pc.shape[1] == 4:
            intensity = pc[:, 3]
        await self.points.update_points(pc, intensity=intensity)

    async def show_pc_with_attrs(self, pc, attrs, attrFields):
        intensity = None
        is_nan_mask = np.isnan(pc).any(1)
        is_not_nan_mask = np.logical_not(is_nan_mask)
        num_nan = is_nan_mask.sum()
        if (num_nan) > 0:
            print("NUM NAN", num_nan)
        if pc.shape[1] == 4:
            intensity = np.ascontiguousarray(pc[:, 3])[is_not_nan_mask]
            pc = np.ascontiguousarray(pc[:, :3])[is_not_nan_mask]
        await self.points.update_points(pc,
                                        intensity=intensity,
                                        attrs=attrs[is_not_nan_mask],
                                        attrFields=attrFields)



class SampleThreePointsApp(EditableApp):
    def __init__(self) -> None:
        super().__init__(reloadable_layout=True)
        self.set_init_window_size([800, 600])
        # makesure three canvas size fit parent.
        self.root.props.minHeight = 0
        # store components here if you want to keep
        # data after reload layout.
        self.points = three.Points(5000000)

    def app_create_layout(self) -> mui.LayoutType:
        cam = three.PerspectiveCamera(True, fov=75, near=0.1, far=1000)
        cam.prop(position=(0, 0, 20), up=(0, 0, 1))
        self.ctrl = three.CameraControl().prop(damping_factor=1.0)
        infgrid = three.InfiniteGridHelper(5, 50, "gray")
        self.canvas = three.Canvas({
            "cam": cam,
            "points": self.points,
            "ctrl": self.ctrl,
            "axes": three.AxesHelper(10),
            "infgrid": infgrid,
        })
        self.show_pcs = [
            np.random.uniform(-100, 100, size=[100000, 3]) for _ in range(10)
        ]
        self.offsets = []
        start = 0
        for p in self.show_pcs:
            self.offsets.append((start, start + p.shape[0]))
            start += p.shape[0]
        slider = mui.Slider("Frames", 0,
                            len(self.show_pcs) - 1, 1, self._on_frame_select)
        self.points.prop(points=np.concatenate(self.show_pcs))
        self.prev_range = None
        return [
            mui.VBox([
                mui.VBox([
                    self.canvas,
                ]).prop(flex=1, minHeight=0, minWidth=0),
                slider,
            ]).prop(flex=1, minHeight=0),
        ]

    async def _on_frame_select(self, index):
        if self.prev_range is not None:
            await self.points.set_colors_in_range("#9099ba", *self.prev_range)
        self.prev_range = self.offsets[index]
        await self.points.set_colors_in_range("red", *self.prev_range)

    @mark_autorun
    def wtf(self):
        print("RTD?")


class SampleTestApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.root.add_layout({
            "plot0":
            VBox({
                "asd": mui.Typography("Hello"),
            }).prop(flex=1),
            "btn":
            Button("Show", lambda: print("?"))
        })
        self.set_init_window_size([480, 320])


class SampleThreeHudApp(EditableApp):
    def __init__(self) -> None:
        super().__init__(reloadable_layout=True)
        self.set_init_window_size([800, 600])
        # makesure three canvas size fit parent.
        self.root.props.minHeight = 0
        # store components here if you want to keep
        # data after reload layout.
        self.points = three.Points(2000000)
        self.lines = three.Segments(20000)

    def app_create_layout(self) -> Dict[str, MUIComponentType]:
        cam = three.PerspectiveCamera(True, fov=75, near=0.1, far=1000)
        cam.prop(position=(0, 0, 20), up=(0, 0, 1))
        # cam = three.OrthographicCamera(True, near=0.1, far=1000,
        #                               zoom=8.0)
        # cam.prop(position=[0, 0, 10], up=[0, 0, 1])
        ctrl = three.MapControl()
        # ctrl = three.FirstPersonControl()

        # ctrl = three.OrbitControl()
        infgrid = three.InfiniteGridHelper(5, 50, "gray")
        self.b2d = three.Boxes2D(1000)
        mesh = three.MeshV1(three.RoundedRectGeometry(2, 1.5, 0.5),
                          three.MeshBasicMaterial().prop(color="#393939"))
        mesh.set_pointer_callback(
            on_click=three.EventHandler(lambda x: print(1), True))
        mesh.prop(hover_color="#222222", click_color="#009A63")
        text = three.Text("WTF")
        text.prop(color="red", fontSize=2)
        text.set_pointer_callback(
            on_click=three.EventHandler(lambda x: print(2)))

        self.text2 = three.Text("T")
        self.text2.prop(color="red", fontSize=0.5)
        self.text2.set_pointer_callback(
            on_click=three.EventHandler(lambda x: print(3)))
        material = three.MeshBasicMaterial()
        material.prop(wireframe=True, color="hotpink")
        mesh2 = three.MeshV1(three.BoxGeometry(), material)
        mesh2.set_pointer_callback(
            on_click=three.EventHandler(lambda x: print(4)))
        self.img_path = mui.Input("Image Path")
        self.img = three.Image()
        self.img.set_pointer_callback(on_click=three.EventHandler(
            lambda x: print("IMAGE!!!", self.img_path.value)))
        self.img.prop(scale=(4, 4, 1))
        self.html = three.Html(
            {"btn": mui.Button("RTX", lambda: print("RTX1"))})
        self.html.prop(transform=True, center=False, insideFlex=True)
        self.html2 = three.Html(
            {"btn2": mui.Button("RTX2", lambda: print("RTX2"))})
        res = self.html2.prop(transform=True, center=False, insideFlex=True)
        self.flex_container = three.Flex({
            "mesh1":
            three.ItemBox({
                "mesh03":
                three.Button("RTX", 2, 1, lambda x: print("HELLO")),
            }).prop(centerAnchor=True),
        })
        self.hud = three.Hud({
            "mesh":
            three.ItemBox({
                "mesh0":
                three.Button("RTX", 2, 1, lambda x: print("HELLO")),
            }).prop(centerAnchor=True),
            "mesh1":
            three.ItemBox({
                "mesh0":
                three.ToggleButton("RTX2", 2, 1, lambda x: print("HELLO2", x)),
            }).prop(centerAnchor=True),
            "text":
            three.ItemBox({
                "text0": self.html,
            }).prop(centerAnchor=True),
            "text4":
            three.ItemBox({
                "text0": self.html2,
            }).prop(centerAnchor=True),
            "text3":
            three.ItemBox({
                "text0": three.BoundingBox((2, 5, 2)),
            }).prop(centerAnchor=True),
            "autoreflow":
            three.FlexAutoReflow(),
        }).prop(render_priority=1,
                flexDirection="row",
                justifyContent="flex-start")
        self.canvas = three.Canvas({
            "cam":
            cam,
            "points":
            self.points,
            # "lines": self.lines,
            # "flexdev": three.Flex({
            #     "box1": three.ItemBox({
            #         "text0": three.Text("WTF1").prop(color="red", fontSize=2),
            #     }).prop(centerAnchor=True),
            #     "box2": three.ItemBox({
            #         "text0": three.Text("WTF2").prop(color="red", fontSize=2),
            #     }).prop(centerAnchor=True),
            #     "box3": three.ItemBox({
            #         "text0": three.Text("WTF3").prop(color="red", fontSize=2),
            #     }).prop(centerAnchor=True),
            #     "box4": three.ItemBox({
            #         "text0": three.Text("WTF4").prop(color="red", fontSize=2),
            #     }).prop(centerAnchor=True),

            # }).prop(flexDirection="row", size=(20, 20, 0), position=(-20, -20, 0), flexWrap="wrap"),
            "ctrl":
            ctrl,
            "axes":
            three.AxesHelper(10),
            "infgrid":
            infgrid,
            "b2d":
            self.b2d,
            "mesh":
            mesh2,
            # "img": self.img,
            "text":
            three.Text("WTF").prop(color="red", fontSize=2),
            "box":
            three.BoundingBox((2, 5, 2)).prop(position=(5, 0, 0)),
            #
            # "text0": self.html,
            "hud":
            self.flex_container,
        })
        return {
            "d3v":
            VBox({
                "d3":
                self.canvas,
                "hud":
                mui.VBox({
                    "inp": self.img_path,
                    "btn1": mui.Button("Read Image", self.on_read_img),
                    "btn2": mui.Button("Debug", self.on_debug),
                    "btn3": mui.Typography("Inp", )
                }).prop(position="absolute",
                        top=0,
                        right=0,
                        zIndex=5,
                        justifyContent="flex-end")
            }).prop(position="relative", flex=1, minHeight=0),
        }

    async def on_read_img(self):
        path = self.img_path.value
        with open(path, "rb") as f:
            img_str = f.read()
        await self.img.show_raw(img_str, "jpg")
        await self.text2.update_value("WTF1")

    async def on_debug(self):
        await self.flex_container.set_new_layout({
            "mesh2":
            three.ItemBox({
                "mesh0":
                three.Button("RTX2", 2, 1, lambda x: print("HELLO")),
            }).prop(centerAnchor=True),
        })


class SampleThree2DApp(EditableApp):
    def __init__(self) -> None:
        super().__init__(reloadable_layout=True)
        self.set_init_window_size([800, 600])
        # makesure three canvas size fit parent.
        self.root.props.minHeight = 0
        # store components here if you want to keep
        # data after reload layout.

    def app_create_layout(self) -> Dict[str, MUIComponentType]:
        cam = three.OrthographicCamera(True, near=0.1, far=1000, zoom=50.0)
        cam.prop(position=(0, 0, 10), up=(0, 0, 1))
        ctrl = three.MapControl()
        ctrl.props.enable_rotate = False
        # ctrl = three.FirstPersonControl()
        self.box2d = three.Boxes2D(20000)

        self.box2d.prop(color="aqua",
                        lineColor="red",
                        alpha=0.1,
                        lineWidth=1,
                        hover_line_color="blue",
                        hover_line_width=2)

        self.canvas = three.Canvas({
            "cam":
            cam,
            "ctrl":
            ctrl,
            "b2d":
            self.box2d,
            # "axes": three.AxesHelper(10),
            "btn0":
            three.Button("RTX", 2, 1, self.on_box2d_update),
            "html0":
            three.Html({
                "btn0": mui.Button("RTX", lambda: print("RTX")),
            }).prop(position=(-5, 0, 0), transform=True)
        })
        return {
            "d3v":
            VBox({
                "d3":
                self.canvas,
                "hud":
                mui.VBox({
                    # "update": mui.Button("Box2d", self.on_box2d_update),
                    "btn3": mui.Typography("Inp", )
                }).prop(position="absolute",
                        top=0,
                        right=0,
                        zIndex=5,
                        justifyContent="flex-end")
            }).prop(position="relative", flex=1, minHeight=0),
        }

    async def on_box2d_update(self, ev=None):
        centers = np.random.randint(1, 10, size=[128 * 32,
                                                 2]).astype(np.float32)
        centers = np.arange(0, 128 * 32).astype(np.int32)
        centers = np.stack([centers // 32, centers % 32],
                           axis=1).astype(np.float32)
        centers += [3, 0]
        # centers = np.array([[0, 0], [2, 2], [3, 3]], np.float32)
        dimensions = np.ones((1, ), np.float32)  #  - 0.1
        attrs = [str(i) for i in range(centers.shape[0])]
        await self.box2d.update_boxes(centers, dimensions, attrs=attrs)


class SampleMapApp:

    @mark_create_layout
    def app_create_layout(self):
        google_url = "http://{s}.google.com/vt?lyrs=m&x={x}&y={y}&z={z}"
        esri_url = "https://{s}.tile-cyclosm.openstreetmap.fr/cyclosm/{z}/{x}/{y}.png"
        self.leaflet = leaflet.MapContainer(
            (30, 100), 13, {
                "tile": leaflet.TileLayer(esri_url),
            }).prop(height="100%", flex=3)
        return mui.HBox({
            "control":
            mui.VBox({
                "btn":
                mui.Button("FlyTo", lambda: self.leaflet.fly_to(
                    (40, 100), zoom=10)),
            }).prop(minHeight=0, flex=1),
            "mmap":
            self.leaflet,
        }).prop(width="640px", height="480px")


class TestEnum(enum.Enum):
    A = "1"
    B = "2"
    C = "3"


class TestEnumInt(enum.IntEnum):
    A = 1
    B = 2
    C = 3


@dataclasses.dataclass
class WTF1:
    d: int


@dataclasses.dataclass
class WTF:
    a: int
    b: Union[int, float]
    g: WTF1
    x: Literal["WTF", "WTF1"]
    f: List[Tuple[int, Dict[str, int]]]
    c: bool = False
    e: str = "RTX"
    h: TestEnum = TestEnum.C
    i: V.Annotated[int, V.RangedInt(0, 10)] = 1
    j: TestEnumInt = TestEnumInt.C
    wtf: V.Annotated[float, V.RangedFloat(0, 1, 0.05, "ftw")] = 0.5
    wtfcolor: V.Annotated[str, V.ColorRGB()] = "red"
    v3: three.Vector3Type = (1, 2, 3)
    v4: V.Annotated[three.Vector3Type, V.Vector3(1.0)] = (1, 2, 3)

class SampleConfigApp(EditableApp):
    def __init__(self) -> None:
        super().__init__(reloadable_layout=True)
        self.set_init_window_size([800, 600])
        # makesure three canvas size fit parent.
        # self.root.props.minHeight = 0
        # store components here if you want to keep
        # data after reload layout.
        self.root.props.flexFlow = "row nowrap"
        self.cfg = WTF(1, 0.5, WTF1(2), "WTF", [])

    def app_create_layout(self) -> Dict[str, MUIComponentType]:
        return {
            "control": plus.ConfigPanel(self.cfg),
            "check": mui.Button("Check Config", lambda: print(self.cfg))
        }


class SampleDataControlApp(EditableApp):
    def __init__(self) -> None:
        super().__init__(reloadable_layout=True)
        # makesure three canvas size fit parent.
        # self.root.props.minHeight = 0
        # store components here if you want to keep
        # data after reload layout.
        self.root.props.flexFlow = "row nowrap"

    def app_create_layout(self) -> Dict[str, MUIComponentType]:
        return {
            "btn1": mui.Button("Add Data To Storage", self.add_data),
            "btn2": mui.Button("Read Data From Storage", self.read_data),
        }

    async def add_data(self):
        await self.save_data_storage("default_flow.Data.arr0",
                                     np.zeros((500, 3)))

    async def read_data(self):
        print(await self.read_data_storage("Data.arr0"))


class AutoComputeApp:
    @mark_create_layout
    def create_layout(self):

        self.options = [
            {
                "label": 'The Shawshank Redemption',
                "year": 1994
            },
            {
                "label": 'The Godfather',
                "year": 1972
            },
            {
                "label": 'The Godfather: Part II',
                "year": 1974
            },
            {
                "label": 'The Dark Knight',
                "year": 2008
            },
            {
                "label": 'Monty Python and the Holy Grail',
                "year": 1975
            },
        ]
        ac = mui.MultipleAutocomplete("Movies", self.options).prop(
                variant="checkbox", disableCloseOnSelect=True)
        
        return mui.VBox([
            ac,
        ]).prop(width=640, height=480)


class AnyLayout:
    def __init__(self) -> None:
        super().__init__()

    @marker.mark_create_layout
    def my_layout(self):
        return mui.FlexBox([mui.Button("Hi2345", self.handle_click)])

    def reload_wtf(self):
        print("??4")

    def handle_click(self):
        print("???22X???")
        self.reload_wtf()


class ObjectInspectApp:
    @marker.mark_create_layout
    def my_latout(self):
        self.array = np.random.uniform(-1, 1, size=[500])
        self.non_contig_arr = np.random.uniform(-1, 1, size=[500, 3])[:, 1:]

        try:
            import torch
            self.ten_cpu = torch.rand(1, 3, 224, 224)
            self.ten_gpu = self.ten_cpu.cuda()
            self.ten_gpu_non_c = self.ten_gpu[..., 1:]

        except:
            pass

        return mui.VBox([plus.ObjectInspector(self)]).prop(width=640,
                                                           height=480)


class PointCloudApp:
    @mark_create_layout
    def my_layout(self):
        cam = three.PerspectiveCamera(fov=75, near=0.1, far=1000)
        self.wtfobj = UserObjTree()

        self.canvas = plus.SimpleCanvas(cam, self._on_video_save)
        self.slider = mui.Slider(0,
                                 1,
                                 1,
                                 callback=self._on_slider_select)

        res = mui.VBox([
            mui.Markdown("PointCloud **:red[App]** :dog: :+1: :green[$\\sqrt{3}$]").prop(padding="10px", katex=True, emoji=True),
            mui.Input("hello world"),
            mui.HBox([
                mui.Button("Change Slider Range",
                           self._on_slider_range_change),
                mui.Button("Video", self._on_save_video),
                self.slider.prop(flex=1),
            ]),
            self.canvas.prop(flex=1),
        ]).prop(minHeight=0,
                minWidth=0,
                flex=1,
                width="100%",
                height="100%",
                overflow="hidden")
        res.set_flow_event_context_creator(
            lambda: self.wtfobj.enter_context(self.wtfobj))
        ctx = self.wtfobj.enter_context(self.wtfobj)
        with ctx:
            print("???")
        return res

    async def _on_slider_range_change(self):

        await self.slider.update_ranges(0, 10, 1)

    async def _on_save_video(self):
        c2e_init = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 10],
            [0, 0, 0, 1],
        ], np.float32)
        self.ts = time.time()
        for i in tqdm.tqdm(range(100), total=100):
            c2e_init[0, 3] = i * 0.2
            await self.canvas.set_cam2world(c2e_init.copy(), 1.0, True)
            # await self.canvas.trigger_screen_shot(i)
            img_bytes = await self.canvas.get_screen_shot()
            print(len(img_bytes))
            # with open(f"/home/yy/test/{i}.png", "wb") as f:
            #     f.write(img_bytes)

    async def _on_video_save(self, img_bytes, userdata):
        print(time.time() - self.ts)
        with open(f"/home/yy/test/{userdata}.png", "wb") as f:
            f.write(img_bytes)

    async def _on_slider_select(self, value):
        print(get_objtree_context())

        print("select slider!!!", value)
        # you need to specify a key for a group of point
        # you also need to specify number limit of current point
        points = np.random.uniform(-1, 1, size=[1000, 3]).astype(np.float32)
        # colors can be:
        # 1. [N, 3] float, value range: [0, 1]
        # 2. [N], int8 (intensity), value range: [0, 255]
        # 3. a color string, e.g. red, green
        colors = np.random.uniform(0, 255, size=[1000]).astype(np.uint8)
        # print(colors)
        # colors = np.random.uniform(254, 255, size=[1000]).astype(np.uint8)
        sizes = np.random.uniform(0.5, 10.5, size=[1000]).astype(
            np.float32) * 1

        await self.canvas.show_points("key0",
                                      points,
                                      limit=100000,
                                    #   colors=colors,
                                    #   attrs=points,
                                    #   attr_fields=["x", "y", "z"]
                                      )
        # lines = np.random.uniform(-10, 10, size=[1000, 2, 3]).astype(np.float32)
        
        # await self.canvas.show_lines("lkey0",
        #                               lines,
        #                               limit=800000)

        # boxes: dims, locs, rots, colors (string list, don't support ndarray currently)
        dims = np.random.uniform(1, 2, size=[5, 3])
        locs = np.random.uniform(-5, 5, size=[5, 3])
        rots = np.random.uniform(-1, 1, size=[5, 3])
        rots[:, :2] = 0
        colors = ["red", "yellow", "red", "blue", "yellow"]
        await self.canvas.show_boxes("key0", dims, locs, rots, colors)
        
        voxel_size = 0.1
        size = np.random.randint(100, 300)
        pcs = np.random.randint(-10, 10, size=[size, 3]) * voxel_size
        pcs[:, 0] += 3
        pcs = pcs.astype(np.float32)
        pc_colors = np.random.uniform(0, 255, size=[pcs.shape[0],
                                                    3]).astype(np.uint8)

        await self.canvas.show_voxels("vox0", pcs, pc_colors, voxel_size, 1000)
        random_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        await self.canvas.show_image("img0", random_img, (0, 0, 0), (0, 0, 0), 3)

class PlotApp:
    @mark_create_layout
    def my_layout(self):
        self.plot = plotly.Plotly().prop(
            data=[
                plotly.Trace(x=[1, 2, 3],
                             y=[2, 7, 3],
                             type="scatter",
                             mode="lines")
            ],
            layout=plotly.Layout(
                height=240,
                autosize=True,
                margin=plotly.Margin(l=0, r=0, b=0, t=0),
                xaxis=plotly.Axis(automargin=True),
                yaxis=plotly.Axis(automargin=True),
            ))
        return mui.VBox([
            self.plot,
            Button("Show", self._show_plot),
        ]).prop(width=640, height=480)

    async def _show_plot(self):
        data = [
            plotly.Trace(x=[1, 2, 3],
                         y=[6, 2, 3],
                         type="scatter",
                         mode="lines",
                         marker=plotly.Marker(color="red"))
        ]
        layout = plotly.Layout(
            height=240,
            autosize=True,
            margin=plotly.Margin(l=0, r=0, b=0, t=0),
            xaxis=plotly.Axis(automargin=True),
            yaxis=plotly.Axis(automargin=True),
        )
        await self.plot.show_raw(data, layout)

class ThreadLockerApp:
    @mark_create_layout
    def my_layout(self):
        return mui.VBox([
            Button("enter lock", self._enter_lock),
        ]).prop(width="100%")

    async def _enter_lock(self):
        return await appctx.run_in_executor(self.long_process)
    
    def long_process(self):

        for i in range(100):
            print(i)
            appctx.thread_locker_wait_sync()


class MatchCaseAppBase:
    @marker.mark_create_layout
    def my_layout(self):
        self.switchcase = mui.MatchCase([
            mui.MatchCase.Case("1", mui.Typography("1")),
            mui.MatchCase.Case("2", mui.Typography("2")),
            mui.MatchCase.Case("3", mui.Typography("3")),
            mui.MatchCase.Case(mui.undefined, mui.Typography("default")),
        ])
        self.switchcase_fp = mui.MatchCase([
            mui.MatchCase.ExprCase("x <= 0.2", mui.Typography("1")),
            mui.MatchCase.ExprCase("x >= 0.2 and x < 0.6", mui.Typography("2")),
            mui.MatchCase.ExprCase("x >= 0.6", mui.Typography("3")),
            mui.MatchCase.Case(mui.undefined, mui.Typography("default")),
        ])

        return mui.VBox([
            mui.RadioGroup(["1", "2", "3"], self._on_select),
            mui.Divider(),
            self.switchcase,
            mui.Divider(),
            mui.Slider(0, 1.0, 0.01, self._on_slider),
            self.switchcase_fp,
        ])
    
    async def _on_slider(self, value):
        print(value, 4)
        await self.switchcase_fp.set_condition(value)

    async def _on_select(self, value):
        print(3)
        await self.switchcase.set_condition(value)

class MatchCaseApp(MatchCaseAppBase):
    async def _on_slider(self, value):
        print(value, 2)
        await self.switchcase_fp.set_condition(value)

class DataListApp:
    @marker.mark_create_layout
    def my_layout(self):
        dataList = [
            {
                "id": "1",
                "name": "name1",
                "isCheck": True,
                "tags": [{
                    "id": "0",
                    "tag": "good",
                }],
            },
            {
                "id": "2",
                "name": "name2",
                "isCheck": True,
                "tags": [{
                    "id": "0",
                    "tag": "good",
                }, {
                    "id": "1",
                    "tag": "small",
                }],

            },
            {
                "id": "3",
                "name": "name3",
                "isCheck": False,
                "tags": [{
                    "id": "0",
                    "tag": "fat",
                }],

            },

        ]
        datalist_checkbox = mui.Checkbox()
        datalist_comp = mui.HBox([
            mui.Typography().set_override_props(value="name"),
            datalist_checkbox,
            mui.DataFlexBox(mui.Chip().prop(size="small").set_override_props(label="tag")).set_override_props(dataList="tags").prop(flexFlow="row", virtualized=False),
        ]).prop(alignItems="center")
        datalist = mui.DataFlexBox(datalist_comp).prop(flexFlow="column", dataList=dataList)
        datalist.bind_prop(datalist_checkbox, "isCheck")
        return mui.VBox([
            datalist,
        ])

class DataGridApp:
    def create_data(self, index: int, name: str, calories: float,
        fat: float,
        carbs: float,
        protein: bool):
        return {
            "id": str(index),
            "name": name,
            "calories": calories,
            "fat": fat,
            "carbs": carbs,
            "protein": protein,
            "nested": [
                {
                    "id": str(i),
                    "iq": random.randint(0, 100),
                } for i in range(random.randint(2, 6))
            ]
        }
    
    def create_many_datas(self, count: int):
        fake = Faker()
        for i in range(count):
            yield self.create_data(i, fake.name(), random.randint(100, 300), random.randint(1, 25), random.randint(22, 44), i % 2 == 0)


    @marker.mark_create_layout
    def my_layout(self):
        rows = list(self.create_many_datas(10))
        btn = mui.Button("Edit").prop(loading=False)
        btn.event_click.on_standard(lambda x: print(x.keys)).configure(True)
        cbox = mui.Checkbox("")
        input_cell = mui.Input("dev")
        fat_cell = mui.Slider(0, 100, 1)

        column_defs = [
            mui.DataGrid.ColumnDef("special", specialType=mui.DataGridColumnSpecialType.MasterDetail.value),
            mui.DataGrid.ColumnDef("id", accessorKey="id"),
            mui.DataGrid.ColumnDef("name", accessorKey="name", width=120, editCell=input_cell),
            mui.DataGrid.ColumnDef("calories", accessorKey="calories"),
            mui.DataGrid.ColumnDef("fat", accessorKey="fat", editCell=fat_cell),
            mui.DataGrid.ColumnDef("carbs", accessorKey="carbs"),
            mui.DataGrid.ColumnDef("protein", accessorKey="protein", align="right", cell=cbox),
            mui.DataGrid.ColumnDef("btn", cell=btn),
        ]
        master_detail = mui.JsonViewer().set_override_props(data=".")
        master_detail = mui.VBox([
            mui.Typography("Master Detail").prop(variant="h4"),
            mui.DataGrid([
                mui.DataGrid.ColumnDef("id", accessorKey="id"),
                mui.DataGrid.ColumnDef("iq", accessorKey="iq"),
            ]).prop(idKey="id", rowHover=True, stickyHeader=False, virtualized=False, size="small", enableFilter=False).set_override_props(dataList="nested")
        ]).prop(width="100%", alignItems="center")
        # master_detail = mui.DataFlexBox(mui.HBox([
        #     mui.Typography().set_override_props(value="id"),
        #     mui.Divider(orientation="vertical"),
        #     mui.Typography().set_override_props(value="iq"),
        # ])).set_override_props(dataList="nested").prop(flexFlow="column")
        btn = mui.Button("NAME!")
        md_root = mui.Markdown("")
        btn.event_click.on(lambda: md_root.write("FOOT!"))

        dgrid = mui.DataGrid(column_defs, rows, master_detail, customHeaders=[
            mui.MatchCase([
                mui.MatchCase.Case("name", btn),
                mui.MatchCase.Case(mui.undefined, mui.Typography("Other H!")),
            ]).set_override_props(condition="condition")
        ], customFooters=[
            mui.MatchCase([
                mui.MatchCase.Case("name", md_root),
                mui.MatchCase.Case(mui.undefined, mui.Typography("Other F!")),
            ]).set_override_props(condition="condition")
        ]).prop(idKey="id", rowHover=True, virtualized=True, enableFilter=True)
        # dgrid.event_fetch_detail.on(self._fetch_detail)
        dgrid.bind_prop(cbox, "protein")
        dgrid.bind_prop(input_cell, "name")
        dgrid.bind_prop(fat_cell, "fat")

        return mui.VBox([
            dgrid.prop(stickyHeader=False, virtualized=False, size="small", tableLayout="fixed"),
        ]).prop(width="100%", height="100%", overflow="hidden")

    def _fetch_detail(self, key: str):
        print("WTF", key)
        return {"key": key}

class NumpyDataGridProxy(mui.DataGridProxy):
    def __init__(self, obj: np.ndarray):
        assert obj.ndim == 2
        self.obj = obj
        default_data = {f"{c}": 0 for c in range(obj.shape[1])}
        super().__init__(numRows=obj.shape[0], numColumns=obj.shape[1], defaultData=default_data)

    async def fetch_data(self, start: int, end: int):
        print("fetch", start, end)
        subarr = self.obj[start:end]
        data_list: List[Dict[str, Any]] = []
        for row in range(subarr.shape[0]):
            col = {f"{c}": subarr[row, c] for c in range(subarr.shape[1])}
            col["id"] = str(start + row)
            data_list.append(col)
        return data_list

    def fetch_data_sync(self, start: int, end: int):
        print("fetch", start, end)

        subarr = self.obj[start:end]
        data_list: List[Dict[str, Any]] = []
        for row in range(subarr.shape[0]):
            col = {f"{c}": subarr[row, c] for c in range(subarr.shape[1])}
            col["id"] = str(start + row)
            data_list.append(col)
        return data_list

class DataGridProxyApp:    
    @marker.mark_create_layout
    def my_layout(self):
        arr = np.random.uniform(0, 1, size=[1000, 3])
        data_list: List[Dict[str, Any]] = []
        for row in range(arr.shape[0]):
            col = {f"{c}": arr[row, c] for c in range(arr.shape[1])}
            col["id"] = str(row)
            data_list.append(col)

        column_defs = [
            mui.DataGrid.ColumnDef(id=f"{c}") for c in range(arr.shape[1])
        ]
        dgrid = mui.DataGrid(column_defs, NumpyDataGridProxy(arr)).prop(idKey="id", rowHover=True, virtualized=True, enableFilter=True)
        return mui.VBox([
            dgrid.prop(stickyHeader=False, virtualized=True, size="small"),
        ]).prop(width="100%", height="100%", overflow="hidden")

class MatrixDataGridApp:    
    @marker.mark_create_layout
    def my_layout(self):
        arr = np.random.uniform(0, 1, size=[100, 3])
        arr2 = np.random.randint(0, 100, size=[100, 1]).astype(np.int64)
        column_def = mui.DataGrid.ColumnDef(id=f"unused", specialType=mui.DataGridColumnSpecialType.Number, width=80, specialProps=mui.DataGridColumnSpecialProps(mui.DataGridNumberCell(fixed=8)))
        custom_footers = [
            mui.MatchCase([
                mui.MatchCase.Case("index", mui.Typography("Max")),
                mui.MatchCase.Case(mui.undefined, mui.Typography().set_override_props(value="data").prop(enableTooltipWhenOverflow=True, tooltipEnterDelay=400, fontSize="12px")),
            ]).set_override_props(condition="condition")
        ]
        custom_footer_datas = [{
            "a-0": str(arr.max(0)[0]),
            "a-1": str(arr.max(0)[1]),
            "a-2": str(arr.max(0)[2]),
            "b-0": str(arr2.max(0)[0]),
        }]
        dgrid = mui.MatrixDataGrid(column_def, {"a": arr, "b": arr2}, 
            customFooters=custom_footers, 
            customFooterDatas=custom_footer_datas,
        )
        dgrid.prop(rowHover=True, virtualized=True, enableFilter=True, tableLayout="fixed")
        dgrid.prop(tableSxProps={
            '& .MuiTableCell-sizeSmall': {
                "padding": '2px 2px',
            },
        })
        return mui.VBox([
            dgrid.prop(stickyHeader=False, virtualized=True, size="small"),
        ]).prop(width="100%", height="100%", overflow="hidden")

class TutorialApp:
    @marker.mark_create_layout
    def my_layout(self):
        code = f"""
from tensorpc.flow.flowapp import appctx
from tensorpc.flow.flowapp.components import mui, three, plus
from tensorpc.flow import mark_create_layout
class App:
    @mark_create_layout
    def my_layout(self):
        return mui.VBox([
            mui.Typography("Hello World"),
        ])
        """
        appctx.get_app().set_enable_language_server(True)
        pyright_setting = appctx.get_app().get_language_server_settings()
        pyright_setting.python.analysis.pythonPath = sys.executable
        pyright_setting.python.analysis.extraPaths = [
            str(PACKAGE_ROOT.parent),
        ]

        return mui.VBox([
            plus.AppInMemory("sample_code", code).prop(flex=1),
        ])
    
class LinkDownloadApp:
    @marker.mark_create_layout
    def my_layout(self):
        appctx.get_app().add_file_resource("sample.py", self.sample_file)
        return mui.VBox([
            mui.Markdown("## WTF"),
            mui.Link.app_download_link("click to download", "sample.py"),
        ])

    def sample_file(self):
        return Path(__file__).read_bytes()

class VirtualizedBoxApp:
    
    def create_many_datas(self, count: int):
        fake = Faker()
        for i in range(count):
            yield fake.text()

    @marker.mark_create_layout
    def my_layout(self):
        rows = list(self.create_many_datas(10))
        row_elems = [mui.HBox([mui.Typography(row).prop(variant="body1")]) for row in rows]
        return mui.VBox([
            mui.VirtualizedBox([*row_elems]),
        ]).prop(width="100%", height="100%", overflow="hidden")

class CollectionApp:
    @mark_create_layout
    def my_layout(self):
        self.anylayout = AnyLayout()
        self.monitor = plus.ComputeResourceMonitor()
        self.example_draggable_pc = np.random.uniform(-3, 3, size=[1000, 3])
        self.example_3d_canvas = PointCloudApp()
        self.example_preview_layout = TestPreview0()
        self.example_object_inspector = ObjectInspectApp()
        self.example_pyplot = PlotApp()
        self.example_auto_complete = AutoComputeApp()
        self.editor = mui.MonacoEditor("RTX = 0", "python",
                                       "default_path").prop(height="100%",
                                                            width="100%",
                                                            overflow="hidden")
        self.sm = plus.ScriptManager("CodeStorage")
        self.switchcase = MatchCaseApp()
        self.datalist = DataListApp()
        self.datagrid = DataGridApp()
        nodes = [
            mui.ControlNode("1",
                            "color",
                            mui.ControlNodeType.ColorRGB.value,
                            initValue="#ffffff")
        ]
        self.code = f"""
    @mark_create_layout
    def my_layout(self):
        self.anylayout = AnyLayout()
        self.monitor = plus.ComputeResourceMonitor()
        self.example_draggable_pc = np.random.uniform(-3, 3, size=[1000, 3])
        self.example_3d_canvas = PointCloudApp()
        self.example_object_inspector = ObjectInspectApp()
        self.example_pyplot = PlotApp()
        self.example_auto_complete = AutoComputeApp()
        nodes = [
            mui.ControlNode("1",
                            "color",
                            mui.ControlNodeType.Color.value,
                            initValue="#ffffff")
        ]
        """
        self.wtf = mui.DynamicControls(init=nodes, callback=lambda x: print(x))
        self.cfg = WTF(1, 0.5, WTF1(2), "WTF", [])
        self.wtf2 = plus.ConfigPanelV2(self.cfg, lambda x, y: print(x, y))
        self.wtf3 = plus.ConfigPanel(self.cfg, lambda x, y: print(x, y))

        self.locker = ThreadLockerApp()
        # self.dev_0 = Dev()
        appctx.get_app().set_enable_language_server(True)
        pyright_setting = appctx.get_app().get_language_server_settings()
        pyright_setting.python.analysis.pythonPath = sys.executable
        pyright_setting.python.analysis.extraPaths = [
            str(PACKAGE_ROOT.parent),
        ]

        res = mui.HBox([
            mui.Allotment([
                plus.ObjectInspector(self, use_fast_tree=True).prop(width="100%",
                                                height="100%",
                                                overflow="hidden"),
                mui.HBox([
                    plus.AnyFlexLayout(
                        mui.FlexLayout.VBox([self.sm, mui.FlexLayout.HBox([
                            mui.Markdown("1"),
                            mui.Markdown("2"),
                        ])])),
                ]).prop(width="100%", height="100%", overflow="hidden")
            ]).prop(defaultSizes=[1, 3], width="100%", height="100%")
        ]).prop(flexFlow="row nowrap")

        return res

    @mark_autorun
    async def _autorun_dev(self):
        

        return await self._autorun_dev2()


    @staticmethod
    @observe_autorun_function
    async def _autorun_dev2():
        print("X2ss25sad")

    @staticmethod
    @observe_autorun_script
    async def _autorun_dev3():
        print("BLOCK0 asasff WTF")
        a = 5
        #%% block split
        print("BLOCK2s asfasf asf")

    @staticmethod
    @observe_autorun_script
    def _autorun_dev4():
        print("BLOCK0 asasff WTF")
        appctx.inspector.set_custom_layout_sync(mui.VBox([
            mui.Markdown("## :red[WTF2]"),
        ]))
        a = 5
        #%% block split
        print("BLOCK2s asfasf asf")

class SchedulerTest:
    @mark_create_layout
    def my_layout(self):

        return mui.VBox([
            Button("Show", self._submit_simple_task),
        ])

    async def _submit_simple_task(self):
        schr = appctx.find_component(plus.TmuxScheduler)
        assert schr is not None
        task1 = plus.Task(
            TaskType.FunctionId,
            "tensorpc.autossh.scheduler.test_data::simple_task_with_client",
            [{}])
        task1.id = "test1"
        task1.num_gpu_used = 3
        task2 = plus.Task(
            TaskType.FunctionId,
            "tensorpc.autossh.scheduler.test_data::simple_task_with_client",
            [{}])
        task2.id = "test2"
        task2.num_gpu_used = 4

        task3 = plus.Task(
            TaskType.FunctionId,
            "tensorpc.autossh.scheduler.test_data::simple_task_with_client",
            [{}])
        task3.id = "test3"
        task3.num_gpu_used = 1

        # task1.keep_tmux_session = False
        await schr.submit_task(task1)
        await schr.submit_task(task2)
        await schr.submit_task(task3)


class SchedulerApp:
    @mark_create_layout
    def my_layout(self):

        # use a function to protect your password (only stored in master disk).
        self.scheduler = plus.TmuxScheduler(
            lambda: appctx.get_app().get_ssh_node_data("Local"))
        self.scheduler_test = mui.flex_wrapper(SchedulerTest())
        res = plus.InspectPanel(
            self,
            mui.FlexLayout.Row([
                mui.FlexLayout.Tab(self.scheduler),
                mui.FlexLayout.Tab(self.scheduler_test),
            ]))
        return res


class CameraBenchmarkApp:
    @mark_create_layout
    def my_layout(self):
        self.img_ui = mui.Image()
        self.task = None
        return mui.VBox([
            mui.Button("OpenCam",
                        self.on_button_click),
            self.img_ui,
        ]).prop(width="400px", height="400px")

    async def on_button_click(self):
        if self.task is None:
            loop = asyncio.get_running_loop()
            self.task = asyncio.create_task(self._video_task())
        else:
            await cancel_task(self.task)
            self.task = None

    async def _video_task(self):
        import time
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        loop = asyncio.get_running_loop()

        t = time.time()
        fr = 0
        dura = 1
        t = time.time()
        fr = 0
        while True:
            t3 = time.time()
            ret, frame = cap.read()
            dura_cv = time.time() - t3
            font = cv2.FONT_HERSHEY_SIMPLEX
            now = datetime.now()
            dt_string = now.strftime("%H:%M:%S")

            frame = cv2.putText(frame, f'{dt_string} FrameRate={1 / dura:.2f}',
                                (10, 30), font, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)
            suffix = "jpg"
            t2 = time.time()
            _, img_str = cv2.imencode(".{}".format(suffix), frame)
            await self.img_ui.show_raw(img_str, "jpg")
            dura_encode = time.time() - t2

            dura = time.time() - t
            t = time.time()
            # await asyncio.sleep(0)
            # print(dura, dura_encode, dura_cv, len(img_str), frame.shape)

class TestNodeNode0(UserObjTree):
    def __init__(self) -> None:
        super().__init__()

    @marker.mark_create_preview_layout
    def layout_func(self):
        return mui.VBox([
            mui.Button("WTF"),
            mui.Markdown("## 6")
        ])

class TestNodeRoot(UserObjTree):
    def __init__(self) -> None:
        super().__init__()
        self.node0 = TestNodeNode0()
        self._childs["node0"] = self.node0

    @marker.mark_create_preview_layout
    def layout_func(self):
        return mui.VBox([
            mui.Button("ROOT"),
            mui.Markdown("## ROOT132")
        ])

class GridPreviewLayoutApp:
    @mark_create_layout
    def my_layout(self):
        root = TestNodeRoot()
        reload_mgr = appctx.get_reload_manager()

        return mui.HBox([plus.GridPreviewLayout({
            "root": root,
            "root.node0": root.get_childs()["node0"]
        })]).prop(width="100%")


if __name__ == "__main__":
    from pydantic import (
        BaseModel,
        GetCoreSchemaHandler,
        GetJsonSchemaHandler,
        TypeAdapter,
        ValidationError,
    )

    props = mui.MultipleAutocompleteProps()
    props.variant = "wtf" 

    TypeAdapter(mui.MultipleAutocompleteProps).validate_python(props)
    # ac = mui.MultipleAutocomplete("Movies", []).prop(
    #             variant="checkbox", disableCloseOnSelect=True)

    
