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
import time
import traceback
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import imageio
import numpy as np
from faker import Faker
from typing_extensions import Literal

import tqdm

import tensorpc
from tensorpc.core import prim
from tensorpc.core.asynctools import cancel_task
from tensorpc.core.inspecttools import get_all_members_by_type
from tensorpc.flow import (App, EditableApp, EditableLayoutApp, leaflet,
                           mark_autorun, mark_create_layout, marker, mui,
                           plotly, plus, three)
from tensorpc.flow.client import AppClient, AsyncAppClient, add_message
from tensorpc.flow.coretypes import MessageLevel, ScheduleEvent
from tensorpc.flow.flowapp.components.mui import (Button, HBox, ListItemButton,
                                                  ListItemText,
                                                  MUIComponentType, VBox,
                                                  VList)
from tensorpc.flow.flowapp.components.plus.config import ConfigPanel
from tensorpc.flow.sampleapp.sample_reload_fn import func_support_reload, Dev

class SampleApp(App):

    def __init__(self) -> None:
        super().__init__()
        self.img_ui = mui.Images()
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
            "slider":
            mui.Slider("Slider", 0, 100, 1, self.on_slider_change),
            "select":
            mui.Select("Select", [("One", 0), ("Two", 1)],
                       self.on_select_change),
            "rg":
            mui.RadioGroup(["Option1", "Option2"], True, self.on_radio),
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
        self.root.props.min_height = 0
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
        self.lines.prop(line_width=1, color="green")
        self.b2d = three.Boxes2D(1000).prop(color="red", alpha=0.5)
        mesh = three.Mesh(three.BoxGeometry(), three.MeshBasicMaterial())
        mesh.set_pointer_callback(
            on_click=three.EventHandler(lambda x: print(x)))
        self.canvas = three.ThreeCanvas({
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
                }).prop(flex=1, min_height=0, min_width=0),
                "btn":
                btn_random_pc,
                "btn2":
                Button("rpcTest", self.rpc_test),
                "btn3":
                Button("Reset Camera", self.reset_camera),
            }).prop(flex=1, min_height=0),
        }

    async def show_Random_pc(self):
        # data = np.load(
        #     "/home/tusimple/tusimple/spconv/test/data/benchmark-pc.npz")
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
        attr_fields = ["x", "y", "z"]
        # print("???", pc.size * pc.itemsize)
        await self.points.update_points(pc,
                                        attrs=attrs,
                                        attr_fields=attr_fields)

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

    async def show_pc_with_attrs(self, pc, attrs, attr_fields):
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
                                        attr_fields=attr_fields)

    async def rpc_test(self):
        print(self._get_app_dynamic_cls().module_key)
        print("???")
        data = np.load(
            "/home/tusimple/tusimple/spconv/test/data/benchmark-pc.npz")
        pc = np.ascontiguousarray(data["pc"])
        print(pc.shape)
        # tmp = pc[:, 0].copy()
        # pc[:, 0] = pc[:, 1]
        # pc[:, 1] = tmp
        addr = "localhost:43727"
        master_addr = "10.130.51.54:51051"
        # we can't use sync code here which will cause deadlock
        async with AsyncAppClient(master_addr, "default_flow",
                                  "D3VisApp") as client:
            await client.app_remote_call("show_pc", pc)


class SampleThreePointsApp(EditableApp):

    def __init__(self) -> None:
        super().__init__(reloadable_layout=True)
        self.set_init_window_size([800, 600])
        # makesure three canvas size fit parent.
        self.root.props.min_height = 0
        # store components here if you want to keep
        # data after reload layout.
        self.points = three.Points(5000000)

    def app_create_layout(self) -> mui.LayoutType:
        cam = three.PerspectiveCamera(True, fov=75, near=0.1, far=1000)
        cam.prop(position=(0, 0, 20), up=(0, 0, 1))
        self.ctrl = three.CameraControl().prop(damping_factor=1.0)
        infgrid = three.InfiniteGridHelper(5, 50, "gray")
        self.canvas = three.ThreeCanvas({
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
                ]).prop(flex=1, min_height=0, min_width=0),
                slider,
            ]).prop(flex=1, min_height=0),
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
        self.root.props.min_height = 0
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
        mesh = three.Mesh(three.RoundedRectGeometry(2, 1.5, 0.5),
                          three.MeshBasicMaterial().prop(color="#393939"))
        mesh.set_pointer_callback(
            on_click=three.EventHandler(lambda x: print(1), True))
        mesh.prop(hover_color="#222222", click_color="#009A63")
        text = three.Text("WTF")
        text.prop(color="red", font_size=2)
        text.set_pointer_callback(
            on_click=three.EventHandler(lambda x: print(2)))

        self.text2 = three.Text("T")
        self.text2.prop(color="red", font_size=0.5)
        self.text2.set_pointer_callback(
            on_click=three.EventHandler(lambda x: print(3)))
        material = three.MeshBasicMaterial()
        material.prop(wireframe=True, color="hotpink")
        mesh2 = three.Mesh(three.BoxGeometry(), material)
        mesh2.set_pointer_callback(
            on_click=three.EventHandler(lambda x: print(4)))
        self.img_path = mui.Input("Image Path")
        self.img = three.Image()
        self.img.set_pointer_callback(on_click=three.EventHandler(
            lambda x: print("IMAGE!!!", self.img_path.value)))
        self.img.prop(scale=(4, 4, 1))
        self.html = three.Html(
            {"btn": mui.Button("RTX", lambda: print("RTX1"))})
        self.html.prop(transform=True, center=False, inside_flex=True)
        self.html2 = three.Html(
            {"btn2": mui.Button("RTX2", lambda: print("RTX2"))})
        res = self.html2.prop(transform=True, center=False, inside_flex=True)
        self.flex_container = three.Flex({
            "mesh1":
            three.ItemBox({
                "mesh03":
                three.Button("RTX", 2, 1, lambda x: print("HELLO")),
            }).prop(center_anchor=True),
        })
        self.hud = three.Hud({
                "mesh":
                three.ItemBox({
                    "mesh0":
                    three.Button("RTX", 2, 1, lambda x: print("HELLO")),
                }).prop(center_anchor=True),
                "mesh1":
                three.ItemBox({
                    "mesh0":
                    three.ToggleButton("RTX2", 2, 1,
                                       lambda x: print("HELLO2", x)),
                }).prop(center_anchor=True),
                "text":
                three.ItemBox({
                    "text0": self.html,
                }).prop(center_anchor=True),
                "text4":
                three.ItemBox({
                    "text0": self.html2,
                }).prop(center_anchor=True),
                "text3":
                three.ItemBox({
                    "text0": three.BoundingBox((2, 5, 2)),
                }).prop(center_anchor=True),
                "autoreflow":
                three.FlexAutoReflow(),
            }).prop(render_priority=1,
                    flex_direction="row",
                    justify_content="flex-start")
        self.canvas = three.ThreeCanvas({
            "cam":
            cam,
            "points":
            self.points,
            # "lines": self.lines,
            # "flexdev": three.Flex({
            #     "box1": three.ItemBox({
            #         "text0": three.Text("WTF1").prop(color="red", font_size=2),
            #     }).prop(center_anchor=True),
            #     "box2": three.ItemBox({
            #         "text0": three.Text("WTF2").prop(color="red", font_size=2),
            #     }).prop(center_anchor=True),
            #     "box3": three.ItemBox({
            #         "text0": three.Text("WTF3").prop(color="red", font_size=2),
            #     }).prop(center_anchor=True),
            #     "box4": three.ItemBox({
            #         "text0": three.Text("WTF4").prop(color="red", font_size=2),
            #     }).prop(center_anchor=True),

            # }).prop(flex_direction="row", size=(20, 20, 0), position=(-20, -20, 0), flex_wrap="wrap"),
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
            three.Text("WTF").prop(color="red", font_size=2),
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
                        z_index=5,
                        justify_content="flex-end")
            }).prop(position="relative", flex=1, min_height=0),
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
                }).prop(center_anchor=True),
        })

class SampleThree2DApp(EditableApp):

    def __init__(self) -> None:
        super().__init__(reloadable_layout=True)
        self.set_init_window_size([800, 600])
        # makesure three canvas size fit parent.
        self.root.props.min_height = 0
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
                        line_color="red",
                        alpha=0.1,
                        line_width=1,
                        hover_line_color="blue",
                        hover_line_width=2)

        self.canvas = three.ThreeCanvas({
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
                        z_index=5,
                        justify_content="flex-end")
            }).prop(position="relative", flex=1, min_height=0),
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


class SampleMapApp(EditableApp):

    def __init__(self) -> None:
        super().__init__(reloadable_layout=True)
        self.set_init_window_size([800, 600])
        # makesure three canvas size fit parent.
        # self.root.props.min_height = 0
        # store components here if you want to keep
        # data after reload layout.
        self.root.props.flex_flow = "row nowrap"

    def app_create_layout(self) -> Dict[str, MUIComponentType]:
        google_url = "http://{s}.google.com/vt?lyrs=m&x={x}&y={y}&z={z}"
        self.leaflet = leaflet.MapContainer(
            (30, 100), 13, {
                "tile": leaflet.TileLayer(google_url),
            }).prop(height="100%", flex=3)
        return {
            "control":
            mui.VBox({
                "btn":
                mui.Button("FlyTo", lambda: self.leaflet.fly_to(
                    (40, 100), zoom=10)),
            }).prop(min_height=0, flex=1),
            "mmap":
            self.leaflet,
        }


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
    i: int = dataclasses.field(default=1,
                               metadata=ConfigPanel.slider_meta(0, 10))
    j: TestEnumInt = TestEnumInt.C


class SampleConfigApp(EditableApp):

    def __init__(self) -> None:
        super().__init__(reloadable_layout=True)
        self.set_init_window_size([800, 600])
        # makesure three canvas size fit parent.
        # self.root.props.min_height = 0
        # store components here if you want to keep
        # data after reload layout.
        self.root.props.flex_flow = "row nowrap"
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
        # self.root.props.min_height = 0
        # store components here if you want to keep
        # data after reload layout.
        self.root.props.flex_flow = "row nowrap"

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

        return mui.VBox([
            mui.MultipleAutocomplete("Movies", self.options).prop(
                variant="checkbox", disable_close_on_select=True),
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

        self.canvas = plus.SimpleCanvas(cam, self._on_video_save)
        self.slider = mui.Slider("Slider",
                                 0,
                                 1,
                                 1,
                                 callback=self._on_slider_select)
        
        return mui.VBox([

            mui.HBox([
                mui.Button("Change Slider Range",
                           self._on_slider_range_change),
                mui.Button("Video",
                           self._on_save_video),

                self.slider.prop(flex=1),
            ]),
            self.canvas.prop(flex=1),
        ]).prop(min_height=0,
                min_width=0,
                flex=1,
                width="100%",
                height="100%",
                overflow="hidden")

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
        print("select slider!!!", value)
        # you need to specify a key for a group of point
        # you also need to specify number limit of current point
        points = np.random.uniform(-1, 1, size=[1000, 3]).astype(np.float32)
        # colors can be:
        # 1. [N, 3] float, value range: [0, 1]
        # 2. [N], int8 (intensity), value range: [0, 255]
        # 3. a color string, e.g. red, green
        colors = np.random.uniform(254, 255, size=[1000]).astype(np.uint8)
        print(colors)
        # colors = np.random.uniform(254, 255, size=[1000]).astype(np.uint8)
        sizes = np.random.uniform(0.5, 10.5, size=[1000]).astype(
            np.float32) * 1

        await self.canvas.show_points("key0",
                                      points,
                                      limit=100000,
                                      colors=colors)
        # boxes: dims, locs, rots, colors (string list, don't support ndarray currently)
        dims = np.random.uniform(1, 2, size=[5, 3])
        locs = np.random.uniform(-5, 5, size=[5, 3])
        rots = np.random.uniform(-1, 1, size=[5, 3])
        rots[:, :2] = 0
        colors = ["red", "yellow", "red", "blue", "yellow"]
        await self.canvas.show_boxes("key0", dims, locs, rots, colors)

        # lines: [N, 2, 3]
        lines = np.random.uniform(-3, 3, size=[10, 2, 3])
        await self.canvas.show_lines("key0", lines, limit=10000, color="aqua")


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


class CollectionApp:

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
        self.wtf2 = plus.ConfigPanel(self.cfg, lambda x, y: print(x, y))
        # self.dev_0 = Dev()
        return mui.HBox([
            mui.Allotment([
                plus.ObjectInspector(self).prop(width="100%",
                                                height="100%",
                                                overflow="hidden"),
                mui.HBox([
                    plus.AnyFlexLayout(),
                ]).prop(width="100%", height="100%", overflow="hidden")
            ]).prop(default_sizes=[1, 3], width="100%", height="100%")
        ]).prop(flex_flow="row nowrap")
    
    @mark_autorun 
    def _autorun_dev(self):
        func_support_reload(1, 2)
        # self.dev_0.dev()


if __name__ == "__main__":
    import time
    tps = get_all_members_by_type(mui.FlexBox)
    for _ in range(10):
        t = time.time()
        tps = get_all_members_by_type(mui.FlexBox)
        print(len(tps), time.time() - t)
