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
import io
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

import cv2
import imageio
from faker import Faker
import tensorpc
from tensorpc.apps.flow.client import AsyncAppClient, add_message, AppClient
from tensorpc.apps.flow.coretypes import MessageLevel, ScheduleEvent
from tensorpc.apps.flow.flowapp import App, EditableApp
from tensorpc.apps.flow.flowapp.components.mui import (
    Button, ChartJSLine, HBox, ListItemButton, ListItemText, MUIComponentType,
    Plotly, Text, VBox, VList)
from ..flowapp.core import Component
from tensorpc.core import prim
from tensorpc.core.asynctools import cancel_task
from tensorpc.apps.flow.flowapp.components import three, mui
import numpy as np


class SampleApp(App):

    def __init__(self) -> None:
        super().__init__()
        self.img_ui = mui.Images()
        self.task_loop = mui.TaskLoop("Test", self.on_task_loop)
        self.root.add_layout({
            "btn": mui.Buttons(["LoadImage", "SendMessage", "OpenCam", "RunCode"],
            self.on_button_click),
            "swi": mui.Switch("Switch", self.on_switch),
            "inp": mui.Input("Image Path", callback=self.on_input_change),
            "img_ui": self.img_ui,
            "taskloop": self.task_loop,
            "slider": mui.Slider("Slider", 0, 100, 1, self.on_slider_change),
            "select": mui.Select("Select", [("One", 0), ("Two", 1)],
                             self.on_select_change),
            "rg": mui.RadioGroup(["Option1", "Option2"], True, self.on_radio),
        })
        self.img_path = ""
        self.set_init_window_size([480, 640])
        self.task = None
        self.code = ""

    async def on_radio(self, name: str):
        print(name)

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
        elif name == "RunCode":
            exec(self.code)

    async def on_switch(self, checked: bool):
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
        self.vlist.newprop(flex=1)
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
                Button("CLICK ME1",
                       lambda: print("HELLO BTN1")).newprop(flex=1),
                "btn1":
                Button("Add", self._ppend_list).newprop(flex=1),
            }),
            "l0":
            HBox({
                "items": self.vlist,
                "text": Text("content").newprop(flex=3),
            }).newprop(height="100%"),
        })
        self.set_init_window_size([480, 640])

    async def _ppend_list(self):
        await self.vlist.update_childs(
            {f"text{self.cnt}": ListItemText(str(self.cnt))})
        self.cnt += 1


class SamplePlotApp(App):

    def __init__(self) -> None:
        super().__init__()
        data = [
            {
                "x": [1, 2, 3],
                "y": [2, 6, 3],
                "type": 'scatter',
                "mode": 'lines+markers',
                "marker": {
                    "color": 'red'
                },
            },
        ]
        layout = {
            "height": 240,
            "autosize": 'true',
            "margin": {
                "l": 0,
                "r": 0,
                "b": 0,
                "t": 0,
                #   "pad": 0
            },
            #  "margin": {
            # #   "l": 0,
            # #   "r": 0,
            # #   "b": 0,
            #   "t": 20,
            # #   "pad": 0
            # },
            "yaxis": {
                "automargin": True,
            },
            "xaxis": {
                "automargin": True,
            },
        }

        self.plot = Plotly(data=data, layout=layout)
        self.root.add_layout({
            "plot0": self.plot,
            "btn": Button("Show", self._show_plot)
        })
        self.set_init_window_size([480, 320])

    async def _show_plot(self):
        data = [
            {
                "x": [1, 2, 3],
                "y": [3, 2, 1],
                "type": 'scatter',
                "mode": 'lines+markers',
                "marker": {
                    "color": 'red'
                },
            },
        ]
        layout = {
            "width": 320,
            "height": 240,
            "yaxis": {
                "automargin": True,
            },
            "xaxis": {
                "automargin": True,
            }
        }
        await self.plot.show_raw(data, layout)


class SampleChartJSApp(App):

    def __init__(self) -> None:
        super().__init__()
        options = {
            "responsive": True,
            "devicePixelRatio": 1.5,
            "plugins": {
                "legend": {
                    "position": 'top',
                },
                "title": {
                    "display": True,
                    "text": 'Chart.js Line Chart',
                },
            },
        }
        labels = [
            'January', 'February', 'March', 'April', 'May', 'June', 'July'
        ]
        data = {
            "labels":
            labels,
            "datasets": [
                {
                    "label": 'Dataset 1',
                    "data": list(range(len(labels))),
                    "borderColor": 'rgb(255, 99, 132)',
                    "backgroundColor": 'rgba(255, 99, 132, 0.5)',
                },
                {
                    "label": 'Dataset 2',
                    "data": list(range(len(labels))),
                    "borderColor": 'rgb(53, 162, 235)',
                    "backgroundColor": 'rgba(53, 162, 235, 0.5)',
                },
            ],
        }
        self.plot = ChartJSLine(data=data, options=options)
        self.root.add_layout({
            "plot0": VBox({
                "asd": self.plot,
            }, flex=1),
            "btn": Button("Show", self._show_plot)
        })
        self.set_init_window_size([480, 320])

    async def _show_plot(self):
        options = {
            "responsive": True,
            "plugins": {
                "legend": {
                    "position": 'top',
                },
                "title": {
                    "display": True,
                    "text": 'Chart.js Line Chart',
                },
            },
        }
        labels = ['0', '1', '2', '3', '4', '5', '6']
        data = {
            "labels":
            labels,
            "datasets": [
                {
                    "label": 'Dataset 1',
                    "data": list(range(len(labels))),
                    "borderColor": 'rgb(255, 99, 132)',
                    "backgroundColor": 'rgba(255, 99, 132, 0.5)',
                },
                {
                    "label": 'Dataset 2',
                    "data": list(range(len(labels))),
                    "borderColor": 'rgb(53, 162, 235)',
                    "backgroundColor": 'rgba(53, 162, 235, 0.5)',
                },
            ],
        }
        await self.plot.show_raw(data, options)


class SampleFlowApp(App):

    def __init__(self) -> None:
        super().__init__()
        self.text = Text("")
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
        self.text = Text("WTF")
        self.root.add_layout({
            "text": self.text,
            "btn": Button("runCB", self.example_cb),
            "btn2": Button("ShowTS", self.show_ts),
        })
        self.set_init_window_size([480, 320])
        self.init_enable_editor()

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
        self.set_init_window_size([1280, 720])
        # makesure three canvas size fit parent.
        self.root.props.min_height = 0
        # store components here if you want to keep
        # data after reload layout.
        self.points = three.Points(2000000)
        self.lines = three.Segments(20000)

    def app_create_layout(self) -> Dict[str, MUIComponentType]:
        cam = three.PerspectiveCamera(True, fov=75, near=0.1, far=1000)
        cam.newprop(position=(0, 0, 20), up=(0, 0, 1))
        # cam = three.OrthographicCamera(True, position=[0, 0, 10], up=[0, 0, 1], near=0.1, far=1000,
        #                               zoom=8.0)
        self.img = three.Image()
        ctrl = three.MapControl(True, 0.25, 1, 100)
        # ctrl2 = three.PointerLockControl()

        # ctrl = three.OrbitControl(True, 0.25, 1, 100)
        infgrid = three.InfiniteGridHelper(5, 50, "gray")
        self.b2d = three.Boxes2D(1000)
        self.canvas = three.ThreeCanvas({
            "cam": cam,
            "points": self.points,
            "lines": self.lines,
            "ctrl": ctrl,
            "axes": three.AxesHelper(10),
            "infgrid": infgrid,
            "img": self.img,
            "b2d": self.b2d,
            # "box": three.BoundingBox([2, 5, 2], [0, 10, 0], [0, 0, 0.5])
        })
        btn_random_pc = Button("showRandomRPC", self.show_Random_pc)
        return {
            "d3v":
            VBox({
                "d3":
                VBox({
                    "d32": self.canvas,
                }).newprop(flex=1, min_height=0, min_width=0),
                "btn":
                btn_random_pc,
                "btn2":
                Button("rpcTest", self.rpc_test),
            }).newprop(flex=1, min_height=0),
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
        # await self.points.update_points(pc,
        #                                 attrs=attrs,
        #                                 attr_fields=attr_fields)

        random_lines = np.random.uniform(-5, 5, size=[5, 2,
                                                      3]).astype(np.float32)
        await self.lines.update_lines(random_lines,
                                      line_width=1,
                                      color="green")
        # print("???????", random_lines)
        # with open("/home/yy/Pictures/Screenshot from 2022-02-11 15-10-06.png", "rb") as f:
        #     await self.img.show_raw(f.read(), "png")
        centers = np.array([[0, 0], [2, 2], [3, 3]], np.float32)
        dimersions = np.array([[1, 1], [1, 1], [1, 1]], np.float32)
        attrs = [str(i) for i in range(centers.shape[0])]
        await self.b2d.update_boxes(centers,
                                    dimersions,
                                    color="red",
                                    alpha=0.5)
        await self.b2d.update_object3d(position=(0, 0, 1))

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


class SampleTestApp(App):

    def __init__(self) -> None:
        super().__init__()
        self.root.add_layout({
            "plot0": VBox({
                "asd": Text("Hello"),
            }).newprop(flex=1),
            "btn": Button("Show", lambda: print("?"))
        })
        self.set_init_window_size([480, 320])
