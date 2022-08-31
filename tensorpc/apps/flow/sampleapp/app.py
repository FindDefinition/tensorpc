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
from functools import partial
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
from tensorpc.apps.flow.flowapp.app import EditableLayoutApp
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
            "btn": mui.ButtonGroup({
                "btn0": mui.Button("LoadImage", partial(self.on_button_click, name="LoadImage")),
                "btn1": mui.Button("SendMessage", partial(self.on_button_click, name="SendMessage")),
                "btn2": mui.Button("OpenCam", partial(self.on_button_click, name="OpenCam")),
                "btn3": mui.Button("Sleep", partial(self.on_button_click, name="Sleep")),
            }),
            "btn2": mui.Button("Sleep", self.on_one_button_click),

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
                Button("CLICK ME1",
                       lambda: print("HELLO BTN1")).prop(flex=1),
                "btn1":
                Button("Add", self._ppend_list).prop(flex=1),
            }),
            "l0":
            HBox({
                "items": self.vlist,
                "text": Text("content").prop(flex=3),
            }).prop(height="100%"),
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
        self.root._get_all_nested_childs()
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

class SampleEditorAppV2(EditableApp):

    def __init__(self) -> None:
        super().__init__(reloadable_layout=True)
        self.text = Text("WTF")
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
        ctrl = three.MapControl()
        # ctrl = three.FirstPersonControl()

        # ctrl = three.OrbitControl()
        infgrid = three.InfiniteGridHelper(5, 50, "gray")
        self.b2d = three.Boxes2D(1000)
        mesh = three.Mesh(three.BoxGeometry(), three.MeshBasicMaterial())
        mesh.set_pointer_callback(on_click=three.EventCallback(lambda x: print(x)))
        self.canvas = three.ThreeCanvas({
            "cam": cam,
            "points": self.points,
            # "lines": self.lines,
            "ctrl": ctrl,
            "axes": three.AxesHelper(10),
            "infgrid": infgrid,
            "img": self.img,
            "b2d": self.b2d,
            "mesh": mesh,
            # "box": three.BoundingBox([2, 5, 2], [0, 10, 0], [0, 0, 0.5])
        })
        btn_random_pc = Button("showRandomRPC", self.show_Random_pc)
        return {
            "d3v":
            VBox({
                "d3":
                VBox({
                    "d32": self.canvas,
                }).prop(flex=1, min_height=0, min_width=0),
                "btn": btn_random_pc,
                "btn2":
                Button("rpcTest", self.rpc_test),
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
        await self.lines.update_lines(random_lines,
                                      line_width=1,
                                      color="green")
        # print("???????", random_lines)
        # with open("/home/yy/Pictures/Screenshot from 2022-02-11 15-10-06.png", "rb") as f:
        #     await self.img.show_raw(f.read(), "png")
        centers = np.array([[0, 0], [2, 2], [3, 3]], np.float32)
        dimensions = np.array([[1, 1], [1, 1], [1, 1]], np.float32)
        attrs = [str(i) for i in range(centers.shape[0])]
        await self.b2d.update_boxes(centers,
                                    dimensions,
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
            }).prop(flex=1),
            "btn": Button("Show", lambda: print("?"))
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
        mesh = three.Mesh(three.RoundedRectGeometry(2, 1.5, 0.5), three.MeshBasicMaterial().prop(color="#393939"))
        mesh.set_pointer_callback(on_click=three.EventCallback(lambda x: print(1), True))
        mesh.prop(hover_color="#222222", click_color="#009A63")
        text = three.Text("WTF")
        text.prop(color="red", font_size=2)
        text.set_pointer_callback(on_click=three.EventCallback(lambda x: print(2)))

        self.text2 = three.Text("T")
        self.text2.prop(color="red", font_size=0.5)
        self.text2.set_pointer_callback(on_click=three.EventCallback(lambda x: print(3)))
        material = three.MeshBasicMaterial()
        material.prop(wireframe=True, color="hotpink")
        mesh2 = three.Mesh(three.BoxGeometry(), material)
        mesh2.set_pointer_callback(on_click=three.EventCallback(lambda x: print(4)))
        self.img_path = mui.Input("Image Path")
        self.img = three.Image()
        self.img.set_pointer_callback(on_click=three.EventCallback(lambda x: print("IMAGE!!!", self.img_path.value)))
        self.img.prop(scale=(4, 4, 1))
        self.html = three.Html({
            "btn": mui.Button("RTX", lambda: print("RTX1"))
        })
        self.html.prop(transform=True, center=False, inside_flex=True)
        self.html2 = three.Html({
            "btn2": mui.Button("RTX2", lambda: print("RTX2"))

        })
        self.html2.prop(transform=True, center=False, inside_flex=True)

        self.canvas = three.ThreeCanvas({
            "cam": cam,
            "points": self.points,
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

            "ctrl": ctrl,
            "axes": three.AxesHelper(10),
            "infgrid": infgrid,
            "b2d": self.b2d,
            "mesh": mesh2,
            # "img": self.img,
            "text": three.Text("WTF").prop(color="red", font_size=2),
            "box": three.BoundingBox([2, 5, 2]).prop(position=(5, 0, 0)),
            # 
            # "text0": self.html,
            "hud": three.Hud({
                "mesh": three.ItemBox({
                    "mesh0": three.Button("RTX", 2, 1, lambda x: print("HELLO")),
                }).prop(center_anchor=True),
                "mesh1": three.ItemBox({
                    "mesh0": three.ToggleButton("RTX2", 2, 1, lambda x: print("HELLO2", x)),
                }).prop(center_anchor=True),
                "text": three.ItemBox({
                    "text0": self.html,
                }).prop(center_anchor=True),
                "text4": three.ItemBox({
                    "text0": self.html2,
                }).prop(center_anchor=True),
                "text3": three.ItemBox({
                    "text0": three.BoundingBox([2, 5, 2]),
                }).prop(center_anchor=True),
                "autoreflow": three.FlexAutoReflow(),
            }).prop(render_priority=1, flex_direction="row", justify_content="flex-start")
        })
        return {
            "d3v":
            VBox({
                "d3": self.canvas,
                "hud": mui.VBox({
                    "inp": self.img_path,
                    "btn1": mui.Button("Read Image", self.on_read_img),
                    "btn3": mui.Text("Inp", )


                }).prop(position="absolute", top=0, right=0, z_index=5, justify_content="flex-end")
            }).prop(position="relative", flex=1, min_height=0),
        }
        
    async def on_read_img(self):
        path = self.img_path.value
        with open(path, "rb") as f:
            img_str = f.read()
        await self.img.show_raw(img_str, "jpg")
        await self.text2.update_value("WTF1")

class SampleThree2DApp(EditableApp):

    def __init__(self) -> None:
        super().__init__(reloadable_layout=True)
        self.set_init_window_size([800, 600])
        # makesure three canvas size fit parent.
        self.root.props.min_height = 0
        # store components here if you want to keep
        # data after reload layout.
        self.box2d = three.Boxes2D(20000)


    def app_create_layout(self) -> Dict[str, MUIComponentType]:
        cam = three.OrthographicCamera(True, near=0.1, far=1000,
                                      zoom=50.0)
        cam.prop(position=[0, 0, 10], up=[0, 0, 1])
        ctrl = three.MapControl()
        ctrl.props.enable_rotate = False
        # ctrl = three.FirstPersonControl()

        self.canvas = three.ThreeCanvas({
            "cam": cam,
            "ctrl": ctrl,
            "b2d": self.box2d,
            # "axes": three.AxesHelper(10),
            "btn0": three.Button("RTX", 2, 1, self.on_box2d_update),
            "html0": three.Html({
                "btn0": mui.Button("RTX", lambda: print("RTX")),
            }).prop(position=(-5, 0, 0), transform=True)
        })
        return {
            "d3v":
            VBox({
                "d3": self.canvas,
                "hud": mui.VBox({
                    # "update": mui.Button("Box2d", self.on_box2d_update),
                    "btn3": mui.Text("Inp", )
                }).prop(position="absolute", top=0, right=0, z_index=5, justify_content="flex-end")
            }).prop(position="relative", flex=1, min_height=0),
        }
        
    async def on_box2d_update(self, ev = None):
        centers = np.random.randint(1, 10, size=[128 * 32, 2]).astype(np.float32)
        centers = np.arange(0, 128 * 32).astype(np.int32)
        centers = np.stack([centers // 32, centers % 32], axis=1).astype(np.float32)
        centers += [3, 0]
        # centers = np.array([[0, 0], [2, 2], [3, 3]], np.float32)
        dimensions = np.ones((1,), np.float32) #  - 0.1
        attrs = [str(i) for i in range(centers.shape[0])]
        await self.box2d.update_boxes(centers,
                                    dimensions,
                                    color="red",
                                    alpha=0.0,
                                    attrs=attrs)
