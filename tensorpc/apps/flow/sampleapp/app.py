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

from pathlib import Path
import traceback
from typing import Any, Union
import cv2
from tensorpc.apps.flow.coretypes import MessageLevel, ScheduleEvent
from tensorpc.apps.flow.flowapp import App, Button, HBox, ListItemButton, ListItemText, Text, VBox, VList, Plotly, ChartJSLine
import imageio
import io
import base64
from tensorpc.apps.flow.client import add_message
from faker import Faker
from datetime import datetime

import asyncio
from tensorpc.core import prim

from tensorpc.core.asynctools import cancel_task

class SampleApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.root.add_buttons(["LoadImage", "SendMessage", "OpenCam", "RunCode"], self.on_button_click)
        self.root.add_switch("Switch", self.on_switch)
        self.root.add_input("Image Path", self.on_input_change)
        self.img_ui = self.root.add_image()
        self.img_path = ""
        self.set_init_window_size([480, 640])
        self.task = None

        self.task_loop = self.root.add_task_loop("Test", self.on_task_loop)
        self.root.add_slider("Slider", 0, 100, 1, self.on_slider_change)
        self.root.add_select("Select", [("One", 0), ("Two", 1)], self.on_select_change)
        self.root.add_radio_group(["Option1", "Option2"], True, self.on_radio)
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
                    raw = b'data:image/gif;base64,' + base64.b64encode(data)
                    await self.img_ui.show_raw(raw)
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

            frame = cv2.putText(frame,f'{dt_string} FrameRate={1 / dura:.2f}',(10,30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            suffix = "jpg"
            _, img_str = cv2.imencode(".{}".format(suffix), frame)

            await self.img_ui.show_raw(b'data:image/jpg;base64,' + base64.b64encode(img_str))
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
        }, flex=1)
        self.cnt = 2
        self.root.add_layout({
            "btn0": Button("CLICK ME", lambda: print("HELLO BTN")),
            # "vlist0": VList({
            #     "btn0": ListItemButton("wtf1", lambda: print("HELLO List BTN1")),
            #     "btn1": ListItemButton("wtf2", lambda: print("HELLO List BTN2")),
            # }),
            "layout0": HBox({
                "btn0": Button("CLICK ME1", lambda: print("HELLO BTN1"), flex=1),
                "btn1": Button("Add", self._ppend_list, flex=1),
            }),
            "l0": HBox({
                "items": self.vlist,
                "text": Text("content", flex=3),
            }, height="100%"),
        })
        self.set_init_window_size([480, 640])

    async def _ppend_list(self):
        await self.vlist.update_childs({f"text{self.cnt}": ListItemText(str(self.cnt))})
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
            "marker": {"color": 'red'},
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
            "marker": {"color": 'red'},
          },
        ]
        layout = {"width": 320, "height": 240,
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
        labels =  ['January', 'February', 'March', 'April', 'May', 'June', 'July']
        data = {
           "labels": labels,
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
            "plot0": self.plot,
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
        labels =  ['0', '1', '2', '3', '4', '5', '6']
        data = {
           "labels": labels,
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
