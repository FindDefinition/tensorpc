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
import cv2
from tensorpc.apps.flow.coretypes import MessageLevel
from tensorpc.apps.flow.flowapp import App 
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
        self.add_buttons(["LoadImage", "SendMessage", "OpenCam"], self.on_button_click)
        self.add_switch("Switch", self.on_switch)
        self.add_input("Image Path", self.on_input_change)
        self.img_ui = self.add_images(1)
        self.img_path = ""
        self.set_init_window_size([480, 640])
        self.task = None

        self.task_loop = self.add_task_loop("Test", self.on_task_loop)

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
                    await self.img_ui.show_raw(0, raw)
                else:
                    img = cv2.imread(str(path))
                    # print(type(img))
                    # print(img.shape)
                    await self.img_ui.show(0, img)
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

    async def on_switch(self, checked: bool):
        print(checked)

    async def on_input_change(self, value: str):
        print(value)
        self.img_path = value

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

            await self.img_ui.show_raw(0, b'data:image/jpg;base64,' + base64.b64encode(img_str))
            dura = time.time() - t
            t = time.time()
            # await asyncio.sleep(0)
            # print(cnt, len(img_str), (time.time() - t) / cnt)


