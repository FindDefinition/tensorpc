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

import cv2
from tensorpc.apps.flow.flowapp import App 


import asyncio 

class SampleApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.add_buttons(["A", "B", "LoadImage"], self.on_button_click)
        self.add_switch("Switch", self.on_switch)
        self.add_input("Image Path", self.on_input_change)
        self.img_ui = self.add_images(1)
        self.img_path = ""
        self.set_init_window_size([480, 640])

    async def on_button_click(self, name: str):
        print(name)
        if name == "LoadImage":
            path = Path(self.img_path)
            print(path)
            if path.exists():
                img = cv2.imread(str(path))
                print(type(img))
                print(img.shape)
                await self.img_ui.show(0, img)

    async def on_switch(self, checked: bool):
        print(checked)

    async def on_input_change(self, value: str):
        print(value)
        self.img_path = value