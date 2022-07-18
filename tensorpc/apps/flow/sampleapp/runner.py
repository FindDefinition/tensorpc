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

class PyRunner(App):
    def __init__(self) -> None:
        super().__init__()
        self.root.add_buttons(["RunCode", "CopyCode"], self.on_button_click)
        self.code = ""
        self.root.add_code_editor("python", self.on_code_change)
        self.set_init_window_size([320, None])

    async def on_button_click(self, name: str):
        if name == "RunCode":
            exec(self.code)
        elif name == "CopyCode":
            await self.copy_text_to_clipboard(self.code)

    async def on_code_change(self, value: str):
        self.code = value

