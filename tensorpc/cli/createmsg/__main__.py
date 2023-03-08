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
import fire
from tensorpc.flow.client import add_message
from tensorpc.flow.coretypes import MessageItemType, MessageLevel, MessageItem
import faker
import base64


def main(title: str, image_path: str = ""):
    fake = faker.Faker()
    items = [MessageItem(MessageItemType.Text, fake.text())]
    if image_path != "":
        path = Path(image_path)
        suffix = path.suffix[1:]
        with path.open("rb") as f:
            encoded_string = base64.b64encode(f.read()).decode("utf-8")
        img_str = f"data:image/{suffix};base64,{encoded_string}"
        items.append(MessageItem(MessageItemType.Image, img_str))
    add_message(title, MessageLevel.Info, items)


if __name__ == "__main__":
    fire.Fire(main)
