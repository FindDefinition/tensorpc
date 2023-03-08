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

import fire
import time
from tensorpc.constants import PACKAGE_ROOT


def get_proto_path():
    paths = list((PACKAGE_ROOT / "protos").glob("*.proto"))
    paths = list(filter(lambda x: x.stem != "remote_object", paths))
    path_strs = list(map(lambda x: f"{str(x)}", paths))
    print(" ".join(path_strs))


def main():
    fire.Fire(get_proto_path)


if __name__ == "__main__":
    main()
