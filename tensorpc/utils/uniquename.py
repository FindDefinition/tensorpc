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

from typing import Set


def _make_unique_name(unique_set, name, max_count=10000) -> str:
    if name not in unique_set:
        unique_set.add(name)
        return name
    for i in range(max_count):
        new_name = name + "_{}".format(i)
        if new_name not in unique_set:
            unique_set.add(new_name)
            return new_name
    raise ValueError("max count reached")


class UniqueNamePool:

    def __init__(self, max_count=10000):
        self.max_count = max_count
        self.unique_set: Set[str] = set()

    def __call__(self, name):
        return _make_unique_name(self.unique_set, name, self.max_count)

    def __contains__(self, key: str):
        return key in self.unique_set

    def pop(self, key: str):
        self.unique_set.remove(key)

    def clear(self):
        self.unique_set.clear()