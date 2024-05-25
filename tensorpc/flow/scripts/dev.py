# Copyright 2024 Yan Yan
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

from typing_extensions import Literal
import tensorpc
import time
from typing import Any, Tuple, Union, List, Dict, get_origin, Generic
import dataclasses
import mashumaro
import enum
from tensorpc.flow.components import plotly
from tensorpc.flow.core.core import as_dict_no_undefined


def get_args(t: Any) -> Tuple[Any, ...]:
    return getattr(t, "__args__", None) or ()


def get_origin(tp):
    if tp is Generic:
        return Generic
    return getattr(tp, "__origin__", None)


_BASE_TYPES = (
    int,
    float,
    bool,
    str,
)


def _check_is_basic_type(tp):
    origin = get_origin(tp)
    if origin is not None:
        if origin in (list, tuple, dict):
            args = get_args(tp)
            return all(_check_is_basic_type(a) for a in args)
        else:
            return origin in _BASE_TYPES or origin is Literal
    else:
        return tp in _BASE_TYPES or issubclass(tp, enum.Enum)


class TestEnum(enum.Enum):
    A = "1"
    B = "2"
    C = "3"


@dataclasses.dataclass
class WTF1:
    d: int
    e: List[Tuple[int, Dict[str, int]]]


@dataclasses.dataclass
class WTF:
    a: int
    b: Union[int, float]
    x: Literal["WTF", "WTF1"]
    c: bool = False
    e: str = "RTX"
    f: TestEnum = TestEnum.C


if __name__ == "__main__":
    # X = List[Tuple[int, Dict[str, int]]]
    # print(X, type(X))
    # breakpoint()

    # for f in dataclasses.fields(WTF):
    #     # if f.name == "b":
    #     print(f.type, get_origin(f.type), get_origin(f.type) is Union, get_origin(f.type) is Literal, type(get_origin(f.type)), _check_is_basic_type(f.type))
    # breakpoint()
    l = plotly.Layout(autosize=True,
                      margin=plotly.Margin(l=0, r=0, b=0, t=0),
                      xaxis=plotly.Axis(automargin=True),
                      yaxis=plotly.Axis(automargin=True))

    print(as_dict_no_undefined(l))
