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

from typing import Any, Dict, List, Tuple, Union
from typing_extensions import Literal
from ..mui import FlexBox
from ..plotly import Plotly
from ...core import DataClassWithUndefined, NumberType, Undefined, undefined
import dataclasses


class HomogeneousFigures(FlexBox):
    """multiple figures with same layout, and same number of data trace,
    only data value / type varies.
    Often be used in metrics.
    """

    pass
