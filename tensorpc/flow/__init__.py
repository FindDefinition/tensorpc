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

from tensorpc.core.moduleid import loose_isinstance

from . import constants, marker
from .flowapp import App, EditableApp, EditableLayoutApp, appctx
from .flowapp.appcore import observe_function, AppSpecialEventType
from .flowapp.components import leaflet, mui, plotly, plus, three
from .flowapp.objtree import UserObjTree
from .marker import (mark_autorun, mark_create_layout, mark_create_object,
                     mark_create_preview_layout, mark_did_mount,
                     mark_will_unmount)
from .flowapp.components.plus.vis import vapi_core as V