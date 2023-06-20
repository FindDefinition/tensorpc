# Copyright 2023 Yan Yan
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
from typing_extensions import TypedDict
class AddableAutocompleteType(TypedDict):
    select_on_focus: bool 
    clear_on_blur: bool
    handle_home_end_keys: bool
    free_solo: bool
    add_option: bool

class CommonOptions:

    AddableAutocomplete: AddableAutocompleteType = {
        "select_on_focus": True,
        "clear_on_blur": True,
        "handle_home_end_keys": True,
        "free_solo": True,
        "add_option": True,
    }
    FitParent = {
        "width": "100%",
        "height": "100%",
        "overflow": "auto",
    }