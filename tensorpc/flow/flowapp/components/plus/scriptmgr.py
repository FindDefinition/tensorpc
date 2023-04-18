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

import asyncio
import dataclasses
import enum
import inspect
import time
from typing import Any, Callable, Coroutine, Dict, Iterable, List, Optional, Set, Tuple, Union
from typing_extensions import Literal

import numpy as np
from tensorpc.flow.flowapp import appctx

from tensorpc.flow import marker
from tensorpc.flow.flowapp.components import mui, three
from tensorpc.flow.flowapp.core import FrontendEventType
from .options import CommonOptions


@dataclasses.dataclass
class Script:
    label: str
    code: str
    lang: str


_LANG_TO_VSCODE_MAPPING = {
    "python": "python",
    "cpp": "cpp",
    "cuda": "cpp",
}


class ScriptManager(mui.FlexBox):

    def __init__(
        self,
        storage_node_rid: str,
    ):
        super().__init__()

        self._storage_node_rid = storage_node_rid

        self.code_editor = mui.MonacoEditor("", "python",
                                            "default").prop(flex=1)
        self.scripts = mui.Autocomplete(
            "Scripts",
            [],
            self._on_script_select,
        ).prop(size="small",
               input_margin="dense",
               padding="0 3px 0 3px",
               **CommonOptions.AddableAutocomplete)
        self.langs = mui.ToggleButtonGroup([
            mui.ToggleButton("cpp", name="CPP"),
            mui.ToggleButton("python", name="PY"),
            mui.ToggleButton("cuda", name="CUDA"),
        ], True, self._on_lang_select).prop(value="python",
                                            enforce_value_set=True)
        self.init_add_layout([
            mui.HBox([
                self.scripts.prop(flex=1),
                mui.IconButton(
                    mui.IconType.PlayArrow,
                    self._on_run_script).prop(progress_color="primary"),
                self.langs,
            ]).prop(align_items="center"),
            self.code_editor,
        ])
        self.prop(flex=1,
                  flex_direction="column",
                  width="100%",
                  height="100%",
                  overflow="hidden")
        self.code_editor.register_event_handler(
            FrontendEventType.EditorSave.value, self._on_editor_save)
        self.scripts.register_event_handler(
            FrontendEventType.SelectNewItem.value, self._on_new_script)
        self.code_editor.register_event_handler(
            FrontendEventType.EditorReady.value, self._on_editor_ready)

    async def _on_editor_ready(self):
        items = await appctx.list_data_storage(self._storage_node_rid)
        items.sort(key=lambda x: x.userdata["timestamp"]
                   if not isinstance(x.userdata, mui.Undefined) else 0,
                   reverse=True)
        options: List[Dict[str, Any]] = []
        for item in items:
            if item.typeStr == Script.__name__:
                options.append({"label": item.name})
        if options:
            await self.scripts.update_options(options, 0)
            await self._on_script_select(options[0])

    async def _on_run_script(self):
        if self.scripts.value is not None:
            label = self.scripts.value["label"]
            item = await appctx.read_data_storage(label,
                                                  self._storage_node_rid)
            assert isinstance(item, Script)
            if item.lang == "python":
                __tensorpc_script_res: List[Optional[Coroutine]] = [None]
                lines = item.code.splitlines()
                lines = [" " * 4 + line for line in lines]
                lines.insert(0, "async def _run():")
                lines.append("__tensorpc_script_res[0] = _run()")
                code = "\n".join(lines)
                exec(code, {},
                     {"__tensorpc_script_res": __tensorpc_script_res})
                res = __tensorpc_script_res[0]
                assert res is not None
                await res

    async def _on_lang_select(self, value):
        await self.send_and_wait(
            self.code_editor.update_event(
                language=_LANG_TO_VSCODE_MAPPING[value]))
        if self.scripts.value is not None:
            label = self.scripts.value["label"]
            item = await appctx.read_data_storage(label,
                                                  self._storage_node_rid)
            assert isinstance(item, Script)
            item.lang = value
            await appctx.save_data_storage(label, self._storage_node_rid, item)

    async def _on_editor_save(self, value: str):
        if self.scripts.value is not None:
            label = self.scripts.value["label"]
            item = await appctx.read_data_storage(label,
                                                  self._storage_node_rid)
            assert isinstance(item, Script)
            item.code = value
            await appctx.save_data_storage(label, self._storage_node_rid, item)

    async def _on_new_script(self, value):

        new_item_name = value["label"]
        await self.scripts.update_options([*self.scripts.props.options, value],
                                          -1)
        lang = self.langs.props.value
        assert isinstance(lang, str)
        code_lines: List[str] = []
        if lang == "python":
            code_lines.append("from tensorpc.flow import appctx")

        script = Script(new_item_name, "".join(code_lines), lang)
        await appctx.save_data_storage(new_item_name, self._storage_node_rid,
                                       script)
        await self.send_and_wait(
            self.code_editor.update_event(
                language=_LANG_TO_VSCODE_MAPPING[lang],
                value=script.code,
                path=script.label))

    async def _on_script_select(self, value):
        label = value["label"]
        item = await appctx.read_data_storage(label, self._storage_node_rid)
        assert isinstance(item, Script)
        await self.send_and_wait(
            self.code_editor.update_event(
                language=_LANG_TO_VSCODE_MAPPING[item.lang],
                value=item.code,
                path=item.label))
