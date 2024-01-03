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
import os
import time
from typing import Any, Callable, Coroutine, Dict, Iterable, List, Optional, Set, Tuple, Union
from typing_extensions import Literal

import numpy as np
from tensorpc.constants import TENSORPC_FILE_NAME_PREFIX
from tensorpc.flow.constants import TENSORPC_FLOW_APP_LANG_SERVER_PORT
from tensorpc.flow.flowapp import appctx

from tensorpc.flow import marker
from tensorpc.flow.flowapp.components import mui, three
from tensorpc.flow.flowapp.core import FrontendEventType
from .options import CommonOptions

from tensorpc.flow.client import MasterMeta

@dataclasses.dataclass
class Script:
    label: str
    code: str
    lang: str


_LANG_TO_VSCODE_MAPPING = {
    "python": "python",
    "cpp": "cpp",
    "cuda": "cpp",
    "bash": "shell",
}

async def _read_stream(stream, cb):  
    while True:
        line = await stream.readline()
        if line:
            try:
                line_print = line.decode().rstrip()
            except UnicodeDecodeError:
                line_print = line
            cb(line_print)
        else:
            break

class ScriptManager(mui.FlexBox):

    def __init__(
        self,
        storage_node_rid: Optional[str] = None,
        graph_id: Optional[str] = None,
        init_python_script: Optional[str] = None
    ):
        """when storage_node_rid is None, use app node storage, else use the specified node storage
        """
        super().__init__()
        if storage_node_rid is None:
            storage_node_rid = MasterMeta().node_id
        if graph_id is None:
            graph_id = MasterMeta().graph_id
        self._storage_node_rid = storage_node_rid

        self._graph_id = graph_id
        self.code_editor = mui.MonacoEditor("", "python",
                                            "default").prop(flex=1, minHeight=0, minWidth=0)
        self.scripts = mui.Autocomplete(
            "Scripts",
            [],
            self._on_script_select,
        ).prop(size="small",
               muiMargin="dense",
               padding="0 3px 0 3px",
               **CommonOptions.AddableAutocomplete)
        self.langs = mui.ToggleButtonGroup([
            mui.ToggleButton("cpp", name="CPP"),
            mui.ToggleButton("python", name="PY"),
            mui.ToggleButton("cuda", name="CUDA"),
            mui.ToggleButton("bash", name="BASH"),

        ], True, self._on_lang_select).prop(value="python",
                                            enforceValueSet=True)
        self._enable_save_watch = mui.ToggleButton(
                    "value",
                    mui.IconType.Visibility).prop(muiColor="secondary", size="small")
        self._run_button = mui.IconButton(
                    mui.IconType.PlayArrow,
                    self._on_run_script).prop(progressColor="primary")
        self.init_add_layout([
            mui.HBox([
                self.scripts.prop(flex=1),
                self._run_button,
                self._enable_save_watch,
                self.langs,
            ]).prop(alignItems="center"),
            self.code_editor,
        ])
        self._init_python_script = init_python_script
        self.prop(flex=1,
                  flexDirection="column",
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
        items = await appctx.list_data_storage(self._storage_node_rid, self._graph_id)
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
        else:
            if self._init_python_script is not None:
                await self._on_new_script({
                        "label": "example",
                    }, init_str=self._init_python_script)

    async def _on_run_script(self):
        await self.code_editor.save()
        if self.scripts.value is not None:
            label = self.scripts.value["label"]
            item = await appctx.read_data_storage(label,
                                                  self._storage_node_rid, self._graph_id)
            assert isinstance(item, Script)
            item_uid = f"{self._graph_id}@{self._storage_node_rid}@{item.label}"
            if item.lang == "python":
                __tensorpc_script_res: List[Optional[Coroutine]] = [None]
                lines = item.code.splitlines()
                lines = [" " * 4 + line for line in lines]
                run_name = f"run_{label}"
                lines.insert(0, f"async def _{run_name}():")
                lines.append(f"__tensorpc_script_res[0] = _{run_name}()")
                code = "\n".join(lines)
                code_comp = compile(code, f"<{TENSORPC_FILE_NAME_PREFIX}-scripts-{item_uid}>", "exec")
                exec(code_comp, {},
                     {"__tensorpc_script_res": __tensorpc_script_res})
                res = __tensorpc_script_res[0]
                assert res is not None
                await res
            elif item.lang == "bash":
                proc = await asyncio.create_subprocess_shell(
                    item.code,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE)
                await asyncio.wait([
                    _read_stream(proc.stdout, print),
                    _read_stream(proc.stderr, print)
                ])
                await proc.wait()
                print(f'[cmd exited with {proc.returncode}]')
                # if stdout:
                #     print(f'[stdout]\n{stdout.decode()}')
                # if stderr:
                #     print(f'[stderr]\n{stderr.decode()}')

    async def _on_lang_select(self, value):
        await self.send_and_wait(
            self.code_editor.update_event(
                language=_LANG_TO_VSCODE_MAPPING[value]))
        if self.scripts.value is not None:
            label = self.scripts.value["label"]
            item = await appctx.read_data_storage(label,
                                                  self._storage_node_rid, self._graph_id)
            assert isinstance(item, Script)
            item.lang = value
            await appctx.save_data_storage(label, item, self._storage_node_rid, self._graph_id)

    async def _on_editor_save(self, value: str):
        if self.scripts.value is not None:
            label = self.scripts.value["label"]
            item = await appctx.read_data_storage(label,
                                                  self._storage_node_rid, self._graph_id)
            assert isinstance(item, Script)
            item.code = value
            await appctx.save_data_storage(label, item, self._storage_node_rid, self._graph_id)
            if self._enable_save_watch.checked:
                await self._run_button.headless_click()

    async def _on_new_script(self, value, init_str: Optional[str] = None):

        new_item_name = value["label"]
        await self.scripts.update_options([*self.scripts.props.options, value],
                                          -1)
        lang = self.langs.props.value
        assert isinstance(lang, str)
        code_lines: List[str] = []
        if lang == "python":
            code_lines.append("from tensorpc.flow import appctx")
            code_lines.append("import asyncio")
            code_lines.append("async def main():")
            if init_str is not None:
                init_str_lines = init_str.splitlines()
                init_str_lines = [" " * 4 + line for line in init_str_lines]
                code_lines.extend(init_str_lines)
            code_lines.append("    pass")
            code_lines.append("")
            code_lines.append("asyncio.get_running_loop().create_task(main())")
            code_lines.append("")
        script = Script(new_item_name, "\n".join(code_lines), lang)
        await appctx.save_data_storage(new_item_name, script, self._storage_node_rid,
                                       self._graph_id)
        await self.send_and_wait(
            self.code_editor.update_event(
                language=_LANG_TO_VSCODE_MAPPING[lang],
                value=script.code,
                path=script.label))

    async def _on_script_select(self, value):
        label = value["label"]
        item = await appctx.read_data_storage(label, self._storage_node_rid, self._graph_id)
        assert isinstance(item, Script)
        await self.langs.set_value(item.lang)
        await self.send_and_wait(
            self.code_editor.update_event(
                language=_LANG_TO_VSCODE_MAPPING[item.lang],
                value=item.code,
                path=item.label))
