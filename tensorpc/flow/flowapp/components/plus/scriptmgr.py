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
import tempfile
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
from tensorpc.flow.flowapp.components.plus.tutorials import AppInMemory
from tensorpc.flow.flowapp.core import FrontendEventType
from .options import CommonOptions

from tensorpc.flow.client import MasterMeta

@dataclasses.dataclass
class Script:
    label: str
    code: Union[str, Dict[str, str]]
    lang: str

    def get_code(self):
        if isinstance(self.code, dict):
            return self.code.get(self.lang, "")
        else:
            return self.code

_LANG_TO_VSCODE_MAPPING = {
    "python": "python",
    "cpp": "cpp",
    "bash": "shell",
    "app": "python",
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

_INITIAL_SCRIPT_PER_LANG = {
    "python": """
from tensorpc.flow import appctx
import asyncio
async def main():
    pass
asyncio.get_running_loop().create_task(main())
    """,
    "app": """
from tensorpc.flow import mui, three, plus, appctx, mark_create_layout
class App:
    @mark_create_layout
    def my_layout(self):
        return mui.VBox([
            mui.Typography("Hello World"),
        ])
    """,
    "cpp": """
#include <iostream>
int main(){
    std::cout << "Hello World" << std::endl;
    return 0;
}

    """,
    "bash": """
echo "Hello World"
    """,
}

class ScriptManager(mui.FlexBox):

    def __init__(
        self,
        storage_node_rid: Optional[str] = None,
        graph_id: Optional[str] = None,
        init_scripts: Optional[Dict[str, str]] = None
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
        self.app_editor = AppInMemory("scriptmgr", "").prop(flex=1, minHeight=0, minWidth=0)
        self.app_show_box = mui.FlexBox() # .prop(flex=1)

        self.code_editor_container = mui.HBox({
            "editor": self.code_editor,
            "divider":  mui.Divider("horizontal"),
            "app_show_box": self.app_show_box
        }).prop(flex=1)
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
            mui.ToggleButton("bash", name="BASH"),
            mui.ToggleButton("app", name="APP"),
        ], True, self._on_lang_select).prop(value="python",
                                            enforceValueSet=True)
        # self._enable_save_watch = mui.ToggleButton(
        #             "value",
        #             mui.IconType.Visibility).prop(muiColor="secondary", size="small")
        self._run_button = mui.IconButton(
                    mui.IconType.PlayArrow,
                    self._on_run_script).prop(progressColor="primary")
        self._delete_button = mui.IconButton(
                    mui.IconType.Delete,
                    self._on_script_delete).prop(progressColor="primary", confirmTitle="Warning", confirmMessage="Are you sure to delete this script?")

        self.init_add_layout({
            "header": mui.HBox([
                self.scripts.prop(flex=1),
                self._run_button,
                # self._enable_save_watch,
                self.langs,
                self._delete_button,

            ]).prop(alignItems="center"),
            "editor": self.code_editor_container,
        })
        self._init_scripts = _INITIAL_SCRIPT_PER_LANG.copy()
        if init_scripts is not None:
            self._init_scripts.update(init_scripts)
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
            await self._on_new_script({
                    "label": "example",
                }, init_str=self._init_scripts["python"])

    async def _on_run_script(self):
        await self.code_editor.save()
        if self.scripts.value is not None:
            label = self.scripts.value["label"]
            item = await appctx.read_data_storage(label,
                                                  self._storage_node_rid, self._graph_id)
            assert isinstance(item, Script)
            item_uid = f"{self._graph_id}@{self._storage_node_rid}@{item.label}"
            fname = f"<{TENSORPC_FILE_NAME_PREFIX}-scripts-{item_uid}>"
            if isinstance(item.code, dict):
                code = item.code.get(item.lang, "")
            else:
                code = item.code
            if item.lang == "python":
                __tensorpc_script_res: List[Optional[Coroutine]] = [None]
                lines = code.splitlines()
                lines = [" " * 4 + line for line in lines]
                run_name = f"run_{label}"
                lines.insert(0, f"async def _{run_name}():")
                lines.append(f"__tensorpc_script_res[0] = _{run_name}()")
                code = "\n".join(lines)
                code_comp = compile(code, fname, "exec")
                exec(code_comp, {},
                     {"__tensorpc_script_res": __tensorpc_script_res})
                res = __tensorpc_script_res[0]
                assert res is not None
                await res
            elif item.lang == "bash":
                proc = await asyncio.create_subprocess_shell(
                    code,
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
            elif item.lang == "cpp":
                import ccimport
                from ccimport.utils import tempdir
                from pathlib import Path 
                import subprocess

                with tempdir() as tempd:
                    path = Path(tempd) / "source.cc"
                    exec_path = Path(tempd) / "executable"
                    with open(path, "w") as f:
                        f.write(code)
                    sources = [
                        path,
                    ]
                    build_meta = ccimport.BuildMeta()
                    source = ccimport.ccimport(sources,
                                            exec_path,
                                            build_meta,
                                            shared=False,
                                            load_library=False,
                                            verbose=False)
                    subprocess.check_call([str(source)])
            if item.lang == "app":
                mod_dict = {}
                code_comp = compile(code, fname, "exec")
                exec(code_comp, mod_dict)
                app_cls = mod_dict["App"]
                layout = mui.flex_wrapper(app_cls())
                await self.app_show_box.set_new_layout({"layout": layout})

    async def _on_lang_select(self, value):
        if value != "app":
            await self.app_show_box.set_new_layout({})
        await self.send_and_wait(self.app_show_box.update_event(flex=1 if value == "app" else mui.undefined))

        if self.scripts.value is not None:
            label = self.scripts.value["label"]

            item = await appctx.read_data_storage(label,
                                                  self._storage_node_rid, self._graph_id)
            assert isinstance(item, Script)
            item.lang = value
            await self.send_and_wait(
                self.code_editor.update_event(
                    language=_LANG_TO_VSCODE_MAPPING[value], value=item.get_code()))
            if value == "app":
                # TODO add better option
                await self._on_run_script()
            await appctx.save_data_storage(label, item, self._storage_node_rid, self._graph_id)
        else:
            await self.send_and_wait(
                self.code_editor.update_event(
                    language=_LANG_TO_VSCODE_MAPPING[value]))

    async def _on_editor_save(self, value: str):
        if self.scripts.value is not None:
            label = self.scripts.value["label"]
            item = await appctx.read_data_storage(label,
                                                  self._storage_node_rid, self._graph_id)
            assert isinstance(item, Script)
            # compact new code dict
            if not isinstance(item.code, dict):
                item.code = self._init_scripts.copy()
            item.code[item.lang] = value
            await appctx.save_data_storage(label, item, self._storage_node_rid, self._graph_id)
            if item.lang == "app":
                await self._on_run_script()
            # if self._enable_save_watch.checked:
            #     await self._run_button.headless_click()

    async def _on_new_script(self, value, init_str: Optional[str] = None):

        new_item_name = value["label"]
        await self.scripts.update_options([*self.scripts.props.options, value],
                                          -1)
        lang = self.langs.props.value
        assert isinstance(lang, str)
        script = Script(new_item_name, self._init_scripts, lang)
        await appctx.save_data_storage(new_item_name, script, self._storage_node_rid,
                                       self._graph_id)
        await self.send_and_wait(self.app_show_box.update_event(flex=1 if lang == "app" else mui.undefined))
        await self.send_and_wait(
            self.code_editor.update_event(
                language=_LANG_TO_VSCODE_MAPPING[lang],
                value=script.get_code(),
                path=script.label))

    async def _on_script_delete(self):
        if self.scripts.value is not None:
            label = self.scripts.value["label"]
            await appctx.remove_data_storage(label, self._storage_node_rid, self._graph_id)
            new_options =  [x for x in self.scripts.props.options if x["label"] != label]
            await self.scripts.update_options(
            new_options, 0)
            if new_options:
                await self._on_script_select(new_options[0])

    async def _on_script_select(self, value):
        label = value["label"]
        item = await appctx.read_data_storage(label, self._storage_node_rid, self._graph_id)
        assert isinstance(item, Script)
        await self.send_and_wait(self.app_show_box.update_event(flex=1 if item.lang == "app" else mui.undefined))
            
        await self.langs.set_value(item.lang)
        await self.send_and_wait(
            self.code_editor.update_event(
                language=_LANG_TO_VSCODE_MAPPING[item.lang],
                value=item.get_code(),
                path=item.label))
        if item.lang != "app":
            await self.app_show_box.set_new_layout({})
        else:
            await self._on_run_script()