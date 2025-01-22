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
from pathlib import Path
import tempfile
import asyncio
import dataclasses
import enum
import inspect
import os
import time
from typing import Any, Callable, Coroutine, Dict, Iterable, List, Mapping, Optional, Set, Tuple, TypeVar, Union
from typing_extensions import Literal, overload

from tensorpc.utils.containers.dict_proxy import DictProxy

import numpy as np
from tensorpc.constants import TENSORPC_FILE_NAME_PREFIX
from tensorpc.flow.constants import TENSORPC_FLOW_APP_LANG_SERVER_PORT
from tensorpc.flow.components import mui
from tensorpc.flow import appctx
# from tensorpc.core import dataclass_dispatch as dataclasses

from tensorpc.flow import marker
from tensorpc.flow.components import three
from tensorpc.flow.components.plus.tutorials import AppInMemory
from tensorpc.flow.core.appcore import AppSpecialEventType, app_is_remote_comp
from tensorpc.flow.core.component import FrontendEventType
from tensorpc.utils.code_fmt import PythonCodeFormatter
from .options import CommonOptions

from tensorpc.flow.client import MasterMeta

class EditorActions(enum.Enum):
    SaveAndRun = "SaveAndRun"

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


SCRIPT_STORAGE_KEY_PREFIX = "__tensorpc_flow_plus_script_manager"
SCRIPT_STORAGE_KEY_PREFIX_V2 = "__tensorpc_flow_plus_script_manager_v2"

SCRIPT_TEMP_STORAGE_KEY = "STORAGE"

_INITIAL_SCRIPT_PER_LANG = {
    "python": f"""
from tensorpc.flow import appctx
from tensorpc.utils.containers.dict_proxy import DictProxy
import asyncio
from typing import Any, Dict
{SCRIPT_TEMP_STORAGE_KEY}: DictProxy[str, Any] = DictProxy() # global storage of manager

async def main():
    pass
asyncio.get_running_loop().create_task(main())
    """,
    "app": f"""
from tensorpc.flow import mui, three, plus, appctx, mark_create_layout
from tensorpc.utils.containers.dict_proxy import DictProxy
from typing import Any, Dict
{SCRIPT_TEMP_STORAGE_KEY}: DictProxy[str, Any] = DictProxy() # global storage of manager

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
    """Deprecated, removed in v0.15
    """
    def __init__(self,
                 storage_node_rid: Optional[str] = None,
                 graph_id: Optional[str] = None,
                 init_scripts: Optional[Dict[str, str]] = None):
        """when storage_node_rid is None, use app node storage, else use the specified node storage
        """
        super().__init__()
        self._init_storage_node_rid = storage_node_rid
        self._init_graph_id = graph_id
        self._storage_node_rid = storage_node_rid
        self._graph_id = graph_id

        self.code_editor = mui.MonacoEditor("", "python",
                                            "default").prop(flex=1,
                                                            minHeight=0,
                                                            minWidth=0)
        self.code_editor.prop(actions=[
            mui.MonacoEditorAction(id=EditorActions.SaveAndRun.value, 
                label="Save And Run", contextMenuOrder=1.5,
                contextMenuGroupId="tensorpc-flow-editor-action", 
                keybindings=[([mui.MonacoKeyMod.Shift], 3)]),
        ])
        self.code_editor.event_editor_action.on(self._handle_editor_action)

        self.app_editor = AppInMemory("scriptmgr", "").prop(flex=1,
                                                            minHeight=0,
                                                            minWidth=0)
        self.app_show_box = mui.FlexBox()  # .prop(flex=1)

        self.code_editor_container = mui.Allotment(mui.Allotment.ChildDef([
            mui.Allotment.Pane(self.code_editor.prop(height="100%")),
            mui.Allotment.Pane(self.app_show_box.prop(height="100%"), visible=False),
        ])).prop(flex=1, minHeight=0)
        self.scripts = mui.Autocomplete(
            "Scripts",
            [],
            self._on_script_select,
        ).prop(size="small",
               textFieldProps=mui.TextFieldProps(muiMargin="dense"),
               padding="0 3px 0 3px",
               **CommonOptions.AddableAutocomplete)
        self.langs = mui.ToggleButtonGroup([
            mui.GroupToggleButtonDef("cpp", name="CPP"),
            mui.GroupToggleButtonDef("python", name="PY"),
            mui.GroupToggleButtonDef("bash", name="BASH"),
            mui.GroupToggleButtonDef("app", name="APP"),
        ], True, self._on_lang_select).prop(value="python",
                                            enforceValueSet=True)
        self._save_and_run_btn = mui.IconButton(
            mui.IconType.PlayArrow,
            self._on_save_and_run).prop(progressColor="primary")
        self._delete_button = mui.IconButton(
            mui.IconType.Delete, self._on_script_delete).prop(
                progressColor="primary",
                confirmTitle="Warning",
                confirmMessage="Are you sure to delete this script?")
        self._show_editor_btn = mui.ToggleButton(icon=mui.IconType.Code, callback=self._handle_show_editor).prop(size="small", selected=True)
        self.init_add_layout({
            "header":
            mui.HBox([
                self.scripts.prop(flex=1),
                self._save_and_run_btn,
                # self._enable_save_watch,
                self.langs,
                self._delete_button,
                self._show_editor_btn,
            ]).prop(alignItems="center"),
            "editor":
            self.code_editor_container,
        })
        self._init_scripts = _INITIAL_SCRIPT_PER_LANG.copy()
        if init_scripts is not None:
            self._init_scripts.update(init_scripts)
        self.prop(flex=1,
                  flexDirection="column",
                  width="100%",
                  height="100%",
                  minHeight=0,
                  minWidth=0,
                  overflow="hidden")
        self.code_editor.event_editor_save.on(self._on_editor_save)
        self.code_editor.event_component_ready.on(self._on_editor_ready)
        self.scripts.event_select_new_item.on(self._on_new_script)
        # used for apps and python scripts
        self._manager_global_storage: Dict[str, Any] = {}

    @marker.mark_did_mount
    async def _on_mount(self):
        if app_is_remote_comp():
            assert self._init_storage_node_rid is None, "remote comp can't specify storage node"
            assert self._init_graph_id is None, "remote comp can't specify graph id"
            self._storage_node_rid = None
            self._graph_id = None
        else:
            if self._init_storage_node_rid is None:
                self._storage_node_rid = MasterMeta().node_id
            if self._init_graph_id is None:
                self._graph_id = MasterMeta().graph_id
        appctx.register_app_special_event_handler(AppSpecialEventType.RemoteCompMount, self._on_remote_comp_mount)
    
    @marker.mark_will_unmount
    async def _on_unmount(self):
        # we clear the global storage when unmount to provide a way for user to reset the global storage
        self._manager_global_storage.clear()
        appctx.unregister_app_special_event_handler(AppSpecialEventType.RemoteCompMount, self._on_remote_comp_mount)

    async def _on_remote_comp_mount(self, data: Any):
        await self._on_editor_ready()

    async def _handle_show_editor(self, selected: bool):
        if self.langs.value == "app":
            await self.code_editor_container.update_pane_props(0, {
                "visible": selected
            })

    async def _on_editor_ready(self):
        items = await appctx.list_data_storage(
            self._storage_node_rid, self._graph_id,
            f"{SCRIPT_STORAGE_KEY_PREFIX}/*")
        items.sort(key=lambda x: x.userdata["timestamp"]
                   if not isinstance(x.userdata, mui.Undefined) else 0,
                   reverse=True)
        options: List[Dict[str, Any]] = []
        for item in items:
            if item.typeStr == Script.__name__:
                options.append({
                    "label": Path(item.name).stem,
                    "storage_key": item.name
                })
        if options:
            await self.scripts.update_options(options, 0)
            await self._on_script_select(options[0])
        else:
            default_opt = {
                "label": "example",
                "storage_key": f"{SCRIPT_STORAGE_KEY_PREFIX}/example"
            }
            await self._on_new_script(default_opt,
                                      init_str=self._init_scripts["python"])

    async def _on_save_and_run(self):
        # we attach userdata to tell save handler run script after save
        # actual run script will be handled in save handler
        await self.code_editor.save({"SaveAndRun": True})
        return

    async def _handle_editor_action(self, act_ev: mui.MonacoEditorActionEvent):
        action = act_ev.action
        if action == EditorActions.SaveAndRun.value:
            await self._on_save_and_run()

    async def _on_run_script(self):
        if self.scripts.value is not None:
            label = self.scripts.value["label"]
            storage_key = self.scripts.value["storage_key"]

            item = await appctx.read_data_storage(storage_key,
                                                  self._storage_node_rid,
                                                  self._graph_id)
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
                gs = {}
                exec(code_comp, gs,
                     {"__tensorpc_script_res": __tensorpc_script_res})
                if SCRIPT_TEMP_STORAGE_KEY in gs:
                    storage_var = gs[SCRIPT_TEMP_STORAGE_KEY]
                    if isinstance(storage_var, DictProxy):
                        storage_var.set_internal(self._manager_global_storage)
                res = __tensorpc_script_res[0]
                assert res is not None
                await res
            elif item.lang == "bash":
                proc = await asyncio.create_subprocess_shell(
                    code,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE)
                await asyncio.gather(_read_stream(proc.stdout, print),
                                     _read_stream(proc.stderr, print))
                await proc.wait()
                print(f'[cmd exited with {proc.returncode}]')
            elif item.lang == "cpp":
                import ccimport # type: ignore
                from ccimport.utils import tempdir # type: ignore
                from pathlib import Path
                import subprocess

                with tempdir() as tempd:
                    path = Path(tempd) / "source.cc"
                    exec_path = Path(tempd) / "executable"
                    with open(path, "w") as f:
                        f.write(code)
                    sources: List[Union[str, Path]] = [
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
            elif item.lang == "app":
                mod_dict = {}
                code_comp = compile(code, fname, "exec")
                exec(code_comp, mod_dict)
                if SCRIPT_TEMP_STORAGE_KEY in mod_dict:
                    storage_var = mod_dict[SCRIPT_TEMP_STORAGE_KEY]
                    if isinstance(storage_var, DictProxy):
                        storage_var.set_internal(self._manager_global_storage)
                app_cls = mod_dict["App"]
                layout = mui.flex_wrapper(app_cls())
                await self.app_show_box.set_new_layout({"layout": layout})

    async def _handle_pane_visible_status(self, lang: str):
        await self.code_editor_container.update_panes_props({
            0: {
                "visible": self._show_editor_btn.value if lang == "app" else True
            },
            1: {
                "visible": True if lang == "app" else False
            },
        })


    async def _on_lang_select(self, value):
        if value != "app":
            await self.app_show_box.set_new_layout({})
        # await self.send_and_wait(
        #     self.app_show_box.update_event(
        #         flex=1 if value == "app" else mui.undefined))
        await self._handle_pane_visible_status(value)

        if self.scripts.value is not None:
            storage_key = self.scripts.value["storage_key"]

            item = await appctx.read_data_storage(storage_key,
                                                  self._storage_node_rid,
                                                  self._graph_id)
            assert isinstance(item, Script)
            item.lang = value
            await self.send_and_wait(
                self.code_editor.update_event(
                    language=_LANG_TO_VSCODE_MAPPING[value],
                    value=item.get_code()))
            await appctx.save_data_storage(storage_key, item,
                                           self._storage_node_rid,
                                           self._graph_id)
            if value == "app":
                # TODO add better option
                await self._on_run_script()

        else:
            await self.send_and_wait(
                self.code_editor.update_event(
                    language=_LANG_TO_VSCODE_MAPPING[value]))

    async def _on_editor_save(self, ev: mui.MonacoEditorSaveEvent):
        value = ev.value
        if self.scripts.value is not None:
            label = self.scripts.value["label"]
            storage_key = f"{SCRIPT_STORAGE_KEY_PREFIX}/{label}"
            item = await appctx.read_data_storage(storage_key,
                                                  self._storage_node_rid,
                                                  self._graph_id)
            assert isinstance(item, Script)
            # compact new code dict
            if not isinstance(item.code, dict):
                item.code = self._init_scripts.copy()
            item.code[item.lang] = value

            await appctx.save_data_storage(storage_key, item,
                                           self._storage_node_rid,
                                           self._graph_id)
            is_save_and_run = ev.userdata is not None and "SaveAndRun" in ev.userdata
            if item.lang == "app" or is_save_and_run:
                await self._on_run_script()

    async def _on_new_script(self, value, init_str: Optional[str] = None):

        new_item_name = value["label"]
        storage_key = f"{SCRIPT_STORAGE_KEY_PREFIX}/{new_item_name}"

        value["storage_key"] = storage_key
        await self.scripts.update_options([*self.scripts.props.options, value],
                                          -1)
        lang = self.langs.props.value
        assert isinstance(lang, str)
        script = Script(new_item_name, self._init_scripts, lang)
        await appctx.save_data_storage(storage_key, script,
                                       self._storage_node_rid, self._graph_id)
        if lang != "app":
            await self.app_show_box.set_new_layout({})
        await self._handle_pane_visible_status(lang)
        # await self.send_and_wait(
        #     self.app_show_box.update_event(
        #         flex=1 if lang == "app" else mui.undefined))
        await self.send_and_wait(
            self.code_editor.update_event(
                language=_LANG_TO_VSCODE_MAPPING[lang],
                value=script.get_code(),
                path=script.label))
        # if value == "app":
        #     # TODO add better option
        #     await self._on_run_script()

    async def _on_script_delete(self):
        if self.scripts.value is not None:
            label = self.scripts.value["label"]
            storage_key = self.scripts.value["storage_key"]

            await appctx.remove_data_storage(storage_key,
                                             self._storage_node_rid,
                                             self._graph_id)
            new_options = [
                x for x in self.scripts.props.options if x["label"] != label
            ]
            await self.scripts.update_options(new_options, 0)
            if new_options:
                await self._on_script_select(new_options[0])

    async def _on_script_select(self, value):
        label = value["label"]
        storage_key = value["storage_key"]

        item = await appctx.read_data_storage(storage_key,
                                              self._storage_node_rid,
                                              self._graph_id)
        assert isinstance(item, Script)
        # await self.send_and_wait(
        #     self.app_show_box.update_event(
        #         flex=1 if item.lang == "app" else mui.undefined))
        await self._handle_pane_visible_status(item.lang)

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

def _create_init_script_states(lang_to_script: Optional[dict[str, str]] = None):
    if lang_to_script is None:
        lang_to_script = _INITIAL_SCRIPT_PER_LANG
    return {
        "cpp": ScriptState(lang_to_script["cpp"], True, False),
        "python": ScriptState(lang_to_script["python"], True, False),
        "bash": ScriptState(lang_to_script["bash"], True, False),
        "app": ScriptState(lang_to_script["app"], True, True),
    }

@dataclasses.dataclass
class ScriptState:
    code: str
    is_editor_visible: bool
    is_app_visible: bool

@dataclasses.dataclass
class ScriptModel:
    label: str 
    language: str = "python"
    states: dict[str, ScriptState] = dataclasses.field(default_factory=_create_init_script_states)

    def get_cur_state(self):
        return self.states[self.language]

@dataclasses.dataclass
class ScriptManagerModel:
    scripts: list[ScriptModel]
    cur_script_idx: int = -1


class ScriptManagerV2(mui.FlexBox):

    def __init__(self,
                 storage_node_rid: Optional[str] = None,
                 graph_id: Optional[str] = None,
                 init_scripts: Optional[Dict[str, str]] = None):
        """when storage_node_rid is None, use app node storage, else use the specified node storage
        """
        super().__init__()
        self._init_storage_node_rid = storage_node_rid
        self._init_graph_id = graph_id
        self._storage_node_rid = storage_node_rid
        self._graph_id = graph_id
        self._editor_path_uid = "scriptmgr_v2"
        init_model = ScriptManagerModel([], -1)
        self.code_editor = mui.MonacoEditor("", "python",
                                            "default").prop(flex=1,
                                                            minHeight=0,
                                                            minWidth=0)
        self._code_fmt = PythonCodeFormatter()
        editor_acts: list[mui.MonacoEditorAction] = [
            mui.MonacoEditorAction(id=EditorActions.SaveAndRun.value, 
                label="Save And Run", contextMenuOrder=1.5,
                contextMenuGroupId="tensorpc-flow-editor-action", 
                keybindings=[([mui.MonacoKeyMod.Shift], 3)]),
        ]
        for backend in self._code_fmt.get_all_supported_backends():
            editor_acts.append(
                mui.MonacoEditorAction(id=f"FormatCode-{backend}",
                                       label=f"Format Code ({backend})",
                                       contextMenuOrder=1.5,
                                       contextMenuGroupId="tensorpc-flow-editor-action",
                                       userdata={"backend": backend})
            )
        self.code_editor.prop(actions=editor_acts)
        self.code_editor.event_editor_action.on(self._handle_editor_action)

        self.app_editor = AppInMemory("scriptmgr", "").prop(flex=1,
                                                            minHeight=0,
                                                            minWidth=0)
        self.app_show_box = mui.FlexBox()  # .prop(flex=1)

        self.code_editor_container = mui.Allotment(mui.Allotment.ChildDef([
            mui.Allotment.Pane(self.code_editor.prop(height="100%")),
            mui.Allotment.Pane(self.app_show_box.prop(height="100%"), visible=False),
        ])).prop(flex=1, minHeight=0)

        self.code_editor_container.bind_fields(visibles="[getitem(states, language).is_editor_visible, getitem(states, language).is_app_visible]")
        self._scripts_select = mui.Autocomplete(
            "Scripts",
            [],
            self._on_script_select,
        ).prop(size="small",
               textFieldProps=mui.TextFieldProps(muiMargin="dense"),
               padding="0 3px 0 3px",
               **CommonOptions.AddableAutocomplete)
        self.langs = mui.ToggleButtonGroup([
            mui.GroupToggleButtonDef("cpp", name="CPP"),
            mui.GroupToggleButtonDef("python", name="PY"),
            mui.GroupToggleButtonDef("bash", name="BASH"),
            mui.GroupToggleButtonDef("app", name="APP"),
        ]).prop(enforceValueSet=True)
        self._save_and_run_btn = mui.IconButton(
            mui.IconType.PlayArrow,
            self._on_save_and_run).prop(progressColor="primary")
        self._delete_button = mui.IconButton(
            mui.IconType.Delete, self._on_script_delete).prop(
                progressColor="primary",
                confirmTitle="Warning",
                confirmMessage="Are you sure to delete this script?")
        self._save_and_run_btn.bind_fields(disabled="cur_script_idx == `-1`")
        self._delete_button.bind_fields(disabled="cur_script_idx == `-1`")
        self.code_editor.bind_fields(readOnly="cur_script_idx == `-1`")
        self._show_editor_btn = mui.ToggleButton(icon=mui.IconType.Code, callback=self._handle_show_editor).prop(size="small")
        self._show_editor_btn.bind_fields(selected="getitem(states, language).is_editor_visible")
        self.model = mui.DataModel(init_model, [
            mui.HBox([
                self._scripts_select.prop(flex=1).bind_fields(
                    options=r"scripts[*].{label: label}", 
                    value=r"getitem(scripts, cur_script_idx).{label: label}"),
                self.langs.bind_fields(disabled="cur_script_idx == `-1`"),
                self._save_and_run_btn,
                # self._enable_save_watch,
                self._delete_button,
                mui.DataSubQuery("getitem(scripts, cur_script_idx)", [
                    self._show_editor_btn,
                ]).bind_fields(enable="cur_script_idx != `-1`"),
            ]).prop(alignItems="center"),
            mui.DataSubQuery("getitem(scripts, cur_script_idx)", [
                self.code_editor_container,
            ]).bind_fields(enable="cur_script_idx != `-1`"),
        ])
        self.model.connect_app_storage(f"{SCRIPT_STORAGE_KEY_PREFIX_V2}")
        draft = self.model.get_draft_type_only()
        self.langs.bind_draft_change(draft.scripts[draft.cur_script_idx].language)
        # make sure lang change is handled before `_on_lang_select`
        self.langs.event_change.on(self._on_lang_select)

        self.init_add_layout([
            self.model,
        ])

        self._init_scripts = _INITIAL_SCRIPT_PER_LANG.copy()
        if init_scripts is not None:
            self._init_scripts.update(init_scripts)
        self.prop(flex=1,
                  flexDirection="column",
                  width="100%",
                  height="100%",
                  minHeight=0,
                  minWidth=0,
                  overflow="hidden")
        self.code_editor.event_editor_save.on(self._on_editor_save)
        self._scripts_select.event_select_new_item.on(self._on_new_script)

        self.model.event_storage_fetched.on(self._on_editor_ready)

        # used for apps and python scripts
        self._manager_global_storage: Dict[str, Any] = {}

    @marker.mark_did_mount
    async def _on_mount(self):
        if app_is_remote_comp():
            assert self._init_storage_node_rid is None, "remote comp can't specify storage node"
            assert self._init_graph_id is None, "remote comp can't specify graph id"
            self._storage_node_rid = None
            self._graph_id = None
        else:
            if self._init_storage_node_rid is None:
                self._storage_node_rid = MasterMeta().node_id
            if self._init_graph_id is None:
                self._graph_id = MasterMeta().graph_id
        appctx.register_app_special_event_handler(AppSpecialEventType.RemoteCompMount, self._on_remote_comp_mount)
    
    @marker.mark_will_unmount
    async def _on_unmount(self):
        # we clear the global storage when unmount to provide a way for user to reset the global storage
        self._manager_global_storage.clear()
        appctx.unregister_app_special_event_handler(AppSpecialEventType.RemoteCompMount, self._on_remote_comp_mount)

    async def _on_remote_comp_mount(self, data: Any):
        await self._on_editor_ready()

    async def _handle_show_editor(self, selected: bool):
        draft = self.model.get_draft()
        if self.model.model.cur_script_idx != -1:
            cur_script = draft.scripts[self.model.model.cur_script_idx]
            cur_script.states[cur_script.language].is_editor_visible = selected

    async def _set_code_editor(self, script: ScriptModel):
        await self.send_and_wait(
            self.code_editor.update_event(
                language=_LANG_TO_VSCODE_MAPPING[script.language],
                value=script.states[script.language].code,
                path=self._get_path(script.label, script.language)))

    async def _on_editor_ready(self):
        draft = self.model.get_draft()
        model = self.model.model
        if model.scripts:
            if model.cur_script_idx == -1:
                draft.cur_script_idx = 0
            cur_script = model.scripts[model.cur_script_idx]
            await self._set_code_editor(cur_script)
            if cur_script.language == "app":
                await self._on_run_script()

    async def _on_save_and_run(self):
        # we attach userdata to tell save handler run script after save
        # actual run script will be handled in save handler
        await self.code_editor.save({"SaveAndRun": True})
        return

    async def _handle_editor_action(self, act_ev: mui.MonacoEditorActionEvent):
        action = act_ev.action
        if action == EditorActions.SaveAndRun.value:
            await self._on_save_and_run()
        elif action.startswith("FormatCode-"):
            assert act_ev.userdata is not None 
            backend = act_ev.userdata["backend"]
            cur_idx = self.model.model.cur_script_idx
            cur_script = self.model.model.scripts[cur_idx]
            cur_state = cur_script.states[cur_script.language]
            new_code = self._code_fmt.format_code(cur_state.code, backend)
            async with self.model.draft_update() as draft:
                draft.scripts[cur_idx].states[cur_script.language].code = new_code
            await self._set_code_editor(cur_script)

    async def _on_lang_select(self, value):
        if value != "app":
            await self.app_show_box.set_new_layout({})
        if self.model.model.cur_script_idx != -1:
            await self._set_code_editor(self.model.model.scripts[self.model.model.cur_script_idx])
            if value == "app":
                # TODO add better option
                await self._on_run_script(value)
        else:
            await self.send_and_wait(
                self.code_editor.update_event(
                    language=_LANG_TO_VSCODE_MAPPING[value]))

    async def _on_editor_save(self, ev: mui.MonacoEditorSaveEvent):
        value = ev.value
        model = self.model.model
        draft = self.model.get_draft()
        if model.cur_script_idx != -1:
            cur_script = model.scripts[model.cur_script_idx]
            cur_script_draft = draft.scripts[model.cur_script_idx]
            cur_script_draft.states[cur_script.language].code = value
            is_save_and_run = ev.userdata is not None and "SaveAndRun" in ev.userdata
            if cur_script.language == "app" or is_save_and_run:
                # when we use draft update, real update (include backend model) is delayed
                await self._on_run_script(code=value)

    def _get_path(self, name: str, lang: str):
        return f"{self._editor_path_uid}/{name}.{lang}"

    async def _on_new_script(self, value):
        new_item_name = value["label"]
        new_script_model = ScriptModel(new_item_name)
        draft = self.model.get_draft()
        draft.scripts.append(new_script_model)
        # draft update is delayed, so we use len(...) instead of len(...) - 1
        draft.cur_script_idx = len(self.model.model.scripts)
        await self.app_show_box.set_new_layout({})
        await self._set_code_editor(new_script_model)

    async def _on_script_delete(self):
        model = self.model.model
        draft = self.model.get_draft()
        if model.cur_script_idx != -1:
            if len(model.scripts) == 1:
                draft.scripts.pop()
                draft.cur_script_idx = -1
                await self.send_and_wait(
                    self.code_editor.update_event(
                        language="python",
                        value="# let's add a new script",
                        path=""))
            else:
                draft.scripts.pop(model.cur_script_idx)
                draft.cur_script_idx = 0

    async def _on_script_select(self, value):
        label = value["label"]
        # find script model by label
        model = self.model.model
        idx = -1
        for i, script in enumerate(model.scripts):
            if script.label == label:
                idx = i
                break
        if idx == -1:
            raise ValueError("shouldn't happen")
        # draft is delayed after all event handlers of each component event.
        # so we use draft_update to do update immediately
        async with self.model.draft_update() as draft:
            draft.cur_script_idx = idx
        cur_script = model.scripts[idx]
        await self._set_code_editor(cur_script)
        if cur_script.language != "app":
            await self.app_show_box.set_new_layout({})
        else:
            await self._on_run_script()

    async def _on_run_script(self, cur_lang: Optional[str] = None, code: Optional[str] = None):
        if self.model.model.cur_script_idx != -1:
            cur_script = self.model.model.scripts[self.model.model.cur_script_idx]
            if cur_lang is None:
                cur_lang = cur_script.language
            item_uid = f"{self._graph_id}@{self._storage_node_rid}@{cur_script.label}"
            fname = f"<{TENSORPC_FILE_NAME_PREFIX}-scripts-{item_uid}>"
            if code is None:
                code = cur_script.states[cur_lang].code
            label = cur_script.label
            if cur_lang == "python":
                __tensorpc_script_res: List[Optional[Coroutine]] = [None]
                lines = code.splitlines()
                lines = [" " * 4 + line for line in lines]
                run_name = f"run_{label}"
                lines.insert(0, f"async def _{run_name}():")
                lines.append(f"__tensorpc_script_res[0] = _{run_name}()")
                code = "\n".join(lines)
                code_comp = compile(code, fname, "exec")
                gs = {}
                exec(code_comp, gs,
                     {"__tensorpc_script_res": __tensorpc_script_res})
                if SCRIPT_TEMP_STORAGE_KEY in gs:
                    storage_var = gs[SCRIPT_TEMP_STORAGE_KEY]
                    if isinstance(storage_var, DictProxy):
                        storage_var.set_internal(self._manager_global_storage)
                res = __tensorpc_script_res[0]
                assert res is not None
                await res
            elif cur_lang == "bash":
                proc = await asyncio.create_subprocess_shell(
                    code,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE)
                await asyncio.gather(_read_stream(proc.stdout, print),
                                     _read_stream(proc.stderr, print))
                await proc.wait()
                print(f'[cmd exited with {proc.returncode}]')
            elif cur_lang == "cpp":
                import ccimport # type: ignore
                from ccimport.utils import tempdir # type: ignore
                from pathlib import Path
                import subprocess

                with tempdir() as tempd:
                    path = Path(tempd) / "source.cc"
                    exec_path = Path(tempd) / "executable"
                    with open(path, "w") as f:
                        f.write(code)
                    sources: List[Union[str, Path]] = [
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
            elif cur_lang == "app":
                mod_dict = {}
                code_comp = compile(code, fname, "exec")
                exec(code_comp, mod_dict)
                if SCRIPT_TEMP_STORAGE_KEY in mod_dict:
                    storage_var = mod_dict[SCRIPT_TEMP_STORAGE_KEY]
                    if isinstance(storage_var, DictProxy):
                        storage_var.set_internal(self._manager_global_storage)
                app_cls = mod_dict["App"]
                layout = mui.flex_wrapper(app_cls())
                await self.app_show_box.set_new_layout({"layout": layout})

