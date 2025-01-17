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

from tensorpc.flow import marker
from tensorpc.flow.components import three
from tensorpc.flow.components.plus.tutorials import AppInMemory
from tensorpc.flow.core.appcore import AppSpecialEventType, app_is_remote_comp
from tensorpc.flow.core.component import FrontendEventType
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

def _create_states():
    return {
        "cpp": ScriptState(True, False),
        "python": ScriptState(True, False),
        "bash": ScriptState(True, False),
        "app": ScriptState(True, True),
    }

@dataclasses.dataclass
class ScriptState:
    is_editor_visible: bool
    is_app_visible: bool

@dataclasses.dataclass
class ScriptModel:
    label: str 
    storage_key: str
    language: Literal["cpp", "python", "bash", "app"] = "python"
    states: dict[str, ScriptState] = dataclasses.field(default_factory=_create_states)

@dataclasses.dataclass
class ScriptManagerModel:
    scripts: list[ScriptModel]
    cur_script_idx: Optional[int] = None


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

        init_model = ScriptManagerModel([], None)
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

        self.code_editor_container.bind_fields(visibles="[getitem(states, language).is_editor_visible, getitem(states, language).is_app_visible]")
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
        ], True, self._on_lang_select).prop(enforceValueSet=True)
        self.langs.bind_fields(value="getitem(scripts, cur_script_idx).language")
        self._save_and_run_btn = mui.IconButton(
            mui.IconType.PlayArrow,
            self._on_save_and_run).prop(progressColor="primary")
        self._delete_button = mui.IconButton(
            mui.IconType.Delete, self._on_script_delete).prop(
                progressColor="primary",
                confirmTitle="Warning",
                confirmMessage="Are you sure to delete this script?")
        self._show_editor_btn = mui.ToggleButton(icon=mui.IconType.Code, callback=self._handle_show_editor).prop(size="small")
        self._show_editor_btn.bind_fields(selected="getitem(states, language).is_editor_visible")
        self.model = mui.DataModel(init_model, [
            mui.HBox([
                self.scripts.prop(flex=1).bind_fields(options="scripts", value="getitem(scripts, cur_script_idx)"),
                mui.DataSubQuery("getitem(scripts, cur_script_idx)", [
                    self._save_and_run_btn,
                    # self._enable_save_watch,
                    self.langs,
                    self._delete_button,
                    self._show_editor_btn,
                ]).bind_fields(enable="cur_script_idx != null"),
            ]).prop(alignItems="center"),
            mui.DataSubQuery("getitem(scripts, cur_script_idx)", [
                self.code_editor_container,
            ]).bind_fields(enable="cur_script_idx != null"),
        ])

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
        self.event_after_mount.on(self._on_editor_ready)
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
        draft = self.model.get_draft()
        if self.model.model.cur_script_idx is not None:
            cur_script_real = self.model.model.scripts[self.model.model.cur_script_idx]
            cur_script = draft.scripts[self.model.model.cur_script_idx]
            cur_script.states[cur_script_real.language].is_editor_visible = selected

    async def _on_editor_ready(self):
        items = await appctx.list_data_storage(
            self._storage_node_rid, self._graph_id,
            f"{SCRIPT_STORAGE_KEY_PREFIX}/*")
        items.sort(key=lambda x: x.userdata["timestamp"]
                   if not isinstance(x.userdata, mui.Undefined) else 0,
                   reverse=True)
        options: List[Dict[str, Any]] = []
        options_v2: List[ScriptModel] = []
        for item in items:
            if item.typeStr == Script.__name__:
                options.append({
                    "label": Path(item.name).stem,
                    "storage_key": item.name
                })
                options_v2.append(ScriptModel(label=Path(item.name).stem, storage_key=item.name))
        if options_v2:
            draft = self.model.get_draft()
            draft.scripts = options_v2
            draft.cur_script_idx = 0
        else:
            raise NotImplementedError
        # if options:
        #     await self.scripts.update_options(options, 0)
        #     await self._on_script_select(options[0])
        # else:
        #     default_opt = {
        #         "label": "example",
        #         "storage_key": f"{SCRIPT_STORAGE_KEY_PREFIX}/example"
        #     }
        #     await self._on_new_script(default_opt,
        #                               init_str=self._init_scripts["python"])

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
