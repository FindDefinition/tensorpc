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
from base64 import b64encode
from pathlib import Path
import tempfile
import asyncio
import dataclasses
import enum
import inspect
import os
import time
from types import FrameType
from typing import Any, Callable, Coroutine, Dict, Iterable, List, Optional, Set, Tuple, Union
from typing_extensions import Literal

import numpy as np
from tensorpc.constants import TENSORPC_FILE_NAME_PREFIX
from tensorpc.core.inspecttools import get_co_qualname_from_frame
from tensorpc.dbg.constants import TENSORPC_DBG_FRAMESCRIPT_STORAGE_PREFIX, DebugFrameState
from tensorpc.dbg.core.frame_id import get_frame_uid
from tensorpc.flow.components.plus.styles import CodeStyles
from tensorpc.flow.constants import TENSORPC_FLOW_APP_LANG_SERVER_PORT
from tensorpc.flow.components import mui
from tensorpc.flow import appctx

from tensorpc.flow import marker
from tensorpc.flow.components import three
from tensorpc.flow.components.plus.tutorials import AppInMemory
from tensorpc.flow.core.appcore import AppSpecialEventType, app_is_remote_comp
from tensorpc.flow.core.component import FrontendEventType
from ..options import CommonOptions
from tensorpc.flow.client import MasterMeta
from urllib.request import pathname2url

@dataclasses.dataclass
class FrameScriptItem:
    label: str
    code: str

class EditorActions(enum.Enum):
    SaveAndRun = "SaveAndRun"

class FrameScript(mui.FlexBox):

    def __init__(self):
        super().__init__()
        self.code_editor = mui.MonacoEditor("", "python",
                                            "default").prop(flex=1,
                                                            minHeight=0,
                                                            minWidth=0)
        # monaco key code 3: enter
        self.code_editor.prop(actions=[
            mui.MonacoEditorAction(id=EditorActions.SaveAndRun.value, 
                label="Save And Run", contextMenuOrder=1.5,
                keybindings=[([mui.MonacoKeyMod.Shift], 3)]),
        ])
        self.code_editor_container = mui.HBox({
            "editor":
            self.code_editor,
        }).prop(flex=1)
        self.code_editor.event_editor_action.on(self._handle_editor_action)
        self.scripts = mui.Autocomplete(
            "Scripts",
            [],
            self._on_script_select,
        ).prop(size="small",
               muiMargin="dense",
               padding="0 3px 0 3px",
               **CommonOptions.AddableAutocomplete)
        # self._enable_save_watch = mui.ToggleButton(
        #             "value",
        #             mui.IconType.Visibility).prop(muiColor="secondary", size="small")
        self._run_button = mui.IconButton(
            mui.IconType.PlayArrow,
            self._on_run_script).prop(progressColor="primary")
        self._delete_button = mui.IconButton(
            mui.IconType.Delete, self._on_script_delete).prop(
                progressColor="primary",
                confirmTitle="Warning",
                confirmMessage="Are you sure to delete this script?")
        self._header = mui.Typography("[wait for breakpoint]").prop(variant="caption", fontFamily=CodeStyles.fontFamily)
        self.init_add_layout([
            self._header,
            mui.Divider(),
            mui.HBox([
                self.scripts.prop(flex=1),
                self._run_button,
                # self._enable_save_watch,
                self._delete_button,
            ]).prop(alignItems="center"),
            self.code_editor_container,
        ])

        self.prop(flex=1,
                  flexDirection="column",
                  width="100%",
                  height="100%",
                  overflow="hidden")
        self.code_editor.event_editor_save.on(
            self._on_editor_save)
        self.code_editor.event_component_ready.on(
            self._on_editor_ready)
        self.scripts.event_select_new_item.on(
            self._on_new_script)

        self._current_frame_id: Optional[str] = None
        self._current_frame_state: Optional[DebugFrameState] = None

    # @marker.mark_did_mount
    # async def _on_mount(self):
    #     pass 

    @marker.mark_did_mount
    async def _on_mount(self):
        appctx.register_app_special_event_handler(AppSpecialEventType.RemoteCompMount, self._on_remote_comp_mount)
    
    @marker.mark_will_unmount
    async def _on_unmount(self):
        appctx.unregister_app_special_event_handler(AppSpecialEventType.RemoteCompMount, self._on_remote_comp_mount)

    async def _on_remote_comp_mount(self, data: Any):
        await self._on_editor_ready()

    async def mount_frame(self, frame_state: DebugFrameState):
        assert frame_state.frame is not None 
        if frame_state.frame.f_code.co_filename.startswith("<"):
            return
        if frame_state.frame.f_code.co_name.startswith("<"):
            return
        frame_id, title = self._get_frame_id_and_title_from_frame(frame_state.frame)
        self._current_frame_id = frame_id
        self._current_frame_state = frame_state
        ev = self.code_editor.update_event(readOnly=False)
        ev += self._header.update_event(value=title)
        await self.send_and_wait(ev)
        if appctx.get_app().app_storage.is_available():
            await self._on_editor_ready()

    async def unmount_frame(self):
        if self._current_frame_id is not None:
            self._current_frame_id = None 
            self._current_frame_state = None
        ev = self.code_editor.update_event(value="", readOnly=True)
        ev += self._header.update_event(value="[wait for breakpoint]")
        await self.send_and_wait(ev)

    async def _on_editor_ready(self):
        if self._current_frame_state is not None:
            assert appctx.get_app().app_storage.is_available()
            frame_id = self._current_frame_id
            item_dict_raw = (await appctx.glob_read_data_storage(f"{TENSORPC_DBG_FRAMESCRIPT_STORAGE_PREFIX}/{frame_id}/*"))
            item_dict_raw_list = list(item_dict_raw.items())
            item_dict_raw_list.sort(key=lambda x: x[1].timestamp, reverse=True)
            options: List[Dict[str, Any]] = []
            for key, item in item_dict_raw_list:
                options.append({"label": item.data["label"], "storage_key": key})
            if options:
                await self.scripts.update_options(options, 0)
                await self._on_script_select(options[0])
            else:
                await self._on_new_script({
                    "label": "example",
                    "storage_key": f"{TENSORPC_DBG_FRAMESCRIPT_STORAGE_PREFIX}/{frame_id}/example"
                },
                                        init_str="")

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
            assert self._current_frame_state is not None
            frame = self._current_frame_state.frame
            assert frame is not None
            storage_key = self.scripts.value["storage_key"]
            item_raw = await appctx.read_data_storage(storage_key)
            item = FrameScriptItem(**item_raw)
            fname = f"<{TENSORPC_FILE_NAME_PREFIX}-scripts-{storage_key}>"
            code = item.code
            code_comp = compile(code, fname, "exec")
            exec(code_comp, frame.f_globals, frame.f_locals)

    def _get_frame_id_and_title_from_frame(self, frame: FrameType):
        uid, meta = get_frame_uid(frame)
        title = f"{meta.module}::{meta.qualname}"
        return uid, title

    def _get_current_frame_id(self):
        assert self._current_frame_state is not None
        frame = self._current_frame_state.frame
        assert frame is not None
        frame_id, title = self._get_frame_id_and_title_from_frame(frame)
        return frame_id

    async def _on_editor_save(self, ev: mui.MonacoEditorSaveEvent):
        value = ev.value
        if self.scripts.value is not None:
            frame_id = self._get_current_frame_id()
            label = self.scripts.value["label"]
            storage_key = f"{TENSORPC_DBG_FRAMESCRIPT_STORAGE_PREFIX}/{frame_id}/{label}"
            item_raw = await appctx.read_data_storage(storage_key)
            item = FrameScriptItem(**item_raw)
            item.code = value
            await appctx.save_data_storage(storage_key, dataclasses.asdict(item))
            is_save_and_run = ev.userdata is not None and "SaveAndRun" in ev.userdata
            if is_save_and_run:
                await self._on_run_script()

    async def _on_new_script(self, value, init_str: Optional[str] = None):
        frame_id = self._get_current_frame_id()

        new_item_name = value["label"]
        await self.scripts.update_options([*self.scripts.props.options, value],
                                          -1)
        script = FrameScriptItem(new_item_name, "")
        storage_key = f"{TENSORPC_DBG_FRAMESCRIPT_STORAGE_PREFIX}/{frame_id}/{new_item_name}"

        await appctx.save_data_storage(storage_key, dataclasses.asdict(script))
        await self.send_and_wait(
            self.code_editor.update_event(
                value=script.code,
                path=script.label))

    async def _on_script_delete(self):
        if self.scripts.value is not None:
            label = self.scripts.value["label"]
            storage_key = self.scripts.value["storage_key"]

            await appctx.remove_data_storage(storage_key)
            new_options = [
                x for x in self.scripts.props.options if x["label"] != label
            ]
            await self.scripts.update_options(new_options, 0)
            if new_options:
                await self._on_script_select(new_options[0])

    async def _on_script_select(self, value):
        storage_key = value["storage_key"]

        item_raw = await appctx.read_data_storage(storage_key)
        item = FrameScriptItem(**item_raw)

        await self.send_and_wait(
            self.code_editor.update_event(
                value=item.code,
                path=item.label))
