import enum
from functools import partial
import inspect
from typing import Any, Awaitable, Callable, Optional, Union

from tensorpc.core.datamodel.events import DraftChangeEvent
from tensorpc.core.tree_id import UniqueTreeId
from tensorpc.dock import terminal
import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.dock.components import mui
import tensorpc.core.datamodel as D
from tensorpc.apps.cm.coretypes import RaftMgrActions, WorkerUISSHState, WorkerUIType
from tensorpc.dock.components.plus.styles import get_tight_icon_tab_theme, get_tight_icon_tab_theme_horizontal


def _get_terminal_menus(term: terminal.AsyncSSHTerminal):
    
    return mui.MenuList([
        mui.MenuItem("Clear", "Clear", iconSize="small", iconFontSize="small")
    ], term).prop(
            flex=1, overflow="auto", display="flex", flexFlow="column nowrap", triggerMethod="contextmenu",
            anchorReference="anchorPosition", dense=True,
            paperProps=mui.PaperProps(width="20ch"))

class RaftManagerPanel(mui.FlexBox):

    def __init__(self, group_id: str, uid: str, dm: mui.DataModel, master_action_fn: Callable[[RaftMgrActions], mui.CORO_ANY],
                 draft: WorkerUISSHState, term: Optional[terminal.AsyncSSHTerminal], default_path: str = "default"):
        self._uid = uid
        self._remote_term_key = UniqueTreeId.from_parts([group_id, WorkerUIType.TERMINAL.value]).uid_encoded
        start_or_cancel_btn = mui.IconButton(
            mui.IconType.PlayArrow, partial(master_action_fn, RaftMgrActions.START_OR_CANCEL)).prop(iconSize="small",
                                                             size="small")
        stop_btn = mui.IconButton(mui.IconType.Stop, partial(master_action_fn, RaftMgrActions.SHUTDOWN_ALL)).prop(
            iconSize="small",
            size="small",
            muiColor="error",
            confirmTitle="Dangerous Operation",
            confirmMessage=
            "Are you sure to shutdown (ctrl-c->terminate->kill) ALL running process?",
            tooltip="shutdown command")
        kill_btn = mui.IconButton(mui.IconType.Delete, partial(master_action_fn, RaftMgrActions.KILL_ALL)).prop(
            iconSize="small",
            size="small",
            confirmTitle="Dangerous Operation",
            confirmMessage="Are you sure to kill ALL running process?",
            tooltip="kill all child process")
        header_str = mui.Typography().prop(variant="body2",
                                        color="primary")
        header_str.bind_pfl_query(dm, value=(WorkerUISSHState.get_common_query, "header"))
        worker_select = mui.Autocomplete("Workers", []).prop(textFieldProps=mui.TextFieldProps(muiMargin="none"),
                                                           size="small",
                                                           groupByKey="ssh_status",
                                                           margin="0 10px 0 10px")
        # header = mui.HBox([
        #     mui.HBox([
        #         header_str,
        #     ]).prop(flex=1),
        #     start_or_cancel_btn,
        #     stop_btn,
        #     kill_btn,
        # ])
        self._code_editor = mui.MonacoEditor("", "shell",
                                             default_path).prop(debounce=300,
                                                          height="300px")
        self._terminal = term
        tab_defs = [
            mui.TabDef("",
                       "control",
                        mui.VBox([
                            self._code_editor,
                        ]).prop(width="100%", height="100%", overflow="hidden"),
                       icon=mui.IconType.Settings,
                       tooltip="Control"),
        ]
        init_value = "control"
        self._remote_terminal_box = mui.VBox([]).prop(flex=1, overflow="hidden")

        if term is not None:
            _terminal_box = _get_terminal_menus(term)
            _terminal_box.event_contextmenu_select.on(self._on_term_menu)
            _terminal_panel = mui.MatchCase.binary_selection(
                True, _terminal_box, self._remote_terminal_box)
            init_value = "terminal"
            _terminal_panel.bind_pfl_query(dm, condition=(WorkerUISSHState.get_common_query, "terminal_is_local"))
        else:
            _terminal_panel =  self._remote_terminal_box
        tab_defs.append(
            mui.TabDef("",
                    "terminal",
                        mui.VBox([
                            _terminal_panel,
                        ]).prop(width="100%", height="100%", overflow="hidden"),
                    icon=mui.IconType.Terminal,
                    tooltip="Terminal"),

        )

        self._tabs = mui.Tabs(tab_defs, init_value=init_value, before=[
            # header,
            header_str,
            worker_select.prop(flex=1),
            start_or_cancel_btn,
            stop_btn,
            kill_btn,
            mui.VDivider(),
        ])
        self._tabs.prop(panelProps=mui.FlexBoxProps(
                                  height="100%", padding=0,),
                        orientation="horizontal",
                        borderBottom=1,
                        flex=1,
                        borderColor='divider',
                        # overflow="hidden",
                        tooltipPlacement="top",
                        display="flex",
                        alignItems="center")
        # ssh_panel = mui.VBox([
        #     header,
        #     mui.HBox([
        #         worker_select.prop(flex=1),
        #     ]),
        #     self._code_editor,
        #     self._terminal_panel,
        # ]).prop(width="100%", height="100%", overflow="hidden")
        ssh_panel = mui.ThemeProvider([
            mui.VBox([
                self._tabs
            ]).prop(flex=1, overflow="hidden")
        ], get_tight_icon_tab_theme_horizontal(size="40px"))


        # self.dm.event_storage_fetched.on(self._init_fields_when_fetch_model)
        master_draft = draft
        start_or_cancel_btn.bind_pfl_query(dm, 
            icon=(WorkerUISSHState.get_common_query, "start_or_cancel_icon"),
            disabled=(WorkerUISSHState.get_common_query, "not_leader_disabled"))
        stop_btn.bind_pfl_query(dm, disabled=(WorkerUISSHState.get_common_query, "stop_btn_disabled"))
        kill_btn.bind_pfl_query(dm, disabled=(WorkerUISSHState.get_common_query, "stop_btn_disabled"))

        self._code_editor.bind_draft_change_uncontrolled(master_draft.user_cmd)
        dm.install_draft_change_handler(
            master_draft.cur_worker,
            self._handle_worker_change)
        worker_select.bind_draft_change(master_draft.cur_worker)
        worker_select.bind_fields(options=master_draft.workers)
        worker_select.bind_pfl_query(dm, disabled=(WorkerUISSHState.get_common_query, "not_leader_disabled"))
        super().__init__([
            ssh_panel,
        ])
        self.prop(flexDirection="column", flex=1)

    async def _on_term_menu(self, item_id: str):
        if self._terminal is None:
            return 
        if item_id == "Clear":
            await self._terminal.clear()
        else:
            raise ValueError(f"Unknown menu item {item_id}")

    async def _handle_worker_change(self, ev: DraftChangeEvent):
        selected_state_dict = ev.new_value
        if selected_state_dict is not None:
            uid = selected_state_dict.id
            url = selected_state_dict.url
            ip, port_str = url.split(":")
            port = int(port_str)
            if ev.old_value is not None:
                if ev.old_value.id == uid:
                    return
            if uid == self._uid:
                await self._remote_terminal_box.set_new_layout([])
            else:
                await self._remote_terminal_box.set_new_layout([
                    mui.RemoteBoxGrpc(ip, port,
                                      self._remote_term_key).prop(flex=1)
                ])
        else:
            await self._remote_terminal_box.set_new_layout([])
