from functools import partial
import inspect
from typing import Any, Awaitable, Callable, Optional, Union

from tensorpc.apps.dbg.components.dbgpanel import MasterDebugPanel
from tensorpc.apps.dbg.components.distpyspy import PyspyViewer
from tensorpc.core.datamodel.events import DraftChangeEvent
from tensorpc.core.tree_id import UniqueTreeId
from tensorpc.dock import terminal
from tensorpc.dock.components import mui
from tensorpc.apps.cm.coretypes import CM_LOGGER, RaftMgrActions, WorkerUISSHState, WorkerUIType
from tensorpc.dock.components.plus.styles import get_tight_icon_tab_theme_horizontal
from tensorpc.utils.pyspyutil import PyspyTraceMode, fetch_pyspy_info


def _get_terminal_menus(term: terminal.AsyncSSHTerminal):
    
    return mui.MenuList([
        mui.MenuItem("Clear", "Clear", iconSize="small", iconFontSize="small")
    ], term).prop(
            flex=1, overflow="auto", display="flex", flexFlow="column nowrap", triggerMethod="contextmenu",
            anchorReference="anchorPosition", dense=True,
            paperProps=mui.PaperProps(width="20ch"))

class RaftManagerPanel(mui.FlexBox):

    def __init__(self, is_raft_node: bool, group_id: str, uid: str, dm: mui.DataModel, 
                master_action_fn: Callable[[RaftMgrActions], mui.CORO_ANY],
                debug_panel_fn: Optional[Callable[..., Awaitable[None]]],
                fetch_pyspy_info_fn: Optional[Callable[[PyspyTraceMode, bool], Awaitable[None]]],
                draft: WorkerUISSHState, term: Optional[terminal.AsyncSSHTerminal], default_path: str = "default"):
        self._uid = uid
        self._draft = draft
        self._remote_term_key = UniqueTreeId.from_parts([group_id, WorkerUIType.TERMINAL.value]).uid_encoded
        start_or_cancel_btn = mui.IconButton(
            mui.IconType.PlayArrow, partial(master_action_fn, RaftMgrActions.START_OR_CANCEL)).prop(iconSize="small",
                                                             size="small")
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
        self._code_editor = mui.MonacoEditor("", "shell",
                                             default_path).prop(debounce=300,
                                                          height="300px")
        self._terminal = term

        menu_items: list[mui.MenuItem] = []
        cared_menu_acts = [RaftMgrActions.RECONNECT_ALL_CLIENT, RaftMgrActions.CLEAR_ALL_TERMINALS]
        for action in RaftMgrActions:
            if action in cared_menu_acts:
                menu_items.append(mui.MenuItem(action.value, action.value, 
                    confirmMessage="Are You Sure?", 
                    confirmTitle=f"Dangerous Operation ({action.value})"))
        menu_items.extend([
            mui.MenuItem("divider1", divider=True),
            mui.MenuItem(RaftMgrActions.PYTORCH_SPY.value, "PyTorch Dist Spy"),
            mui.MenuItem("divider2", divider=True),
            mui.MenuItem(RaftMgrActions.INTERNAL_DEBUG.value, "Internal State Viewer"),
        ])
        self._menu = mui.MenuList(
            menu_items,
            mui.IconButton(mui.IconType.MoreVert).prop(size="small", iconSize="small"))
        self._menu.prop(anchorOrigin=mui.Anchor("top", "right"))
        self._master_action_fn = master_action_fn
        self._menu.event_contextmenu_select.on(self._handle_master_actions)
        dialog_debug = mui.Dialog([
            mui.JsonViewer().bind_fields(data="getRoot()")
        ])
        self._dialog_debug = dialog_debug
        self._pyspy_viewer = PyspyViewer()
        self._fetch_pyspy_info_fn = fetch_pyspy_info_fn
        if is_raft_node:
            assert fetch_pyspy_info_fn is not None
            scan_buttons = mui.HBox([
                mui.Button("Scan Pth Local", partial(self._on_pyspy_scan, PyspyTraceMode.PYTORCH_LOCAL, True)).prop(variant="outlined"),
                mui.Button("Scan Pth Distributed", partial(self._on_pyspy_scan, PyspyTraceMode.PYTORCH_DISTRIBUTED, True)).prop(variant="outlined"),
                mui.Button("Scan Local", partial(self._on_pyspy_scan, PyspyTraceMode.ALL_SUBPROCESS, True)).prop(variant="outlined"),
                mui.Button("Scan Raft Local", partial(self._on_pyspy_scan, PyspyTraceMode.ALL_SUBPROCESS, False)).prop(variant="outlined"),
                mui.Button("Scan Aio Tasks", partial(self._on_pyspy_scan, PyspyTraceMode.LOCAL_AIO_TASKS, True)).prop(variant="outlined"),
            ])
        else:
            scan_buttons = mui.HBox([
                mui.Button("Scan Pth Local", partial(self._on_pyspy_scan, PyspyTraceMode.PYTORCH_LOCAL, True)).prop(variant="outlined"),
                mui.Button("Scan Local", partial(self._on_pyspy_scan, PyspyTraceMode.ALL_SUBPROCESS, True)).prop(variant="outlined"),
                mui.Button("Scan Aio Tasks", partial(self._on_pyspy_scan, PyspyTraceMode.LOCAL_AIO_TASKS, True)).prop(variant="outlined"),
            ])
        pyspy_dbg_dialog = mui.Dialog([
            scan_buttons,
            mui.Divider(orientation="horizontal"),
            self._pyspy_viewer.prop(flex=1)
        ])

        dialog_debug.prop(maxWidth="xl", fullWidth=True)
        pyspy_dbg_dialog.prop(dialogMaxWidth=False, fullWidth=False,
            width="75vw", height="75vh", includeFormControl=False,
            display="flex", flexDirection="column")
        self._pyspy_dbg_dialog = pyspy_dbg_dialog

        tab_defs: list[mui.TabDef] = []
        init_value = "terminal"
        if is_raft_node:
            init_value = "control"

            tab_defs.append(
                mui.TabDef("",
                       "control",
                        mui.VBox([
                            self._code_editor,
                        ]).prop(width="100%", height="100%", overflow="hidden"),
                       icon=mui.IconType.Settings,
                       tooltip="Control"),
            )
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
        # workers can't call distributed debug action.
        self.debug_panel = MasterDebugPanel(rpc_call_external=debug_panel_fn if is_raft_node else None)
        tab_defs.append(
            mui.TabDef("",
                    "debug",
                    self.debug_panel.prop(width="100%", height="100%", overflow="hidden"),
                    icon=mui.IconType.BugReport,
                    tooltip="Debug Panel"),
        )
        control_btn = mui.ToggleButton(icon=mui.IconType.Adb, callback=self._handle_toggle_btn)
        control_btn.prop(muiColor="success", size="small")
        control_btn.bind_draft_change(draft.is_user_control_enabled)
        control_btn_tooltip = mui.TooltipFlexBox("Toggle all pth_control_point in your running program.", [
            control_btn,
        ]).prop(enterDelay=400)

        before: list[mui.Component] = [header_str]
        if is_raft_node:
            before.append(worker_select.prop(flex=1))
            before.append(start_or_cancel_btn)
            before.append(kill_btn)
            before.append(mui.VDivider())
            before.append(control_btn_tooltip)
            before.append(self._menu,)
        else:
            header_str.prop(flex=1)
        before.append(mui.VDivider())
        self._tabs = mui.Tabs(tab_defs, init_value=init_value, before=before)
        self._tabs.prop(panelProps=mui.FlexBoxProps(
                                  height="100%", padding=0, minHeight=0),
                        orientation="horizontal",
                        borderBottom=1,
                        flex=1,
                        borderColor='divider',
                        # overflow="hidden",
                        tooltipPlacement="top",
                        display="flex",
                        alignItems="center")
        ssh_panel = mui.ThemeProvider([
            mui.VBox([
                self._tabs
            ]).prop(flex=1, overflow="hidden", minHeight=0)
        ], get_tight_icon_tab_theme_horizontal(size="40px"))

        # self.dm.event_storage_fetched.on(self._init_fields_when_fetch_model)
        master_draft = draft
        if is_raft_node:
            start_or_cancel_btn.bind_pfl_query(dm, 
                icon=(WorkerUISSHState.get_common_query, "start_or_cancel_icon"),
                disabled=(WorkerUISSHState.get_common_query, "not_leader_disabled"))
            kill_btn.bind_pfl_query(dm, disabled=(WorkerUISSHState.get_common_query, "stop_btn_disabled"))

            self._code_editor.bind_draft_change_uncontrolled(master_draft.user_cmd)
            dm.install_draft_change_handler(
                master_draft.cur_worker,
                self._handle_worker_change)
            worker_select.bind_draft_change(master_draft.cur_worker)
            worker_select.bind_fields(options=master_draft.workers)
            worker_select.bind_pfl_query(dm, disabled=(WorkerUISSHState.get_common_query, "not_leader_disabled"))
            worker_select.bind_pfl_query(dm, label=(WorkerUISSHState.get_common_query, "worker_select_label"))

        super().__init__([
            dialog_debug,
            pyspy_dbg_dialog,
            ssh_panel,
        ])
        self.prop(flexDirection="column", flex=1, minHeight=0, overflow="hidden")

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

    async def _handle_master_actions(self, act_id: str):
        act = RaftMgrActions(act_id)
        if act == RaftMgrActions.INTERNAL_DEBUG:
            await self._dialog_debug.set_open(True)
        elif act == RaftMgrActions.PYTORCH_SPY:
            await self._pyspy_dbg_dialog.set_open(True)
        else:
            coro = self._master_action_fn(act)
            if inspect.iscoroutine(coro):
                await coro

    async def _fetch_pyspy_local(self, mode: PyspyTraceMode):
        if mode == PyspyTraceMode.SERVER_PROCESS or mode == PyspyTraceMode.LOCAL_AIO_TASKS:
            pid = None 
        else:
            assert self._terminal is not None 
            state = self._terminal.get_current_state()
            assert state is not None 
            pid = state.pid
        try:
            return await fetch_pyspy_info(mode, parent_pid=pid)
        except:
            CM_LOGGER.exception("get torchrun traceback failed", exc_info=True)
            return {}

    async def _on_pyspy_scan(self, mode: PyspyTraceMode, is_compute: bool):
        if self._fetch_pyspy_info_fn is not None:
            data = await self._fetch_pyspy_info_fn(mode, is_compute)
        else:
            data = await self._fetch_pyspy_local(mode)
        if data is not None:
            data_with_str_id = {}
            for (rank, pid), v in data.items():
                # only check mainthread
                if v:
                    data_with_str_id[f"{rank}-{pid}"] = v[0]
                else:
                    data_with_str_id[f"{rank}-{pid}"] = {
                        "pid": pid,
                        "thread_id": 0,
                        "thread_name": "Unknown",
                        "frames": [],
                    }
            await self._pyspy_viewer.set_pyspy_raw_data(data_with_str_id)

    async def _handle_toggle_btn(self, enable: bool):
        if not enable:
            await self.debug_panel.release_all_breakpoints()
