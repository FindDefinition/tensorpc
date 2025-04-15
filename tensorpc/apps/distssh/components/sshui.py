from typing import Any, Awaitable, Callable, Optional, Union

from tensorpc.core.datamodel.events import DraftChangeEvent
from tensorpc.dock import terminal
import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.dock.components import mui
import tensorpc.core.datamodel as D
import psutil
from tensorpc.apps.distssh.constants import (TENSORPC_DISTSSH_CLIENT_DEBUG_UI_KEY, TENSORPC_DISTSSH_UI_KEY,
                                             TENSORPC_ENV_DISTSSH_URL_WITH_PORT
                                             )
from ..typedefs import FTState, CmdStatus, MasterUIState, FTStatusBoxState
from tensorpc.apps.dbg.components.dbgpanel_v2 import MasterDebugPanel


class WorkersStatusBox(mui.DataFlexBox):

    def __init__(self,
                 init_data_list: list[FTStatusBoxState],
                 on_click: Callable[[mui.Event], mui.CORO_NONE],
                 box_size: int = 10):
        self._box_template = mui.HBox([])
        box_size_px = f"{box_size}px"
        self._box_template.prop(width=box_size_px,
                                height=box_size_px,
                                margin="2px")
        self._box_template.bind_fields(
            backgroundColor="color",
            border=
            f"where(selected, '2px solid lightpink', '2px solid transparent')",
        )
        self._box_template.event_click.on_standard(on_click)
        self._selected_idx = -1
        super().__init__(self._box_template, init_data_list)
        self.prop(flexFlow="row wrap", padding="10px")


class FaultToleranceUIMaster(mui.FlexBox):

    def __init__(self, master_rank: int, ui_state: MasterUIState,
                 term: terminal.AsyncSSHTerminal, debug_panel: MasterDebugPanel, port: int,
                 start_or_cancel_fn: Callable[[], mui.CORO_ANY],
                 stop_fn: Callable[[], mui.CORO_ANY],
                 kill_fn: Callable[[], mui.CORO_ANY],
                 release_bkpt_fn: Callable[[], Awaitable[None]]):
        master_state = ui_state.client_states[master_rank]
        states = ui_state.client_states
        self._master_rank = master_rank
        self._port = port
        if master_state.is_master:
            title = "Main Worker"
        else:
            title = f"Worker ({master_state.rank})"
        self._release_bkpt_fn = release_bkpt_fn

        start_or_cancel_btn = mui.IconButton(
            mui.IconType.PlayArrow, start_or_cancel_fn).prop(iconSize="small",
                                                             size="small")
        stop_btn = mui.IconButton(mui.IconType.Stop, stop_fn).prop(
            iconSize="small",
            size="small",
            muiColor="error",
            confirmTitle="Dangerous Operation",
            confirmMessage=
            "Are you sure to shutdown (ctrl-c->terminate->kill) ALL running process?",
            tooltip="shutdown command")
        kill_btn = mui.IconButton(mui.IconType.Delete, kill_fn).prop(
            iconSize="small",
            size="small",
            confirmTitle="Dangerous Operation",
            confirmMessage="Are you sure to kill ALL running process?",
            tooltip="kill all child process")
        enable_control_btn = mui.TooltipFlexBox("Toggle all pth_control_point in your running program.", [
            mui.ToggleButton(icon=mui.IconType.Adb, callback=self._handle_toggle_btn).prop(muiColor="success", size="small", )
        ]).prop(enterDelay=400)
        header_str = mui.Typography(title).prop(variant="body2",
                                                color="primary")
        rank_select = mui.Autocomplete("Workers", []).prop(muiMargin="dense",
                                                           size="small")
        self.worker_status_box = WorkersStatusBox(
            [FTStatusBoxState.from_ft_state(state, False) for state in states],
            self._on_status_box_click)
        header = mui.HBox([
            mui.HBox([
                header_str,
            ]).prop(flex=1),
            start_or_cancel_btn,
            stop_btn,
            kill_btn,
        ])
        # self._remote_box = mui.HBox([])
        # self._code_editor = mui.SimpleCodeEditor("echo $HOME", "bash").prop(debounce=300, height="300px", border="1px solid gray")
        self._code_editor = mui.MonacoEditor("echo $HOME", "shell",
                                             "root").prop(debounce=300,
                                                          height="300px")
        self._terminal_box = mui.VBox([
            term,
        ]).prop(flex=1, overflow="auto")
        self._remote_terminal_box = mui.HBox([]).prop(flex=1, overflow="auto")
        self._terminal_panel = mui.MatchCase.binary_selection(
            True, self._terminal_box, self._remote_terminal_box)
        ssh_panel = mui.VBox([
            header,
            mui.HBox([
                rank_select.prop(flex=1),
                enable_control_btn,
            ]),
            self.worker_status_box,
            self._code_editor,
            self._terminal_panel,
        ]).prop(width="100%", height="100%", overflow="hidden")
        self._master_panel = debug_panel
        child_control_panel = mui.VBox([
            self._master_panel.prop(flex=1)
        ]).prop(width="100%", height="100%", overflow="hidden")
        self._child_control_panel = child_control_panel
        global_panel = mui.Allotment(mui.Allotment.ChildDef([
            mui.Allotment.Pane(ssh_panel),
            mui.Allotment.Pane(child_control_panel),
        ])).prop(vertical=False, defaultSizes=[150, 300])
        self.dm = mui.DataModel(ui_state, [
            global_panel
        ])
        self.dm.event_storage_fetched.on(self._init_fields_when_fetch_model)
        master_draft = self.dm.get_draft()
        self._terminal_panel.bind_fields(condition=D.logical_or(
            master_draft.selected_client_state == None,
            D.cast_any_draft(D.dict_get_item(master_draft.selected_client_state, "rank"), int)
            == master_rank))
        start_or_cancel_btn.bind_fields(
            icon=D.where(master_draft.cmd_status == CmdStatus.IDLE,
                         mui.IconType.PlayArrow, mui.IconType.Stop),
            disabled=D.where(
                master_draft.cmd_status == CmdStatus.DURING_RESTART, True,
                False))
        stop_btn.bind_fields(disabled=D.where(
            master_draft.cmd_status == CmdStatus.IDLE, True, False))
        kill_btn.bind_fields(disabled=D.where(
            master_draft.cmd_status == CmdStatus.IDLE, True, False))
        self._code_editor.bind_draft_change_uncontrolled(master_draft.cmd)
        # self._code_editor.bind_draft_change_uncontrolled(master_draft.cmd)
        # FIXME can't install to worker_status_box
        self.dm.install_draft_change_handler(master_draft.client_states,
                                             self._handle_client_state_change,
                                             handle_child_change=True)
        self.dm.install_draft_change_handler(
            master_draft.selected_client_state,
            self._handle_selected_box_change)

        rank_select.bind_draft_change(master_draft.selected_client_state)
        rank_select.bind_fields(options=master_draft.client_states)

        super().__init__([
            self.dm,
        ])
        self.prop(flexDirection="column", flex=1)

    async def _handle_toggle_btn(self, enable: bool):
        if enable:
            # only master rank control point check this value, so no need to sent to all
            # client. 
            self.dm.get_draft().client_states[self._master_rank].is_user_control_enabled = True
        else:
            self.dm.get_draft().client_states[self._master_rank].is_user_control_enabled = False
            await self._release_bkpt_fn()

    def _init_fields_when_fetch_model(self, prev_model: MasterUIState):
        # client_states are runtime state, don't use stored value.
        # TODO exclude client_states in draft store
        self.dm.model.client_states = prev_model.client_states
        self.dm.model.selected_client_state = None

    async def _on_status_box_click(self, ev: mui.Event):
        rank = ev.get_indexes_checked()[0]
        self.dm.get_draft().selected_client_state = dataclasses.asdict(
            self.dm.model.client_states[rank])

    async def _handle_client_state_change(self, ev: DraftChangeEvent):
        if ev.new_value is not None:
            states: list[FTState] = ev.new_value
            selected_idx = -1
            selected_state = self.dm.model.selected_client_state
            if selected_state is not None:
                selected_idx = selected_state["rank"]
            ui_states = [
                FTStatusBoxState.from_ft_state(state, i == selected_idx)
                for i, state in enumerate(states)
            ]
            await self.send_and_wait(
                self.worker_status_box.update_event(dataList=ui_states))
        else:
            await self.send_and_wait(
                self.worker_status_box.update_event(dataList=[]))

    async def _handle_selected_box_change(self, ev: DraftChangeEvent):
        selected_state_dict = ev.new_value
        if selected_state_dict is not None:
            rank = selected_state_dict["rank"]
            ip = selected_state_dict["ip"]
            if ev.old_value is not None:
                if ev.old_value["rank"] == rank:
                    return
            if rank == self._master_rank:
                await self._remote_terminal_box.set_new_layout([])
                print("WTFWTF")
                await self._child_control_panel.set_new_layout([self._master_panel])
            else:
                await self._remote_terminal_box.set_new_layout([
                    mui.RemoteBoxGrpc(ip, self._port,
                                      TENSORPC_DISTSSH_UI_KEY).prop(flex=1)
                ])
                await self._child_control_panel.set_new_layout([
                    mui.RemoteBoxGrpc(ip, self._port,
                                      TENSORPC_DISTSSH_CLIENT_DEBUG_UI_KEY).prop(flex=1)
                ])
            async with self.worker_status_box.draft_update(
                    FTStatusBoxState) as dctx:
                with dctx.group(rank):
                    dctx.draft.selected = True
                with dctx.group(None):
                    dctx.draft.selected = False
        else:
            await self._remote_terminal_box.set_new_layout([])
            await self._child_control_panel.set_new_layout([self._master_panel])


class FaultToleranceUIClient(mui.FlexBox):

    def __init__(self, state: FTState, term: terminal.AsyncSSHTerminal):
        title = f"Worker ({state.rank})"
        header_str = mui.Typography(title).prop(variant="body2",
                                                color="primary")
        self._terminal_box = mui.VBox([
            term,
        ]).prop(flex=1, overflow="auto")
        self.dm = mui.DataModel(state, [header_str, self._terminal_box])
        super().__init__([
            self.dm,
        ])
        self.prop(flexDirection="column", flex=1, border="1px solid gray")
