import enum
from functools import partial
import inspect
from typing import Any, Awaitable, Callable, Optional, Union

from tensorpc.apps.collections.serv.kvstore import KVStoreChangeEvent, KVStoreEventType, KVStoreItem
from tensorpc.apps.dbg.components.distpyspy import PyspyViewer
from tensorpc.core import BuiltinServiceKeys, prim
from tensorpc.core.datamodel.events import DraftChangeEvent
from tensorpc.core.tree_id import UniqueTreeId
from tensorpc.dock import terminal
import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.dock.components import mui
import tensorpc.core.datamodel as D
import psutil
from tensorpc.apps.distssh.constants import (TENSORPC_DISTSSH_CLIENT_DEBUG_UI_KEY, TENSORPC_DISTSSH_UI_KEY,
                                             TENSORPC_ENV_DISTSSH_URL_WITH_PORT,
                                             )
from tensorpc.dock.components.plus.styles import get_tight_icon_tab_theme_horizontal, get_tight_tab_theme_horizontal
from ..typedefs import CheckpointActions, CheckpointMetadata, FTState, CmdStatus, MasterUIState, FTStatusBoxState, PyspyTraceMode, UILocalActions
from tensorpc.apps.dbg.components.dbgpanel import MasterDebugPanel
from tensorpc.apps.distssh.typedefs import CheckpointType, MasterActions
import humanize

class WorkersStatusBox(mui.DataFlexBox):

    def __init__(self,
                 init_data_list: list[FTStatusBoxState],
                 on_click: Callable[[mui.Event], mui.CORO_NONE],
                 box_size: int = 10):
        self._box_template = mui.HBox([])
        box_size_px = f"{box_size}px"
        self._box_template.prop(width=box_size_px,
                                height=box_size_px,
                                outlineWidth="1px",
                                outlineOffset=0,
                                outlineColor="rgba(0, 130, 206, 0)",
                                outlineStyle="solid",
                                margin="2px")
        self._box_template.bind_fields(
            backgroundColor="color",
            animation="where(num_bkpt_proc > 0, 'animateOutline 3s ease infinite', 'none')",
            # border=
            # f"where(num_bkpt_proc > `0`, '2px solid red', where(selected, '2px solid lightpink', '2px solid transparent'))",
            border=
            f"where(selected, '2px solid lightpink', '2px solid transparent')",

        )
        self._box_template.event_click.on_standard(on_click)
        self._selected_idx = -1
        super().__init__(self._box_template, init_data_list)
        self.prop(flexFlow="row wrap", padding="10px")
        self.update_raw_props({
            "@keyframes animateOutline": {
                "0%": {
                    "outlineWidth": "1px",
                    "outlineOffset": 0,
                    "outlineColor": "rgba(0, 130, 206, 0)",
                },
                "10%": {
                    "outlineColor": "rgba(0, 130, 206, 0.75)",
                },

                "50%": {
                    "outlineWidth": "4px",
                    "outlineOffset": "2px",
                    "outlineColor": "rgba(0, 130, 206, 0)",
                },
                "100%": {
                    "outlineWidth": "4px",
                    "outlineOffset": "2px",
                    "outlineColor": "rgba(102, 102, 102, 0)",
                },
            }
        })


@dataclasses.dataclass
class CheckpointItem:
    id: str
    name: str
    type: str
    step: int
    size: Union[int, str]
    local_rank: int = 0

class CheckpointManager(mui.FlexBox):
    def __init__(self, master_dm: mui.DataModel[MasterUIState],
            release_bkpt_fn: Callable[[Any], Awaitable[None]]):

        event_emitter_kvstore = prim.get_service(f"{BuiltinServiceKeys.ShmTrOnlyKVStore.value}.backend_get_event_emitter")()
        btn = mui.Button("Load").prop(loading=False, size="small")
        btn.event_click.on_standard(self._on_ckpt_load).configure(True)
        save_btn = mui.Button("Save", self._on_ckpt_save).prop(size="small")
        self._release_bkpt_fn = release_bkpt_fn
        column_defs = [
            # mui.DataGrid.ColumnDef("id", accessorKey="id"),
            mui.DataGrid.ColumnDef("name", accessorKey="name", width=120),
            mui.DataGrid.ColumnDef("type", accessorKey="type"),
            mui.DataGrid.ColumnDef("step", accessorKey="step"),
            mui.DataGrid.ColumnDef("size", accessorKey="size"),
            mui.DataGrid.ColumnDef("actions", cell=mui.MatchCase.binary_selection(CheckpointType.FIXED.name, 
                mui.Typography("fixed"), mui.TooltipFlexBox("Send load ckpt msg to process inside breakpoint.", [
                    btn
                ])).bind_fields(condition="type")),
        ]
        draft = master_dm.get_draft()
        btn.bind_fields(disabled=(master_dm, f"({draft.client_states[0].num_bkpt_proc} == 0)"),)
        save_btn.bind_fields(disabled=(master_dm, f"({draft.client_states[0].num_bkpt_proc} == 0)"),)
        footer = mui.HBox([
            save_btn
        ]).prop(padding="10px")
        dgrid = mui.DataGrid(column_defs, [], customPaginationFooters=[footer]).prop(idKey="id", rowHover=True)
        self.dgrid = dgrid
        super().__init__([
            dgrid.prop(stickyHeader=True, virtualized=False, size="small", enableGlobalFilter=False,
                    display="flex", flexDirection="column", fullWidth=True,
                    tableContainerProps=mui.FlexBoxProps(overflow="auto", flex=1)),
        ])
        event_emitter_kvstore.on(KVStoreEventType.ITEM_CHANGE, self._on_kvstore_item_change)
        self.prop(flex=1, overflow="hidden")
        self.event_after_mount.on(self._init)

    async def _init(self):
        store = prim.get_service(f"{BuiltinServiceKeys.ShmTrOnlyKVStore.value}.backend_get_store")()
        await self._on_kvstore_item_change(store)


    async def _on_kvstore_item_change(self, store: dict[str, KVStoreItem]):
        item_getsize = prim.get_service(f"{BuiltinServiceKeys.ShmTrOnlyKVStore.value}.get_item_shm_size")
        all_stores = store 
        res_items: list[CheckpointItem] = []
        key_step_to_res_items: dict[tuple[str, int], CheckpointItem] = {}
        for k, v in all_stores.items():
            metadata = v.metadata
            item_size = item_getsize(k)
            if isinstance(metadata, CheckpointMetadata):
                if metadata.step is None:
                    step = -1
                else:
                    step = metadata.step
                key = metadata.key
                uid = UniqueTreeId.from_parts([key, str(step)]).uid_encoded
                item = CheckpointItem(uid, metadata.key, metadata.type.name, step, item_size)
            else:
                step = -1
                uid = UniqueTreeId.from_parts([k, str(step)]).uid_encoded
                item = CheckpointItem(uid, k, CheckpointType.FIXED.name, -1, item_size)
                key = k
            key_tuple = (key, step)
            if key_tuple not in key_step_to_res_items:
                key_step_to_res_items[key_tuple] = item
            else:
                prev_item = key_step_to_res_items[key_tuple]
                prev_item.size += item_size
        for v in key_step_to_res_items.values():
            if isinstance(v.size, int):
                v.size = humanize.naturalsize(v.size)
            res_items.append(v)
        res_items.sort(key=lambda x: -x.step)
        await self.dgrid.send_and_wait(self.dgrid.update_event(dataList=res_items))

    async def _on_ckpt_load(self, ev: mui.Event):
        key_encoded = ev.get_keys_checked()[0]
        dlist = self.dgrid.props.dataList
        assert isinstance(dlist, list)
        for data in dlist:
            assert isinstance(data, CheckpointItem)
            data_dict = dataclasses.asdict(data)
            msg = {
                "type": CheckpointActions.LOAD_ITEM.value,
                "data": data_dict,
            }
            if data.id == key_encoded:
                return await self._release_bkpt_fn(msg)

    async def _on_ckpt_save(self):
        msg = {
            "type": CheckpointActions.SAVE.value,
            "data": None,
        }
        return await self._release_bkpt_fn(msg)

def _get_terminal_menus(term: terminal.AsyncSSHTerminal):
    
    return mui.MenuList([
        mui.MenuItem("Clear", "Clear", iconSize="small", iconFontSize="small")
    ], term).prop(
            flex=1, overflow="auto", display="flex", flexFlow="column nowrap", triggerMethod="contextmenu",
            anchorReference="anchorPosition", dense=True,
            paperProps=mui.PaperProps(width="20ch"))

class FaultToleranceUIMaster(mui.FlexBox):

    def __init__(self, master_rank: int, ui_state: MasterUIState,
                 term: terminal.AsyncSSHTerminal, debug_panel: MasterDebugPanel, port: int,
                 master_action_fn: Callable[[MasterActions], mui.CORO_ANY],
                 release_bkpt_fn: Callable[[Any], Awaitable[None]],
                 fetch_debug_info_fn: Callable[[PyspyTraceMode], Awaitable[Optional[dict[tuple[int, int], Any]]]],
                 enabled: bool = True, default_path: str = "default"):
        master_state = ui_state.client_states[master_rank]
        states = ui_state.client_states
        self._master_rank = master_rank
        self._port = port
        if master_state.is_master:
            title = "Main Worker (%s)"
        else:
            title = f"Worker ({master_state.rank}, %s)"
        self._release_bkpt_fn = release_bkpt_fn
        self._fetch_debug_info_fn = fetch_debug_info_fn
        start_or_cancel_btn = mui.IconButton(
            mui.IconType.PlayArrow, partial(master_action_fn, MasterActions.START_OR_CANCEL)).prop(iconSize="small",
                                                             size="small")
        stop_btn = mui.IconButton(mui.IconType.Stop, partial(master_action_fn, MasterActions.SHUTDOWN_ALL)).prop(
            iconSize="small",
            size="small",
            muiColor="error",
            confirmTitle="Dangerous Operation",
            confirmMessage=
            "Are you sure to shutdown (ctrl-c->terminate->kill) ALL running process?",
            tooltip="shutdown command")
        kill_btn = mui.IconButton(mui.IconType.Delete, partial(master_action_fn, MasterActions.KILL_ALL)).prop(
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
        rank_select = mui.Autocomplete("Workers", []).prop(textFieldProps=mui.TextFieldProps(muiMargin="none"),
                                                           size="small")
        self.worker_status_box = WorkersStatusBox(
            [FTStatusBoxState.from_ft_state(state, False) for state in states],
            self._on_status_box_click)
        dialog_debug = mui.Dialog([
            mui.JsonViewer().bind_fields(data="getRoot()")
        ])
        self._dialog_debug = dialog_debug
        self._pyspy_viewer = PyspyViewer()
        pyspy_dbg_dialog = mui.Dialog([
            mui.HBox([
                mui.Button("Scan Pth Local", partial(self._on_pyspy_scan, PyspyTraceMode.PYTORCH_LOCAL)).prop(variant="outlined"),
                mui.Button("Scan Pth Distributed", partial(self._on_pyspy_scan, PyspyTraceMode.PYTORCH_DISTRIBUTED)).prop(variant="outlined"),
                mui.Button("Scan Local", partial(self._on_pyspy_scan, PyspyTraceMode.ALL_SUBPROCESS)).prop(variant="outlined"),
                mui.Button("Scan Aio Tasks", partial(self._on_pyspy_scan, PyspyTraceMode.LOCAL_AIO_TASKS)).prop(variant="outlined"),
            ]),
            mui.Divider(orientation="horizontal"),
            self._pyspy_viewer.prop(flex=1)
        ])
        dialog_debug.prop(maxWidth="xl", fullWidth=True)
        pyspy_dbg_dialog.prop(dialogMaxWidth=False, fullWidth=False,
            width="75vw", height="75vh", includeFormControl=False,
            display="flex", flexDirection="column")
        self._pyspy_dbg_dialog = pyspy_dbg_dialog
        menu_items: list[mui.MenuItem] = []
        cared_menu_acts = [MasterActions.RECONNECT_ALL_CLIENT, MasterActions.CLEAR_ALL_CKPT, MasterActions.CLEAR_ALL_TERMINALS]
        for action in MasterActions:
            if action in cared_menu_acts:
                menu_items.append(mui.MenuItem(action.value, action.value, 
                    confirmMessage="Are You Sure?", 
                    confirmTitle=f"Dangerous Operation ({action.value})"))
        menu_items.extend([
            mui.MenuItem("divider1", divider=True),
            mui.MenuItem(UILocalActions.PYTORCH_SPY.value, "PyTorch Dist Spy"),
            mui.MenuItem("divider2", divider=True),
            mui.MenuItem(UILocalActions.INTERNAL_DEBUG.value, "Internal State Viewer"),
        ])
        self._menu = mui.MenuList(
            menu_items,
            mui.IconButton(mui.IconType.MoreVert).prop(size="small", iconSize="small"))
        self._menu.prop(anchorOrigin=mui.Anchor("top", "right"))
        self._master_action_fn = master_action_fn
        self._menu.event_contextmenu_select.on(self._handle_master_actions)

        header = mui.HBox([
            mui.HBox([
                header_str,
            ]).prop(flex=1),
            start_or_cancel_btn,
            stop_btn,
            kill_btn,
            self._menu,
        ])
        # self._remote_box = mui.HBox([])
        # self._code_editor = mui.SimpleCodeEditor("echo $HOME", "bash").prop(debounce=300, height="300px", border="1px solid gray")
        self._code_editor = mui.MonacoEditor("echo $HOME", "shell",
                                             default_path).prop(debounce=300,
                                                          height="300px")
        self._terminal = term
        self._terminal_box = _get_terminal_menus(term)
        self._terminal_box.event_contextmenu_select.on(self._on_term_menu)
        self._remote_terminal_box = mui.VBox([]).prop(flex=1, overflow="hidden")
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
        if enabled:
            self._master_panel.event_breakpoint_process_change.on(self._on_has_bkpt_change)

        child_control_panel = mui.VBox([

        ]).prop(width="100%", height="100%", overflow="hidden")
        self._child_control_panel = child_control_panel
        global_panel = mui.Allotment(mui.Allotment.ChildDef([
            mui.Allotment.Pane(ssh_panel),
            mui.Allotment.Pane(child_control_panel),
        ])).prop(vertical=False, defaultSizes=[150, 300])
        self.dm = mui.DataModel(ui_state, [
            global_panel,
            dialog_debug,
            pyspy_dbg_dialog,
        ])
        if enabled:
            # FIXME: better code
            ckpt_mgr = CheckpointManager(self.dm, release_bkpt_fn)
        else:
            ckpt_mgr = mui.HBox([])
        debug_view_tab_defs = [
            mui.TabDef("Checkpoint",
                    "Checkpoint",
                    mui.VBox([ckpt_mgr]).prop(width="100%",
                                                        height="100%",
                                                        overflow="hidden"),
                    tooltip="Checkpoint Manager"),
        ]
        self._debug_view_tabs = mui.Tabs(debug_view_tab_defs, init_value="Checkpoint").prop(
            panelProps=mui.FlexBoxProps(flex=1, padding=0, overflow="hidden"),
            borderBottom=1,
            borderColor='divider',
            # flex=1,
            display="flex",
            flexFlow="row nowrap",
            tooltipPlacement="bottom",
        )
        self._master_all_panel = mui.Fragment([
            mui.ThemeProvider([
                mui.VBox([
                    self._debug_view_tabs
                ]).prop(flex=1, overflow="hidden")
            ], get_tight_tab_theme_horizontal(tab_padding="10px")),
            mui.Divider("horizontal"),
            self._master_panel.prop(flex=2),

        ])
        child_control_panel.init_add_layout([
            self._master_all_panel,
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
        header_str.bind_fields(value=D.literal_val(title) % master_draft.client_states[master_rank].title_msg)

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

    async def _on_pyspy_scan(self, mode: PyspyTraceMode):
        data = await self._fetch_debug_info_fn(mode)
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

    async def _on_has_bkpt_change(self, num_bkpt_proc):
        prev = self.dm.model.client_states[self._master_rank].num_bkpt_proc
        if prev != num_bkpt_proc:
            async with self.dm.draft_update() as draft:
                draft.client_states[self._master_rank].num_bkpt_proc = num_bkpt_proc

    async def _on_term_menu(self, item_id: str):
        if item_id == "Clear":
            await self._terminal.clear()
        else:
            raise ValueError(f"Unknown menu item {item_id}")

    async def _handle_toggle_btn(self, enable: bool):
        if enable:
            # only master rank control point check this value, so no need to sent to all
            # client. 
            self.dm.get_draft().client_states[self._master_rank].is_user_control_enabled = True
        else:
            self.dm.get_draft().client_states[self._master_rank].is_user_control_enabled = False
            await self._release_bkpt_fn(None)

    def _init_fields_when_fetch_model(self, prev_model: MasterUIState):
        # client_states are runtime state, don't use stored value.
        # TODO exclude client_states in draft store
        self.dm.model.client_states = prev_model.client_states
        self.dm.model.selected_client_state = None
        self.dm.model.cmd_status = CmdStatus.IDLE

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
                await self._child_control_panel.set_new_layout([
                    self._master_all_panel,
                ])
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
            await self._child_control_panel.set_new_layout([
                self._master_all_panel,
            ])

    async def _handle_master_actions(self, act_id: str):
        if act_id.startswith("_local"):
            act_local = UILocalActions(act_id)
            if act_local == UILocalActions.INTERNAL_DEBUG:
                await self._dialog_debug.set_open(True)
            elif act_local == UILocalActions.PYTORCH_SPY:
                await self._pyspy_dbg_dialog.set_open(True)
            return 
        act = MasterActions(act_id)
        coro = self._master_action_fn(act)
        if inspect.iscoroutine(coro):
            await coro

class FaultToleranceUIClient(mui.FlexBox):

    def __init__(self, state: FTState, term: terminal.AsyncSSHTerminal):
        title = f"Worker ({state.rank}, %s)"
        header_str = mui.Typography(title).prop(variant="body2",
                                                color="primary")
        self._terminal = term
        self._terminal_box = _get_terminal_menus(term)
        self._terminal_box.event_contextmenu_select.on(self._on_term_menu)

        self.dm = mui.DataModel(state, [header_str, self._terminal_box])
        draft = self.dm.get_draft()
        header_str.bind_fields(value=D.literal_val(title) % draft.title_msg)

        super().__init__([
            self.dm,
        ])
        self.prop(flexDirection="column", flex=1, overflow="hidden", border="1px solid gray")

    async def _on_term_menu(self, item_id: str):
        if item_id == "Clear":
            await self._terminal.clear()
        else:
            raise ValueError(f"Unknown menu item {item_id}")
