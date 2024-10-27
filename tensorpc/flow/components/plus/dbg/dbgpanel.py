import asyncio
import dataclasses
import enum
import gzip
import json
import time
import traceback
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union
import uuid

import grpc
from regex import P
import rich
from tensorpc.constants import TENSORPC_BG_PROCESS_NAME_PREFIX
from tensorpc.core.asyncclient import simple_chunk_call_async, simple_remote_call_async
from tensorpc.core.client import simple_remote_call
from tensorpc.dbg.constants import TENSORPC_DBG_FRAME_INSPECTOR_KEY, TENSORPC_DBG_SPLIT, DebugFrameMeta, TracerConfig
from tensorpc.flow import marker, appctx
from tensorpc.flow.components import chart, mui
from tensorpc.flow.components.plus.styles import CodeStyles, get_tight_icon_tab_theme
from tensorpc.dbg.serv_names import serv_names as dbg_serv_names
from tensorpc.compat import InWindows
import psutil

from tensorpc.flow.core.appcore import AppSpecialEventType
from tensorpc.flow.vscode.coretypes import VscodeBreakpoint, VscodeTensorpcMessage, VscodeTensorpcMessageType


@dataclasses.dataclass
class DebugServerProcessMeta:
    id: str
    name: str
    pid: int
    uid: str
    server_id: str
    port: int
    secondary_name: str = "running"
    @property 
    def url_with_port(self):
        return f"localhost:{self.port}"


def list_all_dbg_server_in_machine():
    res: List[DebugServerProcessMeta] = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        proc_name = proc.info["name"]
        proc_cmdline = proc.info["cmdline"]
        if proc_name.startswith(TENSORPC_BG_PROCESS_NAME_PREFIX):
            parts = proc_name.split(TENSORPC_DBG_SPLIT)[1:]
            meta = DebugServerProcessMeta(str(proc.info["pid"]), proc_name,
                                          proc.info["pid"], parts[-1],
                                          parts[0], int(parts[1]))
            res.append(meta)
            continue 
        if proc_cmdline and proc_cmdline[0].startswith(TENSORPC_BG_PROCESS_NAME_PREFIX):
            # some platform need cmdline
            parts = proc_cmdline[0].split(TENSORPC_DBG_SPLIT)[1:]
            meta = DebugServerProcessMeta(str(proc.info["pid"]), proc_cmdline[0],
                                          proc.info["pid"], parts[-1],
                                          parts[0], int(parts[1]))
            res.append(meta)
    return res


class ServerItemActions(enum.Enum):
    RELEASE_BREAKPOINT = "release_breakpoint"
    SKIP_BREAKPOINT = "skip_breakpoint"
    ENABLE_BREAKPOINT = "enable_breakpoint"
    UNMOUNT_REMOTE_SERVER = "unmount_remote_server"
    RECORD = "record"


class MasterDebugPanel(mui.FlexBox):

    def __init__(self):
        assert not InWindows, "MasterDebugPanel is not supported in Windows due to setproctitle."
        name = mui.ListItemText("").prop(
            primaryTypographyProps=mui.TypographyProps(
                variant="body1", fontFamily=CodeStyles.fontFamily,
                overflow="hidden",
                whiteSpace="nowrap", textOverflow="ellipsis"),
            secondaryTypographyProps=mui.TypographyProps(
                variant="caption", fontFamily=CodeStyles.fontFamily,
                overflow="hidden",
                whiteSpace="nowrap", textOverflow="ellipsis"))
        name.set_override_props(value="server_id", secondary="secondary_name")
        remote_server_item = mui.ListItemButton([
            name,
        ])
        self._remote_server_discover_lst = mui.DataFlexBox(
            remote_server_item, [])
        filter_input = mui.TextField("filter").prop(
            valueChangeTarget=(self._remote_server_discover_lst, "filter"))
        self._remote_server_discover_lst.prop(
            filterKey="server_id",
            variant="list",
            dense=False,
            disablePadding=True,
            overflow="auto",
            virtualized=False,
            secondaryIconButtonProps=[
            ])
        remote_server_item.event_click.on_standard(
            self._on_server_item_click).configure(True)
        self._menu = mui.MenuList([
            mui.MenuItem(id=ServerItemActions.RELEASE_BREAKPOINT.value, label="Release Breakpoint"),
            mui.MenuItem(id=ServerItemActions.SKIP_BREAKPOINT.value, label="Disable All Breakpoints"),
            mui.MenuItem(id=ServerItemActions.ENABLE_BREAKPOINT.value, label="Enable All Breakpoints"),
            mui.MenuItem(id=ServerItemActions.UNMOUNT_REMOTE_SERVER.value, label="Unmount Remote Panel"),
            mui.MenuItem(id=ServerItemActions.RECORD.value, label="Release And Start Record"),
        ], mui.IconButton(mui.IconType.MoreVert))
        self._menu.prop(anchorOrigin=mui.Anchor("top", "right"))
        self._menu.event_contextmenu_select.on(self._handle_secondary_actions)
        self._drawer = mui.Collapse([
            mui.VBox([
                mui.HBox([
                    mui.HBox([
                        mui.Typography("Debug Servers").prop(variant="body1")
                    ]).prop(flex=1),
                    mui.IconButton(mui.IconType.ChevronLeft,
                                   self._close_drawer).prop(
                                       size="small",
                                       iconFontSize="18px",
                                       alignSelf="flex-end")
                ]).prop(alignItems="center"),
                mui.HBox([
                    filter_input.prop(flex=1),
                    self._menu,
                ]).prop(alignItems="center"),
                mui.Divider(),
                self._remote_server_discover_lst.prop(flex="1 1 1", minHeight=0),
            ]).prop(width="240px", alignItems="stretch", overflow="hidden", height="100%")
        ]).prop(triggered=True, orientation="horizontal", overflow="hidden")
        self._remote_comp_container = mui.VBox([]).prop(width="100%", height="100%", overflow="hidden")
        
        self._perfetto_select = mui.Select("trace", []).prop(size="small", muiMargin="dense")
        self._dist_perfetto = chart.Perfetto().prop(width="100%", height="100%")
        
        self._dist_perfetto_container = mui.VBox([
            mui.HBox([
                self._perfetto_select.prop(flex=1),
                mui.IconButton(mui.IconType.Refresh, self._on_dist_perfetto_reflesh).prop(size="small")
            ]).prop(alignItems="center"),
            mui.Divider(),
            self._dist_perfetto,
        ]).prop(width="100%", height="100%", overflow="hidden")
        tab_defs = [
            mui.TabDef("",
                       "remote",
                       self._remote_comp_container,
                       icon=mui.IconType.Terminal,
                       tooltip="Remote Viewer"),
            mui.TabDef("",
                       "perfetto",
                       self._dist_perfetto_container,
                       icon=mui.IconType.Timeline,
                       tooltip="Distributed Perfetto Viewer"),
        ]
        before = [
            mui.IconButton(mui.IconType.Menu,
                            self._open_drawer).prop(size="small",
                                                    iconFontSize="18px"),
            mui.Divider(),

        ]
        self._tabs = mui.Tabs(tab_defs, init_value="remote", before=before).prop(
                    panelProps=mui.FlexBoxProps(width="100%", padding=0),
                    orientation="vertical",
                    borderRight=1,
                    flex=1,
                    borderColor='divider',
                    tooltipPlacement="right")
        self._tabs.event_change.on(self._on_tab_change)
        super().__init__([
            self._drawer,
            mui.Divider(orientation="vertical"),
            # mui.VBox([
            #     mui.IconButton(mui.IconType.Menu).prop(size="small",
            #                                            iconFontSize="18px"),
            # ]),
            mui.Divider(orientation="vertical"),
            # self._remote_comp_container,
            mui.ThemeProvider([
                mui.HBox([
                    self._tabs
                ]).prop(flex=1)
            ], get_tight_icon_tab_theme())
        ])
        self.prop(flexDirection="row", overflow="hidden", alignItems="stretch")
        self._cur_leave_bkpt_cb: Optional[Callable[[], Coroutine[None, None,
                                                                 Any]]] = None

        self._current_mount_uid = ""
        self._current_metas: List[DebugServerProcessMeta] = []

        self._scan_duration = 2  # seconds

        self._scan_shutdown_ev = asyncio.Event()
        self._scan_loop_task: Optional[asyncio.Task] = None

        self._serv_list_lock = asyncio.Lock()
        self._vscode_handler_registered = False

    @marker.mark_did_mount
    async def _on_init(self):
        self._register_vscode_handler()
        self._scan_shutdown_ev.clear()
        self._scan_loop_task = asyncio.create_task(
            self._scan_loop(self._scan_shutdown_ev))

    @marker.mark_will_unmount
    async def _on_unmount(self):
        self._unregister_vscode_handler()
        self._scan_shutdown_ev.set()
        if self._scan_loop_task is not None:
            await self._scan_loop_task

    async def _on_server_item_click(self, ev: mui.Event):
        indexes = ev.indexes
        assert not isinstance(indexes, mui.Undefined)
        meta = self._current_metas[indexes[0]]
        if self._current_mount_uid == meta.uid:
            return
        async with self._serv_list_lock:
            await self._remote_comp_container.set_new_layout([
                mui.RemoteBoxGrpc("localhost", meta.port,
                                TENSORPC_DBG_FRAME_INSPECTOR_KEY).prop(flex=1)
            ])
        self._current_mount_uid = meta.uid

    async def _scan_loop(self, shutdown_ev: asyncio.Event):
        shutdown_task = asyncio.create_task(shutdown_ev.wait())
        sleep_task = asyncio.create_task(asyncio.sleep(self._scan_duration))
        wait_tasks = [shutdown_task, sleep_task]
        while True:
            done, pending = await asyncio.wait(
                wait_tasks, return_when=asyncio.FIRST_COMPLETED)
            if shutdown_task in done:
                break
            if sleep_task in done:
                wait_tasks.remove(sleep_task)
                sleep_task = asyncio.create_task(
                    asyncio.sleep(self._scan_duration))
                wait_tasks.append(sleep_task)
                await self._update_remote_server_discover_lst()

    async def _update_remote_server_discover_lst(self):
        async with self._serv_list_lock:
            metas = list_all_dbg_server_in_machine()
            metas.sort(key=lambda x: x.server_id)
            self._current_metas = metas
            bkpts = appctx.get_vscode_state().get_all_breakpoints()
            for i, meta in enumerate(metas):
                try:
                    frame_meta: Optional[DebugFrameMeta] = await simple_remote_call_async(
                        meta.url_with_port, dbg_serv_names.DBG_SET_BKPTS_AND_CURRENT_FRAME_META, bkpts, rpc_timeout=1)
                    if frame_meta is not None:
                        meta.secondary_name = f"{meta.pid}|{frame_meta.name}:{frame_meta.lineno}"
                    else:
                        meta.secondary_name = f"{meta.pid}|running"
                except grpc.aio.AioRpcError as e:
                    if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                        meta.secondary_name = f"{meta.pid}|disconnected"
                    elif e.code() == grpc.StatusCode.UNAVAILABLE:
                        meta.secondary_name = f"{meta.pid}|unavailable"
                    else:
                        traceback.print_exc()
                        meta.secondary_name = f"{meta.pid}|{e.code().name}"
                except:
                    print("Failed to connect to", meta.url_with_port)
                    traceback.print_exc()
                    meta.secondary_name = f"{meta.pid}|error"
                    continue
            metas_dict = [dataclasses.asdict(meta) for meta in metas]
            await self.send_and_wait(
                self._remote_server_discover_lst.update_event(dataList=metas_dict))
            found = False
            for meta in metas:
                if meta.uid == self._current_mount_uid:
                    found = True
                    break
            if not found:
                await self._remote_comp_container.set_new_layout({})
                self._current_mount_uid = ""

    async def _open_drawer(self):
        await self.send_and_wait(self._drawer.update_event(triggered=True))

    async def _close_drawer(self):
        await self.send_and_wait(self._drawer.update_event(triggered=False))

    async def _handle_secondary_actions(self, item_id: str):
        async with self._serv_list_lock:
            if item_id == ServerItemActions.UNMOUNT_REMOTE_SERVER.value:
                await self._remote_comp_container.set_new_layout({}) 
                self._current_mount_uid = ""
            elif item_id == ServerItemActions.RELEASE_BREAKPOINT.value:
                await self.release_all_breakpoints()
            elif item_id == ServerItemActions.SKIP_BREAKPOINT.value:
                await self.skip_all_breakpoints()
            elif item_id == ServerItemActions.ENABLE_BREAKPOINT.value:
                await self.enable_all_breakpoints()
            elif item_id == ServerItemActions.RECORD.value:
                await self.start_record()
        await self._update_remote_server_discover_lst()

    async def start_record(self):
        ts = time.time_ns()
        for meta in self._current_metas:
            trace_cfg = TracerConfig(True, breakpoint_count=1, trace_timestamp=ts)
            try:
                await simple_remote_call_async(meta.url_with_port, dbg_serv_names.DBG_LEAVE_BREAKPOINT,
                                        trace_cfg, rpc_timeout=1)
            except TimeoutError:
                traceback.print_exc()
            except grpc.aio.AioRpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    continue
                else:
                    traceback.print_exc()

    async def query_record_data_keys(self):
        all_keys = []
        for meta in self._current_metas:
            try:
                keys = await simple_remote_call_async(meta.url_with_port, dbg_serv_names.DBG_GET_TRACE_DATA_KEYS,
                                        rpc_timeout=1)
                all_keys.extend(keys)
            except TimeoutError:
                traceback.print_exc()
            except grpc.aio.AioRpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    continue
                else:
                    traceback.print_exc()
        return list(set(all_keys))

    async def query_record_data_by_key(self, key: str):
        all_data = []
        for meta in self._current_metas:
            try:
                data_gzipped = await simple_chunk_call_async(meta.url_with_port, dbg_serv_names.DBG_GET_TRACE_DATA,
                                        key, rpc_timeout=10)
                if data_gzipped is not None:
                    data = gzip.decompress(data_gzipped[1])
                    all_data.append(data)
            except TimeoutError:
                traceback.print_exc()
            except grpc.aio.AioRpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    continue
                else:
                    traceback.print_exc()
        # merge trace events
        all_trace_events = []
        for data in all_data:
            data_json = json.loads(data)
            all_trace_events.extend(data_json["traceEvents"])
        return json.dumps({
            "traceEvents": all_trace_events
        }).encode()

    async def _on_dist_perfetto_select(self, value: Any):
        data = await self.query_record_data_by_key(value)
        await self._dist_perfetto.set_trace_data(data, value)

    async def _on_dist_perfetto_reflesh(self):
        await self._on_tab_change("perfetto")

    async def _on_tab_change(self, value: str):
        if value == "perfetto":
            keys = await self.query_record_data_keys() 
            if keys:
                options = [(key, key) for key in keys]
                await self._perfetto_select.update_items(options, 0)
                await self._on_dist_perfetto_select(options[0][1])

    async def release_all_breakpoints(self):
        for meta in self._current_metas:
            try:
                await simple_remote_call_async(meta.url_with_port, dbg_serv_names.DBG_LEAVE_BREAKPOINT,
                                        rpc_timeout=1)
            except TimeoutError:
                traceback.print_exc()
            except grpc.aio.AioRpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    continue
                else:
                    traceback.print_exc()

    async def skip_all_breakpoints(self):
        for meta in self._current_metas:
            try:
                await simple_remote_call_async(meta.url_with_port, dbg_serv_names.DBG_SET_SKIP_BREAKPOINT, True,
                                        rpc_timeout=1)
            except TimeoutError:
                traceback.print_exc()

    async def enable_all_breakpoints(self):
        for meta in self._current_metas:
            try:
                await simple_remote_call_async(meta.url_with_port, dbg_serv_names.DBG_SET_SKIP_BREAKPOINT, False,
                                        rpc_timeout=1)
            except TimeoutError:
                traceback.print_exc()
        # await self._update_remote_server_discover_lst()

    async def release_server_breakpoint(self, event: mui.Event):
        indexes = event.indexes
        assert not isinstance(indexes, mui.Undefined)
        meta = self._current_metas[indexes[0]]
        url = meta.url_with_port
        await simple_remote_call_async(url,
                                       dbg_serv_names.DBG_LEAVE_BREAKPOINT,
                                       rpc_timeout=1)
        # await self._update_remote_server_discover_lst()

    def _register_vscode_handler(self):
        if self._vscode_handler_registered:
            return
        appctx.register_app_special_event_handler(
            AppSpecialEventType.VscodeTensorpcMessage,
            self._handle_vscode_message)
        appctx.register_app_special_event_handler(
            AppSpecialEventType.VscodeBreakpointChange,
            self._handle_vscode_bkpt_change)

        self._vscode_handler_registered = True

    def _unregister_vscode_handler(self):
        if not self._vscode_handler_registered:
            return
        self._vscode_handler_registered = False
        appctx.unregister_app_special_event_handler(
            AppSpecialEventType.VscodeTensorpcMessage,
            self._handle_vscode_message)
        appctx.unregister_app_special_event_handler(
            AppSpecialEventType.VscodeBreakpointChange,
            self._handle_vscode_bkpt_change)

    async def _handle_vscode_bkpt_change(self, bkpts: List[VscodeBreakpoint]):
        async with self._serv_list_lock:
            for meta in self._current_metas:
                try:
                    await simple_remote_call_async(meta.url_with_port, dbg_serv_names.DBG_SET_VSCODE_BKPTS, bkpts,
                                            rpc_timeout=1)
                except grpc.aio.AioRpcError as e:
                    if e.code() == grpc.StatusCode.UNAVAILABLE:
                        continue
                    else:
                        traceback.print_exc()


    async def _handle_vscode_message(self, data: VscodeTensorpcMessage):
        if data.type == VscodeTensorpcMessageType.UpdateCursorPosition:
            if data.selections is not None and len(
                    data.selections) > 0 and data.currentUri.startswith(
                        "file://"):
                path = data.currentUri[7:]
                sel = data.selections[0]
                lineno = sel.start.line + 1
                col = sel.start.character
                end_lineno = sel.end.line + 1
                end_col = sel.end.character
                code_range = (lineno, col, end_lineno, end_col)
                for meta in self._current_metas:
                    try:
                        await simple_remote_call_async(meta.url_with_port, dbg_serv_names.DBG_HANDLE_CODE_SELECTION_MSG, 
                                data.selectedCode, path, code_range,
                                                rpc_timeout=1)
                    except TimeoutError:
                        traceback.print_exc()


if __name__ == "__main__":
    print(list_all_dbg_server_in_machine())
