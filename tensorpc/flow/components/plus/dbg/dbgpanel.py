import asyncio
import dataclasses
import enum
import traceback
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union
from tensorpc.constants import TENSORPC_BG_PROCESS_NAME_PREFIX
from tensorpc.core.asyncclient import simple_remote_call_async
from tensorpc.core.client import simple_remote_call
from tensorpc.dbg.constants import TENSORPC_DBG_FRAME_INSPECTOR_KEY, DebugFrameMeta
from tensorpc.flow import marker, appctx
from tensorpc.flow.components import mui
from tensorpc.flow.components.plus.styles import CodeStyles
from tensorpc.dbg.serv_names import serv_names as dbg_serv_names
from tensorpc.compat import InWindows
import psutil

from tensorpc.flow.core.appcore import AppSpecialEventType
from tensorpc.flow.vscode.coretypes import VscodeTensorpcMessage, VscodeTensorpcMessageType


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
            parts = proc_name.split("-")[1:]
            meta = DebugServerProcessMeta(str(proc.info["pid"]), proc_name,
                                          proc.info["pid"], parts[-1],
                                          parts[0], int(parts[1]))
            res.append(meta)
            continue 
        if proc_cmdline and proc_cmdline[0].startswith(TENSORPC_BG_PROCESS_NAME_PREFIX):
            # some platform need cmdline
            parts = proc_cmdline[0].split("-")[1:]
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
            secondaryIconButtonProps=[
                # mui.IconButtonBaseProps(
                #     name=ServerItemActions.RELEASE_BREAKPOINT.value,
                #     icon=mui.IconType.PlayArrow,
                #     size="small"),
                mui.IconButtonBaseProps(
                    name=ServerItemActions.UNMOUNT_REMOTE_SERVER.value,
                    icon=mui.IconType.Close,
                    size="small"),
            ])
        self._remote_server_discover_lst.event_secondary_action_click.on_standard(
            self._handle_secondary_actions)
        remote_server_item.event_click.on_standard(
            self._on_server_item_click).configure(True)
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
                    mui.IconButton(mui.IconType.PlayArrow,
                                   self.release_all_breakpoints).prop(
                                       size="small",
                                       iconFontSize="18px"),
                    mui.IconButton(mui.IconType.DoubleArrow,
                                   self.skip_all_breakpoints).prop(
                                       size="small",
                                       iconFontSize="18px"),
                    mui.IconButton(mui.IconType.Pause,
                                   self.enable_all_breakpoints).prop(
                                       size="small",
                                       iconFontSize="18px"),
                ]).prop(alignItems="center"),
                mui.Divider(),
                self._remote_server_discover_lst,
            ]).prop(width="240px", flexShrink=0, alignItems="stretch")
        ]).prop(triggered=True, orientation="horizontal")
        self._remote_comp_container = mui.VBox([]).prop(flex=1)
        super().__init__([
            self._drawer,
            mui.Divider(orientation="vertical"),
            mui.VBox([
                mui.IconButton(mui.IconType.Menu,
                               self._open_drawer).prop(size="small",
                                                       iconFontSize="18px"),
                mui.Divider(),
                mui.IconButton(mui.IconType.Menu).prop(size="small",
                                                       iconFontSize="18px"),
            ]),
            mui.Divider(orientation="vertical"),
            self._remote_comp_container,
        ])
        self.prop(flexDirection="row")
        self._cur_leave_bkpt_cb: Optional[Callable[[], Coroutine[None, None,
                                                                 Any]]] = None

        self._current_mount_uid = ""
        self._current_metas: List[DebugServerProcessMeta] = []

        self._scan_duration = 1  # 1s

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
            for i, meta in enumerate(metas):
                try:
                    frame_meta: Optional[DebugFrameMeta] = await simple_remote_call_async(
                        meta.url_with_port, dbg_serv_names.DBG_CURRENT_FRAME_META, rpc_timeout=1)
                    if frame_meta is not None:
                        meta.secondary_name = f"{meta.pid}|{frame_meta.name}:{frame_meta.lineno}"
                    else:
                        meta.secondary_name = f"{meta.pid}|running"
                except:
                    traceback.print_exc()
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

    async def _handle_secondary_actions(self, ev: mui.Event):
        indexes = ev.indexes
        assert not isinstance(indexes, mui.Undefined)
        action = ev.data
        async with self._serv_list_lock:
            if action == ServerItemActions.RELEASE_BREAKPOINT.value:
                await self.release_server_breakpoint(ev)
            elif action == ServerItemActions.UNMOUNT_REMOTE_SERVER.value:
                await self._remote_comp_container.set_new_layout({}) 
                self._current_mount_uid = ""

    async def release_all_breakpoints(self):
        for meta in self._current_metas:
            try:
                await simple_remote_call_async(meta.url_with_port, dbg_serv_names.DBG_LEAVE_BREAKPOINT,
                                        rpc_timeout=1)
            except TimeoutError:
                traceback.print_exc()
        await self._update_remote_server_discover_lst()

    async def skip_all_breakpoints(self):
        for meta in self._current_metas:
            try:
                await simple_remote_call_async(meta.url_with_port, dbg_serv_names.DBG_SET_SKIP_BREAKPOINT, True,
                                        rpc_timeout=1)
            except TimeoutError:
                traceback.print_exc()
        await self._update_remote_server_discover_lst()

    async def enable_all_breakpoints(self):
        for meta in self._current_metas:
            try:
                await simple_remote_call_async(meta.url_with_port, dbg_serv_names.DBG_SET_SKIP_BREAKPOINT, False,
                                        rpc_timeout=1)
            except TimeoutError:
                traceback.print_exc()
        await self._update_remote_server_discover_lst()

    async def release_server_breakpoint(self, event: mui.Event):
        indexes = event.indexes
        assert not isinstance(indexes, mui.Undefined)
        meta = self._current_metas[indexes[0]]
        url = meta.url_with_port
        await simple_remote_call_async(url,
                                       dbg_serv_names.DBG_LEAVE_BREAKPOINT,
                                       rpc_timeout=1)
        await self._update_remote_server_discover_lst()

    def _register_vscode_handler(self):
        if self._vscode_handler_registered:
            return
        appctx.register_app_special_event_handler(
            AppSpecialEventType.VscodeTensorpcMessage,
            self._handle_vscode_message)
        self._vscode_handler_registered = True

    def _unregister_vscode_handler(self):
        if not self._vscode_handler_registered:
            return
        appctx.unregister_app_special_event_handler(
            AppSpecialEventType.VscodeTensorpcMessage,
            self._handle_vscode_message)
        self._vscode_handler_registered = False

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
