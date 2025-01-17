import asyncio
import dataclasses
import datetime
import enum
import gzip
import io
import time
import traceback
import uuid
import zipfile
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Union

import grpc
import psutil
import rich
import yaml

from tensorpc.compat import InWindows
from tensorpc.constants import TENSORPC_BG_PROCESS_NAME_PREFIX
from tensorpc.core.asyncclient import (simple_chunk_call_async,
                                       simple_remote_call_async)
from tensorpc.core.client import simple_remote_call
from tensorpc.dbg.constants import (TENSORPC_DBG_FRAME_INSPECTOR_KEY,
                                    TENSORPC_DBG_SPLIT, TENSORPC_DBG_TRACE_VIEW_KEY, DebugDistributedInfo, DebugFrameInfo,
                                    DebugInfo, RecordFilterConfig, RecordMode, RemoteDebugEventType, RemoteDebugTargetTrace, TargetTraceConfig, TraceLaunchType,
                                    TracerConfig, TraceResult, TracerType,
                                    TracerUIConfig)
from tensorpc.dbg.serv_names import serv_names as dbg_serv_names
from tensorpc.flow import appctx, marker
from tensorpc.flow.components import chart, mui
from tensorpc.flow.components.plus.config import ConfigDialogEvent, ConfigPanelDialog, ConfigPanelDialogPersist
from tensorpc.flow.components.plus.styles import (CodeStyles,
                                                  get_tight_icon_tab_theme)
from tensorpc.flow.core.appcore import AppSpecialEventType
from tensorpc.flow.jsonlike import as_dict_no_undefined
from tensorpc.flow.vscode.coretypes import (VscodeBreakpoint,
                                            VscodeTensorpcMessage,
                                            VscodeTensorpcMessageType)
from tensorpc.utils.rich_logging import get_logger

try:
    import orjson as json  # type: ignore

    def json_dump_to_bytes(obj: Any) -> bytes:
        # json dump/load is very slow when trace data is large
        # so we use orjson if available
        return json.dumps(obj)
except ImportError:
    import json  # type: ignore

    def json_dump_to_bytes(obj: Any) -> bytes:
        return json.dumps(obj).encode()

LOGGER = get_logger("tensorpc.dbg")

FILE_RESOURCE_KEY = "tensorpc_dbg_trace.json"

@dataclasses.dataclass
class DebugServerProcessInfo:
    id: str
    name: str
    pid: int
    uid: str
    server_id: str
    port: int
    secondary_name: str = "running"
    is_tracing: bool = False
    primaryColor: Union[mui.Undefined, mui._StdColorNoDefault] = mui.undefined
    secondaryColor: Union[mui.Undefined, mui._StdColorNoDefault] = mui.undefined
    dist_info: Optional[DebugDistributedInfo] = None
    is_mounted: bool = False
    @property
    def url_with_port(self):
        return f"localhost:{self.port}"


_INIT_YAML_CONFIG = """
# e.g. use Module._call_impl to remove all events starting with it
exclude_name_prefixes:
- Module._
- Module.named_
- DisableContext
- _disable_dynamo
- _CachingTorchDispatchMode
- OpOverload
- find_torch_dispatch_rule
- _is_compiling
- SimpleLibraryRegistry
- no_grad.__
- SelectiveCheckpointContext
- set_grad_enabled
- _NoParamDecoratorContextManager
- Tensor.__hash__
- DTensorSpec
- DTensor.__torch_dispatch__
- OpDispatcher
- Logger
- OpSchema

exclude_file_names:
- pytree.py
- simple_registry.py
- _jit_internal.py

# e.g. add "torch.nn" to trace all submodules of torch.nn only
include_modules:

exclude_modules:
- torch.distributed.tensor._dispatch # comment this to debug pytorch distributed
include_files:

exclude_files:
"""

def list_all_dbg_server_in_machine():
    res: List[DebugServerProcessInfo] = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        proc_name = proc.info["name"]
        proc_cmdline = proc.info["cmdline"]
        if proc_name.startswith(TENSORPC_BG_PROCESS_NAME_PREFIX):
            parts = proc_name.split(TENSORPC_DBG_SPLIT)[1:]
            meta = DebugServerProcessInfo(str(proc.info["pid"]), proc_name,
                                          proc.info["pid"], parts[-1],
                                          parts[0], int(parts[1]))
            res.append(meta)
            continue
        if proc_cmdline and proc_cmdline[0].startswith(
                TENSORPC_BG_PROCESS_NAME_PREFIX):
            # some platform need cmdline
            parts = proc_cmdline[0].split(TENSORPC_DBG_SPLIT)[1:]
            meta = DebugServerProcessInfo(str(proc.info["pid"]),
                                          proc_cmdline[0], proc.info["pid"],
                                          parts[-1], parts[0], int(parts[1]))
            res.append(meta)
    return res


class ServerItemActions(enum.Enum):
    RELEASE_BREAKPOINT = "release_breakpoint"
    SKIP_BREAKPOINT = "skip_breakpoint"
    ENABLE_BREAKPOINT = "enable_breakpoint"
    UNMOUNT_REMOTE_SERVER = "unmount_remote_server"
    RECORD = "record"
    RECORD_INFINITE = "record_infinite"
    RECORD_CUSTOM = "record_custom"
    FORCE_STOP_RECORD = "force_stop_record"

@dataclasses.dataclass
class _DebugPerfettoScrollToRange:
    start_us: mui.NumberType
    end_us: mui.NumberType


class MasterDebugPanel(mui.FlexBox):

    def __init__(self, app_storage_key: str = "MasterDebugPanel"):
        self._app_storage_key = app_storage_key
        assert not InWindows, "MasterDebugPanel is not supported in Windows due to setproctitle."
        lst_name_primary_prop = mui.TypographyProps(
            variant="body1",
            fontFamily=CodeStyles.fontFamily,
            overflow="hidden",
            whiteSpace="nowrap",
            textOverflow="ellipsis")
        lst_name_secondary_prop = mui.TypographyProps(
            variant="caption",
            fontFamily=CodeStyles.fontFamily,
            overflow="hidden",
            whiteSpace="nowrap",
            textOverflow="ellipsis")
        name = mui.ListItemText("").prop(
            primaryTypographyProps=lst_name_primary_prop,
            secondaryTypographyProps=lst_name_secondary_prop)
        name.bind_fields(value="server_id", secondary="secondary_name", primaryColor="primaryColor", secondaryColor="secondaryColor")
        remote_server_item = mui.ListItemButton([
            name,
        ])
        remote_server_item.bind_fields(selected="is_mounted")
        self._remote_server_discover_lst = mui.DataFlexBox(
            remote_server_item, [])
        filter_input = mui.Input("filter").prop(
            valueChangeTarget=(self._remote_server_discover_lst, "filter"))
        filter_input.prop(muiMargin="dense", paddingLeft="5px")
        self._remote_server_discover_lst.prop(filterKey="server_id",
                                              variant="list",
                                              dense=True,
                                              disablePadding=True,
                                              overflow="auto",
                                              virtualized=False)
        remote_server_item.event_click.on_standard(
            self._on_server_item_click).configure(True)
        self._menu = mui.MenuList(
            [
                # mui.MenuItem(id=ServerItemActions.RELEASE_BREAKPOINT.value,
                #              label="Release Breakpoint"),
                mui.MenuItem(id=ServerItemActions.SKIP_BREAKPOINT.value,
                             label="Disable All Breakpoints"),
                mui.MenuItem(id=ServerItemActions.ENABLE_BREAKPOINT.value,
                             label="Enable All Breakpoints"),
                # mui.MenuItem(id=ServerItemActions.UNMOUNT_REMOTE_SERVER.value,
                #              label="Unmount Remote Panel"),
                mui.MenuItem(id=ServerItemActions.RECORD.value,
                             label="Release And Start Record"),
                mui.MenuItem(id=ServerItemActions.RECORD_CUSTOM.value,
                             label="Launch Custom Record"),
                # mui.MenuItem(id=ServerItemActions.RECORD_INFINITE.value,
                #              label="Start Infinite Record"),
                # mui.MenuItem(id=ServerItemActions.FORCE_STOP_RECORD.value,
                #              label="Force Stop Record"),
            ],
            mui.IconButton(mui.IconType.MoreVert).prop(size="small"))
        self._menu.prop(anchorOrigin=mui.Anchor("top", "right"))
        self._menu.event_contextmenu_select.on(self._handle_secondary_actions)
        # self._trace_yaml_cfg_editor = mui.SimpleCodeEditor(_INIT_YAML_CONFIG, "yaml")
        # self._trace_yaml_cfg_editor.prop(backgroundColor="#fafafa", flex=1, overflow="auto", editorFontSize="12px", debounce=100)
        self._trace_yaml_cfg_editor = mui.MonacoEditor(_INIT_YAML_CONFIG, "yaml", "")
        self._trace_yaml_cfg_editor.prop(width="100%", height="40vh")
        # self._trace_yaml_cfg_editor.prop(backgroundColor="#fafafa", flex=1, overflow="auto", editorFontSize="12px", debounce=100)

        self._trace_launch_dialog = ConfigPanelDialogPersist(
            TracerUIConfig(), self._on_trace_launch, children=[
                mui.Divider(),
                mui.Typography("Record Filter (YAML, For Viztracer)").prop(variant="body1"),
                mui.Divider(),
                self._trace_yaml_cfg_editor
            ]).prop(okLabel="Launch Record", title="Record Launch Config", dividers=True)

        self._record_data_cache: Dict[str, Tuple[List[int], bytes]] = {}
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
                    # mui.IconButton(mui.IconType.FiberManualRecord, self.start_inf_record).prop(size="small", muiColor="success"),
                    # mui.IconButton(mui.IconType.Stop, self.force_trace_stop).prop(size="small"),
                    self._menu,
                ]).prop(alignItems="center"),
                mui.HBox([
                    mui.IconButton(mui.IconType.NavigateNext,
                                   self.release_all_breakpoints).prop(
                                       size="small",
                                       tooltip="Release All Breakpoints"),
                    mui.IconButton(mui.IconType.RadioButtonChecked,
                                   self.start_inf_record).prop(
                                       size="small",
                                       tooltip="Start Infinite Record",
                                       muiColor="success"),
                    mui.IconButton(mui.IconType.StopCircleOutlined,
                                   self.force_trace_stop).prop(
                                       size="small", tooltip="Stop Record"),
                    mui.IconButton(mui.IconType.LinkOff,
                                   self._unmount_remote_comp).prop(
                                       size="small", tooltip="Unmount Remote Panel"),
                ]).prop(alignItems="center"),
                mui.Divider(),
                self._remote_server_discover_lst.prop(flex="1 1 1",
                                                      minHeight=0),
            ]).prop(width="240px",
                    alignItems="stretch",
                    overflow="hidden",
                    height="100%")
        ]).prop(triggered=True, orientation="horizontal", overflow="hidden")
        self._remote_comp_container = mui.VBox([]).prop(width="100%",
                                                        height="100%",
                                                        overflow="hidden")
        self._remote_comp_tv_container = mui.VBox([]).prop(width="100%",
                                                        height="100%",
                                                        overflow="hidden")

        self._perfetto_select = mui.Select("trace", [], self._on_dist_perfetto_select)
        self._perfetto_select.prop(size="small", muiMargin="dense")
        self._dist_perfetto = chart.Perfetto().prop(width="100%",
                                                    height="100%")
        self._dist_trace_data_for_download: Optional[bytes] = None
        self._debug_use_zip_instead_of_merge = True
        self._dist_perfetto_container = mui.VBox([
            mui.HBox([
                self._perfetto_select.prop(flex=1),
                mui.IconButton(
                    mui.IconType.Refresh,
                    self._on_dist_perfetto_reflesh).prop(size="small"),
                mui.IconButton(
                    mui.IconType.Download).prop(size="small", href=f"tensorpc://{FILE_RESOURCE_KEY}", target="_blank"),
            ]).prop(alignItems="center"),
            mui.Divider(),
            self._dist_perfetto,
        ]).prop(width="100%", height="100%", overflow="hidden")
        tab_defs = [
            mui.TabDef("",
                       "remote_breakpoint_view",
                       self._remote_comp_container,
                       icon=mui.IconType.Terminal,
                       tooltip="Remote Debug Viewer"),
            mui.TabDef("",
                       "remote_trace_view",
                       self._remote_comp_tv_container,
                       icon=mui.IconType.TableView,
                       tooltip="Remote Trace Viewer"),
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
        self._tabs = mui.Tabs(tab_defs, init_value="remote_breakpoint_view",
                              before=before).prop(panelProps=mui.FlexBoxProps(
                                  width="100%", padding=0),
                                                  orientation="vertical",
                                                  borderRight=1,
                                                  flex=1,
                                                  borderColor='divider',
                                                  tooltipPlacement="right")
        self._tabs.event_change.on(self._on_tab_change)
        super().__init__([
            self._drawer,
            mui.Divider(orientation="vertical"),
            mui.ThemeProvider([mui.HBox([self._tabs]).prop(flex=1)],
                              get_tight_icon_tab_theme()),
            self._trace_launch_dialog,
        ])
        self.prop(flexDirection="row", overflow="hidden", alignItems="stretch")
        self._cur_leave_bkpt_cb: Optional[Callable[[], Coroutine[None, None,
                                                                 Any]]] = None

        self._current_mount_uid = ""
        self._current_proc_infos: List[DebugServerProcessInfo] = []

        self._scan_duration = 2  # seconds

        self._scan_shutdown_ev = asyncio.Event()
        self._scan_loop_task: Optional[asyncio.Task] = None

        self._serv_list_lock = asyncio.Lock()
        self._vscode_handler_registered = False

    @marker.mark_did_mount
    async def _on_init(self):
        self._register_handlers()
        appctx.get_app().add_file_resource(FILE_RESOURCE_KEY, self._trace_download)
        self._scan_shutdown_ev.clear()
        self._scan_loop_task = asyncio.create_task(
            self._scan_loop(self._scan_shutdown_ev))
        filter_cfg_str = await appctx.read_data_storage(f"{self._app_storage_key}/record_filter", raise_if_not_found=False)
        if filter_cfg_str is not None:
            await self.send_and_wait(self._trace_yaml_cfg_editor.update_event(value=filter_cfg_str))

    @marker.mark_will_unmount
    async def _on_unmount(self):
        self._unregister_handlers()
        appctx.get_app().remove_file_resource(FILE_RESOURCE_KEY)
        self._scan_shutdown_ev.set()
        if self._scan_loop_task is not None:
            await self._scan_loop_task

    def _trace_download(self, req: mui.FileResourceRequest):
        if self._dist_trace_data_for_download is not None:
            suffix = ".zip" if self._debug_use_zip_instead_of_merge else ".tar.gz"
            return mui.FileResource(name=f"{self._perfetto_select.value}{suffix}", content=self._dist_trace_data_for_download)
        return mui.FileResource(name=f"{self._perfetto_select.value}.json", content="{}".encode()) 

    async def _mount_remote_server_apps(self, meta: DebugServerProcessInfo):
        async with self._serv_list_lock:
            await self._remote_comp_container.set_new_layout([
                mui.RemoteBoxGrpc(
                    "localhost", meta.port,
                    TENSORPC_DBG_FRAME_INSPECTOR_KEY).prop(flex=1)
            ])
            await self._remote_comp_tv_container.set_new_layout([
                mui.RemoteBoxGrpc(
                    "localhost", meta.port,
                    TENSORPC_DBG_TRACE_VIEW_KEY).prop(flex=1)
            ])

    async def _unmount_remote_server_apps(self):
        await self._remote_comp_container.set_new_layout([])
        await self._remote_comp_tv_container.set_new_layout([])
        await self._set_selected_remote_list(None)

    async def _set_selected_remote_list(self, selected_meta: Optional[DebugServerProcessInfo] = None):
        process_infos = self._current_proc_infos
        metas_with_selected = []
        for m in process_infos:
            if selected_meta is not None and m.uid == selected_meta.uid:
                m = dataclasses.replace(m, is_mounted=True)
            else:
                m = dataclasses.replace(m, is_mounted=False)
            metas_with_selected.append(m)
        metas_dict = [as_dict_no_undefined(meta) for meta in metas_with_selected]
        await self.send_and_wait(
            self._remote_server_discover_lst.update_event(
                dataList=metas_dict))

    async def _on_server_item_click(self, ev: mui.Event):
        indexes = ev.indexes
        assert not isinstance(indexes, mui.Undefined)
        meta = self._current_proc_infos[indexes[0]]
        if self._current_mount_uid == meta.uid:
            return
        await self._mount_remote_server_apps(meta)
        self._current_mount_uid = meta.uid
        # update list selected status
        await self._set_selected_remote_list(meta)

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
            process_infos = list_all_dbg_server_in_machine()
            process_infos.sort(key=lambda x: x.server_id)
            self._current_proc_infos = process_infos
            bkpts = appctx.get_vscode_state().get_all_breakpoints()
            for i, info in enumerate(process_infos):
                try:
                    debug_info: DebugInfo = await simple_remote_call_async(
                        info.url_with_port,
                        dbg_serv_names.DBG_SET_BKPTS_AND_GET_CURRENT_INFO,
                        bkpts,
                        rpc_timeout=1)
                    frame_meta = debug_info.frame_meta
                    dist_info = debug_info.dist_info
                    trace_cfg = debug_info.trace_cfg
                    info.dist_info = dist_info
                    skipped_count = str(debug_info.metric.total_skipped_bkpt)
                    if debug_info.metric.total_skipped_bkpt > 100:
                        skipped_count = "100+"
                    status_str = f"running ({skipped_count})"
                    info.primaryColor = mui.undefined
                    prefix = str(info.pid)
                    if dist_info is not None and dist_info.run_id is not None:
                        prefix = f"{dist_info.run_id[:6]}"
                    if trace_cfg is not None:
                        tracer = trace_cfg.tracer
                        if tracer == TracerType.VIZTRACER:
                            tracer_str = "viz"
                        elif tracer == TracerType.PYTORCH:
                            tracer_str = "pth"
                        elif tracer == TracerType.VIZTRACER_PYTORCH:
                            tracer_str = "v+p"
                        else:
                            tracer_str = "unknown"
                        if trace_cfg.mode == RecordMode.INFINITE:
                            status_str = f"rec-{tracer_str} ({skipped_count}-inf)"
                        else:
                            status_str = f"rec-{tracer_str} ({skipped_count})"
                        info.is_tracing = True
                        info.primaryColor = "success"
                    if frame_meta is not None:
                        info.secondary_name = f"{prefix}|{frame_meta.name}:{frame_meta.lineno}"
                        info.primaryColor = "primary"
                    else:
                        info.secondary_name = f"{prefix}|{status_str}"
                    if self._current_mount_uid == info.uid:
                        info.is_mounted = True
                except grpc.aio.AioRpcError as e:
                    if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                        info.secondary_name = f"{info.pid}|disconnected"
                    elif e.code() == grpc.StatusCode.UNAVAILABLE:
                        info.secondary_name = f"{info.pid}|unavailable"
                    else:
                        traceback.print_exc()
                        info.secondary_name = f"{info.pid}|{e.code().name}"
                    info.primaryColor = "error"
                except:
                    print("Failed to connect to", info.url_with_port)
                    traceback.print_exc()
                    info.secondary_name = f"{info.pid}|error"
                    info.primaryColor = "error"
                    continue
            infos_dict = [as_dict_no_undefined(info) for info in process_infos]
            await self.send_and_wait(
                self._remote_server_discover_lst.update_event(
                    dataList=infos_dict))
            found = False
            for info in process_infos:
                if info.uid == self._current_mount_uid:
                    found = True
                    break
            if not found:
                await self._unmount_remote_server_apps()
                self._current_mount_uid = ""

    async def _open_drawer(self):
        await self.send_and_wait(self._drawer.update_event(triggered=True))

    async def _close_drawer(self):
        await self.send_and_wait(self._drawer.update_event(triggered=False))

    async def _unmount_remote_comp(self):
        async with self._serv_list_lock:
            if self._current_mount_uid != "":
                await self._unmount_remote_server_apps()

                self._current_mount_uid = ""

    async def _handle_secondary_actions(self, item_id: str):
        async with self._serv_list_lock:
            if item_id == ServerItemActions.UNMOUNT_REMOTE_SERVER.value:
                await self._unmount_remote_server_apps()
                self._current_mount_uid = ""
            elif item_id == ServerItemActions.RELEASE_BREAKPOINT.value:
                await self.release_all_breakpoints()
            elif item_id == ServerItemActions.SKIP_BREAKPOINT.value:
                await self.skip_all_breakpoints()
            elif item_id == ServerItemActions.ENABLE_BREAKPOINT.value:
                await self.enable_all_breakpoints()
            elif item_id == ServerItemActions.RECORD.value:
                await self.start_record()
            elif item_id == ServerItemActions.RECORD_CUSTOM.value:
                await self._trace_launch_dialog.open_config_dialog()
            elif item_id == ServerItemActions.RECORD_INFINITE.value:
                await self.start_inf_record()
            elif item_id == ServerItemActions.FORCE_STOP_RECORD.value:
                await self.force_trace_stop()
        await self._update_remote_server_discover_lst()

    async def _on_debug_perfetto_select(self, cfg: _DebugPerfettoScrollToRange):
        await self._dist_perfetto.scroll_to_range(cfg.start_us / 1e9, cfg.end_us / 1e9, 0.5)

    async def start_record(self, trace_cfg: Optional[TracerConfig] = None, dist_id: Optional[str] = None):
        ts = time.time_ns()
        for info in self._current_proc_infos:
            if dist_id is not None and info.dist_info is not None:
                if info.dist_info.run_id != dist_id:
                    continue
            if trace_cfg is None:
                trace_cfg = TracerConfig(enable=True,
                                         breakpoint_count=1,
                                         trace_timestamp=ts)
            else:
                trace_cfg = dataclasses.replace(trace_cfg, trace_timestamp=ts)
            try:
                await simple_remote_call_async(
                    info.url_with_port,
                    dbg_serv_names.DBG_LEAVE_BREAKPOINT,
                    trace_cfg,
                    rpc_timeout=1)
            except TimeoutError:
                traceback.print_exc()
            except grpc.aio.AioRpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    continue
                else:
                    traceback.print_exc()

    async def force_trace_stop(self):
        await self._run_rpc_on_processes(self._current_proc_infos,
                                     dbg_serv_names.DBG_FORCE_TRACE_STOP,
                                     rpc_timeout=3)

    async def _on_trace_launch(self, cfg_ev: ConfigDialogEvent[TracerUIConfig]):
        config = cfg_ev.cfg
        filter_cfg_value = self._trace_yaml_cfg_editor.props.value
        if not isinstance(filter_cfg_value, mui.Undefined):
            filter_cfg = yaml.safe_load(filter_cfg_value)
            filter_obj = RecordFilterConfig(**filter_cfg)
            await appctx.save_data_storage(f"{self._app_storage_key}/record_filter", filter_cfg_value)
            cfg = TracerConfig(enable=True,
                                record_filter=filter_obj,
                            **dataclasses.asdict(config))
            if cfg.tracer == TracerType.VIZTRACER:
                cfg.trace_name = f"{cfg.trace_name}|viz"
            elif cfg.tracer == TracerType.PYTORCH:
                cfg.trace_name = f"{cfg.trace_name}|pth"
            elif cfg.tracer == TracerType.VIZTRACER_PYTORCH:
                cfg.trace_name = f"{cfg.trace_name}|v+p"

            await self.start_record(cfg)

    async def start_inf_record(self):
        cfg = TracerConfig(enable=True, mode=RecordMode.INFINITE)
        await self.start_record(cfg)

    async def query_record_data_keys(self):
        all_keys_may_none = await self._run_rpc_on_processes(
            self._current_proc_infos,
            dbg_serv_names.DBG_GET_TRACE_DATA_KEYS,
            rpc_timeout=1)
        all_keys = []
        for keys in all_keys_may_none:
            if keys is None:
                continue
            all_keys.extend(keys)
        return list(set(all_keys))

    async def query_record_data_by_key(self, key: str):
        # perfetto support zip of gzip trace in their source code.
        # so we can just use already gzipped data to avoid expensive zip again.
        _use_perfetto_undoc_zip_of_gzip = False
        LOGGER.warning("Querying record data by key %s", key)
        if key in self._record_data_cache:
            all_timestamps_with_none: List[Optional[int]] = await self._run_rpc_on_metas_chunk_call(
                self._current_proc_infos,
                dbg_serv_names.DBG_GET_TRACE_DATA_TIMESTAMP,
                key,
                rpc_timeout=1)
            all_timestamps = [ts for ts in all_timestamps_with_none if ts is not None]
            cached_timestamps, cached_data = self._record_data_cache[key]
            # if cached_timestamps contains all timestamps in all_timestamps, we can use cache
            if all(ts in cached_timestamps for ts in all_timestamps):
                return cached_data, cached_timestamps
        
        all_data_gzipped: List[Optional[Tuple[int, TraceResult]]] = await self._run_rpc_on_metas_chunk_call(
            self._current_proc_infos,
            dbg_serv_names.DBG_GET_TRACE_DATA,
            key,
            rpc_timeout=30)
        LOGGER.warning("Finish querying record data by key %s", key)

        # print("RPC TIME", time.time() - t)
        all_data = []
        all_data_external_evs = []
        all_timestamps = []
        for data_gzipped in all_data_gzipped:
            if data_gzipped is None:
                continue
            if not _use_perfetto_undoc_zip_of_gzip:
                datas = [gzip.decompress(d.data) for d in data_gzipped[1].single_results]
            else:
                datas = [d.data for d in data_gzipped[1].single_results]
            for single_res in data_gzipped[1].single_results:
                if single_res.external_events is not None:
                    all_data_external_evs.append(single_res.external_events)
            all_data.extend(datas)
            all_timestamps.append(data_gzipped[0])
        LOGGER.warning("Decompressing record data by key %s", key)

        if not all_data:
            raise ValueError("No trace data found for key", key)
        if self._debug_use_zip_instead_of_merge:
            zip_ss = io.BytesIO()
            zip_mode = zipfile.ZIP_DEFLATED if not _use_perfetto_undoc_zip_of_gzip else zipfile.ZIP_STORED
            compresslevel = 9 if not _use_perfetto_undoc_zip_of_gzip else None
            with zipfile.ZipFile(zip_ss, mode="w", compression=zip_mode, compresslevel=compresslevel) as zf:
                for i, data in enumerate(all_data):
                    zf.writestr(f"{i}.json", data)
                for i, data in enumerate(all_data_external_evs):
                    if data:
                        zf.writestr(f"{i}_extra.json", json.dumps({
                            "traceEvents": data
                        }))
            res = zip_ss.getvalue()
            if _use_perfetto_undoc_zip_of_gzip:
                LOGGER.warning("Zip (store mode) to file with length %d", len(res))
            else:
                LOGGER.warning("Zip (best compress) to file with length %d", len(res))
            self._record_data_cache[key] = (all_timestamps, res)
            return res, all_timestamps
        else:
            # print("DECOMPRESS TIME", time.time() - t)
            # merge trace events
            all_trace_events = []
            data_json_meta = {}
            for i, data in enumerate(all_data):
                data_json = json.loads(data)
                trace_ev = data_json.pop("traceEvents")
                external_evs = all_data_external_evs[i]
                trace_ev.extend(external_evs)
                all_trace_events.extend(trace_ev)
                if i == 0:
                    data_json_meta = data_json
            # print("JSON LOAD TIME", time.time() - t)
            res_trace = {"traceEvents": all_trace_events}
            res_trace.update(data_json_meta)
            res_data = json_dump_to_bytes(res_trace)
            # print("JSON DUMP TIME", time.time() - t)
            res = gzip.compress(res_data)
            # print("ALL GZIP TIME", time.time() - t)
            self._record_data_cache[key] = (all_timestamps, res)
            return res, all_timestamps

    async def _on_dist_perfetto_select(self, value: Any):
        data, timestamps = await self.query_record_data_by_key(value)
        self._dist_trace_data_for_download = data
        title = value
        if timestamps:
            time_str = datetime.datetime.fromtimestamp(timestamps[0] / 1e9).strftime('%m-%d %H:%M:%S')
            title = f"{value} ({time_str})"
        await self._dist_perfetto.set_trace_data(data, title)

    async def _on_dist_perfetto_reflesh(self):
        await self._on_tab_change("perfetto")

    async def _on_tab_change(self, value: str):
        if value == "perfetto":
            keys = await self.query_record_data_keys()
            if keys:
                options = [(key, key) for key in keys]
                await self._perfetto_select.update_items(options)
                await self._on_dist_perfetto_select(self._perfetto_select.value)

    async def release_all_breakpoints(self):
        await self._run_rpc_on_processes(self._current_proc_infos,
                                     dbg_serv_names.DBG_LEAVE_BREAKPOINT,
                                     rpc_timeout=1)

    async def skip_all_breakpoints(self):
        await self._run_rpc_on_processes(self._current_proc_infos,
                                     dbg_serv_names.DBG_SET_SKIP_BREAKPOINT,
                                     True,
                                     rpc_timeout=1)

    async def enable_all_breakpoints(self):
        await self._run_rpc_on_processes(self._current_proc_infos,
                                     dbg_serv_names.DBG_SET_SKIP_BREAKPOINT,
                                     False,
                                     rpc_timeout=1)

    async def _run_rpc_on_process(self,
                               meta: DebugServerProcessInfo,
                               service_key: str,
                               *args,
                               rpc_timeout: float = 1,
                               rpc_is_chunk_call: bool = False):
        if rpc_is_chunk_call:
            rpc_func = simple_chunk_call_async
        else:
            rpc_func = simple_remote_call_async
        try:
            return await rpc_func(meta.url_with_port,
                                  service_key,
                                  *args,
                                  rpc_timeout=rpc_timeout)
        except TimeoutError:
            traceback.print_exc()
            return None
        except grpc.aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                return None
            else:
                traceback.print_exc()
                return None

    async def _run_rpc_on_processes(self,
                                process_infos: List[DebugServerProcessInfo],
                                service_key: str,
                                *args,
                                rpc_timeout: float = 1):
        all_tasks = []
        for info in process_infos:
            all_tasks.append(
                self._run_rpc_on_process(info,
                                      service_key,
                                      *args,
                                      rpc_timeout=rpc_timeout))
        return await asyncio.gather(*all_tasks)

    async def _run_rpc_on_metas_chunk_call(self,
                                           process_infos: List[DebugServerProcessInfo],
                                           service_key: str,
                                           *args,
                                           rpc_timeout: float = 10):
        all_tasks = []
        for info in process_infos:
            all_tasks.append(
                self._run_rpc_on_process(info,
                                      service_key,
                                      *args,
                                      rpc_timeout=rpc_timeout,
                                      rpc_is_chunk_call=True))
        return await asyncio.gather(*all_tasks)

    async def release_server_breakpoint(self, event: mui.Event):
        indexes = event.indexes
        assert not isinstance(indexes, mui.Undefined)
        info = self._current_proc_infos[indexes[0]]
        url = info.url_with_port
        await simple_remote_call_async(url,
                                       dbg_serv_names.DBG_LEAVE_BREAKPOINT,
                                       rpc_timeout=1)
        # await self._update_remote_server_discover_lst()

    async def _handle_target_trace_from_distributed_group_worker(self, ev: mui.RemoteCompEvent):
        trace_ev = ev.data
        assert isinstance(trace_ev, RemoteDebugTargetTrace), f"WTF, {ev}"
        cfg = TracerConfig(enable=True,
                        **dataclasses.asdict(self._trace_launch_dialog.config))
        cfg.breakpoint_count = 1
        cfg.mode = RecordMode.NEXT_BREAKPOINT
        cfg.tracer = TracerType.TARGET_TRACER
        cfg.launch_type = TraceLaunchType.TARGET_VARIABLE
        target_trace_cfg = TargetTraceConfig(trace_ev.target_filename, 
            trace_ev.target_func_qname,
                trace_ev.target_expr,
                trace_ev.is_distributed,
                trace_ev.max_num_variable)
        cfg.target_trace_cfg = target_trace_cfg
        await self.start_record(cfg, trace_ev.dist_info.run_id)

    def _register_handlers(self):
        if self._vscode_handler_registered:
            return
        appctx.register_app_special_event_handler(
            AppSpecialEventType.VscodeTensorpcMessage,
            self._handle_vscode_message)
        appctx.register_app_special_event_handler(
            AppSpecialEventType.VscodeBreakpointChange,
            self._handle_vscode_bkpt_change)
        appctx.register_remote_comp_event_handler(RemoteDebugEventType.DIST_TARGET_VARIABLE_TRACE.value, 
            self._handle_target_trace_from_distributed_group_worker)
        self._vscode_handler_registered = True

    def _unregister_handlers(self):
        if not self._vscode_handler_registered:
            return
        self._vscode_handler_registered = False
        appctx.unregister_app_special_event_handler(
            AppSpecialEventType.VscodeTensorpcMessage,
            self._handle_vscode_message)
        appctx.unregister_app_special_event_handler(
            AppSpecialEventType.VscodeBreakpointChange,
            self._handle_vscode_bkpt_change)
        appctx.unregister_remote_comp_event_handler(RemoteDebugEventType.DIST_TARGET_VARIABLE_TRACE.value, 
            self._handle_target_trace_from_distributed_group_worker)

    async def _handle_vscode_bkpt_change(self, bkpts: Dict[str, tuple[list[VscodeBreakpoint], int]]):
        async with self._serv_list_lock:
            await self._run_rpc_on_processes(self._current_proc_infos,
                                         dbg_serv_names.DBG_SET_VSCODE_BKPTS,
                                         bkpts,
                                         rpc_timeout=1)

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
                for meta in self._current_proc_infos:
                    try:
                        await simple_remote_call_async(
                            meta.url_with_port,
                            dbg_serv_names.DBG_HANDLE_CODE_SELECTION_MSG,
                            data.selectedCode,
                            path,
                            code_range,
                            rpc_timeout=1)
                    except TimeoutError:
                        traceback.print_exc()


if __name__ == "__main__":
    print(list_all_dbg_server_in_machine())
