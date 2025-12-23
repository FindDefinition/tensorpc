import ast
import asyncio
from collections.abc import Sequence
import enum
from functools import partial
import importlib
import inspect
import json
import linecache
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile
import threading
import time
import traceback
import types
from typing import Any, Callable, Union, cast
import watchdog
import watchdog.events
from watchdog.observers import Observer
from watchdog.observers.api import ObservedWatch
import dataclasses as dataclasses_plain
from tensorpc.apps.mls.backends._triton.runtime import TritonInlineRunEnv
from tensorpc.core.astex.sourcecache import SCDItem, SourceChangeDiffCache
from tensorpc.apps.mls import tsim
from tensorpc.apps.mls.components.global_mem import Label, MatrixPanel, GlobalMemContainer, GlobalMemoryModel, Matrix, layout_table_inplace
from tensorpc.apps.mls.tsim.core import TensorSimIoOp, get_flush_sim_io_ops, get_tensorsim_context_checked
from tensorpc.constants import PACKAGE_ROOT
from tensorpc.core.annolib import is_undefined
from tensorpc.core.astex.astcache import AstCache, AstCacheItem
from tensorpc.core.datamodel.draft import DraftBase, DraftFieldMeta
from tensorpc.core.funcid import find_toplevel_func_node_by_lineno, find_toplevel_func_node_container_by_lineno
from tensorpc.core.moduleid import get_module_id_of_type
from tensorpc.core.pfl.evaluator import PFLAsyncThread, PFLBreakpointDesc, PFLRunnerExprHit, PFLRunnerStateType, PFLRunnerBreakpoint, PFLRunnerCtrlFor, PFLRunnerCtrlBase
from tensorpc.core.pfl.backends.js import ColorUtil, Math, MathUtil
from tensorpc.core.pfl.pfl_ast import unparse_pfl_ast
from tensorpc.core.tree_id import UniqueTreeId, UniqueTreeIdForTree
from tensorpc.dock import mui, three, plus, mark_create_layout, mark_did_mount, appctx
from typing import Annotated, Any, Optional, Union
from tensorpc.core import pfl
from tensorpc.dock.components.mui import MonacoBreakpoint
from tensorpc.dock.components.plus.styles import get_tight_icon_tab_theme_horizontal
from tensorpc.dock.flowapp.appstorage import AppDraftFileStoreBackend

from tensorpc.apps.mls.backends import tritonstd
import numpy as np 
from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.dock.core.appcore import AppSpecialEventType
from tensorpc.dock.vscode.coretypes import VscodeTensorpcMessage, VscodeTensorpcMessageType
from tensorpc.utils.package_finder import find_submodule_from_file
import importlib.machinery

from tensorpc.utils.wait_tools import debounce

class EditorActions(enum.Enum):
    RUN_TO = "Run To"
    EXPR_TRACE = "Expr Trace"

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class ExprTraceData:
    for_steps: tuple[int, ...]
    grid_id: int
    value: Union[tritonstd.Tensor, int, float, bool]

@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class LocalMatrix(Matrix):
    id: str = ""
    global_indices: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)

@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class LocalMemoryModel:
    matrix: LocalMatrix
    minimap: plus.hud.MinimapModel
    hover: Optional[str] = None

    @classmethod 
    def empty(cls):
        return cls(
            matrix=LocalMatrix.empty(),
            minimap=plus.hud.MinimapModel(),
        )

@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class LabelWithId(Label):
    id: str

@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class ExprTraceMemoryModel:
    matrices: list[LocalMatrix]
    labels: list[LabelWithId]
    minimap: plus.hud.MinimapModel
    title: str = "expr trace"
    hover: Optional[str] = None

    @staticmethod 
    def empty():
        return ExprTraceMemoryModel(
            matrices=[],
            labels=[],
            minimap=plus.hud.MinimapModel(
                fit_mode=int(plus.hud.MinimapFitMode.HEIGHT),
                align_mode=int(plus.hud.MinimapAlignMode.LEFT_TOP)
            )
        )

class LocalMemContainer(mui.TooltipFlexBox):
    def __init__(self, key: str, draft: LocalMemoryModel):
        assert isinstance(draft, DraftBase)
        panel = MatrixPanel(draft.matrix, enable_hover_line=False)
        minimap = plus.hud.MiniMap(draft.minimap, [
            panel
        ], minimap_event_key="local_mem_minimap")
        self.panel = panel
        self.minimap = minimap
        self._draft = draft
        cam = three.OrthographicCamera(near=0.1, far=1000, children=[
            minimap,
        ]).prop(position=(0, 0, 10))
        canvas = three.View([
            cam.prop(makeDefault=True),
            # three.InfiniteGridHelper(1, 10, "green")
        ]).prop(allowKeyboardEvent=True)
        minimap.install_canvas_events(draft.minimap, canvas)
        layout = [
            mui.Typography(key).prop(variant="caption"),
            canvas.prop(flex=1)
        ]
        super().__init__("", layout)
        self.bind_fields(title=draft.hover)
        self.prop(minHeight=0,
                minWidth=0,
                flexFlow="column nowrap",
                width="100%",
                height="100%",
                overflow="hidden",
                followCursor=True)


class LocalMemContainerV2(mui.TooltipFlexBox):
    def __init__(self, key_draft: str, draft: LocalMemoryModel):
        assert isinstance(draft, DraftBase)
        panel = MatrixPanel(draft.matrix, enable_hover_line=False)
        minimap = plus.hud.MiniMap(draft.minimap, [
            panel
        ], minimap_event_key="local_mem_minimap")
        self.panel = panel
        self.minimap = minimap
        self._draft = draft
        cam = three.OrthographicCamera(near=0.1, far=1000, children=[
            minimap,
        ]).prop(position=(0, 0, 10))
        canvas = three.View([
            cam.prop(makeDefault=True),
            # three.InfiniteGridHelper(1, 10, "green")
        ]).prop(allowKeyboardEvent=True)
        minimap.install_canvas_events(draft.minimap, canvas)
        layout = [
            mui.Typography().prop(variant="caption").bind_fields(value=key_draft),
            canvas.prop(flex=1)
        ]
        super().__init__("", layout)
        self.bind_fields(title=draft.hover)
        self.prop(minHeight=0,
                minWidth=0,
                flexFlow="column nowrap",
                width="100%",
                height="100%",
                overflow="hidden",
                followCursor=True)

class MatrixTableContainer(mui.TooltipFlexBox):
    def __init__(self, draft: ExprTraceMemoryModel):
        assert isinstance(draft, DraftBase)
        dm = mui.DataModel.get_datamodel_from_draft(draft)
        draft_nested = dm.create_external_draft_with_self(Matrix)
        label_draft_nested = dm.create_external_draft_with_self(LabelWithId)
        panel = MatrixPanel(draft_nested, enable_hover_line=False, label_with_shape=False)
        label = three.Text("").bind_fields(value=label_draft_nested.text, fontSize=label_draft_nested.fontSize)
        label.prop(color="blue", fillOpacity=0.5)
        label.bind_fields_unchecked_dict({
            "position-x": label_draft_nested.offsetX,
            "position-y": f"-{label_draft_nested.offsetY}",
        })
        minimap = plus.hud.MiniMap(draft.minimap, [
            three.DataListGroup(panel).bind_fields(dataList=draft.matrices),
            three.DataListGroup(label).bind_fields(dataList=draft.labels),
        ], minimap_event_key="matrix_table")
        self.panel = panel
        self.minimap = minimap
        self._draft = draft
        cam = three.OrthographicCamera(near=0.1, far=1000, children=[
            minimap,
        ]).prop(position=(0, 0, 10))
        canvas = three.View([
            cam.prop(makeDefault=True),
            # three.InfiniteGridHelper(1, 10, "green")
        ]).prop(allowKeyboardEvent=True)
        minimap.install_canvas_events(draft.minimap, canvas)
        layout = [
            mui.Typography("").prop(variant="caption").bind_fields(value=draft.title),
            canvas.prop(flex=1)
        ]
        super().__init__("", layout)
        self.bind_fields(title=draft.hover)
        self.prop(minHeight=0,
                minWidth=0,
                flexFlow="column nowrap",
                width="100%",
                height="100%",
                overflow="hidden",
                followCursor=True)

@dataclasses.dataclass(kw_only=True)
class TritonSimThreadStateModel:
    label: str 
    is_paused: bool = False
    cur_local_tensor_id: Optional[tuple[str, str]] = None

@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class TritonSimModel:
    grid_idx_x: int = 0
    grid_idx_y: int = 0
    grid_idx_z: int = 0

    grid_size_x_range: tuple[int, int, int] = (0, 0, 1)
    grid_size_y_range: tuple[int, int, int] = (0, 0, 1)
    grid_size_z_range: tuple[int, int, int] = (0, 0, 1)
    global_mem: GlobalMemoryModel
    # stack tensors during triton simulation
    # local_matrices: dict[str, LocalMemoryModel] = dataclasses.field(default_factory=dict)
    local_var_key: str = ""
    local_mat: LocalMemoryModel = dataclasses.field(default_factory=LocalMemoryModel.empty)
    expr_trace_matrices: ExprTraceMemoryModel
    thread_options: Annotated[list[TritonSimThreadStateModel], DraftFieldMeta(is_store_external=True)]  = dataclasses.field(default_factory=list)
    cur_thread_option: Annotated[Optional[TritonSimThreadStateModel], DraftFieldMeta(is_store_external=True)] = None

    @pfl.js.mark_js_compilable
    def _on_hover_pfl_single(self, data: three.PointerEvent):
        local_mat = self.local_mat
        point_unified_x = data.pointLocal[0] + 0.5
        point_unified_y = -data.pointLocal[1] + 0.5
        idx_x = Math.floor(point_unified_x * local_mat.matrix.width)
        idx_y = Math.floor(point_unified_y * local_mat.matrix.height)
        flat_idx = idx_y * local_mat.matrix.width + idx_x
        if local_mat.matrix.data is not None:
            data_arr = MathUtil.getTypedArray(local_mat.matrix.data)
            value = data_arr[flat_idx]
            local_mat.hover = str(value)
        for global_key, indices in local_mat.matrix.global_indices.items():
            # print(global_key in self.global_mem.matrices)
            if global_key in self.global_mem.matrices:
                global_mat = self.global_mem.matrices[global_key]
                inds_flat_buffer = MathUtil.getTypedArray(indices[flat_idx])
                line_pos = np.empty([inds_flat_buffer.length, 2], np.float32)
                line_size = np.empty([inds_flat_buffer.length, 2], np.float32)

                line_pos_buffer = MathUtil.getTypedArray(line_pos)
                line_size_buffer = MathUtil.getTypedArray(line_size)
                for j in range(inds_flat_buffer.length):
                    x = inds_flat_buffer[j] % global_mat.width + 0.5 - global_mat.width / 2
                    y = (Math.floor(inds_flat_buffer[j] / global_mat.width) + 0.5 - global_mat.height / 2)
                    if global_mat.transposed:
                        tmp = x
                        x = y
                        y = tmp
                    line_pos_buffer[j * 2] = x
                    line_pos_buffer[j * 2 + 1] = -y
                    line_size_buffer[j * 2] = 1
                    line_size_buffer[j * 2 + 1] = 1
                # print(fill_pos)
                # print("COLOR", fill_pos_buffer[0], fill_pos_buffer[1], fill_pos_buffer[2])
                # print("POS", inds_flat_buffer[0] % global_mat.width, Math.floor(inds_flat_buffer[0] / global_mat.width))
                global_mat.temp_aabb_line_pos = line_pos
                global_mat.temp_aabb_line_size = line_size


    @pfl.js.mark_js_compilable
    def _on_matrix_table_elem_enter_pfl(self, data: three.PointerEvent):
        if data.dataIndexes:
            dataIndexes = pfl.compiler_remove_optional(data.dataIndexes)
            local_mat = self.expr_trace_matrices.matrices[dataIndexes[0]]
            for mat in self.expr_trace_matrices.matrices:
                mat.hovered = False
            local_mat.hovered = True

    @pfl.js.mark_js_compilable
    def _on_matrix_table_elem_click_pfl(self, data: three.PointerEvent):
        if data.dataIndexes:
            dataIndexes = pfl.compiler_remove_optional(data.dataIndexes)
            local_mat = self.expr_trace_matrices.matrices[dataIndexes[0]]
            for mat in self.expr_trace_matrices.matrices:
                mat.selected = False
            local_mat.selected = True
            for global_key, indices in local_mat.global_indices.items():
                if global_key in self.global_mem.matrices:
                    global_mat = self.global_mem.matrices[global_key]
                    inds_flat_buffer = MathUtil.getTypedArray(indices)
                    line_pos = np.empty([inds_flat_buffer.length, 2], np.float32)
                    line_pos_buffer = MathUtil.getTypedArray(line_pos)
                    for j in range(inds_flat_buffer.length):
                        x = inds_flat_buffer[j] % global_mat.width + 0.5 - global_mat.width / 2
                        y = (Math.floor(inds_flat_buffer[j] / global_mat.width) + 0.5 - global_mat.height / 2)
                        if global_mat.transposed:
                            tmp = x
                            x = y
                            y = tmp
                        line_pos_buffer[j * 2] = x
                        line_pos_buffer[j * 2 + 1] = -y
                    global_mat.temp_mask_pos = line_pos

    @pfl.js.mark_js_compilable
    def _on_matrix_table_bkgd_click(self, data: three.PointerEvent):
        for mat in self.expr_trace_matrices.matrices:
            mat.selected = False

    @pfl.js.mark_js_compilable
    def _on_matrix_table_elem_hover_pfl(self, data: three.PointerEvent):
        if data.dataIndexes:
            dataIndexes = pfl.compiler_remove_optional(data.dataIndexes)
            local_mat = self.expr_trace_matrices.matrices[dataIndexes[0]]
            # hover vis
            point_unified_x = data.pointLocal[0] + 0.5
            point_unified_y = -data.pointLocal[1] + 0.5
            idx_x = Math.floor(point_unified_x * local_mat.width)
            idx_y = Math.floor(point_unified_y * local_mat.height)
            local_mat.linePosX = (idx_x + 0.5) - local_mat.width / 2
            local_mat.linePosY = ((-(idx_y + 0.5)) + local_mat.height / 2) * local_mat.height_scale
            flat_idx = idx_y * local_mat.width + idx_x
            # self.linePosX = (idx_x + 0.5) - local_mat.width / 2
            # self.linePosY = (-(idx_y + 0.5)) + local_mat.height / 2
            if local_mat.data is not None:
                data_arr = MathUtil.getTypedArray(local_mat.data)
                value = data_arr[flat_idx]
                self.expr_trace_matrices.hover = str(value)
            for global_key, indices in local_mat.global_indices.items():
                # print(global_key in self.global_mem.matrices)
                if global_key in self.global_mem.matrices:
                    global_mat = self.global_mem.matrices[global_key]
                    inds_flat_buffer = MathUtil.getTypedArray(indices[flat_idx])
                    line_pos = np.empty([inds_flat_buffer.length, 2], np.float32)
                    line_size = np.empty([inds_flat_buffer.length, 2], np.float32)

                    line_pos_buffer = MathUtil.getTypedArray(line_pos)
                    line_size_buffer = MathUtil.getTypedArray(line_size)
                    for j in range(inds_flat_buffer.length):
                        x = inds_flat_buffer[j] % global_mat.width + 0.5 - global_mat.width / 2
                        y = (Math.floor(inds_flat_buffer[j] / global_mat.width) + 0.5 - global_mat.height / 2)
                        if global_mat.transposed:
                            tmp = x
                            x = y
                            y = tmp
                        line_pos_buffer[j * 2] = x
                        line_pos_buffer[j * 2 + 1] = -y
                        line_size_buffer[j * 2] = 1
                        line_size_buffer[j * 2 + 1] = 1
                    global_mat.temp_aabb_line_pos = line_pos
                    global_mat.temp_aabb_line_size = line_size

    @pfl.js.mark_js_compilable
    def _on_matrix_table_elem_hover_leave_pfl(self, data: three.PointerEvent):
        self.expr_trace_matrices.hover = None
        for mat in self.expr_trace_matrices.matrices:
            mat.linePosX = None
            mat.linePosY = None
            mat.hovered = False

        for global_key, mat in self.global_mem.matrices.items():
            mat.temp_aabb_line_pos = None
            mat.temp_aabb_line_size = None

    @pfl.js.mark_js_compilable
    def _on_hover_leave_pfl_single(self, data: three.PointerEvent):
        self.local_mat.hover = None
        for global_key, mat in self.global_mem.matrices.items():
            # mat.temp_fill_pos = None 
            # mat.temp_fill_color = None
            mat.temp_aabb_line_pos = None
            mat.temp_aabb_line_size = None

    def get_global_fill(self, global_key: str, inds: np.ndarray, is_persist: bool = True, color_advance: Optional[np.ndarray] = None):
        global_mat = self.global_mem.matrices[global_key]
        return global_mat.get_global_fill(global_key, inds, is_persist=is_persist, color_advance=color_advance)


@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class SingleBlockRunState:
    # used to record all memory access in history.
    global_access_indices: dict[str, np.ndarray]
    global_access_advances: dict[str, np.ndarray]
    global_access_cnt: dict[str, int]


class InlineCompPrefix(enum.Enum):
    CONTROLS = "ctrls"
    LOCAL_TENSORS = "local_tensors"

def get_prefix_data_from_key(key: str) -> tuple[InlineCompPrefix, str]:
    """
    Get the prefix and data from the key.
    The key is in the format of "prefix-data".
    """
    if "-" not in key:
        raise ValueError(f"Key {key} must contain a '-' to separate prefix and data")
    prefix_str, data = key.split("-", 1)
    try:
        prefix = InlineCompPrefix(prefix_str)
    except ValueError:
        raise ValueError(f"Invalid prefix {prefix_str} in key {key}")
    return prefix, data

def get_key_from_prefix_data(prefix: InlineCompPrefix, data: str) -> str:
    """ Get the key from the prefix and data.
    The key is in the format of "prefix-data".
    """
    return f"{prefix.value}-{data}"

_WATCHDOG_MODIFY_EVENT_TYPES = Union[watchdog.events.DirModifiedEvent,
                                     watchdog.events.FileModifiedEvent]

class _WatchDogForKernelFile(watchdog.events.FileSystemEventHandler):

    def __init__(
            self, on_modified: Callable[[_WATCHDOG_MODIFY_EVENT_TYPES],
                                        None],
                ignore_mtime: float) -> None:
        super().__init__()
        self._on_modified = on_modified
        self._ignore_mtime = ignore_mtime
        self._disable = False
        self._lock = threading.Lock()

    def on_modified(self, event: _WATCHDOG_MODIFY_EVENT_TYPES):
        if isinstance(event, watchdog.events.FileModifiedEvent) and not self._disable:
            with self._lock:
                src_path = event.src_path
                if isinstance(src_path, bytes):
                    src_path = src_path.decode("utf-8")
                cur_mtime = Path(src_path).stat().st_mtime
                if cur_mtime == self._ignore_mtime and self._ignore_mtime != -1:
                    return 
                # print(f"File modified: {src_path}, mtime: {cur_mtime}, ignore_mtime: {self._ignore_mtime}")
                self._ignore_mtime = cur_mtime
                return self._on_modified(event)

def _temp_module_path_getter(fn: Any, path_editor: str, code_editor: str):
    mod = inspect.getmodule(fn)
    assert mod is not None, "module_code_path_getter must be provided if func isn't a module function"
    path = inspect.getabsfile(mod)
    if Path(path).resolve() == Path(path_editor).resolve():
        return code_editor, path_editor
    module_code = inspect.getsource(mod)
    return module_code, inspect.getabsfile(mod)


@dataclasses_plain.dataclass
class TritonKernelManagerState:
    fn: Callable 
    cur_fn_name: str 
    fn_names: list[str]
    path: str 
    content: str 
    content_lines: list[str]
    lib: pfl.PFLLibrary
    module_dict: dict[str, Any]
    runner: tritonstd.TritonKernelRunner
    # used to capture specific variables in the kernel, such as local tensors.
    expr_trace_runner: tritonstd.TritonKernelRunner
    finder_dict: dict[str, pfl.PFLTreeNodeFinder]
    expr_finder_dict: dict[str, pfl.PFLTreeExprFinder]
    runner_task: Optional[asyncio.Task] = None
    expr_trace_task: Optional[asyncio.Task] = None
    mapper_new_to_old: Optional[SCDItem] = None
    watchdog_watcher: Optional[_WatchDogForKernelFile] = None
    watchdog_observer: Optional[Any] = None
    bkpts: Optional[list[mui.MonacoBreakpoint]] = None

    def setup_watchdog(self, handler: Callable[[_WATCHDOG_MODIFY_EVENT_TYPES], None], ignore_mtime: float) -> None:
        assert self.watchdog_observer is None and self.watchdog_watcher is None
        self.watchdog_watcher = _WatchDogForKernelFile(debounce(0.1)(handler), ignore_mtime)
        self.watchdog_observer = Observer()
        self.watchdog_observer.schedule(self.watchdog_watcher, self.path, recursive=False)
        self.watchdog_observer.start()

    def stop_watchdog(self):
        if self.watchdog_observer is not None:
            self.watchdog_observer.stop()
            self.watchdog_observer.join()
        self.watchdog_observer = None 
        if self.watchdog_watcher is not None:
            self.watchdog_watcher._disable = True
        self.watchdog_watcher = None

    def switch_lib_in_same_file(self, new_fn_name: str):
        new_fn = self.module_dict[new_fn_name]
        runner = tritonstd.parse_triton_compilable_to_runner(new_fn, do_meta_eval=True, 
            # module_code_path_getter=lambda x: (self.content, self.path)
            module_code_path_getter=partial(_temp_module_path_getter, path_editor=self.path, code_editor=self.content)
            )
        # print(ast.ret_st)
        lib = runner._library
        finder_dict: dict[str, pfl.PFLTreeNodeFinder] = {}
        expr_finder_dict: dict[str, pfl.PFLTreeExprFinder] = {}
        for k, v in lib.all_compiled.items():
            finder_dict[k] = pfl.PFLTreeNodeFinder(v, (pfl.PFLName, pfl.PFLAttribute, pfl.PFLArg)) 
            expr_finder_dict[k] = pfl.PFLTreeExprFinder(v)
        self.fn = new_fn
        self.cur_fn_name = new_fn_name
        self.lib = lib
        self.runner = runner
        self.expr_trace_runner = runner.copy()
        self.finder_dict = finder_dict
        self.expr_finder_dict = expr_finder_dict
        self.mapper_new_to_old = None

class TritonKernelManager:
    def __init__(self, item: AstCacheItem, path: str, lineno: int, fn_name: Optional[str] = None, bkpts: Optional[list[mui.MonacoBreakpoint]] = None,
            fixed_inline_env: Optional[TritonInlineRunEnv] = None):
        self._compiled_tmp_file_path: Optional[str] = None
        if fn_name is None:
            func_nodes = find_toplevel_func_node_container_by_lineno(cast(ast.Module, item.tree), lineno)
            assert func_nodes is not None 

            fn_node = func_nodes[-1]
            assert isinstance(fn_node, ast.FunctionDef), "Function node must be a FunctionDef"

            fn_name = fn_node.name
        self.state = self._get_state_from_code(path, item.content, fn_name, bkpts, fixed_inline_env)
        self._runner_task_ev = asyncio.Event()
        self._fixed_inline_env = fixed_inline_env

    @property 
    def grid_size(self):
        return self.state.runner.triton_sim_info.grid_size 

    @property 
    def runner(self):
        return self.state.runner 

    def setup_watchdog(self, handler: Callable[[_WATCHDOG_MODIFY_EVENT_TYPES], None], ignore_mtime: float) -> None:
        self.state.setup_watchdog(handler, ignore_mtime)

    def stop_watchdog(self) -> None:
        self.state.stop_watchdog()

    async def validate(self):
        await self.runner.copy().validate_kernel_by_test_data(self.state.fn, external_inline_env=self._fixed_inline_env)

    async def _run_triton_bench(self):
        _, _, compile_info = await self.runner.bench_kernel_in_triton_process(self.state.fn,
            ext_inline_env=self._fixed_inline_env)
        return compile_info

    async def run_triton_bench_sync(self):
        return await self._run_triton_bench()

    def close(self):
        if self._compiled_tmp_file_path is not None:
            # remove from linecache
            linecache.checkcache(self._compiled_tmp_file_path)
            self._compiled_tmp_file_path = None
        self.state.stop_watchdog()

    def _get_module_dict_init(self, path: str, code: Optional[str] = None):
        module_import_path = find_submodule_from_file(path)
        if module_import_path is None:
            assert code is not None 
            mod_dict = self._get_module_dict_by_code(path, code)
        else:
            mod_dict = importlib.import_module(module_import_path).__dict__
        fn_options: list[str] = []
        for k, v in mod_dict.items():
            v = tritonstd.may_triton_func(v)
            meta = pfl.get_compilable_meta(v)
            
            if meta is not None and isinstance(meta, tritonstd.TritonSimFuncMeta):
                if meta.inline_run_env_fn is not None:
                    fn_options.append(k)
        return mod_dict, fn_options

    def _get_module_dict_by_code(self, path: str, code: str):
        # use dynamic file import
        module = types.ModuleType(path)
        spec = importlib.machinery.ModuleSpec(path, None, origin=path)
        module.__spec__ = spec
        use_tmp_file: bool = False 
        if use_tmp_file:
            if self._compiled_tmp_file_path is not None:
                # remove from linecache
                linecache.checkcache(self._compiled_tmp_file_path)
            with NamedTemporaryFile(mode="w", suffix=".py", delete=True) as f:
                f.write(code)
                code_comp = compile(code, f.name, "exec")
                module.__file__ = f.name
                exec(code_comp, module.__dict__)
                self._compiled_tmp_file_path = f.name
        else:
            # assume user already save code to filesystem.
            linecache.checkcache(path)
            code_comp = compile(code, path, "exec")
            module.__file__ = path
            exec(code_comp, module.__dict__)

        mod_dict = module.__dict__
        return mod_dict

    def _get_state_from_code(self, path: str, code: str, fn_name: str, 
            bkpts: Optional[list[mui.MonacoBreakpoint]] = None,
            fixed_inline_env: Optional[TritonInlineRunEnv] = None):
        mod_dict, fn_names = self._get_module_dict_init(path, code)
        fn = mod_dict[fn_name]
        runner = tritonstd.parse_triton_compilable_to_runner(fn, do_meta_eval=True, 
            module_code_path_getter=partial(_temp_module_path_getter, path_editor=path, code_editor=code),
            external_inline_env=fixed_inline_env)
        lib = runner._library
        finder_dict: dict[str, pfl.PFLTreeNodeFinder] = {}
        expr_finder_dict: dict[str, pfl.PFLTreeExprFinder] = {}
        for k, v in lib.all_compiled.items():
            finder_dict[k] = pfl.PFLTreeNodeFinder(v, (pfl.PFLName, pfl.PFLAttribute, pfl.PFLArg)) 
            expr_finder_dict[k] = pfl.PFLTreeExprFinder(v)
        state = TritonKernelManagerState(
            fn=fn,
            module_dict=mod_dict,
            cur_fn_name=fn_name,
            fn_names=fn_names,
            path=path,
            content=code,
            content_lines=code.split("\n"),
            lib=lib,
            finder_dict=finder_dict,
            expr_finder_dict=expr_finder_dict,
            runner=runner,
            expr_trace_runner=runner.copy(),
            bkpts=bkpts,
        )
        if bkpts is not None:
            pfl_bkpt_desc_map: dict[tuple[str, int], PFLBreakpointDesc] = {}
            for bkpt in bkpts:
                pfl_bkpt_desc = PFLBreakpointDesc(bkpt.lineNumber, bkpt.enabled)
                pfl_bkpt_desc_map[(path, bkpt.lineNumber)] = pfl_bkpt_desc
            runner.sync_breakpoints(pfl_bkpt_desc_map)
        return state

    def recompile(self, new_code: str):
        self._runner_task_ev.clear()
        prev_state = self.state
        self.state = self._get_state_from_code(self.state.path, new_code, self.state.cur_fn_name, self.state.bkpts, self._fixed_inline_env)
        self.state.watchdog_observer = prev_state.watchdog_observer
        self.state.watchdog_watcher = prev_state.watchdog_watcher
        
    def run_to_use_task(self, grid_idxes: Sequence[int], lineno: int, thread_id: Optional[str] = None):
        unwrapped_fn = self.runner.get_unwrapped_triton_fn(self.state.fn)
        stmt = self.state.lib.find_stmt_by_path_lineno(self.state.lib.get_module_by_func(unwrapped_fn).compile_info.path, lineno)
        if stmt is not None:
            if self.runner.has_paused_thread():
                assert thread_id is not None, "Thread ID must be provided when runner is paused."
                # if paused, continue from the current position
                self.runner.continue_until(thread_id, self.state.path, lineno)
            else:
                self._runner_task_ev.clear()
                with tsim.enter_tensorsim_context(grid_idxes, self.runner.triton_sim_info.grid_size):
                    if self._fixed_inline_env is not None:
                        inline_env = self._fixed_inline_env
                    else:
                        inline_env = self.runner.get_triton_fn_inline_env(unwrapped_fn, tritonstd.TritonSimExecType.SIM)
                    # use data in inline_env to create tensor visualization.
                    func_node = self.runner._library.get_compiled_unit_specs(unwrapped_fn)[0]
                    self._runner_task = asyncio.create_task(self.runner.run_until(lineno, func_node.uid, 
                        exit_event=self._runner_task_ev, external_inline_env=inline_env))
                    return inline_env
        return None 

    async def run_expr_trace_kernel_test(self):
        return await self.state.expr_trace_runner.run_kernel_test(self.state.fn, external_inline_env=self._fixed_inline_env)

    async def run_single_block(self, grid_idxes: Sequence[int]):
        unwrapped_fn = self.runner.get_unwrapped_triton_fn(self.state.fn)
        assert not self.runner.is_running(), "Runner is running, cannot run single block."
        self._runner_task_ev.clear()
        with tsim.enter_tensorsim_context(grid_idxes, self.runner.triton_sim_info.grid_size):
            if self._fixed_inline_env is not None:
                inline_env = self._fixed_inline_env
            else:
                inline_env = self.runner.get_triton_fn_inline_env(unwrapped_fn, tritonstd.TritonSimExecType.SIM)
            # use data in inline_env to create tensor visualization.
            func_node = self.runner._library.get_compiled_unit_specs(unwrapped_fn)[0]
            self._runner_task = asyncio.create_task(self.runner.run_func(func_node.uid, 
                exit_event=self._runner_task_ev, external_inline_env=inline_env))
            return inline_env
        return None 

    def find_nearest_node_by_line_col(self, lineno: int, col_offset: int):
        for k, finder in self.state.finder_dict.items():
            res = finder.find_nearest_node_by_line_col(lineno, col_offset)
            if res is not None:
                return k, res
        return None, None 

    def find_expr_node_by_source_loc(self, sloc: pfl.SourceLocType):
        for k, finder in self.state.expr_finder_dict.items():
            res = finder.find_expr_by_source_loc(sloc)
            if res is not None:
                return k, res
        return None, None 

    async def stop_run(self):
        if self.runner.has_paused_thread():
            self.runner.release_all_breakpoint(stop=True)
            await self._runner_task_ev.wait()

    def get_cur_func_pfl_node(self):
        unwrapped_fn = self.runner.get_unwrapped_triton_fn(self.state.fn)
        func_node = self.runner._library.get_compiled_unit_specs(unwrapped_fn)[0]
        return func_node

class TritonCompiledViewer(mui.FlexBox):
    def __init__(self):
        self.editor = mui.MonacoEditor("", "c", "")
        self.editor.prop(readOnly=True, minWidth=0, minHeight=0)
        self.select = mui.Autocomplete("triton", [], self._on_select)
        self.select.prop(size="small", 
                textFieldProps=mui.TextFieldProps(
                            muiMargin="dense",
                            variant="outlined")
            )
        self._json_viewer = mui.JsonViewer()
        super().__init__([
            mui.VBox([
                mui.HBox([
                    self.select.prop(flex=1),
                    mui.Button("ptxas", self._on_ptxas_info).prop(size="small")
                ]),
                self.editor.prop(flex=1)

            ]).prop(flex=3),
            mui.HBox([
                self._json_viewer
            ]).prop(minWidth=0, minHeight=0, flex=1, overflow="auto"),
        ])
        self.prop(flexFlow="row nowrap", overflow="hidden")
        self._asm_dict: dict[str, str] = {}

    async def _on_ptxas_info(self):
        if self._asm_dict:
            with NamedTemporaryFile(suffix=".ptx") as f:
                # get target from ptx code
                # triton write target to ptx and use it to compile
                # instead of specifying it in ptxas.
                ptx_lines = self._asm_dict["ptx"].splitlines()
                arch = "sm_80"
                for l in ptx_lines:
                    if l.startswith(".target"):
                        arch = l.split()[1]
                        break
                f.write(self._asm_dict["ptx"].encode("utf-8"))
                out = subprocess.check_output(
                    ["ptxas", f.name, "-v", "--warn-on-spills", f"-arch={arch}"],
                    stderr=subprocess.STDOUT,
                    universal_newlines=True) 
                print(out)

    async def _on_select(self, option):
        content = self._asm_dict[option["label"]]
        await self.editor.write(content, path=option["path"])

    async def set_triton_compile_info(self, func_id: str, info: tritonstd.TritonKernelCompileInfo):
        asm_dict = info.asm
        self._asm_dict = asm_dict.copy()
        options = []
        for k, v in asm_dict.items():
            options.append({
                "label": k,
                "path": f"{func_id}-{k}",
            })
        await self.select.update_options(options, 0)
        await self.editor.write(asm_dict[options[0]["label"]], path=options[0]["path"])
        await self._json_viewer.write(info.metadata)



@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class TritonSimAppModel:
    path: Optional[str] = None 
    fn_name: Optional[str] = None
    lineno: Optional[int] = None
    bkpts: dict[str, list[mui.MonacoBreakpoint]] = dataclasses.field(default_factory=dict)
    fn_options: Annotated[list[Any], DraftFieldMeta(is_store_external=True)]  = dataclasses.field(default_factory=list)
    cur_fn_option: Annotated[Optional[Any], DraftFieldMeta(is_store_external=True)] = None
    is_external_kernel: bool = False 

class TritonSim:
    @mark_create_layout
    def my_layout(self):
        gmem_model = GlobalMemoryModel.empty()
        gmem_model.minimap.fit_mode = plus.hud.MinimapFitMode.AUTO
        self.dm = mui.DataModel(TritonSimModel(global_mem=gmem_model, expr_trace_matrices=ExprTraceMemoryModel.empty()), [])
        draft = self.dm.get_draft()
        self.app_dm = mui.DataModel(TritonSimAppModel(), [self.dm])
        app_draft = self.app_dm.get_draft()
        self.app_dm.connect_draft_store("_tensorpc_mls_triton_sim_v2", AppDraftFileStoreBackend())
        self.editor = mui.MonacoEditor("", "python", "")

        container = LocalMemContainerV2(draft.local_var_key, draft.local_mat)
        container.prop(border="1px solid blue")
        container.panel.event_plane.event_move.add_frontend_handler(self.dm, TritonSimModel._on_hover_pfl_single)
        container.panel.event_plane.event_leave.add_frontend_handler(self.dm, TritonSimModel._on_hover_leave_pfl_single)
        local_container = mui.MatchCase.binary_selection(True, container)
        local_container.bind_fields(condition=f"{draft.local_var_key} != \"\"")

        self.tree = plus.ObjectInspector(with_builtins=False, show_terminal=False, default_tab_preview=False, default_sizes=[100, 100],
            init_fast_layout=[local_container])
        self.io_ops_tree = mui.TanstackJsonLikeTree().prop(ignoreRoot=True)
        self.io_ops_tree.event_select.on(self._handle_io_tree_select)
        self._cur_recorded_io_ops: list[TensorSimIoOp] = []
        self.editor.update_raw_props({
            ".monaco-editor-content-decoration": {
                "background": "lightblue"
            }
        })
        self.app_dm.event_storage_fetched.on(self._handle_app_dm_storage_fetched)
        editor_acts: list[mui.MonacoEditorAction] = [
            mui.MonacoEditorAction(id=EditorActions.RUN_TO.value, 
                label="Run Towards Here", contextMenuOrder=1.5,
                contextMenuGroupId="tensorpc-pfl-editor-action", 
                keybindings=[([mui.MonacoKeyMod.Shift], 3)]),
            mui.MonacoEditorAction(id=EditorActions.EXPR_TRACE.value, 
                label="Trace Selected Expr", contextMenuOrder=1.5,
                contextMenuGroupId="tensorpc-pfl-editor-action"),
        ]
        self._triton_viewer = TritonCompiledViewer()
        self._ptx_viewer_dialog = mui.Dialog([
            self._triton_viewer.prop(flex=1),

        ]).prop(dialogMaxWidth=False, fullWidth=False,
            width="75vw", height="75vh", display="flex")

        debug_toolbar = mui.HBox([
            mui.IconButton(mui.IconType.PlayArrow, self._on_debug_just_run)
                .prop(tooltip="Run Single Block", size="small", iconSize="small", muiColor="primary")
                .bind_fields(disabled=draft.cur_thread_option != None),
            mui.IconButton(mui.IconType.KeyboardArrowRight, self._on_debug_next_line)
                .prop(tooltip="Next Line", size="small", iconSize="small", muiColor="primary")
                .bind_fields(disabled=f"(not {draft.cur_thread_option.is_paused}) if {draft.cur_thread_option} else True"),
            mui.IconButton(mui.IconType.KeyboardDoubleArrowRight, self._on_debug_continue)
                .prop(tooltip="Continue", size="small", iconSize="small", muiColor="primary")
                .bind_fields(disabled=f"(not {draft.cur_thread_option.is_paused}) if {draft.cur_thread_option} else True"),
            mui.IconButton(mui.IconType.RestartAlt,)
                .prop(tooltip="Restart", size="small", iconSize="small", muiColor="success")
                .bind_fields(disabled=f"(not {draft.cur_thread_option.is_paused}) if {draft.cur_thread_option} else True"),
            mui.IconButton(mui.IconType.Stop, self._on_debug_stop)
                .prop(tooltip="Stop", size="small", iconSize="small", muiColor="error")
                .bind_fields(disabled=f"(not {draft.cur_thread_option.is_paused}) if {draft.cur_thread_option} else True"),
            mui.HBox([]).prop(flex=1),
            mui.IconButton(mui.IconType.QueryStats, self._on_launch_triton)
                .prop(tooltip="Run Triton Kernel", size="small", iconSize="small", muiColor="error", 
                      progressColor="primary", progressSize=28),
            mui.IconButton(mui.IconType.DataObject, lambda: self._ptx_viewer_dialog.set_open(True))
                .prop(tooltip="Check Triton Viewer", size="small", iconSize="small", muiColor="error", 
                      progressColor="primary"),
        ])
        self.editor.prop(minWidth=0, minHeight=0, actions=editor_acts, options=mui.MonacoEditorOptions(glyphMargin=True))
        self.editor.event_editor_hover_query.on(self.hover_query)
        self.editor.event_editor_action.on(self._handle_editor_acts)
        self.editor.event_editor_inlay_hints_query.on(self.inlay_hint_query)
        self.editor.event_editor_cursor_selection.on(self._handle_editor_cursor_selection)
        self.editor.event_editor_save.on(self._handle_editor_save)
        self.editor.event_change.on(self._handle_editor_debounced_change)
        self.editor.event_editor_breakpoint_change.on(self._handle_editor_bkpt_change)
        self._runner: Optional[TritonKernelManager] = None
        self._global_mem = GlobalMemContainer(external_dm=self.dm, external_draft=draft.global_mem, use_view=True)
        self._ast_cache = AstCache()
        self._editor_lock = asyncio.Lock()
        appctx.use_app_special_event_handler(self.tree, AppSpecialEventType.VscodeTensorpcMessage, self._handle_vscode_message)
        # gpu grid sliders
        x_slider = mui.BlenderSlider(0, 0, 1).prop(showTotal=True, isInteger=True, infSlider=False, showControlButton=True)
        y_slider = mui.BlenderSlider(0, 0, 1).prop(showTotal=True, isInteger=True, infSlider=False, showControlButton=True)
        z_slider = mui.BlenderSlider(0, 0, 1).prop(showTotal=True, isInteger=True, infSlider=False, showControlButton=True)
        x_slider.bind_fields(ranges=draft.grid_size_x_range)
        y_slider.bind_fields(ranges=draft.grid_size_y_range)
        z_slider.bind_fields(ranges=draft.grid_size_z_range)
        z_slider.bind_draft_change(draft.grid_idx_z)
        y_slider.bind_draft_change(draft.grid_idx_y)
        x_slider.bind_draft_change(draft.grid_idx_x)
        self.y_slider = y_slider
        self.x_slider = x_slider
        self.z_slider = z_slider
        x_slider.event_change.on(partial(self._handle_slider_change, axis=0))
        y_slider.event_change.on(partial(self._handle_slider_change, axis=1))
        z_slider.event_change.on(partial(self._handle_slider_change, axis=2))

        self._block_run_backend_state: Optional[SingleBlockRunState] = None
        self._cur_observed_local_tensor_key: Optional[tuple[str, str]] = None
        self._cur_traced_expr: Optional[tuple[pfl.SourceLocType, str, list[str]]] = None

        self._next_bkpt_set_lineno_tid: Optional[str] = None

        self._watchdog_lock = threading.Lock()
        # some network based fs modify event may trigger multiple times
        # during write, so we need to debounce it to avoid incomplete write.
        self._fs_event_debounce = 0.1
        self._kernel_select = mui.Autocomplete("kernel", [])
        self._kernel_select.prop(size="small", 
                                textFieldProps=mui.TextFieldProps(
                                                muiMargin="dense",
                                                variant="outlined")
                                )
        self._kernel_select.bind_fields(options=self.app_dm.get_draft().fn_options, disabled=draft.cur_thread_option != None)
        self._kernel_select.bind_draft_change(self.app_dm.get_draft().cur_fn_option)
        self._kernel_select.event_change.on(self._handle_kernel_select_change)
        self._thread_select = mui.Autocomplete("thread", [])
        self._thread_select.prop(size="small", 
                                textFieldProps=mui.TextFieldProps(
                                                muiMargin="dense",
                                                variant="outlined")
                                )
        self._thread_select.bind_fields(options=draft.thread_options)
        self._thread_select.bind_draft_change(draft.cur_thread_option)
        self._thread_select.event_change.on(self._handle_thread_select_change)

        matrix_table = MatrixTableContainer(draft.expr_trace_matrices)
        matrix_table.panel.event_plane.event_move.add_frontend_handler(self.dm, TritonSimModel._on_matrix_table_elem_hover_pfl)
        matrix_table.panel.event_plane.event_leave.add_frontend_handler(self.dm, TritonSimModel._on_matrix_table_elem_hover_leave_pfl)
        matrix_table.panel.event_plane.event_enter.add_frontend_handler(self.dm, TritonSimModel._on_matrix_table_elem_enter_pfl)
        matrix_table.panel.event_plane.event_click.add_frontend_handler(self.dm, TritonSimModel._on_matrix_table_elem_click_pfl)
        matrix_table.panel.event_plane.event_click.configure(stop_propagation=True)
        matrix_table.minimap.event_plane.event_click.add_frontend_handler(self.dm, TritonSimModel._on_matrix_table_bkgd_click)

        self.dm.init_add_layout([
            mui.VBox([
                mui.DataPortal(self.app_dm, [
                    self._kernel_select,
                ]),
                self._thread_select,
                mui.HDivider(),
                debug_toolbar,
                mui.HDivider(),
                x_slider,
                y_slider, 
                z_slider,
                mui.HDivider(),
                self.tree.prop(flex=2),
                mui.HDivider(),
                self.io_ops_tree.prop(flex=1),
            ]).prop(flex=1),
            mui.VDivider(),
            mui.VBox([
                # mui.ThemeProvider([
                #     tabs
                # ], tab_theme),
                self._ptx_viewer_dialog,
                self.editor.prop(flex=3),
                mui.HDivider(),
                mui.AppTerminal().prop(flex=1),
            ]).prop(flex=2, minWidth=0, minHeight=0, overflow="hidden"),
            mui.VDivider(),
            mui.VBox([
                mui.HBox([self._global_mem]).prop(flex=1),
                mui.HBox([matrix_table]).prop(flex=1),
            ]).prop(flex=2)
        ])
        return mui.VBox([
            three.ViewCanvas([
                self.app_dm,
            ]).prop(display="flex", flexFlow="row nowrap", flex=1, overflow="hidden"),
        ]).prop(width="100%", height="100%", overflow="hidden", minWidth=0, minHeight=0)

    async def _handle_app_dm_storage_fetched(self, prev_model: TritonSimAppModel):
        # fetch from storage
        model = self.app_dm.model
        if model.path is not None and model.fn_name is not None and model.lineno is not None and not model.is_external_kernel:
            try:
                runner = TritonKernelManager(
                    self._ast_cache.query_path(Path(model.path)),
                    model.path,
                    model.lineno,
                    model.fn_name
                )
                async with self.app_dm.draft_update() as draft:
                    draft.fn_options = [{"label": fn_name} for fn_name in runner.state.fn_names]
                    draft.cur_fn_option = {
                        "label": runner.state.cur_fn_name,
                    }
            except:
                async with self.app_dm.draft_update() as draft:
                    draft.path = None
                    draft.fn_name = None
                    draft.lineno = None
                    draft.fn_options = []
                    draft.cur_fn_option = None
                raise 
            await self._init_new_runner(
                runner,
                model.lineno
            )
            if model.bkpts is not None and model.path in model.bkpts:
                bkpts = model.bkpts[model.path]
                await self.editor.send_and_wait(self.editor.update_event(bkpts=bkpts))

    async def _on_debug_continue(self):
        if self._runner is None:
            return
        self._validate_editor_has_unsave()
        if self.dm.model.cur_thread_option is not None:
            cur_thread = self.dm.model.cur_thread_option
            self._runner.runner.release_breakpoint(cur_thread.label)
            self._next_bkpt_set_lineno_tid = cur_thread.label

    async def _on_debug_next_line(self):
        if self._runner is None:
            return
        self._validate_editor_has_unsave()
        if self.dm.model.cur_thread_option is not None:
            cur_thread = self.dm.model.cur_thread_option
            self._runner.runner.release_breakpoint(cur_thread.label)

            assert self._runner.runner.is_thread_paused(cur_thread.label), "Runner must be paused to continue to next line"
            self._runner.runner.continue_next_line(cur_thread.label)
            self._next_bkpt_set_lineno_tid = cur_thread.label


    async def _on_debug_stop(self):
        if self._runner is None:
            return
        await self._runner.stop_run()

    async def _on_debug_just_run(self):
        if self._runner is None:
            return
        self._validate_editor_has_unsave()
        grid_idxes = [self.x_slider.int(), self.y_slider.int(), self.z_slider.int()]

        await self._runner.run_single_block(grid_idxes)

    async def _init_sim_info(self):
        assert self._runner is not None 
        async with self.dm.draft_update() as draft:
            draft.grid_size_x_range = (0, self._runner.grid_size[0] - 1, 1)
            draft.grid_size_y_range = (0, self._runner.grid_size[1] - 1, 1)
            draft.grid_size_z_range = (0, self._runner.grid_size[2] - 1, 1)
            draft.grid_idx_x = min(self.dm.model.grid_idx_x, self._runner.grid_size[0] - 1)
            draft.grid_idx_y = min(self.dm.model.grid_idx_y, self._runner.grid_size[1] - 1)
            draft.grid_idx_z = min(self.dm.model.grid_idx_z, self._runner.grid_size[2] - 1)
        sim_info = self._runner.runner.triton_sim_info
        gmem = sim_info.global_mem
        assert gmem is not None 
        mat_dict: dict[str, np.ndarray] = {}
        for k, block in gmem.memory_blocks.items():
            mat_dict[k] = block.get_data_view_checked()
        await self._global_mem.set_matrix_dict(mat_dict, sim_info.vis_layout)

    async def _handle_editor_save(self, ev: mui.MonacoSaveEvent):
        assert self._runner is not None, "Runner must be initialized before saving"
        prev_is_paused = self._runner.runner.is_paused()
        new_bkpt_lineno = None 
        if prev_is_paused:
            decors = ev.decorationsRanges
            if decors is not None:
                common_ranges = decors["common"]
                assert len(common_ranges) == 1
                new_bkpt_lineno = common_ranges[0].startLineNumber
        return await self._recompile(ev.value, bkpt_lineno=new_bkpt_lineno)

    def _handle_external_file_change(self, event: _WATCHDOG_MODIFY_EVENT_TYPES, loop: asyncio.AbstractEventLoop):
        if self._runner is None:
            return 
        if isinstance(event, watchdog.events.FileModifiedEvent):
            with self._watchdog_lock:
                if isinstance(event.src_path, bytes):
                    src_path = event.src_path.decode()
                else:
                    src_path = cast(str, event.src_path)
                content = Path(src_path).read_text(encoding="utf-8")
                try:
                    fut = asyncio.run_coroutine_threadsafe(
                        self.editor.write(content),
                        loop
                    )
                    fut.result()
                    fut = asyncio.run_coroutine_threadsafe(
                        self._recompile(content, write_to_file=False),
                        loop
                    )
                    fut.result()
                except BaseException as e:
                    traceback.print_exc()

    async def _reset_vis_matrics(self):
        async with self.dm.draft_update() as draft:
            draft.expr_trace_matrices.matrices.clear()
            draft.expr_trace_matrices.labels.clear()
            draft.expr_trace_matrices.title = "expr trace"
            draft.global_mem.matrices.clear()

    async def _recompile(self, new_content: str, bkpt_lineno: Optional[int] = None, write_to_file: bool = True):
        assert self._runner is not None, "Runner must be initialized before saving"
        main_thread = self._runner.runner.get_main_thread()
        prev_is_paused = main_thread is not None and main_thread.is_paused()
        if main_thread is not None and prev_is_paused:
            if bkpt_lineno is None:
                bkpt_lineno = main_thread.get_cur_bkpt_checked().node.source_loc[0]

        # print(0)
        await self._on_debug_stop()
        # print(1)
        # always write to file even if bug.
        if write_to_file:
            self._runner.stop_watchdog()
            with open(self._runner.state.path, "w", encoding="utf-8") as f:
                f.write(new_content)
        self._runner.recompile(new_content)
        if write_to_file:
            file_cur_mtime = Path(self._runner.state.path).stat().st_mtime
            self._runner.setup_watchdog(partial(self._handle_external_file_change, loop=asyncio.get_running_loop()), file_cur_mtime)

        await self._runner.validate()
        await self._reset_vis_matrics()
        await self._init_sim_info()

        self._install_event_handlers_to_runner(self._runner)
        if prev_is_paused and bkpt_lineno is not None:
            cur_grid_idxes = [self.x_slider.int(), self.y_slider.int(), self.z_slider.int()]
            for j in range(len(cur_grid_idxes)):
                cur_grid_idxes[j] = max(cur_grid_idxes[j], 0)
                cur_grid_idxes[j] = min(cur_grid_idxes[j], self._runner.grid_size[j] - 1)
            self._runner.run_to_use_task(cur_grid_idxes, bkpt_lineno)

        if self._cur_traced_expr is not None:
            # rerun trace
            sloc, key, old_lines = self._cur_traced_expr
            # try to match the lineno of previous traced expr
            mapper = SourceChangeDiffCache.get_raw_item_for_mapping(old_lines, self._runner.state.content_lines)
            mapped_lineno = mapper.bisect_mapped_lineno(sloc[0])
            if mapped_lineno == -1:
                print("Cannot map the line number to previous traced expr")
                self._cur_traced_expr = None
            # here we don't use end lineno because we shouldn't trace multiple lines expr
            await self._expr_trace(key, (mapped_lineno, sloc[1], mapped_lineno, sloc[3]))
            
    async def _handle_kernel_select_change(self, option):
        new_fn_name = option["label"]
        assert self._runner is not None, "Runner must be initialized before saving"
        await self._on_debug_stop()
        self._runner.state.switch_lib_in_same_file(new_fn_name)
        self._install_event_handlers_to_runner(self._runner)
        await self._runner.validate()
        lineno = self._runner.get_cur_func_pfl_node().source_loc[0]
        await self.editor.set_line_number(lineno)

        await self._init_sim_info()
        async with self.app_dm.draft_update() as draft:
            draft.path = self._runner.state.path
            draft.fn_name = self._runner.state.cur_fn_name
            draft.lineno = lineno


    def _install_event_handlers_to_runner(self, runner: TritonKernelManager):
        runner.runner.event_run_start.on(self._handle_eval_start)
        runner.runner.event_run_stop.on(self._handle_eval_stop)
        runner.runner.event_thread_enter_bkpt.on(self._handle_enter_bkpt)
        runner.runner.event_thread_leave_bkpt.on(self._handle_leave_bkpt)
        runner.runner.event_thread_changed.on(self._handle_thread_changed)

    async def _handle_thread_changed(self, new_threads: dict[str, PFLAsyncThread]):
        await self._sync_threads_status()

    async def launch_external_sim(self, path: str, lineno: int, inline_env: TritonInlineRunEnv):
        item = self._ast_cache.query_path(Path(path))
        runner = TritonKernelManager(item, path, lineno, fixed_inline_env=inline_env)
        async with self.app_dm.draft_update() as draft:
            draft.path = path
            draft.fn_name = runner.state.cur_fn_name
            draft.lineno = lineno
            draft.fn_options = [{"label": fn_name} for fn_name in runner.state.fn_names]
            draft.cur_fn_option = {
                "label": runner.state.cur_fn_name,
            }
            draft.is_external_kernel = True
        await self._init_new_runner(runner, lineno)

    async def _handle_vscode_message(self, data: VscodeTensorpcMessage):
        if data.type == VscodeTensorpcMessageType.PFLLaunchSimulation:
            if self._runner is not None:
                await self._runner.stop_run()
                self._runner.close()
            assert data.selections is not None 
            # vscode.Selection use zero-based line numbers and col offset
            # monaco.Selection use 1-based for both. oh my god
            lineno = data.selections[0].start.line + 1 
            path = data.currentUri
            assert path.startswith("file://"), "Current URI must be a file URI"
            path = path[7:]
            item = self._ast_cache.query_path(Path(path))
            runner = TritonKernelManager(item, path, lineno)

            async with self.app_dm.draft_update() as draft:
                draft.path = path
                draft.fn_name = runner.state.cur_fn_name
                draft.lineno = lineno
                draft.fn_options = [{"label": fn_name} for fn_name in runner.state.fn_names]
                draft.cur_fn_option = {
                    "label": runner.state.cur_fn_name,
                }
                draft.is_external_kernel = False
            await self._init_new_runner(runner, lineno)
        elif data.type == VscodeTensorpcMessageType.PFLRunExprTrace:
            if self._runner is None:
                return

            assert data.selections is not None 
            lineno = data.selections[0].start.line + 1 
            end_lineno = data.selections[0].end.line + 1
            col_start = data.selections[0].start.character
            if end_lineno != lineno:
                print("Cannot trace multi-line expression")
                return None
            col_end = data.selections[0].end.character
            func_uid, expr_node = self._runner.find_expr_node_by_source_loc((lineno, col_start, end_lineno, col_end))
            if expr_node is None or func_uid is None:
                print(f"Cannot find expression at line {lineno}, column {col_start + 1}, {data.selections[0]}")
                return None
            fuid_name = self._remove_spec_suffix_of_func_uid(func_uid)
            fuid_name_qname = "::".join(fuid_name.split("::")[1:])
            key = f"{fuid_name_qname}-{unparse_pfl_ast(expr_node)}"
            await self._expr_trace(key, expr_node.source_loc)
            # await self.editor.set_line_number(lineno)

    async def _init_new_runner(self, runner: TritonKernelManager, lineno: int):
        self._runner = runner
        # do simple validate in each recompile 
        # use copy to avoid trigger ui events
        await self._runner.validate()

        self._install_event_handlers_to_runner(self._runner)
        await self.editor.write(runner.state.content, runner.state.path, "python")
        await self.editor.set_line_number(lineno)
        self._cur_observed_local_tensor_key = None
        await self._init_sim_info()
        if runner.state.path in self.app_dm.model.bkpts:
            bkpts = self.app_dm.model.bkpts[runner.state.path]
            bkpts_desc = self._get_sim_bkpt_descs(runner.state.path, bkpts)
            runner.runner.sync_breakpoints(bkpts_desc)
        runner.setup_watchdog(partial(self._handle_external_file_change, loop=asyncio.get_running_loop()), -1)

    async def _handle_slider_change(self, value: Any, axis: int):
        if self._runner is None:
            return 
        self._validate_editor_has_unsave()
        old_value = [self.x_slider.int(), self.y_slider.int(), self.z_slider.int()]
        old_value[axis] = value
        if self._runner.runner.has_paused_thread():
            cur_bkpt_lineno = None
            cur_thread_opt = self.dm.model.cur_thread_option
            if cur_thread_opt is not None and cur_thread_opt.is_paused:
                cur_thread = self._runner.runner.get_thread(cur_thread_opt.label)
                if cur_thread.is_main_thread:
                    cur_bkpt_lineno = cur_thread.get_cur_bkpt_checked().node.source_loc[0]
            await self._runner.stop_run()
            if cur_bkpt_lineno is not None:
                self._runner.run_to_use_task(old_value, cur_bkpt_lineno)
            else:
                await self._runner.run_single_block(old_value)
        else:
            await self._runner.run_single_block(old_value)

    def _set_local_memory_model_draft(self, draft: LocalMatrix, ten: tritonstd.Tensor):
        storage = ten._wrapped.get_storage_checked()
        data = storage.data
        if data.dtype == np.bool_:
            # dont support bool in frontend
            data = data.astype(np.uint8)
        elif data.dtype == np.int64 or data.dtype == np.uint64:
            # dont support bigint (64bit int) in frontend
            data = data.astype(np.int32 if data.dtype == np.int64 else np.uint32)
        draft.data = data
        draft.global_indices = storage.indices
        local_pos, local_color, mask_pos = Matrix.get_value_pos_and_color_gray(storage.data)
        draft.persist_fill_pos = local_pos
        draft.persist_fill_color = local_color
        draft.temp_mask_pos = mask_pos
        return storage

    def _bkpt_handle_local_tensor_panel_local(self, user_func_uid_no_suffix: str, k: str, draft: TritonSimModel, bkpt: PFLRunnerBreakpoint):
        for stack in bkpt.stack[::-1]:
            func_uid_no_suffix = self._remove_spec_suffix_of_func_uid(stack.node.uid)
            if k in stack.scope and func_uid_no_suffix == user_func_uid_no_suffix:
                local_key = f"{func_uid_no_suffix}-{k}"
                ten = bkpt.scope[k]
                assert isinstance(ten, (tritonstd.Tensor, tritonstd.PointerTensor))
                if isinstance(ten, tritonstd.Tensor):
                    for global_key in self.dm.model.global_mem.matrices.keys():
                        draft.global_mem.matrices[global_key].temp_fill_pos = None 
                        draft.global_mem.matrices[global_key].temp_fill_color = None
                    storage = self._set_local_memory_model_draft(draft.local_mat.matrix, ten)
                    indices_dict = storage.indices
                    for global_key, inds in indices_dict.items():
                        inds_no_invalid = inds[inds != -1]
                        fill_pos, fill_color = self.dm.model.get_global_fill(global_key, inds_no_invalid, is_persist=False)
                        # draft.global_mem.matrices[global_key].temp_mask_pos = fill_pos
                        draft.global_mem.matrices[global_key].temp_fill_pos = fill_pos
                        draft.global_mem.matrices[global_key].temp_fill_color = fill_color
                    break
                else:
                    # TODO
                    raise NotImplementedError

    async def _handle_io_tree_select(self, selected: dict[str, bool]):
        for k, v in selected.items():
            if not v:
                continue
            id_obj = UniqueTreeIdForTree(k).parts[0]
            idx = int(id_obj)
            op = self._cur_recorded_io_ops[idx]
            fill_pos, fill_color = self.dm.model.get_global_fill(op.name, op.io_indices)
            async with self.dm.draft_update() as draft:
                for global_key in self.dm.model.global_mem.matrices.keys():
                    draft.global_mem.matrices[global_key].temp_mask_pos = None

                draft.global_mem.matrices[op.name].temp_mask_pos = fill_pos
            await self.editor.set_line_number(op.ast_node.source_loc[0], select_line=True)

    async def _recorded_io_ops_to_global(self, io_ops: list[TensorSimIoOp]):
        # add persist memory access data to global matrix.
        async with self.dm.draft_update() as draft:
            updated_keys: set[str] = set()
            assert self._block_run_backend_state is not None 
            for op in io_ops:
                # accumulate and unique the indices
                msg = f"Global memory {op.name} not found in backend state {self._block_run_backend_state.global_access_indices.keys()}"
                assert op.name in self._block_run_backend_state.global_access_indices, msg
                old_inds = self._block_run_backend_state.global_access_indices[op.name]
                cnt = self._block_run_backend_state.global_access_cnt[op.name]

                old_advs = self._block_run_backend_state.global_access_advances[op.name]
                new_inds, uniq_idxes = np.unique(np.concatenate([old_inds, op.io_indices.reshape(-1)]), return_inverse=True)
                new_advs = np.full_like(new_inds, (cnt % 4) / 4, dtype=np.float32)
                new_advs[uniq_idxes[:len(old_inds)]] = old_advs
                # new_inds = np.concatenate([old_inds, op.io_indices.reshape(-1)])
                # new_advs = np.concatenate([old_advs, np.full_like(op.io_indices.reshape(-1), (cnt % 4) / 4, dtype=np.float32)])
                self._block_run_backend_state.global_access_indices[op.name] = new_inds
                self._block_run_backend_state.global_access_advances[op.name] = new_advs
                self._block_run_backend_state.global_access_cnt[op.name] += 1

                updated_keys.add(op.name)
            for k in updated_keys:
                inds = self._block_run_backend_state.global_access_indices[k]
                advs = self._block_run_backend_state.global_access_advances[k]
                fill_pos, fill_color = self.dm.model.get_global_fill(k, inds, is_persist=True, color_advance=advs)
                draft.global_mem.matrices[k].persist_fill_pos = fill_pos
                draft.global_mem.matrices[k].persist_fill_color = fill_color


    async def _bkpt_handle_recorded_io_ops(self, io_ops: list[TensorSimIoOp]):
        items: list[mui.JsonLikeNode] = []
        old_length = len(self._cur_recorded_io_ops)
        # old ops
        for i, op in enumerate(self._cur_recorded_io_ops + io_ops):
            is_old = i < old_length
            name = f"{op.name}"
            if not is_old:
                name = "+" + name
            if op.matrix_info is not None:
                s0 = op.matrix_info.offsets[0]
                s1 = op.matrix_info.offsets[1]
                e0 = s0 + op.matrix_info.shape[0]
                e1 = s1 + op.matrix_info.shape[1]
                name += f"[{s0}:{e0}, {s1}:{e1}]"
            node = mui.JsonLikeNode(
                id=UniqueTreeIdForTree.from_parts([str(i)]),
                name=name,
                type=mui.JsonLikeType.Object.value,
                typeStr="Load" if op.is_load else "Store",
                value=str(op.shape)
            ) 
            items.append(node)
        self._cur_recorded_io_ops.extend(io_ops)
        dummy_node = mui.JsonLikeNode.create_dummy()
        dummy_node.children = items
        await self.io_ops_tree.send_and_wait(self.io_ops_tree.update_event(tree=dummy_node, ignoreRoot=True))
        await self._recorded_io_ops_to_global(io_ops)

    def _sync_threads_status_draft(self, draft: TritonSimModel, select_thread_id: Optional[str] = None):
        if self._runner is None:
            return 
        all_threads = self._runner.runner.get_all_threads()
        cur_model = self.dm.model
        if not all_threads:
            draft.thread_options = []
            draft.cur_thread_option = None
        else:
            prev_cur_option = cur_model.cur_thread_option
            if prev_cur_option is not None:
                if prev_cur_option.label in all_threads:
                    prev_thread = all_threads[prev_cur_option.label]
                    draft.cur_thread_option.is_paused = prev_thread.is_paused()
                else:
                    one_thread = next(iter(all_threads.values()))
                    draft.cur_thread_option = TritonSimThreadStateModel(
                        label=one_thread.thread_id,
                        is_paused=one_thread.is_paused()
                    )
            else:
                one_thread = next(iter(all_threads.values()))
                draft.cur_thread_option = TritonSimThreadStateModel(
                    label=one_thread.thread_id,
                    is_paused=one_thread.is_paused()
                )
            prev_opts = {t.label: t for t in cur_model.thread_options}
            if select_thread_id is not None:
                selected_thread = all_threads[select_thread_id]
                if select_thread_id in prev_opts:
                    draft.cur_thread_option = dataclasses.replace(prev_opts[select_thread_id], is_paused=selected_thread.is_paused())
                else:
                    draft.cur_thread_option = TritonSimThreadStateModel(
                        label=selected_thread.thread_id,
                        is_paused=selected_thread.is_paused()
                    )
            # draft.thread_options.clear()
            draft.thread_options = []
            for k, t in all_threads.items():
                if k in prev_opts:
                    new_t = dataclasses.replace(prev_opts[k], is_paused=t.is_paused())
                else:
                    new_t = TritonSimThreadStateModel(
                        label=k,
                        is_paused=t.is_paused()
                    )
                draft.thread_options.append(new_t)

    async def _sync_threads_status(self, select_thread_id: Optional[str] = None):
        if self._runner is None:
            return 
        async with self.dm.draft_update() as draft:
            self._sync_threads_status_draft(draft, select_thread_id=select_thread_id)


    async def _switch_to_thread(self, thread: PFLAsyncThread):
        await self._sync_threads_status(select_thread_id=thread.thread_id)

        if thread.is_paused():
            await self._update_thread_breakpoint(thread)

    async def _handle_thread_select_change(self, option):
        thread_id = option["label"]
        if self._runner is None:
            return 
        thread = self._runner.runner.get_thread(thread_id)
        await self._switch_to_thread(thread)

    async def _update_breakpoint_and_scope(self, bkpt: PFLRunnerBreakpoint):
        await self.tree.tree.set_root_object_dict(bkpt.scope)
        await self.editor.set_decorations("common", [
            mui.MonacoModelDeltaDecoration(mui.MonacoRange(bkpt.node.source_loc[0], 1, bkpt.node.source_loc[0], 1),
            mui.MonacoModelDecoration(className="monaco-editor-content-decoration", isWholeLine=True,
            minimap=mui.MonacoModelDecorationMinimapOptions(mui.MonacoMinimapPosition.Inline)))
        ])

    async def _update_editor_inline_comps(self, thread: PFLAsyncThread, clear_previous: bool = False):
        if clear_previous:
            self.editor.childs_complex.icomps.clear()
        cur_ctrl_points = thread.get_state().cur_ctrl_points
        for ctrl_id, ctrl in cur_ctrl_points.items():
            assert isinstance(ctrl, PFLRunnerCtrlFor), "Only PFLRunnerCtrlFor is supported in this editor"
            key = get_key_from_prefix_data(InlineCompPrefix.CONTROLS, str(ctrl_id))
            if key not in self.editor.childs_complex.icomps:
                node = ctrl.node
                slider = mui.BlenderSlider(ctrl.range.start, ctrl.range.stop, ctrl.range.step)
                slider.event_change.on(partial(self._handle_slider, slider=slider, ctrl=ctrl))
                slider.prop(width="50%", showControlButton=True, showTotal=True, showStep=True, 
                    isInteger=True, forwardOnly=True, zIndex=10, alwaysShowButton=True)
                inline_comp = mui.MonacoEditor.InlineComponent(
                    slider,
                    afterLineNumber=node.source_loc[0], heightInPx=24)
                self.editor.childs_complex.icomps[key] = inline_comp
            else:
                slider = self.editor.childs_complex.icomps[key].comp
                assert isinstance(slider, mui.BlenderSlider)
                await slider.send_and_wait(slider.update_event(disabled=False))
                await slider.update_value(ctrl.step)
        all_keys = list(self.editor.childs_complex.icomps.keys())
        for k in all_keys:
            prefix, data = get_prefix_data_from_key(k)
            if prefix == InlineCompPrefix.CONTROLS:
                if int(data) not in cur_ctrl_points:
                    # remove the slider if the ctrl point is not in the current state
                    self.editor.childs_complex.icomps.pop(k, None)
        await self.editor.set_new_layout(self.editor.childs_complex)

    async def _update_thread_breakpoint(self, thread: PFLAsyncThread):
        bkpt = thread.get_cur_bkpt_checked()
        await self._update_breakpoint_and_scope(bkpt)
        async with self._editor_lock:
            await self._update_editor_inline_comps(thread, clear_previous=True)
        cur_thread_info = self.dm.model.cur_thread_option
        if cur_thread_info is not None and cur_thread_info.cur_local_tensor_id is not None:
            func_uid, obj_key = cur_thread_info.cur_local_tensor_id
            for stack in bkpt.stack:
                stack_uid_no_suffix = self._remove_spec_suffix_of_func_uid(stack.node.uid)
                func_uid_no_suffix = self._remove_spec_suffix_of_func_uid(func_uid)
                if stack_uid_no_suffix == func_uid_no_suffix and obj_key in stack.scope:
                    obj = stack.scope[obj_key]
                    if isinstance(obj, tritonstd.Tensor):
                        await self._create_local_tensor_panel(func_uid, obj_key, obj.shape)

    async def _handle_enter_bkpt(self, thread_id: str, bkpt: PFLRunnerBreakpoint):
        if self._runner is None:
            return 
        await self._sync_threads_status()
        cur_thread_opt = self.dm.model.cur_thread_option
        if cur_thread_opt is not None and cur_thread_opt.label != thread_id:
            return 
        if self._next_bkpt_set_lineno_tid is not None and self._next_bkpt_set_lineno_tid == thread_id:
            await self.editor.set_line_number(bkpt.node.source_loc[0])
            self._next_bkpt_set_lineno_tid = None

        await self._update_thread_breakpoint(self._runner.runner.get_thread(thread_id))
        tsim_io_ops = get_flush_sim_io_ops()
        await self._bkpt_handle_recorded_io_ops(tsim_io_ops)

    async def _handle_eval_start(self):
        await self.io_ops_tree.clear()
        self._cur_recorded_io_ops.clear()
        global_access_indices: dict[str, np.ndarray] = {}
        global_access_advances: dict[str, np.ndarray] = {}
        global_access_cnt: dict[str, int] = {}
        for global_key, mat in self.dm.model.global_mem.matrices.items():
            global_access_indices[global_key] = np.empty([0], dtype=np.int32)
            global_access_advances[global_key] = np.empty([0], dtype=np.float32)
            global_access_cnt[global_key] = 0

        self._block_run_backend_state = SingleBlockRunState(global_access_indices=global_access_indices,
            global_access_advances=global_access_advances, global_access_cnt=global_access_cnt)
        async with self.dm.draft_update() as draft:
            draft.cur_thread_option = None
            draft.thread_options = []
            draft.local_var_key = ""
            for global_key in self.dm.model.global_mem.matrices.keys():
                draft.global_mem.matrices[global_key].temp_fill_color = None
                draft.global_mem.matrices[global_key].temp_fill_pos = None
                draft.global_mem.matrices[global_key].persist_fill_color = None
                draft.global_mem.matrices[global_key].persist_fill_pos = None
                draft.global_mem.matrices[global_key].temp_mask_pos = None

    async def _handle_eval_stop(self):
        tsim_io_ops = get_flush_sim_io_ops()
        if tsim_io_ops:
            await self._bkpt_handle_recorded_io_ops(tsim_io_ops)

        self.editor.childs_complex.icomps.clear()
        await self.editor.set_new_layout(self.editor.childs_complex)
        await self.editor.set_decorations("common", [])
        await self.tree.tree.set_root_object_dict({})
        # await self.tree.clear_custom_layout()
        self._next_bkpt_set_lineno_tid = None
        async with self.dm.draft_update() as draft:
            draft.cur_thread_option = None
            draft.thread_options = []
            draft.local_var_key = ""
            # keep persist data here until new run.
            for global_key in self.dm.model.global_mem.matrices.keys():
                draft.global_mem.matrices[global_key].temp_fill_color = None
                draft.global_mem.matrices[global_key].temp_fill_pos = None
                draft.global_mem.matrices[global_key].temp_mask_pos = None

    
    async def _handle_leave_bkpt(self, thread_id: str, bkpt: PFLRunnerBreakpoint):
        if self._runner is None:
            return 
        await self._sync_threads_status()
        cur_thread_opt = self.dm.model.cur_thread_option
        if cur_thread_opt is not None and cur_thread_opt.label != thread_id:
            return 
        await self.tree.tree.set_root_object_dict({})
        await self.editor.set_decorations("common", [])
        for k, v in self.editor.childs_complex.icomps.items():
            slider = v.comp
            if isinstance(slider, mui.BlenderSlider):
                await slider.send_and_wait(slider.update_event(disabled=True))
        thread = self._runner.runner.get_thread(thread_id)
        if thread.is_main_thread:
            async with self.dm.draft_update() as draft:
                for global_key in self.dm.model.global_mem.matrices.keys():
                    draft.global_mem.matrices[global_key].temp_mask_pos = None
                    draft.global_mem.matrices[global_key].temp_aabb_line_pos = None
                    draft.global_mem.matrices[global_key].temp_aabb_line_size = None

    async def _handle_slider(self, value: mui.NumberType, slider: mui.BlenderSlider, ctrl: PFLRunnerCtrlFor):
        if self._runner is None:
            return 
        if self._runner._runner_task is None:
            return 
        cur_thread = self.dm.model.cur_thread_option
        if cur_thread is None:
            return
        cur_thread_id = cur_thread.label
        old = slider.int()
        new = int(value)
        if new > old:
            ctrl.step = new
            ctrl.should_pause = False
            self._runner.runner.release_breakpoint(cur_thread_id)
    
    def _get_cur_grid_idxes(self) -> tuple[int, int, int]:
        return self.dm.model.grid_idx_x, self.dm.model.grid_idx_y, self.dm.model.grid_idx_z

    def _validate_editor_has_unsave(self):
        if self._runner is None:
            return False
        cur_content = self._runner.state.content
        editor_value = self.editor.props.value
        assert isinstance(editor_value, str)
        if editor_value != cur_content:
            raise ValueError("Editor content has unsaved changes, please save before running.")

    def _capture_expr_val(self, thread_id: str, expr_hit: PFLRunnerExprHit, results: list[ExprTraceData]):
        for_steps = [(x[1]) for x in expr_hit.for_stack]
        tctx = get_tensorsim_context_checked()
        grid_id = tctx.get_flatted_grid_id()
        if isinstance(expr_hit.data, (tritonstd.Tensor, int, float, bool)):
            if isinstance(expr_hit.data, tritonstd.Tensor):
                value = expr_hit.data._clone()
            else:
                value = expr_hit.data
            results.append(ExprTraceData(tuple(for_steps), grid_id, value))

    async def _handle_editor_acts(self, act: mui.MonacoActionEvent):
        if self._runner is None:
            return 
        if act.action == EditorActions.RUN_TO.value:
            if act.selection is not None:
                self._validate_editor_has_unsave()
                lineno = act.selection.selections[0].startLineNumber
                inline_env = self._runner.run_to_use_task(self._get_cur_grid_idxes(), lineno)
                if inline_env is not None:
                    sim_info = inline_env.get_userdata_typed(tritonstd.TritonSimInfo)
                    gmem = sim_info.global_mem
                    assert gmem is not None 
                    mat_dict: dict[str, np.ndarray] = {}
                    for k, block in gmem.memory_blocks.items():
                        mat_dict[k] = block.get_data_view_checked()
                    await self._global_mem.set_matrix_dict(mat_dict, sim_info.vis_layout)

        if act.action == EditorActions.EXPR_TRACE.value:
            if act.selection is not None:

                lineno = act.selection.selections[0].startLineNumber
                end_lineno = act.selection.selections[0].endLineNumber
                col_start = act.selection.selections[0].startColumn - 1 # 1-based to 0-based
                if end_lineno != lineno:
                    print("Cannot trace multi-line expression")
                    return None
                col_end = act.selection.selections[0].endColumn - 1
                func_uid, expr_node = self._runner.find_expr_node_by_source_loc((lineno, col_start, end_lineno, col_end))
                if expr_node is None or func_uid is None:
                    print(f"Cannot find expression at line {lineno}, column {col_start + 1}, {act.selection.selections[0]}")
                    return None
                fuid_name = self._remove_spec_suffix_of_func_uid(func_uid)
                fuid_name_qname = "::".join(fuid_name.split("::")[1:])
                key = f"{fuid_name_qname}-{unparse_pfl_ast(expr_node)}"
                await self._expr_trace(key, expr_node.source_loc)

    async def _expr_trace(self, key: str, source_loc: pfl.SourceLocType):
        assert self._runner is not None, "Runner must be initialized before expr trace"
        results: list[ExprTraceData] = []
        ev_handler = partial(self._capture_expr_val, results=results)
        self._runner.state.expr_trace_runner.set_observed_source_locs({
            key: source_loc
        })
        self._runner.state.expr_trace_runner.event_thread_expr_hit.on(ev_handler)
        try:
            sim_info = await self._runner.run_expr_trace_kernel_test()
        finally:
            self._runner.state.expr_trace_runner.event_thread_expr_hit.off(ev_handler)
        if results:
            async with self.dm.draft_update() as draft:

                results.sort(key=lambda x: x.grid_id)
                grid_id_to_res: dict[int, list[ExprTraceData]] = {}
                for res in results:
                    if res.grid_id not in grid_id_to_res:
                        grid_id_to_res[res.grid_id] = []
                    grid_id_to_res[res.grid_id].append(res)
                # for v in grid_id_to_res.values():
                #     v.sort(key=lambda x: x.for_steps)
                max_cnt = max(len(v) for v in grid_id_to_res.values())
                grid_is_row = True 
                draft.expr_trace_matrices.matrices.clear()
                draft.expr_trace_matrices.labels.clear()
                grid_id_cnt = 0
                labels: list[LabelWithId] = []
                if grid_is_row:
                    result_matrix: list[list[Any]] = [[None for j in range(max_cnt + 1)] for i in range(len(grid_id_to_res))]
                    grid_keys = list(grid_id_to_res.keys())
                    for j in range(len(grid_id_to_res)):
                        gid = grid_keys[j]
                        label = LabelWithId(text=f"B{gid}", id=str(gid)) 
                        result_matrix[j][0] = label
                        labels.append(label)
                else:
                    result_matrix: list[list[Any]] = [[None for j in range(len(grid_id_to_res))] for i in range(max_cnt + 1)]
                    result_matrix[0] = [LabelWithId(text=f"B{gid}", id=str(gid)) for gid in grid_id_to_res.keys()]
                mat_cnt = 0
                item_cnt = 0

                mat_fontsize_min = None
                for grid_id, res_list in grid_id_to_res.items():
                    for j in range(len(res_list)):
                        item = res_list[j]
                        name = "-".join(map(str, item.for_steps))
                        uid = UniqueTreeId.from_parts([name, str(item_cnt), str(grid_id)]).uid_encoded

                        if isinstance(item.value, tritonstd.Tensor):
                            mat = LocalMatrix.from_shape(name, list(item.value.shape), label_with_shape=False)
                            mat.id = uid
                            mat.fontSize /= 1.5
                            if mat_fontsize_min is None:
                                mat_fontsize_min = mat.fontSize
                            else:
                                mat_fontsize_min = min(mat_fontsize_min, mat.fontSize)
                            draft.expr_trace_matrices.matrices.append(mat)
                            self._set_local_memory_model_draft(draft.expr_trace_matrices.matrices[mat_cnt], item.value)

                            mat_cnt += 1
                        else:
                            if isinstance(item.value, float):
                                mat = LabelWithId(text=f"{item.value:.4f}", id=uid)
                            else:
                                mat = LabelWithId(text=str(item.value), id=uid)
                            labels.append(mat)
                        if grid_is_row:
                            result_matrix[grid_id_cnt][j + 1] = mat
                        else:
                            result_matrix[j + 1][grid_id_cnt] = mat
                        item_cnt += 1
                    grid_id_cnt += 1 
                draft.expr_trace_matrices.labels = labels
                for label in labels:
                    if mat_fontsize_min is None:
                        mat_fontsize_min = 5
                    label.fontSize = mat_fontsize_min / 2
                    
                max_w, max_h = layout_table_inplace(result_matrix)
                for i, row in enumerate(result_matrix):
                    for j, cell in enumerate(row):
                        if isinstance(cell, (LocalMatrix, LabelWithId)):
                            cell.offsetX -= max_w / 2
                            cell.offsetY -= max_h / 2
                draft.expr_trace_matrices.minimap.width = max_w
                draft.expr_trace_matrices.minimap.height = max_h
                draft.expr_trace_matrices.title = key
            # save code here to match when code changed
            self._cur_traced_expr = (source_loc, key, self._runner.state.content_lines)

    async def _handle_editor_cursor_selection(self, ev: mui.MonacoSelectionEvent):
        if self._runner is None:
            return 
        try:
            lineno = ev.selections[0].startLineNumber # 1-based in monaco.Selection
            col = ev.selections[0].startColumn # 1-based
            if self._runner.state.mapper_new_to_old is not None:
                lineno_mapped = self._runner.state.mapper_new_to_old.bisect_mapped_lineno(lineno)
                if lineno_mapped != -1:
                    lineno = lineno_mapped
            func_uid, node = self._runner.find_nearest_node_by_line_col(lineno, col - 1)
            if node is not None and func_uid is not None:
                if not isinstance(node, pfl.PFLName):
                    return 
                if not node.st.has_metadata(tritonstd.Tensor, ):
                    return 
                meta = node.st.metadata_checked
                assert isinstance(meta, (tritonstd.Tensor, ))
                await self._create_local_tensor_panel(func_uid, node.id, meta.shape)
                self._cur_observed_local_tensor_key = (func_uid, node.id)
            
        except Exception as e:
            traceback.print_exc()
            raise 

    async def _create_local_tensor_panel(self, func_uid: str, obj_name: str, shape: list[int]):
        assert self._runner is not None
        func_uid_parts = UniqueTreeId(func_uid).parts
        func_local_qname = ".".join(func_uid_parts[0].split("::")[1:])
        func_uid_no_suffix = self._remove_spec_suffix_of_func_uid(func_uid)
        cur_thread_opt = self.dm.model.cur_thread_option
        async with self.dm.draft_update() as draft:
            # draft.local_matrices.clear()
            mat = LocalMatrix.from_shape(obj_name, shape)
            mat_layout_shape = mat.get_vis_wh()
            draft.local_mat = LocalMemoryModel(matrix=mat,
                minimap=plus.hud.MinimapModel(mat_layout_shape[0], mat_layout_shape[1]))
            key = f"{func_local_qname}-{obj_name}"
            draft.local_var_key = key
            if cur_thread_opt is not None:
                draft.cur_thread_option.cur_local_tensor_id = (func_uid, obj_name)
                for opt in self.dm.model.thread_options:
                    if opt.label == cur_thread_opt.label:
                        draft.cur_thread_option = dataclasses.replace(opt, cur_local_tensor_id=(func_uid, obj_name))
                        break
                if cur_thread_opt.is_paused:
                    cur_thread = self._runner.runner.get_thread(cur_thread_opt.label)
                    bkpt = cur_thread.get_cur_bkpt_checked()
                    self._bkpt_handle_local_tensor_panel_local(func_uid_no_suffix, obj_name, draft, bkpt)
        # await self.tree.set_custom_layout(mui.HBox([
        #     local_container
        # ]).prop(width="100%", height="100%", overflow="hidden"))


    def _remove_spec_suffix_of_func_uid(self, func_uid: str) -> str:
        parts = UniqueTreeId(func_uid).parts
        return parts[0] 

    async def hover_query(self, hqevent: mui.MonacoHoverQueryEvent) -> Optional[mui.MonacoHover]:
        if self._runner is None:
            return 
        lineno = hqevent.position.lineNumber

        col = hqevent.position.column
        if self._runner.state.mapper_new_to_old is not None:
            lineno_mapped = self._runner.state.mapper_new_to_old.bisect_mapped_lineno(lineno)
            if lineno_mapped != -1:
                lineno = lineno_mapped
        func_uid, node = self._runner.find_nearest_node_by_line_col(lineno, col - 1)
        if func_uid is not None and node is not None and isinstance(node, (pfl.PFLExpr, pfl.PFLArg)):
            node_expr = pfl.unparse_pfl_ast(node)
            loc = node.get_source_loc_checked()
            msg =f"### {node_expr}\n`{node.st}`"
            if not is_undefined(node.st.metadata):
                msg += f"\n{node.st.metadata}"
            cur_thread_opt = self.dm.model.cur_thread_option

            if cur_thread_opt is not None and cur_thread_opt.is_paused:
                cur_thread = self._runner.runner.get_thread(cur_thread_opt.label)
                stack = cur_thread.get_cur_bkpt_checked().stack
                for s in stack:
                    if self._remove_spec_suffix_of_func_uid(s.node.uid) == self._remove_spec_suffix_of_func_uid(func_uid):
                        if isinstance(node, pfl.PFLName) and node.id in s.scope:
                            val = s.scope[node.id]
                            msg += f"\n\n**Value in stack**: \n`{val}`"
                        if isinstance(node, pfl.PFLArg) and node.arg in s.scope:
                            val = s.scope[node.arg]
                            msg += f"\n\n**Value in stack**: \n`{val}`"

            return mui.MonacoHover([
                mui.MonacoMarkdownString(value=msg)
            ], range=mui.MonacoRange(loc[0], loc[1] + 1, loc[2], loc[3] + 1))
        return None 

    async def inlay_hint_query(self, ev: mui.MonacoInlayHintQueryEvent) -> Optional[mui.MonacoInlayHintList]:
        res: list[mui.MonacoInlayHint] = []
        if self._runner is None:
            return 
        cur_thread_opt = self.dm.model.cur_thread_option

        if cur_thread_opt is not None and cur_thread_opt.is_paused:
            cur_thread = self._runner.runner.get_thread(cur_thread_opt.label)
            stack = cur_thread.get_cur_bkpt_checked().stack
            mapper: Optional[SCDItem] = None
            new_value = ev.value
            query_range = ev.range
            if new_value is not None:
                new_value_lines = new_value.split("\n")
                # mapper = SourceChangeDiffCache.get_raw_item_for_mapping(new_value_lines, self._runner._content_lines, )
                mapper = SourceChangeDiffCache.get_raw_item_for_mapping(self._runner.state.content_lines, new_value_lines, )

            for s in stack:
                finder = self._runner.state.finder_dict[s.node.uid]

                for node in finder._all_nodes:
                    if isinstance(node, pfl.PFLName) and node.id in s.scope :
                        val = s.scope[node.id]
                        if isinstance(val, (int, bool, str)):
                            val_str = str(val)
                        elif isinstance(val, tritonstd.Tensor):
                            shape = ",".join(map(str, val.shape))
                            val_str = f"[{shape}]"
                        else:
                            continue
                        end_lineno = node.get_source_loc_checked()[2]
                        if mapper is not None:
                            end_lineno_mapped = mapper.bisect_mapped_lineno(end_lineno)
                            if end_lineno_mapped != -1:
                                end_lineno = end_lineno_mapped
                        if end_lineno < query_range.startLineNumber or end_lineno > query_range.endLineNumber:
                            continue
                        # print(end_lineno, node.get_source_loc_checked()[3] + 1)
                        res.append(mui.MonacoInlayHint(
                            label=f":{val_str}",
                            position=mui.MonacoPosition(end_lineno, node.get_source_loc_checked()[3] + 1),
                            kind=1,
                        ))
        if res:
            return mui.MonacoInlayHintList(res)
        return None 


    def _handle_editor_debounced_change(self, change):
        if self._runner is None:
            return 
        new_value = change["value"]
        new_value_lines = new_value.split("\n")
        self._runner.state.mapper_new_to_old = SourceChangeDiffCache.get_raw_item_for_mapping(new_value_lines, self._runner.state.content_lines)

    async def _on_launch_triton(self):
        if self._runner is None:
            return
        res = await self._runner.run_triton_bench_sync()
        if res is not None:
            await self._triton_viewer.set_triton_compile_info(self._runner.state.cur_fn_name, res)
    
    def _get_sim_bkpt_descs(self, path: str, bkpts: list[mui.MonacoBreakpoint]):
        bkpts_desc: dict[tuple[str, int], PFLBreakpointDesc] = {}
        cur_path = path
        for bkpt in bkpts:
            if bkpt.enabled:
                bkpts_desc[(cur_path, bkpt.lineNumber)] = PFLBreakpointDesc(bkpt.lineNumber) 
        return bkpts_desc

    async def _handle_editor_bkpt_change(self, bkpts: list[mui.MonacoBreakpoint]):
        if self._runner is None:
            return 
        cur_path = self._runner.state.path
        bkpts_desc = self._get_sim_bkpt_descs(cur_path, bkpts)
        self._runner.runner.sync_breakpoints(bkpts_desc)
        async with self.app_dm.draft_update() as draft:
            draft.bkpts[cur_path] = bkpts

    async def set_triton_compile_info(self, func_id: str, info: tritonstd.TritonKernelCompileInfo):
        await self._triton_viewer.set_triton_compile_info(func_id, info)

def _main():
    from tensorpc.dock.core.datamodel import _compile_pfllibrary
    _compile_pfllibrary(TritonSimModel)


if __name__ == "__main__":
    _main()