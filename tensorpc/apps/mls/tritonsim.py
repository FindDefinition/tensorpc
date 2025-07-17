import ast
import asyncio
from collections.abc import Sequence
import enum
from functools import partial
import importlib
import json
import linecache
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
import traceback
import types
from typing import Any, Union, cast
import uuid
from tensorpc.core.astex.sourcecache import SCDItem, SourceChangeDiffCache
from tensorpc.apps.mls import tsim
from tensorpc.apps.mls.components.global_mem import GlobalMatrixPanel, GlobalMemContainer, GlobalMemoryModel, Matrix
from tensorpc.apps.mls.tsim.core import TensorSimIoOp, get_flush_sim_io_ops
from tensorpc.constants import PACKAGE_ROOT
from tensorpc.core.annolib import is_undefined
from tensorpc.core.astex.astcache import AstCache, AstCacheItem
from tensorpc.core.datamodel.draft import DraftBase
from tensorpc.core.funcid import find_toplevel_func_node_by_lineno, find_toplevel_func_node_container_by_lineno
from tensorpc.core.moduleid import get_module_id_of_type
from tensorpc.core.pfl.evaluator import PFLAsyncRunnerStateType, PFLBreakpoint, PFLCtrlFor, PFLCtrlBase
from tensorpc.core.pfl.backends.js import ColorUtil, Math, MathUtil
from tensorpc.core.tree_id import UniqueTreeId, UniqueTreeIdForTree
from tensorpc.dock import mui, three, plus, mark_create_layout, mark_did_mount, appctx
from typing import Annotated, Any, Optional, Union
from tensorpc.core import pfl

from tensorpc.apps.mls.backends import tritonstd
import numpy as np 
from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.dock.core.appcore import AppSpecialEventType
from tensorpc.dock.vscode.coretypes import VscodeTensorpcMessage, VscodeTensorpcMessageType
from tensorpc.utils.package_finder import find_submodule_from_file
import importlib.machinery

@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class LocalMatrix(Matrix):
    global_indices: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)

@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class LocalMemoryModel:
    matrix: LocalMatrix
    minimap: plus.hud.MinimapModel
    hover: Optional[str] = None


class LocalMemContainer(mui.TooltipFlexBox):
    def __init__(self, key: str, draft: LocalMemoryModel, use_view: bool = False):
        assert isinstance(draft, DraftBase)
        panel = GlobalMatrixPanel(draft.matrix, enable_hover_line=True)
        minimap = plus.hud.MiniMap(draft.minimap, [
            panel
        ])
        self.panel = panel
        self.minimap = minimap
        self._draft = draft
        cam = three.OrthographicCamera(near=0.1, far=1000, children=[
            minimap,
        ]).prop(position=(0, 0, 10))
        if use_view:
            canvas = three.View([
                cam.prop(makeDefault=True),
                # three.InfiniteGridHelper(1, 10, "green")
            ]).prop(allowKeyboardEvent=True)
        else:
            canvas = three.Canvas([
                cam.prop(makeDefault=True),
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


@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class TritonSimModel:
    grid_idx_x: int = 0
    grid_idx_y: int = 0
    grid_idx_z: int = 0

    grid_size_x_range: tuple[int, int, int] = (0, 0, 1)
    grid_size_y_range: tuple[int, int, int] = (0, 0, 1)
    grid_size_z_range: tuple[int, int, int] = (0, 0, 1)
    global_mem: GlobalMemoryModel
    is_paused: bool = False 
    # stack tensors during triton simulation
    local_matrices: dict[str, LocalMemoryModel] = dataclasses.field(default_factory=dict)

    @pfl.mark_pfl_compilable
    def _on_hover_pfl(self, data: three.PointerEvent, key: str):
        # print("WTF", self.global_mem)

        if key in self.local_matrices:
            local_mat = self.local_matrices[key]
            point_unified_x = data.pointLocal[0] + 0.5
            point_unified_y = -data.pointLocal[1] + 0.5
            idx_x = Math.floor(point_unified_x * local_mat.matrix.width)
            idx_y = Math.floor(point_unified_y * local_mat.matrix.height)
            flat_idx = idx_y * local_mat.matrix.width + idx_x
            # self.linePosX = (idx_x + 0.5) - local_mat.width / 2
            # self.linePosY = (-(idx_y + 0.5)) + local_mat.height / 2
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
                        line_pos_buffer[j * 2] = inds_flat_buffer[j] % global_mat.width + 0.5 - global_mat.width / 2
                        line_pos_buffer[j * 2 + 1] = -(Math.floor(inds_flat_buffer[j] / global_mat.width) + 0.5 - global_mat.height / 2)
                        line_size_buffer[j * 2] = 1
                        line_size_buffer[j * 2 + 1] = 1
                    # print(fill_pos)
                    # print("COLOR", fill_pos_buffer[0], fill_pos_buffer[1], fill_pos_buffer[2])
                    # print("POS", inds_flat_buffer[0] % global_mat.width, Math.floor(inds_flat_buffer[0] / global_mat.width))
                    global_mat.temp_aabb_line_pos = line_pos
                    global_mat.temp_aabb_line_size = line_size

    @pfl.mark_pfl_compilable
    def _on_hover_leave_pfl(self, data: three.PointerEvent, key: str):
        # print("WTF", self.global_mem)
        if key in self.local_matrices:
            local_mat = self.local_matrices[key]
            local_mat.hover = None
        for global_key, mat in self.global_mem.matrices.items():
            # mat.temp_fill_pos = None 
            # mat.temp_fill_color = None
            mat.temp_aabb_line_pos = None
            mat.temp_aabb_line_size = None

    def get_global_fill(self, global_key: str, inds: np.ndarray, is_persist: bool = True):
        global_mat = self.global_mem.matrices[global_key]
        return global_mat.get_global_fill(global_key, inds, is_persist=is_persist)


@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class SingleBlockRunState:
    # used to record all memory access in history.
    global_access_indices: dict[str, np.ndarray]


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

class TritonKernelRunner:
    def __init__(self, item: AstCacheItem, path: str, lineno: int):
        self._compiled_tmp_file_path: Optional[str] = None
        mod_dict = self._get_module_dict_init(path, item.content)
        func_nodes = find_toplevel_func_node_container_by_lineno(cast(ast.Module, item.tree), lineno)
        assert func_nodes is not None 

        fn_node = func_nodes[-1]
        assert isinstance(fn_node, ast.FunctionDef), "Function node must be a FunctionDef"
        fn = mod_dict[fn_node.name]
        self._fn = fn
        self._fn_name = fn_node.name
        self._path = path
        self._content = item.content
        self._content_lines = item.content.split("\n")

        runner = tritonstd.parse_triton_compilable_to_runner(fn, do_meta_eval=True, module_code_getter=lambda x: item.content)
        # print(ast.ret_st)
        lib = runner._library
        self.lib = lib
        finder_dict: dict[str, pfl.PFLTreeNodeFinder] = {}
        for k, v in lib.all_compiled.items():
            finder_dict[k] = pfl.PFLTreeNodeFinder(v, (pfl.PFLName, pfl.PFLAttribute, pfl.PFLArg)) 
        self.finder_dict = finder_dict
        self.runner = runner
        self._runner_task: Optional[asyncio.Task] = None
        self._runner_task_ev = asyncio.Event()

        self.grid_size = runner.triton_sim_info.grid_size

        self._mapper_new_to_old: Optional[SCDItem] = None


    def close(self):
        if self._compiled_tmp_file_path is not None:
            # remove from linecache
            linecache.checkcache(self._compiled_tmp_file_path)
            self._compiled_tmp_file_path = None

    def _get_module_dict_init(self, path: str, code: Optional[str] = None):
        module_import_path = find_submodule_from_file(path)
        if module_import_path is None:
            assert code is not None 
            mod_dict = self._get_module_dict_by_code(path, code)
        else:
            mod_dict = importlib.import_module(module_import_path).__dict__
        return mod_dict

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

    def recompile(self, new_code: str):
        self._runner_task_ev.clear()
        mod_dict = self._get_module_dict_by_code(self._path, new_code)
        fn = mod_dict[self._fn_name]
        runner = tritonstd.parse_triton_compilable_to_runner(fn, do_meta_eval=True, module_code_getter=lambda x: new_code)
        self._fn = fn
        # print(ast.ret_st)
        lib = runner._library
        self.lib = lib
        finder_dict: dict[str, pfl.PFLTreeNodeFinder] = {}
        for k, v in lib.all_compiled.items():
            finder_dict[k] = pfl.PFLTreeNodeFinder(v, (pfl.PFLName, pfl.PFLAttribute, pfl.PFLArg)) 
        self.finder_dict = finder_dict
        self.runner = runner
        self._runner_task = None
        self._content = new_code
        self.grid_size = runner.triton_sim_info.grid_size
        self._content_lines = new_code.split("\n")
        self._mapper_new_to_old = None 

    async def run_to(self, grid_idxes: Sequence[int], lineno: int):
        stmt = self.lib.find_stmt_by_path_lineno(self.lib.get_module_by_func(self._fn.fn).uid, lineno)
        if stmt is not None:
            if self.runner.is_paused():
                # if paused, continue from the current position
                await self.runner.continue_until(lineno)
            else:
                self._runner_task_ev.clear()
                assert self.runner._state.type == pfl.PFLAsyncRunnerStateType.IDLE, \
                    f"Runner is not in IDLE state, current state: {self.runner._state.type}"
                with tsim.enter_tensorsim_context(grid_idxes, self.runner.triton_sim_info.grid_size):
                    inline_env = self.lib.get_compiled_unit_inline_env(self._fn.fn)
                    # use data in inline_env to create tensor visualization.
                    func_node = self.runner._library.get_compiled_unit_specs(self._fn.fn)[0]
                    self._runner_task = asyncio.create_task(self.runner.run_until(lineno, func_node.uid, 
                        exit_event=self._runner_task_ev, external_inline_env=inline_env))
                    return inline_env
        return None 

    async def run_single_block(self, grid_idxes: Sequence[int]):
        assert not self.runner.is_paused()
        self._runner_task_ev.clear()
        assert self.runner._state.type == pfl.PFLAsyncRunnerStateType.IDLE, \
            f"Runner is not in IDLE state, current state: {self.runner._state.type}"
        with tsim.enter_tensorsim_context(grid_idxes, self.runner.triton_sim_info.grid_size):
            inline_env = self.lib.get_compiled_unit_inline_env(self._fn.fn)
            # use data in inline_env to create tensor visualization.
            func_node = self.runner._library.get_compiled_unit_specs(self._fn.fn)[0]
            self._runner_task = asyncio.create_task(self.runner.run_func(func_node.uid, 
                exit_event=self._runner_task_ev, external_inline_env=inline_env))
            return inline_env
        return None 

    def find_nearest_node_by_line_col(self, lineno: int, col_offset: int):
        for k, finder in self.finder_dict.items():
            res = finder.find_nearest_node_by_line_col(lineno, col_offset)
            if res is not None:
                return k, res
        return None, None 

    async def stop_run(self):
        if self.runner.is_paused():
            self.runner.release_breakpoint(stop=True)
            await self._runner_task_ev.wait()

class TritonSim:
    @mark_create_layout
    def my_layout(self):
        # mui.MonacoEditor.InlineComponent(
        #     mui.BlenderSlider(0, 10, 1).prop(width="50%"), 
        #     afterLineNumber=2, heightInPx=24)

        self.dm = mui.DataModel(TritonSimModel(global_mem=GlobalMemoryModel.empty(), local_matrices={}), [])
        draft = self.dm.get_draft()
        self.editor = mui.MonacoEditor("", "python", "")
        self.tree = plus.ObjectInspector(with_builtins=False, show_terminal=False, default_tab_preview=False, default_sizes=[100, 100])
        self.io_ops_tree = mui.TanstackJsonLikeTree().prop(ignoreRoot=True)
        self.io_ops_tree.event_select.on(self._handle_io_tree_select)
        self._cur_recorded_io_ops: list[TensorSimIoOp] = []
        self.editor.update_raw_props({
            ".monaco-editor-content-decoration": {
                "background": "lightblue"
            }
        })
        editor_acts: list[mui.MonacoEditorAction] = [
            mui.MonacoEditorAction(id="Run To", 
                label="Run Towards Here", contextMenuOrder=1.5,
                contextMenuGroupId="tensorpc-pfl-editor-action", 
                keybindings=[([mui.MonacoKeyMod.Shift], 3)]),
        ]
        debug_toolbar = mui.HBox([
            mui.IconButton(mui.IconType.PlayArrow, self._on_debug_just_run)
                .prop(tooltip="Run Single Block", size="small", iconSize="small", muiColor="primary")
                .bind_fields(disabled=f"({draft.is_paused})"),
            mui.IconButton(mui.IconType.KeyboardArrowRight, self._on_debug_next_line)
                .prop(tooltip="Next Line", size="small", iconSize="small", muiColor="primary")
                .bind_fields(disabled=f"!({draft.is_paused})"),
            mui.IconButton(mui.IconType.KeyboardDoubleArrowRight, self._on_debug_continue)
                .prop(tooltip="Continue", size="small", iconSize="small", muiColor="primary")
                .bind_fields(disabled=f"!({draft.is_paused})"),
            mui.IconButton(mui.IconType.RestartAlt,)
                .prop(tooltip="Restart", size="small", iconSize="small", muiColor="success")
                .bind_fields(disabled=f"!({draft.is_paused})"),
            mui.IconButton(mui.IconType.Stop, self._on_debug_stop)
                .prop(tooltip="Stop", size="small", iconSize="small", muiColor="error")
                .bind_fields(disabled=f"!({draft.is_paused})"),
        ])
        self.editor.prop(minWidth=0, minHeight=0, actions=editor_acts)
        self.editor.event_editor_hover_query.on(self.hover_query)
        self.editor.event_editor_action.on(self._handle_editor_acts)
        self.editor.event_editor_inlay_hints_query.on(self.inlay_hint_query)
        self.editor.event_editor_cursor_selection.on(self._handle_editor_cursor_selection)
        self.editor.event_editor_save.on(self._handle_editor_save)
        self.editor.event_change.on(self._handle_editor_debounced_change)

        self._runner: Optional[TritonKernelRunner] = None
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
        self._next_bkpt_set_lineno: bool = False

        self.dm.init_add_layout([
            mui.VBox([
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
                self.editor.prop(flex=3),
                mui.HDivider(),
                mui.AppTerminal().prop(flex=1),
            ]).prop(flex=2),
            mui.VDivider(),
            self._global_mem.prop(flex=2),
        ])
        return mui.VBox([
            three.ViewCanvas([
                self.dm
            ]).prop(display="flex", flexFlow="row nowrap", flex=1)
        ]).prop(width="100%", height="100%", overflow="hidden")

    async def _on_debug_continue(self):
        if self._runner is None:
            return
        self._validate_editor_has_unsave()
        self._runner.runner.release_breakpoint()
        self._next_bkpt_set_lineno = True

    async def _on_debug_next_line(self):
        if self._runner is None:
            return
        self._validate_editor_has_unsave()
        assert self._runner.runner.is_paused(), "Runner must be paused to continue to next line"
        await self._runner.runner.continue_next_line()
        self._next_bkpt_set_lineno = True


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
            draft.grid_idx_x = 0
            draft.grid_idx_y = 0
            draft.grid_idx_z = 0
        sim_info = self._runner.runner.triton_sim_info
        gmem = sim_info.global_mem
        assert gmem is not None 
        mat_dict: dict[str, np.ndarray] = {}
        for k, block in gmem.memory_blocks.items():
            mat_dict[k] = block.get_data_view_checked()
        await self._global_mem.set_matrix_dict(mat_dict)

    async def _handle_editor_save(self, ev: mui.MonacoSaveEvent):
        assert self._runner is not None, "Runner must be initialized before saving"
        prev_is_paused = self._runner.runner.is_paused()
        # print(0)
        await self._on_debug_stop()
        # print(1)
        with open(self._runner._path, "w", encoding="utf-8") as f:
            f.write(ev.value)
        self._runner.recompile(ev.value)
        await self._init_sim_info()
        self._install_event_handlers_to_runner(self._runner)

        if prev_is_paused:
            decors = ev.decorationsRanges
            if decors is not None:
                common_ranges = decors["common"]
                assert len(common_ranges) == 1
                new_bkpt_lineno = common_ranges[0].startLineNumber
                cur_grid_idxes = [self.x_slider.int(), self.y_slider.int(), self.z_slider.int()]
                for j in range(len(cur_grid_idxes)):
                    cur_grid_idxes[j] = max(cur_grid_idxes[j], 0)
                    cur_grid_idxes[j] = min(cur_grid_idxes[j], self._runner.grid_size[j] - 1)
                # print(2)
                await self._runner.run_to(cur_grid_idxes, new_bkpt_lineno)
                # print(cur_grid_idxes, new_bkpt_lineno)
    
    def _install_event_handlers_to_runner(self, runner: TritonKernelRunner):
        runner.runner.event_eval_start.on(self._handle_eval_start)
        runner.runner.event_eval_stop.on(self._handle_eval_stop)
        runner.runner.event_enter_bkpt.on(self._handle_enter_bkpt)
        runner.runner.event_leave_bkpt.on(self._handle_leave_bkpt)

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
            self._runner = TritonKernelRunner(item, path, lineno)
            self._install_event_handlers_to_runner(self._runner)

            await self.editor.write(item.content, path, "python")
            await self.editor.set_line_number(data.selections[0].start.line)
            await self._init_sim_info()

    async def _handle_slider_change(self, value: Any, axis: int):
        if self._runner is None:
            return 
        self._validate_editor_has_unsave()
        old_value = [self.x_slider.int(), self.y_slider.int(), self.z_slider.int()]
        old_value[axis] = value

        if self._runner.runner.is_paused():
            cur_bkpt_lineno = self._runner.runner.get_state().cur_bkpt.node.source_loc[0]
            await self._runner.stop_run()
            await self._runner.run_to(old_value, cur_bkpt_lineno)
        else:
            await self._runner.run_single_block(old_value)

    def _bkpt_handle_local_tensor_panel_local(self, k: str, draft: TritonSimModel, bkpt: PFLBreakpoint):
        for stack in bkpt.stack[::-1]:
            if k in stack.scope:
                func_uid_no_suffix = self._remove_spec_suffix_of_func_uid(stack.node.uid)

                local_key = f"{func_uid_no_suffix}-{k}"
                ten = bkpt.scope[k]
                assert isinstance(ten, (tritonstd.Tensor, tritonstd.PointerTensor))
                if isinstance(ten, tritonstd.Tensor):
                    for global_key in self.dm.model.global_mem.matrices.keys():
                        draft.global_mem.matrices[global_key].temp_fill_pos = None 
                        draft.global_mem.matrices[global_key].temp_fill_color = None

                    storage = ten._wrapped.get_storage_checked()
                    data = storage.data
                    if data.dtype == np.bool_:
                        # dont support bool in frontend
                        data = data.astype(np.uint8)
                    elif data.dtype == np.int64 or data.dtype == np.uint64:
                        # dont support bigint (64bit int) in frontend
                        data = data.astype(np.int32 if data.dtype == np.int64 else np.uint32)
                    draft.local_matrices[local_key].matrix.data = data
                    draft.local_matrices[local_key].matrix.global_indices = storage.indices
                    local_pos, local_color = Matrix.get_value_pos_and_color_gray(storage.data)
                    draft.local_matrices[local_key].matrix.persist_fill_pos = local_pos
                    draft.local_matrices[local_key].matrix.persist_fill_color = local_color
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

    async def _bkpt_handle_local_tensor_preview_panel(self, draft: TritonSimModel, bkpt: PFLBreakpoint):
        for local_key, v in self.dm.model.local_matrices.items():
            obj_id = local_key.split("-")[-1]
            self._bkpt_handle_local_tensor_panel_local(obj_id, draft, bkpt)


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
                assert op.name in self._block_run_backend_state.global_access_indices
                old_inds = self._block_run_backend_state.global_access_indices[op.name]
                new_inds = np.unique(np.concatenate([old_inds, op.io_indices.reshape(-1)]))
                self._block_run_backend_state.global_access_indices[op.name] = new_inds
                updated_keys.add(op.name)
            for k in updated_keys:
                inds = self._block_run_backend_state.global_access_indices[k]
                fill_pos, fill_color = self.dm.model.get_global_fill(k, inds, is_persist=True)
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

    async def _handle_enter_bkpt(self, bkpt: PFLBreakpoint):
        if self._runner is None:
            return 
        tsim_io_ops = get_flush_sim_io_ops()
        if self._next_bkpt_set_lineno:
            await self.editor.set_line_number(bkpt.node.source_loc[0])
            self._next_bkpt_set_lineno = False
        async with self._editor_lock:
            await self.tree.tree.set_root_object_dict(bkpt.scope)
            await self.editor.set_decorations("common", [
                mui.MonacoModelDeltaDecoration(mui.MonacoRange(bkpt.node.source_loc[0], 1, bkpt.node.source_loc[0], 1),
                mui.MonacoModelDecoration(className="monaco-editor-content-decoration", isWholeLine=True,
                minimap=mui.MonacoModelDecorationMinimapOptions(mui.MonacoMinimapPosition.Inline)))
            ])
            for ctrl_id, ctrl in self._runner.runner.get_state().cur_ctrl_points.items():
                assert isinstance(ctrl, PFLCtrlFor), "Only PFLCtrlFor is supported in this editor"
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
                    if int(data) not in self._runner.runner.get_state().cur_ctrl_points:
                        # remove the slider if the ctrl point is not in the current state
                        self.editor.childs_complex.icomps.pop(k, None)
            await self.editor.set_new_layout(self.editor.childs_complex)
            async with self.dm.draft_update() as draft:
                draft.is_paused = True
                # await self._bkpt_handle_local_tensor_panels(draft, bkpt)
                await self._bkpt_handle_local_tensor_preview_panel(draft, bkpt)
            await self._bkpt_handle_recorded_io_ops(tsim_io_ops)
        if self._cur_observed_local_tensor_key is not None:
            func_uid, obj_key = self._cur_observed_local_tensor_key
            for stack in bkpt.stack:
                stack_uid_no_suffix = self._remove_spec_suffix_of_func_uid(stack.node.uid)
                func_uid_no_suffix = self._remove_spec_suffix_of_func_uid(func_uid)
                if stack_uid_no_suffix == func_uid_no_suffix and obj_key in stack.scope:
                    obj = stack.scope[obj_key]
                    if isinstance(obj, tritonstd.Tensor):
                        await self._create_local_tensor_panel(func_uid, obj_key, obj.shape)

    async def _handle_eval_start(self):
        await self.io_ops_tree.clear()
        self._cur_recorded_io_ops.clear()
        global_access_indices: dict[str, np.ndarray] = {}
        for global_key, mat in self.dm.model.global_mem.matrices.items():
            global_access_indices[global_key] = np.empty([0], dtype=np.int32)
        self._block_run_backend_state = SingleBlockRunState(global_access_indices=global_access_indices)
        async with self.dm.draft_update() as draft:
            draft.is_paused = False
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
        await self.tree.clear_custom_layout()
        self._next_bkpt_set_lineno = False
        async with self.dm.draft_update() as draft:
            draft.is_paused = False
            # keep persist data here until new run.
            for global_key in self.dm.model.global_mem.matrices.keys():
                draft.global_mem.matrices[global_key].temp_fill_color = None
                draft.global_mem.matrices[global_key].temp_fill_pos = None
                draft.global_mem.matrices[global_key].temp_mask_pos = None

    
    async def _handle_leave_bkpt(self, bkpt: PFLBreakpoint):
        await self.tree.tree.set_root_object_dict({})
        # await self.io_ops_tree.clear()
        # self._cur_recorded_io_ops.clear()

        await self.editor.set_decorations("common", [])
        for k, v in self.editor.childs_complex.icomps.items():
            slider = v.comp
            if isinstance(slider, mui.BlenderSlider):
                await slider.send_and_wait(slider.update_event(disabled=True))
        async with self.dm.draft_update() as draft:
            draft.is_paused = False
            for global_key in self.dm.model.global_mem.matrices.keys():
                draft.global_mem.matrices[global_key].temp_mask_pos = None
                draft.global_mem.matrices[global_key].temp_aabb_line_pos = None
                draft.global_mem.matrices[global_key].temp_aabb_line_size = None

    async def _handle_slider(self, value: mui.NumberType, slider: mui.BlenderSlider, ctrl: PFLCtrlFor):
        if self._runner is None:
            return 
        if self._runner._runner_task is None:
            return 
        old = slider.int()
        new = int(value)
        if new > old:
            ctrl.step = new
            ctrl.should_pause = False
            print("RELEASE", old, new)
            self._runner.runner.release_breakpoint()
    
    def _get_cur_grid_idxes(self) -> tuple[int, int, int]:
        return self.dm.model.grid_idx_x, self.dm.model.grid_idx_y, self.dm.model.grid_idx_z

    def _validate_editor_has_unsave(self):
        if self._runner is None:
            return False
        cur_content = self._runner._content
        editor_value = self.editor.props.value
        assert isinstance(editor_value, str)
        if editor_value != cur_content:
            raise ValueError("Editor content has unsaved changes, please save before running.")

    async def _handle_editor_acts(self, act: mui.MonacoActionEvent):
        if self._runner is None:
            return 
        if act.action == "Run To":
            if act.selection is not None:
                self._validate_editor_has_unsave()
                lineno = act.selection.selections[0].startLineNumber
                inline_env = await self._runner.run_to(self._get_cur_grid_idxes(), lineno)
                if inline_env is not None:
                    gmem = inline_env.get_userdata_typed(tritonstd.TritonSimInfo).global_mem
                    assert gmem is not None 
                    mat_dict: dict[str, np.ndarray] = {}
                    for k, block in gmem.memory_blocks.items():
                        mat_dict[k] = block.get_data_view_checked()
                    await self._global_mem.set_matrix_dict(mat_dict)

    async def _handle_editor_cursor_selection(self, ev: mui.MonacoSelectionEvent):
        if self._runner is None:
            return 
        try:
            lineno = ev.selections[0].startLineNumber # 1-based in monaco.Selection
            col = ev.selections[0].startColumn # 1-based
            if self._runner._mapper_new_to_old is not None:
                lineno_mapped = self._runner._mapper_new_to_old.bisect_mapped_lineno(lineno)
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
        await self.tree.clear_custom_layout()
        func_uid_parts = UniqueTreeId(func_uid).parts
        func_local_qname = ".".join(func_uid_parts[0].split("::")[1:])
        func_uid_no_suffix = self._remove_spec_suffix_of_func_uid(func_uid)
        async with self.dm.draft_update() as draft:
            draft.local_matrices.clear()
            obj_key = f"{func_uid_no_suffix}-{obj_name}"
            mat = LocalMatrix.from_shape(obj_name, shape)
            mat_layout_shape = mat.get_vis_wh()
            draft.local_matrices[obj_key] = LocalMemoryModel(matrix=mat,
                minimap=plus.hud.MinimapModel(mat_layout_shape[0], mat_layout_shape[1]))
            key = f"{func_local_qname}-{obj_name}"
            container = LocalMemContainer(key, draft.local_matrices[obj_key], use_view=True)
            container.prop(border="1px solid blue")
            container.panel._event_plane.event_move.add_frontend_handler_v2(self.dm, partial(TritonSimModel._on_hover_pfl, key=obj_key))
            container.panel._event_plane.event_leave.add_frontend_handler_v2(self.dm, partial(TritonSimModel._on_hover_leave_pfl, key=obj_key))
            local_container = mui.MatchCase.binary_selection(True, container)
            local_container.bind_fields(condition=f"{draft.local_matrices[obj_key]} != null")
            if self._runner.runner.is_paused():
                bkpt = self._runner.runner.get_state().cur_bkpt
                assert bkpt is not None
                self._bkpt_handle_local_tensor_panel_local(obj_name, draft, bkpt)
        await self.tree.set_custom_layout(mui.HBox([
            local_container
        ]).prop(width="100%", height="100%", overflow="hidden"))


    def _remove_spec_suffix_of_func_uid(self, func_uid: str) -> str:
        parts = UniqueTreeId(func_uid).parts
        return parts[0] 

    async def hover_query(self, hqevent: mui.MonacoHoverQueryEvent) -> Optional[mui.MonacoHover]:
        if self._runner is None:
            return 
        lineno = hqevent.position.lineNumber

        col = hqevent.position.column
        if self._runner._mapper_new_to_old is not None:
            lineno_mapped = self._runner._mapper_new_to_old.bisect_mapped_lineno(lineno)
            if lineno_mapped != -1:
                lineno = lineno_mapped
        func_uid, node = self._runner.find_nearest_node_by_line_col(lineno, col - 1)
        if func_uid is not None and node is not None and isinstance(node, (pfl.PFLExpr, pfl.PFLArg)):
            node_expr = pfl.unparse_pfl_ast(node)
            loc = node.get_source_loc_checked()
            msg =f"### {node_expr}\n`{node.st}`"
            if not is_undefined(node.st.metadata):
                msg += f"\n{node.st.metadata}"
            if self._runner.runner.is_paused():
                stack = self._runner.runner.get_state().stack
                for s in stack:
                    if self._remove_spec_suffix_of_func_uid(s.node.uid) == self._remove_spec_suffix_of_func_uid(func_uid):
                        if isinstance(node, pfl.PFLName) and node.id in s.scope:
                            val = s.scope[node.id]
                            msg += f"\n\n**Value in stack**: \n`{val}`"

            return mui.MonacoHover([
                mui.MonacoMarkdownString(value=msg)
            ], range=mui.MonacoRange(loc[0], loc[1] + 1, loc[2], loc[3] + 1))
        return None 

    async def inlay_hint_query(self, ev: mui.MonacoInlayHintQueryEvent) -> Optional[mui.MonacoInlayHintList]:
        res: list[mui.MonacoInlayHint] = []
        if self._runner is None:
            return 
        if self._runner.runner.is_paused():
            mapper: Optional[SCDItem] = None
            new_value = ev.value
            query_range = ev.range
            if new_value is not None:
                new_value_lines = new_value.split("\n")
                # mapper = SourceChangeDiffCache.get_raw_item_for_mapping(new_value_lines, self._runner._content_lines, )
                mapper = SourceChangeDiffCache.get_raw_item_for_mapping(self._runner._content_lines, new_value_lines, )

            stack = self._runner.runner.get_state().stack
            for s in stack:
                finder = self._runner.finder_dict[s.node.uid]

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
        self._runner._mapper_new_to_old = SourceChangeDiffCache.get_raw_item_for_mapping(new_value_lines, self._runner._content_lines)
