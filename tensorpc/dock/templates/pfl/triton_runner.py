import asyncio
from functools import partial
from typing import Any, Dict, List, Union
from tensorpc.apps.ppcl import tsim
from tensorpc.core.moduleid import get_module_id_of_type
from tensorpc.core.pfl.evaluator import PFLBreakpoint, PFLCtrlFor, PFLCtrlBase
from tensorpc.dock import mui, three, plus, mark_create_layout, mark_did_mount, appctx
from typing import Annotated, Any, Optional, Union
from tensorpc.core import pfl

from tensorpc.apps.ppcl.backends import tritonstd
import triton.language as tl
import triton
import numpy as np 

def _matmul_kernel_test_fn() -> pfl.PFLInlineRunEnv:
    M = 256
    N = 256
    K = 128
    a = np.random.uniform(-1, 1, size=[M, K]).astype(np.float32)
    b = np.random.uniform(-1, 1, size=[K, N]).astype(a.dtype)
    c = np.empty([M, N], dtype=a.dtype)
    test_kwargs = {
        "a_ptr": a,
        "b_ptr": b,
        "c_ptr": c,
        "M": M,
        "N": N,
        "K": K,
        "stride_am": a.strides[0],
        "stride_ak": a.strides[1],
        "stride_bk": b.strides[0],
        "stride_bn": b.strides[1],
        "stride_cm": c.strides[0],
        "stride_cn": c.strides[1],
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }
    return pfl.PFLInlineRunEnv(test_kwargs)

@triton.jit
@tritonstd.mark_triton_compilable(inline_run_env_fn=_matmul_kernel_test_fn)
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M: int, N: int, K: int,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am: tl.constexpr, stride_ak: tl.constexpr,  #
        stride_bk: tl.constexpr, stride_bn: tl.constexpr,  #
        stride_cm: tl.constexpr, stride_cn: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    # wtf1 = ppcl.arange(0, BLOCK_SIZE_M)
    # wtf = wtf1 + pid_m
    # return
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    pfl.compiler_print_type(a_ptrs)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = tl.abs(accumulator.to(tl.float16))
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    # # return c_mask
    tl.store(c_ptrs, c, mask=c_mask)

class App:
    @mark_create_layout
    def my_layout(self):
        runner = tritonstd.parse_triton_compilable_to_runner(matmul_kernel.fn, do_meta_eval=True)
        # print(ast.ret_st)
        lib = runner._library
        compiled = lib.get_compiled_unit(matmul_kernel.fn)
        module = lib.get_module_by_func(matmul_kernel.fn)
        self._lib = lib
        finder = pfl.PFLTreeNodeFinder(compiled, (pfl.PFLName, pfl.PFLAttribute, pfl.PFLArg))
        self._finder = finder
        # mui.MonacoEditor.InlineComponent(
        #     mui.BlenderSlider(0, 10, 1).prop(width="50%"), 
        #     afterLineNumber=2, heightInPx=24)
        self.editor = mui.MonacoEditor(module.compile_info.code, "python", "")
        self.tree = plus.ObjectInspector(with_builtins=False)
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
            # mui.MonacoEditorAction(id="Stop", 
            #     label="Stop Current Run", contextMenuOrder=1.5,
            #     contextMenuGroupId="tensorpc-pfl-editor-action"),
        ]

        self.editor.prop(minWidth=0, minHeight=0, actions=editor_acts)
        self.editor.event_editor_hover_query.on(self.hover_query)
        self.editor.event_editor_action.on(self._handle_editor_acts)
        # self.editor.event_editor_inlay_hints_query.on(self.inlay_hint_query)
        self._runner = pfl.PFLAsyncRunner(lib)

        self._ctrl_to_slider: dict[int, mui.BlenderSlider] = {}
        self._runner.event_eval_stop.on(self._handle_eval_stop)

        # self._runner.event_new_ctrl_point.on(partial(self._handle_ctrl_point, is_new=True))
        # self._runner.event_delete_ctrl_point.on(partial(self._handle_ctrl_point, is_new=False))
        # self._runner.event_ctrl_point_change.on(self._handle_ctrl_point_update)

        self._runner.event_enter_bkpt.on(self._handle_enter_bkpt)
        self._runner.event_leave_bkpt.on(self._handle_leave_bkpt)
        self._runner_task: Optional[asyncio.Task] = None
        self._runner_task_ev = asyncio.Event()
        return mui.VBox([
            mui.HBox([
                self.editor.prop(flex=2),
                self.tree.prop(flex=1),
            ]).prop(flex=1)
        ]).prop(width="100%", height="100%", overflow="hidden")

    async def _handle_enter_bkpt(self, bkpt: PFLBreakpoint):
        await self.tree.tree.set_root_object_dict(bkpt.scope)
        await self.editor.set_decorations("common", [
            mui.MonacoModelDeltaDecoration(mui.MonacoRange(bkpt.node.source_loc[0], 1, bkpt.node.source_loc[0], 1),
            mui.MonacoModelDecoration(className="monaco-editor-content-decoration", isWholeLine=True))
        ])
        for ctrl_id, ctrl in self._runner.get_state().cur_ctrl_points.items():
            assert isinstance(ctrl, PFLCtrlFor), "Only PFLCtrlFor is supported in this editor"
            key = str(ctrl_id)
            if key not in self.editor.childs_complex.icomps:
                node = ctrl.node
                slider = mui.BlenderSlider(ctrl.range.start, ctrl.range.stop, ctrl.range.step)
                slider.event_change.on(partial(self._handle_slider, slider=slider, ctrl=ctrl))
                slider.prop(width="50%", showControlButton=True, showTotal=True, isInteger=True, forwardOnly=True, zIndex=10, alwaysShowButton=True)
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
            if int(k) not in self._runner.get_state().cur_ctrl_points:
                # remove the slider if the ctrl point is not in the current state
                self.editor.childs_complex.icomps.pop(k, None)
        await self.editor.set_new_layout(self.editor.childs_complex)

    async def _handle_eval_stop(self):
        self.editor.childs_complex.icomps.clear()
        await self.editor.set_new_layout(self.editor.childs_complex)

    async def _handle_leave_bkpt(self, bkpt: PFLBreakpoint):
        await self.tree.tree.set_root_object_dict({})

        await self.editor.set_decorations("common", [])
        for k, v in self.editor.childs_complex.icomps.items():
            slider = v.comp
            assert isinstance(slider, mui.BlenderSlider)
            await slider.send_and_wait(slider.update_event(disabled=True))


    async def _handle_slider(self, value: mui.NumberType, slider: mui.BlenderSlider, ctrl: PFLCtrlFor):
        if self._runner_task is None:
            return 
        old = slider.int()
        new = int(value)
        if new > old:
            ctrl.step = new
            ctrl.should_pause = False
            print("RELEASE", old, new)
            self._runner.release_breakpoint()
        
    async def _handle_editor_acts(self, act: mui.MonacoActionEvent):
        if act.action == "Run To":
            if act.selection is not None:
                lineno = act.selection.selections[0].startLineNumber
                stmt = self._lib.find_stmt_by_path_lineno(self._lib.get_module_by_func(matmul_kernel.fn).uid, lineno)
                if stmt is not None:
                    if self._runner.is_paused():
                        # if paused, continue from the current position
                        await self._runner.continue_until(lineno)
                    else:
                        self._runner_task_ev.clear()
                        assert self._runner._state.type == pfl.PFLAsyncRunnerStateType.IDLE, \
                            f"Runner is not in IDLE state, current state: {self._runner._state.type}"
                        with tsim.enter_tensorsim_context([0, 0, 0], [1, 1, 1]):
                            self._runner_task = asyncio.create_task(self._runner.run_until(lineno, get_module_id_of_type(matmul_kernel.fn), 
                                exit_event=self._runner_task_ev))

    async def hover_query(self, hqevent: mui.MonacoHoverQueryEvent) -> Optional[mui.MonacoHover]:
        lineno = hqevent.position.lineNumber
        col = hqevent.position.column
        node = self._finder.find_nearest_node_by_line_col(lineno, col - 1)
        if node is not None and isinstance(node, (pfl.PFLExpr, pfl.PFLArg)):
            node_expr = pfl.unparse_pfl_ast(node)
            loc = node.get_source_loc_checked()
            return mui.MonacoHover([
                mui.MonacoMarkdownString(value=f"### {node_expr}\n`{node.st}`\n{node.st.metadata}")
            ], range=mui.MonacoRange(loc[0], loc[1] + 1, loc[2], loc[3] + 1))
        return None 
