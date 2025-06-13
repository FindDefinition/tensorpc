from typing import Any, Dict, List, Union
from tensorpc.core.pfl.evaluator import PFLCtrlFor
from tensorpc.dock import mui, three, plus, mark_create_layout, mark_did_mount, appctx
from tensorpc.apps.ppcl.std import Tensor, PointerTensor, ppcl
from tensorpc.apps.ppcl.core import TensorMeta, DTypeEnum, ConstExprMeta
from typing import Annotated, Any, Optional, Union
from tensorpc.core import pfl

def matmul_kernel(
        # Pointers to matrices
        a_ptr: PointerTensor, b_ptr: PointerTensor, c_ptr: PointerTensor,
        # Matrix dimensions
        M: int, N: int, K: int,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am: Annotated[int, ConstExprMeta()], stride_ak: Annotated[int, ConstExprMeta()],  #
        stride_bk: Annotated[int, ConstExprMeta()], stride_bn: Annotated[int, ConstExprMeta()],  #
        stride_cm: Annotated[int, ConstExprMeta()], stride_cn: Annotated[int, ConstExprMeta()],
        # Meta-parameters
        BLOCK_SIZE_M: Annotated[int, ConstExprMeta()], BLOCK_SIZE_N: Annotated[int, ConstExprMeta()], BLOCK_SIZE_K: Annotated[int, ConstExprMeta()],  #
        GROUP_SIZE_M: Annotated[int, ConstExprMeta()],  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = ppcl.program_id(0)
    num_pid_m = ppcl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = ppcl.cdiv(N, BLOCK_SIZE_N)
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
    offs_am = (pid_m * BLOCK_SIZE_M + ppcl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + ppcl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = ppcl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    pfl.compiler_print_type(a_ptrs)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = ppcl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=ppcl.float32)
    for k in range(0, ppcl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = ppcl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = ppcl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = ppcl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = ppcl.abs(accumulator.to(ppcl.float16))
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + ppcl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + ppcl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    # # return c_mask
    ppcl.store(c_ptrs, c, mask=c_mask)

def func3(a: float, b: float):
    if a > 10:
        d = 5 
    else:
        d = 3
    for j in range(10):
        d += 1
    return d + b

class App:
    @mark_create_layout
    def my_layout(self):
        lib = pfl.parse_func_to_pfl_library(func3, backend="ppcl")
        # print(ast.ret_st)
        compiled = lib.get_compiled_unit(func3)
        module = lib.get_module_by_func(func3)
        self._lib = lib
        # pfl.metaeval_total_tree(compiled, {
        #     "BLOCK_SIZE_M": 128,
        #     "BLOCK_SIZE_N": 128,
        #     "BLOCK_SIZE_K": 32,
        #     "GROUP_SIZE_M": 8,
        #     "a_ptr": TensorMeta([], DTypeEnum.float16, is_pointer=True),
        #     "b_ptr": TensorMeta([], DTypeEnum.float16, is_pointer=True),
        #     "c_ptr": TensorMeta([], DTypeEnum.float16, is_pointer=True),
        # }, backend="ppcl", code_for_error=module.compile_info.code)
        finder = pfl.PFLTreeNodeFinder(compiled, (pfl.PFLName, pfl.PFLAttribute, pfl.PFLArg))
        self._finder = finder
        # mui.MonacoEditor.InlineComponent(
        #     mui.BlenderSlider(0, 10, 1).prop(width="50%"), 
        #     afterLineNumber=2, heightInPx=24)
        self.editor = mui.MonacoEditor(module.compile_info.code, "python", "")
        editor_acts: list[mui.MonacoEditorAction] = [
            mui.MonacoEditorAction(id="Run To", 
                label="Run Towards Here", contextMenuOrder=1.5,
                contextMenuGroupId="tensorpc-pfl-editor-action", 
                keybindings=[([mui.MonacoKeyMod.Shift], 3)]),
        ]

        self.editor.prop(minWidth=0, minHeight=0, actions=editor_acts)
        self.editor.event_editor_hover_query.on(self.hover_query)
        self.editor.event_editor_action.on(self._handle_editor_acts)
        # self.editor.event_editor_inlay_hints_query.on(self.inlay_hint_query)
        self._runner = pfl.PFLAsyncRunner(lib, ctrl_point_cb=self._handle_ctrl_point)
        return mui.VBox([
            self.editor.prop(flex=1)
        ]).prop(width="100%", height="100%", overflow="hidden")

    # @mark_did_mount
    # async def _on_init(self):
    #     await self.panel.inspector.add_object_to_tree(self.tutorials,
    #                                           key="tutorials",
    #                                           expand_level=2)
    async def _handle_ctrl_point(self, ctrl: PFLCtrlFor, is_new: bool):
        if is_new:
            node = ctrl.node
            inline_comp = mui.MonacoEditor.InlineComponent(
                mui.BlenderSlider(ctrl.range.start, ctrl.range.stop, ctrl.range.step).prop(width="50%"),
                afterLineNumber=node.source_loc[0], heightInPx=24, afterColumn=node.source_loc[1] + 1)
            self.editor.childs_complex.icomps[str(id(ctrl.node))] = inline_comp
        else:
            node = ctrl.node
            self.editor.childs_complex.icomps.pop(str(id(node)), None)
        await self.editor.set_new_layout(self.editor.childs_complex)
        
    async def _handle_editor_acts(self, act: mui.MonacoActionEvent):
        if act.selection is not None:
            lineno = act.selection.selections[0].startLineNumber
            stmt = self._lib.find_stmt_by_path_lineno(self._lib.get_module_by_func(func3).uid, lineno)
            print(stmt)

    async def hover_query(self, hqevent: mui.MonacoHoverQueryEvent) -> Optional[mui.MonacoHover]:
        lineno = hqevent.position.lineNumber
        col = hqevent.position.column
        node = self._finder.find_nearest_node_by_line_col(lineno, col - 1)
        print(lineno, col)
        if node is not None and isinstance(node, (pfl.PFLExpr, pfl.PFLArg)):
            node_expr = pfl.unparse_pfl_ast(node)
            loc = node.get_source_loc_checked()
            print(loc, type(node))
            return mui.MonacoHover([
                mui.MonacoMarkdownString(value=f"### {node_expr}\n`{node.st}`\n{node.st.metadata}")
            ], range=mui.MonacoRange(loc[0], loc[1] + 1, loc[2], loc[3] + 1))
        return None 

    async def inlay_hint_query(self, ev: mui.MonacoInlayHintQueryEvent) -> Optional[mui.MonacoInlayHintList]:
        res: list[mui.MonacoInlayHint] = []
        for node in self._finder._all_nodes:
            if isinstance(node, pfl.PFLExpr) and node.st.has_metadata():
                meta = node.st.metadata_checked
                if isinstance(meta, TensorMeta):
                    res.append(mui.MonacoInlayHint(
                        label=f": {meta}",
                        position=mui.MonacoPosition(node.get_source_loc_checked()[2], node.get_source_loc_checked()[3] + 1),
                        kind=1,
                    ))
        if res:
            return mui.MonacoInlayHintList(res)
        return None 

