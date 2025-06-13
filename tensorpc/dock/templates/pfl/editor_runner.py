import asyncio
from functools import partial
from typing import Any, Dict, List, Union
from tensorpc.core.moduleid import get_module_id_of_type
from tensorpc.core.pfl.evaluator import PFLBreakpoint, PFLCtrlFor, PFLCtrlBase
from tensorpc.dock import mui, three, plus, mark_create_layout, mark_did_mount, appctx
from typing import Annotated, Any, Optional, Union
from tensorpc.core import pfl

def func3(a: float, b: float):
    if a > 10:
        d = 5 
    else:
        d = 3
    for j in range(10):
        for k in range(5):
            d += 1
    return d + b

class App:
    @mark_create_layout
    def my_layout(self):
        lib = pfl.parse_func_to_pfl_library(func3, backend="js")
        # print(ast.ret_st)
        compiled = lib.get_compiled_unit(func3)
        module = lib.get_module_by_func(func3)
        self._lib = lib
        finder = pfl.PFLTreeNodeFinder(compiled, (pfl.PFLName, pfl.PFLAttribute, pfl.PFLArg))
        self._finder = finder
        # mui.MonacoEditor.InlineComponent(
        #     mui.BlenderSlider(0, 10, 1).prop(width="50%"), 
        #     afterLineNumber=2, heightInPx=24)
        self.editor = mui.MonacoEditor(module.compile_info.code, "python", "")
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

        self.editor.prop(minWidth=0, minHeight=0, actions=editor_acts)
        self.editor.event_editor_hover_query.on(self.hover_query)
        self.editor.event_editor_action.on(self._handle_editor_acts)
        # self.editor.event_editor_inlay_hints_query.on(self.inlay_hint_query)
        self._runner = pfl.PFLAsyncRunner(lib)
        self._runner.event_new_ctrl_point.on(partial(self._handle_ctrl_point, is_new=True))
        self._runner.event_delete_ctrl_point.on(partial(self._handle_ctrl_point, is_new=False))
        self._runner.event_enter_bkpt.on(self._handle_enter_bkpt)
        self._runner.event_leave_bkpt.on(self._handle_leave_bkpt)
        self._runner_task: Optional[asyncio.Task] = None
        return mui.VBox([
            self.editor.prop(flex=1)
        ]).prop(width="100%", height="100%", overflow="hidden")

    async def _handle_enter_bkpt(self, bkpt: PFLBreakpoint):
        await self.editor.set_decorations("common", [
            mui.MonacoModelDeltaDecoration(mui.MonacoRange(bkpt.node.source_loc[0], 1, bkpt.node.source_loc[0], 1),
            mui.MonacoModelDecoration(className="monaco-editor-content-decoration", isWholeLine=True))
        ])

    async def _handle_leave_bkpt(self, bkpt: PFLBreakpoint):
        await self.editor.set_decorations("common", [])

    async def _handle_slider(self, value: mui.NumberType, slider: mui.BlenderSlider, ctrl: PFLCtrlFor):
        old = slider.int()
        new = int(value)
        print(new, old)

        if new > old:
            ctrl.step += ctrl.range.step
            self._runner.release_breakpoint()
        elif new < old:
            pass 

    async def _handle_ctrl_point(self, ctrl: PFLCtrlBase, is_new: bool):
        assert isinstance(ctrl, PFLCtrlFor), "Only PFLCtrlFor is supported in this editor"
        if is_new:
            node = ctrl.node
            slider = mui.BlenderSlider(ctrl.range.start, ctrl.range.stop, ctrl.range.step)
            slider.event_change.on(partial(self._handle_slider, slider=slider, ctrl=ctrl))
            slider.prop(width="50%", showControlButton=True, showTotal=True, isInteger=True)
            inline_comp = mui.MonacoEditor.InlineComponent(
                slider,
                afterLineNumber=node.source_loc[0], heightInPx=24)
            self.editor.childs_complex.icomps[str(id(ctrl.node))] = inline_comp
        else:
            node = ctrl.node
            self.editor.childs_complex.icomps.pop(str(id(node)), None)
        await self.editor.set_new_layout(self.editor.childs_complex)
        
    async def _handle_editor_acts(self, act: mui.MonacoActionEvent):
        if act.selection is not None:
            lineno = act.selection.selections[0].startLineNumber
            stmt = self._lib.find_stmt_by_path_lineno(self._lib.get_module_by_func(func3).uid, lineno)
            if stmt is not None:
                self._runner_task = asyncio.create_task(self._runner.run_until(lineno, get_module_id_of_type(func3), {
                    "a": 5,
                    "b": 3
                }))

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
