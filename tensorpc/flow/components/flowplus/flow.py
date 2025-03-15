from tensorpc.flow import mui, three, plus, appctx, mark_create_layout, flowui, models
from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.core.datamodel.draft import create_literal_draft
import tensorpc.core.datamodel.funcs as D
from functools import partial
from tensorpc.core.tree_id import UniqueTreeIdForTree

from typing import Optional, Any

from tensorpc.flow.components.flowplus.model import ComputeFlowModelRoot, DetailType, get_compute_flow_drafts
from tensorpc.utils.code_fmt import PythonCodeFormatter


class ComputeFlow(mui.FlexBox):

    def __init__(self):
        items = [
            mui.MenuItem(id="plain", label="Add Plain Node"),
            mui.MenuItem(id="nested", label="Add Nested Flow Node"),
        ]

        self.graph = flowui.Flow([], [], [
            flowui.MiniMap(),
            flowui.Controls(),
            flowui.Background(),
        ]).prop(paneContextMenuItems=items, zoomActivationKeyCode="z",
                                        disableKeyboardA11y=True,
                                        zoomOnScroll=False,
                                        preventCycle=True)
        self.graph_preview = flowui.Flow([], [], [
            flowui.MiniMap(),
            flowui.Controls(),
            flowui.Background(),
        ]).prop(paneContextMenuItems=items, zoomActivationKeyCode="z",
                                        disableKeyboardA11y=True,
                                        zoomOnScroll=False,
                                        preventCycle=True)

        path_breadcrumb = mui.Breadcrumbs([]).prop(keepHistoryPath=True)
        self.user_detail = mui.VBox([]).prop(flex=1, overflow="hidden")
        detail_ct = mui.MatchCase(
            [
                mui.MatchCase.Case(DetailType.NONE.value, mui.VBox([])),
                mui.MatchCase.Case(DetailType.SUBFLOW.value, mui.VBox([
                    self.graph_preview,
                ]).prop(flex=1, overflow="hidden")),
                mui.MatchCase.Case(DetailType.USER_LAYOUT.value, self.user_detail),
            ]
        )
        self.code_editor = mui.MonacoEditor("", "python",
                                            "default").prop(flex=1,
                                                            minHeight=0,
                                                            minWidth=0)
        self._code_fmt = PythonCodeFormatter()
        editor_acts: list[mui.MonacoEditorAction] = []
        for backend in self._code_fmt.get_all_supported_backends():
            editor_acts.append(
                mui.MonacoEditorAction(id=f"FormatCode-{backend}",
                                       label=f"Format Code ({backend})",
                                       contextMenuOrder=1.5,
                                       contextMenuGroupId="tensorpc-flow-editor-action",
                                       userdata={"backend": backend})
            )
        self.code_editor.prop(actions=editor_acts)
        # self.code_editor.event_editor_action.on(self._handle_editor_action)

        detail_ct_with_editor = mui.Allotment(mui.Allotment.ChildDef([
            mui.Allotment.Pane(mui.HBox([
                detail_ct
            ]).prop(height="100%", width="100%", overflow="hidden")),
            mui.Allotment.Pane(self.code_editor.prop(height="100%")),
        ])).prop(flex=1, minHeight=0, vertical=False)


        global_container = mui.Allotment(mui.Allotment.ChildDef([
            mui.Allotment.Pane(self.graph),
            mui.Allotment.Pane(detail_ct_with_editor),
        ])).prop(flex=1, minHeight=0)

        self.dm = mui.DataModel(ComputeFlowModelRoot(edges={}, nodes={}), [
            mui.VBox([
                mui.HBox([
                    path_breadcrumb
                ]).prop(minHeight="24px"),
                global_container,
            ]).prop(flex=1),
            detail_ct,
        ])
        draft = self.dm.get_draft()
        flow_draft = get_compute_flow_drafts(draft)
        detail_ct_with_editor.bind_fields(visibles=f"[`true`, {flow_draft.show_detail}]")

        detail_ct.bind_fields(condition=flow_draft.selected_node_detail_type)
        self.graph.event_pane_context_menu.on(partial(self.add_node, target_flow_draft=flow_draft.cur_model))
        self.graph_preview.event_pane_context_menu.on(partial(self.add_node, target_flow_draft=flow_draft.preview_model))
        path_breadcrumb.bind_fields(value=f"concat(`[\"root\"]`, {draft.cur_path}[1::3])")
        path_breadcrumb.event_change.on(self.handle_breadcrumb_click)
        # since we may switch preview flow repeatedly, we need to set a unique flow id to avoid handle wrong frontend event
        # e.g. the size/position change event is debounced
        binder = models.flow.BaseFlowModelBinder(
            self.graph, self.dm.get_model, 
            cur_model_draft, 
            self.model_to_ui_node,
            flow_uid_getter=lambda: self.dm.get_model().get_uid_from_path())
        binder.bind_flow_comp_with_base_model(self.dm, cur_model_draft.selected_node)
        preview_binder = models.flow.BaseFlowModelBinder(
            self.graph_preview, self.dm.get_model, 
            preview_model_draft, 
            self.model_to_ui_node,
            flow_uid_getter=partial(self._get_preview_flow_uid, prev_path_draft))
        preview_binder.bind_flow_comp_with_base_model(self.dm, preview_model_draft.selected_node)

        return mui.HBox([
            self.dm,
        ]).prop(width="100%", height="100%", overflow="hidden")

    def handle_breadcrumb_click(self, data: list[str]):
        logic_path = data[1:] # remove root
        res_path: list[str] = []
        for item in logic_path:
            res_path.extend(['nodes', item, 'flow'])
        draft = self.dm.get_draft()
        draft.cur_path = res_path
