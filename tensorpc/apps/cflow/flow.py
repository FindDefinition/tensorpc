from tensorpc.core.datamodel.draftast import evaluate_draft_ast
from tensorpc.dock import mui, three, plus, appctx, mark_create_layout, flowui, models
from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.core.datamodel.draft import create_literal_draft
import tensorpc.core.datamodel.funcs as D
from functools import partial
from tensorpc.core.tree_id import UniqueTreeIdForTree

from typing import Optional, Any

from tensorpc.apps.cflow.binder import ComputeFlowBinder
from tensorpc.apps.cflow.model import ComputeFlowDrafts, ComputeFlowModelRoot, ComputeFlowNodeModel, ComputeNodeType, DetailType, InlineCode, get_compute_flow_drafts
from tensorpc.apps.cflow.nodes.cnode.default_code import get_default_custom_node_code
from tensorpc.apps.cflow.nodes.cnode.handle import HandleTypePrefix
from tensorpc.dock.components.flowplus.style import default_compute_flow_css
from tensorpc.utils.code_fmt import PythonCodeFormatter
from tensorpc.apps.cflow.nodes.cnode.registry import NODE_REGISTRY, parse_code_to_compute_cfg
import tensorpc.apps.cflow.nodes.defaultnodes

_SYS_NODE_PREFIX = "sys-"
_USER_NODE_PREFIX = "user-"



class ComputeFlow(mui.FlexBox):

    def __init__(self):
        items = [
            mui.MenuItem(id=f"{_SYS_NODE_PREFIX}markdown", label="Add Markdown"),
            mui.MenuItem(id=f"{_SYS_NODE_PREFIX}compute", label="Add Compute"),
        ]
        if NODE_REGISTRY.global_dict:
            items.append(mui.MenuItem(id="divider", divider=True))
            for key, cfg in NODE_REGISTRY.global_dict.items():
                items.append(mui.MenuItem(id=f"{_USER_NODE_PREFIX}{key}", label=cfg.name))

        self.graph = flowui.Flow([], [], [
            flowui.MiniMap(),
            flowui.Controls(),
            flowui.Background(),
        ]).prop(paneContextMenuItems=items, zoomActivationKeyCode="z",
                                        disableKeyboardA11y=True,
                                        zoomOnScroll=False,
                                        preventCycle=True)
        target_conn_valid_map = {
            HandleTypePrefix.Input: {
                # each input (target) can only connect one output (source)
                HandleTypePrefix.Output:
                1
            },
            HandleTypePrefix.SpecialDict: {
                # inf number of handle
                HandleTypePrefix.Output: -1
            },
            HandleTypePrefix.DriverInput: {
                HandleTypePrefix.DriverOutput: -1
            }
        }
        self.graph.prop(targetValidConnectMap=target_conn_valid_map)
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
                ]).prop(height="100%", width="100%", overflow="hidden").update_raw_props(default_compute_flow_css())),
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
        self.code_editor.prop(actions=editor_acts, height="100%")
        # self.code_editor.event_editor_action.on(self._handle_editor_action)
        flow_with_editor = mui.Allotment(mui.Allotment.ChildDef([
            mui.Allotment.Pane(mui.HBox([
                self.graph
            ]).prop(height="100%", width="100%", overflow="hidden").update_raw_props(default_compute_flow_css())),
            mui.Allotment.Pane(mui.HBox([
                self.code_editor
            ]).prop(height="100%", width="100%", overflow="hidden"), visible=False),
        ])).prop(vertical=False, defaultSizes=[200, 100])
        global_container = mui.Allotment(mui.Allotment.ChildDef([
            mui.Allotment.Pane(flow_with_editor),
            mui.Allotment.Pane(detail_ct),
        ])).prop(vertical=True)
        self.dm = mui.DataModel(ComputeFlowModelRoot(edges={}, nodes={}), [
            mui.VBox([
                mui.HBox([
                    path_breadcrumb
                ]).prop(minHeight="24px"),
                global_container,
            ]).prop(flex=1),
        ])
        draft = self.dm.get_draft()
        flow_draft = get_compute_flow_drafts(draft)
        flow_with_editor.bind_fields(visibles=f"[`true`, {flow_draft.show_editor}]")
        global_container.bind_fields(visibles=f"[`true`, {flow_draft.show_detail}]")
        detail_ct.bind_fields(condition=flow_draft.selected_node_detail_type)
        self.graph.event_pane_context_menu.on(partial(self.add_node, target_flow_draft=flow_draft.cur_model))
        self.graph_preview.event_pane_context_menu.on(partial(self.add_node, target_flow_draft=flow_draft.preview_model))
        path_breadcrumb.bind_fields(value=f"concat(`[\"root\"]`, {draft.cur_path}[1::3])")
        path_breadcrumb.event_change.on(self.handle_breadcrumb_click)
        self.code_editor.bind_draft_change_uncontrolled(self.dm, flow_draft.selected_node_code, 
            path_draft=flow_draft.selected_node_code_path, 
            lang_draft=flow_draft.selected_node_code_language,
            save_event_prep=partial(self._process_save_ev_before_save, drafts=flow_draft))
        binder = ComputeFlowBinder(self.graph, self.graph_preview, flow_draft)
        binder.bind_flow_comp_with_datamodel(self.dm)
        super().__init__([self.dm])

        self.prop(width="100%", height="100%", overflow="hidden")

    def _process_save_ev_before_save(self, ev: mui.MonacoEditorSaveEvent, drafts: ComputeFlowDrafts):
        cur_flow_draft = drafts.cur_model
        sel_node = drafts.selected_node
        cur_selected_node_draft = cur_flow_draft.selected_node
        if ev.lang == "python":
            # compute node code, parse and get new state
            # TODO if old and new state are same, don't update
            cfg = parse_code_to_compute_cfg(ev.value)
            if cfg.state_dcls is not None:
                state = cfg.state_dcls()
                cur_flow_draft.node_states[cur_selected_node_draft] = state
            sel_node.name = cfg.name
            sel_node.key = cfg.key
            sel_node.moduleId = cfg.module_id

    def handle_breadcrumb_click(self, data: list[str]):
        logic_path = data[1:] # remove root
        res_path: list[str] = []
        for item in logic_path:
            res_path.extend(['nodes', item, 'flow'])
        draft = self.dm.get_draft()
        draft.cur_path = res_path

    def add_node(self, data: flowui.PaneContextMenuEvent, target_flow_draft: Any):
        item_id = data.itemId
        node_type = item_id
        pos = data.flowPosition
        # print(f"Add Node: {node_type} at {pos}")
        if pos is None:
            return 
        target_flow = D.evaluate_draft(target_flow_draft, self.dm.model)
        assert target_flow is not None
        node_id = target_flow.make_unique_node_name(node_type)

        if item_id.startswith(_SYS_NODE_PREFIX):
            node_type = item_id[len(_SYS_NODE_PREFIX):]
            if node_type == "markdown":
                new_node = ComputeFlowNodeModel(nType=ComputeNodeType.MARKDOWN, type="app", id=node_id, position=pos, impl=InlineCode(code="## MarkdownNode"))
            elif node_type == "compute":
                code = get_default_custom_node_code()
                parsed_cfg = parse_code_to_compute_cfg(code)
                new_node = ComputeFlowNodeModel(nType=ComputeNodeType.COMPUTE, type="app", id=node_id, position=pos, impl=InlineCode(code=code),
                    name=parsed_cfg.name, key=parsed_cfg.key, moduleId=parsed_cfg.module_id)
                target_flow_draft.node_states[node_id] = {}
            else:
                raise NotImplementedError
        else:
            node_type = item_id[len(_USER_NODE_PREFIX):]
            cfg = NODE_REGISTRY.global_dict[node_type]
            new_node = ComputeFlowNodeModel(
                nType=ComputeNodeType.COMPUTE, type="app", id=node_id, position=pos, moduleId=cfg.module_id, key=cfg.key,
                    name=cfg.name)
            if cfg.state_dcls is not None:
                target_flow_draft.node_states[node_id] = cfg.state_dcls()

        target_flow_draft.nodes[node_id] = new_node
