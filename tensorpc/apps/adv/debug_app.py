from tensorpc.apps.adv.codemgr.flow import ADVProjectBackendManager
from tensorpc.apps.adv.nodes.base import BaseNodeWrapper
from tensorpc.constants import PACKAGE_ROOT
from tensorpc.dock import mui, three, plus, appctx, mark_create_layout, flowui, models
from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.core.datamodel.draft import create_literal_draft
import tensorpc.core.datamodel.funcs as D
from functools import partial
from tensorpc.core.tree_id import UniqueTreeIdForTree

from typing import Optional, Any
from tensorpc.apps.adv.model import ADVEdgeModel, ADVHandlePrefix, ADVNodeHandle, ADVNodeType, ADVRoot, ADVProject, ADVNodeModel, ADVFlowModel, InlineCode
from tensorpc.core.datamodel.draft import (get_draft_pflpath)
from tensorpc.dock.components.flowplus.style import default_compute_flow_css

def _test_model_simple():
    return ADVProject(
        flow=ADVFlowModel(nodes={
            "n1": ADVNodeModel(
                id="n1", 
                position=flowui.XYPosition(0, 0), 
                name="Node 1",
                impl=InlineCode(),
                handles=[
                    ADVNodeHandle(
                        id="in1",
                        name="Input 1",
                        type="number",
                        is_input=True,
                    ),
                    ADVNodeHandle(
                        id="out1",
                        name="Output 1",
                        type="number",
                        is_input=False,
                    ),

                ]
            ),

            "n2": ADVNodeModel(id="n2", position=flowui.XYPosition(0, 100), name="Node 2 (Nested)",
                flow=ADVFlowModel(nodes={
                    "n2_1": ADVNodeModel(id="n2_1", position=flowui.XYPosition(0, 0), name="Nested Node 1"),
                }, edges={}),
            ),
            "n3": ADVNodeModel(
                id="n3", 
                position=flowui.XYPosition(200, 0), 
                name="Node 3",
                impl=InlineCode(),
                handles=[
                    ADVNodeHandle(
                        id="in1",
                        name="Input 1",
                        type="number",
                        is_input=True,
                    ),
                    ADVNodeHandle(
                        id="out1",
                        name="Output 1",
                        type="number",
                        is_input=False,
                    ),

                ]
            ),

            "n1-ref": ADVNodeModel(id="n1-ref", position=flowui.XYPosition(0, 200), name="Node 1 (ref)",
                ref_node_id="n1"),


        }, edges={
            "e0": ADVEdgeModel(
                id="e0", 
                source="n1",
                sourceHandle="out1",
                target="n3",
                targetHandle="in1",
                isAutoEdge=True,
            )
        }),
        import_prefix="tensorpc.adv.test_project",
        path=str(PACKAGE_ROOT / "adv" / "test_project"),
    )


def _test_model_symbol_group():
    global_script_0 = f"""
import numpy as np 
    """
    symbolgroup0 = f"""
@dataclasses.dataclass
class SymbolGroup0:
    a: int 
    b: float
    c: float
    d: int
    """

    fragment0 = f"""
ADV.mark_outputs("c")
return a + b
    """
    fragment1 = f"""
ADV.mark_outputs("d")
return c + a
    """

    return ADVProject(
        flow=ADVFlowModel(nodes={
            "g1": ADVNodeModel(
                id="g1", 
                nType=ADVNodeType.GLOBAL_SCRIPT,
                position=flowui.XYPosition(0, 200), 
                name="GlobalScript0",
                impl=InlineCode(global_script_0),
            ),

            "n1": ADVNodeModel(
                id="n1", 
                nType=ADVNodeType.SYMBOLS,
                position=flowui.XYPosition(0, 0), 
                name="SymbolGroup",
                impl=InlineCode(symbolgroup0),
            ),
            "f0": ADVNodeModel(
                id="f0", 
                nType=ADVNodeType.FRAGMENT,
                position=flowui.XYPosition(200, 0), 
                name="add_func",
                inline_subflow_name="inline0",
                impl=InlineCode(fragment0),
            ),
            "f1": ADVNodeModel(
                id="f1", 
                nType=ADVNodeType.FRAGMENT,
                position=flowui.XYPosition(400, 100), 
                name="add_func2",
                inline_subflow_name="inline0",

                impl=InlineCode(fragment1),
            ),


        }, edges={
        }),
        import_prefix="tensorpc.adv.test_project",
        path=str(PACKAGE_ROOT / "adv" / "test_project"),
    )


class App:
    @mark_create_layout
    def my_layout(self):
        adv_proj = {
            # "project": _test_model_simple()
            "project": _test_model_symbol_group()

        }
        nid_to_path, nid_to_fpath = adv_proj["project"].assign_path_to_all_node()
        adv_proj["project"].node_id_to_path = nid_to_path
        adv_proj["project"].node_id_to_frontend_path = nid_to_fpath
        adv_proj["project"].update_ref_path(nid_to_fpath)
        
        
        model = ADVRoot(cur_adv_project="project", adv_projects=adv_proj)
        node_cm_items = [
            mui.MenuItem(id="nested", label="Enter Nested"),
        ]
        items = [
            mui.MenuItem(id="plain", label="Add Plain Node"),
            mui.MenuItem(id="nested", label="Add Nested Flow Node"),
        ]

        self.graph = flowui.Flow([], [], [
            flowui.MiniMap(),
            flowui.Controls(),
            flowui.Background(),
        ]).prop(nodeContextMenuItems=node_cm_items, paneContextMenuItems=items)
        target_conn_valid_map = {
            ADVHandlePrefix.Input: {
                # each input (target) can only connect one output (source)
                ADVHandlePrefix.Output: 1
            },
            ADVHandlePrefix.OutIndicator: {
                # output indicator can connect to unlimited inputs
                ADVHandlePrefix.Output: -1
            },
        }
        self.graph.prop(targetValidConnectMap=target_conn_valid_map)

        self.graph.event_node_context_menu.on(self.handle_node_cm)
        path_breadcrumb = mui.Breadcrumbs([]).prop(keepHistoryPath=True)
        detail = mui.JsonEditor()
        editor = mui.MonacoEditor("", "python", "default").prop(flex=1, minHeight=0, minWidth=0)
        editor_ct = mui.MatchCase.binary_selection(True, mui.VBox([
            editor.prop(flex=1),
        ]).prop(flex=1, overflow="hidden"))

        detail_ct = mui.MatchCase.binary_selection(True, mui.VBox([
            mui.HBox([
                detail,
            ]).prop(flex=1, overflow="hidden"),
            editor_ct,
        ]).prop(flex=1, overflow="hidden"))
        graph_container = mui.VBox([
                mui.HBox([
                    path_breadcrumb
                ]).prop(minHeight="24px"),
                self.graph,
            ]).prop(flex=1)
        self.dm = mui.DataModel(model, [
            graph_container,
            detail_ct,
        ])
        manager = ADVProjectBackendManager(lambda: self.dm.get_model().adv_projects["project"].flow)
        manager.sync_project_model()
        manager.parse_all()
        manager.init_all_nodes()
        import rich 
        rich.print(dataclasses.asdict(self.dm.get_model().adv_projects["project"].flow))
        self._manager = manager
        graph_container.update_raw_props(default_compute_flow_css())

        draft = self.dm.get_draft()
        cur_root_proj = draft.draft_get_cur_adv_project()
        cur_model_draft = draft.draft_get_cur_model()
        self.graph.event_pane_context_menu.on(partial(self.handle_context_menu, target_flow_draft=cur_model_draft))
        # self.graph_preview.event_pane_context_menu.on(partial(self.add_node, target_flow_draft=preview_model_draft))
        # draft only support raw path, so we use [1::3] to convert from raw path to real node path
        # we also need to add root to the beginning
        path_breadcrumb.bind_fields(value=f"[\"root\"] + {cur_root_proj.cur_path}[1::3]")
        path_breadcrumb.event_change.on(self.handle_breadcrumb_click)
        # since we may switch preview flow repeatedly, we need to set a unique flow id to avoid handle wrong frontend event
        # e.g. the size/position change event is debounced
        detail_ct.bind_fields(condition=f"{cur_root_proj.draft_get_selected_node()} is not None")

        binder = models.flow.BaseFlowModelBinder(
            self.graph, 
            self.dm.get_model,
            cur_model_draft, 
            self.model_to_ui_node,
            to_ui_edge=self.model_to_ui_edge,
            to_model_edge=self.ui_to_model_edge,
            flow_uid_getter=lambda: self.dm.get_model().get_cur_flow_uid(),
            debug_id="main_flow")
        binder.bind_flow_comp_with_base_model(self.dm, cur_model_draft.selected_nodes)
        # detail.bind_fields(data=cur_root_proj.draft_get_selected_node())
        detail.bind_pfl_query(self.dm, data=(ADVRoot.get_cur_node_flows, "selectedNode"))
        has_code, code_draft, path_draft = cur_root_proj.draft_get_node_impl_editor(cur_root_proj.draft_get_selected_node().id)
        editor.bind_draft_change_uncontrolled(code_draft, path_draft=path_draft)
        # editor_ct.bind_fields(condition=has_code)
        editor_ct.bind_pfl_query(self.dm, condition=(ADVRoot.get_cur_node_flows, "enableCodeEditor"))
        # self.dm.debug_print_draft_change(has_code)

        return mui.HBox([
            self.dm,
        ]).prop(width="100%", height="100%", overflow="hidden")
    
    def _get_preview_flow_uid(self, path_draft):
        path = D.evaluate_draft(path_draft, self.dm.model)
        if path is None:
            return "root"
        return UniqueTreeIdForTree.from_parts(path).uid_encoded

    def model_to_ui_node(self, flow: ADVFlowModel, node_id: str):
        node = flow.nodes[node_id]
        # draft = self.dm.get_draft()
        comp = mui.VBox([
            mui.Typography(f"Node-{node.name}" if node.flow is None else "Nested Flow"),
        ])
        comp = BaseNodeWrapper(
            node_id,
            self.dm,
            ADVNodeType(node.nType),
        )
        ui_node = flowui.Node(type="app", 
            id=node.id, 
            data=flowui.NodeData(component=comp, label=node.name), 
            position=node.position)
        return ui_node

    def model_to_ui_edge(self, edge: ADVEdgeModel):
        ui_edge = flowui.Edge(
            id=edge.id,
            source=edge.source,
            target=edge.target,
            sourceHandle=edge.sourceHandle,
            targetHandle=edge.targetHandle,
        )
        if edge.isAutoEdge:
            ui_edge.style = {
                "strokeDasharray": "5",
            }
        return ui_edge

    def ui_to_model_edge(self, ui_edge: flowui.Edge) -> ADVEdgeModel:
        edge = ADVEdgeModel(
            id=ui_edge.id,
            source=ui_edge.source,
            target=ui_edge.target,
            sourceHandle=ui_edge.sourceHandle,
            targetHandle=ui_edge.targetHandle,
            isAutoEdge=False,
        )
        return edge

    async def handle_node_cm(self, data: flowui.NodeContextMenuEvent):
        item_id = data.itemId
        node_id = data.nodeId

        cur_path_val = self.dm.model.get_cur_adv_project().cur_path
        new_path_val = cur_path_val + ['nodes', node_id, 'flow']
        new_logic_path = new_path_val[1::3]
        # validate node contains nested flow
        cur_model = self.dm.model.get_cur_adv_project().flow
        for item in new_logic_path:
            cur_model = cur_model.nodes[item].flow
            if cur_model is None:
                return

        draft = self.dm.get_draft().draft_get_cur_adv_project()
        # we have to clear selection before switch flow because xyflow don't support controlled selection.
        # xyflow will clear previous selection and send clear-selection event when flow is switched.
        D.getitem_path_dynamic(draft.flow, draft.cur_path, Optional[ADVFlowModel]).selected_nodes = []
        draft.cur_path = new_path_val

    def handle_breadcrumb_click(self, data: list[str]):
        logic_path = data[1:] # remove root
        res_path: list[str] = []
        for item in logic_path:
            res_path.extend(['nodes', item, 'flow'])
        draft = self.dm.get_draft().draft_get_cur_adv_project()
        # we have to clear selection before switch flow because xyflow don't support controlled selection.
        # xyflow will clear previous selection and send clear-selection event when flow is switched.
        D.getitem_path_dynamic(draft.flow, draft.cur_path, Optional[ADVFlowModel]).selected_nodes = []
        draft.cur_path = res_path

    async def handle_context_menu(self, data: flowui.PaneContextMenuEvent, target_flow_draft: Any):
        
        cur_model = self.dm.model.get_cur_adv_project().flow
        node_ids = [n.id for n in cur_model.nodes.values()]
        await self.graph.update_node_internals(node_ids)


