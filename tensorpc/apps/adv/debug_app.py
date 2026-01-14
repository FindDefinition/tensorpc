from tensorpc.apps.adv.codemgr.flow import ADV_MAIN_FLOW_NAME, ADVProjectBackendManager
from tensorpc.apps.adv.codemgr.proj_parse import ADVProjectParser
from tensorpc.apps.adv.nodes.base import BaseNodeWrapper, IndicatorWrapper
from tensorpc.constants import PACKAGE_ROOT
from tensorpc.core.datamodel.events import DraftChangeEvent
from tensorpc.dock import mui, three, plus, appctx, mark_create_layout, flowui, models
from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.core.datamodel.draft import create_draft_type_only, create_literal_draft
import tensorpc.core.datamodel.funcs as D
from functools import partial
from tensorpc.core.tree_id import UniqueTreeIdForTree

from typing import Literal, Optional, Any
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

def _get_simple_flow(name: str, op: Literal["+", "-", "*", "/"], sym_import_path: list[str]):
    fragment = f"""
ADV.mark_outputs("c")
return a {op} b
    """
    if op == "+":
        op_name = "add"
    elif op == "-":
        op_name = "sub"
    elif op == "*":
        op_name = "mul"
    elif op == "/":
        op_name = "div"
    else:
        raise ValueError(f"Unsupported op: {op}")
    return ADVFlowModel(nodes={
        "sym_def": ADVNodeModel(
            id="sym_def", 
            nType=ADVNodeType.SYMBOLS,
            position=flowui.XYPosition(0, 0), 
            ref_node_id="sym_def",
            ref_import_path=sym_import_path,
        ),
        "func": ADVNodeModel(
            id="func", 
            nType=ADVNodeType.FRAGMENT,
            position=flowui.XYPosition(200, 0), 
            name=f"{name}_func",
            inlinesf_name=name,
            impl=InlineCode(fragment),
        ),
        "o0": ADVNodeModel(
            id="o0", 
            nType=ADVNodeType.OUT_INDICATOR,
            position=flowui.XYPosition(400, 0), 
            name="Outputs",
        ),
        }, edges={
            "e0": ADVEdgeModel(
                id="e0", 
                source="func",
                sourceHandle=f"{ADVHandlePrefix.Output}-c",
                target="o0",
                targetHandle=f"{ADVHandlePrefix.OutIndicator}-outputs",
                isAutoEdge=False,
            )
        })


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

    fragment_add = f"""
ADV.mark_outputs("c")
return a + b
    """
    fragment1 = f"""
ADV.mark_outputs("d->D")
return c + a
    """

    nested_model_symbol_lib = ADVNodeModel(
        id="sym_lib", 
        nType=ADVNodeType.FRAGMENT,
        position=flowui.XYPosition(0, 200), 
        name="sym_lib",
        flow=ADVFlowModel(nodes={
            "sym_def": ADVNodeModel(
                id="sym_def", 
                nType=ADVNodeType.SYMBOLS,
                position=flowui.XYPosition(0, 0), 
                name="sym_def",
                impl=InlineCode(symbolgroup0),
            ),
        }, edges={})
    )
    nested_model_op_lib = ADVNodeModel(
        id="op_lib", 
        nType=ADVNodeType.FRAGMENT,
        position=flowui.XYPosition(600, 600), 
        name="op_lib",
        flow=ADVFlowModel(nodes={
            "sym_lib": nested_model_symbol_lib,
            "add": ADVNodeModel(
                id="add", 
                nType=ADVNodeType.FRAGMENT,
                position=flowui.XYPosition(0, 0), 
                name="add",
                flow=_get_simple_flow("add", "+", ["op_lib", "sym_lib"]),
            ),
            "sub": ADVNodeModel(
                id="sub", 
                nType=ADVNodeType.FRAGMENT,
                position=flowui.XYPosition(200, 0), 
                name="sub",
                flow=_get_simple_flow("sub", "-", ["op_lib", "sym_lib"]),
            ),
            "div": ADVNodeModel(
                id="div", 
                nType=ADVNodeType.FRAGMENT,
                position=flowui.XYPosition(400, 0), 
                name="div",
                flow=_get_simple_flow("div", "/", ["op_lib", "sym_lib"]),
            ),
            "mul": ADVNodeModel(
                id="mul", 
                nType=ADVNodeType.FRAGMENT,
                position=flowui.XYPosition(600, 0), 
                name="mul",
                flow=_get_simple_flow("mul", "*", ["op_lib", "sym_lib"]),
            ),
        }, edges={})
    )

    nested_model = ADVNodeModel(
        id="nf1", 
        nType=ADVNodeType.FRAGMENT,
        position=flowui.XYPosition(600, 100), 
        name="nested0",
        flow=ADVFlowModel(nodes={
            "s1": ADVNodeModel(
                id="s1", 
                nType=ADVNodeType.SYMBOLS,
                position=flowui.XYPosition(0, 0), 
                name="SymbolGroup",
                impl=InlineCode(symbolgroup0),
            ),
            "mul": ADVNodeModel(
                id="mul", 
                nType=ADVNodeType.FRAGMENT,
                position=flowui.XYPosition(200, 0), 
                name="mul_func",
                inlinesf_name="nested0",
                ref_node_id="func",
                ref_import_path=["op_lib", "mul"],
            ),
            "oic0": ADVNodeModel(
                id="oic0", 
                nType=ADVNodeType.OUT_INDICATOR,
                position=flowui.XYPosition(400, 100), 
                name="Outputs",
            ),

        }, edges={
            "ea": ADVEdgeModel(
                id="ea", 
                source="s1",
                sourceHandle=f"{ADVHandlePrefix.Output}-a",
                target="mul",
                targetHandle=f"{ADVHandlePrefix.Input}-a",
                isAutoEdge=False,
            ),
            "eb": ADVEdgeModel(
                id="eb", 
                source="s1",
                sourceHandle=f"{ADVHandlePrefix.Output}-b",
                target="mul",
                targetHandle=f"{ADVHandlePrefix.Input}-b",
                isAutoEdge=False,
            ),

            "eo": ADVEdgeModel(
                id="eo", 
                source="mul",
                sourceHandle=f"{ADVHandlePrefix.Output}-c",
                target="oic0",
                targetHandle=f"{ADVHandlePrefix.OutIndicator}-outputs",
                isAutoEdge=False,
            )

        })
    )
    # return ADVProject(
    #     flow=nested_model.flow,
    #     import_prefix="tensorpc.adv.test_project",
    #     path=str(PACKAGE_ROOT / "adv" / "test_project"),

    # )
    res_proj = ADVProject(
        flow=ADVFlowModel(nodes={
            "op_lib": nested_model_op_lib,
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
            "add": ADVNodeModel(
                id="add", 
                nType=ADVNodeType.FRAGMENT,
                position=flowui.XYPosition(200, 0), 
                name="add_func",
                inlinesf_name="inline0",
                impl=InlineCode(fragment_add),
            ),
            "f1": ADVNodeModel(
                id="f1", 
                nType=ADVNodeType.FRAGMENT,
                position=flowui.XYPosition(400, 100), 
                name="add_func2",
                inlinesf_name="inline0",
                impl=InlineCode(fragment1),
            ),
            "add-ref": ADVNodeModel(
                id="add-ref", 
                nType=ADVNodeType.FRAGMENT,
                position=flowui.XYPosition(200, 200), 
                name="add_func",
                inlinesf_name="inline0",
                ref_node_id="add",
                ref_import_path=[],
                alias_map="c->C", 
            ),
            "oic0": ADVNodeModel(
                id="oic0", 
                nType=ADVNodeType.OUT_INDICATOR,
                position=flowui.XYPosition(800, 200), 
                name="Outputs",
            ),
            "nf1": nested_model,
            "nf1-mul-ref": ADVNodeModel(
                id="nf1-mul-ref", 
                nType=ADVNodeType.FRAGMENT,
                position=flowui.XYPosition(800, 400), 
                name="fn_nested",
                # inlinesf_name="inline0",
                ref_node_id="mul",
                ref_import_path=["op_lib"],
                # alias_map="c->C", 
            ),
            # "sub-ref": ADVNodeModel(
            #     id="nf2-sub-ref", 
            #     nType=ADVNodeType.FRAGMENT,
            #     position=flowui.XYPosition(800, 600), 
            #     name="fn_nested",
            #     # inlinesf_name="inline0",
            #     ref_node_id="sub",
            #     ref_import_path=["nested0", "nested1"],
            #     # alias_map="c->C", 
            # ),

        }, edges={
            # "e0": ADVEdgeModel(
            #     id="e0", 
            #     source="f1",
            #     sourceHandle=f"{ADVHandlePrefix.Output}-d",
            #     target="oic0",
            #     targetHandle=f"{ADVHandlePrefix.OutIndicator}-outputs",
            #     isAutoEdge=False,
            # )
            "e-add-ref-a": ADVEdgeModel(
                id="e-add-ref-a", 
                source="n1",
                sourceHandle=f"{ADVHandlePrefix.Output}-a",
                target="add-ref",
                targetHandle=f"{ADVHandlePrefix.Input}-a",
                isAutoEdge=False,
            )

        }),

        import_prefix="tensorpc.adv.test_project",
        path=str(PACKAGE_ROOT / "adv" / "test_project"),
    )
    ngid_to_path, ngid_to_fpath = res_proj.assign_path_to_all_node()
    res_proj.node_gid_to_path = ngid_to_path
    res_proj.node_gid_to_frontend_path = ngid_to_fpath
    res_proj.update_ref_path(ngid_to_path, ngid_to_fpath)

    # manager = ADVProjectBackendManager(lambda: res_proj.flow)
    # manager.sync_project_model()
    # manager.parse_all()
    # manager.init_all_nodes()
    # path_to_code: dict[str, str] = {}
    # for flow_id, parser in manager._flow_node_gid_to_parser.items():
    #     assert parser._flow_parse_result is not None 
    #     parse_res = parser._flow_parse_result
    #     path = ".".join(parse_res.get_path_list())
    #     code_lines = parse_res.generated_code_lines
    #     code = "\n".join(code_lines)
    #     path_to_code[path] = code

    # proj_parser = ADVProjectParser(lambda path: path_to_code[".".join(path)])
    # flow = proj_parser._parse_desc_to_flow_model([], set())
    # res_proj.flow = flow
    # ngid_to_path, ngid_to_fpath = res_proj.assign_path_to_all_node()
    # res_proj.node_gid_to_path = ngid_to_path
    # res_proj.node_gid_to_frontend_path = ngid_to_fpath
    # res_proj.update_ref_path(ngid_to_path, ngid_to_fpath)

    return res_proj

class App:
    @mark_create_layout
    def my_layout(self):
        adv_proj = {
            # "project": _test_model_simple()
            "project": _test_model_symbol_group()

        }
        ngid_to_path, ngid_to_fpath = adv_proj["project"].assign_path_to_all_node()
        adv_proj["project"].node_gid_to_path = ngid_to_path
        adv_proj["project"].node_gid_to_frontend_path = ngid_to_fpath
        adv_proj["project"].update_ref_path(ngid_to_path, ngid_to_fpath)
        
        
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
                # each out indicator can only connect one output (source)
                ADVHandlePrefix.Output: 1
            },
        }
        self.graph.prop(targetValidConnectMap=target_conn_valid_map)

        self.graph.event_node_context_menu.on(self.handle_node_cm)
        path_breadcrumb = mui.Breadcrumbs([]).prop(keepHistoryPath=True)
        detail = mui.JsonEditor()
        editor = mui.MonacoEditor("", "python", "default").prop(flex=1, minHeight=0, minWidth=0)
        editor_acts: list[mui.MonacoEditorAction] = [
            mui.MonacoEditorAction(id="ToggleEditableAreas", 
                label="Toggle Editable Areas", contextMenuOrder=1.5,
                contextMenuGroupId="tensorpc-editor-action", 
            ),
        ]

        self.editor = editor.prop(enableConstrainedEditing=True, actions=editor_acts)
        self.editor.event_editor_action.on(self._handle_editor_acts)
        self.editor.update_raw_props({
            ".monaco-editor-content-decoration": {
                "background": "lightblue"
            }
        })
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
        ], json_only=True)
        draft = self.dm.get_draft()
        cur_root_proj = draft.draft_get_cur_adv_project()
        cur_model_draft = draft.draft_get_cur_model()

        manager = ADVProjectBackendManager(lambda: self.dm.get_model().adv_projects["project"].flow, cur_root_proj.flow)
        manager.sync_project_model()
        manager.parse_all()
        manager.init_all_nodes()
        # import rich 
        # debug_flow = self.dm.get_model().adv_projects["project"].flow
        # rich.print({
        #     "nodes": debug_flow.nodes,
        #     "edges": debug_flow.edges,
        # })
        self._manager = manager
        graph_container.update_raw_props(default_compute_flow_css())

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
        # has_code, code_draft, path_draft = cur_root_proj.draft_get_node_impl_editor(cur_root_proj.draft_get_selected_node().id)
        # editor.bind_draft_change_uncontrolled(code_draft, path_draft=path_draft)
        # editor_ct.bind_fields(condition=has_code)
        handler, _ = self.dm.install_draft_change_handler(
            {
                "sel_node": cur_model_draft.selected_nodes,
                "cur_path": cur_root_proj.cur_path,
            },
            partial(self._code_editor_draft_change),
            installed_comp=editor)

        editor_ct.bind_pfl_query(self.dm, condition=(ADVRoot.get_cur_node_flows, "enableCodeEditor"))
        # self.dm.debug_print_draft_change(has_code)

        return mui.HBox([
            self.dm,
        ]).prop(width="100%", height="100%", overflow="hidden")
    
    async def _code_editor_draft_change(self, draft_ev: DraftChangeEvent):
        select_node = draft_ev.new_value_dict["sel_node"]
        cur_fe_path = draft_ev.new_value_dict["cur_path"]
        adv_proj = self.dm.get_model().get_cur_adv_project()
        if select_node is not None and len(select_node) == 1:
            # print(cur_fe_path)
            pair = ADVProject.get_flow_node_by_fe_path(adv_proj.flow, cur_fe_path + ["nodes", select_node[0]])
            assert pair is not None 
            node_gid = pair[1].get_global_uid()
            flow_code, path, code_range = self._manager._get_flow_code_lineno_by_node_gid(node_gid)
            # print(pair[1].id, pair[1].nType, path, lineno)
            constrained_ranges = [
                mui.MonacoConstrainedRange(code_range, "editarea", allowMultiline=True, decorationOptions=mui.MonacoModelDecoration(
                    className="monaco-editor-content-decoration", isWholeLine=True,
                    minimap=mui.MonacoModelDecorationMinimapOptions(mui.MonacoMinimapPosition.Inline
                )))
            ]
            if code_range[0] > 0:
                await self.editor.write(flow_code, path, line=code_range[0], 
                    language="python", constrained_ranges=constrained_ranges)
            else:
                await self.editor.write(flow_code, path, language="python")
        else:
            await self.editor.write("", "", language="python", constrained_ranges=[])

    def _get_preview_flow_uid(self, path_draft):
        path = D.evaluate_draft(path_draft, self.dm.model)
        if path is None:
            return "root"
        return UniqueTreeIdForTree.from_parts(path).uid_encoded

    def model_to_ui_node(self, flow: ADVFlowModel, node_id: str):
        node = flow.nodes[node_id]
        node_gid = node.get_global_uid()
        if node.nType == ADVNodeType.OUT_INDICATOR:
            comp = IndicatorWrapper(
                node_gid, self.dm, f"{ADVHandlePrefix.OutIndicator}-outputs"
            )
        else:
            comp = BaseNodeWrapper(
                node_gid,
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

    async def _handle_editor_acts(self, act: mui.MonacoActionEvent):
        if act.action == "ToggleEditableAreas":
            await self.editor.toggle_editable_areas()


def _main():
    import rich 
    model = _test_model_symbol_group()

    manager = ADVProjectBackendManager(lambda: model.flow, create_draft_type_only(type(model.flow)))
    manager.sync_project_model()
    manager.parse_all()
    manager.init_all_nodes()
    import rich 
    path_to_code: dict[str, str] = {}
    for flow_id, fcache in manager._flow_node_gid_to_cache.items():

        assert fcache.parser._flow_parse_result is not None 
        parse_res = fcache.parser._flow_parse_result
        path = ".".join(parse_res.get_path_list())
        code_lines = parse_res.generated_code_lines
        code = "\n".join(code_lines)
        path_to_code[path] = code
        # print("+" * 80)
        # print("+" * 80)

        # print(code)

    # proj_parser = ADVProjectParser(lambda path: path_to_code[".".join(path)])
    # flow = proj_parser._parse_desc_to_flow_model([], set())
    # model.flow = flow
    # ngid_to_path, ngid_to_fpath = model.assign_path_to_all_node()
    # model.node_gid_to_path = ngid_to_path
    # model.node_gid_to_frontend_path = ngid_to_fpath
    # model.update_ref_path(ngid_to_path, ngid_to_fpath)

    # rich.print(desc)
    # proj_parser._parse_desc_to_flow_model(["test", "adv"], set())
    # rich.print({
    #     "nodes": debug_flow.nodes,
    #     "edges": debug_flow.edges,
    # })

def _main_change_debug():
    import rich 
    model = _test_model_symbol_group()

    manager = ADVProjectBackendManager(lambda: model.flow, create_draft_type_only(type(model.flow)))
    manager.sync_project_model()
    manager.parse_all()
    manager.init_all_nodes()
    add_func2_node = model.flow.nodes["f1"]
    fragment_changed = f"""
ADV.mark_outputs("d->D")
return c + b
    """
    # add_func2_node.impl.code = fragment_changed
    print(add_func2_node.get_global_uid())
    changed_nodes, changed_edges = manager.modify_code_impl(add_func2_node.get_global_uid(), fragment_changed)
    import rich 
    rich.print(changed_nodes, changed_edges)

if __name__ == "__main__":
    _main()