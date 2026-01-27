import enum
from pathlib import Path
import rich
from tensorpc.apps.adv.codemgr.flow import ADV_MAIN_FLOW_NAME, ADVProjectBackendManager, ADVProjectChange
from tensorpc.apps.adv.codemgr.proj_parse import ADVProjectParser, create_adv_model
from tensorpc.apps.adv.constants import TENSORPC_ADV_MD_MIN_SIZE
from tensorpc.apps.adv.nodes.base import BaseNodeWrapper, IONodeWrapper, IndicatorWrapper, MarkdownNodeWrapper
from tensorpc.apps.adv.test_data.simple import get_simple_nested_model
from tensorpc.constants import PACKAGE_ROOT
from tensorpc.core.datamodel.events import DraftChangeEvent
from tensorpc.core.funcid import remove_common_indent_from_code
from tensorpc.dock import mui, three, plus, appctx, mark_create_layout, flowui, models
from tensorpc.core.datamodel.draft import create_draft_type_only, create_literal_draft
import tensorpc.core.datamodel.funcs as D
from functools import partial
from tensorpc.core.tree_id import UniqueTreeIdForTree

from typing import Callable, Coroutine, Literal, Optional, Any
from tensorpc.apps.adv.model import ADVEdgeModel, ADVHandlePrefix, ADVNewNodeConfig, ADVNodeHandle, ADVNodeType, ADVRoot, ADVProject, ADVNodeModel, ADVFlowModel, InlineCode
from tensorpc.core.datamodel.draft import (get_draft_pflpath)
from tensorpc.dock.components.flowplus.style import default_compute_flow_css
from tensorpc.dock.components.plus.config import ConfigDialogEvent, ConfigPanelDialog
import rich 
import sys 


class NodeContextMenu(enum.Enum):
    EnterNested = "Enter Nested"
    DeleteNode = "Delete Node"

class PaneContextMenu(enum.Enum):
    AddFragment = "Add Fragment Node"
    AddNestedFragment = "Add Fragment Subflow"
    AddGlobalScript = "Add Global Script"
    AddSymbolGroup = "Add Symbol Group"
    AddOutput = "Add Output"
    AddRef = "Add Ref Node"
    AddMarkdown = "Add Markdown Node"
    Debug = "Debug"

class RefNodeSelectDialog(mui.Dialog):
    def __init__(self, on_click: Callable[[str, tuple[mui.NumberType, mui.NumberType]], Coroutine[Any, Any, None]]):
        self.title = mui.Typography().prop(variant="body1")
        self.qname = mui.Typography().prop(variant="caption", enableTooltipWhenOverflow=True)


        self.info_icon = mui.Icon()
        self.title.bind_fields(value="title")
        self.qname.bind_fields(value="qname")
        self.info_icon.bind_fields(tooltip="info")

        self.card = mui.Paper([
            self.title,
            self.qname,
            self.info_icon
        ]).prop(width="150px", height="150px", margin="10px", flexFlow="column",)
        # card get bigger when hover
        self.card.update_raw_props({
            ":hover": {
                "transform": "scale(1.05)",
                "transition": "transform 0.2s",
            }
        })
        self.card.event_click.on_standard(self._handle_click)
        self.content = mui.DataFlexBox(self.card).prop(flex=1, overflowY="auto", flexFlow="row wrap",
            filterKey="qname", flexDirection="row")
        self.search = mui.TextField("nodes")
        self.search.prop(valueChangeTarget=(self.content, "filter"))
        self._on_click = on_click
        super().__init__([
            self.search,
            self.content,
        ])
        self.prop(overflow="hidden", title="Create Ref Node", 
            display="flex", dialogMaxWidth=False, fullWidth=False,
            width="75vw", height="75vh", includeFormControl=False, flexDirection="column")

        self.position: tuple[mui.NumberType, mui.NumberType] = (0, 0)

    async def _handle_click(self, event: mui.Event):
        gid = event.get_keys_checked()[0]
        try:
            await self._on_click(gid, self.position)
        finally:
            await self.set_open(False)

class App:
    @mark_create_layout
    def my_layout(self):
        adv_proj = {
            # "project": _test_model_simple()
            "project": get_simple_nested_model()

        }
        # adv_proj = create_adv_model({
        #     PACKAGE_ROOT / "apps" / "adv" / "managed": ("project", "tensorpc.apps.adv.managed"),
        # }).adv_projects
        # rich.print(adv_proj["project"])
        ngid_to_path, ngid_to_fpath = adv_proj["project"].assign_path_to_all_node()
        adv_proj["project"].node_gid_to_path = ngid_to_path
        adv_proj["project"].node_gid_to_frontend_path = ngid_to_fpath
        adv_proj["project"].update_ref_path(ngid_to_fpath)
        appctx.get_app().set_enable_language_server(True)
        pyright_setting = appctx.get_app().get_language_server_settings()
        pyright_setting.python.analysis.pythonPath = sys.executable
        pyright_setting.python.analysis.extraPaths = [
            str(PACKAGE_ROOT.parent),
        ]
        model = ADVRoot(cur_adv_project="project", adv_projects=adv_proj)
        node_cm_items = [
            mui.MenuItem(id=NodeContextMenu.EnterNested.name, label=NodeContextMenu.EnterNested.value),
            mui.MenuItem(id=NodeContextMenu.DeleteNode.name, label=NodeContextMenu.DeleteNode.value),

        ]
        items = [
            mui.MenuItem(id=PaneContextMenu.AddRef.name, label=PaneContextMenu.AddRef.value),
            mui.MenuItem(id="divider0", divider=True),

            mui.MenuItem(id=PaneContextMenu.AddFragment.name, label=PaneContextMenu.AddFragment.value),
            mui.MenuItem(id=PaneContextMenu.AddGlobalScript.name, label=PaneContextMenu.AddGlobalScript.value),
            mui.MenuItem(id=PaneContextMenu.AddSymbolGroup.name, label=PaneContextMenu.AddSymbolGroup.value),
            mui.MenuItem(id=PaneContextMenu.AddOutput.name, label=PaneContextMenu.AddOutput.value),
            mui.MenuItem(id=PaneContextMenu.AddMarkdown.name, label=PaneContextMenu.AddMarkdown.value),

            mui.MenuItem(id="divider1", divider=True),
            mui.MenuItem(id=PaneContextMenu.AddNestedFragment.name, label=PaneContextMenu.AddNestedFragment.value),
            mui.MenuItem(id="divider2", divider=True),
            mui.MenuItem(id=PaneContextMenu.Debug.name, label=PaneContextMenu.Debug.value),


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
        self.graph.prop(targetValidConnectMap=target_conn_valid_map, selectNodesOnDrag=False, debounce=300)

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
        self.editor.event_editor_save.on(self._handle_editor_save)

        self.editor.update_raw_props({
            ".monaco-editor-content-decoration": {
                "background": "lightblue",
                "width": "5px !important",
                # "marginLeft": "3px",
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
        self.new_node_dialog = ConfigPanelDialog(self._on_new_node_create)
        self.new_ref_dialog = RefNodeSelectDialog(self._handle_new_ref_node)
        self.dm = mui.DataModel(model, [
            graph_container,
            detail_ct,
            self.new_node_dialog,
            self.new_ref_dialog,
        ], json_only=True)
        draft = self.dm.get_draft()
        cur_root_proj = draft.draft_get_cur_adv_project()
        cur_model_draft = draft.draft_get_cur_model()

        manager = ADVProjectBackendManager(lambda: self.dm.get_model().adv_projects["project"], cur_root_proj.flow)
        manager.sync_project_model()
        manager.parse_all()
        manager.init_all_nodes()
        # debug_flow = self.dm.get_model().adv_projects["project"].flow
        # rich.print({
        #     "nodes": debug_flow.nodes,
        #     "edges": debug_flow.edges,
        # })
        self._manager = manager
        graph_container.update_raw_props(default_compute_flow_css())

        self.graph.event_pane_context_menu.on(partial(self._handle_pane_context_menu, target_flow_draft=cur_model_draft))
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
        self._binder = binder
        binder.event_position_change.on(self._handle_position_change)
        binder.event_edge_delete.on(self._handle_edge_change)
        binder.event_edge_new.on(self._handle_edge_change)

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
            # print("ASFASFASDASD")
            pair = ADVProject.get_flow_node_by_fe_path(adv_proj.flow, cur_fe_path + ["nodes", select_node[0]])
            assert pair is not None 
            node = pair[1]
            node_gid = node.get_global_uid()
            frag = self._manager._get_flow_code_lineno_by_node_gid(node_gid)
            if frag is not None:
                if node.nType == ADVNodeType.MARKDOWN:
                    # {node_gid}.md, not real path
                    return await self.editor.write(frag.code, frag.path, language="markdown", constrained_ranges=[])
                else:
                    real_path_fs = str(Path(adv_proj.path) / Path(frag.path))
                    print("!", real_path_fs)
                    # print(pair[1].id, pair[1].nType, path, lineno)
                    constrained_ranges = [
                        mui.MonacoConstrainedRange(frag.code_range, node_gid, allowMultiline=True, decorationOptions=mui.MonacoModelDecoration(
                            linesDecorationsClassName="monaco-editor-content-decoration", isWholeLine=True,
                            minimap=mui.MonacoModelDecorationMinimapOptions(mui.MonacoMinimapPosition.Inline
                        )))
                    ]
                    # constrained_ranges = []
                    if frag.code_range[0] > 0:
                        return await self.editor.write(frag.code, real_path_fs, line=frag.code_range[0], 
                            language="python", constrained_ranges=constrained_ranges)
                    else:
                        return await self.editor.write(frag.code, real_path_fs, language="python")
        return await self.editor.write("", "", language="python", constrained_ranges=[])

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
                node_gid, self.dm
            )
        elif node.nType == ADVNodeType.MARKDOWN:
            comp = MarkdownNodeWrapper(
                node_gid, self.dm
            )
        else:
            comp = IONodeWrapper(
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
            ui_edge.deletable = False
        return ui_edge

    def ui_to_model_edge(self, ui_edge: flowui.Edge) -> ADVEdgeModel:
        # TODO generate edge id here
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
        if item_id == NodeContextMenu.EnterNested.name:
            node_id = data.nodeId
            cur_proj = self.dm.model.get_cur_adv_project()
            cur_path_val = self.dm.model.get_cur_adv_project().cur_path
            new_path_val = cur_path_val + ['nodes', node_id, 'flow']
            pair = cur_proj.get_flow_node_by_fe_path(cur_proj.flow, new_path_val)
            # validate node contains nested flow
            if pair is None:
                return 
            node = pair[1]
            if node.ref_fe_path is not None:
                path = node.ref_fe_path
            else:
                path = node.frontend_path
            draft = self.dm.get_draft().draft_get_cur_adv_project()
            # we have to clear selection before switch flow because xyflow don't support controlled selection.
            # xyflow will clear previous selection and send clear-selection event when flow is switched.
            D.getitem_path_dynamic(draft.flow, draft.cur_path, Optional[ADVFlowModel]).selected_nodes = []
            draft.cur_path = path + ['flow']
        elif item_id == NodeContextMenu.DeleteNode.name:
            
            node_id = data.nodeId
            cur_proj = self.dm.model.get_cur_adv_project()
            cur_path_val = cur_proj.cur_path
            new_path_val = cur_path_val + ['nodes', node_id, ]
            pair = cur_proj.get_flow_node_by_fe_path(cur_proj.flow, new_path_val)
            assert pair is not None 

            change = self._manager.delete_node(pair[1].get_global_uid())
            await self._apply_flow_change(change)

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

    def _get_cur_flow_gid(self):
        cur_proj = self.dm.model.get_cur_adv_project()
        cur_path_val = cur_proj.cur_path
        if not cur_path_val:
            flow_gid = ""
        else:
            pair = cur_proj.get_flow_node_by_fe_path(cur_proj.flow, cur_path_val)
            assert pair is not None 
            flow_gid = pair[1].get_global_uid()
        return flow_gid

    async def _handle_pane_context_menu(self, data: flowui.PaneContextMenuEvent, target_flow_draft: Any):
        
        # cur_model = self.dm.model.get_cur_adv_project().flow
        # node_ids = [n.id for n in cur_model.nodes.values()]
        # await self.graph.update_node_internals(node_ids)
        cur_flow_gid = self._get_cur_flow_gid()
        add_node_items = [
            PaneContextMenu.AddFragment.name, PaneContextMenu.AddGlobalScript.name, 
            PaneContextMenu.AddSymbolGroup.name, PaneContextMenu.AddNestedFragment.name,
            PaneContextMenu.AddOutput.name, PaneContextMenu.AddMarkdown.name
        ]
        if data.itemId in add_node_items:
            if data.itemId == PaneContextMenu.AddFragment.name or data.itemId == PaneContextMenu.AddNestedFragment.name:
                ntype = ADVNodeType.FRAGMENT.value 
                if data.itemId == PaneContextMenu.AddNestedFragment.name:
                    name_default = "new_subflow"
                else:
                    name_default = "new_fragment"
            elif data.itemId == PaneContextMenu.AddGlobalScript.name:
                ntype = ADVNodeType.GLOBAL_SCRIPT.value
                name_default = "new_global_script"
            elif data.itemId == PaneContextMenu.AddSymbolGroup.name:
                ntype = ADVNodeType.SYMBOLS.value
                name_default = "NewSymbolGroup"
            elif data.itemId == PaneContextMenu.AddOutput.name:
                name_default = ""
                ntype = ADVNodeType.OUT_INDICATOR.value
            elif data.itemId == PaneContextMenu.AddMarkdown.name:
                name_default = ""
                ntype = ADVNodeType.MARKDOWN.value

            else:
                raise NotImplementedError

            assert data.flowPosition is not None 
            position = (data.flowPosition.x, data.flowPosition.y)
            await self.new_node_dialog.open_config_dialog(ADVNewNodeConfig(
                name=name_default,
            ), userdata={
                "flow_gid": cur_flow_gid,
                "ntype": ntype,
                "position": position,
                "is_subflow": data.itemId == PaneContextMenu.AddNestedFragment.name,
            })
        elif data.itemId == PaneContextMenu.AddRef.name:
            all_ref_nodes = self._manager.collect_possible_ref_nodes(cur_flow_gid)
            assert data.flowPosition is not None 

            rich.print([n.id for n in all_ref_nodes])
            datas = []
            for n in all_ref_nodes:
                name = n.name
                qname = ".".join(n.path + [n.name])
                info = "TODO"
                datas.append({
                    "id": n.get_global_uid(),
                    "title": name,
                    "qname": qname,
                    "info": info,
                })
            await self.new_ref_dialog.send_and_wait(self.new_ref_dialog.content.update_event(dataList=datas))
            self.new_ref_dialog.position = (data.flowPosition.x, data.flowPosition.y)
            await self.new_ref_dialog.set_open(True)
        elif data.itemId == PaneContextMenu.Debug.name:
            cur_model = self.dm.model.get_cur_adv_project().flow
            node_ids = [n.id for n in cur_model.nodes.values()]
            await self.graph.update_node_internals(node_ids)

    async def _handle_editor_acts(self, act: mui.MonacoActionEvent):
        if act.action == "ToggleEditableAreas":
            await self.editor.toggle_editable_areas()

    async def _handle_editor_save(self, ev: mui.MonacoSaveEvent):
        # print(ev.path, ev.constrainedValues)
        if ev.constrainedValues is not None:
            for node_gid, node_impl_val in ev.constrainedValues.items():

                change = self._manager.modify_code_impl(node_gid, remove_common_indent_from_code(node_impl_val))
                print(ev.path, ev.path in change.flow_code_changes)
                if ev.path in change.flow_code_changes:
                    frag = self._manager._get_flow_code_lineno_by_node_gid(node_gid)
                    assert frag is not None 
                    constrained_ranges = [
                        mui.MonacoConstrainedRange(frag.code_range, node_gid, allowMultiline=True, decorationOptions=mui.MonacoModelDecoration(
                            className="monaco-editor-content-decoration", isWholeLine=True,
                            minimap=mui.MonacoModelDecorationMinimapOptions(mui.MonacoMinimapPosition.Inline
                        )))
                    ]
                    # print(change.flow_code_changes[ev.path])
                    await self.editor.write(change.flow_code_changes[ev.path], constrained_ranges=constrained_ranges)
                await self._apply_flow_change(change)
        else:
            # markdown
            assert ev.path is not None 
            node_gid = Path(ev.path).stem
            new_md_value = ev.value
            change = self._manager.modify_code_impl(node_gid, new_md_value)
            await self._apply_flow_change(change)

    async def _on_new_node_create(self, event: ConfigDialogEvent[ADVNewNodeConfig]):
        # TODO check name conflict
        flow_gid = event.userdata["flow_gid"]
        ntype = ADVNodeType(event.userdata["ntype"])
        is_subflow = event.userdata["is_subflow"]
        node_id_base = event.cfg.name
        if ntype != ADVNodeType.OUT_INDICATOR and ntype != ADVNodeType.MARKDOWN:
            assert event.cfg.name.isidentifier()
        else:
            if event.cfg.name != "":
                assert event.cfg.name.isidentifier()
        subflow = None
        position = event.userdata["position"]
        impl: Optional[InlineCode] = None
        if ntype == ADVNodeType.FRAGMENT:
            if is_subflow:
                subflow = ADVFlowModel(nodes={}, edges={})
            else:
                impl = InlineCode(f"ADV.mark_outputs() # mark output symbols here.")
        elif ntype == ADVNodeType.GLOBAL_SCRIPT:
            impl = InlineCode(f"")
        elif ntype == ADVNodeType.SYMBOLS:
            impl = InlineCode(f"""
@dataclasses.dataclass
class {event.cfg.name}:
    pass
            """)
        elif ntype == ADVNodeType.OUT_INDICATOR:
            node_id_base = "Out"
        elif ntype == ADVNodeType.MARKDOWN:
            node_id_base = "Markdown"
            impl = InlineCode(f"""
## Hello!
            """)
        else:
            raise NotImplementedError
        node = ADVNodeModel(
            id=node_id_base,
            position=flowui.XYPosition(position[0], position[1]),
            nType=ntype,
            name=event.cfg.name,
            impl=impl,
            flow=subflow,
        )
        if ntype == ADVNodeType.MARKDOWN:
            node.width = TENSORPC_ADV_MD_MIN_SIZE[0]
            node.height = TENSORPC_ADV_MD_MIN_SIZE[1]
        change = self._manager.add_new_node(flow_gid, node)
        await self._apply_flow_change(change)

    async def _handle_new_ref_node(self, gid: str, position: tuple[mui.NumberType, mui.NumberType]):
        cur_flow_gid = self._get_cur_flow_gid()

        import_path, node_id = ADVNodeModel.extract_path_and_id(gid)
        cache = self._manager._node_gid_to_cache[gid]
        node = ADVNodeModel(
            id=node_id,
            position=flowui.XYPosition(position[0], position[1]),
            nType=cache.node.nType,
            name=cache.node.name,
            ref_node_id=node_id,
            ref_import_path=import_path,
        )
        change = self._manager.add_new_node(cur_flow_gid, node)
        rich.print(node)
        await self._apply_flow_change(change)

        # print("_handle_new_ref_node", gid)
    async def _handle_position_change(self, changed_nodes: dict[str, flowui.XYPosition]):
        cur_flow_gid = self._get_cur_flow_gid()
        change = self._manager.flow_nodes_position_change(cur_flow_gid)
        await self._apply_flow_change(change)

    async def _apply_flow_change(self, change: ADVProjectChange):
        post_ev_ct = lambda: self.graph.get_update_node_internals_event(change.changed_node_ids_in_cur_flow)
        async with self.dm.draft_update(post_ev_creator=post_ev_ct) as draft:
            # draft = self.dm.get_draft().draft_get_cur_adv_project()
            self._manager.create_draft_updates_from_change(draft.draft_get_cur_adv_project(), change)
        rich.print(change.get_short_repr()) 

    async def _handle_edge_change(self, changed_edges: list[str]):
        cur_flow_gid = self._get_cur_flow_gid()
        change = self._manager.flow_edge_change(cur_flow_gid)
        await self._apply_flow_change(change)

def _main():
    model = get_simple_nested_model()

    manager = ADVProjectBackendManager(lambda: model, create_draft_type_only(type(model.flow)))
    manager.sync_project_model()
    manager.parse_all()
    manager.init_all_nodes()
    rich.print(manager.sync_all_files())

    # path_to_code: dict[str, str] = {}
    folders, path_to_code = manager._get_all_files_and_folders()
    rich.print(folders)
    rich.print(path_to_code.keys())
    # for flow_id, fcache in manager._flow_node_gid_to_cache.items():

    #     assert fcache.parser._flow_parse_result is not None 
    #     parse_res = fcache.parser._flow_parse_result
    #     path = ".".join(parse_res.get_path_list())
    #     code_lines = parse_res.generated_code_lines
    #     code = "\n".join(code_lines)
    #     path_to_code[path] = code
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
    model = get_simple_nested_model()

    manager = ADVProjectBackendManager(lambda: model, create_draft_type_only(type(model.flow)))
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
    change = manager.modify_code_impl(add_func2_node.get_global_uid(), fragment_changed)
    rich.print(change.get_short_repr())

def _main_change_debug2():
    model = get_simple_nested_model()

    manager = ADVProjectBackendManager(lambda: model, create_draft_type_only(type(model.flow)))
    manager.sync_project_model()
    manager.parse_all()
    manager.init_all_nodes()
    add_node = model.flow.nodes["add"]
    change = manager.delete_node(add_node.get_global_uid())
    rich.print(change.get_short_repr())


if __name__ == "__main__":
    _main()