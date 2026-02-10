import enum
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable, Coroutine, Literal, Optional

import rich

import tensorpc.core.datamodel.funcs as D
from tensorpc.apps.adv.codemgr.flow import (ADVProjectBackendManager,
                                            ADVProjectChange)
from tensorpc.apps.adv.codemgr.proj_parse import (ADVProjectParser,
                                                  create_adv_model)
from tensorpc.apps.adv.constants import (TENSORPC_ADV_MD_MIN_SIZE,
                                         TENSORPC_ADV_ROOT_FLOW_ID)
from tensorpc.apps.adv.ide.ref_select import RefNodeSelectDialog
from tensorpc.apps.adv.config import ADVNewNodeConfig, ADVNodeCMConfig, ADVPaneContextMenu

from tensorpc.apps.adv.model import (ADVEdgeModel, ADVFlowModel,
                                     ADVHandlePrefix, ADVNodeFlags,
                                     ADVNodeHandle, ADVNodeModel,
                                     ADVNodeRefInfo, ADVNodeType, 
                                     ADVProject, ADVRoot,
                                     InlineCode)

from tensorpc.apps.adv.ide.nodes.base import (IndicatorWrapper,
                                          IONodeWrapper, MarkdownNodeWrapper)
from tensorpc.apps.adv.test_data.simple import get_simple_nested_model
from tensorpc.constants import PACKAGE_ROOT
from tensorpc.core.datamodel.draft import (create_draft_type_only,
                                           create_literal_draft,
                                           get_draft_pflpath)
from tensorpc.core.datamodel.events import DraftChangeEvent
from tensorpc.core.funcid import remove_common_indent_from_code
from tensorpc.core.tree_id import UniqueTreeIdForTree
from tensorpc.dock import (appctx, flowui, mark_create_layout, models, mui,
                           plus, three)
from tensorpc.dock.components.flowplus.style import default_compute_flow_css
from tensorpc.dock.components.plus.config import (ConfigDialogEvent,
                                                  ConfigPanelDialog)


class NodeContextMenu(enum.Enum):
    EnterNested = "Enter Nested"
    DeleteNode = "Delete Node"
    ConfigureNode = "Configure"

class ADVIdeApp:
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
        base_node_cm_items = [
            # mui.MenuItem(id=NodeContextMenu.EnterNested.name, label=NodeContextMenu.EnterNested.value),
            mui.MenuItem(id=NodeContextMenu.ConfigureNode.value, label=NodeContextMenu.ConfigureNode.value),
            mui.MenuItem(id="divider0", divider=True),
            mui.MenuItem(id=NodeContextMenu.DeleteNode.value, label=NodeContextMenu.DeleteNode.value, 
                confirmTitle="Are you sure to delete this node?", confirmMessage="This action cannot be undo."),

        ]
        self.graph = flowui.Flow([], [], [
            flowui.MiniMap(),
            flowui.Controls(),
            flowui.Background(),
        ]).prop(nodeContextMenuItems=base_node_cm_items)
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
        self.graph.prop(targetValidConnectMap=target_conn_valid_map, 
            selectNodesOnDrag=False, debounce=300)

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
        self.node_configure_dialog = ConfigPanelDialog(self._on_node_configure)

        self.new_ref_dialog = RefNodeSelectDialog(self._handle_new_ref_node)
        self.dm = mui.DataModel(model, [
            graph_container,
            detail_ct,
            self.new_node_dialog,
            self.new_ref_dialog,
            self.node_configure_dialog,
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
        self.graph.bind_pfl_query(self.dm, paneContextMenuItems=(ADVRoot.get_cur_flow_props, "paneMenuItems"))
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
        detail.bind_pfl_query(self.dm, data=(ADVRoot.get_cur_flow_props, "selectedNode"))
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

        editor_ct.bind_pfl_query(self.dm, condition=(ADVRoot.get_cur_flow_props, "enableCodeEditor"))
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
                if node.nType == ADVNodeType.MARKDOWN or node.nType == ADVNodeType.CLASS:
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
        node_cm_items = mui.undefined
        if node.flow is not None:
            node_cm_items = [
                mui.MenuItem(id=NodeContextMenu.EnterNested.value, label=NodeContextMenu.EnterNested.value),
                mui.MenuItem(id=NodeContextMenu.ConfigureNode.value, label=NodeContextMenu.ConfigureNode.value),
                mui.MenuItem(id="divider0", divider=True),
                mui.MenuItem(id=NodeContextMenu.DeleteNode.value, label=NodeContextMenu.DeleteNode.value, 
                    confirmTitle="Are you sure to delete this node?", confirmMessage="This action cannot be undo."),

            ]
        ui_node = flowui.Node(type="app", 
            id=node.id, 
            data=flowui.NodeData(component=comp, label=node.name, contextMenuItems=node_cm_items), 
            position=node.position,
            # disable interaction for inherited nodes.
            # user can unlock this by override in context menu.
            deletable=not node.is_inherited_node(),
            draggable=not node.is_inherited_node(),
            connectable=not node.is_inherited_node())
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
        node_id = data.nodeId
        cur_proj = self.dm.model.get_cur_adv_project()
        cur_path_val = self.dm.model.get_cur_adv_project().cur_path
        new_path_val = cur_path_val + ['nodes', node_id]
        pair = cur_proj.get_flow_node_by_fe_path(cur_proj.flow, new_path_val)
        if pair is None:
            return 

        if item_id == NodeContextMenu.EnterNested.value:
            node = pair[1]
            if node.ref is not None and node.ref.fe_path is not None:
                path = node.ref.fe_path
            else:
                path = node.frontend_path
            draft = self.dm.get_draft().draft_get_cur_adv_project()
            # we have to clear selection before switch flow because xyflow don't support controlled selection.
            # xyflow will clear previous selection and send clear-selection event when flow is switched.
            D.getitem_path_dynamic(draft.flow, draft.cur_path, Optional[ADVFlowModel]).selected_nodes = []
            draft.cur_path = path + ['flow']
        elif item_id == NodeContextMenu.DeleteNode.value:
            change = self._manager.delete_node(pair[1].get_global_uid())
            await self._apply_flow_change(change)
        elif item_id == NodeContextMenu.ConfigureNode.value:
            node = pair[1]
            container_node = pair[0]
            if container_node is None:
                flow_gid = TENSORPC_ADV_ROOT_FLOW_ID
            else:
                flow_gid = container_node.get_global_uid()
            parent_node = pair[0]
            if parent_node is None:
                parent_node_type = ADVNodeType.FRAGMENT
            else:
                parent_node_type = ADVNodeType(parent_node.nType)
            inline_flows = self._manager.collect_all_inline_flow_names(flow_gid)
            selects = [(n, n) for n in inline_flows]
            dynamic_enum_dict = {
                "inline_flow_name": selects
            }
            cfg, exclude_ids = ADVNodeCMConfig.from_node(node, parent_node_type)
            await self.node_configure_dialog.open_config_dialog(cfg, userdata={
                "node_gid": node.get_global_uid(),
            }, dynamic_enum_dict=dynamic_enum_dict, root_exclude_ids=exclude_ids)

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
        node: Optional[ADVNodeModel] = None
        if not cur_path_val:
            flow_gid = ""
            
        else:
            pair = cur_proj.get_flow_node_by_fe_path(cur_proj.flow, cur_path_val)
            assert pair is not None 
            flow_gid = pair[1].get_global_uid()
            node = pair[1]
        return flow_gid, node

    async def _handle_pane_context_menu(self, data: flowui.PaneContextMenuEvent, target_flow_draft: Any):
        
        # cur_model = self.dm.model.get_cur_adv_project().flow
        # node_ids = [n.id for n in cur_model.nodes.values()]
        # await self.graph.update_node_internals(node_ids)
        cur_flow_gid, cur_flow_node = self._get_cur_flow_gid()
        if cur_flow_node is None:
            flow_node_type = ADVNodeType.FRAGMENT
        else:
            flow_node_type = ADVNodeType(cur_flow_node.nType)
        add_node_items = [
            ADVPaneContextMenu.AddFragment.value, ADVPaneContextMenu.AddGlobalScript.value, 
            ADVPaneContextMenu.AddSymbolGroup.value, ADVPaneContextMenu.AddNestedFragment.value,
            ADVPaneContextMenu.AddOutput.value, ADVPaneContextMenu.AddMarkdown.value,
            ADVPaneContextMenu.AddInlineFlowDesc.value, ADVPaneContextMenu.AddClass.value,
        ]
        if data.itemId in add_node_items:
            cfg, nType, exc_fields = ADVNodeCMConfig.from_pane_action(ADVPaneContextMenu(data.itemId), flow_node_type)
            flags = 0
            if data.itemId == ADVPaneContextMenu.AddInlineFlowDesc.value:
                flags = int(ADVNodeFlags.IS_INLINE_FLOW_DESC)
            assert data.flowPosition is not None 
            position = (data.flowPosition.x, data.flowPosition.y)
            inline_flows = self._manager.collect_all_inline_flow_names(cur_flow_gid)
            selects = [(n, n) for n in inline_flows]
            dynamic_enum_dict = {
                "inline_flow_name": selects
            }
            is_subflow = (data.itemId == ADVPaneContextMenu.AddNestedFragment.value or 
                    data.itemId == ADVPaneContextMenu.AddClass.value) 
            await self.new_node_dialog.open_config_dialog(cfg, userdata={
                "flow_gid": cur_flow_gid,
                "ntype": nType,
                "position": position,
                "flags": flags,
                "is_subflow": data.itemId == is_subflow,
            }, root_exclude_ids=exc_fields, dynamic_enum_dict=dynamic_enum_dict)
        elif data.itemId == ADVPaneContextMenu.AddRef.value:
            all_ref_nodes = self._manager.collect_possible_ref_nodes(cur_flow_gid)
            assert data.flowPosition is not None 

            rich.print("ALL REFS", [n.id for n in all_ref_nodes])
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
            ev = self.new_ref_dialog.content.update_event(dataList=datas)
            inline_flow_names = self._manager.collect_all_inline_flow_names(cur_flow_gid)
            ev += self.new_ref_dialog.inline_flow_select.update_event(options=[{"id": n, "label": n} for n in inline_flow_names])
            await self.new_ref_dialog.send_and_wait(ev)
            self.new_ref_dialog.position = (data.flowPosition.x, data.flowPosition.y)
            await self.new_ref_dialog.set_open(True)
        elif data.itemId == ADVPaneContextMenu.Debug.value:
            cur_model = self.dm.model.get_cur_adv_project().flow
            node_ids = [n.id for n in cur_model.nodes.values()]
            rich.print("update_node_internals", node_ids)
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

    async def _on_new_node_create(self, event: ConfigDialogEvent[ADVNodeCMConfig]):
        # TODO check name conflict
        flow_gid = event.userdata["flow_gid"]
        ntype = ADVNodeType(event.userdata["ntype"])
        is_subflow = event.userdata["is_subflow"]
        flags = event.userdata["flags"]
        cfg = event.cfg
        if cfg.func_base_type == "Class Method":
            flags |= int(ADVNodeFlags.IS_CLASSMETHOD)
        elif cfg.func_base_type == "Static Method":
            flags |= int(ADVNodeFlags.IS_STATICMETHOD)
        elif cfg.func_base_type == "Method":
            flags |= int(ADVNodeFlags.IS_METHOD)
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
                if not flags & ADVNodeFlags.IS_INLINE_FLOW_DESC:
                    impl = InlineCode(f"ADV.mark_outputs() # mark output symbols here.\n"
                                      "\n"
                                      "return None\n")
        elif ntype == ADVNodeType.GLOBAL_SCRIPT:
            impl = InlineCode(f"")
        elif ntype == ADVNodeType.CLASS:
            subflow = ADVFlowModel(nodes={}, edges={})

        elif ntype == ADVNodeType.SYMBOLS:
            impl = InlineCode("pass")
        elif ntype == ADVNodeType.OUT_INDICATOR:
            node_id_base = "Out"
        elif ntype == ADVNodeType.MARKDOWN:
            node_id_base = "Markdown"
            impl = InlineCode(f"""
## Hello!
            """)
        else:
            raise NotImplementedError
        inlinesf_name = None 
        if cfg.inline_flow_name != "":
            inlinesf_name = cfg.inline_flow_name
        node = ADVNodeModel(
            id=node_id_base,
            position=flowui.XYPosition(position[0], position[1]),
            nType=ntype,
            name=event.cfg.name,
            impl=impl,
            flow=subflow,
            flags=flags,
            inlinesf_name=inlinesf_name,
        )
        if ntype == ADVNodeType.MARKDOWN:
            node.width = TENSORPC_ADV_MD_MIN_SIZE[0]
            node.height = TENSORPC_ADV_MD_MIN_SIZE[1]
        change = self._manager.add_new_node(flow_gid, node)
        await self._apply_flow_change(change)

    async def _handle_new_ref_node(self, gid: str, position: tuple[mui.NumberType, mui.NumberType], inline_flow_name: Optional[str]):
        cur_flow_gid, _ = self._get_cur_flow_gid()

        import_path, node_id = ADVNodeModel.extract_path_and_id(gid)
        cache = self._manager._node_gid_to_cache[gid]
        node = ADVNodeModel(
            id=node_id,
            position=flowui.XYPosition(position[0], position[1]),
            nType=cache.node.nType,
            name=cache.node.name,
            ref=ADVNodeRefInfo(node_id, import_path),
            flags=cache.node.flags,
            inlinesf_name=inline_flow_name,
        )
        change = self._manager.add_new_node(cur_flow_gid, node)
        await self._apply_flow_change(change)

        # print("_handle_new_ref_node", gid)
    async def _handle_position_change(self, changed_nodes: dict[str, flowui.XYPosition]):
        cur_flow_gid, _ = self._get_cur_flow_gid()
        change = self._manager.flow_nodes_position_change(cur_flow_gid)
        await self._apply_flow_change(change)

    async def _apply_flow_change(self, change: ADVProjectChange):
        post_ev_ct = lambda: self.graph.get_update_node_internals_event(change.changed_node_ids_in_cur_flow)
        async with self.dm.draft_update(post_ev_creator=post_ev_ct) as draft:
            # draft = self.dm.get_draft().draft_get_cur_adv_project()
            self._manager.create_draft_updates_from_change(draft.draft_get_cur_adv_project(), change)
        rich.print(change.get_short_repr()) 

    async def _handle_edge_change(self, changed_edges: list[str]):
        cur_flow_gid, _ = self._get_cur_flow_gid()
        change = self._manager.flow_edge_change(cur_flow_gid)
        await self._apply_flow_change(change)

    async def _on_node_configure(self, event: ConfigDialogEvent[ADVNodeCMConfig]):
        cfg = event.cfg
        node_gid = event.userdata["node_gid"]
        change = self._manager.modify_node_config(node_gid, cfg)
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