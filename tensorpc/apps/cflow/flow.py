import asyncio
from tensorpc.apps.cflow.executors.base import NodeExecutorBase
from tensorpc.core.datamodel.draftast import evaluate_draft_ast
from tensorpc.dock import mui, three, plus, appctx, mark_create_layout, flowui, models
from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.core.datamodel.draft import create_literal_draft
import tensorpc.core.datamodel.funcs as D
from functools import partial
from tensorpc.core.tree_id import UniqueTreeIdForTree

from typing import Optional, Any

from tensorpc.apps.cflow.binder import ComputeFlowBinder, FlowPanelComps
from tensorpc.apps.cflow.model import ComputeFlowDrafts, ComputeFlowModelRoot, ComputeFlowNodeModel, ComputeNodeType, DetailType, InlineCode, ResourceDesp, get_compute_flow_drafts
from tensorpc.apps.cflow.nodes.cnode.default_code import get_default_custom_node_code
from tensorpc.apps.cflow.nodes.cnode.handle import HandleTypePrefix
from tensorpc.dock.components.flowplus.style import default_compute_flow_css
from tensorpc.utils.code_fmt import PythonCodeFormatter
from tensorpc.apps.cflow.nodes.cnode.registry import NODE_REGISTRY, get_compute_node_runtime, parse_code_to_compute_cfg
import tensorpc.apps.cflow.nodes.defaultnodes
from tensorpc.utils.gpuusage import get_nvidia_gpu_measures

from .schedulers import SimpleScheduler
from .executors import LocalNodeExecutor

_SYS_NODE_PREFIX = "sys-"
_USER_NODE_PREFIX = "user-"


def _get_local_resource_desp():
    gpus = get_nvidia_gpu_measures()
    # TODO add memory schedule
    desp = ResourceDesp(-1, -1, len(gpus), sum([a.memtotal for a in gpus]))
    return desp

class NodeContextMenuItemNames:
    Run = "Run Sub Graph"
    RunThisNode = "Run Cached Node"
    StopGraphRun = "Stop Graph Run"

    CopyNode = "Copy Node"
    DeleteNode = "Delete Node"
    RenameNode = "Rename Node"
    ToggleCached = "Toggle Cached Inputs"
    DebugUpdateNodeInternals = "Debug Update Internals"


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
        self._node_menu_items = [
            mui.MenuItem(NodeContextMenuItemNames.Run,
                         NodeContextMenuItemNames.Run,
                         icon=mui.IconType.PlayArrow),
        ]

        self.graph.prop(nodeContextMenuItems=self._node_menu_items)

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
        self.graph_preview.prop(nodeContextMenuItems=self._node_menu_items)

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

        self.graph.event_node_context_menu.on(self._on_node_contextmenu)
        self.graph_preview.event_node_context_menu.on(self._on_node_contextmenu)

        path_breadcrumb.bind_fields(value=f"concat(`[\"root\"]`, {draft.cur_path}[1::3])")
        path_breadcrumb.event_change.on(self.handle_breadcrumb_click)
        self.code_editor.bind_draft_change_uncontrolled(self.dm, flow_draft.selected_node_code, 
            path_draft=flow_draft.selected_node_code_path, 
            lang_draft=flow_draft.selected_node_code_language,
            save_event_prep=partial(self._process_save_ev_before_save, drafts=flow_draft))
        binder = ComputeFlowBinder(self.graph, self.graph_preview, flow_draft, FlowPanelComps())
        binder.bind_flow_comp_with_datamodel(self.dm)
        self._shutdown_ev = asyncio.Event()
        self.scheduler = SimpleScheduler(self.dm, self._shutdown_ev)

        self.executors: list[NodeExecutorBase] = [
            LocalNodeExecutor("local", _get_local_resource_desp())
        ]
        super().__init__([self.dm])
        self.event_after_unmount.on(self._on_flow_unmount)
        self.prop(width="100%", height="100%", overflow="hidden")

    async def _on_flow_unmount(self):
        self._shutdown_ev.set()

    def _process_save_ev_before_save(self, ev: mui.MonacoEditorSaveEvent, drafts: ComputeFlowDrafts):
        cur_flow_draft = drafts.cur_model
        sel_node = drafts.selected_node
        cur_selected_node_draft = cur_flow_draft.selected_node
        cur_flow = D.evaluate_draft(cur_flow_draft, self.dm.get_model())
        if ev.lang == "python" and cur_flow is not None:
            # compute node code, parse and get new state
            # TODO if old and new state are same, don't update
            cfg = parse_code_to_compute_cfg(ev.value)
            rt = get_compute_node_runtime(cfg)
            sel_node_value = cur_flow.selected_node
            new_inp_handles = [a.name for a in rt.inp_handles]
            new_out_handles = [a.name for a in rt.out_handles]
            # when node impl code changed, we need to remove invalid edges.
            assert cur_flow is not None 
            assert sel_node_value is not None 
            removed_edge_ids = cur_flow.runtime.change_node_handles(sel_node_value, new_inp_handles, new_out_handles)
            for edge_id in removed_edge_ids:
                cur_flow_draft.edges.pop(edge_id)
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
                new_node = ComputeFlowNodeModel(nType=ComputeNodeType.MARKDOWN, id=node_id, position=pos, impl=InlineCode(code="## MarkdownNode"))
            elif node_type == "compute":
                code = get_default_custom_node_code()
                parsed_cfg = parse_code_to_compute_cfg(code)
                new_node = ComputeFlowNodeModel(nType=ComputeNodeType.COMPUTE, id=node_id, position=pos, impl=InlineCode(code=code),
                    name=parsed_cfg.name, key=parsed_cfg.key, moduleId=parsed_cfg.module_id)
                target_flow_draft.node_states[node_id] = {}
            else:
                raise NotImplementedError
        else:
            node_type = item_id[len(_USER_NODE_PREFIX):]
            cfg = NODE_REGISTRY.global_dict[node_type]
            new_node = ComputeFlowNodeModel(
                nType=ComputeNodeType.COMPUTE, id=node_id, position=pos, moduleId=cfg.module_id, key=cfg.key,
                    name=cfg.name)
            if cfg.state_dcls is not None:
                target_flow_draft.node_states[node_id] = cfg.state_dcls()

        target_flow_draft.nodes[node_id] = new_node

    async def _on_node_contextmenu(self, data: flowui.NodeContextMenuEvent):
        item_id = data.itemId
        node_id = data.nodeId

        if item_id == NodeContextMenuItemNames.Run:
            # if node is cached, only run it from cached input
            await self.scheduler.run_sub_graph(self.dm.model, node_id, self.executors)
        # elif item_id == NodeContextMenuItemNames.RunThisNode:
        #     await self.run_cached_node(node_id)
        # elif item_id == NodeContextMenuItemNames.StopGraphRun:
        #     self._shutdown_ev.set()
        # elif item_id == NodeContextMenuItemNames.DeleteNode:
        #     await self.graph.delete_nodes_by_ids([node_id])
        #     await self.save_graph()
        # elif item_id == NodeContextMenuItemNames.RenameNode:
        #     node = self.graph.get_node_by_id(node_id)
        #     wrapper = node.get_component_checked(ComputeNodeWrapper)
        #     await self._node_setting_name.send_and_wait(
        #         self._node_setting_name.update_event(value=wrapper.cnode.name))
        #     await self._node_setting_dialog.set_open(True,
        #                                              {"node_id": node_id})
        #     await self.save_graph()
        # elif item_id == NodeContextMenuItemNames.ToggleCached:
        #     node = self.graph.get_node_by_id(node_id)
        #     wrapper = node.get_component_checked(ComputeNodeWrapper)
        #     await wrapper.set_cached(not wrapper.is_cached_node)
        #     await self.graph.set_node_context_menu_items(
        #         node_id, wrapper.get_context_menus())
        #     await self.save_graph()
        # elif item_id == NodeContextMenuItemNames.DebugUpdateNodeInternals:
        #     await self.graph.update_node_internals([node_id])
