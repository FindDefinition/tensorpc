import asyncio
from collections.abc import Sequence
from tensorpc.apps.cflow.executors.base import NodeExecutorBase
from tensorpc.apps.cflow.schedulers.base import SchedulerBase
from tensorpc.core.datamodel.draftast import evaluate_draft_ast
from tensorpc.dock import mui, three, plus, appctx, mark_create_layout, flowui, models
from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.core.datamodel.draft import create_literal_draft
import tensorpc.core.datamodel.funcs as D
from functools import partial
from tensorpc.core.tree_id import UniqueTreeIdForTree

from typing import Annotated, Optional, Any

from tensorpc.apps.cflow.binder import ComputeFlowBinder, FlowPanelComps
from tensorpc.apps.cflow.model import DEFAULT_EXECUTOR_ID, ComputeFlowDrafts, ComputeFlowModelRoot, ComputeNodeModel, ComputeNodeType, DetailType, InlineCode, ResourceDesc, get_compute_flow_drafts
from tensorpc.apps.cflow.nodes.cnode.default_code import get_default_custom_node_code
from tensorpc.apps.cflow.nodes.cnode.handle import HandleTypePrefix
from tensorpc.dock.components.flowplus.style import default_compute_flow_css
from tensorpc.dock.components.models.flow import BaseEdgeModel
from tensorpc.dock.components.plus.config import ConfigDialogEvent, ConfigPanel, ConfigPanelDialog
from tensorpc.dock.components.plus.styles import get_tight_icon_tab_theme
from tensorpc.dock.components.terminal import TerminalBuffer
from tensorpc.dock.flowapp.appstorage import AppDraftFileStoreBackend
from tensorpc.utils.code_fmt import PythonCodeFormatter
from tensorpc.apps.cflow.nodes.cnode.registry import NODE_REGISTRY, get_compute_node_runtime, parse_code_to_compute_cfg
import tensorpc.apps.cflow.nodes.defaultnodes
from tensorpc.utils.gpuusage import get_nvidia_gpu_measures

from .schedulers import SimpleScheduler
from .executors import LocalNodeExecutor

_SYS_NODE_PREFIX = "sys-"
_USER_NODE_PREFIX = "user-"

@dataclasses.dataclass
class NodeSettings:
    executor_id: Annotated[str, ConfigPanel.base_meta(alias="Executor Id")]

def _get_local_resource_desp():
    gpus = get_nvidia_gpu_measures()
    # TODO add memory schedule
    desc = ResourceDesc(-1, -1, len(gpus), sum([a.memtotal for a in gpus]))
    return desc

class NodeContextMenuItemNames:
    Run = "Run Sub Graph"
    Setting = "Node Settings"
    RunThisNode = "Run Cached Node"
    StopGraphRun = "Stop Graph Run"

    CopyNode = "Copy Node"
    DeleteNode = "Delete Node"
    RenameNode = "Rename Node"
    ToggleCached = "Toggle Cached Inputs"
    DebugUpdateNodeInternals = "Debug Update Internals"


class ComputeFlow(mui.FlexBox):
    executors: Sequence[NodeExecutorBase]
    scheduler: SchedulerBase
    def __init__(self, scheduler: Optional[SchedulerBase] = None, executors: Optional[Sequence[NodeExecutorBase]] = None,
            nodes: Optional[Sequence[ComputeNodeModel]] = None, edges: Optional[Sequence[BaseEdgeModel]] = None,
            enable_three: bool = True):
        nodes_dict = {}
        if nodes is not None:
            for node in nodes:
                nodes_dict[node.id] = node
        edges_dict = {}
        if edges is not None:
            for edge in edges:
                edges_dict[edge.id] = edge
        
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
            mui.MenuItem(NodeContextMenuItemNames.Setting,
                         NodeContextMenuItemNames.Setting,
                         icon=mui.IconType.Settings),

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
        tab_theme = get_tight_icon_tab_theme()
        detail_box = mui.HBox([mui.Markdown(" ## Detail")]).prop(overflow="hidden",
                                    padding="3px",
                                    flex=1,
                                    width="100%",
                                    height="100%")
        debug_box = mui.HBox([mui.Markdown(" ## Debug")]).prop(overflow="hidden",
                                    padding="3px",
                                    flex=1,
                                    width="100%",
                                    height="100%")
        panel_comps = FlowPanelComps(detail=detail_box, debug=debug_box)
        self.code_editor = mui.MonacoEditor("", "python",
                                            "default").prop(flex=1,
                                                            minHeight=0,
                                                            minWidth=0)
        code_box = mui.HBox([
            self.code_editor
        ]).prop(height="100%", width="100%", overflow="hidden")
        node_settings = mui.Input("executor id").prop(debounce=300)
        tabdefs: list[mui.TabDef] = [
            mui.TabDef("",
                       "0",
                       debug_box,
                       icon=mui.IconType.BugReport,
                       tooltip="compute node debug panel"),
            mui.TabDef("",
                       "1",
                       detail_box,
                       icon=mui.IconType.Dashboard,
                       tooltip="node detail layout"),

        ]
        self.user_detail = mui.HBox([
            mui.ThemeProvider([
                mui.Tabs(tabdefs, init_value="0").prop(
                    panelProps=mui.FlexBoxProps(width="100%", padding=0),
                    orientation="vertical",
                    borderRight=1,
                    borderColor='divider',
                    tooltipPlacement="right")
            ], tab_theme)
        ]).prop(height="100%", width="100%", overflow="hidden")
        detail_ct = mui.MatchCase(
            [
                mui.MatchCase.Case(DetailType.NONE.value, mui.VBox([])),
                mui.MatchCase.Case(DetailType.SUBFLOW.value, mui.VBox([
                    self.graph_preview,
                ]).prop(height="100%", width="100%", overflow="hidden").update_raw_props(default_compute_flow_css())),
                mui.MatchCase.Case(DetailType.USER_LAYOUT.value, self.user_detail),
            ]
        )
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
            mui.Allotment.Pane(code_box, visible=False),
        ])).prop(vertical=False, defaultSizes=[200, 100])
        global_container = mui.Allotment(mui.Allotment.ChildDef([
            mui.Allotment.Pane(flow_with_editor),
            mui.Allotment.Pane(detail_ct),
        ])).prop(vertical=True, defaultSizes=[300, 200])

        self._header_search_bar = mui.Input("Search").prop(muiMargin="dense", flex=1)
        show_bottom_panel_btn = mui.ToggleButton(name="BOTTOM").prop(size="small", height="28px")
        show_right_panel_btn = mui.ToggleButton(name="RIGHT").prop(size="small", height="28px")
        control_bar = mui.HBox([
            mui.HBox([]).prop(flex=1),
            show_bottom_panel_btn,
            show_right_panel_btn,
        ]).prop(flex=1)
        self._node_setting_dialog = ConfigPanelDialog(self._on_node_config)

        self.dm = mui.DataModel(ComputeFlowModelRoot(edges=edges_dict, nodes=nodes_dict), [
            mui.VBox([
                mui.HBox([
                    path_breadcrumb.prop(flex=1),
                    mui.Divider(orientation="vertical"),
                    self._header_search_bar,
                    mui.Divider(orientation="vertical"),
                    control_bar,
                ]).prop(minHeight="24px"),
                global_container,
            ]).prop(flex=1),
        ])
        # self.dm.connect_draft_store(f"__tensorpc_cflow", AppDraftFileStoreBackend(), clear_previous=False)
        # print(self.dm.model)
        draft = self.dm.get_draft()
        flow_draft = get_compute_flow_drafts(draft)
        flow_with_editor.bind_fields(visibles=f"[True, {flow_draft.show_editor}]")
        global_container.bind_fields(visibles=f"[True, {flow_draft.show_detail}]")
        detail_ct.bind_fields(condition=flow_draft.selected_node_detail_type)
        node_settings.bind_draft_change_uncontrolled(flow_draft.selected_node.vExecId)
        self.graph.event_pane_context_menu.on(partial(self.add_node, target_flow_draft=flow_draft.cur_model))
        self.graph_preview.event_pane_context_menu.on(partial(self.add_node, target_flow_draft=flow_draft.preview_model))
        self._flow_draft = flow_draft
        self.graph.event_node_context_menu.on(self._on_node_contextmenu)
        self.graph_preview.event_node_context_menu.on(self._on_node_contextmenu)

        path_breadcrumb.bind_fields(value=f"[\"root\"] + {draft.cur_path}[1::3]")
        path_breadcrumb.event_change.on(self.handle_breadcrumb_click)
        show_bottom_panel_btn.bind_draft_change(draft.settings.isBottomPanelVisible)
        show_right_panel_btn.bind_draft_change(draft.settings.isRightPanelVisible)
        self.code_editor.bind_draft_change_uncontrolled(flow_draft.selected_node_code, 
            path_draft=flow_draft.selected_node_code_path, 
            lang_draft=flow_draft.selected_node_code_language,
            save_event_prep=partial(self._process_save_ev_before_save, drafts=flow_draft))
        binder = ComputeFlowBinder(self.graph, self.graph_preview, flow_draft, panel_comps)
        binder.bind_flow_comp_with_datamodel(self.dm)
        self._shutdown_ev = asyncio.Event()
        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = SimpleScheduler(self.dm, self._shutdown_ev)
        self._executor_term_buffers: dict[str, TerminalBuffer] = {}
        if executors is not None:
            self.executors = list(executors)
            has_local_executor = False
            for executor in executors:
                buf = executor.get_terminal_buffer()
                ssh_term = executor.get_ssh_terminal()
                if ssh_term is not None:
                    self._executor_term_buffers[executor.get_id()] = buf
                    ssh_term.set_state_buffers(self._executor_term_buffers)
            for executor in executors:
                if executor.is_local():
                    has_local_executor = True
                    break
            if not has_local_executor:
                # add local executor if not exist
                # insert it to the first, currently all built-in schedulers will check executors in order.
                self.executors.insert(0, LocalNodeExecutor(DEFAULT_EXECUTOR_ID, _get_local_resource_desp()))
        else:
            self.executors = [
                LocalNodeExecutor(DEFAULT_EXECUTOR_ID, _get_local_resource_desp())
            ]
        if enable_three:
            # use a special canvas here, it doesn't render 3d child directly,
            # but you can use three.View inside nodes to render a 3d view.
            super().__init__([
                three.ViewCanvas([
                    self.dm,
                ]).prop(display="flex",
                    flexDirection="column", width="100%", height="100%", overflow="hidden"),
                self._node_setting_dialog,
            ])
        else:
            super().__init__([
                self.dm,
                self._node_setting_dialog,
            ])
        self.event_before_unmount.on(self._on_flow_unmount)
        if enable_three:
            self.prop(width="100%", height="100%", overflow="hidden", position="relative", minHeight=0,
                    minWidth=0,)
        else:
            self.prop(width="100%", height="100%", overflow="hidden")

    async def _on_flow_unmount(self):
        self._shutdown_ev.set()
        print("CLOSE SCHEDULER")
        await self.scheduler.close()
        for executor in self.executors:
            print(f"CLOSE EXECUTOR {executor.get_id()}")
            await executor.close()

    def _process_save_ev_before_save(self, ev: mui.MonacoSaveEvent, drafts: ComputeFlowDrafts):
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

    def _debug_add_sys_node(self, key: str,  pos: tuple[int, int], target_flow_draft: Optional[Any] = None) -> ComputeNodeModel:
        """For debugging purpose, add a node to the flow by programming.
        """
        if target_flow_draft is None:
            target_flow_draft = self.dm.get_draft_type_only()
        target_flow = D.evaluate_draft(target_flow_draft, self.dm.model)
        assert target_flow is not None
        node_id = target_flow.make_unique_node_name(key)
        cfg = NODE_REGISTRY.global_dict[key]
        new_node = ComputeNodeModel(
            nType=ComputeNodeType.COMPUTE, id=node_id, position=flowui.XYPosition(*pos), moduleId=cfg.module_id, key=cfg.key,
                name=cfg.name)
        if cfg.state_dcls is not None:
            target_flow_draft.node_states[node_id] = cfg.state_dcls()
        target_flow_draft.nodes[node_id] = new_node
        return new_node

    def _debug_add_edge(self, edge: BaseEdgeModel, target_flow_draft: Optional[Any] = None):
        """For debugging purpose, add a node to the flow by programming.
        """
        if target_flow_draft is None:
            target_flow_draft = self.dm.get_draft_type_only()
        target_flow_draft.edges[edge.id] = edge

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
                new_node = ComputeNodeModel(nType=ComputeNodeType.MARKDOWN, id=node_id, position=pos, impl=InlineCode(code="## MarkdownNode"))
            elif node_type == "compute":
                code = get_default_custom_node_code()
                parsed_cfg = parse_code_to_compute_cfg(code)
                new_node = ComputeNodeModel(nType=ComputeNodeType.COMPUTE, id=node_id, position=pos, impl=InlineCode(code=code),
                    name=parsed_cfg.name, key=parsed_cfg.key, moduleId=parsed_cfg.module_id)
                target_flow_draft.node_states[node_id] = {}
            else:
                raise NotImplementedError
        else:
            node_type = item_id[len(_USER_NODE_PREFIX):]
            cfg = NODE_REGISTRY.global_dict[node_type]
            new_node = ComputeNodeModel(
                nType=ComputeNodeType.COMPUTE, id=node_id, position=pos, moduleId=cfg.module_id, key=cfg.key,
                    name=cfg.name)
            if cfg.state_dcls is not None:
                target_flow_draft.node_states[node_id] = cfg.state_dcls()

        target_flow_draft.nodes[node_id] = new_node

    async def _on_node_config(self, data: ConfigDialogEvent):
        setting_obj = data.cfg
        assert isinstance(setting_obj, NodeSettings)
        node_draft = data.userdata
        node_draft.vExecId = setting_obj.executor_id

    async def _on_node_contextmenu(self, data: flowui.NodeContextMenuEvent):
        item_id = data.itemId
        node_id = data.nodeId
        cur_flow = self.dm.model.get_cur_flow()
        assert cur_flow is not None and node_id in cur_flow.nodes
        node = cur_flow.nodes[node_id]
        if item_id == NodeContextMenuItemNames.Run:
            await self.scheduler.run_sub_graph(cur_flow, node_id, self.executors, self._executor_term_buffers)
        elif item_id == NodeContextMenuItemNames.Setting:
            node_draft = self._flow_draft.cur_model.nodes[node_id]
            await self._node_setting_dialog.open_config_dialog(
                NodeSettings(node.vExecId), node_draft
            )
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
