from functools import partial
import traceback
from tensorpc.apps.cflow.executors.base import NodeExecutorBase
from tensorpc.core.datamodel.draft import get_draft_ast_node
from tensorpc.core.datamodel.draftast import evaluate_draft_ast_noexcept
from tensorpc.core.datamodel.events import DraftChangeEvent
from tensorpc.core.tree_id import UniqueTreeIdForTree
from tensorpc.dock.components import flowui
from tensorpc.apps.cflow.nodes.cnode.registry import NODE_REGISTRY, get_compute_node_runtime, parse_code_to_compute_cfg
from tensorpc.apps.cflow.nodes.mdnode import MarkdownNodeWrapper
from tensorpc.dock.components.flowplus.style import ComputeFlowClasses
from tensorpc.dock.components.flowui import Node, Edge, Flow
from tensorpc.dock import mui
from typing import Annotated, Any, Callable, Optional, cast
from tensorpc.dock import models
import tensorpc.core.datamodel as D

from .model import ComputeFlowDrafts, ComputeFlowNodeDrafts, ComputeFlowModelRoot, ComputeNodeModel, ComputeNodeType, ComputeFlowModel
from .nodes.cnode.wrapper import ComputeNodeWrapper
import dataclasses as dataclasses_plain

@dataclasses_plain.dataclass
class FlowPanelComps:
    detail: mui.FlexBox 
    debug: mui.FlexBox

class ComputeFlowBinder:

    def __init__(self, flow_comp: Flow, flow_comp_preview: Flow,
                 drafts: ComputeFlowDrafts, panel_comps: FlowPanelComps):
        self.flow_comp = flow_comp
        self.flow_comp_preview = flow_comp_preview
        self.drafts = drafts
        self.panel_comps = panel_comps

    def _get_preview_flow_uid(self, path_draft,
                              dm_comp: mui.DataModel[ComputeFlowModelRoot]):
        path = D.evaluate_draft(path_draft, dm_comp.model)
        if path is None:
            return "root"
        return UniqueTreeIdForTree.from_parts(path).uid_encoded

    def to_ui_node(self, flow: ComputeFlowModel, node_id: str,
                   dm_comp: mui.DataModel[ComputeFlowModelRoot]) -> Node:
        node_model = flow.nodes[node_id]
        if node_model.nType == ComputeNodeType.COMPUTE:
            root_model = dm_comp.get_model()
            draft = self.drafts.get_node_drafts(node_model.id)
            if node_model.runtime is None:
                runtime = node_model.get_node_runtime(root_model)
                node_model.runtime = runtime
            state_d = flow.create_or_convert_node_state(node_model.id)
            wrapper = ComputeNodeWrapper(node_model.id, node_model.runtime.cfg, state_d, node_model.runtime.cnode, draft)
            if node_model.codeKey is not None or node_model.key == "":
                dm_comp.install_draft_change_handler(
                    draft.code,
                    partial(self._handle_node_code_draft_change,
                            wrapper=wrapper,
                            draft=draft,
                            node_model=node_model),
                    installed_comp=wrapper,
                    user_eval_vars={
                        "state": draft.node_state
                    })
            # dm_comp.debug_print_draft_change(draft.node.name)
            # deletable: we use custom delete instead of delete in flowui.
            ui_node = Node(node_model.id,
                           type="app",
                           data=flowui.NodeData(component=wrapper),
                           deletable=False,
                           position=node_model.position)
            ui_node.dragHandle = f".{ComputeFlowClasses.Header}"
        elif node_model.nType == ComputeNodeType.MARKDOWN:
            # markdown node won't be stored in registry, it's fully controlled.
            draft = self.drafts.get_node_drafts(node_model.id)
            wrapper = MarkdownNodeWrapper(node_model.id, draft)
            ui_node = Node(node_model.id,
                           type="app",
                           data=flowui.NodeData(component=wrapper),
                           deletable=False,
                           position=node_model.position)
        else:
            raise NotImplementedError
        return ui_node

    async def _handle_node_code_draft_change(self, ev: DraftChangeEvent,
                                             wrapper: ComputeNodeWrapper,
                                             draft: ComputeFlowNodeDrafts,
                                             node_model: ComputeNodeModel):
        # evaluate new state
        assert ev.user_eval_vars is not None 
        new_state = ev.user_eval_vars["state"]
        cfg = parse_code_to_compute_cfg(ev.new_value)
        runtime = get_compute_node_runtime(cfg)
        node_model.runtime = runtime
        await wrapper.set_node_from_code(cfg, new_state, runtime.cnode, draft)


    async def _handle_node_executor_change(self, ev: DraftChangeEvent):
        # evaluate new state
        executor = ev.new_value
        if executor is None:
            # TODO reset preview layout and detail layout
            # reset executor debug panel
            if ev.old_value is not None and ev.old_value.is_local():
                await self.panel_comps.debug.set_new_layout([
                    mui.Markdown("## Debug")
                ])
        else:
            if ev.old_value is ev.new_value:
                # no change
                return
            # TODO set remote preview layout to wrapper
            assert isinstance(executor, NodeExecutorBase)
            bottom_layout = executor.get_bottom_layout()
            # print("EXECUTOR!!!", ev.old_value, ev.new_value, bottom_layout)

            if bottom_layout is not None:
                await self.panel_comps.debug.set_new_layout([
                    bottom_layout
                ])

    def bind_flow_comp_with_datamodel(self, dm_comp: mui.DataModel[ComputeFlowModelRoot]):
        binder = models.flow.BaseFlowModelBinder(
            self.flow_comp,
            dm_comp.get_model,
            self.drafts.cur_model,
            partial(self.to_ui_node, dm_comp=dm_comp),
            flow_uid_getter=lambda: dm_comp.get_model().get_uid_from_path(),
            debug_id="flow")
        preview_binder = models.flow.BaseFlowModelBinder(
            self.flow_comp_preview,
            dm_comp.get_model,
            self.drafts.preview_model,
            partial(self.to_ui_node, dm_comp=dm_comp),
            flow_uid_getter=partial(self._get_preview_flow_uid,
                                    path_draft=self.drafts.preview_path,
                                    dm_comp=dm_comp),
            debug_id="flow_preview")
        # dm_comp.debug_print_draft_change(self.drafts.preview_path)
        
        # dm_comp.debug_print_draft_change(self.drafts.root.settings.isRightPanelVisible)

        # dm_comp.debug_print_draft_change(self.drafts.selected_node_code)

        binder.bind_flow_comp_with_base_model(
            dm_comp, self.drafts.cur_model.selected_node)
        preview_binder.bind_flow_comp_with_base_model(
            dm_comp, self.drafts.preview_model.selected_node)

        dm_comp.install_draft_change_handler(
            self.drafts.selected_node.runtime.executor,
            self._handle_node_executor_change,
            installed_comp=self.flow_comp)
