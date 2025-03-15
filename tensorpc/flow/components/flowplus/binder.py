
from functools import partial
from tensorpc.core.datamodel.events import DraftChangeEvent
from tensorpc.core.tree_id import UniqueTreeIdForTree
from tensorpc.flow.components import flowui
from tensorpc.flow.components.flowplus.nodes.cnode.registry import NODE_REGISTRY
from tensorpc.flow.components.flowplus.nodes.mdnode import MarkdownNodeWrapper
from tensorpc.flow.components.flowui import Node, Edge, Flow
from tensorpc.flow import mui 
from typing import Annotated, Any, Callable, Optional, cast
from tensorpc.flow import models
import tensorpc.core.datamodel as D

from tensorpc.flow.core.datamodel import DataModel
from .model import ComputeFlowDrafts, ComputeFlowNodeDrafts, ComputeFlowModelRoot, ComputeFlowNodeModel, ComputeNodeType
from .nodes.cnode.wrapper import ComputeNodeWrapper

class ComputeFlowBinder:
    def __init__(self, flow_comp: Flow, flow_comp_preview: Flow, dm_comp: DataModel[ComputeFlowModelRoot], drafts: ComputeFlowDrafts):
        self.flow_comp = flow_comp
        self.drafts = drafts
        self.root_model_getter = dm_comp.get_model
        self.dm_comp = dm_comp

        self.binder = models.flow.BaseFlowModelBinder(
            flow_comp, dm_comp.get_model, 
            drafts.cur_model_draft, 
            self.to_ui_node,
            flow_uid_getter=lambda: dm_comp.get_model().get_uid_from_path())
        self.preview_binder = models.flow.BaseFlowModelBinder(
            flow_comp_preview, dm_comp.get_model, 
            drafts.preview_model_draft, 
            self.to_ui_node,
            flow_uid_getter=partial(self._get_preview_flow_uid, drafts.prev_path_draft))

    def _get_preview_flow_uid(self, path_draft):
        path = D.evaluate_draft(path_draft, self.dm_comp.model)
        if path is None:
            return "root"
        return UniqueTreeIdForTree.from_parts(path).uid_encoded

    def to_ui_node(self, node_model: ComputeFlowNodeModel):
        if node_model.node_type == ComputeNodeType.COMPUTE:
            root_model = self.root_model_getter()
            draft = self.drafts.get_node_drafts(node_model.id)
            if node_model.code_key is not None:
                code = root_model.shared_node_code[node_model.code_key]
                wrapper = ComputeNodeWrapper(node_model.id, code, draft)
            elif node_model.key != "":
                # node stored in registry.
                cfg = NODE_REGISTRY.global_dict[node_model.key]
                wrapper = ComputeNodeWrapper(node_model.id, cfg, draft)
            else:
                code = draft.code
                wrapper = ComputeNodeWrapper(node_model.id, code, draft)
            # deletable: we use custom delete instead of delete in flowui.
            self.dm_comp.install_draft_change_handler(draft.code, partial(self._handle_node_draft_change, wrapper=wrapper, draft=draft))
            return Node(node_model.id, type="app", data=flowui.NodeData(component=wrapper), deletable=False)
        elif node_model.node_type == ComputeNodeType.MARKDOWN:
            # markdown node won't be stored in registry, it's fully controlled.
            draft = self.drafts.get_node_drafts(node_model.id)
            wrapper = MarkdownNodeWrapper(node_model.id, draft)
            return Node(node_model.id, type="app", data=flowui.NodeData(component=wrapper), deletable=False)
        else:
            raise NotImplementedError


    async def _handle_node_draft_change(self, ev: DraftChangeEvent, wrapper: ComputeNodeWrapper, draft: ComputeFlowNodeDrafts):
        await wrapper.set_node_from_code(ev.new_value, draft)

    def bind_flow_comp_with_datamodel(self, dm_comp: mui.DataModel):

        pass 
