import contextvars
import contextlib

from typing import Any, Callable, Optional, TypeVar, Union
from tensorpc.core.annolib import DataclassType
from tensorpc.core.datamodel.draft import get_draft_ast_node
from tensorpc.core.datamodel.draftast import evaluate_draft_ast_noexcept
from tensorpc.apps.cflow.model import ComputeFlowNodeDrafts

T = TypeVar("T", bound=DataclassType)

class ComputeFlowNodeContext:
    def __init__(self, node_id: str, drafts: ComputeFlowNodeDrafts, root_model_getter: Any) -> None:
        self.node_id = node_id

        self.drafts = drafts
        self.root_model_getter = root_model_getter


COMPUTE_FLOW_NODE_CONTEXT_VAR: contextvars.ContextVar[
    Optional[ComputeFlowNodeContext]] = contextvars.ContextVar(
        "computeflow_node_context_v2", default=None)


def get_compute_flow_node_context() -> Optional[ComputeFlowNodeContext]:
    return COMPUTE_FLOW_NODE_CONTEXT_VAR.get()


@contextlib.contextmanager
def enter_flow_ui_node_context_object(ctx: ComputeFlowNodeContext):
    token = COMPUTE_FLOW_NODE_CONTEXT_VAR.set(ctx)
    try:
        yield ctx
    finally:
        COMPUTE_FLOW_NODE_CONTEXT_VAR.reset(token)

def get_node_state_draft(state_ty: type[T]) -> tuple[T, T]:
    ctx = get_compute_flow_node_context()
    assert ctx is not None, "No context found for node state draft"
    state = evaluate_draft_ast_noexcept(get_draft_ast_node(ctx.drafts.node_state), ctx.root_model_getter())
    assert state is not None, "you must register a state dataclass if use this function"
    if isinstance(state, dict):
        pass 
    assert isinstance(state, state_ty), f"Invalid state type {type(state)}, expected {state_ty}"
    return state, ctx.drafts.node_state