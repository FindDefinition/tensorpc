import contextvars
import contextlib

from typing import Any, Callable, Optional, TypeVar, Union
from tensorpc.core.annolib import DataclassType
from tensorpc.core.datamodel.draft import cast_any_draft_to_dataclass, get_draft_ast_node
from tensorpc.core.datamodel.draftast import evaluate_draft_ast_noexcept

T = TypeVar("T", bound=DataclassType)

class ComputeFlowNodeContext:
    def __init__(self, node_id: str, state: Any, state_draft: Any) -> None:
        self.node_id = node_id

        self.state = state
        self.state_draft = state_draft


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

@contextlib.contextmanager
def enter_flow_ui_node_context(node_id: str, state: Any, state_draft: Any):
    ctx = ComputeFlowNodeContext(node_id, state, state_draft)
    token = COMPUTE_FLOW_NODE_CONTEXT_VAR.set(ctx)
    try:
        yield ctx
    finally:
        COMPUTE_FLOW_NODE_CONTEXT_VAR.reset(token)

def get_node_state_draft(state_ty: type[T]) -> tuple[T, T]:
    ctx = get_compute_flow_node_context()
    assert ctx is not None, "No context found for node state draft"
    state = ctx.state 
    assert isinstance(state, state_ty), f"Node state is not of type {state_ty}"
    return state, cast_any_draft_to_dataclass(ctx.state_draft, state_ty)