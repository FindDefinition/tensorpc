import contextvars
import contextlib

from typing import Callable, Optional, TypeVar, Union


class ComputeFlowNodeContext:
    def __init__(self, node_id: str) -> None:
        self.node_id = node_id


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
