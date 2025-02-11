"""Core functions for draft expression change event.

We use shallow compare by default (use python id built-in function) to compare the old and new value.

Example:

def handle_code_change(new_value):
    ...

model_component.register_draft_change_event(model_draft.a[model_draft.cur_key].code, handle_code_change)

"""

import enum
from typing import Any, Callable, Coroutine, Optional, Union

from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.core.datamodel.draft import (
    DraftASTNode, DraftBase, DraftUpdateOp,
    apply_draft_update_ops_with_changed_obj_ids, evaluate_draft_ast_noexcept,
    get_draft_ast_node)

CORO_NONE = Union[Coroutine[None, None, None], None]


class DraftEventType(enum.IntEnum):
    NoChange = 0
    ValueChange = 1
    ObjectIdChange = 2
    ValueChangeCustom = 3
    ObjectInplaceChange = 4
    ChildObjectChange = 5


@dataclasses.dataclass
class DraftChangeEvent:
    type: DraftEventType
    new_value: Any
    user_eval_vars: Optional[dict[str, Optional[DraftASTNode]]] = None


@dataclasses.dataclass
class DraftChangeEventHandler:
    draft_expr: DraftASTNode
    handler: Callable[[DraftChangeEvent], CORO_NONE]
    equality_fn: Optional[Callable[[Any, Any], bool]] = None
    handle_child_change: bool = False
    user_eval_vars: Optional[dict[str, DraftASTNode]] = None


def create_draft_change_event_handler(
        draft_obj: Any,
        handler: Callable[[DraftChangeEvent], CORO_NONE],
        equality_fn: Optional[Callable[[Any, Any], bool]] = None,
        handle_child_change: bool = False):
    assert isinstance(draft_obj, DraftBase)
    return DraftChangeEventHandler(get_draft_ast_node(draft_obj), handler,
                                   equality_fn, handle_child_change)


def update_model_with_change_event(
        model: Any, ops: list[DraftUpdateOp],
        event_handlers: list[DraftChangeEventHandler]):
    # 1. eval all draft expressions and record old value (or id)
    handler_change_type_and_new_val: list[tuple[DraftEventType, Any]] = []
    handler_old_values: list[tuple[Any, bool]] = []
    for handler in event_handlers:
        obj = evaluate_draft_ast_noexcept(handler.draft_expr, model)
        if obj is not None:
            if isinstance(obj, (bool, int, float, str, type(None))):
                handler_old_values.append((obj, False))
            else:
                handler_old_values.append((obj, True))
    changed_parent_obj_ids, changed_obj_ids = apply_draft_update_ops_with_changed_obj_ids(
        model, ops)
    for handler, (old_value, is_obj) in zip(event_handlers,
                                            handler_old_values):
        new_value = evaluate_draft_ast_noexcept(handler.draft_expr, model)
        ev_type = DraftEventType.NoChange
        if handler.equality_fn is not None:
            if not handler.equality_fn(old_value, new_value):
                ev_type = DraftEventType.ValueChangeCustom
            handler_change_type_and_new_val.append((ev_type, new_value))
            continue
        if is_obj:
            is_not_equal = id(old_value) != id(new_value)
            if not is_not_equal:
                if id(old_value) in changed_obj_ids:
                    ev_type = DraftEventType.ObjectInplaceChange
                elif handler.handle_child_change:
                    if id(old_value) in changed_parent_obj_ids:
                        ev_type = DraftEventType.ChildObjectChange
            else:
                ev_type = DraftEventType.ObjectIdChange
        else:
            if old_value != new_value:
                ev_type = DraftEventType.ValueChange
        handler_change_type_and_new_val.append((ev_type, new_value))
    return handler_change_type_and_new_val

