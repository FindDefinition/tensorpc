"""Core functions for draft expression change event.

We use shallow compare by default (use python id built-in function) to compare the old and new value.

Example:

def handle_code_change(new_value):
    ...

model_component.register_draft_change_event(model_draft.a[model_draft.cur_key].code, handle_code_change)

"""

import enum
import time
from typing import Any, Callable, Coroutine, Optional, TypeVar, Union

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
    # when we init model, we should trigger all change event handlers, the type will be this.
    MountChange = 6
    UnmountChange = 7

@dataclasses.dataclass
class DraftChangeEvent:
    type_dict: dict[str, DraftEventType]
    old_value_dict: dict[str, Any]
    new_value_dict: dict[str, Any]
    user_eval_vars: Optional[dict[str, Any]] = None

    def is_item_changed(self, key: str):
        return key in self.type_dict and self.type_dict[key] != DraftEventType.NoChange

    @property 
    def is_changed(self):
        return any(v != DraftEventType.NoChange for v in self.type_dict.values())

    @property 
    def type(self):
        assert len(self.type_dict) == 1, f"you provide more than one draft expr, {self.type_dict}"
        return list(self.type_dict.values())[0]

    @property 
    def new_value(self):
        assert len(self.new_value_dict) == 1, f"you provide more than one draft expr, {self.new_value_dict}"
        return list(self.new_value_dict.values())[0]

    @property
    def old_value(self):
        assert len(self.old_value_dict) == 1, f"you provide more than one draft expr, {self.old_value_dict}, {self.new_value_dict}"
        return list(self.old_value_dict.values())[0]

@dataclasses.dataclass
class DraftChangeEventHandler:
    draft_expr_dict: dict[str, DraftASTNode]
    handler: Callable[[DraftChangeEvent], CORO_NONE]
    equality_fn: Optional[Callable[[Any, Any], bool]] = None
    handle_child_change: bool = False
    user_eval_vars: Optional[dict[str, DraftASTNode]] = None

    _draft_expr_compiled_cache: dict[str, Callable[[Any], Any]] = dataclasses.field(default_factory=dict)
    _user_eval_vars_compiled_cache: dict[str, Callable[[Any], Any]] = dataclasses.field(default_factory=dict)

    def evaluate_draft_expr_noexcept(self, key: str, model: Any):
        if key not in self._draft_expr_compiled_cache:
            self._draft_expr_compiled_cache[key] = self.draft_expr_dict[key].compile()
        try:
            return self._draft_expr_compiled_cache[key](model)
        except BaseException as e:
            return None

    def evaluate_user_eval_var_noexcept(self, key: str, model: Any):
        assert self.user_eval_vars is not None 
        if key not in self._user_eval_vars_compiled_cache:
            self._user_eval_vars_compiled_cache[key] = self.user_eval_vars[key].compile()
        try:
            return self._user_eval_vars_compiled_cache[key](model)
        except BaseException as e:
            return None


def create_draft_change_event_handler(
        draft_obj: Union[Any, dict[str, Any]],
        handler: Callable[[DraftChangeEvent], CORO_NONE],
        equality_fn: Optional[Callable[[Any, Any], bool]] = None,
        handle_child_change: bool = False):
    assert isinstance(draft_obj, DraftBase)
    if isinstance(draft_obj, dict):
        draft_expr_dict = {k: get_draft_ast_node(v) for k, v in draft_obj.items()}
    else:
        draft_expr_dict = {"": get_draft_ast_node(draft_obj)}
    return DraftChangeEventHandler(draft_expr_dict, handler,
                                   equality_fn, handle_child_change)


def update_model_with_change_event(
        model: Any, ops: list[DraftUpdateOp],
        event_handlers: list[DraftChangeEventHandler]):
    # 1. eval all draft expressions and record old value (or id)
    change_events: list[DraftChangeEvent] = []
    handler_old_values: list[dict[str, tuple[Any, bool]]] = []
    for handler in event_handlers:
        old_val_dict = {}
        for k in handler.draft_expr_dict.keys():
            obj = handler.evaluate_draft_expr_noexcept(k, model)
            if isinstance(obj, (bool, int, float, str, type(None))):
                old_val_dict[k] = (obj, False)
            else:
                old_val_dict[k] = (obj, True)
        handler_old_values.append(old_val_dict)
    # 2. perform model update, we also record changed obj ids by parsing draft update ops.
    changed_parent_obj_ids, changed_obj_ids = apply_draft_update_ops_with_changed_obj_ids(
        model, ops)
    # 3. eval all draft expressions again and compare with old value (or id)
    for handler, old_val_dict in zip(event_handlers,
                                            handler_old_values):
        
        new_val_dict: dict[str, Any] = {}
        type_dict: dict[str, DraftEventType] = {}
        for k in handler.draft_expr_dict.keys():
            old_value, is_obj = old_val_dict[k]
            new_value = handler.evaluate_draft_expr_noexcept(k, model)
            ev_type = DraftEventType.NoChange
            if handler.equality_fn is not None:
                if not handler.equality_fn(old_value, new_value):
                    ev_type = DraftEventType.ValueChangeCustom
                type_dict[k] = ev_type
                new_val_dict[k] = new_value
                continue
            if is_obj:
                obj_id_not_equal = id(old_value) != id(new_value)
                if not obj_id_not_equal:
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
            type_dict[k] = ev_type
            new_val_dict[k] = new_value
        change_events.append(DraftChangeEvent(type_dict, {k: v[0] for k, v in old_val_dict.items()}, new_val_dict))
    return change_events

