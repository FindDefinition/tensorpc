"""Core functions for draft expression change event.

We use shallow compare by default (use python id built-in function) to compare the old and new value.

Example:

def handle_code_change(new_value):
    ...

model_component.register_draft_change_event(model_draft.a[model_draft.cur_key].code, handle_code_change)

"""