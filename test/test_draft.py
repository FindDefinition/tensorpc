import copy
from typing import Annotated, cast
from tensorpc.core.datamodel.draft import DraftObject, capture_draft_update, apply_draft_update_ops, apply_draft_jmes_ops, create_draft, create_draft_type_only, get_draft_anno_type, get_draft_ast_node, insert_assign_draft_op
from tensorpc.core import dataclass_dispatch as dataclasses
from deepdiff.diff import DeepDiff

@dataclasses.dataclass
class NestedModel:
    a: int 
    b: float 
    
@dataclasses.dataclass
class Model:
    a: int 
    b: float 
    c: bool 
    d: str 
    e: list[int]
    f: Annotated[dict[str, int], "WTFWTF"]
    g: list[dict[str, int]]
    h: NestedModel
    i: list[NestedModel]


def test_draft(type_only_draft: bool = True):
    model = Model(
        a=0, b=2.0, c=True, d="test", e=[1, 2, 3], f={"a": 1, "b": 2}, 
        g=[{"a": 1, "b": 2}, {"a": 3, "b": 4}], h=NestedModel(a=1, b=2.0), 
        i=[NestedModel(a=1, b=2.0), NestedModel(a=3, b=4.0)]
    )

    model_ref = copy.deepcopy(model)
    if type_only_draft:
        draft = create_draft_type_only(type(model))
    else:
        draft = create_draft(model)

    with capture_draft_update() as ctx:
        draft.a = 1
        draft.b += 5
        draft.h.a = 3
        draft.i[0].a = 4
        draft.f["a"] = 5
        insert_assign_draft_op(draft.i[draft.a].b, 7)
        draft.i[draft.a].a = 5
        draft.e[draft.a] = 8
        draft.e.extend([5, 6, 7])
        draft.e[0] -= 3

        model_ref.a = 1
        model_ref.b += 5
        model_ref.h.a = 3
        model_ref.i[0].a = 4
        model_ref.f["a"] = 5
        model_ref.i[model_ref.a].b = 7
        model_ref.i[model_ref.a].a = 5
        model_ref.e[model_ref.a] = 8
        model_ref.e.extend([5, 6, 7])
        model_ref.e[0] -= 3
    print(get_draft_anno_type(draft.f))
    print(get_draft_ast_node(draft.e[draft.a]).get_jmes_path())
    model_for_jpath = copy.deepcopy(model)
    model_for_jpath_dict = dataclasses.asdict(model_for_jpath)
    apply_draft_update_ops(model, ctx._ops)
    apply_draft_jmes_ops(model_for_jpath_dict, [op.to_jmes_path_op() for op in ctx._ops])

    ddiff = DeepDiff(dataclasses.asdict(model), dataclasses.asdict(model_ref), ignore_order=True)
    assert not ddiff, str(ddiff)

    ddiff = DeepDiff(model_for_jpath_dict, dataclasses.asdict(model_ref), ignore_order=True)
    assert not ddiff


if __name__ == "__main__":
    test_draft(False)
    test_draft(True)