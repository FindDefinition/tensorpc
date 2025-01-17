import copy
from typing import cast
from tensorpc.core.datamodel.draft import DraftObject, capture_draft_update, apply_draft_jmes_ops_backend, apply_draft_jmes_ops
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
    f: dict[str, int]
    g: list[dict[str, int]]
    h: NestedModel
    i: list[NestedModel]


def test_draft():
    model = Model(
        a=1, b=2.0, c=True, d="test", e=[1, 2, 3], f={"a": 1, "b": 2}, 
        g=[{"a": 1, "b": 2}, {"a": 3, "b": 4}], h=NestedModel(a=1, b=2.0), 
        i=[NestedModel(a=1, b=2.0), NestedModel(a=3, b=4.0)]
    )

    model_ref = copy.deepcopy(model)

    draft = cast(Model, DraftObject(model))

    with capture_draft_update() as ctx:
        draft.a = 2
        draft.h.a = 3
        draft.i[0].a = 4
        draft.e.extend([5, 6, 7])

        model_ref.a = 2
        model_ref.h.a = 3
        model_ref.i[0].a = 4
        model_ref.e.extend([5, 6, 7])
    model_for_jpath = copy.deepcopy(model)
    model_for_jpath_dict = dataclasses.asdict(model_for_jpath)
    apply_draft_jmes_ops_backend(model, ctx._ops)
    apply_draft_jmes_ops(model_for_jpath_dict, [op.to_jmes_path_op() for op in ctx._ops])

    ddiff = DeepDiff(dataclasses.asdict(model), dataclasses.asdict(model_ref), ignore_order=True)
    assert not ddiff

    ddiff = DeepDiff(model_for_jpath_dict, dataclasses.asdict(model_ref), ignore_order=True)
    assert not ddiff


if __name__ == "__main__":
    test_draft()