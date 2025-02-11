import asyncio
import base64
import copy
from typing import Annotated, Any, cast

from deepdiff.diff import DeepDiff

from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.core.datamodel.draft import (
    DraftObject, apply_draft_jmes_ops, apply_draft_update_ops,
    apply_draft_update_ops_with_changed_obj_ids, capture_draft_update, cast_any_draft_to_dataclass,
    create_draft, create_draft_type_only, create_literal_draft, get_draft_anno_path_metas,
    get_draft_anno_type, get_draft_ast_node, insert_assign_draft_op, materialize_any_draft_to_dataclass)
from tensorpc.core.datamodel.draftast import evaluate_draft_ast, evaluate_draft_ast_json
from tensorpc.core.datamodel.draftstore import (DraftFileStorage,
                                                DraftFileStoreBackendInMemory,
                                                DraftStoreMapMeta,
                                                get_splitted_update_model_ops,
                                                validate_splitted_model_type)
from tensorpc.core.datamodel.events import DraftChangeEventHandler, DraftEventType, create_draft_change_event_handler, update_model_with_change_event
from tensorpc.core.datamodel import jmes as jmespath

@dataclasses.dataclass
class NestedModel:
    a: int 
    b: float
    c: Annotated[str, "AnnoNestedMeta1"] = ""

@dataclasses.dataclass
class Model:
    a: int 
    b: float 
    c: bool 
    d: str 
    e: "list[int]"
    f: Annotated[dict[str, int], "AnnoMeta1"]
    g: list[dict[str, int]]
    h: NestedModel
    i: list[NestedModel]
    j: Annotated[dict[str, NestedModel], "AnnoMeta2"]
    fmt: str = "%s"

    def set_i_b(self, idx: int, value: float):
        self.i[idx].b = value

def modify_func(mod: Model):
    mod.a = 1
    mod.b += 5
    mod.h.a = 3
    mod.i[0].a = 4
    mod.f["a"] = 5
    mod.i[mod.a].b = 7
    mod.i[mod.a].a = 5
    mod.e[mod.a] = 8
    mod.e.extend([5, 6, 7])
    mod.e[0] -= 3
    mod.set_i_b(0, 7)
    mod.e.pop()
    mod.f.pop("b")
    mod.f.update({
        "c": 5
    })
    mod.f["d"] = 6
    del mod.f["a"]

def _draft_expr_examples(mod: Model, is_draft: bool):
    return {
        "fmt1": mod.fmt % mod.a,
        "fmt2": mod.fmt % mod.b,
        "fmt3": mod.fmt % mod.d,
        "fmt4": create_literal_draft("Hello World! %s %s") % (mod.a, mod.b) if is_draft else "Hello World! %s %s" % (mod.a, mod.b),
    }


def _get_test_model():
    model = Model(
        a=0, b=2.0, c=True, d="test", e=[1, 2, 3], f={"a": 1, "b": 2}, 
        g=[{"a": 1, "b": 2}, {"a": 3, "b": 4}], h=NestedModel(a=1, b=2.0), 
        i=[NestedModel(a=1, b=2.0), NestedModel(a=3, b=4.0)],
        j={"a": NestedModel(a=1, b=2.0), "b": NestedModel(a=3, b=4.0)}
    )
    return model

def test_draft(type_only_draft: bool = True):
    model = _get_test_model()
    model_ref = copy.deepcopy(model)
    if type_only_draft:
        draft = create_draft_type_only(type(model))
    else:
        draft = create_draft(model)
    # print(get_draft_anno_path_metas(draft.f))
    assert get_draft_anno_path_metas(draft.f) == [("AnnoMeta1",)]
    assert get_draft_anno_path_metas(draft.j["a"].c) == [("AnnoMeta2",), ("AnnoNestedMeta1",)]
    assert get_draft_anno_path_metas(draft.j["a"].a) == [("AnnoMeta2",)]

    with capture_draft_update() as ctx:
        modify_func(model_ref)
        modify_func(draft)
    # print(get_draft_anno_type(draft.f))
    # print(get_draft_ast_node(draft.e[draft.a]).get_jmes_path())
    model_for_jpath = copy.deepcopy(model)
    model_for_jpath_dict = dataclasses.asdict(model_for_jpath)
    apply_draft_update_ops_with_changed_obj_ids(model, ctx._ops)
    apply_draft_jmes_ops(model_for_jpath_dict, [op.to_jmes_path_op() for op in ctx._ops])

    ddiff = DeepDiff(dataclasses.asdict(model), dataclasses.asdict(model_ref), ignore_order=True)
    assert not ddiff, str(ddiff)

    ddiff = DeepDiff(model_for_jpath_dict, dataclasses.asdict(model_ref), ignore_order=True)
    assert not ddiff

def test_draft_expr():
    model = _get_test_model()
    model_ref = copy.deepcopy(model)
    exprs_res_ref = _draft_expr_examples(model_ref, False)
    draft = create_draft_type_only(type(model))
    exprs = _draft_expr_examples(draft, True)

    exprs_res = {}
    for k, expr in exprs.items():
        exprs_res[k] = evaluate_draft_ast(get_draft_ast_node(cast(Any, expr)), model)
    ddiff = DeepDiff(exprs_res_ref, exprs_res, ignore_order=True)
    assert not ddiff, str(ddiff)
    exprs_res = {}
    for k, expr in exprs.items():
        exprs_res[k] = evaluate_draft_ast_json(get_draft_ast_node(cast(Any, expr)), dataclasses.asdict(model))
    ddiff = DeepDiff(exprs_res_ref, exprs_res, ignore_order=True)
    assert not ddiff, str(ddiff)

    exprs_res = {}
    for k, expr in exprs.items():
        jmes_path = get_draft_ast_node(cast(Any, expr)).get_jmes_path()
        jmes_expr = jmespath.compile(jmes_path)
        exprs_res[k] = jmespath.search(jmes_expr, dataclasses.asdict(model))
    ddiff = DeepDiff(exprs_res_ref, exprs_res, ignore_order=True)
    assert not ddiff, str(ddiff)


@dataclasses.dataclass
class SplittedSubModel:
    d: int
    e_split: Annotated[dict[str, Any], DraftStoreMapMeta()] 
    f: list[int]


@dataclasses.dataclass
class SplittedModel:
    a: int 
    b: float 
    c_split: Annotated[dict[str, int], DraftStoreMapMeta("modified_name")] 
    d_split: Annotated[dict[str, SplittedSubModel], DraftStoreMapMeta()] 


def modify_func_splitted(mod: SplittedModel):
    mod.a = 1
    mod.b += 5
    mod.c_split["a"] += 3
    mod.c_split["c"] = 5
    mod.c_split.update({
        "b": 4
    })
    mod.c_split["b"] = 4
    mod.d_split["a"].d = 5
    mod.d_split["a"].e_split["b"] = 6
    mod.d_split["a"].f[0] = 6
    mod.c_split.pop("a")
    mod.d_split["a"].e_split.pop("a")

def _b64encode(key: str):
    return base64.b64encode(key.encode()).decode()

def test_splitted_draft(type_only_draft: bool = True):
    validate_splitted_model_type(SplittedModel)
    model = SplittedModel(a=0, b=2.0, c_split={"a": 1, "b": 2}, d_split={"a": SplittedSubModel(d=1, e_split={"a": 1, "b": 2}, f=[5])})
    model_ref = copy.deepcopy(model)
    if type_only_draft:
        draft = create_draft_type_only(type(model))
    else:
        draft = create_draft(model)
    with capture_draft_update() as ctx:
        modify_func_splitted(model_ref)
        modify_func_splitted(draft)

    model_for_jpath = copy.deepcopy(model)
    model_for_jpath_dict = dataclasses.asdict(model_for_jpath)
    splitted_update_ops, _, _ = get_splitted_update_model_ops("root", ctx._ops, model_ref)
    # print(splitted_update_ops)
    assert f"root/modified_name/{_b64encode('a')}" in splitted_update_ops
    assert f"root/modified_name/{_b64encode('b')}" in splitted_update_ops
    assert f"root/d_split/{_b64encode('a')}" in splitted_update_ops
    assert f"root/d_split/{_b64encode('a')}/e_split/{_b64encode('b')}" in splitted_update_ops

    apply_draft_update_ops(model, ctx._ops)
    apply_draft_jmes_ops(model_for_jpath_dict, [op.to_jmes_path_op() for op in ctx._ops])

    ddiff = DeepDiff(dataclasses.asdict(model), dataclasses.asdict(model_ref), ignore_order=True)
    assert not ddiff, str(ddiff)

    ddiff = DeepDiff(model_for_jpath_dict, dataclasses.asdict(model_ref), ignore_order=True)
    assert not ddiff

async def test_splitted_draft_store(type_only_draft: bool = True):
    store_backend = DraftFileStoreBackendInMemory()
    validate_splitted_model_type(SplittedModel)
    model = SplittedModel(a=0, b=2.0, c_split={"a": 1, "b": 2}, d_split={"a": SplittedSubModel(d=1, e_split={"a": 1, "b": 2}, f=[5])})
    model_ref = copy.deepcopy(model)

    store = DraftFileStorage("test", model, store_backend)
    model_loaded = await store.fetch_model()
    # import rich 
    # rich.print(store_backend._data)
    # rich.print(model_loaded)
    ddiff = DeepDiff(dataclasses.asdict(model_loaded), dataclasses.asdict(model_ref), ignore_order=True)
    assert not ddiff, str(ddiff)
    if type_only_draft:
        draft = create_draft_type_only(type(model))
    else:
        draft = create_draft(model)
    with capture_draft_update() as ctx:
        modify_func_splitted(model_ref)
        modify_func_splitted(draft)
    splitted_update_ops = get_splitted_update_model_ops("test", ctx._ops, store._model)
    # rich.print(splitted_update_ops)
    await store.update_model(ctx._ops)
    store_my = DraftFileStorage("test", model, store_backend)
    model_my = await store_my.fetch_model()
    # import rich 
    # rich.print(dataclasses.asdict(model_my),)
    # rich.print(dataclasses.asdict(model_ref),)
    # rich.print(store_backend._data)

    ddiff = DeepDiff(dataclasses.asdict(model_my), dataclasses.asdict(model_ref), ignore_order=True)
    assert not ddiff, str(ddiff)

def modify_func_for_event(mod: Model):
    mod.a = 1
    mod.i[0].a = 4
    mod.f["a"] = 5
    mod.e.pop()


def test_draft_event():
    model = _get_test_model()
    draft = create_draft_type_only(type(model))

    handler1 = create_draft_change_event_handler(draft.a, lambda x: None)
    handler2 = create_draft_change_event_handler(draft.i, lambda x: None, handle_child_change=True)
    handler3 = create_draft_change_event_handler(draft.i, lambda x: None, handle_child_change=False)
    handler4 = create_draft_change_event_handler(draft.e, lambda x: None)
    with capture_draft_update() as ctx:
        modify_func_for_event(draft)
    ev_type_and_val = update_model_with_change_event(model, ctx._ops, [handler1, handler2, handler3, handler4])
    ev_types = [ev_type for ev_type, _ in ev_type_and_val]
    assert ev_types == [
        DraftEventType.ValueChange, 
        DraftEventType.ChildObjectChange, 
        DraftEventType.NoChange, 
        DraftEventType.ObjectInplaceChange,
    ]
    assert ev_type_and_val[0][1] == 1

@dataclasses.dataclass
class ModelWithAny:
    a: Any 
    b: dict[str, Any]

@dataclasses.dataclass
class ModelWithAnyA:
    c: int 
    d: str 

@dataclasses.dataclass
class ModelWithAnyB:
    e: str 


def test_draft_any_materialize():
    model = ModelWithAny({"c": 1, "d": "test"}, {"a": {"c": 1, "d": "test"}, "b": {"e": "testX"}})
    # currently we only support materialize type-only draft.
    draft = create_draft_type_only(type(model))
    draft_a_typed = cast_any_draft_to_dataclass(draft.a, ModelWithAnyA)
    materialize_any_draft_to_dataclass(model, draft.a, ModelWithAnyA)

    assert isinstance(model.a, ModelWithAnyA)
    assert model.a.c == 1
    assert model.a.d == "test"

    with capture_draft_update() as ctx:
        draft_a_typed.c = 2
        draft_a_typed.d = "test2"

    apply_draft_update_ops(model, ctx._ops)

    assert model.a.c == 2
    assert model.a.d == "test2"
    # print(get_draft_anno_type(draft_a_typed.c), get_draft_ast_node(draft_a_typed.c).get_jmes_path())

    materialize_any_draft_to_dataclass(model, draft.b["a"], ModelWithAnyA)
    materialize_any_draft_to_dataclass(model, draft.b["b"], ModelWithAnyB)

    assert isinstance(model.b["a"], ModelWithAnyA)
    assert model.b["a"].c == 1
    assert model.b["a"].d == "test"
    assert isinstance(model.b["b"], ModelWithAnyB)
    assert model.b["b"].e == "testX"
    # draft_b_typed = cast_any_draft_to_dataclass(draft.b["a"], ModelWithAnyA)
    # print(get_draft_anno_type(draft_b_typed.c), get_draft_ast_node(draft_b_typed.c).get_jmes_path())


if __name__ == "__main__":
    test_draft(False)
    test_draft(True)

    asyncio.run(test_splitted_draft_store())

    test_splitted_draft(True)
    test_splitted_draft(False)

    test_draft_event()

    test_draft_any_materialize()

    test_draft_expr()

