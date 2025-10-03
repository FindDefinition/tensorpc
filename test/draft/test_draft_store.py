import asyncio
import base64
import copy
from dataclasses import is_dataclass
import inspect
from typing import Annotated, Any, Generic, Optional, TypeVar, cast, get_origin

from deepdiff.diff import DeepDiff

from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.core.datamodel.draft import (
    DraftFieldMeta, DraftObject, apply_draft_path_ops, apply_draft_update_ops,
    apply_draft_update_ops_with_changed_obj_ids, capture_draft_update, cast_any_draft_to_dataclass,
    create_draft, create_draft_type_only, create_literal_draft, get_draft_anno_path_metas,
    get_draft_anno_type, get_draft_ast_node, insert_assign_draft_op, materialize_any_draft_to_dataclass, rebuild_and_stabilize_draft_expr)

import tensorpc.core.datamodel.funcs as D
from tensorpc.core.datamodel.draftast import evaluate_draft_ast, evaluate_draft_ast_json
from tensorpc.core.datamodel.draftstore import (DraftFileStorage,
                                                DraftFileStoreBackendInMemory,
                                                DraftStoreMapMeta, DraftStoreScalarMeta, SplitNewDeleteOp,
                                                get_splitted_update_model_ops,
                                                analysis_model_store_meta)
from tensorpc.core.datamodel.events import DraftChangeEventHandler, DraftEventType, create_draft_change_event_handler, update_model_with_change_event
from tensorpc.core.datamodel import jmes as jmespath
import pytest_asyncio
import pytest
import rich 
from rich.pretty import pprint


@dataclasses.dataclass
class SplittedSubModel:
    d: int
    e_split: Annotated[dict[str, Any], DraftStoreMapMeta(store_id="another_store")] 
    f: list[int]


@dataclasses.dataclass
class SplittedModel:
    a: int 
    b: float 
    c_split: Annotated[dict[str, int], DraftStoreMapMeta(attr_key="modified_name")] 
    d_split: Annotated[dict[str, SplittedSubModel], DraftStoreMapMeta()] 
    e: Annotated[dict[str, int], DraftFieldMeta(is_external=True)] = dataclasses.field(default_factory=dict)

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
    mod.e["a"] = 9

def _b64encode(key: str):
    return base64.b64encode(key.encode()).decode()

def test_splitted_draft(type_only_draft: bool = True):
    _, field_meta_dict = analysis_model_store_meta(SplittedModel, set())
    model = SplittedModel(a=0, b=2.0, c_split={"a": 1, "b": 2}, d_split={"a": SplittedSubModel(d=1, e_split={"a": 1, "b": 2}, f=[5])}, e={})
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
    splitted_update_ops = get_splitted_update_model_ops("root", ctx._ops, "", field_meta_dict)
    first_group = splitted_update_ops[0]
    assert isinstance(first_group, dict) and len(splitted_update_ops) == 1
    # print(splitted_update_ops)
    assert f"root/modified_name/{_b64encode('a')}" in first_group
    assert f"root/modified_name/{_b64encode('b')}" in first_group
    assert f"root/d_split/{_b64encode('a')}" in first_group
    assert f"root/d_split/{_b64encode('a')}/e_split/{_b64encode('b')}" in first_group

    apply_draft_update_ops(model, ctx._ops)
    apply_draft_path_ops(model_for_jpath_dict, [op.to_jmes_path_op() for op in ctx._ops])

    ddiff = DeepDiff(dataclasses.asdict(model), dataclasses.asdict(model_ref), ignore_order=True)
    assert not ddiff, str(ddiff)

    ddiff = DeepDiff(model_for_jpath_dict, dataclasses.asdict(model_ref), ignore_order=True)
    assert not ddiff

@pytest.mark.asyncio
async def test_splitted_draft_store(type_only_draft: bool = True):
    store_backend = DraftFileStoreBackendInMemory()
    another_store_backend = DraftFileStoreBackendInMemory()
    _, field_store_metas = analysis_model_store_meta(SplittedModel, set())
    model = SplittedModel(a=0, b=2.0, c_split={"a": 1, "b": 2}, d_split={"a": SplittedSubModel(d=1, e_split={"a": 1, "b": 2}, f=[5])}, e={})
    model_ref = copy.deepcopy(model)
    backend_dict = {
        "": store_backend,
        "another_store": another_store_backend
    }
    store = DraftFileStorage("test", model, backend_dict)
    model_loaded = await store.fetch_model()
    # pprint(store_backend._data, expand_all=True)
    assert "e" not in store_backend._data["test"]
    # pprint(store_backend._data, expand_all=True)
    # pprint(another_store_backend._data, expand_all=True)

    # raise NotImplementedError
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
    splitted_update_ops = get_splitted_update_model_ops("test", ctx._ops, "", field_store_metas)
    # rich.print(splitted_update_ops)
    await store.update_model(draft, ctx._ops)
    pprint(store_backend._data, expand_all=True)

    store_my = DraftFileStorage("test", model, backend_dict)

    model_my = await store_my.fetch_model()
    # rich.print(dataclasses.asdict(model_my),)
    # rich.print(dataclasses.asdict(model_ref),)
    # rich.print(store_backend._data)
    # rich.print(another_store_backend._data)
    model_my_dict = dataclasses.asdict(model_my)
    model_my_dict["e"] = {"a": 9} # e don't exists on store, so we add it for test.
    ddiff = DeepDiff(model_my_dict, dataclasses.asdict(model_ref), ignore_order=True)
    assert not ddiff, str(ddiff)

@dataclasses.dataclass
class NestedNode:
    a: int
    models: dict[str, "NestedModelX"]

@dataclasses.dataclass
class NestedModelX:
    nodes: Annotated[dict[str, NestedNode], DraftStoreMapMeta()]

@dataclasses.dataclass
class NestedModelRoot:
    path: list[str]
    model: NestedModelX

@pytest.mark.asyncio
async def test_nested_model():
    model = NestedModelRoot(model=
        NestedModelX(
            nodes={
                "a": NestedNode(a=1, models={"b": NestedModelX(nodes={"c": NestedNode(a=2, models={})})})
            }
        ),
        path=["nodes", "a", "models", "b", "nodes", "c"]
    )
    model_ref = copy.deepcopy(model)
    store_backend = DraftFileStoreBackendInMemory()
    store = DraftFileStorage("test", model, store_backend)
    await store.fetch_model()
    draft = create_draft_type_only(type(model))
    expr_getitempath = D.getitem_path_dynamic(draft.model, draft.path, NestedNode)
    res1 = evaluate_draft_ast(get_draft_ast_node(expr_getitempath), model)

    ddiff = DeepDiff(dataclasses.asdict(res1), dataclasses.asdict(model.model.nodes["a"].models["b"].nodes["c"]), ignore_order=True)
    assert not ddiff, str(ddiff)
    assert get_draft_ast_node(expr_getitempath).get_jmes_path() == "getItemPath($.model,$.path)"
    expr_getitempath_stable = rebuild_and_stabilize_draft_expr(get_draft_ast_node(expr_getitempath), draft, model)
    expr_getitempath_stable_node = get_draft_ast_node(expr_getitempath_stable)
    assert expr_getitempath_stable_node.get_jmes_path() == "$.model.nodes.\"a\".models.\"b\".nodes.\"c\""
    res2 = evaluate_draft_ast(expr_getitempath_stable_node, model)
    ddiff = DeepDiff(dataclasses.asdict(res2), dataclasses.asdict(model.model.nodes["a"].models["b"].nodes["c"]), ignore_order=True)
    assert not ddiff, str(ddiff)

    with capture_draft_update() as ctx:
        expr_getitempath.a = 5
    model_ref.model.nodes["a"].models["b"].nodes["c"].a = 5
    apply_draft_update_ops(model, ctx._ops)
    diff = DeepDiff(dataclasses.asdict(model), dataclasses.asdict(model_ref), ignore_order=True)
    assert not diff, str(diff)

    with capture_draft_update() as ctx:
        expr_getitempath.a = 7
    await store.update_model(draft, ctx._ops)

@dataclasses.dataclass
class NestedNode2:
    id: str
    nodes: Annotated[Optional[dict[str, "NestedNode2"]], DraftStoreMapMeta(base64_key=False)] = None

def _assert_may_dataclass_equal(a: Any, b: Any):
    if dataclasses.is_dataclass(a) and not inspect.isclass(a):
        aa = dataclasses.asdict(a)
    else:
        aa = a
    if dataclasses.is_dataclass(b) and not inspect.isclass(b):
        bb = dataclasses.asdict(b)
    else:
        bb = b
    diff = DeepDiff(aa, bb, ignore_order=True)
    assert not diff, str(diff)

@pytest.mark.asyncio
async def test_nested_model_store():
    model = NestedNode2(id="root")
    model_ref = copy.deepcopy(model)
    draft = create_draft_type_only(type(model))

    store_backend = DraftFileStoreBackendInMemory()
    store = DraftFileStorage("test", model, store_backend)
    model = await store.fetch_model()
    _assert_may_dataclass_equal(store_backend._data, {'test': {'id': 'root', 'nodes': None}})

    nested_model = NestedNode2(id="1")
    with capture_draft_update() as ctx:
        draft.nodes = {"1": nested_model}
        model_ref.nodes = {"1": nested_model}
    apply_draft_update_ops(model, ctx._ops)
    splitted_update_ops = get_splitted_update_model_ops("test", ctx._ops, "", store._field_store_meta)
    assert len(splitted_update_ops) == 2
    first_op = splitted_update_ops[0]
    assert isinstance(first_op, SplitNewDeleteOp)
    assert first_op.is_new == False 
    assert first_op.key == "test/nodes"
    assert first_op.is_remove_all_childs

    pprint(splitted_update_ops, expand_all=True)
    await store.update_model(draft, ctx._ops)
    # pprint(store_backend._data, expand_all=True)
    pprint(store_backend._data, expand_all=True)

    model_cur = await store.fetch_model()
    # rich.print(dataclasses.asdict(model_cur))
    # rich.print(dataclasses.asdict(model_ref))
    pprint(store_backend._data, expand_all=True)

    _assert_may_dataclass_equal(model_ref, model_cur)

    with capture_draft_update() as ctx:
        draft.nodes["1"].id = "X"
        model_ref.nodes["1"].id = "X"
    apply_draft_update_ops(model, ctx._ops)
    await store.update_model(draft, ctx._ops)
    model_cur = await store.fetch_model()
    _assert_may_dataclass_equal(model_ref, model_cur)

    nested_model3 = NestedNode2(id="2")
    with capture_draft_update() as ctx:
        draft.nodes["1"].nodes = {"2": nested_model3}
        model_ref.nodes["1"].nodes = {"2": nested_model3}

    apply_draft_update_ops(model, ctx._ops)
    await store.update_model(draft, ctx._ops)
    model_cur = await store.fetch_model()
    _assert_may_dataclass_equal(model_ref, model_cur)
    nested_model4 = NestedNode2(id="3")

    with capture_draft_update() as ctx:
        draft.nodes["1"].nodes = {"3": nested_model4}
        model_ref.nodes["1"].nodes = {"3": nested_model4}

    apply_draft_update_ops(model, ctx._ops)
    await store.update_model(draft, ctx._ops)
    model_cur = await store.fetch_model()
    _assert_may_dataclass_equal(model_ref, model_cur)

    pprint(store_backend._data, expand_all=True)

    with capture_draft_update() as ctx:
        draft.nodes["1"].nodes = None
        model_ref.nodes["1"].nodes = None
    splitted_update_ops = get_splitted_update_model_ops("test", ctx._ops, "", store._field_store_meta)
    apply_draft_update_ops(model, ctx._ops)
    await store.update_model(draft, ctx._ops)
    # pprint(splitted_update_ops, expand_all=True)
    model_cur = await store.fetch_model()
    _assert_may_dataclass_equal(model_ref, model_cur)
    pprint(store_backend._data, expand_all=True)

@dataclasses.dataclass
class NonNested:
    id: str


@dataclasses.dataclass
class NestedNode3:
    id: str
    attr: Optional[NestedNode2] = None
    attr2: Optional[NonNested] = None

@pytest.mark.asyncio
async def test_nested_model_store_set_attr():
    model = NestedNode3(id="root", attr=NestedNode2(id="root", nodes={"1": NestedNode2(id="1")}), attr2=NonNested(id="X"))
    model_ref = copy.deepcopy(model)
    draft = create_draft_type_only(type(model))

    store_backend = DraftFileStoreBackendInMemory()
    store = DraftFileStorage("test", model, store_backend)
    model = await store.fetch_model()
    pprint(store_backend._data, expand_all=True)

    with capture_draft_update() as ctx:
        draft.attr = None 
        draft.attr2 = None
        model_ref.attr = None
        model_ref.attr2 = None
    splitted_update_ops = get_splitted_update_model_ops("test", ctx._ops, "", store._field_store_meta)
    pprint(splitted_update_ops, expand_all=True)
    assert len(splitted_update_ops) == 2
    first_op = splitted_update_ops[0]
    assert isinstance(first_op, SplitNewDeleteOp)
    assert first_op.is_new == False 
    assert first_op.key == "test/attr"
    assert first_op.is_remove_all_childs

    apply_draft_update_ops(model, ctx._ops)
    await store.update_model(draft, ctx._ops)
    model_cur = await store.fetch_model()
    _assert_may_dataclass_equal(model_ref, model_cur)
    pprint(store_backend._data, expand_all=True)

@dataclasses.dataclass
class NodeWithScalarStore:
    id: str
    attr: Annotated[int, DraftStoreScalarMeta()]
    attr2: Annotated[int, DraftStoreScalarMeta()]

@pytest.mark.asyncio
async def test_scalar_meta():
    model = NodeWithScalarStore(id="root", attr=1, attr2=1)
    model_ref = copy.deepcopy(model)
    draft = create_draft_type_only(type(model))

    store_backend = DraftFileStoreBackendInMemory()
    store = DraftFileStorage("test", model, store_backend)
    model = await store.fetch_model()
    pprint(store_backend._data, expand_all=True)

    with capture_draft_update() as ctx:
        draft.attr = 3
        draft.attr2 += 5
        model_ref.attr = 3
        model_ref.attr2 += 5
    splitted_update_ops = get_splitted_update_model_ops("test", ctx._ops, "", store._field_store_meta)
    pprint(splitted_update_ops, expand_all=True)

    apply_draft_update_ops(model, ctx._ops)
    await store.update_model(draft, ctx._ops)
    model_cur = await store.fetch_model()
    _assert_may_dataclass_equal(model_ref, model_cur)
    pprint(store_backend._data, expand_all=True)

if __name__ == "__main__":

    asyncio.run(test_splitted_draft_store())

    test_splitted_draft(True)
    test_splitted_draft(False)

    asyncio.run(test_nested_model())
    asyncio.run(test_nested_model_store())
    asyncio.run(test_nested_model_store_set_attr())

    asyncio.run(test_scalar_meta())