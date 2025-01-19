"""Draft Proxy to record changes for dataclass.
inspired by [immer](https://www.npmjs.com/package/immer).

Only support standard scalar types and list/dict. don't support set, tuple, etc.

Supported update:

1. direct assignment

```Python
draft.a.b = 1
draft.arr[1] = 2
draft.dic['key'] = 3
draft.a.b += 4
```

2. List/Dict methods (except sort)

```Python
draft.arr.append(1)
draft.arr.extend([1, 2])
draft.arr.pop()
draft.arr.remove(1)
draft.arr.clear()
draft.arr.insert(1, 2)

draft.dic.pop('key')
draft.dic.clear()
```

"""


import contextlib
import contextvars
import enum
from typing import Any, Optional, Type, TypeVar, Union
import tensorpc.core.dataclass_dispatch as dataclasses
from collections.abc import Sequence, Mapping
import tensorpc.core.datamodel.jmes as jmespath

T = TypeVar("T")

class JMESPathOpType(enum.IntEnum):
    Set = 0
    Delete = 1
    Extend = 2
    Slice = 3
    ArraySet = 4
    ArrayPop = 5
    ArrayInsert = 6
    ArrayRemove = 7
    ContainerClear = 8
    DictUpdate = 10
    Assign = 11
    ScalarInplaceOp = 20

class ScalarInplaceOpType(enum.IntEnum):
    Add = 0
    Sub = 1
    Mul = 2
    Div = 3

# currently jmespath don't support ast to code, so we use a simple ast here.

class DraftASTType(enum.IntEnum):
    GET_ITEM = 0
    ARRAY_GET_ITEM = 1
    DICT_GET_ITEM = 2
    FUNC_CALL = 3
    NAME = 4

@dataclasses.dataclass
class DraftASTNode:
    type: DraftASTType
    children: list["DraftASTNode"]
    value: Any

    def get_jmes_path(self) -> str:
        if self.type == DraftASTType.NAME:
            return self.value if self.value != "" else "$"
        return _draft_ast_to_jmes_path_recursive(self)

_GET_ITEMS = set([DraftASTType.GET_ITEM, DraftASTType.ARRAY_GET_ITEM, DraftASTType.DICT_GET_ITEM])

def _draft_ast_to_jmes_path_recursive(node: DraftASTNode) -> str:
    if node.type == DraftASTType.NAME:
        return node.value
    elif node.type in _GET_ITEMS:
        child_value = _draft_ast_to_jmes_path_recursive(node.children[0])
        is_root = child_value == ""
        if node.type == DraftASTType.GET_ITEM:
            if is_root:
                return f"{node.value}"
            else:
                return f"{child_value}.{node.value}"
        elif node.type == DraftASTType.ARRAY_GET_ITEM:
            return f"{child_value}[{node.value}]"
        else:
            return f"{child_value}.\"{node.value}\""
    elif node.type == DraftASTType.FUNC_CALL:
        return f"{node.value}(" + ",".join([_draft_ast_to_jmes_path_recursive(child) for child in node.children]) + ")"
    else:
        raise NotImplementedError(f"node type {node.type} not implemented")
    
def _evaluate_draft_ast(node: DraftASTNode, obj: Any, root_obj: Any) -> Any:
    if node.type == DraftASTType.NAME:
        if node.value == "" or node.value == "$":
            return obj
        return getattr(obj, node.value)
    elif node.type == DraftASTType.GET_ITEM:
        return getattr(_evaluate_draft_ast(node.children[0], root_obj, root_obj), node.value)
    elif node.type == DraftASTType.ARRAY_GET_ITEM or node.type == DraftASTType.DICT_GET_ITEM:
        return _evaluate_draft_ast(node.children[0], root_obj, root_obj)[node.value]
    elif node.type == DraftASTType.FUNC_CALL:
        if node.value == "getitem":
            return _evaluate_draft_ast(node.children[0], root_obj, root_obj)[_evaluate_draft_ast(node.children[1], root_obj, root_obj)]
        elif node.value == "getattr":
            return getattr(_evaluate_draft_ast(node.children[0], root_obj, root_obj), _evaluate_draft_ast(node.children[1], root_obj, root_obj))
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError(f"node type {node.type} not implemented")

@dataclasses.dataclass
class JMESPathOp:
    path: str 
    op: JMESPathOpType
    opData: Any

    def to_dict(self):
        return {
            "path": self.path,
            "op": int(self.op),
            "opData": self.opData
        }


@dataclasses.dataclass
class JMESPathOpForBackend:
    op: JMESPathOpType
    opData: Any
    node: DraftASTNode
    userdata: Any = None
    # only used when evaluate object (not json object), jmes only support json.
    additionalNodes: list[DraftASTNode] = dataclasses.field(default_factory=list)

    def __repr__(self) -> str:
        path_str = self.node.get_jmes_path()
        # jpath_str = _get_jmes_path(self.path)
        return f"JOp[{path_str}|{self.op.name}]:{self.opData}"

    def to_jmes_path_op(self) -> JMESPathOp:
        return JMESPathOp(self.node.get_jmes_path(), self.op, self.opData)

    def get_userdata_typed(self, t: Type[T]) -> T:
        assert self.userdata is not None and isinstance(self.userdata, t), f"userdata is not {t}"
        return self.userdata


class DraftUpdateContext:
    def __init__(self):
        self._ops: list[JMESPathOpForBackend] = []

_DRAGT_UPDATE_CONTEXT: contextvars.ContextVar[Optional[DraftUpdateContext]] = contextvars.ContextVar("DraftUpdateContext", default=None)

@contextlib.contextmanager
def capture_draft_update():
    ctx = DraftUpdateContext()
    token = _DRAGT_UPDATE_CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _DRAGT_UPDATE_CONTEXT.reset(token)

def get_draft_update_context_noexcept() -> Optional[DraftUpdateContext]:
    return _DRAGT_UPDATE_CONTEXT.get()

def get_draft_update_context() -> DraftUpdateContext:
    ctx = get_draft_update_context_noexcept()
    assert ctx is not None, "This operation is only allowed in Draft context"
    return ctx

def _tensorpc_draft_dispatch(new_obj: Any, node: DraftASTNode, userdata: Any) -> "DraftBase":
    # TODO add annotation validate
    if dataclasses.is_dataclass(new_obj):
        return DraftObject(new_obj, userdata, node)
    elif isinstance(new_obj, Sequence) and not isinstance(new_obj, str):
        return DraftSequence(new_obj, userdata, node)
    elif isinstance(new_obj, Mapping):
        return DraftDict(new_obj, userdata, node)
    elif isinstance(new_obj, (int, float)):
        return DraftMutableScalar(new_obj, userdata, node)
    else:
        return DraftImmutableScalar(new_obj, userdata, node)


class DraftBase:
    __known_attrs__ = {"_tensorpc_draft_attr_real_obj", "_tensorpc_draft_attr_userdata", "_tensorpc_draft_attr_cur_node"}

    def __init__(self, obj: Any, userdata: Any = None, node: Optional[DraftASTNode] = None) -> None:
        self._tensorpc_draft_attr_real_obj = obj 
        self._tensorpc_draft_attr_userdata = userdata
        self._tensorpc_draft_attr_cur_node: DraftASTNode = node or DraftASTNode(DraftASTType.NAME, [], "")

    def __str__(self) -> str:
        return get_draft_jmespath(self)

    def _tensorpc_draft_get_jmes_op(self, op_type: JMESPathOpType, opdata: Any, drop_last: bool = False, addi_nodes: Optional[list[DraftASTNode]] = None) -> JMESPathOpForBackend:
        node = self._tensorpc_draft_attr_cur_node
        if drop_last:
            node = node.children[0]
        return JMESPathOpForBackend(op_type, opdata, node, self._tensorpc_draft_attr_userdata, addi_nodes if addi_nodes is not None else [])

    def _tensorpc_draft_dispatch(self, new_obj: Any, new_node: DraftASTNode) -> "DraftBase":
        return _tensorpc_draft_dispatch(new_obj, new_node, self._tensorpc_draft_attr_userdata)

class DraftObject(DraftBase):
    __known_attrs__ = {*DraftBase.__known_attrs__, "_tensorpc_draft_attr_obj_fields_dict"}
    def __init__(self, obj: Any, userdata: Any = None, node: Optional[DraftASTNode] = None) -> None:
        # TODO should we limit obj is a pydantic model to perform validate?
        super().__init__(obj, userdata, node)
        self._tensorpc_draft_attr_obj_fields_dict = {field.name: field for field in dataclasses.fields(self._tensorpc_draft_attr_real_obj)}
        assert dataclasses.is_dataclass(obj), f"DraftObject only support dataclass, got {type(obj)}"


    def __getattr__(self, name: str):
        if name not in self._tensorpc_draft_attr_obj_fields_dict:
            raise AttributeError(f"{type(self._tensorpc_draft_attr_real_obj)} has no attribute {name}")
        obj_child = getattr(self._tensorpc_draft_attr_real_obj, name)
        new_ast_node = DraftASTNode(DraftASTType.GET_ITEM, [self._tensorpc_draft_attr_cur_node], name)
        return self._tensorpc_draft_dispatch(obj_child, new_ast_node)

    def __setattr__(self, name: str, value: Any):
        if name in DraftObject.__known_attrs__:
            super().__setattr__(name, value)
            return 
        if name not in self._tensorpc_draft_attr_obj_fields_dict:
            raise AttributeError(f"{type(self._tensorpc_draft_attr_real_obj)} has no attribute {name}")
        ctx = get_draft_update_context()
        if isinstance(value, DraftBase):
            if ctx._ops and ctx._ops[-1].op == JMESPathOpType.ScalarInplaceOp:
                key = ctx._ops[-1].opData["key"]
                if name == key and ctx._ops[-1].node.get_jmes_path() == self._tensorpc_draft_attr_cur_node.get_jmes_path():
                    # inplace operation, do nothing
                    return
        assert not isinstance(value, DraftBase), "you can't assign a Draft object to another Draft object, assign real value instead."
        # TODO do validate here
        ctx._ops.append(self._tensorpc_draft_get_jmes_op(JMESPathOpType.Set, {
            "items": [(name, value)]
        }))

def _assert_not_draft(*value: Any):
    for v in value:
        assert not isinstance(v, DraftBase), "you can't change a Draft object to another Draft object, use real value instead."

class DraftSequence(DraftBase):
    def __getitem__(self, index: Union[DraftBase, int, slice | tuple[Union[int, slice]]]):
        if isinstance(index, DraftBase):
            ast_node = DraftASTNode(DraftASTType.FUNC_CALL, [self._tensorpc_draft_attr_cur_node, index._tensorpc_draft_attr_cur_node], "getitem")
            return self._tensorpc_draft_dispatch(self._tensorpc_draft_attr_real_obj[index._tensorpc_draft_attr_real_obj], ast_node)
        if isinstance(index, tuple):
            raise NotImplementedError("DraftSequence don't support N-D slice")
        if isinstance(index, slice):
            raise NotImplementedError("DraftSequence don't support slice")
        ast_node = DraftASTNode(DraftASTType.ARRAY_GET_ITEM, [self._tensorpc_draft_attr_cur_node], index)
        return self._tensorpc_draft_dispatch(self._tensorpc_draft_attr_real_obj[index], ast_node)

    def __setitem__(self, index: Union[int, DraftBase], value: Any):
        # TODO support list item inplace (`a[1] += 1`)
        ctx = get_draft_update_context()
        if isinstance(value, DraftBase):
            if ctx._ops and ctx._ops[-1].op == JMESPathOpType.ScalarInplaceOp:
                key = ctx._ops[-1].opData["key"]
                if index == key and ctx._ops[-1].node.get_jmes_path() == self._tensorpc_draft_attr_cur_node.get_jmes_path():
                    # inplace operation, do nothing
                    return
        _assert_not_draft(value)
        if isinstance(index, DraftBase):
            ctx._ops.append(self._tensorpc_draft_get_jmes_op(JMESPathOpType.Assign, {
                "keyPath": index._tensorpc_draft_attr_cur_node.get_jmes_path(),
                "value": value
            }, addi_nodes=[index._tensorpc_draft_attr_cur_node]))
            return 
        _assert_not_draft(index)
        ctx._ops.append(self._tensorpc_draft_get_jmes_op(JMESPathOpType.ArraySet, {
            "items": [(index, value)]
        }))

    def append(self, value: Any):
        _assert_not_draft(value)
        ctx = get_draft_update_context()
        ctx._ops.append(self._tensorpc_draft_get_jmes_op(JMESPathOpType.Extend, {
            "items": [value]
        }))

    def extend(self, value: list):
        _assert_not_draft(value)
        ctx = get_draft_update_context()
        ctx._ops.append(self._tensorpc_draft_get_jmes_op(JMESPathOpType.Extend, {
            "items": value
        }))

    def pop(self, index: Optional[int] = None):
        _assert_not_draft(index)
        ctx = get_draft_update_context()
        ctx._ops.append(self._tensorpc_draft_get_jmes_op(JMESPathOpType.ArrayPop, {
            "index": index
        }))

    def remove(self, item: Any):
        _assert_not_draft(item)
        ctx = get_draft_update_context()
        ctx._ops.append(self._tensorpc_draft_get_jmes_op(JMESPathOpType.ArrayRemove, {
            "item": item
        }))

    def clear(self):
        ctx = get_draft_update_context()
        ctx._ops.append(self._tensorpc_draft_get_jmes_op(JMESPathOpType.ContainerClear, {}))

    def insert(self, index: int, item: Any):
        _assert_not_draft(index, item)
        ctx = get_draft_update_context()
        ctx._ops.append(self._tensorpc_draft_get_jmes_op(JMESPathOpType.ArrayInsert, {
            "index": index,
            "item": item
        }))


class DraftDict(DraftBase):
    def validate_obj(self):
        assert isinstance(self._tensorpc_draft_attr_real_obj, Mapping), f"DraftDict only support Mapping, got {type(self._tensorpc_draft_attr_real_obj)}"

    def __getitem__(self, key: Union[str, DraftBase]):
        if isinstance(key, DraftBase):
            ast_node = DraftASTNode(DraftASTType.FUNC_CALL, [self._tensorpc_draft_attr_cur_node, key._tensorpc_draft_attr_cur_node], "getitem")
            return self._tensorpc_draft_dispatch(self._tensorpc_draft_attr_real_obj[key._tensorpc_draft_attr_real_obj], ast_node)
        ast_node = DraftASTNode(DraftASTType.DICT_GET_ITEM, [self._tensorpc_draft_attr_cur_node], key)
        return self._tensorpc_draft_dispatch(self._tensorpc_draft_attr_real_obj[key], ast_node)

    def __setitem__(self, key: Union[str, DraftBase], value: Any):
        ctx = get_draft_update_context()
        if isinstance(value, DraftBase):
            if ctx._ops and ctx._ops[-1].op == JMESPathOpType.ScalarInplaceOp:
                key = ctx._ops[-1].opData["key"]
                if key == key and ctx._ops[-1].node.get_jmes_path() == self._tensorpc_draft_attr_cur_node.get_jmes_path():
                    # inplace operation, do nothing
                    return
        _assert_not_draft(value)
        if isinstance(key, DraftBase):
            ctx._ops.append(self._tensorpc_draft_get_jmes_op(JMESPathOpType.Assign, {
                "keyPath": key._tensorpc_draft_attr_cur_node.get_jmes_path(),
                "value": value
            }, addi_nodes=[key._tensorpc_draft_attr_cur_node]))
            return 
        _assert_not_draft(key)
        ctx._ops.append(self._tensorpc_draft_get_jmes_op(JMESPathOpType.DictUpdate, {
            "items": {key: value}
        }))

    def pop(self, key: str):
        _assert_not_draft(key)
        ctx = get_draft_update_context()
        ctx._ops.append(self._tensorpc_draft_get_jmes_op(JMESPathOpType.Delete, {
            "keys": [key]
        }))

    def clear(self):
        ctx = get_draft_update_context()
        ctx._ops.append(self._tensorpc_draft_get_jmes_op(JMESPathOpType.ContainerClear, {}))

    def update(self, items: dict[str, Any]):
        _assert_not_draft(items)
        ctx = get_draft_update_context()
        ctx._ops.append(self._tensorpc_draft_get_jmes_op(JMESPathOpType.DictUpdate, {
            "items": items
        }))

class DraftImmutableScalar(DraftBase):
    """Leaf draft object, user can't do any operation on it."""
    pass

class DraftMutableScalar(DraftBase):
    def __iadd__(self, other: Union[int, float]):
        _assert_not_draft(other)
        ctx = get_draft_update_context()
        ctx._ops.append(self._tensorpc_draft_get_jmes_op(JMESPathOpType.ScalarInplaceOp, {
            "op": ScalarInplaceOpType.Add,
            "key": self._tensorpc_draft_attr_cur_node.value,
            "value": other
        }, drop_last=True))
        return self

    def __isub__(self, other: Union[int, float]):
        _assert_not_draft(other)
        ctx = get_draft_update_context()
        ctx._ops.append(self._tensorpc_draft_get_jmes_op(JMESPathOpType.ScalarInplaceOp, {
            "op": ScalarInplaceOpType.Sub,
            "key": self._tensorpc_draft_attr_cur_node.value,
            "value": other
        }, drop_last=True))
        return self

    def __imul__(self, other: Union[int, float]):
        _assert_not_draft(other)
        ctx = get_draft_update_context()
        ctx._ops.append(self._tensorpc_draft_get_jmes_op(JMESPathOpType.ScalarInplaceOp, {
            "op": ScalarInplaceOpType.Mul,
            "key": self._tensorpc_draft_attr_cur_node.value,
            "value": other
        }, drop_last=True))
        return self

    def __itruediv__(self, other: Union[int, float]):
        _assert_not_draft(other)
        ctx = get_draft_update_context()
        ctx._ops.append(self._tensorpc_draft_get_jmes_op(JMESPathOpType.ScalarInplaceOp, {
            "op": ScalarInplaceOpType.Div,
            "key": self._tensorpc_draft_attr_cur_node.value,
            "value": other
        }, drop_last=True))
        return self

    def __ifloordiv__(self, other: Union[int, float]):
        _assert_not_draft(other)
        ctx = get_draft_update_context()
        ctx._ops.append(self._tensorpc_draft_get_jmes_op(JMESPathOpType.ScalarInplaceOp, {
            "op": ScalarInplaceOpType.Div,
            "key": self._tensorpc_draft_attr_cur_node.value,
            "value": other
        }, drop_last=True))
        return self

def apply_draft_jmes_ops_backend(obj: Any, ops: list[JMESPathOpForBackend]):
    # we delay real operation on original object to make sure
    # all validation is performed before real operation
    for op in ops:
        # cur_obj = obj
        # for i in range(len(op.path)):
        #     path_item = op.path[i]
        #     if path_item.type == DraftASTType.GET_ITEM:
        #         assert isinstance(path_item.key, str)
        #         cur_obj = getattr(cur_obj, path_item.key)
        #     elif path_item.type == DraftASTType.ARRAY_GET_ITEM or path_item.type == DraftASTType.DICT_GET_ITEM:
        #         cur_obj = cur_obj[path_item.key]
        # print(op)
        cur_obj = _evaluate_draft_ast(op.node, obj, obj)
        # new cur_obj is target, apply op.
        if op.op == JMESPathOpType.Set:
            for k, v in op.opData["items"]:
                setattr(cur_obj, k, v)
        elif op.op == JMESPathOpType.Delete:
            for k in op.opData["keys"]:
                cur_obj.pop(k)
        elif op.op == JMESPathOpType.Extend:
            cur_obj.extend(op.opData["items"])
        elif op.op == JMESPathOpType.ArraySet:
            for idx, item in op.opData["items"]:
                cur_obj[idx] = item
        elif op.op == JMESPathOpType.ArrayPop:
            cur_obj.pop(op.opData.get("index", None))
        elif op.op == JMESPathOpType.ArrayInsert:
            cur_obj.insert(op.opData["index"], op.opData["item"])
        elif op.op == JMESPathOpType.ArrayRemove:
            cur_obj.remove(op.opData["item"])
        elif op.op == JMESPathOpType.ContainerClear:
            cur_obj.clear()
        elif op.op == JMESPathOpType.DictUpdate:
            for k, v in op.opData["items"].items():
                cur_obj[k] = v
        elif op.op == JMESPathOpType.Assign:
            key = _evaluate_draft_ast(op.additionalNodes[0], obj, obj)
            cur_obj[key] = op.opData["value"]
        elif op.op == JMESPathOpType.ScalarInplaceOp:
            key = op.opData["key"]
            value = op.opData["value"]
            if op.opData["op"] == ScalarInplaceOpType.Add:
                setattr(cur_obj, key, getattr(cur_obj, key) + value)
            elif op.opData["op"] == ScalarInplaceOpType.Sub:
                setattr(cur_obj, key, getattr(cur_obj, key) - value)
            elif op.opData["op"] == ScalarInplaceOpType.Mul:
                setattr(cur_obj, key, getattr(cur_obj, key) * value)
            elif op.opData["op"] == ScalarInplaceOpType.Div:
                setattr(cur_obj, key, getattr(cur_obj, key) / value)
        else:
            raise NotImplementedError(f"op {op.op} not implemented")

def apply_draft_jmes_ops(obj: dict, ops: list[JMESPathOp]):
    # we delay real operation on original object to make sure
    # all validation is performed before real operation
    for op in ops:
        cur_obj = jmespath.search(op.path, obj)
        # new cur_obj is target, apply op.
        if op.op == JMESPathOpType.Set:
            for k, v in op.opData["items"]:
                cur_obj[k] = v
        elif op.op == JMESPathOpType.Delete:
            for k in op.opData["keys"]:
                cur_obj.pop(k)
        elif op.op == JMESPathOpType.Extend:
            cur_obj.extend(op.opData["items"])
        elif op.op == JMESPathOpType.ArraySet:
            for idx, item in op.opData["items"]:
                cur_obj[idx] = item
        elif op.op == JMESPathOpType.ArrayPop:
            cur_obj.pop(op.opData.get("index", None))
        elif op.op == JMESPathOpType.ArrayInsert:
            cur_obj.insert(op.opData["index"], op.opData["item"])
        elif op.op == JMESPathOpType.ArrayRemove:
            cur_obj.remove(op.opData["item"])
        elif op.op == JMESPathOpType.ContainerClear:
            cur_obj.clear()
        elif op.op == JMESPathOpType.DictUpdate:
            for k, v in op.opData["items"].items():
                cur_obj[k] = v
        elif op.op == JMESPathOpType.Assign:
            key = jmespath.search(op.opData["keyPath"], obj)
            cur_obj[key] = op.opData["value"]
        elif op.op == JMESPathOpType.ScalarInplaceOp:
            key = op.opData["key"]
            value = op.opData["value"]
            if op.opData["op"] == ScalarInplaceOpType.Add:
                cur_obj[key] += value
            elif op.opData["op"] == ScalarInplaceOpType.Sub:
                cur_obj[key] -= value
            elif op.opData["op"] == ScalarInplaceOpType.Mul:
                cur_obj[key] *= value
            elif op.opData["op"] == ScalarInplaceOpType.Div:
                cur_obj[key] /= value
        else:
            raise NotImplementedError(f"op {op.op} not implemented")

def get_draft_jmespath(draft: DraftBase) -> str:
    return draft._tensorpc_draft_attr_cur_node.get_jmes_path()

def create_draft(obj: Any, userdata: Any = None):
    new_node = DraftASTNode(DraftASTType.NAME, [], "")
    return _tensorpc_draft_dispatch(obj, new_node, userdata)

def get_draft_ast_node(draft: Any) -> DraftASTNode:
    return draft._tensorpc_draft_attr_cur_node

def insert_assign_draft_op(draft: Any, value: Any):
    """used to insert a assign op to ctx without explicit assignment.
    Usually used when user only provide a draft object and want to assign a value to it.
    """
    _assert_not_draft(value)
    assert isinstance(draft, DraftBase), "draft should be a Draft object"
    ctx = get_draft_update_context()
    cur_node = draft._tensorpc_draft_attr_cur_node
    assert cur_node.type != DraftASTType.NAME and cur_node.type != DraftASTType.FUNC_CALL, "can't assign to root or getitem/getattr object"
    node_prev = cur_node.children[0]

    if cur_node.type == DraftASTType.GET_ITEM:
        ctx._ops.append(JMESPathOpForBackend(
            JMESPathOpType.Set, {
                "items": [(cur_node.value, value)]
            }, node_prev, 
            draft._tensorpc_draft_attr_userdata))
    elif cur_node.type == DraftASTType.ARRAY_GET_ITEM:
        ctx._ops.append(JMESPathOpForBackend(
            JMESPathOpType.ArraySet, {
                "items": [(cur_node.value, value)]
            }, node_prev, 
            draft._tensorpc_draft_attr_userdata))
    elif cur_node.type == DraftASTType.DICT_GET_ITEM:
        ctx._ops.append(JMESPathOpForBackend(
            JMESPathOpType.DictUpdate, {
                "items": {cur_node.value: value}
            }, node_prev, 
            draft._tensorpc_draft_attr_userdata))
    else:
        raise NotImplementedError(f"Draft type {type(draft)} not implemented")