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
    ScalarInplaceOp = 20

class ScalarInplaceOpType(enum.IntEnum):
    Add = 0
    Sub = 1
    Mul = 2
    Div = 3

class _PathType(enum.IntEnum):
    GET_ITEM = 0
    ARRAY_GET_ITEM = 1
    DICT_GET_ITEM = 2


@dataclasses.dataclass
class _PathItem:
    type: _PathType
    key: Union[int, str]


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
    path: list[_PathItem]
    op: JMESPathOpType
    opData: Any
    userdata: Any = None

    @staticmethod
    def get_path_list_str(path: list[_PathItem]):

        res = ["$"]
        for path_item in path:
            if path_item.type == _PathType.GET_ITEM:
                res.append(f".{path_item.key}")
            elif path_item.type == _PathType.ARRAY_GET_ITEM:
                res.append(f"[{path_item.key}]")
            elif path_item.type == _PathType.DICT_GET_ITEM:
                res.append(f".{path_item.key}")
        return "".join(res)

    def __repr__(self) -> str:
        path_str = self.get_path_list_str(self.path)
        # jpath_str = _get_jmes_path(self.path)
        return f"JOp[{path_str}|{self.op.name}]:{self.opData}"

    def to_jmes_path_op(self) -> JMESPathOp:
        return JMESPathOp(_get_jmes_path(self.path), self.op, self.opData)

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

def _get_jmes_path(path: list[_PathItem]) -> str:
    res: list[str] = []

    for i in range(len(path)):
        path_item = path[i]
        if path_item.type == _PathType.GET_ITEM:
            if i != 0:
                res.append(".")
            res.append(str(path_item.key))
        elif path_item.type == _PathType.ARRAY_GET_ITEM:
            res.append(f"[{path_item.key}]")
        elif path_item.type == _PathType.DICT_GET_ITEM:
            if i != 0:
                res.append(".")
            res.append(f"\"{path_item.key}\"")
    if not res:
        # $ means root
        return "$"
    return "".join(res)


def _tensorpc_draft_dispatch(new_obj: Any, path: list[_PathItem], userdata: Any) -> "DraftBase":
    # TODO add annotation validate
    if dataclasses.is_dataclass(new_obj):
        return DraftObject(new_obj, path, userdata)
    elif isinstance(new_obj, Sequence) and not isinstance(new_obj, str):
        return DraftSequence(new_obj, path, userdata)
    elif isinstance(new_obj, Mapping):
        return DraftDict(new_obj, path, userdata)
    elif isinstance(new_obj, (int, float)):
        return DraftMutableScalar(new_obj, path, userdata)
    else:
        return DraftImmutableScalar(new_obj, path, userdata)


class DraftBase:
    __known_attrs__ = {"_tensorpc_draft_attr_real_obj", "_tensorpc_draft_attr_cur_path", "_tensorpc_draft_attr_userdata"}

    def __init__(self, obj: Any, path: Optional[list[_PathItem]] = None, userdata: Any = None) -> None:
        self._tensorpc_draft_attr_real_obj = obj 
        self._tensorpc_draft_attr_userdata = userdata
        self._tensorpc_draft_attr_cur_path: list[_PathItem] = []
        if path is not None:
            self._tensorpc_draft_attr_cur_path = path

    def __str__(self) -> str:
        return get_draft_jmespath(self)

    def _tensorpc_draft_get_jmes_op(self, op_type: JMESPathOpType, opdata: Any, drop_last: bool = False) -> JMESPathOpForBackend:
        if drop_last:
            jpath = self._tensorpc_draft_attr_cur_path[:-1]
        else:
            jpath = self._tensorpc_draft_attr_cur_path
        return JMESPathOpForBackend(jpath, op_type, opdata, self._tensorpc_draft_attr_userdata)

    def _tensorpc_draft_dispatch(self, new_obj: Any, new_path_item: _PathItem) -> "DraftBase":
        return _tensorpc_draft_dispatch(new_obj, self._tensorpc_draft_attr_cur_path + [new_path_item], self._tensorpc_draft_attr_userdata)

class DraftObject(DraftBase):
    __known_attrs__ = {*DraftBase.__known_attrs__, "_tensorpc_draft_attr_obj_fields_dict"}
    def __init__(self, obj: Any, path: Optional[list[_PathItem]] = None, userdata: Any = None) -> None:
        # TODO should we limit obj is a pydantic model to perform validate?
        super().__init__(obj, path, userdata)
        self._tensorpc_draft_attr_obj_fields_dict = {field.name: field for field in dataclasses.fields(self._tensorpc_draft_attr_real_obj)}
        assert dataclasses.is_dataclass(obj), f"DraftObject only support dataclass, got {type(obj)}"


    def __getattr__(self, name: str):
        if name not in self._tensorpc_draft_attr_obj_fields_dict:
            raise AttributeError(f"{type(self._tensorpc_draft_attr_real_obj)} has no attribute {name}")
        obj_child = getattr(self._tensorpc_draft_attr_real_obj, name)
        return self._tensorpc_draft_dispatch(obj_child, _PathItem(_PathType.GET_ITEM, name))

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
                if name == key and ctx._ops[-1].path == self._tensorpc_draft_attr_cur_path:
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
    def __getitem__(self, index: Union[int, slice | tuple[Union[int, slice]]]):
        if isinstance(index, tuple):
            raise NotImplementedError("DraftSequence don't support N-D slice")
        if isinstance(index, slice):
            raise NotImplementedError("DraftSequence don't support slice")
        return self._tensorpc_draft_dispatch(self._tensorpc_draft_attr_real_obj[index], _PathItem(_PathType.ARRAY_GET_ITEM, index))

    def __setitem__(self, index: int, value: Any):
        # TODO support list item inplace (`a[1] += 1`)
        ctx = get_draft_update_context()
        if isinstance(value, DraftBase):
            if ctx._ops and ctx._ops[-1].op == JMESPathOpType.ScalarInplaceOp:
                key = ctx._ops[-1].opData["key"]
                if index == key and ctx._ops[-1].path == self._tensorpc_draft_attr_cur_path:
                    # inplace operation, do nothing
                    return
        _assert_not_draft(index, value)
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

    def __getitem__(self, key: str):
        _assert_not_draft(key)
        return self._tensorpc_draft_dispatch(self._tensorpc_draft_attr_real_obj[key], _PathItem(_PathType.DICT_GET_ITEM, key))

    def __setitem__(self, key: str, value: Any):
        _assert_not_draft(key)
        ctx = get_draft_update_context()
        if isinstance(value, DraftBase):
            if ctx._ops and ctx._ops[-1].op == JMESPathOpType.ScalarInplaceOp:
                key = ctx._ops[-1].opData["key"]
                if key == key and ctx._ops[-1].path == self._tensorpc_draft_attr_cur_path:
                    # inplace operation, do nothing
                    return
        assert not isinstance(value, DraftBase), "you can't assign a Draft object to another Draft object, assign real value instead."
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
            "key": self._tensorpc_draft_attr_cur_path[-1].key,
            "value": other
        }, drop_last=True))
        return self

    def __isub__(self, other: Union[int, float]):
        _assert_not_draft(other)
        ctx = get_draft_update_context()
        ctx._ops.append(self._tensorpc_draft_get_jmes_op(JMESPathOpType.ScalarInplaceOp, {
            "op": ScalarInplaceOpType.Sub,
            "key": self._tensorpc_draft_attr_cur_path[-1].key,
            "value": other
        }, drop_last=True))
        return self

    def __imul__(self, other: Union[int, float]):
        _assert_not_draft(other)
        ctx = get_draft_update_context()
        ctx._ops.append(self._tensorpc_draft_get_jmes_op(JMESPathOpType.ScalarInplaceOp, {
            "op": ScalarInplaceOpType.Mul,
            "key": self._tensorpc_draft_attr_cur_path[-1].key,
            "value": other
        }, drop_last=True))
        return self

    def __itruediv__(self, other: Union[int, float]):
        _assert_not_draft(other)
        ctx = get_draft_update_context()
        ctx._ops.append(self._tensorpc_draft_get_jmes_op(JMESPathOpType.ScalarInplaceOp, {
            "op": ScalarInplaceOpType.Div,
            "key": self._tensorpc_draft_attr_cur_path[-1].key,
            "value": other
        }, drop_last=True))
        return self

    def __ifloordiv__(self, other: Union[int, float]):
        _assert_not_draft(other)
        ctx = get_draft_update_context()
        ctx._ops.append(self._tensorpc_draft_get_jmes_op(JMESPathOpType.ScalarInplaceOp, {
            "op": ScalarInplaceOpType.Div,
            "key": self._tensorpc_draft_attr_cur_path[-1].key,
            "value": other
        }, drop_last=True))
        return self

def apply_draft_jmes_ops_backend(obj: Any, ops: list[JMESPathOpForBackend]):
    # we delay real operation on original object to make sure
    # all validation is performed before real operation
    for op in ops:
        cur_obj = obj
        for i in range(len(op.path)):
            path_item = op.path[i]
            if path_item.type == _PathType.GET_ITEM:
                assert isinstance(path_item.key, str)
                cur_obj = getattr(cur_obj, path_item.key)
            elif path_item.type == _PathType.ARRAY_GET_ITEM or path_item.type == _PathType.DICT_GET_ITEM:
                cur_obj = cur_obj[path_item.key]
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
    return _get_jmes_path(draft._tensorpc_draft_attr_cur_path)

def create_draft(obj: Any, userdata: Any = None):
    return _tensorpc_draft_dispatch(obj, [], userdata)