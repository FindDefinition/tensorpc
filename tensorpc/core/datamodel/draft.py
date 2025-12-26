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


* Main Path: for a draft ast expr, all nodes that can be assigned constructs a main path.

e.g. when you use a dynamic `getItem`, the target of `getItem` is a main path node, the key isn't a main path node

Our draft change detection only check main path nodes, other node will be treated as constant in a draft expr.

"""

import contextlib
import contextvars
import copy
from dataclasses import is_dataclass
import enum
import json
import traceback
import types
from typing import Any, Callable, MutableSequence, Optional, Type, TypeVar, Union, cast, get_type_hints
from typing_extensions import Literal, Self
from tensorpc.core import inspecttools
from tensorpc.core.annolib import AnnotatedType, Undefined, parse_type_may_optional_undefined, resolve_type_hints
from tensorpc.core.core_io import JsonSpecialData
from tensorpc.core.datamodel.asdict import as_dict_no_undefined
import tensorpc.core.dataclass_dispatch as dataclasses
from collections.abc import MutableMapping, Sequence, Mapping
import tensorpc.core.datamodel.jmes as jmespath
from .draftast import DraftASTFuncType, DraftASTNode, evaluate_draft_ast, evaluate_draft_ast_json, evaluate_draft_ast_with_obj_id_trace, evaluate_draft_ast_noexcept, DraftASTType
from tensorpc.core.pfl import pflpath

T = TypeVar("T")


class JMESPathOpType(enum.IntEnum):
    # only used for backend, frontend don't support this.
    RootAssign = -1
    RootInplaceOp = -2

    SetAttr = 0
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

_DYNAMIC_FUNC_TYPES = {DraftASTFuncType.GET_ATTR.value, DraftASTFuncType.GET_ITEM_PATH.value, DraftASTFuncType.WHERE.value}

def _scalar_inplace_op_to_str(op: ScalarInplaceOpType) -> str:
    if op == ScalarInplaceOpType.Add:
        return "+="
    if op == ScalarInplaceOpType.Sub:
        return "-="
    if op == ScalarInplaceOpType.Mul:
        return "*="
    if op == ScalarInplaceOpType.Div:
        return "/="
    raise ValueError(f"Unknown ScalarInplaceOpType {op}")

@dataclasses.dataclass
class TypeMeta:
    type: AnnotatedType
    has_undefined_or_optional: bool


@dataclasses.dataclass
class JMESPathOp:
    path: str
    op: JMESPathOpType
    opData: Any

    def to_dict(self):
        return {"path": self.path, "op": int(self.op), "opData": self.opData}

@dataclasses.dataclass
class PFLPathOp:
    path: str
    op: JMESPathOpType
    opData: Any

    def to_dict(self):
        return {"path": self.path, "op": int(self.op), "opData": self.opData}

@dataclasses.dataclass(kw_only=True)
class DraftFieldMeta:
    # external field won't be sent to frontend or store.
    is_external: bool = False
    # external field won't be sent to store.
    is_store_external: bool = False

@dataclasses.dataclass
class DraftUpdateOp:
    op: JMESPathOpType
    opData: Any
    node: DraftASTNode
    userdata: Any = None
    # only used when evaluate object (not json object), jmes only support json.
    additionalNodes: list[DraftASTNode] = dataclasses.field(
        default_factory=list)
    # when user use type-only draft and have `Annotated`, this field will contains metadata of `Annotated`
    # user can control update behavior by use custom metadata in `Annotated`.
    anno_type_metas: Optional[tuple[Any, ...]] = None
    # provide path-like meta trace
    anno_type_metas_trace: Optional[list[tuple[Any, ...]]] = None

    # when you setattr on draft object, this store the field id of that field.
    field_id: Optional[int] = None
    # when assign/modify target is external, this field will be True.
    is_external: bool = False
    # external field won't be sent to store.
    is_store_external: bool = False

    def __repr__(self) -> str:
        path_str = self.node.get_jmes_path()
        prefix = f"JOp[{path_str}|{self.op.name}|{self.is_external}|{self.is_store_external}]"
        # jpath_str = _get_jmes_path(self.path)
        if self.op == JMESPathOpType.SetAttr:
            key, value = self.opData["items"][0]
            return f"{prefix}:{key}={value}"
        elif self.op == JMESPathOpType.RootInplaceOp or self.op == JMESPathOpType.ScalarInplaceOp:
            op = self.opData["op"]
            value = self.opData["value"]
            if self.op == JMESPathOpType.RootInplaceOp:
                key = "$"
            else:
                key = self.opData["key"]
            return f"{prefix}:{key}{_scalar_inplace_op_to_str(op)}{value}"
        return f"{prefix}:{self.opData}"

    def to_jmes_path_op(self) -> JMESPathOp:
        # app internal will handle non-dict data in self.opData.
        return JMESPathOp(self.node.get_jmes_path(), self.op, self.opData)

    def to_pfl_path_op(self) -> PFLPathOp:
        # app internal will handle non-dict data in self.opData.
        return PFLPathOp(self.node.get_pfl_path(), self.op, self.opData)
    
    def to_pfl_frontend_path_op(self) -> PFLPathOp:
        # app internal will handle non-dict data in self.opData.
        op_data = self.opData
        if "keyPath" in self.opData:
            op_data = op_data.copy()
            op_data["keyPath"] = pflpath.compile_pflpath_to_compact_str(self.opData["keyPath"])
        return PFLPathOp(pflpath.compile_pflpath_to_compact_str(self.node.get_pfl_path()), self.op, op_data)

    def to_userdata_removed(self) -> "DraftUpdateOp":
        return dataclasses.replace(self, userdata=None, node=self.node.to_userdata_removed())

    def get_userdata_typed(self, t: Type[T]) -> Optional[T]:
        if self.userdata is not None and isinstance(
            self.userdata, t):
            return self.userdata
        return None

    def to_json_update_op(self):
        # assume your data is strong-typed dataclass, then we can store
        # data as dict in database and restore it to dataclass via pydantic or mashumaro.
        return dataclasses.replace(self,
                                   opData=as_dict_no_undefined(self.opData),
                                   field_id=None)

    def has_dynamic_node_in_main_path(self):
        # has `getattr` or `getItemPath` or `where` func call node
        for node in self.node.get_child_nodes_in_main_path():
            if node.type == DraftASTType.FUNC_CALL:
                if node.value in _DYNAMIC_FUNC_TYPES:
                    return True
        return False 

    def freeze_assign_data(self, is_json_only: bool = False) -> Self:
        if self.op == JMESPathOpType.SetAttr or self.op == JMESPathOpType.ArraySet:
            # freeze the assign data
            new_items = [(k, JsonSpecialData.from_option(v, is_json_only, True)) for k, v in self.opData["items"]]
            new_opdata = self.opData.copy()
            new_opdata["items"] = new_items
            res = dataclasses.replace(self, opData=new_opdata)
        elif self.op == JMESPathOpType.DictUpdate:
            # freeze the assign data
            new_items = {k: JsonSpecialData.from_option(v, is_json_only, True) for k, v in self.opData["items"]}
            new_opdata = self.opData.copy()
            new_opdata["items"] = new_items
            res = dataclasses.replace(self, opData=new_opdata)
        else:
            res = dataclasses.replace(self)
        return res 

    def to_data_deepcopied(self) -> Self:
        # we may need to deepcopy opData if we do backend update
        # before send to frontend, to avoid backend update modify opData.
        return dataclasses.replace(self,
                                   opData=copy.deepcopy(self.opData))

class DraftUpdateProcessContext:
    def __init__(self, proc: Callable[[DraftUpdateOp], DraftUpdateOp]):
        self._op_process: Callable[[DraftUpdateOp], DraftUpdateOp] = proc


_DRAGT_UPDATE_PROC_CONTEXT: contextvars.ContextVar[
    Optional[DraftUpdateProcessContext]] = contextvars.ContextVar(
        "DraftUpdateProcessContext", default=None)

class DraftUpdateContext:

    def __init__(self, prevent_inner_draft: bool = False, use_jmes_path: bool = True):
        self._ops: list[DraftUpdateOp] = []
        self._prevent_inner_draft = prevent_inner_draft
        self._use_jmes_path = use_jmes_path

    def add_op(self, op: DraftUpdateOp):
        assert not self._prevent_inner_draft, "Draft operation is disabled by a prevent_draft_update context, usually exists in draft event handler."
        update_proc_ctx = _DRAGT_UPDATE_PROC_CONTEXT.get()
        if update_proc_ctx is not None:
            op = update_proc_ctx._op_process(op)
        self._ops.append(op)


_DRAGT_UPDATE_CONTEXT: contextvars.ContextVar[
    Optional[DraftUpdateContext]] = contextvars.ContextVar(
        "DraftUpdateContext", default=None)

@contextlib.contextmanager
def prevent_draft_update():
    ctx = DraftUpdateContext(prevent_inner_draft=True)
    token = _DRAGT_UPDATE_CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _DRAGT_UPDATE_CONTEXT.reset(token)

@contextlib.contextmanager
def capture_draft_update(allow_nested: bool = True, use_jmes_path: bool = True):
    cur_ctx = _DRAGT_UPDATE_CONTEXT.get()
    if cur_ctx is not None and not allow_nested:
        raise RuntimeError("Nested DraftUpdateContext is not allowed")
    ctx = DraftUpdateContext(use_jmes_path=use_jmes_path)
    token = _DRAGT_UPDATE_CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _DRAGT_UPDATE_CONTEXT.reset(token)

@contextlib.contextmanager
def enter_op_process_ctx(proc: Callable[[DraftUpdateOp], DraftUpdateOp]):
    ctx = DraftUpdateProcessContext(proc)
    token = _DRAGT_UPDATE_PROC_CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _DRAGT_UPDATE_PROC_CONTEXT.reset(token)

def get_draft_update_context_noexcept() -> Optional[DraftUpdateContext]:
    return _DRAGT_UPDATE_CONTEXT.get()


def get_draft_update_context() -> DraftUpdateContext:
    ctx = get_draft_update_context_noexcept()
    assert ctx is not None, "This operation is only allowed in Draft context"
    return ctx


@dataclasses.dataclass
class _DraftAnnoState:
    is_type_only: bool
    anno_type: Optional[AnnotatedType] = None
    path_metas: list[tuple[Any, ...]] = dataclasses.field(default_factory=list)
    can_assign: bool = True
    # prevent assign by `insert_draft_assign_op` if False.
    can_direct_assign: bool = True
    is_external: bool = False
    is_store_external: bool = False


def _tensorpc_draft_dispatch(
        new_obj: Any,
        node: DraftASTNode,
        userdata: Any,
        prev_anno_state: _DraftAnnoState,
        anno_type: Optional[AnnotatedType] = None,
        can_assign: bool = True) -> "DraftBase":
    # TODO add annotation validate
    path_metas = prev_anno_state.path_metas.copy()
    if anno_type is not None and anno_type.annometa is not None:
        path_metas = path_metas + [anno_type.annometa]
    new_anno_state = dataclasses.replace(prev_anno_state,
                                         anno_type=anno_type,
                                         path_metas=path_metas)
    if not prev_anno_state.can_assign:
        new_anno_state.can_assign = False
    else:
        new_anno_state.can_assign = can_assign
    if dataclasses.is_dataclass(new_obj):
        return DraftObject(new_obj, userdata, node, new_anno_state)
    elif isinstance(new_obj, Sequence) and not isinstance(new_obj, str):
        return DraftSequence(new_obj, userdata, node, new_anno_state)
    elif isinstance(new_obj, Mapping):
        return DraftDict(new_obj, userdata, node, new_anno_state)
    elif isinstance(new_obj, (int, float)):
        return DraftMutableScalar(new_obj, userdata, node, new_anno_state)
    elif isinstance(new_obj, str):
        return DraftImmutableString(new_obj, userdata, node, new_anno_state)
    else:
        return DraftImmutableScalar(new_obj, userdata, node, new_anno_state)

def _extract_field_meta(anno_type: AnnotatedType, anno_state: _DraftAnnoState):
    is_external = anno_state.is_external
    if not is_external and anno_type.annometa is not None:
        for annmeta in anno_type.annometa:
            if isinstance(annmeta, DraftFieldMeta):
                is_external = annmeta.is_external
                break
    is_store_external = anno_state.is_store_external
    if not is_store_external and anno_type.annometa is not None:
        for annmeta in anno_type.annometa:
            if isinstance(annmeta, DraftFieldMeta):
                is_store_external = annmeta.is_store_external
                break
    return is_external, is_store_external

def _tensorpc_draft_anno_dispatch(
        anno_type: AnnotatedType, node: DraftASTNode, userdata: Any,
        prev_anno_state: _DraftAnnoState,
        can_assign: bool = True) -> "DraftBase":
    """For anno dispatch, we only support List, Dict and primitive scalar types.
    """
    new_obj = None
    path_metas = prev_anno_state.path_metas.copy()
    if anno_type is not None and anno_type.annometa is not None:
        path_metas = path_metas + [anno_type.annometa]
    is_external, is_store_external = _extract_field_meta(anno_type, prev_anno_state)

    new_anno_state = dataclasses.replace(prev_anno_state,
                                         anno_type=anno_type,
                                         path_metas=path_metas,
                                         is_external=is_external,
                                         is_store_external=is_store_external)
    if not prev_anno_state.can_assign:
        new_anno_state.can_assign = False
    else:
        new_anno_state.can_assign = can_assign
    if anno_type.annometa is not None:
        path_metas = path_metas + [anno_type.annometa]
    if dataclasses.is_dataclass(anno_type.origin_type):
        return DraftObject(None, userdata, node, new_anno_state)
    elif anno_type.is_sequence_type():
        return DraftSequence(new_obj, userdata, node, new_anno_state)
    elif anno_type.is_mapping_type():
        return DraftDict(new_obj, userdata, node, new_anno_state)
    elif anno_type.is_number_type():
        # bool is subclass of int
        return DraftMutableScalar(new_obj, userdata, node, new_anno_state)
    elif anno_type.is_union_type():
        raise NotImplementedError("Union type not supported for now, it reqiures pydantic dataclass with tagged union and discriminator.")
        return DraftUnion(new_obj, userdata, node, new_anno_state)
    elif anno_type.is_any_type():
        return DraftAny(new_obj, userdata, node, new_anno_state)
    elif anno_type.origin_type is str:
        return DraftImmutableString(new_obj, userdata, node, new_anno_state)
    else:
        return DraftImmutableScalar(new_obj, userdata, node, new_anno_state)


class _DraftNotValid:
    """A placeholder object that indicate this value is not valid which is different with original value,
    Usually used for `pop` operation.
    """
    pass


class DraftBase:
    __known_attrs__ = {
        "_tensorpc_draft_attr_real_obj",
        "_tensorpc_draft_attr_userdata",
        "_tensorpc_draft_attr_cur_node",
        "_tensorpc_draft_attr_anno_state",
        "_tensorpc_draft_dispatch",
        "_tensorpc_draft_logic_op",
        "_tensorpc_draft_binary_op",
    }

    def __init__(self,
                 obj: Any,
                 userdata: Any = None,
                 node: Optional[DraftASTNode] = None,
                 anno_state: Optional[_DraftAnnoState] = None) -> None:
        if anno_state is None:
            anno_state = _DraftAnnoState(False, None)
        anno_type = anno_state.anno_type
        type_only = anno_state.is_type_only
        path_metas = anno_state.path_metas
        assert obj is not None or anno_type is not None, "obj or anno_type should be provided"
        if type_only:
            assert anno_type is not None, "anno_type must be provided in type-only mode"
        self._tensorpc_draft_attr_real_obj = obj
        self._tensorpc_draft_attr_userdata = userdata
        self._tensorpc_draft_attr_cur_node: DraftASTNode = node or DraftASTNode(
            DraftASTType.NAME, [], "")
        self._tensorpc_draft_attr_anno_state = anno_state

    def __str__(self) -> str:
        return get_draft_pflpath(self)

    def _tensorpc_draft_get_update_op(
            self,
            op_type: JMESPathOpType,
            opdata: Any,
            drop_last: bool = False,
            addi_nodes: Optional[list[DraftASTNode]] = None,
            field_id: Optional[int] = None,
            field_anno_type: Optional[AnnotatedType] = None) -> DraftUpdateOp:
        node = self._tensorpc_draft_attr_cur_node
        if drop_last:
            node = node.children[0]
        annometa = None
        if self._tensorpc_draft_attr_anno_state.anno_type is not None:
            annometa = self._tensorpc_draft_attr_anno_state.anno_type.annometa
        is_external = self._tensorpc_draft_attr_anno_state.is_external
        is_store_external = self._tensorpc_draft_attr_anno_state.is_store_external
        if field_anno_type is not None:
            annometa = field_anno_type.annometa
            is_external, is_store_external = _extract_field_meta(field_anno_type, self._tensorpc_draft_attr_anno_state)
        return DraftUpdateOp(op_type, opdata, node,
                             self._tensorpc_draft_attr_userdata,
                             addi_nodes if addi_nodes is not None else [],
                             annometa, field_id=field_id,
                             is_external=is_external,
                             is_store_external=is_store_external)

    def _tensorpc_draft_dispatch(
            self,
            new_obj: Any,
            new_node: DraftASTNode,
            anno_type: Optional[AnnotatedType] = None) -> "DraftBase":
        if self._tensorpc_draft_attr_anno_state.is_type_only:
            assert anno_type is not None
            res = _tensorpc_draft_anno_dispatch(
                anno_type, new_node, self._tensorpc_draft_attr_userdata,
                self._tensorpc_draft_attr_anno_state)
            return res
        return _tensorpc_draft_dispatch(new_obj, new_node,
                                        self._tensorpc_draft_attr_userdata,
                                        self._tensorpc_draft_attr_anno_state,
                                        anno_type)

    def _tensorpc_draft_logic_op(self, other: Any, op: Literal["&&", "||"]):
        assert isinstance(other, DraftBase)
        new_node = other._tensorpc_draft_attr_cur_node
        this_node = self._tensorpc_draft_attr_cur_node
        ast_node = DraftASTNode(DraftASTType.BINARY_OP,
                                [this_node, new_node], op)
        if self._tensorpc_draft_attr_anno_state.is_type_only:
            return self._tensorpc_draft_dispatch(None, ast_node, AnnotatedType(bool, []))
        return self._tensorpc_draft_dispatch(
            self._tensorpc_draft_attr_real_obj, ast_node, AnnotatedType(bool, []))

    def _tensorpc_draft_binary_op(self, other: Any, op: Literal["==", "!=", ">", "<", ">=", "<=", "+", "-", "*", "/", "//"]):
        binary_ops = set(["==", "!=", ">", "<", ">=", "<="])
        math_ops = set(["+", "-", "*", "/", "//"])
        if op in math_ops:
            assert isinstance(self, DraftMutableScalar) and isinstance(
                other, (DraftMutableScalar, int, float)), "only support arithmetic op for DraftMutableScalar or number"
        
            self_type = self._tensorpc_draft_attr_anno_state.anno_type
            assert self_type is not None
            other_is_float = False
            if isinstance(other, DraftMutableScalar):
                other_annotype = other._tensorpc_draft_attr_anno_state.anno_type
                assert other_annotype is not None
                other_is_float = issubclass(other_annotype.origin_type, float)
                other_node = other._tensorpc_draft_attr_cur_node
            else:
                other_node = DraftASTNode(DraftASTType.JSON_LITERAL, [], other)

            self_is_float = issubclass(self_type.origin_type, float)
            if other_is_float or self_is_float:
                res_annotype = AnnotatedType(float, [])
            else:
                res_annotype = AnnotatedType(int, [])
            this_node = self._tensorpc_draft_attr_cur_node
            ast_node = DraftASTNode(DraftASTType.BINARY_OP,
                                    [this_node, other_node], op)
            if self._tensorpc_draft_attr_anno_state.is_type_only:
                return self._tensorpc_draft_dispatch(None, ast_node, res_annotype)
            return self._tensorpc_draft_dispatch(
                self._tensorpc_draft_attr_real_obj, ast_node, res_annotype)

        elif op in binary_ops:
            if not isinstance(other, DraftComparableScalar):
                if isinstance(other, str):
                    ast_type = DraftASTType.STRING_LITERAL
                else:
                    # obj must be json serializable
                    json.dumps(other)
                    ast_type = DraftASTType.JSON_LITERAL
                new_node = DraftASTNode(ast_type, [], other)
            else:
                new_node = other._tensorpc_draft_attr_cur_node
            this_node = self._tensorpc_draft_attr_cur_node
            ast_node = DraftASTNode(DraftASTType.BINARY_OP,
                                    [this_node, new_node], op)
            
            if self._tensorpc_draft_attr_anno_state.is_type_only:
                return self._tensorpc_draft_dispatch(None, ast_node, AnnotatedType(bool, []))
            return self._tensorpc_draft_dispatch(
                self._tensorpc_draft_attr_real_obj, ast_node, AnnotatedType(bool, []))
        else:
            raise NotImplementedError

    def __eq__(self, other: Any): # type: ignore
        assert other is None, "only allow compare with None for all draft object"
        return self._tensorpc_draft_binary_op(other, "==")
    
    def __ne__(self, other: Any): # type: ignore
        assert other is None, "only allow compare with None for all draft object"
        return self._tensorpc_draft_binary_op(other, "!=")

class DraftObject(DraftBase):
    __known_attrs__ = {
        *DraftBase.__known_attrs__, "_tensorpc_draft_attr_obj_fields_dict"
    }

    def __init__(self,
                 obj: Any,
                 userdata: Any = None,
                 node: Optional[DraftASTNode] = None,
                 anno_state: Optional[_DraftAnnoState] = None) -> None:
        # TODO should we limit obj is a pydantic model to perform validate?
        super().__init__(obj, userdata, node, anno_state)

        if self._tensorpc_draft_attr_anno_state.is_type_only:
            anno_type = self._tensorpc_draft_attr_anno_state.anno_type
            assert anno_type is not None
            # in anno mode, we don't modify or parse obj.
            assert anno_type.is_dataclass_type()
            fields = dataclasses.fields(anno_type.origin_type)
            if anno_type.child_types:
                # generic dataclass
                type_hints = resolve_type_hints(anno_type.origin_type[tuple(anno_type.child_types)])
            else:
                type_hints = resolve_type_hints(anno_type.origin_type)

        else:
            assert dataclasses.is_dataclass(
                obj), f"DraftObject only support dataclass, got {type(obj)}"
            fields = dataclasses.fields(self._tensorpc_draft_attr_real_obj)
            type_hints = resolve_type_hints(type(
                self._tensorpc_draft_attr_real_obj))

        self._tensorpc_draft_attr_obj_fields_dict = {
            field.name:
            (field, parse_type_may_optional_undefined(type_hints[field.name]))
            for field in fields
        }

    def __getattr__(self, name: str):
        if name not in self._tensorpc_draft_attr_obj_fields_dict:
            if self._tensorpc_draft_attr_anno_state.anno_type is not None:
                # only support get bound method through anno type
                dcls_type = self._tensorpc_draft_attr_anno_state.anno_type.origin_type
                if hasattr(dcls_type, name):
                    unbound_func = getattr(dcls_type, name)
                    if inspecttools.isstaticmethod(dcls_type, name):
                        return unbound_func
                    return types.MethodType(unbound_func, self)
                else:
                    raw_type = self._tensorpc_draft_attr_anno_state.anno_type.raw_type
                    raise AttributeError(
                        f"field `{name}` doesn't exist in {self}({raw_type})."
                    )
            if self._tensorpc_draft_attr_real_obj is None:
                raise AttributeError(
                    f"want to get {self}.{name}, but {self} don't have annotated type"
                )
            else:
                raise AttributeError(
                    f"dataclass {type(self._tensorpc_draft_attr_real_obj)} has no attribute {name}"
                )
        anno_type = None
        field_id = None
        if self._tensorpc_draft_attr_anno_state.anno_type is not None:
            field_type = self._tensorpc_draft_attr_obj_fields_dict[name][1]
            anno_type = field_type
            field_id = id(self._tensorpc_draft_attr_obj_fields_dict[name][0])
        new_ast_node = DraftASTNode(DraftASTType.GET_ATTR,
                                    [self._tensorpc_draft_attr_cur_node], name,
                                    anno_type, field_id=field_id)
        if self._tensorpc_draft_attr_anno_state.is_type_only:
            assert anno_type is not None
            return self._tensorpc_draft_dispatch(None, new_ast_node, anno_type)
        else:
            obj_child = getattr(self._tensorpc_draft_attr_real_obj, name)
            return self._tensorpc_draft_dispatch(obj_child, new_ast_node,
                                                 anno_type)

    def __setattr__(self, name: str, value: Any):
        if name in DraftObject.__known_attrs__:
            super().__setattr__(name, value)
            return
        if name not in self._tensorpc_draft_attr_obj_fields_dict:
            raise AttributeError(
                f"{type(self._tensorpc_draft_attr_real_obj)} has no attribute {name}"
            )
        assert self._tensorpc_draft_attr_anno_state.can_assign, "your expr isn't assignable, maybe where, you may need to stabilize it first."
        ctx = get_draft_update_context()
        if isinstance(value, DraftBase):
            if ctx._ops and ctx._ops[-1].op == JMESPathOpType.ScalarInplaceOp:
                key = ctx._ops[-1].opData["key"]
                if name == key and ctx._ops[-1].node.get_jmes_path(
                ) == self._tensorpc_draft_attr_cur_node.get_jmes_path():
                    # inplace operation, do nothing
                    return
        assert not isinstance(
            value, DraftBase
        ), "you can't assign a Draft object to another Draft object, assign real value instead."
        # TODO do validate here
        assert not isinstance(value, Undefined), "currently we don't support assign Undefined to dataclass field."
        field_id = None
        field_anno_type = None
        if self._tensorpc_draft_attr_anno_state.anno_type is not None:
            field_anno_type = self._tensorpc_draft_attr_obj_fields_dict[name][1]
            field_id = id(self._tensorpc_draft_attr_obj_fields_dict[name][0])
        ctx.add_op(
            self._tensorpc_draft_get_update_op(JMESPathOpType.SetAttr,
                                               {"items": [(name, value)]},
                                               field_id=field_id,
                                               field_anno_type=field_anno_type))


def _assert_not_draft(*value: Any):
    for v in value:
        assert not isinstance(
            v, DraftBase
        ), "you can't change a Draft object to another Draft object, use real value instead."


class DraftSequence(DraftBase):

    def __getitem__(self, index: Union[DraftBase, int,
                                       slice | tuple[Union[int, slice]]]):
        anno_type = None
        if self._tensorpc_draft_attr_anno_state.anno_type is not None:
            if len(self._tensorpc_draft_attr_anno_state.anno_type.child_types
                   ) == 0:
                anno_type = AnnotatedType.get_any_type()
            else:
                anno_type = self._tensorpc_draft_attr_anno_state.anno_type.get_child_annotated_type(
                    0)

        if isinstance(index, DraftBase):
            ast_node = DraftASTNode(DraftASTType.FUNC_CALL, [
                self._tensorpc_draft_attr_cur_node,
                index._tensorpc_draft_attr_cur_node
            ], "getItem")
            if self._tensorpc_draft_attr_anno_state.is_type_only:
                assert anno_type is not None
                return self._tensorpc_draft_dispatch(None, ast_node, anno_type)
            else:
                return self._tensorpc_draft_dispatch(
                    self._tensorpc_draft_attr_real_obj[
                        index._tensorpc_draft_attr_real_obj], ast_node)
        if isinstance(index, tuple):
            raise NotImplementedError("DraftSequence don't support N-D slice")
        if isinstance(index, slice):
            raise NotImplementedError("DraftSequence don't support slice")
        ast_node = DraftASTNode(DraftASTType.ARRAY_GET_ITEM,
                                [self._tensorpc_draft_attr_cur_node], index,
                                self._tensorpc_draft_attr_anno_state.anno_type)
        if self._tensorpc_draft_attr_anno_state.is_type_only:
            return self._tensorpc_draft_dispatch(None, ast_node, anno_type)
        else:
            return self._tensorpc_draft_dispatch(
                self._tensorpc_draft_attr_real_obj[index], ast_node, anno_type)

    def __setitem__(self, index: Union[int, DraftBase], value: Any):
        ctx = get_draft_update_context()
        if isinstance(value, DraftBase):
            if ctx._ops and ctx._ops[-1].op == JMESPathOpType.ScalarInplaceOp:
                key = ctx._ops[-1].opData["key"]
                if index == key and ctx._ops[-1].node.get_jmes_path(
                ) == self._tensorpc_draft_attr_cur_node.get_jmes_path():
                    # inplace operation, do nothing
                    return
        _assert_not_draft(value)
        if isinstance(index, DraftBase):
            if ctx._use_jmes_path:
                path = index._tensorpc_draft_attr_cur_node.get_jmes_path()
            else:
                path = index._tensorpc_draft_attr_cur_node.get_pfl_path()
            ctx.add_op(
                self._tensorpc_draft_get_update_op(
                    JMESPathOpType.Assign, {
                        "keyPath": path,
                        "value": value
                    },
                    addi_nodes=[index._tensorpc_draft_attr_cur_node]))
            return
        _assert_not_draft(index)
        ctx.add_op(
            self._tensorpc_draft_get_update_op(JMESPathOpType.ArraySet,
                                               {"items": [(index, value)]}))

    def append(self, value: Any):
        _assert_not_draft(value)
        ctx = get_draft_update_context()
        ctx.add_op(
            self._tensorpc_draft_get_update_op(JMESPathOpType.Extend,
                                               {"items": [value]}))

    def extend(self, value: list):
        _assert_not_draft(value)
        ctx = get_draft_update_context()
        ctx.add_op(
            self._tensorpc_draft_get_update_op(JMESPathOpType.Extend,
                                               {"items": value}))

    def pop(self, index: Optional[int] = None):
        """pop last element if index is None
        
        WARNING: unlike list.pop, this method will not return the popped value
        because we assume your draft model always stored in single instance.
        so poped item isn't valid anymore.
        """
        _assert_not_draft(index)
        ctx = get_draft_update_context()
        ctx.add_op(
            self._tensorpc_draft_get_update_op(JMESPathOpType.ArrayPop,
                                               {"index": index}))
        return _DraftNotValid()

    def remove(self, item: Any):
        _assert_not_draft(item)
        ctx = get_draft_update_context()
        ctx.add_op(
            self._tensorpc_draft_get_update_op(JMESPathOpType.ArrayRemove,
                                               {"item": item}))

    def clear(self):
        ctx = get_draft_update_context()
        ctx.add_op(
            self._tensorpc_draft_get_update_op(JMESPathOpType.ContainerClear,
                                               {}))

    def insert(self, index: int, item: Any):
        _assert_not_draft(index, item)
        ctx = get_draft_update_context()
        ctx.add_op(
            self._tensorpc_draft_get_update_op(JMESPathOpType.ArrayInsert, {
                "index": index,
                "item": item
            }))

    def __add__(self, other: Any):
        return _draft_seq_add(self, other, False)

    def __radd__(self, other: Any):
        return _draft_seq_add(self, other, True)

    def _get_length(self):
        ast_node = DraftASTNode(DraftASTType.FUNC_CALL,
                                [self._tensorpc_draft_attr_cur_node], "len")
        if self._tensorpc_draft_attr_anno_state.is_type_only:
            return self._tensorpc_draft_dispatch(None, ast_node, AnnotatedType(int, []))
        return self._tensorpc_draft_dispatch(
            len(self._tensorpc_draft_attr_real_obj), ast_node, AnnotatedType(int, []))

def _draft_seq_add(x: Any, other: Any, is_reverse: bool):
    assert isinstance(x, DraftSequence)
    if not isinstance(other, DraftSequence):
        assert isinstance(other, list), f"DraftSequence only support list, got {type(other)}"
        json.dumps(other)
        ast_type = DraftASTType.JSON_LITERAL
        new_node = DraftASTNode(ast_type, [], other)
    else:
        new_node = other._tensorpc_draft_attr_cur_node
    this_node = x._tensorpc_draft_attr_cur_node
    args = [this_node, new_node] if not is_reverse else [new_node, this_node]
    ast_node = DraftASTNode(DraftASTType.FUNC_CALL,
                            args, "concat")
    
    if x._tensorpc_draft_attr_anno_state.is_type_only:
        return x._tensorpc_draft_dispatch(None, ast_node, x._tensorpc_draft_attr_anno_state.anno_type)
    return x._tensorpc_draft_dispatch(
        x._tensorpc_draft_attr_real_obj, ast_node, x._tensorpc_draft_attr_anno_state.anno_type)


class DraftDict(DraftBase):

    def validate_obj(self):
        assert isinstance(
            self._tensorpc_draft_attr_real_obj, Mapping
        ), f"DraftDict only support Mapping, got {type(self._tensorpc_draft_attr_real_obj)}"

    def __getitem__(self, key: Union[str, DraftBase]):
        anno_type = None
        if self._tensorpc_draft_attr_anno_state.anno_type is not None:
            if len(self._tensorpc_draft_attr_anno_state.anno_type.child_types
                   ) != 2:
                anno_type = AnnotatedType.get_any_type()
            else:
                anno_type = self._tensorpc_draft_attr_anno_state.anno_type.get_child_annotated_type(
                    1)
        if isinstance(key, DraftBase):
            ast_node = DraftASTNode(
                DraftASTType.FUNC_CALL, [
                    self._tensorpc_draft_attr_cur_node,
                    key._tensorpc_draft_attr_cur_node
                ], "getItem", self._tensorpc_draft_attr_anno_state.anno_type)
            if self._tensorpc_draft_attr_anno_state.is_type_only:
                assert anno_type is not None
                return self._tensorpc_draft_dispatch(None, ast_node, anno_type)

            return self._tensorpc_draft_dispatch(
                self._tensorpc_draft_attr_real_obj[
                    key._tensorpc_draft_attr_real_obj], ast_node, anno_type)
        ast_node = DraftASTNode(DraftASTType.DICT_GET_ITEM,
                                [self._tensorpc_draft_attr_cur_node], key,
                                self._tensorpc_draft_attr_anno_state.anno_type)
        if self._tensorpc_draft_attr_anno_state.is_type_only:
            return self._tensorpc_draft_dispatch(None, ast_node, anno_type)
        return self._tensorpc_draft_dispatch(
            self._tensorpc_draft_attr_real_obj[key], ast_node, anno_type)

    def __setitem__(self, key: Union[str, DraftBase], value: Any):
        ctx = get_draft_update_context()
        if isinstance(value, DraftBase):
            if ctx._ops and ctx._ops[-1].op == JMESPathOpType.ScalarInplaceOp:
                key = ctx._ops[-1].opData["key"]
                if key == key and ctx._ops[-1].node.get_jmes_path(
                ) == self._tensorpc_draft_attr_cur_node.get_jmes_path():
                    # inplace operation, do nothing
                    return
        _assert_not_draft(value)
        if isinstance(key, DraftBase):
            ctx.add_op(
                self._tensorpc_draft_get_update_op(
                    JMESPathOpType.Assign, {
                        "keyPath":
                        key._tensorpc_draft_attr_cur_node.get_jmes_path(),
                        "value":
                        value
                    },
                    addi_nodes=[key._tensorpc_draft_attr_cur_node]))
            return
        _assert_not_draft(key)
        ctx.add_op(
            self._tensorpc_draft_get_update_op(JMESPathOpType.DictUpdate,
                                               {"items": {
                                                   key: value
                                               }}))

    def pop(self, key: str):
        _assert_not_draft(key)
        ctx = get_draft_update_context()
        ctx.add_op(
            self._tensorpc_draft_get_update_op(JMESPathOpType.Delete,
                                               {"keys": [key]}))
        return _DraftNotValid()

    def __delitem__(self, key: str):
        self.pop(key)

    def clear(self):
        ctx = get_draft_update_context()
        ctx.add_op(
            self._tensorpc_draft_get_update_op(JMESPathOpType.ContainerClear,
                                               {}))

    def update(self, items: dict[str, Any]):
        _assert_not_draft(items)
        ctx = get_draft_update_context()
        ctx.add_op(
            self._tensorpc_draft_get_update_op(JMESPathOpType.DictUpdate,
                                               {"items": items}))


class DraftImmutableScalar(DraftBase):
    """Leaf draft object, user can't do any operation on it."""
    pass


class DraftAny(DraftBase):
    """Leaf draft object, user can't do any operation on it."""
    pass

class DraftUnion(DraftBase):
    """Leaf draft object, user can't do any operation on it."""
    pass

class DraftComparableScalar(DraftBase):
    def __eq__(self, other: Any): # type: ignore
        return self._tensorpc_draft_binary_op(other, "==")
    
    def __ne__(self, other: Any): # type: ignore
        return self._tensorpc_draft_binary_op(other, "!=")

    def __gt__(self, other: Any):
        return self._tensorpc_draft_binary_op(other, ">")

    def __lt__(self, other: Any):
        return self._tensorpc_draft_binary_op(other, "<")
    
    def __ge__(self, other: Any):
        return self._tensorpc_draft_binary_op(other, ">=")

    def __le__(self, other: Any):
        return self._tensorpc_draft_binary_op(other, "<=")
    
class DraftImmutableString(DraftComparableScalar):
    """string object, only support c-style format.
    When you use c-style format, keep in mind that we don't support mapping key in format string.
    """
    def __mod__(self, args: tuple[Any, ...]):
        if not isinstance(args, tuple):
            assert not isinstance(args, dict), "only support dict or tuple"
            args = (args,)
        res_nodes: list[DraftASTNode] = [self._tensorpc_draft_attr_cur_node]
        for item in args:
            if isinstance(item, DraftBase):
                res_nodes.append(item._tensorpc_draft_attr_cur_node)
            else:
                # python literals
                assert isinstance(item, (int, float, str)), "only support scalar type (int/float/str)"
                if isinstance(item, str):
                    ast_type = DraftASTType.STRING_LITERAL
                else:
                    ast_type = DraftASTType.JSON_LITERAL
                node = DraftASTNode(ast_type, [], item)
                res_nodes.append(node)
        ast_node = DraftASTNode(DraftASTType.FUNC_CALL,
                                res_nodes, DraftASTFuncType.CFORMAT.value)
        if self._tensorpc_draft_attr_anno_state.is_type_only:
            return self._tensorpc_draft_dispatch(None, ast_node, AnnotatedType(str, []))
        # FIXME we shouldn't eval ast here to get real obj.
        return self._tensorpc_draft_dispatch(
            self._tensorpc_draft_attr_real_obj, ast_node, AnnotatedType(str, []))

    def join(self, arg: Any):
        assert isinstance(arg, DraftSequence), "only support join DraftSequence"
        res_nodes: list[DraftASTNode] = [
            self._tensorpc_draft_attr_cur_node,
            arg._tensorpc_draft_attr_cur_node,
        ]
        ast_node = DraftASTNode(DraftASTType.FUNC_CALL,
                                res_nodes, DraftASTFuncType.JOIN.value)
        if self._tensorpc_draft_attr_anno_state.is_type_only:
            return self._tensorpc_draft_dispatch(None, ast_node, AnnotatedType(str, []))
        # FIXME we shouldn't eval ast here to get real obj.
        return self._tensorpc_draft_dispatch(
            self._tensorpc_draft_attr_real_obj, ast_node, AnnotatedType(str, []))


class DraftMutableScalar(DraftComparableScalar):

    def __add__(self, other: Union[int, float]):
        return self._tensorpc_draft_binary_op(other, "+")

    def __sub__(self, other: Union[int, float]):
        return self._tensorpc_draft_binary_op(other, "-")

    def __mul__(self, other: Union[int, float]):
        return self._tensorpc_draft_binary_op(other, "*")

    def __truediv__(self, other: Union[int, float]):
        return self._tensorpc_draft_binary_op(other, "/")

    def __floordiv__(self, other: Union[int, float]):
        return self._tensorpc_draft_binary_op(other, "//")

    def __iadd__(self, other: Union[int, float]):
        _assert_not_draft(other)
        ctx = get_draft_update_context()
        field_id = self._tensorpc_draft_attr_cur_node.field_id
        ctx.add_op(
            self._tensorpc_draft_get_update_op(
                JMESPathOpType.ScalarInplaceOp, {
                    "op": ScalarInplaceOpType.Add,
                    "key": self._tensorpc_draft_attr_cur_node.value,
                    "value": other
                },
                drop_last=True, field_id=field_id))
        return self

    def __isub__(self, other: Union[int, float]):
        _assert_not_draft(other)
        ctx = get_draft_update_context()
        field_id = self._tensorpc_draft_attr_cur_node.field_id
        ctx.add_op(
            self._tensorpc_draft_get_update_op(
                JMESPathOpType.ScalarInplaceOp, {
                    "op": ScalarInplaceOpType.Sub,
                    "key": self._tensorpc_draft_attr_cur_node.value,
                    "value": other
                },
                drop_last=True, field_id=field_id))
        return self

    def __imul__(self, other: Union[int, float]):
        _assert_not_draft(other)
        ctx = get_draft_update_context()
        field_id = self._tensorpc_draft_attr_cur_node.field_id
        ctx.add_op(
            self._tensorpc_draft_get_update_op(
                JMESPathOpType.ScalarInplaceOp, {
                    "op": ScalarInplaceOpType.Mul,
                    "key": self._tensorpc_draft_attr_cur_node.value,
                    "value": other
                },
                drop_last=True, field_id=field_id))
        return self

    def __itruediv__(self, other: Union[int, float]):
        _assert_not_draft(other)
        ctx = get_draft_update_context()
        field_id = self._tensorpc_draft_attr_cur_node.field_id
        ctx.add_op(
            self._tensorpc_draft_get_update_op(
                JMESPathOpType.ScalarInplaceOp, {
                    "op": ScalarInplaceOpType.Div,
                    "key": self._tensorpc_draft_attr_cur_node.value,
                    "value": other
                },
                drop_last=True, field_id=field_id))
        return self

    def __ifloordiv__(self, other: Union[int, float]):
        _assert_not_draft(other)
        ctx = get_draft_update_context()
        field_id = self._tensorpc_draft_attr_cur_node.field_id
        ctx.add_op(
            self._tensorpc_draft_get_update_op(
                JMESPathOpType.ScalarInplaceOp, {
                    "op": ScalarInplaceOpType.Div,
                    "key": self._tensorpc_draft_attr_cur_node.value,
                    "value": other
                },
                drop_last=True, field_id=field_id))
        return self


def _apply_draft_update_op(cur_obj: Any,
                           op: DraftUpdateOp,
                           dynamic_key: Optional[Any] = None):
    # new cur_obj is target, apply op.
    if op.op == JMESPathOpType.SetAttr:
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
        idx = op.opData.get("index", None)
        if idx is None:
            cur_obj.pop()
        else:
            cur_obj.pop(idx)
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
        assert dynamic_key is not None
        cur_obj[dynamic_key] = op.opData["value"]
    elif op.op == JMESPathOpType.ScalarInplaceOp:
        key = op.opData["key"]
        value = op.opData["value"]
        if isinstance(cur_obj, (MutableSequence, MutableMapping)):
            if op.opData["op"] == ScalarInplaceOpType.Add:
                cur_obj[key] += value
            elif op.opData["op"] == ScalarInplaceOpType.Sub:
                cur_obj[key] -= value
            elif op.opData["op"] == ScalarInplaceOpType.Mul:
                cur_obj[key] *= value
            elif op.opData["op"] == ScalarInplaceOpType.Div:
                cur_obj[key] /= value
        else:
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


def _apply_draft_update_op_to_json(cur_obj: Any,
                                   op: DraftUpdateOp,
                                   dynamic_key: Optional[Any] = None):
    # new cur_obj is target, apply op.
    if op.op == JMESPathOpType.SetAttr:
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
        idx = op.opData.get("index", None)
        if idx is None:
            cur_obj.pop()
        else:
            cur_obj.pop(idx)
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
        cur_obj[dynamic_key] = op.opData["value"]
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


def apply_draft_update_ops(obj: Any,
                           ops: list[DraftUpdateOp],
                           ignore_exc: bool = True):
    # we delay real operation on original object to make sure
    # all validation is performed before real operation
    for op in ops:
        assert op.op >= 0, "only apply_draft_update_ops_to_json_with_root support root ops"
    for op in ops:
        try:
            cur_obj = evaluate_draft_ast(op.node, obj)
        except:
            if ignore_exc:
                continue
            raise
        dynamic_key = None
        if op.op == JMESPathOpType.Assign:
            dynamic_key = evaluate_draft_ast(op.additionalNodes[0], obj)
        _apply_draft_update_op(cur_obj, op, dynamic_key)

def apply_draft_update_ops_with_changed_obj_ids(obj: Any,
                                                ops: list[DraftUpdateOp],
                                                ignore_exc: bool = True):
    """Apply draft update ops and return changed object ids in main path.
    """
    for op in ops:
        assert op.op >= 0, "only apply_draft_update_ops_to_json_with_root support root ops"
    # we delay real operation on original object to make sure
    # all validation is performed before real operation
    changed_parent_obj_ids: set[int] = set()
    changed_obj_ids: set[int] = set()
    for op in ops:
        try:
            cur_obj, obj_id_trace = evaluate_draft_ast_with_obj_id_trace(
                op.node, obj)
        except:
            traceback.print_exc()
            if ignore_exc:
                continue
            raise
        changed_parent_obj_ids.update(obj_id_trace)
        dynamic_key = None
        if op.op == JMESPathOpType.Assign:
            dynamic_key = evaluate_draft_ast(op.additionalNodes[0], obj)
        if op.op != JMESPathOpType.ScalarInplaceOp:
            changed_obj_ids.add(id(cur_obj))
        _apply_draft_update_op(cur_obj, op, dynamic_key)
    return changed_parent_obj_ids, changed_obj_ids

def apply_draft_update_ops_to_json(obj: Any, ops: list[DraftUpdateOp]):
    # we delay real operation on original object to make sure
    # all validation is performed before real operation
    for op in ops:
        assert op.op >= 0, "only apply_draft_update_ops_to_json_with_root support root ops"
    for op in ops:
        cur_obj = evaluate_draft_ast_json(op.node, obj)
        dynamic_key = None
        if op.op == JMESPathOpType.Assign:
            dynamic_key = evaluate_draft_ast_json(op.additionalNodes[0], obj)
        _apply_draft_update_op_to_json(cur_obj, op, dynamic_key)

def apply_draft_update_ops_to_json_with_root(obj: Any, ops: list[DraftUpdateOp]):
    # we delay real operation on original object to make sure
    # all validation is performed before real operation
    is_root_changed = False 
    if obj is None:
        assert ops[0].op == JMESPathOpType.RootAssign, "root object is None, first op must be root assign"
    for op in ops:
        # only used for backend store.
        if op.op == JMESPathOpType.RootAssign:
            is_root_changed = True 
            obj = op.opData
            continue
        elif op.op == JMESPathOpType.RootInplaceOp:
            is_root_changed = True 
            value = op.opData["value"]
            if op.opData["op"] == ScalarInplaceOpType.Add:
                obj += value
            elif op.opData["op"] == ScalarInplaceOpType.Sub:
                obj -= value
            elif op.opData["op"] == ScalarInplaceOpType.Mul:
                obj *= value
            elif op.opData["op"] == ScalarInplaceOpType.Div:
                obj /= value
            continue 
        cur_obj = evaluate_draft_ast_json(op.node, obj)
        dynamic_key = None
        if op.op == JMESPathOpType.Assign:
            dynamic_key = evaluate_draft_ast_json(op.additionalNodes[0], obj)
        _apply_draft_update_op_to_json(cur_obj, op, dynamic_key)
    return is_root_changed, obj

def apply_draft_path_ops(obj: dict, ops: list[Union[JMESPathOp, PFLPathOp]]):
    # we delay real operation on original object to make sure
    # all validation is performed before real operation
    for op in ops:
        is_jmes_op = isinstance(op, JMESPathOp)
        if is_jmes_op:
            cur_obj = jmespath.search(op.path, obj)
        else:
            cur_obj = pflpath.search(op.path, obj)
        # new cur_obj is target, apply op.
        if op.op == JMESPathOpType.SetAttr:
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
            idx = op.opData.get("index", None)
            if idx is None:
                cur_obj.pop()
            else:
                cur_obj.pop(idx)
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
            if is_jmes_op:
                key = jmespath.search(op.opData["keyPath"], obj)
            else:
                key = pflpath.search(op.opData["keyPath"], obj)
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

def get_draft_pflpath(draft: DraftBase) -> str:
    return draft._tensorpc_draft_attr_cur_node.get_pfl_path()

def create_draft(obj: T, userdata: Any = None, obj_type: Optional[type[T]] = None) -> T:
    if obj_type is None:
        obj_type = type(obj)
    new_node = DraftASTNode(DraftASTType.NAME, [], "")
    prev_anno_state = _DraftAnnoState(False, None)
    return cast(
        T,
        _tensorpc_draft_dispatch(obj,
                                 new_node,
                                 userdata,
                                 prev_anno_state,
                                 anno_type=parse_type_may_optional_undefined(
                                     obj_type)))

def create_literal_draft(obj: T, userdata: Any = None) -> T:
    if isinstance(obj, str):
        ast_type = DraftASTType.STRING_LITERAL
    else:
        # obj must be json serializable
        json.dumps(obj)
        ast_type = DraftASTType.JSON_LITERAL
    new_node = DraftASTNode(ast_type, [], obj)
    prev_anno_state = _DraftAnnoState(False, None, can_assign=False)
    return cast(
        T,
        _tensorpc_draft_dispatch(obj,
                                 new_node,
                                 userdata,
                                 prev_anno_state,
                                 anno_type=parse_type_may_optional_undefined(
                                     type(obj))))

def create_draft_type_only(obj_type: type[T], userdata: Any = None) -> T:
    new_node = DraftASTNode(DraftASTType.NAME, [], "")
    prev_anno_state = _DraftAnnoState(True, None)
    anno_type = parse_type_may_optional_undefined(obj_type)        
    assert dataclasses.is_dataclass(anno_type.origin_type)
    return cast(T, _tensorpc_draft_anno_dispatch(
        anno_type, new_node, userdata,
        prev_anno_state))


def get_draft_ast_node(draft: Any) -> DraftASTNode:
    assert isinstance(draft, DraftBase), "draft should be a Draft object"
    return draft._tensorpc_draft_attr_cur_node


def get_draft_anno_type(draft: Any) -> Optional[AnnotatedType]:
    assert isinstance(draft, DraftBase), "draft should be a Draft object"
    return draft._tensorpc_draft_attr_anno_state.anno_type

def get_draft_anno_type_checked(draft: Any) -> AnnotatedType:
    assert isinstance(draft, DraftBase), "draft should be a Draft object"
    res = draft._tensorpc_draft_attr_anno_state.anno_type
    assert res is not None, "draft should have anno type"
    return res

def get_draft_anno_path_metas(draft: Any) -> list[tuple[Any, ...]]:
    assert isinstance(draft, DraftBase), "draft should be a Draft object"
    return draft._tensorpc_draft_attr_anno_state.path_metas

def _validate_cast_and_materialize(draft: Any, target_type: type[T]):
    assert dataclasses.is_dataclass(
        target_type), "target_type should be a dataclass"
    if isinstance(draft, DraftUnion):
        anno_type = draft._tensorpc_draft_attr_anno_state.anno_type
        assert anno_type is not None 
        for ctype in anno_type.child_types:
            assert dataclasses.is_dataclass(ctype), "all child types in union should be dataclass"
        assert target_type in anno_type.child_types, f"target_type {target_type} should be in union types {anno_type.child_types}"
    assert draft._tensorpc_draft_attr_anno_state.is_type_only, "materialize only support type-only mode"

def get_materialize_any_draft_to_dataclass_ops(model: Any, draft: Any, target_type: type[T]):
    """Materialize a Any type with raw dict stored to real dataclasses.

    WARNING: don't implement custom __init__ method in your dataclass.

    TODO model update will fail silently if user forget to materialize `Any` draft to dataclass.
    """
    assert dataclasses.is_dataclass(
        model), "model should be a dataclass"
    assert not isinstance(model, DraftBase), "model should not be a Draft object"
    assert isinstance(
        draft, (DraftAny, DraftUnion)), "draft should be a DraftAny or DraftUnion"
    _validate_cast_and_materialize(draft, target_type)
    anno_type = get_draft_anno_type(draft)
    # TODO add union support
    assert anno_type is not None and (anno_type.is_any_type(
    ) or anno_type.is_union_type()), "draft type should be any type"
    jmes_path = draft._tensorpc_draft_attr_cur_node.get_jmes_path()
    raw_dict = evaluate_draft_ast(draft._tensorpc_draft_attr_cur_node,
                                    model)
    if isinstance(raw_dict, target_type):
        # already materialized
        return []
    assert isinstance(raw_dict, dict), f"path {jmes_path} should be a dict or {target_type}, but got {type(raw_dict)}"
    materialized_obj = target_type(**raw_dict)
    with capture_draft_update() as ctx:
        insert_assign_draft_op(draft, materialized_obj)
    return ctx._ops

def materialize_any_draft_to_dataclass(model: Any, draft: Any, target_type: type[T]):
    """Materialize a Any type with raw dict stored to real dataclasses.
    When you have a model contains some field with type `Any` or `dict[str, Any]`,
    we don't know the real type of the field, when you use `cast_any_draft_to_dataclass`
    to get a dataclass based draft, you still can't use that draft to update model.
    you must use `materialize_any_draft_to_dataclass` to convert raw dict to dataclass before
    use casted draft.

    WARNING: your dataclass must not contains a user-defined `__init__`.
    """
    ops = get_materialize_any_draft_to_dataclass_ops(model, draft, target_type)
    apply_draft_update_ops(model, ops)
    return 

def cast_any_draft_to_dataclass(draft: Any, target_type: type[T]) -> T:
    _validate_cast_and_materialize(draft, target_type)
    assert isinstance(
        draft, (DraftAny, DraftUnion)), "draft should be a DraftAny or DraftUnion"
    anno_type = get_draft_anno_type(draft)
    assert anno_type is not None and anno_type.is_any_type(
    ), "draft type should be any type"
    new_anno_type = parse_type_may_optional_undefined(target_type)
    new_state = dataclasses.replace(draft._tensorpc_draft_attr_anno_state,
                                    anno_type=new_anno_type)
    new_draft = DraftObject(draft._tensorpc_draft_attr_real_obj,
                            draft._tensorpc_draft_attr_userdata,
                            draft._tensorpc_draft_attr_cur_node, new_state)
    return cast(T, new_draft)

def cast_any_draft(draft: Any, target_type: type[T]) -> T:
    if dataclasses.is_dataclass(target_type):
        return cast_any_draft_to_dataclass(draft, target_type)
    assert issubclass(target_type, (int, bool, str, float))
    assert isinstance(
        draft, (DraftAny, DraftUnion)), "draft should be a DraftAny or DraftUnion"
    anno_type = get_draft_anno_type(draft)
    assert anno_type is not None and anno_type.is_any_type(
    ), "draft type should be any type"
    new_anno_type = parse_type_may_optional_undefined(target_type)
    new_state = dataclasses.replace(draft._tensorpc_draft_attr_anno_state,
                                    anno_type=new_anno_type)
    res = _tensorpc_draft_anno_dispatch(
        new_anno_type, draft._tensorpc_draft_attr_cur_node, draft._tensorpc_draft_attr_userdata,
        new_state)
    return cast(T, res)

def draft_from_node_and_type(node: DraftASTNode, target_type: Any) -> Any:
    new_anno_type = parse_type_may_optional_undefined(target_type)
    prev_anno_state = _DraftAnnoState(True, new_anno_type)
    return _tensorpc_draft_anno_dispatch(
        new_anno_type, node, None,
        prev_anno_state)

def insert_assign_draft_op(draft: Any, value: Any):
    """used to insert a assign op to ctx without explicit assignment.
    Usually used when user only provide a draft object and want to assign a value to it.
    """
    _assert_not_draft(value)
    assert isinstance(draft, DraftBase), "draft should be a Draft object"
    ctx = get_draft_update_context()
    cur_node = draft._tensorpc_draft_attr_cur_node
    assert cur_node.type != DraftASTType.NAME and cur_node.type != DraftASTType.FUNC_CALL, "can't assign to root or getItem/getattr object"
    anno_state = draft._tensorpc_draft_attr_anno_state
    assert anno_state.can_assign and anno_state.can_direct_assign, "assign to this draft is disabled."
    node_prev = cur_node.children[0]

    if cur_node.type == DraftASTType.GET_ATTR:
        ctx.add_op(
            DraftUpdateOp(JMESPathOpType.SetAttr,
                          {"items": [(cur_node.value, value)]}, node_prev,
                          draft._tensorpc_draft_attr_userdata))
    elif cur_node.type == DraftASTType.ARRAY_GET_ITEM:
        ctx.add_op(
            DraftUpdateOp(JMESPathOpType.ArraySet,
                          {"items": [(cur_node.value, value)]}, node_prev,
                          draft._tensorpc_draft_attr_userdata))
    elif cur_node.type == DraftASTType.DICT_GET_ITEM:
        ctx.add_op(
            DraftUpdateOp(JMESPathOpType.DictUpdate,
                          {"items": {
                              cur_node.value: value
                          }}, node_prev, draft._tensorpc_draft_attr_userdata))
    else:
        raise NotImplementedError(f"Draft type {type(draft)} not implemented")

def copy_draft(draft: T) -> T:
    assert isinstance(draft, DraftBase), "draft should be a Draft object"
    new_node = draft._tensorpc_draft_attr_cur_node.clone_tree_only()
    new_state = dataclasses.replace(draft._tensorpc_draft_attr_anno_state)
    return draft.__class__(draft._tensorpc_draft_attr_real_obj,
                           draft._tensorpc_draft_attr_userdata, new_node, new_state)

def get_assign_disabled_draft(draft: Any):
    # disable direct assign
    assert isinstance(draft, DraftBase), "draft should be a Draft object"
    new_draft = copy_draft(draft)
    new_draft._tensorpc_draft_attr_anno_state.can_direct_assign = False
    return new_draft 

def _rebuild_draft_expr_recursive(node: DraftASTNode, root_draft: DraftBase, model: Any) -> DraftBase:
    if node.type == DraftASTType.NAME:
        if node.value == "" or node.value == "$":
            return root_draft
        return getattr(root_draft, node.value)
    elif node.type == DraftASTType.JSON_LITERAL or node.type == DraftASTType.STRING_LITERAL:
        return node.value
    elif node.type == DraftASTType.GET_ATTR:
        return getattr(_rebuild_draft_expr_recursive(node.children[0], root_draft, model), node.value)
    elif node.type == DraftASTType.ARRAY_GET_ITEM or node.type == DraftASTType.DICT_GET_ITEM:
        draft_target = _rebuild_draft_expr_recursive(node.children[0], root_draft, model)
        assert isinstance(draft_target, (DraftSequence, DraftDict))
        return draft_target[node.value]

    elif node.type == DraftASTType.BINARY_OP:
        op = node.value
        x = _rebuild_draft_expr_recursive(
            node.children[0], root_draft, model)
        y = _rebuild_draft_expr_recursive(
            node.children[1], root_draft, model)
        assert isinstance(x, DraftComparableScalar) and isinstance(y, DraftComparableScalar)
        if op == "==":
            return cast(DraftBase, x == y)
        elif op == "!=":
            return cast(DraftBase, x != y)
        elif op == ">":
            return cast(DraftBase, x > y)
        elif op == "<":
            return cast(DraftBase, x < y)
        elif op == ">=":
            return cast(DraftBase, x >= y)
        elif op == "<=":
            return cast(DraftBase, x <= y)
        else:
            raise NotImplementedError(f"op {op} not implemented")
    elif node.type == DraftASTType.FUNC_CALL:
        if node.value == "getItem":
            # for dynamic ops, we need real model value as key.
            k = evaluate_draft_ast(node.children[1], model)
            draft_target = _rebuild_draft_expr_recursive(node.children[0], root_draft, model)
            assert isinstance(draft_target, (DraftSequence, DraftDict))
            return draft_target[k]
        elif node.value == "getattr":
            return getattr(_rebuild_draft_expr_recursive(node.children[0], root_draft, model),
                           evaluate_draft_ast(node.children[1], model))
        elif node.value == "cformat":
            fmt = _rebuild_draft_expr_recursive(node.children[0], root_draft, model)
            args = [
                _rebuild_draft_expr_recursive(child, root_draft, model) for child in node.children[1:]
            ]
            assert isinstance(fmt, DraftImmutableString)
            return fmt % tuple(args)
        elif node.value == "getItemPath":
            target_node = node.children[0]
            draft_expr = _rebuild_draft_expr_recursive(target_node, root_draft, model)
            path_items = evaluate_draft_ast(node.children[1], model)
            for path_item in path_items:
                if isinstance(draft_expr, DraftObject):
                    assert isinstance(path_item, str)
                    draft_expr = getattr(draft_expr, path_item)
                elif isinstance(draft_expr, (DraftSequence, DraftDict)):
                    draft_expr = draft_expr[path_item]
                else:
                    raise NotImplementedError(f"invalid draft expr {draft_expr}")
            return draft_expr
        elif node.value == "not_null":
            draft_exprs: list[DraftBase] = []
            for child in node.children:
                draft_expr = _rebuild_draft_expr_recursive(child, root_draft, model)
                draft_exprs.append(draft_expr)
            new_node = DraftASTNode(DraftASTType.FUNC_CALL, [d._tensorpc_draft_attr_cur_node for d in draft_exprs], "not_null")
            new_ann_type = AnnotatedType(Any, [])
            new_state = dataclasses.replace(draft_exprs[0]._tensorpc_draft_attr_anno_state, anno_type=new_ann_type)
            res = DraftImmutableScalar(None, draft_exprs[0]._tensorpc_draft_attr_userdata, new_node, new_state)
            return res
        elif node.value == "where":
            # for where, we must evaluate condition to determine use which branch.
            cond = evaluate_draft_ast(
                node.children[0], model)
            if cond:
                return _rebuild_draft_expr_recursive(node.children[1], root_draft, model)
            else:
                return _rebuild_draft_expr_recursive(node.children[2], root_draft, model)
        else:
            raise NotImplementedError(f"{node.value} not supported in rebuild")
    else:
        raise NotImplementedError(f"node type {node.type} not implemented")


def rebuild_and_stabilize_draft_expr(node: DraftASTNode, root_model_draft: Any, model: Any):
    """Rebuild draft expr from node, all dynamic op (getattr, getItemPath, where) will be converted to static.
    Note: this function must be called in runtime because it depends on real model value.
    """
    assert isinstance(node, DraftASTNode)
    assert isinstance(root_model_draft, DraftBase)
    assert root_model_draft._tensorpc_draft_attr_anno_state.is_type_only, "stabilize only support type-only mode"
    assert dataclasses.is_dataclass(model), "model must be real dataclasses, not draft"
    return _rebuild_draft_expr_recursive(node, root_model_draft, model)


def stabilize_getitem_path_in_op_main_path(op: DraftUpdateOp, root_model_draft: Any, model: Any):
    """Convert dynamic ops `getattr` and `getItemPath(tgt, [...])` to static draft expr.

    `getItemPath` is usually used in nested data structure. If your draft
    expr contains dynamic path, we can't do static type analysis on it. So we need
    to convert it to static draft expr from real model.

    WARNING: attr key in your path must be correct, but value of dict/list key can be invalid because
    we only evaluate dynamic key/attr itself, the container is still draft expr.
    """
    assert isinstance(root_model_draft, DraftBase)
    assert dataclasses.is_dataclass(model), "model must be real dataclasses, not draft"
    node = op.node 
    new_draft_expr = rebuild_and_stabilize_draft_expr(node, root_model_draft, model)
    return dataclasses.replace(op, node=new_draft_expr._tensorpc_draft_attr_cur_node)
