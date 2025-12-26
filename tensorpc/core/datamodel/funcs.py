
from typing import Any, Optional, TypeVar, Union, cast

from typing_extensions import Literal

from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.core.annolib import (AnnotatedType,
                                   parse_type_may_optional_undefined)
from tensorpc.core.datamodel.draftast import evaluate_draft_ast_noexcept

from .draft import (DraftASTFuncType, DraftASTNode, DraftASTType, DraftBase, DraftDict,
                    DraftImmutableScalar, DraftSequence,
                    _tensorpc_draft_anno_dispatch,
                    create_literal_draft, get_draft_anno_type,
                    get_draft_anno_type_checked)

T = TypeVar('T')

def evaluate_draft(draft: T, model: Any) -> Optional[T]:
    assert isinstance(draft, DraftBase)
    return evaluate_draft_ast_noexcept(draft._tensorpc_draft_attr_cur_node, model)

def literal_val(obj: T) -> T:
    return create_literal_draft(obj)

def getitem_path_dynamic(target: Any, path: Any, result_type: Any) -> Any:
    assert isinstance(target, DraftBase), "target should be a Draft object"
    assert isinstance(path, DraftSequence), "path should be a DraftSequence"
    tgt_node = target._tensorpc_draft_attr_cur_node
    path_node = path._tensorpc_draft_attr_cur_node
    assert target._tensorpc_draft_attr_anno_state.is_type_only, "getitem_path_dynamic should be used in type only mode"
    new_node = DraftASTNode(DraftASTType.FUNC_CALL, [tgt_node, path_node], DraftASTFuncType.GET_ITEM_PATH.value)
    new_anno_type = parse_type_may_optional_undefined(
                                     result_type)
    new_node.userdata = new_anno_type
    prev_anno_state = target._tensorpc_draft_attr_anno_state
    return cast(Any, _tensorpc_draft_anno_dispatch(new_anno_type,
                                 new_node,
                                 target._tensorpc_draft_attr_userdata,
                                 prev_anno_state))

def _simple_expr_func_with_any_res(func_name: str, *args: Any, return_type: Optional[Any] = None) -> Any:
    nodes: list[DraftASTNode] = []
    draft_exprs: list[DraftBase] = []
    userdata = None
    for a in args:
        if isinstance(a, DraftBase):
            nodes.append(a._tensorpc_draft_attr_cur_node)
            draft_exprs.append(a)
            if userdata is None:
                userdata = a._tensorpc_draft_attr_userdata
        else:
            expr = create_literal_draft(a)
            nodes.append(expr._tensorpc_draft_attr_cur_node)
            draft_exprs.append(expr)
    new_node = DraftASTNode(DraftASTType.FUNC_CALL, [d._tensorpc_draft_attr_cur_node for d in draft_exprs], func_name)
    new_ann_type = parse_type_may_optional_undefined(Any if return_type is None else return_type)
    res = _tensorpc_draft_anno_dispatch(new_ann_type,
                                 new_node,
                                 userdata,
                                 draft_exprs[0]._tensorpc_draft_attr_anno_state)
    return cast(Any, res)

def not_null(*args: Any):
    """resolve first not null value"""
    return _simple_expr_func_with_any_res("not_null", *args)

def where(cond: Any, x: Any, y: Any, return_type: Optional[Any] = None):
    return _simple_expr_func_with_any_res("where", cond, x, y, return_type=return_type)

def array(*args: Any):
    return cast(list[Any], _simple_expr_func_with_any_res("array", *args, return_type=list[Any]))

def _logical_op(a: Any, b: Any, op: Literal["&&", "||"]) -> bool:
    assert isinstance(a, DraftBase) and isinstance(b, DraftBase), "logical_and should be used with Draft objects"
    assert get_draft_anno_type_checked(a).is_bool_type() and get_draft_anno_type_checked(b).is_bool_type(), "logical_and should be used with bool type Draft objects"
    return cast(bool, a._tensorpc_draft_logic_op(b, op))
    
def logical_and(a: Any, b: Any) -> bool:
    return _logical_op(a, b, "&&")

def logical_or(a: Any, b: Any) -> bool:
    return _logical_op(a, b, "||")

def dict_get_item(a: Any, attr: str):
    assert isinstance(a, DraftDict)
    return cast(Any, a[attr])

def length(a: Any) -> int:
    assert isinstance(a, DraftSequence), "length should be used with DraftSequence objects"
    return cast(int, a._get_length())