
from typing import TypeVar, Any, cast
from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.core.annolib import AnnotatedType, parse_type_may_optional_undefined 
from .draft import DraftBase, DraftImmutableScalar, DraftSequence, DraftASTNode, DraftASTType, DraftASTFuncType, _tensorpc_draft_dispatch, create_literal_draft


T = TypeVar('T')

def getitem_path_dynamic(target: Any, path: Any, result_type: type[T]) -> T:
    assert isinstance(target, DraftBase), "target should be a Draft object"
    assert isinstance(path, DraftSequence), "path should be a DraftSequence"
    tgt_node = target._tensorpc_draft_attr_cur_node
    path_node = path._tensorpc_draft_attr_cur_node
    new_node = DraftASTNode(DraftASTType.FUNC_CALL, [tgt_node, path_node], DraftASTFuncType.GET_ITEM_PATH.value)
    new_anno_type = parse_type_may_optional_undefined(
                                     result_type)
    new_node.userdata = new_anno_type
    prev_anno_state = target._tensorpc_draft_attr_anno_state
    return cast(T, _tensorpc_draft_dispatch(None,
                                 new_node,
                                 target._tensorpc_draft_attr_userdata,
                                 prev_anno_state,
                                 anno_type=new_anno_type))

def not_null(*args: Any):
    """resolve first not null value"""
    nodes: list[DraftASTNode] = []
    draft_exprs: list[DraftBase] = []
    for a in args:
        if isinstance(a, DraftBase):
            nodes.append(a._tensorpc_draft_attr_cur_node)
            draft_exprs.append(a)
        else:
            expr = create_literal_draft(a)
            nodes.append(expr._tensorpc_draft_attr_cur_node)
            draft_exprs.append(expr)
    new_node = DraftASTNode(DraftASTType.FUNC_CALL, [d._tensorpc_draft_attr_cur_node for d in draft_exprs], "not_null")
    new_ann_type = AnnotatedType(Any, [])
    new_state = dataclasses.replace(draft_exprs[0]._tensorpc_draft_attr_anno_state, anno_type=new_ann_type)
    res = DraftImmutableScalar(None, draft_exprs[0]._tensorpc_draft_attr_userdata, new_node, new_state)
    return cast(Any, res)