from collections.abc import Mapping, Sequence
import enum
import json
import types
from typing import Any, Callable, MutableSequence, Optional, Type, TypeVar, Union, cast, get_type_hints
import tensorpc.core.dataclass_dispatch as dataclasses

T = TypeVar("T")

# currently jmespath don't support ast to code, so we use a simple ast here.

class DraftASTType(enum.IntEnum):
    GET_ATTR = 0
    ARRAY_GET_ITEM = 1
    DICT_GET_ITEM = 2
    FUNC_CALL = 3
    NAME = 4
    JSON_LITERAL = 5
    STRING_LITERAL = 6
    BINARY_OP = 7
    UNARY_OP = 8


class DraftASTFuncType(enum.Enum):
    GET_ITEM = "getitem"
    GET_ATTR = "getattr"
    CFORMAT = "cformat"
    GET_ITEM_PATH = "getitem_path"
    NOT_NULL = "not_null"
    WHERE = "where"
    CREATE_ARRAY = "create_array"
    CONCAT = "concat"

_FRONTEND_SUPPORTED_FUNCS = {
    DraftASTFuncType.GET_ITEM.value, DraftASTFuncType.GET_ATTR.value,
    DraftASTFuncType.CFORMAT.value, DraftASTFuncType.GET_ITEM_PATH.value,
    DraftASTFuncType.NOT_NULL.value, DraftASTFuncType.WHERE.value,
    DraftASTFuncType.CREATE_ARRAY.value, DraftASTFuncType.CONCAT.value
}

@dataclasses.dataclass
class DraftASTNode:
    type: DraftASTType
    children: list["DraftASTNode"]
    value: Any
    userdata: Any = None
    field_id: Optional[int] = None

    def get_jmes_path(self) -> str:
        if self.type == DraftASTType.NAME:
            return self.value if self.value != "" else "$"
        return _draft_ast_to_jmes_path_recursive(self)

    def iter_child_nodes(self):
        for child in self.children:
            yield child
            yield from child.iter_child_nodes()

    def get_child_nodes_in_main_path(self):
        res: list[DraftASTNode] = [self]
        cur_node = self
        while cur_node.children:
            res.append(cur_node.children[0])
            cur_node = cur_node.children[0]
        return res

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()


    def __repr__(self):
        return self.get_jmes_path()

    def to_userdata_removed(self):
        child_removed = [child.to_userdata_removed() for child in self.children]
        return DraftASTNode(self.type, child_removed, self.value)

    def clone_tree_only(self):
        # only clone tree structure, not include userdata and value
        child_cloned = [child.clone_tree_only() for child in self.children]
        return DraftASTNode(self.type, child_cloned, self.value, self.userdata, self.field_id)

ROOT_NODE = DraftASTNode(DraftASTType.NAME, [], "")

_GET_ITEMS = set([
    DraftASTType.GET_ATTR, DraftASTType.ARRAY_GET_ITEM,
    DraftASTType.DICT_GET_ITEM
])


def _draft_ast_to_jmes_path_recursive(node: DraftASTNode) -> str:
    if node.type == DraftASTType.NAME:
        return "$" if node.value == "" else node.value
    elif node.type == DraftASTType.JSON_LITERAL:
        if isinstance(node.value, (int, float)):
            if isinstance(node.value, bool):
                return f"`{str(node.value).lower()}`"                
            return f"`{node.value}`"
        else:
            return f"`{json.dumps(node.value)}`"
    elif node.type == DraftASTType.STRING_LITERAL:
        return f"\'{node.value}\'"
    elif node.type in _GET_ITEMS:
        child_value = _draft_ast_to_jmes_path_recursive(node.children[0])
        is_root = child_value == ""
        if node.type == DraftASTType.GET_ATTR:
            if is_root:
                return f"{node.value}"
            else:
                return f"{child_value}.{node.value}"
        elif node.type == DraftASTType.ARRAY_GET_ITEM:
            return f"{child_value}[{node.value}]"
        else:
            return f"{child_value}.\"{node.value}\""
    elif node.type == DraftASTType.FUNC_CALL:
        assert node.value in _FRONTEND_SUPPORTED_FUNCS, f"unsupported func {node.value}, only support {_FRONTEND_SUPPORTED_FUNCS}"
        return f"{node.value}(" + ",".join([
            _draft_ast_to_jmes_path_recursive(child) for child in node.children
        ]) + ")"
    
    elif node.type == DraftASTType.BINARY_OP:
        op = node.value
        return f"{_draft_ast_to_jmes_path_recursive(node.children[0])}{op}{_draft_ast_to_jmes_path_recursive(node.children[1])}"
    elif node.type == DraftASTType.UNARY_OP:
        op = node.value
        return f"{op}{_draft_ast_to_jmes_path_recursive(node.children[0])}"
    
    else:
        raise NotImplementedError(f"node type {node.type} not implemented")


def evaluate_draft_ast(node: DraftASTNode, obj: Any) -> Any:
    if node.type == DraftASTType.NAME:
        if node.value == "" or node.value == "$":
            return obj
        return getattr(obj, node.value)
    elif node.type == DraftASTType.JSON_LITERAL or node.type == DraftASTType.STRING_LITERAL:
        return node.value
    elif node.type == DraftASTType.GET_ATTR:
        return getattr(evaluate_draft_ast(node.children[0], obj), node.value)
    elif node.type == DraftASTType.ARRAY_GET_ITEM or node.type == DraftASTType.DICT_GET_ITEM:
        return evaluate_draft_ast(node.children[0], obj)[node.value]
    elif node.type == DraftASTType.BINARY_OP:
        op = node.value
        x = evaluate_draft_ast(node.children[0], obj)
        y = evaluate_draft_ast(node.children[1], obj)
        if op == "==":
            return x == y
        elif op == "!=":
            return x != y
        elif op == ">":
            return x > y
        elif op == "<":
            return x < y
        elif op == ">=":
            return x >= y
        elif op == "<=":
            return x <= y
        elif op == "&&":
            return x and y
        elif op == "||":
            return x or y
        else:
            raise NotImplementedError
    elif node.type == DraftASTType.UNARY_OP:
        op = node.value
        x = evaluate_draft_ast(node.children[0], obj)
        if op == "!":
            return not x
        else:
            raise NotImplementedError
    elif node.type == DraftASTType.FUNC_CALL:
        if node.value == "getitem":
            k = evaluate_draft_ast(node.children[1], obj)
            return evaluate_draft_ast(node.children[0], obj)[k]
        elif node.value == "getattr":
            return getattr(evaluate_draft_ast(node.children[0], obj),
                           evaluate_draft_ast(node.children[1], obj))
        elif node.value == "cformat":
            fmt = evaluate_draft_ast(node.children[0], obj)
            args = [
                evaluate_draft_ast(child, obj) for child in node.children[1:]
            ]
            return fmt % tuple(args)
        elif node.value == "getitem_path":
            target = evaluate_draft_ast(node.children[0], obj)
            path_list = evaluate_draft_ast(node.children[1], obj)
            assert isinstance(path_list, list)
            cur_obj = target
            for p in path_list:
                if isinstance(cur_obj, (Sequence, Mapping)):
                    cur_obj = cur_obj[p]
                else:
                    assert dataclasses.is_dataclass(cur_obj)
                    cur_obj = getattr(cur_obj, p)
            return cur_obj
        elif node.value == "not_null":
            for child in node.children:
                res = evaluate_draft_ast(child, obj)
                if res is not None:
                    return res
            return None
        elif node.value == "where":
            cond = evaluate_draft_ast(
                node.children[0], obj)
            x = evaluate_draft_ast(
                node.children[1], obj)
            y = evaluate_draft_ast(
                node.children[2], obj)
            return x if cond else y
        elif node.value == "create_array":
            return [evaluate_draft_ast(child, obj) for child in node.children]
        elif node.value == "concat":
            return sum([evaluate_draft_ast(child, obj) for child in node.children], [])
        else:
            raise NotImplementedError(f"func {node.value} not implemented")
    else:
        raise NotImplementedError(f"node type {node.type} not implemented")


def evaluate_draft_ast_noexcept(node: DraftASTNode, obj: Any) -> Optional[Any]:
    try:
        return evaluate_draft_ast(node, obj)
    except NotImplementedError:
        raise
    except Exception:
        return None


def evaluate_draft_ast_with_obj_id_trace(node: DraftASTNode,
                                         obj: Any) -> tuple[Any, list[int]]:
    """Evaluate ast and record object trace (dynamic slice isn't included).
    Usually used to implement obj change event.
    """
    if node.type == DraftASTType.NAME:
        if node.value == "" or node.value == "$":
            return (obj, [id(obj)])
        return getattr(obj, node.value), [id(obj)]
    elif node.type == DraftASTType.JSON_LITERAL or node.type == DraftASTType.STRING_LITERAL:
        return node.value, []
    elif node.type == DraftASTType.GET_ATTR:
        target, obj_id_trace = evaluate_draft_ast_with_obj_id_trace(
            node.children[0], obj)
        res = getattr(target, node.value)
        return res, obj_id_trace + [id(res)]
    elif node.type == DraftASTType.ARRAY_GET_ITEM or node.type == DraftASTType.DICT_GET_ITEM:
        target, obj_id_trace = evaluate_draft_ast_with_obj_id_trace(
            node.children[0], obj)
        res = target[node.value]
        return res, obj_id_trace + [id(res)]
    elif node.type == DraftASTType.BINARY_OP:
        op = node.value
        x = evaluate_draft_ast(
            node.children[0], obj)
        y = evaluate_draft_ast(
            node.children[1], obj)
        if op == "==":
            return x == y, []
        elif op == "!=":
            return x != y, []
        elif op == ">":
            return x > y, []
        elif op == "<":
            return x < y, []
        elif op == ">=":
            return x >= y, []
        elif op == "<=":
            return x <= y, []
        elif op == "&&":
            return x and y, []
        elif op == "||":
            return x or y, []
        else:
            raise NotImplementedError
    elif node.type == DraftASTType.UNARY_OP:
        op = node.value
        x = evaluate_draft_ast(node.children[0], obj)
        if op == "!":
            return not x, []
        else:
            raise NotImplementedError
    elif node.type == DraftASTType.FUNC_CALL:
        if node.value == "getitem":
            target, obj_id_trace = evaluate_draft_ast_with_obj_id_trace(
                node.children[0], obj)
            k, _ = evaluate_draft_ast_with_obj_id_trace(node.children[1], obj)
            res = target[k]
            return target[k], obj_id_trace + [id(res)]
        elif node.value == "getattr":
            target, obj_id_trace = evaluate_draft_ast_with_obj_id_trace(
                node.children[0], obj)
            k, _ = evaluate_draft_ast_with_obj_id_trace(node.children[1], obj)
            res = getattr(target, k)
            return res, obj_id_trace + [id(res)]
        elif node.value == "cformat":
            fmt = evaluate_draft_ast(node.children[0], obj)
            args = [
                evaluate_draft_ast(child, obj) for child in node.children[1:]
            ]
            return fmt % tuple(args), []
        elif node.value == "getitem_path":
            target = evaluate_draft_ast(node.children[0], obj)
            path_list = evaluate_draft_ast(node.children[1], obj)
            assert isinstance(path_list, list)
            cur_obj = target
            obj_id_trace = []
            for p in path_list:
                if isinstance(cur_obj, (Sequence, Mapping)):
                    cur_obj = cur_obj[p]
                else:
                    assert dataclasses.is_dataclass(cur_obj), f"{type(cur_obj)} is not a dataclass"
                    cur_obj = getattr(cur_obj, p)
                obj_id_trace.append(id(cur_obj))
            return cur_obj, obj_id_trace
        elif node.value == "not_null":
            for child in node.children:
                res, obj_id_trace = evaluate_draft_ast_with_obj_id_trace(
                    child, obj)
                if res is not None:
                    return res, obj_id_trace
            return None, []
        elif node.value == "where":
            cond = evaluate_draft_ast(
                node.children[0], obj)
            x = evaluate_draft_ast(
                node.children[1], obj)
            y = evaluate_draft_ast(
                node.children[2], obj)
            return x if cond else y, []
        elif node.value == "create_array":
            return [evaluate_draft_ast(child, obj) for child in node.children], []
        elif node.value == "concat":
            return sum([evaluate_draft_ast(child, obj) for child in node.children], []), []
        else:
            raise NotImplementedError(f"func {node.value} not implemented")
    else:
        raise NotImplementedError(f"node type {node.type} not implemented")


def evaluate_draft_ast_json(node: DraftASTNode, obj: Any) -> Any:
    if node.type == DraftASTType.NAME:
        if node.value == "" or node.value == "$":
            return obj
        return obj[node.value]
    elif node.type == DraftASTType.JSON_LITERAL or node.type == DraftASTType.STRING_LITERAL:
        return node.value
    elif node.type == DraftASTType.GET_ATTR:
        return evaluate_draft_ast_json(node.children[0], obj)[node.value]
    elif node.type == DraftASTType.ARRAY_GET_ITEM or node.type == DraftASTType.DICT_GET_ITEM:
        return evaluate_draft_ast_json(node.children[0], obj)[node.value]
    elif node.type == DraftASTType.BINARY_OP:
        op = node.value
        x = evaluate_draft_ast_json(node.children[0], obj)
        y = evaluate_draft_ast_json(node.children[1], obj)
        if op == "==":
            return x == y
        elif op == "!=":
            return x != y
        elif op == ">":
            return x > y
        elif op == "<":
            return x < y
        elif op == ">=":
            return x >= y
        elif op == "<=":
            return x <= y
        elif op == "&&":
            return x and y, []
        elif op == "||":
            return x or y, []
        else:
            raise NotImplementedError
    elif node.type == DraftASTType.UNARY_OP:
        op = node.value
        x = evaluate_draft_ast_json(node.children[0], obj)
        if op == "!":
            return not x, []
        else:
            raise NotImplementedError
    elif node.type == DraftASTType.FUNC_CALL:
        if node.value == "getitem":
            k = evaluate_draft_ast_json(node.children[1], obj)
            return evaluate_draft_ast_json(node.children[0], obj)[k]
        elif node.value == "getattr":
            src = evaluate_draft_ast_json(node.children[0], obj)
            tgt = evaluate_draft_ast_json(node.children[1], obj)
            return src[tgt]
        elif node.value == "cformat":
            fmt = evaluate_draft_ast_json(node.children[0], obj)
            args = [
                evaluate_draft_ast_json(child, obj) for child in node.children[1:]
            ]
            return fmt % tuple(args)
        elif node.value == "getitem_path":
            target = evaluate_draft_ast_json(node.children[0], obj)
            path_list = evaluate_draft_ast_json(node.children[1], obj)
            assert isinstance(path_list, list)
            cur_obj = target
            for p in path_list:
                if isinstance(cur_obj, (Sequence, Mapping)):
                    cur_obj = cur_obj[p]
                else:
                    assert dataclasses.is_dataclass(cur_obj)
                    cur_obj = getattr(cur_obj, p)
            return cur_obj
        elif node.value == "not_null":
            for child in node.children:
                res = evaluate_draft_ast_json(child, obj)
                if res is not None:
                    return res
            return None
        elif node.value == "where":
            cond = evaluate_draft_ast_json(
                node.children[0], obj)
            x = evaluate_draft_ast_json(
                node.children[1], obj)
            y = evaluate_draft_ast_json(
                node.children[2], obj)
            return x if cond else y
        elif node.value == "create_array":
            return [evaluate_draft_ast(child, obj) for child in node.children]
        elif node.value == "concat":
            return sum([evaluate_draft_ast_json(child, obj) for child in node.children], [])
        else:
            raise NotImplementedError(f"func {node.value} not implemented")
    else:
        raise NotImplementedError(f"node type {node.type} not implemented")

