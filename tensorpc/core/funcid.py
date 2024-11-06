import ast
import copy
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union
import tokenize
import io

from tensorpc import compat


def from_constant(node):
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            res = from_constant(node.operand)
            assert isinstance(res, (int, float, complex))
            return -res
        else:
            raise ValueError("node not a constant")
    if not isinstance(node, ast.Constant):
        raise ValueError("node not a constant")
    return node.value


def get_toplevel_func_node(tree: ast.Module):
    from collections import deque
    res: List[Tuple[Union[ast.FunctionDef, ast.AsyncFunctionDef],
                    List[ast.ClassDef]]] = []
    todo: Deque[Tuple[List[ast.AST],
                      List[ast.ClassDef]]] = deque([([*tree.body], [])])
    while todo:
        body, cur_parent_ns = todo.popleft()
        for node in body:
            if isinstance(node, (ast.ClassDef)):
                todo.append(([*node.body], [*cur_parent_ns, node]))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                res.append((node, cur_parent_ns))
    return res


def get_toplevel_class_node(tree: ast.Module):
    from collections import deque
    res: List[Tuple[ast.ClassDef, List[ast.ClassDef]]] = []
    todo: Deque[Tuple[List[ast.AST],
                      List[ast.ClassDef]]] = deque([([*tree.body], [])])
    while todo:
        body, cur_parent_ns = todo.popleft()
        for node in body:
            if isinstance(node, (ast.ClassDef)):
                todo.append(([*node.body], [*cur_parent_ns, node]))
                res.append((node, cur_parent_ns))
    return res


def find_toplevel_func_node_by_lineno(tree: ast.Module, lineno: int):
    # TODO should we check try block?
    from collections import deque
    todo: Deque[Tuple[List[ast.AST],
                      List[ast.ClassDef]]] = deque([([*tree.body], [])])
    while todo:
        body, cur_parent_ns = todo.popleft()
        for node in body:
            if isinstance(node, (ast.ClassDef)):
                todo.append(([*node.body], [*cur_parent_ns, node]))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_lineno = node.lineno
                deco_list = node.decorator_list
                # fix lineno to match inspect
                if len(deco_list) > 0:
                    func_lineno = min([d.lineno for d in deco_list])
                if func_lineno == lineno:
                    return (node, cur_parent_ns)
                elif func_lineno > lineno:
                    break
            elif isinstance(node, (ast.If, )):
                todo.append(([*node.body], cur_parent_ns))
                todo.append(([*node.orelse], cur_parent_ns))

    return None


def find_toplevel_func_node_container_by_lineno(tree: ast.Module, lineno: int):
    # TODO should we check try block?
    from collections import deque
    todo: Deque[Tuple[List[ast.AST],
                      List[ast.ClassDef]]] = deque([([*tree.body], [])])
    while todo:
        body, cur_parent_ns = todo.popleft()
        for node in body:
            if isinstance(node, (ast.ClassDef)):
                todo.append(([*node.body], [*cur_parent_ns, node]))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_lineno = node.lineno
                deco_list = node.decorator_list
                # fix lineno to match inspect
                if len(deco_list) > 0:
                    func_lineno = min([d.lineno for d in deco_list])
                func_end_lineno = node.end_lineno
                if func_end_lineno is None:
                    in_range = (func_lineno <= lineno)
                else:
                    in_range = (func_lineno <= lineno) and (lineno
                                                            <= func_end_lineno)
                if in_range:
                    return [*cur_parent_ns, node]
                else:
                    continue
            elif isinstance(node, (ast.If, )):
                todo.append(([*node.body], cur_parent_ns))
                todo.append(([*node.orelse], cur_parent_ns))
    return None


class _NodeNameAccessor(ast.NodeVisitor):
    """remove all nodes except node contains target identifier.
    """

    def __init__(self, target_identifier: str):
        self._target_identifier = target_identifier
        self._name_node = None

    def visit_Name(self, node):
        if node.id == self._target_identifier:
            self._name_node = node


class NodeFoldingTransformer(ast.NodeTransformer):
    """remove all nodes except node contains target identifier.
    remove all nested func/class/async func def
    """

    def __init__(self, root_func_node: Union[ast.FunctionDef,
                                             ast.AsyncFunctionDef],
                 target_identifier: str):
        self._target_identifier = target_identifier
        self._root_func_node = root_func_node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.generic_visit(node)
        if node is self._root_func_node:
            return node
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self.generic_visit(node)
        if node is self._root_func_node:
            return node
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.generic_visit(node)
        return None

    def _only_keep_node_contains_target_identifier(self, node: ast.AST):
        self.generic_visit(node)
        accessor = _NodeNameAccessor(self._target_identifier)
        accessor.visit(node)
        if accessor._name_node is not None:
            return node
        return None

    def visit_Return(self, node: ast.Return) -> Any:
        return self._only_keep_node_contains_target_identifier(node)

    def visit_Assign(self, node: ast.Assign) -> Any:
        return self._only_keep_node_contains_target_identifier(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        return self._only_keep_node_contains_target_identifier(node)

    def visit_Expr(self, node: ast.Expr) -> Any:
        return self._only_keep_node_contains_target_identifier(node)

    def visit_If(self, node: ast.If) -> Any:
        return self._only_keep_node_contains_target_identifier(node)

    def visit_For(self, node: ast.For) -> Any:
        return self._only_keep_node_contains_target_identifier(node)

    def visit_While(self, node: ast.While) -> Any:
        return self._only_keep_node_contains_target_identifier(node)

    def visit_With(self, node: ast.With) -> Any:
        return self._only_keep_node_contains_target_identifier(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> Any:
        return self._only_keep_node_contains_target_identifier(node)

    def visit_Try(self, node: ast.Try) -> Any:
        return self._only_keep_node_contains_target_identifier(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> Any:
        return self._only_keep_node_contains_target_identifier(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> Any:
        return self._only_keep_node_contains_target_identifier(node)

    def visit_AugAssign(self, node: ast.AST) -> Any:
        return self._only_keep_node_contains_target_identifier(node)

    def visit_TypeAlias(self, node: ast.AST) -> Any:
        return self._only_keep_node_contains_target_identifier(node)

    def visit_TryStar(self, node: ast.AST) -> Any:
        return self._only_keep_node_contains_target_identifier(node)

    def visit_Match(self, node: ast.AST) -> Any:
        return self._only_keep_node_contains_target_identifier(node)

    def visit_Raise(self, node: ast.Raise) -> Any:
        return self._only_keep_node_contains_target_identifier(node)

    def visit_Import(self, node: ast.Import) -> Any:
        return self._only_keep_node_contains_target_identifier(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        return self._only_keep_node_contains_target_identifier(node)

    def visit_Continue(self, node: ast.Continue) -> Any:
        return None


def fold_func_node_with_target_identifier(tree: Union[ast.FunctionDef,
                                                      ast.AsyncFunctionDef],
                                          target_identifier: str):
    tree = copy.deepcopy(tree)
    transformer = NodeFoldingTransformer(tree, target_identifier)
    return ast.fix_missing_locations(transformer.visit(tree))


def fold_func_node_with_target_identifier_to_code(
        tree: Union[ast.FunctionDef,
                    ast.AsyncFunctionDef], target_identifier: str, with_func: bool = True):
    node = fold_func_node_with_target_identifier(tree, target_identifier)
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    if not with_func:
        return ast.unparse(ast.Module(node.body, []))
    else:
        return ast.unparse(node)


def split_func_id(
        fid: str,
        path_delimiter: str = ".",
        local_delimiter: str = "-") -> Tuple[List[str], str, List[str]]:
    relative_path_parts = list(fid.split(path_delimiter))
    filename_local_id = relative_path_parts[-1]
    local_parts = filename_local_id.split(local_delimiter)
    filename = local_parts[0]
    return relative_path_parts[:-1], filename, local_parts[1:]


def get_tokens(source: str, toknums: Tuple[int]):
    tokens = tokenize.tokenize(io.BytesIO(source.encode('utf-8')).readline)
    for toknum, tokval, (srow, scol), (erow, ecol), line in tokens:
        if toknum in toknums:
            yield (tokval, (srow, scol), (erow, ecol), line)


def get_all_comments(source: str) -> List[Tuple[str, int, int]]:
    res = []
    for tokval, (srow, scol), _, _ in get_tokens(source, (tokenize.COMMENT, )):
        res.append((tokval, srow, scol))
    return res


def clean_source_code(lines: List[str],
                      remove_comment: bool = True,
                      remove_empty_line: bool = True,
                      source: Optional[str] = None,
                      rstrip: bool = True):
    if source is None:
        source = "\n".join(lines)
    lines = lines.copy()
    if remove_comment:
        comments = get_all_comments(source)
        for _, srow, scol in comments:
            lines[srow - 1] = lines[srow - 1][:scol]
    if rstrip:
        lines = [l.rstrip() for l in lines]
    if remove_empty_line:
        new_lines = []  # type: List[str]
        for line in lines:
            line_test = line.strip(" \t")
            if line_test != "":
                new_lines.append(line)
    else:
        new_lines = lines
    return new_lines


def _get_attribute_name(node, parts):
    if isinstance(node, ast.Attribute):
        parts.append(node.attr)
        return _get_attribute_name(node.value, parts)
    elif isinstance(node, ast.Name):
        parts.append(node.id)
    else:
        raise NotImplementedError


def get_attribute_name_parts(node):
    parts = []
    _get_attribute_name(node, parts)
    return parts[::-1]


def get_attribute_name(node):
    return ".".join(get_attribute_name_parts(node))


def determine_code_common_indent(code: str):
    lines = code.split("\n")
    indent = None
    for line in lines:
        if line.strip() == "":
            continue
        line_indent = len(line) - len(line.lstrip())
        if indent is None:
            indent = line_indent
        else:
            indent = min(indent, line_indent)
    return indent


def remove_common_indent_from_code(code: str):
    common_indent = determine_code_common_indent(code)
    code_without_indent = "\n".join(
        [l[common_indent:] for l in code.split("\n")])
    return code_without_indent


def get_body_blocks_from_code(code: str, autorun_block_symbol: str = ""):
    code = remove_common_indent_from_code(code)
    tree = ast.parse(code)
    func_node = get_toplevel_func_node(tree)[0][0]
    body_start = func_node.body[0].lineno
    body_code_lines = code.split("\n")[body_start - 1:]
    # if a line start with '#%%', it's a block splitter.

    if autorun_block_symbol != "":
        body_code_blocks = []
        current_block = []
        for line in body_code_lines:
            if line.strip().startswith(autorun_block_symbol):
                body_code_blocks.append("\n".join(current_block))
                current_block = []
            else:
                current_block.append(line)
        if len(current_block) > 0:
            body_code_blocks.append("\n".join(current_block))
        # body_code_blocks = [b.strip() for b in body_code_blocks if b.strip() != ""]
    else:
        body_code = "\n".join(body_code_lines)
        body_code_blocks = [body_code]

    return body_code_blocks


def _main():

    code = """
def find_toplevel_func_node_container_by_lineno(tree: ast.Module, lineno: int):
    # TODO should we check try block?
    from collections import deque
    todo: Deque[Tuple[List[ast.AST],
                      List[ast.ClassDef]]] = deque([([*tree.body], [])])
    while todo:
        body, cur_parent_ns = todo.popleft()
        for node in body:
            if isinstance(node, (ast.ClassDef)):
                todo.append(([*node.body], [*cur_parent_ns, node]))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_lineno = node.lineno
                deco_list = node.decorator_list
                # fix lineno to match inspect
                if len(deco_list) > 0:
                    func_lineno = min([d.lineno for d in deco_list])
                func_end_lineno = node.end_lineno
                if func_end_lineno is None:
                    in_range = (func_lineno <= lineno)
                else:
                    in_range = (func_lineno <= lineno) and (lineno <= func_end_lineno) 
                if in_range:
                    return [*cur_parent_ns, node]
                else:
                    print("WTFWTF")
                    continue
            elif isinstance(node, (ast.If, )):
                todo.append(([*node.body], cur_parent_ns))
                todo.append(([*node.orelse], cur_parent_ns))
    return None

"""

    code2 = code
    tree = ast.parse(code2)
    node = tree.body[0]
    assert isinstance(node, ast.FunctionDef)
    print(
        fold_func_node_with_target_identifier_to_code(node,
                                                      "node"))


if __name__ == "__main__":
    _main()
