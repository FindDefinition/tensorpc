import ast
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
    if compat.Python3_8AndLater:
        if not isinstance(node, ast.Constant):
            raise ValueError("node not a constant")
        return node.value
    else:
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Bytes):
            return node.s
        elif isinstance(node, ast.Ellipsis):
            return Ellipsis
        elif isinstance(node, ast.NameConstant):
            return node.value
        else:
            raise ValueError("node not a constant")


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
