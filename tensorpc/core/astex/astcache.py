import ast

import dataclasses
from pathlib import Path
from typing import Dict, List, Tuple, Union

@dataclasses.dataclass
class AstCacheItem:
    path: Path
    st_size: int
    st_mtime: float
    st_ctime: float
    content: str
    num_lines: int
    tree: ast.AST
    all_nodes: List[ast.AST]
    code_range_to_nodes: Dict[Tuple[int, int, int, int], List[ast.AST]]
    first_lineno_col_to_nodes: Dict[Tuple[int, int], List[ast.AST]]
    func_defs: List[Union[ast.FunctionDef, ast.AsyncFunctionDef]]
    assign_ctx_nodes: List[Union[ast.Assign, ast.AugAssign, ast.AnnAssign]]

    def query_code_range_nodes(self, lineno: int, col: int, end_lineno: int,
                               end_col: int) -> List[ast.AST]:
        key = (lineno, col, end_lineno, end_col)
        if key in self.code_range_to_nodes:
            return self.code_range_to_nodes[key]
        return []

    def query_first_lineno_col_nodes(self, lineno: int,
                                     col: int) -> List[ast.AST]:
        # some names such as func arg aren't ast node, so we also need this
        # function to search arg by lineno and col.
        key = (lineno, col)
        if key in self.first_lineno_col_to_nodes:
            return self.first_lineno_col_to_nodes[key]
        return []

    def query_func_def_nodes_by_lineno_range(self, lineno: int, end_lineno: int) -> List[Union[ast.FunctionDef, ast.AsyncFunctionDef]]:
        if not self.func_defs:
            return []
        res: List[Union[ast.FunctionDef, ast.AsyncFunctionDef]] = []
        # if we need better performance, we can use interval tree here.
        # for function range only, just use o(n) search.
        for func_def in self.func_defs:
            if func_def.end_lineno is None:
                func_end_lineno = self.num_lines
            else:
                func_end_lineno = func_def.end_lineno
            if func_def.lineno <= lineno and func_end_lineno >= end_lineno:
                res.append(func_def)
        return res


class AstCache:
    def __init__(self) -> None:
        self._cache: Dict[Path, AstCacheItem] = {}

    def query_path(self, path: Path) -> AstCacheItem:
        # check mtime
        path = path.resolve()
        stat = path.stat()
        if path in self._cache:
            item = self._cache[path]
            if item.st_mtime == stat.st_mtime:
                return item
        # read from file
        res = self.read_from_file(path)
        self._cache[path] = res
        return res

    def read_from_file(self, path: Path) -> AstCacheItem:
        path = path.resolve()
        with open(path, "r") as f:
            content = f.read()
        tree = ast.parse(content)
        num_lines = content.count("\n") + 1
        all_nodes = list(ast.walk(tree))
        code_range_to_nodes: Dict[Tuple[int, int, int, int],
                                  List[ast.AST]] = {}
        first_lineno_col_to_nodes: Dict[Tuple[int, int], List[ast.AST]] = {}
        func_defs: List[Union[ast.FunctionDef, ast.AsyncFunctionDef]] = []
        assign_ctx_nodes: List[Union[ast.Assign, ast.AugAssign, ast.AnnAssign]] = []
        for node in all_nodes:
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                assign_ctx_nodes.append(node)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_defs.append(node)
            if hasattr(node, "lineno") and hasattr(node, "col_offset"):
                lineno = getattr(node, "lineno")
                col_offset = getattr(node, "col_offset")
                key = (lineno, col_offset)
                if key not in first_lineno_col_to_nodes:
                    first_lineno_col_to_nodes[key] = []
                first_lineno_col_to_nodes[key].append(node)

                if hasattr(node, "end_lineno") and hasattr(
                        node, "end_col_offset"):
                    end_lineno = getattr(node, "end_lineno")
                    end_col_offset = getattr(node, "end_col_offset")
                    key = (lineno, col_offset, end_lineno, end_col_offset)

                    if key not in code_range_to_nodes:
                        code_range_to_nodes[key] = []
                    code_range_to_nodes[key].append(node)
        item = AstCacheItem(path, len(content),
                            path.stat().st_mtime,
                            path.stat().st_ctime, content, num_lines, tree, all_nodes,
                            code_range_to_nodes, first_lineno_col_to_nodes,
                            func_defs, 
                            assign_ctx_nodes)
        return item
