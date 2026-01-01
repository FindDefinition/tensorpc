import ast
import inspect
from typing import Any
from tensorpc.apps.adv.codemgr.core import BaseParseResult
import tensorpc.core.dataclass_dispatch as dataclasses

from tensorpc.apps.adv.logger import ADV_LOGGER
from tensorpc.apps.adv.model import ADVNodeModel, ADVNodeHandle
import hashlib
from tensorpc.core.annolib import dataclass_flatten_fields_generator, unparse_type_expr

__TENSORPC_ADV_SYMBOL_DCLS_META__ = "__tensorpc_adv_symbol_dcls_meta__"

@dataclasses.dataclass
class SymbolDclsMeta:
    pass 

@dataclasses.dataclass
class ADVSymbolMeta:
    alias: str

@dataclasses.dataclass(kw_only=True)
class SymbolParseResult(BaseParseResult):
    symbols: list[ADVNodeHandle]
    import_stmts: list[str]
    import_stmts_for_ref: list[str]


class SymbolParser:
    def __init__(self):
        self._cached_symbol_parse_res: dict[str, list[tuple[str, SymbolParseResult]]] = {}

    def parse_symbol_node(self, code: str, global_scope: dict[str, Any], global_scripts: list[str]):
        code_to_exec = "\n".join(global_scripts + [code])
        code_to_exec_md5 = hashlib.md5(code_to_exec.encode()).hexdigest()
        if code_to_exec_md5 in self._cached_symbol_parse_res:
            for candidate_code, candidate_res in self._cached_symbol_parse_res[code_to_exec_md5]:
                if candidate_code == code_to_exec:
                    return candidate_res
        code_ast = ast.parse(code)
        # find first class def
        name_to_field_anno: dict[str, ast.expr] = {}
        for node in code_ast.body:
            if isinstance(node, ast.ClassDef):
                for stmt in node.body:
                    if isinstance(stmt, ast.AnnAssign):
                        assert isinstance(stmt.target, ast.Name), f"Unsupported target type: {type(stmt.target)}"
                        name_to_field_anno[stmt.target.id] = stmt.annotation
                break
        name_to_field_anno_str = {k: ast.unparse(v) for k, v in name_to_field_anno.items()}
        compiled = compile(code, "<string>", "exec")
        local_env: dict[str, Any] = {}
        exec(compiled, global_scope, local_env)
        symbol_handles: list[ADVNodeHandle] = []
        import_stmts: list[str] = []
        import_stmts_for_ref: list[str] = []
        def _get_type_str(type_obj):
            module = type_obj.__module__
            qname = type_obj.__qualname__
            qname_parts = qname.split(".")
            import_stmts.append(f"from {module} import {qname_parts[0]}")
            return qname
        # get import stmts for ref node (external usage)
        def _get_type_str_for_ref(type_obj):
            module = type_obj.__module__
            qname = type_obj.__qualname__
            qname_parts = qname.split(".")
            import_stmts.append(f"from {module} import {qname_parts[0]}")
            return qname


        for obj in local_env.values():
            if hasattr(obj, __TENSORPC_ADV_SYMBOL_DCLS_META__):
                assert dataclasses.is_dataclass(obj) and inspect.isclass(obj)
                for field, qname, field_type, field_metas in dataclass_flatten_fields_generator(obj):
                    symbol_name = field.name
                    if field_metas is not None:
                        for meta in field_metas:
                            if isinstance(meta, ADVSymbolMeta):
                                assert meta.alias.strip().isidentifier()
                                symbol_name = meta.alias.strip()
                                break
                    default_str = None 
                    if field.default is not dataclasses.MISSING:
                        default_str = str(field.default)
                    if qname in name_to_field_anno_str:
                        type_str = name_to_field_anno_str[qname]
                        unparse_type_expr(field_type, get_type_str=_get_type_str_for_ref)
                    else:
                        type_str = unparse_type_expr(field_type, get_type_str=_get_type_str)
                    
                    symbol_handles.append(ADVNodeHandle(
                        id=qname,
                        name=symbol_name,
                        type=type_str,
                        is_input=False,
                        symbol_name=symbol_name,
                        default=default_str
                    ))
                break
        parse_res = SymbolParseResult(
            succeed=True,
            symbols=symbol_handles,
            import_stmts=list(set(import_stmts)),
            import_stmts_for_ref=list(set(import_stmts + import_stmts_for_ref)),
        )
        if code_to_exec_md5 not in self._cached_symbol_parse_res:
            self._cached_symbol_parse_res[code_to_exec_md5] = []
        self._cached_symbol_parse_res[code_to_exec_md5].append((code_to_exec, parse_res))
        return parse_res