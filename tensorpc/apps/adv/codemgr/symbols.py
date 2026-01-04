import ast
from functools import partial
import inspect
from typing import Any, Callable, Optional, Self, TypeVar
from tensorpc.apps.adv.codemgr.core import BackendHandle, BaseNodeCodeMeta, BaseParseResult
import tensorpc.core.dataclass_dispatch as dataclasses

from tensorpc.apps.adv.logger import ADV_LOGGER
from tensorpc.apps.adv.model import ADVHandlePrefix, ADVNodeModel, ADVNodeHandle
import hashlib
from tensorpc.core.annolib import dataclass_flatten_fields_generator, unparse_type_expr

__TENSORPC_ADV_SYMBOL_DCLS_META__ = "__tensorpc_adv_symbol_dcls_meta__"

T = TypeVar("T")

def mark_symbol_group(id: str, position: tuple[float, float]) -> Callable[[T], T]:
    def wrapper(fn_wrapped: T) -> T:
        setattr(fn_wrapped, __TENSORPC_ADV_SYMBOL_DCLS_META__, BaseNodeCodeMeta(id, position))
        return fn_wrapped   
    return wrapper


@dataclasses.dataclass
class ADVSymbolMeta:
    alias: str

@dataclasses.dataclass(kw_only=True)
class SymbolParseResult(BaseParseResult):
    symbol_cls_name: str
    symbols: list[BackendHandle]
    local_symbols: list[BackendHandle]
    dep_qnames: list[str]
    dep_qnames_for_ext: list[str]
    num_symbols: int

    def copy(self, node_id: Optional[str] = None, offset: Optional[int] = None, is_sym_handle: bool = False) -> Self:
        new_symbols = [
            s.copy(node_id, offset, is_sym_handle)
            for s in self.symbols
        ]
        new_local_symbols = [
            s.copy(node_id, offset, is_sym_handle)
            for s in self.local_symbols
        ]
        return dataclasses.replace(
            self,
            symbols=new_symbols,
            local_symbols=new_local_symbols,
        )

_default_typing_imports = [
    "Literal",
    "Union",
    "Optional",
    "Any",
    "Callable",
    "Generator",
    "Iterable",
    "Iterator",
    "Type",
    "AsyncGenerator",
    "Awaitable",
    "Coroutine",
    "AsyncIterable",
    "AsyncIterator",
]
_default_typing_ext_imports = [
    "Self",
]

_default_collections_imports = [
    "Sequence",
    "Mapping",
]



def _get_type_str(type_obj, out_list: list[str]):
    module = type_obj.__module__
    qname = type_obj.__qualname__
    qname_parts = qname.split(".")
    first_qname_with_module = f"{module}.{qname_parts[0]}"
    out_list.append(first_qname_with_module)
    return qname

class SymbolParser:
    def __init__(self):
        self._cached_symbol_parse_res: dict[str, list[tuple[str, SymbolParseResult]]] = {}

    def parse_symbol_node(self, node_id: str, code: str, global_scope: dict[str, Any], global_scripts: list[str]):
        code_to_exec = "\n".join(global_scripts + [code])
        code_to_exec_md5 = hashlib.md5(code_to_exec.encode()).hexdigest()
        if code_to_exec_md5 in self._cached_symbol_parse_res:
            for candidate_code, candidate_res in self._cached_symbol_parse_res[code_to_exec_md5]:
                if candidate_code == code_to_exec:
                    return candidate_res
        code_ast = ast.parse(code)
        # find first class def
        name_to_field_anno: dict[str, ast.expr] = {}
        assert len(code_ast.body) == 1, "Only one class definition is allowed in symbol group definition"
        cdef_node = code_ast.body[0]
        assert isinstance(cdef_node, ast.ClassDef), "Only class definition is allowed in symbol group definition"
        cls_name = cdef_node.name
        for stmt in cdef_node.body:
            if isinstance(stmt, ast.AnnAssign):
                assert isinstance(stmt.target, ast.Name), f"Unsupported target type: {type(stmt.target)}"
                name_to_field_anno[stmt.target.id] = stmt.annotation
        name_to_field_anno_str = {k: ast.unparse(v) for k, v in name_to_field_anno.items()}
        compiled = compile(code, "<string>", "exec")
        local_env: dict[str, Any] = {}
        exec(compiled, global_scope, local_env)
        obj = local_env[cls_name]
        symbol_handles: list[BackendHandle] = []
        symbol_handles_local_flow: list[BackendHandle] = []

        import_qnames: list[str] = []
        import_qnames_for_ref: list[str] = []
        cnt = 0
        assert dataclasses.is_dataclass(obj) and inspect.isclass(obj)
        for field, qname, field_type, field_metas in dataclass_flatten_fields_generator(obj):
            symbol_name = field.name
            depth = len(qname.split("."))
            if field_metas is not None:
                for meta in field_metas:
                    if isinstance(meta, ADVSymbolMeta):
                        assert meta.alias.strip().isidentifier()
                        symbol_name = meta.alias.strip()
                        break
            default_str = None 
            if field.default is not dataclasses.MISSING:
                default_str = str(field.default)
            local_qnames: list[str] = []
            type_str = unparse_type_expr(field_type, get_type_str=partial(_get_type_str, out_list=local_qnames))
            local_type_str = type_str
            type_str_is_local = False
            if qname in name_to_field_anno_str:
                local_type_str = name_to_field_anno_str[qname]
                type_str_is_local = True
            handle_id = f"{ADVHandlePrefix.Output}-{qname}"
            handle = ADVNodeHandle(
                id=handle_id,
                name=symbol_name,
                type=type_str,
                is_input=False,
                symbol_name=symbol_name,
                default=default_str,
                type_dep_qnames=local_qnames,
                source_node_id=node_id,
                source_handle_id=handle_id,
                is_sym_handle=True,
                sym_depth=depth - 1,
            )
            symbol_handles.append(BackendHandle(
                handle=handle,
                index=cnt,
            ))
            local_handle = dataclasses.replace(handle, 
                type=local_type_str, type_dep_qnames=[] if type_str_is_local else local_qnames)
            local_backend_handle = BackendHandle(
                handle=local_handle,
                index=cnt,
            )
            symbol_handles_local_flow.append(local_backend_handle)
            cnt += 1

        parse_res = SymbolParseResult(
            symbol_cls_name=cls_name,
            succeed=True,
            symbols=symbol_handles,
            local_symbols=symbol_handles_local_flow,
            dep_qnames=list(set(import_qnames)),
            dep_qnames_for_ext=list(set(import_qnames + import_qnames_for_ref)),
            num_symbols=cnt,
        )
        if code_to_exec_md5 not in self._cached_symbol_parse_res:
            self._cached_symbol_parse_res[code_to_exec_md5] = []
        self._cached_symbol_parse_res[code_to_exec_md5].append((code_to_exec, parse_res))
        return parse_res