import ast
from functools import partial
import inspect
from typing import Any, Callable, Optional, Self, TypeVar
from tensorpc.apps.adv.codemgr.core import BackendHandle, BaseNodeCodeMeta, BaseParseResult, BaseParser, ImplCodeSpec
import tensorpc.core.dataclass_dispatch as dataclasses
import dataclasses as dataclasses_plain
from tensorpc.apps.adv.logger import ADV_LOGGER
from tensorpc.apps.adv.model import ADVHandleFlags, ADVHandlePrefix, ADVNodeModel, ADVNodeHandle
import hashlib
from tensorpc.core.annolib import dataclass_flatten_fields_generator, unparse_type_expr
from tensorpc.apps.adv.codemgr.markers import mark_symbol_group
from tensorpc.core.funcid import get_attribute_name


@dataclasses.dataclass
class ADVSymbolMeta:
    alias: str

@dataclasses.dataclass(kw_only=True)
class SymbolParseResult(BaseParseResult):
    symbol_cls_name: str
    symbols: list[BackendHandle]
    local_symbols: list[BackendHandle]
    dep_qnames_for_ext: list[str]
    dep_class_nodes: list[ADVNodeModel]
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

    def to_code_lines(self):
        assert self.node is not None 
        kwarg_str = ", ".join(self.get_node_meta_kwargs(self.node))
        decorator = f"ADV.{mark_symbol_group.__name__}({kwarg_str})"
        if self.node.ref is not None:
            # only generate line
            return ImplCodeSpec([
                f"{decorator}({self.symbol_cls_name})",
            ], -1, -1, 1, -1)
        else:
            impl = self.node.impl
            assert impl is not None
            lines = [
                f"@{decorator}",
                "@dataclasses.dataclass",
                f"class {self.symbol_cls_name}:",
            ]
            line_offset = len(lines)

            impl_lines = impl.code.splitlines()
            impl_lines_indented = [f"    {line}" for line in impl_lines]

            lines.extend(impl_lines_indented)
            end_column = len(lines[-1]) + 1
            return ImplCodeSpec(lines, line_offset, 1, len(impl_lines_indented), end_column)

    @staticmethod 
    def early_validate_code(code: str):
        code_lines = code.splitlines()
        lines = [
            f"class Foo:",
            *[f"    {line}" for line in code_lines]
        ]
        final_code = "\n".join(lines)
        ast.parse(final_code)

    def is_io_handle_changed(self, other_res: Self):
        """Compare io handles between two flow parse result. 
        if changed, all flows that depend on this fragment node (may flow) need to be re-parsed.
        TODO default change?
        """
        if len(self.symbols) != len(other_res.symbols):
            return True
        for h1, h2 in zip(self.symbols, other_res.symbols):
            if h1.symbol_name != h2.symbol_name or h1.handle.type != h2.handle.type or h1.handle.default != h2.handle.default:
                return True                
        return False


def _get_type_str(type_obj, out_list: list[str]):
    module = type_obj.__module__
    qname = type_obj.__qualname__
    qname_parts = qname.split(".")
    first_qname_with_module = f"{module}.{qname_parts[0]}"
    out_list.append(first_qname_with_module)
    return qname


_default_typing_imports = {
    "Literal",
    "Union",
    "Optional",
    "Any",
    "Annotated",
    "Concatenate",
}
_default_typing_ext_imports = {
    "Self",
}

_default_collections_imports = {
    "deque"
}

_default_collections_abc_imports = {
    "Sequence",
    "Mapping",
    "MutableMapping",
    "MutableSequence",
    "MutableSet",

    "AsyncGenerator",
    "AsyncIterable",
    "AsyncIterator",

    "Generator",
    "Iterable",
    "Iterator",
    "Awaitable",
    "Coroutine",
    "Callable",
    "Hashable",
    "Sized",
}

_builtin_types = {
    "type",
    "list",
    "set",
    "dict",
    "Annotated",
    "Concatenate",
}


_default_type_imports = _default_typing_imports | _default_typing_ext_imports | _default_collections_imports | _default_collections_abc_imports

@dataclasses_plain.dataclass
class FieldAnnoParseRes:
    type: str 
    annometas: Optional[list[ast.expr]] = None

@dataclasses_plain.dataclass
class _FieldAnnoParseDesc:
    pass 

@dataclasses_plain.dataclass
class _FieldAnnoParseTemp:
    field_name: str
    found_cls: list[str]
    found_qualnames: list[str]
    handles: list[ADVNodeHandle]

class AnnoAstParser:
    def __init__(self, global_scope: dict[str, Any], cls_node_map: dict[str, ADVNodeModel]):
        self.global_scope = global_scope
        self.cls_node_map = cls_node_map

    def _get_slice_elts(self, slice: ast.expr) -> list[ast.expr]:
        if isinstance(slice, ast.Tuple):
            return list(slice.elts)
        else:
            return [slice]

    def _parse_anno_expr_internal(self, anno: ast.expr, state: _FieldAnnoParseTemp) -> str:
        if isinstance(anno, (ast.Subscript, ast.Name, ast.Attribute)):
            if isinstance(anno, ast.Subscript):
                value = get_attribute_name(anno.value)
                slice_elts = self._get_slice_elts(anno.slice)
            else:
                value = get_attribute_name(anno)
                slice_elts = []
            value_parts = value.split(".")

            if value.startswith("typing.") or value.startswith("collections.") or value.startswith("typing_extensions.") or value in _default_type_imports:
                child_exprs = [self._parse_anno_expr_internal(elt, state) for elt in slice_elts]
                if child_exprs:
                    return f"{value}[{', '.join(child_exprs)}]"
                else:
                    return value
            elif value_parts[0] in self.global_scope:
                # eval value in global scope 
                obj = self.global_scope[value_parts[0]]
                for attr in value_parts[1:]:
                    obj = getattr(obj, attr)
                obj_type_str = _get_type_str(obj, state.found_qualnames)
                if inspect.isclass(obj) and dataclasses.is_dataclass(obj):
                    # only predefined dataclass support metas. adv class don't support nested dataclass.
                    for field, qname, field_type, field_metas in dataclass_flatten_fields_generator(obj):
                        symbol_name = field.name
                        field_qname = f"{state.field_name}.{qname}"
                        depth = len(qname.split(".")) + 1
                        if field_metas is not None:
                            for meta in field_metas:
                                if isinstance(meta, ADVSymbolMeta):
                                    assert meta.alias.strip().isidentifier()
                                    symbol_name = meta.alias.strip()
                                    break
                        # TODO parse default?
                        default_str = None 
                        if field.default is not dataclasses.MISSING:
                            # TODO this only valids for simple defaults
                            default_str = repr(field.default)
                        local_qnames: list[str] = []
                        type_str = unparse_type_expr(field_type, get_type_str=partial(_get_type_str, out_list=local_qnames))
                        state.found_qualnames.extend(local_qnames)
                        handle_id = f"{ADVHandlePrefix.Output}-{field_qname}"

                        handle = ADVNodeHandle(
                            id=handle_id,
                            name=symbol_name,
                            type=type_str,
                            symbol_name=symbol_name,
                            default=default_str,
                            sym_depth=depth - 1,
                            flags=int(ADVHandleFlags.IS_SYM_HANDLE),
                        )
                        state.handles.append(handle)

                child_exprs = [self._parse_anno_expr_internal(elt, state) for elt in slice_elts]
                if child_exprs:
                    return f"{obj_type_str}[{', '.join(child_exprs)}]"
                else:
                    return obj_type_str

            elif value_parts[0] in self.cls_node_map:
                assert len(value_parts) == 1, "Only support simple symbol class reference in annotation for now"
                state.found_cls.append(value_parts[0])
                child_exprs = [self._parse_anno_expr_internal(elt, state) for elt in slice_elts]
                if child_exprs:
                    return f"{value}[{', '.join(child_exprs)}]"
                else:
                    return value
            else:
                # this shouldn't happen because it should available in global scope.
                type_str = ast.unparse(anno)
                child_exprs = [self._parse_anno_expr_internal(elt, state) for elt in slice_elts]
                if child_exprs:

                    return f"{type_str}[{', '.join(child_exprs)}]"
                else:
                    return type_str

        elif isinstance(anno, ast.BoolOp):
            assert isinstance(anno.op, ast.BitOr), "Only support | in type annotation"
            # handle | (union)
            child_exprs = [self._parse_anno_expr_internal(v, state) for v in anno.values]
            return f"{' | '.join(child_exprs)}"
        else:
            raise ValueError(f"Unsupported annotation expression: {ast.dump(anno)}")


    def _parse_anno_expr(self, anno: ast.expr, state: _FieldAnnoParseTemp):
        res = FieldAnnoParseRes("")
        # 1. process Annotated first. 
        if isinstance(anno, ast.Subscript):
            value = get_attribute_name(anno.value)
            if value == "Annotated":
                slice_elts = self._get_slice_elts(anno.slice)
                type_str = self._parse_anno_expr_internal(slice_elts[0], state)
                if len(slice_elts) > 1:
                    res.annometas = slice_elts[1:]
                res.type = type_str
            else:
                res.type = self._parse_anno_expr_internal(anno, state)
        else:
            res.type = self._parse_anno_expr_internal(anno, state)
        return res

    def parse_symbol_group_type_ast(self, name_to_field_anno: dict[str, ast.expr]):
        found_cls: list[str] = []
        global_qnames: set[str] = set()
        handles: list[ADVNodeHandle] = []
        symbol_handles: list[BackendHandle] = []
        symbol_handles_local_flow: list[BackendHandle] = []
        name_to_field_anno_str = {k: ast.unparse(v) for k, v in name_to_field_anno.items()}
        for k, v in name_to_field_anno.items():
            top_field_handle = ADVNodeHandle(
                id=f"{ADVHandlePrefix.Output}-{k}",
                name=k,
                type="",
                symbol_name=k,
                default=None,
                sym_depth=0,
                flags=int(ADVHandleFlags.IS_SYM_HANDLE),
            )
            handles.append(top_field_handle)
            nested_handles: list[ADVNodeHandle] = []
            found_qualnames: list[str] = []
            state = _FieldAnnoParseTemp(
                field_name=k,
                found_cls=found_cls,
                found_qualnames=found_qualnames,
                handles=nested_handles,
            )
            global_qnames.update(found_qualnames)
            type_str = self._parse_anno_expr(v, state).type
            top_field_handle.type = type_str

            bh = BackendHandle(
                handle=top_field_handle,
                index=-1,
                type_dep_qnames=found_qualnames,
            )
            symbol_handles.append(bh)

            local_type_str = name_to_field_anno_str[k]
            local_top_field_handle = dataclasses.replace(top_field_handle, 
                type=local_type_str)
            local_backend_handle = BackendHandle(
                handle=local_top_field_handle,
                index=-1,
                type_dep_qnames=[]
            )
            symbol_handles_local_flow.append(local_backend_handle)

            for nested_handle in nested_handles:
                nested_bh = BackendHandle(
                    handle=nested_handle,
                    index=-1,
                    type_dep_qnames=found_qualnames,
                )
                symbol_handles.append(nested_bh)
                local_handle = dataclasses.replace(nested_handle, 
                    type=nested_handle.type)

                local_backend_handle = BackendHandle(
                    handle=local_handle,
                    index=-1,
                    type_dep_qnames=found_qualnames
                )
                symbol_handles_local_flow.append(local_backend_handle)
        for cnt, h in enumerate(symbol_handles):
            h.index = cnt
        for cnt, h in enumerate(symbol_handles_local_flow):
            h.index = cnt
        return found_cls, global_qnames, symbol_handles, symbol_handles_local_flow

class SymbolParser(BaseParser):
    def __init__(self):
        self._cached_symbol_parse_res: dict[str, list[tuple[str, SymbolParseResult]]] = {}
    
    @staticmethod
    def get_typing_import_stmts():
        default_typing_imports = set([*_default_typing_imports, "cast"])
        default_typing_import = f"from typing import {', '.join(default_typing_imports)}"
        default_typing_ex_import = f"from typing_extensions import {', '.join(_default_typing_ext_imports)}"
        default_collections_import = f"from collections import {', '.join(_default_collections_imports)}"
        default_collections_abc_import = f"from collections.abc import {', '.join(_default_collections_abc_imports)}"
        return [
            default_typing_import,
            default_typing_ex_import,
            default_collections_import,
            default_collections_abc_import,
        ]

    def parse_symbol_node(self, node: ADVNodeModel, code: str, global_scope: dict[str, Any], global_scripts: list[str],
            parent_node: Optional[ADVNodeModel], cls_node_map: dict[str, ADVNodeModel]):
        node_id = node.id
        code_lines = code.splitlines()
        assert len(code_lines) > 0
        end_column = len(code_lines[-1]) + 1
        code_to_exec = "\n".join(global_scripts + [code])
        code_to_exec_md5 = hashlib.md5(code_to_exec.encode()).hexdigest()
        if code_to_exec_md5 in self._cached_symbol_parse_res:
            for candidate_code, candidate_res in self._cached_symbol_parse_res[code_to_exec_md5]:
                if candidate_code == code_to_exec:
                    return candidate_res
        code_with_dataclass = [
            "@dataclasses.dataclass",
            f"class {node.name}:",
            *[f"    {line}" for line in code_lines]
        ]
        code_ast = ast.parse("\n".join(code_with_dataclass))
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

        anno_parser = AnnoAstParser(global_scope, cls_node_map)
        found_cls, global_qnames, symbol_handles, symbol_handles_local_flow = anno_parser.parse_symbol_group_type_ast(name_to_field_anno)
        for bh in symbol_handles:
            bh.handle.set_source_info_inplace(node_id, bh.handle.id)
        for bh in symbol_handles_local_flow:
            bh.handle.set_source_info_inplace(node_id, bh.handle.id)

        parse_res = SymbolParseResult(
            node=dataclasses.replace(node),
            symbol_cls_name=cls_name,
            succeed=True,
            symbols=symbol_handles,
            local_symbols=symbol_handles_local_flow,
            dep_qnames_for_ext=list(global_qnames),
            dep_class_nodes=[dataclasses.replace(cls_node_map[c]) for c in found_cls],
            num_symbols=len(symbol_handles),
        )
        if code_to_exec_md5 not in self._cached_symbol_parse_res:
            self._cached_symbol_parse_res[code_to_exec_md5] = []
        self._cached_symbol_parse_res[code_to_exec_md5].append((code_to_exec, parse_res))
        return parse_res