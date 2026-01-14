import ast
import inspect
from typing import Any, Callable, Optional, Self, TypeVar, Union
from typing_extensions import Literal
import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.apps.adv.codemgr.core import BackendHandle, BaseParseResult, BaseParser, ImplCodeSpec
from tensorpc.apps.adv.codemgr.markers import mark_fragment_def
from tensorpc.apps.adv.model import ADVEdgeModel, ADVHandlePrefix, ADVNodeModel, ADVNodeHandle
import hashlib
import dataclasses as dataclasses_plain
from tensorpc.core.funcid import clean_source_code

@dataclasses.dataclass
class FragmentInputDesc:
    mapping: dict[str, str]

@dataclasses.dataclass
class FragmentOutputDesc:
    type: Literal["single", "tuple", "dict"]
    # symbol to alias
    mapping: dict[str, tuple[str, str]]

T = TypeVar("T")


@dataclasses.dataclass(kw_only=True)
class FragmentParseResult(BaseParseResult):
    func_name: str
    input_handles: list[BackendHandle]
    output_handles: list[BackendHandle]
    out_type: Literal["single", "tuple", "dict"]
    out_type_anno: str

    def copy(self, node_id: str) -> Self:
        new_output_handles: list[BackendHandle] = []
        for h in self.output_handles:
            new_h = h.copy(node_id)
            new_output_handles.append(new_h)

        new_input_handles: list[BackendHandle] = []
        for h in self.input_handles:
            new_h = h.copy()
            new_h.handle.source_node_id = None
            new_h.handle.source_handle_id = None
            new_input_handles.append(new_h)
        return dataclasses.replace(
            self,
            input_handles=new_input_handles,
            output_handles=new_output_handles,
        )

    def do_alias_map(self, alias_map: dict[str, str]) -> Self:
        new_output_handles: list[BackendHandle] = []
        for h in self.output_handles:
            new_h = h.copy()
            if h.handle.symbol_name in alias_map:
                new_alias = alias_map[h.handle.symbol_name]
                new_h.handle.name = new_alias
                new_h.handle.symbol_name = new_alias
            new_output_handles.append(new_h)
        return dataclasses.replace(
            self,
            output_handles=new_output_handles,
        )

    @staticmethod
    def get_signature_lines_from_handles(input_handles: list[BackendHandle]) -> list[str]:
        lines: list[str] = []
        for i, bh in enumerate(input_handles):
            handle = bh.handle
            if handle.default is not None:
                line = f"{handle.name}: {handle.type} = {handle.default}, "
            else:
                line = f"{handle.name}: {handle.type}, "
            lines.append(f"    {line}")
        return lines

    def to_code_lines(self, id_to_parse_res: dict[str, "BaseParseResult"]):
        assert self.node is not None 
        kwarg_str = ", ".join(self.get_node_meta_kwargs(self.node))
        if self.node.alias_map is not None:
            kwarg_str += f', alias_map="{self.node.alias_map}"'
        if self.node.inlinesf_name is not None:
            kwarg_str += f', inlineflow_name="{self.node.inlinesf_name}"'
        decorator = f"ADV.{mark_fragment_def.__name__}({kwarg_str})"
        # for fragment ref, we use inline Annotated instead of decorator
        if self.node.ref_node_id is not None:
            return ImplCodeSpec([
                f"{decorator}({self.func_name})",
            ], -1, -1, 1, -1)
        else:
            impl = self.node.impl
            assert impl is not None 
            code = impl.code
            code_lines = code.splitlines()
            # generate signature from handles
            # TODO class support 
            lines = [
                f"@{decorator}",
                f"def {self.func_name}(",
            ]
            lines += self.get_signature_lines_from_handles(self.input_handles)
            lines.append(f") -> {self.out_type_anno}:")
            line_offset = len(lines)
            code_lines_indented = [f"    {line}" for line in code_lines]
            lines.extend(code_lines_indented)
            end_column = len(code_lines_indented[-1]) + 1
            return ImplCodeSpec(lines, line_offset, 1, len(code_lines_indented), end_column)

    def is_io_handle_changed(self, other_res: Self):
        """Compare io handles between two flow parse result. 
        if changed, all flows that depend on this fragment node (may flow) need to be re-parsed.
        TODO default change?
        """
        if len(self.input_handles) != len(other_res.input_handles):
            return True
        if len(self.output_handles) != len(other_res.output_handles):
            return True
        for h1, h2 in zip(self.input_handles, other_res.input_handles):
            if h1.symbol_name != h2.symbol_name or h1.handle.type != h2.handle.type or h1.handle.default != h2.handle.default:
                return True
        for h1, h2 in zip(self.output_handles, other_res.output_handles):
            if h1.symbol_name != h2.symbol_name or h1.handle.type != h2.handle.type or h1.handle.default != h2.handle.default:
                return True
                
        return False

def _parse_single_desc(desc: str) -> tuple[str, str]:
    if "->" in desc:
        parts = desc.split("->")
        assert len(parts) == 2, f"Invalid output description: {desc}"
        symbol_name = parts[0].strip()
        alias = parts[1].strip()
    else:
        symbol_name = desc.strip()
        alias = symbol_name
    assert symbol_name.isidentifier() and alias.isidentifier(), \
        f"Invalid symbol name or alias in output description: {desc}"
    return symbol_name, alias

def parse_alias_map(alias_map: str) -> dict[str, str]:
    # alias_map: use alias->new_alias,alias2->new_alias2
    mapping: dict[str, str] = {}
    items = alias_map.split(",")
    for item in items:
        alias, new_alias = _parse_single_desc(item)
        mapping[alias] = new_alias
    return mapping

def mark_stable_symbols(*args: Any):
    # used to make auto-generated inputs stable.
    # we will look for all ast.Name to determine inputs of fragment,
    # so we can use this function to create dummy references to symbols
    pass 

def mark_outputs(desc: Union[str, tuple[str, ...], dict[str, str]], /) -> FragmentOutputDesc:
    # we actually don't run this function, we will parse ast
    # and extract the info from function call node.
    out_desc = FragmentOutputDesc("single", {})
    if isinstance(desc, str):
        symbol_name, alias = _parse_single_desc(desc)
        out_desc.type = "single"  
        out_desc.mapping = {
            "": (symbol_name, alias)
        }
    elif isinstance(desc, tuple):
        out_desc.type = "tuple"
        for i, item in enumerate(desc):
            symbol_name, alias = _parse_single_desc(item)
            out_desc.mapping[str(i)] = (symbol_name, alias)
    elif isinstance(desc, dict):
        out_desc.type = "dict"
        for key, desc in desc.items():
            symbol_name, alias = _parse_single_desc(desc)
            out_desc.mapping[key] = (symbol_name, alias)
    else:
        raise ValueError(f"Invalid output description type: {type(desc)}, must be one of str, tuple, dict")
    return out_desc

class _ChainAttrFinder(ast.NodeVisitor):
    def __init__(self, global_scope: dict[str, Any]):
        self._global_scope = global_scope 
        self.map_res: dict[ast.AST, tuple[str, Any]] = {}

    def _extract_attr_chain(self, node: ast.AST):
        parts: list[str] = []
        parts_node: list[ast.AST] = []
        cur_node = node
        name_found = False
        while isinstance(cur_node, (ast.Attribute, ast.Name)):
            if isinstance(cur_node, ast.Attribute):
                parts.append(cur_node.attr)
                parts_node.append(cur_node)
                cur_node = cur_node.value
            else:
                parts.append(cur_node.id)
                parts_node.append(cur_node)
                name_found = True
                break
        return parts, parts_node, name_found

    def _visit_Attribute_or_name(self, node: Union[ast.Attribute, ast.Name]):
        # TODO block nesetd function def support
        parts, parts_node, name_found = self._extract_attr_chain(
            node)
        
        if not name_found:
            return self.generic_visit(node)
        parts = parts[::-1]
        parts_node = parts_node[::-1]
        cur_obj = self._global_scope

        for part in parts:
            if isinstance(cur_obj, dict):
                if part in cur_obj:
                    cur_obj = cur_obj[part]
                else:
                    return self.generic_visit(node)
            else:
                if hasattr(cur_obj, part):
                    cur_obj = getattr(cur_obj, part)
                else:
                    return self.generic_visit(node)
        full_name = ".".join(parts)
        self.map_res[node] = (full_name, cur_obj)

    def visit_Name(self, node: ast.Name):
        return self._visit_Attribute_or_name(node)

    def visit_Attribute(self, node: ast.Attribute):
        return self._visit_Attribute_or_name(node)

class _ChainAttrFinderV2(ast.NodeVisitor):
    def __init__(self):
        self.map_res: dict[ast.AST, list[str]] = {}

    def _extract_attr_chain(self, node: ast.AST):
        parts: list[str] = []
        cur_node = node
        name_found = False
        while isinstance(cur_node, (ast.Attribute, ast.Name)):
            if isinstance(cur_node, ast.Attribute):
                parts.append(cur_node.attr)
                cur_node = cur_node.value
            else:
                parts.append(cur_node.id)
                name_found = True
                break
        return parts, name_found

    def _visit_Attribute_or_name(self, node: Union[ast.Attribute, ast.Name]):
        # TODO block nesetd function def support
        parts, name_found = self._extract_attr_chain(
            node)
        if not name_found:
            return self.generic_visit(node)
        parts = parts[::-1]
        self.map_res[node] = parts

    def visit_Name(self, node: ast.Name):
        return self._visit_Attribute_or_name(node)

    def visit_Attribute(self, node: ast.Attribute):
        return self._visit_Attribute_or_name(node)


class _FragmentApiFinder(ast.NodeVisitor):
    def __init__(self, chain_attr_map: dict[ast.AST, tuple[str, Any]]):
        self._chain_attr_map = chain_attr_map 
        self.map_res: dict[ast.Call, Any] = {}

    def visit_Call(self, node: ast.Call):
        if node.func in self._chain_attr_map:
            full_name, obj = self._chain_attr_map[node.func]
            if obj is mark_outputs:
                self.map_res[node] = obj
        self.generic_visit(node)

class _FragmentApiFinderV2(ast.NodeVisitor):
    def __init__(self, chain_attr_map: dict[ast.AST, list[str]]):
        self._chain_attr_map = chain_attr_map 
        self.map_res: dict[ast.Call, Any] = {}

    def visit_Call(self, node: ast.Call):
        if node.func in self._chain_attr_map:
            parts = self._chain_attr_map[node.func]
            full_name = ".".join(parts)
            if full_name.startswith("ADV.mark_outputs"):
                self.map_res[node] = mark_outputs
        self.generic_visit(node)

def _parse_mark_outputs_ast(node: ast.Call) -> FragmentOutputDesc:
    assert len(node.args) == 1, "mark_outputs() must have exactly one argument"
    arg = node.args[0]
    if isinstance(arg, ast.Constant):
        assert isinstance(arg.value, str), "mark_outputs() argument must be a string if not tuple/dict"
        return mark_outputs(arg.value)
    elif isinstance(arg, ast.Tuple):
        items: list[str] = []
        for elt in arg.elts:
            assert isinstance(elt, ast.Constant) and isinstance(elt.value, str), \
                "mark_outputs() tuple elements must be strings"
            items.append(elt.value)
        return mark_outputs(tuple(items))
    elif isinstance(arg, ast.Dict):
        mapping: dict[str, str] = {}
        for key_node, value_node in zip(arg.keys, arg.values):
            assert isinstance(key_node, ast.Constant) and isinstance(key_node.value, str), \
                "mark_outputs() dict keys must be strings"
            assert isinstance(value_node, ast.Constant) and isinstance(value_node.value, str), \
                "mark_outputs() dict values must be strings"
            mapping[key_node.value] = value_node.value
        return mark_outputs(mapping)
    else:
        raise ValueError("mark_outputs() argument must be a string, tuple of strings, or dict of strings")

class _ExtractSymbolFromFragment:
    def __init__(self):
        self._assigned_symbols: dict[str, BackendHandle] = {} 

    def _extract_symbol_from_node(self, root: ast.AST, scope: dict[str, BackendHandle]):
        for node in ast.walk(root):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                name = node.id
                if name in scope:
                    self._assigned_symbols[name] = scope[name]
                    scope.pop(name)

    def parse_block(self, body: list[ast.stmt], scope: dict[str, Any]):
        for stmt in body:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                if isinstance(stmt, ast.Assign):
                    targets = stmt.targets
                else:
                    targets = [stmt.target]
                for target in targets:
                    if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store):
                        # name is overrided by local assign, ignore this symbol.
                        if target.id in scope:
                            scope.pop(target.id)
                if stmt.value is not None:
                    self._extract_symbol_from_node(stmt.value, scope)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                # ignore nested cls/func/lambda for now.
                continue
            else:
                self._extract_symbol_from_node(stmt, scope)

@dataclasses_plain.dataclass
class _FragmentPrepCache:
    code_clean: str 
    tree: ast.Module
    output_desc: FragmentOutputDesc
    end_column: int
    num_lines: int

class FragmentParser(BaseParser):
    def __init__(self):
        self._cache : Optional[_FragmentPrepCache] = None
    
    def _cached_parse_output_desc(self, code: str) -> _FragmentPrepCache:
        lines = code.splitlines()
        assert len(lines) > 0
        code_for_compare = "\n".join(clean_source_code(lines))
        if self._cache is not None and self._cache.code_clean == code_for_compare:
            return self._cache
        tree = ast.parse(code)
        chain_finder = _ChainAttrFinderV2()
        chain_finder.visit(tree)
        chain_map = chain_finder.map_res

        api_finder = _FragmentApiFinderV2(chain_map)
        api_finder.visit(tree)
        api_map = api_finder.map_res
        # TODO check number of api calls, should be exactly one.

        first_node = next(iter(api_map.keys()))
        output_desc = _parse_mark_outputs_ast(first_node)
        cache = _FragmentPrepCache(
            code_clean=code_for_compare,
            tree=tree,
            output_desc=output_desc,
            end_column=len(lines[-1]) + 1,
            num_lines=len(lines),
        )
        return cache

    def _cached_parse_ast(self, code: str) -> ast.Module:
        code_for_compare = "\n".join(clean_source_code(code.splitlines()))
        if self._prev_code == code_for_compare and self._prev_ast is not None:
            return self._prev_ast
        tree = ast.parse(code)
        self._prev_code = code_for_compare
        self._prev_ast = tree
        return tree

    def parse_fragment(self, node: ADVNodeModel, code: str, global_scope: dict[str, Any], cur_scope: dict[str, BackendHandle],
            ):
        node_id = node.id
        # tree = self._cached_parse_ast(code)
        # chain_finder = _ChainAttrFinder(global_scope)
        # chain_finder.visit(tree)
        # chain_map = chain_finder.map_res

        # api_finder = _FragmentApiFinder(chain_map)
        # api_finder.visit(tree)
        # api_map = api_finder.map_res
        # # TODO check number of api calls, should be exactly one.

        # first_node = next(iter(api_map.keys()))
        # output_desc = _parse_mark_outputs_ast(first_node)
        prep_cache = self._cached_parse_output_desc(code)
        tree = prep_cache.tree
        output_desc = prep_cache.output_desc
        local_scope = cur_scope.copy()
        symbol_extractor = _ExtractSymbolFromFragment()
        symbol_extractor.parse_block(tree.body, local_scope)
        name_to_sym = symbol_extractor._assigned_symbols

        output_handles: list[BackendHandle] = []
        for dict_key, (sym_name, alias) in output_desc.mapping.items():
            # TODO better error here.
            assert sym_name in cur_scope, \
                f"Output symbol {sym_name} not found in global scope"
            sym_handle = cur_scope[sym_name].copy()
            sym_handle.handle.source_node_id = node_id
            sym_handle.handle.source_handle_id = sym_handle.handle.id
            name = alias
            sym_handle.handle.name = name
            sym_handle.handle.symbol_name = alias
            if output_desc.type == "dict":
                sym_handle.handle.dict_key = dict_key
            output_handles.append(sym_handle)
        input_handles: list[BackendHandle] = []
        for sym_name, sym_handle in name_to_sym.items():
            sym_handle.target_node_handle_id.add((node_id, sym_handle.handle.id))
            sym_handle = sym_handle.copy(prefix=ADVHandlePrefix.Input)
            sym_handle.handle.is_input = True
            input_handles.append(sym_handle)
        # make order of input handles stable.
        input_handles.sort(key=lambda h: (h.handle.default is not None, h.index))
        if output_desc.type == "single":
            out_type_anno = output_handles[0].handle.type
        elif output_desc.type == "tuple":
            out_type_anno = f"tuple[{', '.join([h.handle.type for h in output_handles])}]"
        else:
            out_type_anno = f"dict[str, Any]"
        return FragmentParseResult(
            node=node,
            succeed=True,
            func_name=node.name,
            input_handles=input_handles,
            output_handles=output_handles,
            out_type=output_desc.type,
            out_type_anno=out_type_anno,
            end_column=prep_cache.end_column,
            num_lines=prep_cache.num_lines,

        )