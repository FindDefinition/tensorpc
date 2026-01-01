import ast
from dataclasses import is_dataclass
from functools import partial
import inspect
import dataclasses as dataclasses_plain
from collections.abc import Sequence
from typing import Any, Callable, ForwardRef, Optional, Type, Union, cast

import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core.annolib import (AnnotatedArg, AnnotatedType, DataclassType, T_dataclass,
                                   Undefined,
                                   child_type_generator_with_dataclass,
                                   is_undefined, parse_annotated_function,
                                   parse_type_may_optional_undefined,
                                   undefined)
from tensorpc.core.funcid import (clean_source_code,
                                  determine_code_common_indent,
                                  remove_common_indent_from_code)
from tensorpc.core.inspecttools import (findsource_by_lines,
                                        getsourcelinesby_lines,
                                        unwrap_fn_static_cls_property)
from tensorpc.core.moduleid import get_module_id_of_type, get_qualname_of_type
from tensorpc.core.pfl.constants import (PFL_COMPILE_META_ATTR,
                                         PFL_STDLIB_FUNC_META_ATTR)
from tensorpc.core.tree_id import UniqueTreeId

from .core import (BACKEND_CONFIG_REGISTRY, BASE_ANNO_TYPE_TO_PFLSTATIC_TYPE,
                   PFL_LOGGER, PFLCompileConstant, PFLCompileConstantType,
                   PFLCompileFuncMeta, PFLCompileReq, PFLErrorFormatContext,
                   PFLExprFuncArgInfo, PFLExprFuncInfo, PFLExprInfo,
                   PFLExprType, PFLInlineRunEnv, PFLMetaInferResult,
                   PFLParseConfig, PFLParseContext, PFLStdlibFuncMeta,
                   StaticEvalConfig, PFLProcessedVarMeta, enter_parse_context,
                   evaluate_annotation_expr, get_compilable_meta,
                   get_eval_cfg_in_parse_ctx, get_parse_cache_checked,
                   get_parse_context, get_parse_context_checked,
                   is_dcls_init_defined_by_user,
                   is_dcls_post_init_defined_by_user)
from .pfl_ast import (BinOpType, BoolOpType, CompareType, PFLAnnAssign, PFLArg,
                      PFLArray, PFLAssign, PFLAstNodeBase, PFLAstParseError,
                      PFLAstStmt, PFLASTType, PFLAttribute, PFLAugAssign,
                      PFLBinOp, PFLBoolOp, PFLBreak, PFLCall, PFLClass, PFLCompare,
                      PFLConstant, PFLContinue, PFLDict, PFLEvalError, PFLExpr,
                      PFLExprStmt, PFLFor, PFLFunc, PFLIf, PFLIfExp, PFLModule,
                      PFLName, PFLReturn, PFLSlice, PFLStaticVar, PFLSubscript,
                      PFLTreeNodeFinder, PFLTuple, PFLUnaryOp, PFLWhile,
                      UnaryOpType, iter_child_nodes, unparse_pfl_expr, walk)
from .pfl_reg import (ALL_COMPILE_TIME_FUNCS, STD_REGISTRY, StdRegistryItem,
                      compiler_isinstance, compiler_cast,
                      compiler_print_type, compiler_remove_optional)

_ALL_SUPPORTED_AST_TYPES = {
    ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.Name,
    ast.Constant, ast.Subscript, ast.Attribute, ast.List, ast.Dict, ast.Assign,
    ast.AugAssign, ast.If, ast.Expr, ast.IfExp, ast.For, ast.While,
    ast.AnnAssign, ast.Return, ast.FunctionDef
}

_AST_BINOP_TO_PFL_BINOP = {
    ast.Add: BinOpType.ADD,
    ast.Sub: BinOpType.SUB,
    ast.Mult: BinOpType.MULT,
    ast.Div: BinOpType.DIV,
    ast.FloorDiv: BinOpType.FLOOR_DIV,
    ast.Mod: BinOpType.MOD,
    ast.Pow: BinOpType.POW,
    ast.LShift: BinOpType.LSHIFT,
    ast.RShift: BinOpType.RSHIFT,
    ast.BitOr: BinOpType.BIT_OR,
    ast.BitXor: BinOpType.BIT_XOR,
    ast.BitAnd: BinOpType.BIT_AND,
    ast.MatMult: BinOpType.MATMUL,
}

_AST_UNARYOP_TO_PFL_UNARYOP = {
    ast.Invert: UnaryOpType.INVERT,
    ast.Not: UnaryOpType.NOT,
    ast.UAdd: UnaryOpType.UADD,
    ast.USub: UnaryOpType.USUB,
}

_AST_COMPARE_TO_PFL_COMPARE = {
    ast.Eq: CompareType.EQUAL,
    ast.NotEq: CompareType.NOT_EQUAL,
    ast.Lt: CompareType.LESS,
    ast.LtE: CompareType.LESS_EQUAL,
    ast.Gt: CompareType.GREATER,
    ast.GtE: CompareType.GREATER_EQUAL,
    ast.Is: CompareType.IS,
    ast.IsNot: CompareType.IS_NOT,
    ast.In: CompareType.IN,
    ast.NotIn: CompareType.NOT_IN,
}

_ALLOWED_SPEC_AST_TYPES = set([PFLASTType.ARRAY, PFLASTType.TUPLE])

class PFLLibrary:

    def __init__(self, modules: dict[str, PFLModule], check_empty: bool = True):
        all_compiled_units: dict[str, Union[PFLFunc, PFLClass]] = {}
        all_func_uid_to_specs: dict[str, list[PFLFunc]] = {}
        all_func_uid_to_cls_specs: dict[str, list[PFLClass]] = {}

        backend: str = ""
        for k, v in modules.items():
            all_units = v.get_all_compiled()
            for k1, v1 in all_units.items():
                k1_parts = UniqueTreeId(k1).parts
                if isinstance(v1, PFLFunc):
                    if backend == "":
                        backend = v1.backend
                    else:
                        assert v1.backend == backend, "all compiled units must have the same backend"
                    if k1_parts[0] not in all_func_uid_to_specs:
                        all_func_uid_to_specs[k1_parts[0]] = []
                    all_func_uid_to_specs[k1_parts[0]].append(v1)
                else:
                    k1_parts = UniqueTreeId(k1).parts
                    if k1_parts[0] not in all_func_uid_to_cls_specs:
                        all_func_uid_to_cls_specs[k1_parts[0]] = []
                    all_func_uid_to_cls_specs[k1_parts[0]].append(v1)
                all_compiled_units[k1] = v1
        if check_empty:
            assert modules and all_compiled_units
        self._stmt_finder: Optional[dict[str, PFLTreeNodeFinder]] = None
        self._modules = modules
        self._path_to_modules = {m.compile_info.path: m for m in modules.values()}
        assert len(self._path_to_modules) == len(modules), "module path duplicate, shouldn't happen"
        self._compiled_units = all_compiled_units
        self._func_uid_to_compiled_units = all_func_uid_to_specs
        self._func_uid_to_compiled_classes = all_func_uid_to_cls_specs
        self._backend = backend
        self._inline_env_cache: dict[str, PFLInlineRunEnv] = {}

    @staticmethod 
    def split_pfl_func_uid(uid: str):
        parts = UniqueTreeId(uid).parts
        return parts

    @staticmethod 
    def extract_fn_uid_nospec(uid: str):
        return PFLLibrary.split_pfl_func_uid(uid)[0]

    @staticmethod 
    def extract_fn_name(uid: str):
        parts = PFLLibrary.split_pfl_func_uid(uid)
        qname = parts[0]
        return ".".join(qname.split("::")[1:])

    def _cached_get_stmt_finder(self):
        if self._stmt_finder is not None:
            return self._stmt_finder
        self._stmt_finder = {}
        for k, v in self._modules.items():
            self._stmt_finder[v.compile_info.path] = PFLTreeNodeFinder(v, (PFLAstStmt, ))
        return self._stmt_finder

    def dump_to_json_dict(self):
        res = {}
        for k, v in self._compiled_units.items():
            res[k] = pfl_ast_to_dict(v)
        return res

    def get_compiled_unit_specs(self, key: Union[str,
                                                 Callable]) -> list[PFLFunc]:
        if not isinstance(key, str):
            key = get_module_id_of_type(key)
        if key not in self._func_uid_to_compiled_units:
            raise KeyError(
                f"Function UID {key} not found in compiled units., available: {list(self._func_uid_to_compiled_units.keys())}"
            )
        return self._func_uid_to_compiled_units[key]

    def get_compiled_unit_inline_env(
            self, key: Union[str, Callable], kwargs: Optional[dict[str, Any]] = None) -> PFLInlineRunEnv:
        if not isinstance(key, str):
            key = get_module_id_of_type(key)
        if kwargs is None and key in self._inline_env_cache:
            return self._inline_env_cache[key]
        pfl_fns = self._func_uid_to_compiled_units[key]
        assert len(
            pfl_fns
        ) == 1, "get_compiled_unit_inline_env only supports normal function compiled unit, template isn't supported."
        pfl_fn = pfl_fns[0]
        fn_meta = pfl_fn.compile_info.meta
        assert fn_meta is not None and fn_meta.inline_run_env_fn is not None
        if kwargs is not None:
            inline_env = fn_meta.inline_run_env_fn(**kwargs)
        else:
            inline_env = fn_meta.inline_run_env_fn()
        assert isinstance(
            inline_env,
            PFLInlineRunEnv), "inline run env must be PFLInlineRunEnv"
        if kwargs is None:
            self._inline_env_cache[key] = inline_env
        return inline_env

    def get_module_by_func(self, key: Callable) -> PFLModule:
        key_str = get_module_id_of_type(key)
        module_key = key_str.split("::")[0]
        # print(module_key, self._modules.keys())
        return self._modules[module_key]

    def get_module_by_func_uid(self, compile_uid: str) -> PFLModule:
        parts = UniqueTreeId(compile_uid).parts
        module_key = parts[0].split("::")[0]
        # print(module_key, self._modules.keys())
        return self._modules[module_key]

    def find_stmt_by_path_lineno(self, module_path: str, lineno: int):
        finders = self._cached_get_stmt_finder()
        finder = finders[module_path]
        return finder.find_nearest_node_by_line(lineno)

    @property
    def modules(self) -> dict[str, PFLModule]:
        return self._modules

    @property
    def all_compiled(self) -> dict[str, Union[PFLFunc, PFLClass]]:
        return self._compiled_units

    @property
    def all_func_uid_to_compiled(self) -> dict[str, list[PFLFunc]]:
        return self._func_uid_to_compiled_units

    @property
    def backend(self) -> str:
        return self._backend

    def get_compiled_func_by_uid(self, uid: str) -> PFLFunc:
        res = self._compiled_units[uid]
        assert isinstance(res, PFLFunc)
        return res

    def get_compiled_cls_by_uid(self, uid: str) -> PFLClass:
        res = self._compiled_units[uid]
        assert isinstance(res, PFLClass)
        return res

@dataclasses.dataclass
class ReturnInfo:
    complete: bool
    all_return_stmts: list[PFLReturn]

_SUPPORTED_PYTHON_BUILTINS = {
    "int": int,
    "float": float,
    "bool": bool,
    "range": range,
    "list": list,
    "dict": dict,
    "print": print,
    "len": len,
    "min": min,
    "max": max,
    "abs": abs,
    "str": str,
    "isinstance": isinstance,

}

def default_pfl_var_proc(val: Any):
    is_static = False 
    is_classm = False 
    is_property = False

    if isinstance(val, (classmethod, staticmethod)):
        res_fn = val.__func__
        is_static = isinstance(val, staticmethod)
        is_classm = isinstance(val, classmethod)
    elif isinstance(val, property):
        assert val.fget is not None 
        res_fn =  val.fget
        is_property = True
    else:
        res_fn = val

    if inspect.isfunction(res_fn):
        res_fn = inspect.unwrap(res_fn)
        fn_meta = get_compilable_meta(res_fn)
        return PFLProcessedVarMeta(value=res_fn, 
            is_static=is_static,
            is_classmethod=is_classm,
            is_property=is_property,
            compilable_meta=fn_meta)
    return PFLProcessedVarMeta(value=res_fn)

class MapAstNodeToConstant(ast.NodeVisitor):
    """map every ast name/attribute to a stdlib item.
    we don't map function args (annotations) because they will be 
    parsed after anno evaluation.
    """

    def __init__(self,
                 func_globals: dict[str, Any],
                 global_or_nonlocal_names: set[str],
                 error_ctx: PFLErrorFormatContext,
                 backend: str = "js",
                 var_preproc: Callable[[Any], PFLProcessedVarMeta] = default_pfl_var_proc):
        super().__init__()
        self.func_globals = {**func_globals}
        self._global_or_nonlocal_names = global_or_nonlocal_names
        for k, v in _SUPPORTED_PYTHON_BUILTINS.items():
            if k not in func_globals:
                self.func_globals[k] = v
        self.backend = backend
        self.var_preproc = var_preproc
        self._node_to_compile_constant: dict[ast.AST, PFLCompileConstant] = {}

        self.error_ctx = error_ctx

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # visit func node except decorators
        # compiled decorators isn't supported.
        self.visit(node.args)
        for stmt in node.body:
            self.visit(stmt)
        # if node.returns is not None:
        #     self.visit(node.returns)
        if hasattr(node, "type_params"):
            for tp in getattr(node, "type_params"):
                self.visit(tp)

    def visit_arg(self, node: ast.arg):
        # avoid visit arg annotation
        pass 

    def visit_AnnAssign(self, node: ast.AnnAssign):
        # visit annassign without annotation
        self.visit(node.target)
        if node.value is not None:
            self.visit(node.value)

    def _extract_attr_chain(self, node: ast.AST):
        parts: list[str] = []
        parts_node: list[ast.AST] = []
        cur_node = node
        name_found = False
        is_external_name: bool = False
        while isinstance(cur_node, (ast.Attribute, ast.Name)):
            if isinstance(cur_node, ast.Attribute):
                parts.append(cur_node.attr)
                parts_node.append(cur_node)
                cur_node = cur_node.value
            else:
                parts.append(cur_node.id)
                parts_node.append(cur_node)
                name_found = True
                is_external_name = cur_node.id in self._global_or_nonlocal_names or cur_node.id in _SUPPORTED_PYTHON_BUILTINS
                break
        return parts, parts_node, name_found, is_external_name

    def _visit_Attribute_or_name(self, node: Union[ast.Attribute, ast.Name]):
        # TODO block nesetd function def support
        parts, parts_node, name_found, is_external_name = self._extract_attr_chain(
            node)
        
        if not name_found or not is_external_name:
            return self.generic_visit(node)

        parts = parts[::-1]
        parts_node = parts_node[::-1]
        cur_obj = self.func_globals
        has_std_parent = False
        for part in parts:
            if isinstance(cur_obj, dict):
                if part in cur_obj:
                    cur_obj = cur_obj[part]
                else:
                    return self.generic_visit(node)
            else:
                cur_obj_is_cls_or_fn = inspect.isclass(cur_obj) or inspect.isfunction(cur_obj) or inspect.ismodule(cur_obj)
                if not has_std_parent and cur_obj_is_cls_or_fn and (
                        cur_obj,
                        self.backend) in STD_REGISTRY._type_backend_to_item:
                    has_std_parent = True
                if hasattr(cur_obj, part):
                    cur_obj = getattr(cur_obj, part)
                else:
                    return self.generic_visit(node)
        cur_obj_is_cls_or_fn = inspect.isclass(cur_obj) or inspect.isfunction(cur_obj) or inspect.ismodule(cur_obj)

        cur_is_std = cur_obj_is_cls_or_fn and (cur_obj,
                      self.backend) in STD_REGISTRY._type_backend_to_item
        if has_std_parent and not cur_is_std:
            return self.generic_visit(node)
        res_constant = self.preproc_var_to_constant(cur_obj)
        if res_constant is None:
            return self.generic_visit(node)
        self._node_to_compile_constant[node] = res_constant
        return node

    def preproc_var_to_constant(self, cur_obj: Any):
        bound_self = None
        if inspect.ismethod(cur_obj):
            # method will never be stdlib.
            const_type = PFLCompileConstantType.FUNCTION
            bound_self = cur_obj.__self__
            cur_obj = cur_obj.__func__
        # if self.var_preproc is not None:
        # used to handle custom wrappers. e.g. triton.jit or triton.aggregate
        # for triton.aggregate, we can convert it to standard dataclass 
        prep_res = self.var_preproc(cur_obj)
        cur_obj = prep_res.value 
        if cur_obj is partial:
            return None

        is_stdlib = False
        # TODO detect classmethod and reject it.
        # if isinstance(cur_obj, staticmethod):
        #     cur_obj = cur_obj.__func__
        if inspect.isfunction(cur_obj) or isinstance(
                cur_obj, type) or inspect.ismodule(cur_obj) or inspect.isclass(
                    cur_obj) or inspect.isbuiltin(cur_obj):
            # check is stdlib.
            item = STD_REGISTRY.get_item_by_dcls(cur_obj, self.backend)
            if item is not None:
                dcls = item.dcls
                if dataclasses.is_dataclass(dcls):
                    const_type = PFLCompileConstantType.DATACLASS_TYPE
                else:
                    const_type = PFLCompileConstantType.FUNCTION
                is_stdlib = True
                cur_obj = item
            else:
                # TODO support class
                # TODO should we do func unwrapping here?
                # here we only support function marked with pfl (may be staticmethod).
                assert inspect.isfunction(cur_obj) or (
                    inspect.isclass(cur_obj)
                    and dataclasses.is_dataclass(cur_obj)
                ), f"only function/dcls can be compiled, {type(cur_obj)}, {cur_obj}"
                if inspect.isclass(cur_obj) and dataclasses.is_dataclass(
                        cur_obj):
                    const_type = PFLCompileConstantType.DATACLASS_TYPE
                else:
                    const_type = PFLCompileConstantType.FUNCTION
                # cur_obj = self.func_unwrapper(cur_obj)
        else:
            # builtin types and tuple of them.
            # validate by PFLConstant.
            PFLExprInfo.from_value(cur_obj)
            const_type = PFLCompileConstantType.BUILTIN_VALUE
        return PFLCompileConstant(
            value=cur_obj,
            type=const_type,
            is_stdlib=is_stdlib,
            bound_self=bound_self,
            var_meta=prep_res)


    def visit_Name(self, node: ast.Name):
        return self._visit_Attribute_or_name(node)

    def visit_Attribute(self, node: ast.Attribute):
        return self._visit_Attribute_or_name(node)


def _get_module_code_path_by_fn(func: Callable):
    mod = inspect.getmodule(func)
    assert mod is not None, "module_code_path_getter must be provided if func isn't a module function"
    module_code = inspect.getsource(mod)
    return module_code, inspect.getabsfile(mod)

@dataclasses_plain.dataclass
class _CompileFuncCache:
    tree: ast.FunctionDef
    module_code: str
    module_path: str

    module_code_lines: list[str]
    code: str 
    first_lineno: int
    func_lines: list[str]


class PFLParser:

    def __init__(self,
                 backend: str = "js",
                 parse_cfg: Optional[PFLParseConfig] = None,
                 func_code_getter: Optional[Callable[[Any],
                                                     tuple[list[str],
                                                           int]]] = None,
                 module_code_path_getter: Optional[Callable[[Any], tuple[str, str]]] = None,
                 var_preproc: Optional[Callable[[Any], Any]] = None,
                anno_transform: Optional[Callable[[PFLExprInfo, Any, Union[Undefined, Any]],
                                PFLExprInfo]] = None):
        self._backend = backend
        if parse_cfg is None:
            assert backend in BACKEND_CONFIG_REGISTRY, "you must register backend config first if parse_cfg isn't provided."
            parse_cfg = BACKEND_CONFIG_REGISTRY[backend]
        self._parse_cfg = parse_cfg
        if func_code_getter is None:
            func_code_getter = inspect.getsourcelines
        if module_code_path_getter is None:
            module_code_path_getter = _get_module_code_path_by_fn
        if var_preproc is None:
            var_preproc = default_pfl_var_proc
        self._var_preproc = var_preproc
        self._func_code_getter = func_code_getter
        self._module_code_path_getter = module_code_path_getter
        # anno_transform: (infered_anno, original_anno) -> new_anno
        # used for convert third-party annotations such as tl.constexpr in triton.jit
        # also user can determine constexpr data based on annotation.
        self._anno_transform = anno_transform

        self._all_compiled: dict[str, Union[PFLFunc, PFLClass]] = {}
        self._current_compiling: set[str] = set()

        self.func_node_to_meta: dict[ast.AST, PFLCompileFuncMeta] = {}

        self._cache_fn_precompile_info: dict[Callable, _CompileFuncCache] = {}

    def _parse_expr_to_pfl_notype(self, expr: ast.expr) -> PFLExpr:
        """Parse an expression to PFLExpr without type inference.
        Most features are disabled. currently only used to replace
        jmespath.
        """
        raise NotImplementedError

    def _parse_expr_to_pfl(self, expr: ast.expr,
                              scope: dict[str, PFLExprInfo]) -> PFLExpr:
        source_loc = (expr.lineno, expr.col_offset, expr.end_lineno,
                      expr.end_col_offset)
        try:
            ctx = get_parse_context_checked()
            parse_cache = get_parse_cache_checked()
            if isinstance(expr, ast.Name) or isinstance(expr, ast.Attribute):
                if expr in ctx.node_to_constants:
                    item = ctx.node_to_constants[expr]
                    if item.is_stdlib:
                        assert isinstance(
                            item.value, StdRegistryItem
                        ), "constant value must be StdRegistryItem"
                        new_name = item.value.mapped_name
                        st = ctx.cache.cached_parse_std_item(item.value)
                        res = PFLName(PFLASTType.NAME,
                                      source_loc,
                                      id=new_name,
                                      st=dataclasses.replace(st))
                    elif item.type == PFLCompileConstantType.BUILTIN_VALUE:
                        res = PFLConstant(PFLASTType.CONSTANT,
                                          source_loc,
                                          value=item.value)
                    elif item.type == PFLCompileConstantType.FUNCTION:
                        # TODO currently we assume user never use Class.method(obj_self, ...)
                        # to call a method, so this item won't be method.
                        self_type = None
                        self_annotype = None
                        if item.bound_self is not None:
                            self_annotype = parse_type_may_optional_undefined(
                                item.bound_self)

                            self_type = PFLExprInfo.from_annotype(
                                self_annotype)
                        meta: Optional[
                            PFLCompileFuncMeta] = item.var_meta.compilable_meta
                        new_st = parse_cache.cached_parse_func(item.value,
                                            is_bound_method=item.bound_self
                                            is not None,
                                            self_type=self_annotype,
                                            ext_preproc_res=item.var_meta)
                        new_finfo = new_st.get_func_info_checked()

                        if meta is None or not meta.need_delayed_processing(
                        ):
                            creq = ctx.enqueue_func_compile(
                                item.value,
                                new_finfo,
                                is_method_def=False,
                                self_type=self_type,
                                is_prop=False,
                                bound_self=item.bound_self)

                            res = PFLName(PFLASTType.NAME,
                                          source_loc,
                                          id=new_finfo.func_uid,
                                          st=new_st)
                            res.st.compiled_uid = creq.get_func_compile_uid()
                        else:
                            req = ctx.get_compile_req(
                                item.value,
                                new_finfo,
                                meta,
                                is_method_def=False,
                                self_type=self_type,
                                is_prop=False,
                                bound_self=item.bound_self)
                            new_st.delayed_compile_req = req
                            res = PFLName(PFLASTType.NAME,
                                          source_loc,
                                          id=req.uid,
                                          st=new_st)
                        # make all function object constexpr.
                        new_st._constexpr_data = item.value
                    elif item.type == PFLCompileConstantType.DATACLASS_TYPE:
                        meta = item.var_meta.compilable_meta
                        if meta is None or not meta.need_delayed_processing(
                        ):
                            new_st = parse_cache.cached_parse_dcls(item.value)
                            new_finfo = new_st.get_dcls_info_checked()
                            # set self_type for init and post_init
                            creq = ctx.enqueue_dcls_compile(
                                item.value,
                                new_finfo,
                                self_type=new_st)
                            res = PFLName(PFLASTType.NAME,
                                          source_loc,
                                          id=new_finfo.func_uid,
                                          st=new_st)
                            res.st.compiled_uid = creq.get_func_compile_uid()
                        else:
                            new_st = PFLExprInfo.from_dcls_type(item.value, delay_parse_field=True)
                            new_finfo = new_st.get_dcls_info_checked()
                            req = ctx.get_compile_req(
                                item.value,
                                new_finfo,
                                meta,
                                is_dcls=True,
                                self_type=new_st)
                            new_st.delayed_compile_req = req
                            res = PFLName(PFLASTType.NAME,
                                          source_loc,
                                          id=req.uid,
                                          st=new_st)
                    else:
                        raise NotImplementedError
                else:
                    if isinstance(expr, ast.Name):
                        if expr.id not in scope:
                            if not ctx.cfg.allow_partial_type_infer:
                                raise PFLAstParseError(f"undefined name {expr.id}",
                                                    expr)
                            else:
                                st = PFLExprInfo(PFLExprType.UNKNOWN)
                        else:
                            st = scope[expr.id]
                        res = PFLName(PFLASTType.NAME,
                                      source_loc,
                                      id=expr.id,
                                      st=dataclasses.replace(st))
                    else:
                        value = self._parse_expr_to_pfl(expr.value, scope)
                        attr = expr.attr
                        st = value.st
                        res = PFLAttribute(PFLASTType.ATTR,
                                           source_loc,
                                           value=value,
                                           attr=attr,
                                           st=st)
                res.check_and_infer_type()

            elif isinstance(expr, ast.Constant):
                res = PFLConstant(PFLASTType.CONSTANT,
                                  source_loc,
                                  value=expr.value)
                res.check_and_infer_type()
            elif isinstance(expr, ast.Slice):
                assert get_parse_context_checked(
                ).cfg.allow_slice, "slice is disabled in config"
                lo = self._parse_expr_to_pfl(
                    expr.lower, scope) if expr.lower is not None else undefined
                hi = self._parse_expr_to_pfl(
                    expr.upper, scope) if expr.upper is not None else undefined
                step = self._parse_expr_to_pfl(
                    expr.step, scope) if expr.step is not None else undefined
                res = PFLSlice(PFLASTType.SLICE,
                               source_loc,
                               lo=lo,
                               hi=hi,
                               step=step)
                res.check_and_infer_type()
            elif isinstance(expr, ast.Subscript):
                value = self._parse_expr_to_pfl(expr.value, scope)
                slice: Union[Sequence[PFLExpr], PFLExpr]
                if isinstance(expr.slice, ast.Tuple):
                    assert get_parse_context_checked(
                    ).cfg.allow_nd_slice, "nd slice is disabled in config"
                    slice = []
                    for item in expr.slice.elts:
                        slice.append(self._parse_expr_to_pfl(item, scope))
                else:
                    slice = self._parse_expr_to_pfl(expr.slice, scope)
                is_store = undefined
                if isinstance(expr.ctx, ast.Store):
                    is_store = True
                res = PFLSubscript(PFLASTType.SUBSCRIPT,
                                   source_loc,
                                   value=value,
                                   slice=slice,
                                   is_store=is_store)
                res.check_and_infer_type()
            elif isinstance(expr, ast.List):
                elts = [
                    self._parse_expr_to_pfl(elt, scope) for elt in expr.elts
                ]
                res = PFLArray(PFLASTType.ARRAY, source_loc, elts=elts)
                res.check_and_infer_type()
            elif isinstance(expr, ast.Dict):
                keys = [
                    self._parse_expr_to_pfl(key, scope)
                    if key is not None else None for key in expr.keys
                ]
                values = [
                    self._parse_expr_to_pfl(value, scope)
                    for value in expr.values
                ]
                res = PFLDict(PFLASTType.DICT,
                              source_loc,
                              keys=keys,
                              values=values)
                res.check_and_infer_type()
            elif isinstance(expr, ast.Tuple):
                elts = [
                    self._parse_expr_to_pfl(elt, scope) for elt in expr.elts
                ]
                res = PFLTuple(PFLASTType.TUPLE, source_loc, elts=elts)
                res.check_and_infer_type()
            elif isinstance(expr, ast.BoolOp):
                op = BoolOpType.AND if isinstance(expr.op,
                                                  ast.And) else BoolOpType.OR
                values = [
                    self._parse_expr_to_pfl(value, scope)
                    for value in expr.values
                ]
                res = PFLBoolOp(PFLASTType.BOOL_OP,
                                source_loc,
                                op=op,
                                values=values)
                res.check_and_infer_type()
            elif isinstance(expr, ast.BinOp):
                op = _AST_BINOP_TO_PFL_BINOP[type(expr.op)]
                left = self._parse_expr_to_pfl(expr.left, scope)
                right = self._parse_expr_to_pfl(expr.right, scope)
                res = PFLBinOp(PFLASTType.BIN_OP,
                               source_loc,
                               op=op,
                               left=left,
                               right=right)
                res.check_and_infer_type()
            elif isinstance(expr, ast.UnaryOp):
                op = _AST_UNARYOP_TO_PFL_UNARYOP[type(expr.op)]
                operand = self._parse_expr_to_pfl(expr.operand, scope)
                res = PFLUnaryOp(PFLASTType.UNARY_OP,
                                 source_loc,
                                 op=op,
                                 operand=operand)
                res.check_and_infer_type()
            elif isinstance(expr, ast.Compare):
                left = self._parse_expr_to_pfl(expr.left, scope)
                assert len(expr.ops) == 1
                op = _AST_COMPARE_TO_PFL_COMPARE[type(expr.ops[0])]
                assert len(expr.comparators) == 1
                right = self._parse_expr_to_pfl(expr.comparators[0], scope)
                res = PFLCompare(PFLASTType.COMPARISON,
                                 source_loc,
                                 op=op,
                                 left=left,
                                 right=right)
                res.check_and_infer_type()
            elif isinstance(expr, ast.Call):
                ctx = get_parse_context_checked()
                # TODO support template func here.
                # we need to assign a unique id for each specialized function
                func = self._parse_expr_to_pfl(expr.func, scope)
                if ctx.cfg.allow_partial_type_infer:
                    assert func.st.type != PFLExprType.UNKNOWN, f"function \"{unparse_pfl_expr(func)}\" in call can't be unknown in partial type infer."
                raw_func = None
                if func.st.func_info is not None:
                    raw_func = func.st.func_info.raw_func
                if raw_func is compiler_cast:
                    arg_val = self._parse_expr_to_pfl(expr.args[1], scope)
                    assert len(expr.args) == 2
                    anno_in_ast = evaluate_annotation_expr(expr.args[0])
                    assert anno_in_ast is not None, "annotation must be evaluated to a valid type"
                    if self._anno_transform is not None:
                        anno_st = self._anno_transform(
                            arg_val.st, anno_in_ast, undefined)
                    else:
                        anno_st = PFLExprInfo.from_annotype(
                            parse_type_may_optional_undefined(anno_in_ast))
                    res = dataclasses.replace(
                        arg_val,
                        st=anno_st)
                else:
                    args = [
                        self._parse_expr_to_pfl(arg, scope) for arg in expr.args
                    ]
                    kw_keys: list[str] = []
                    vals: list[PFLExpr] = []
                    for arg in expr.args:
                        assert not isinstance(
                            arg, ast.Starred), "don't support *arg for now"
                    kw_sts: dict[str, PFLExprInfo] = {}
                    for kw in expr.keywords:
                        assert kw.arg is not None, "don't support **kw"
                        kw_keys.append(kw.arg)
                        val_expr = self._parse_expr_to_pfl(kw.value, scope)
                        vals.append(val_expr)
                        kw_sts[kw.arg] = val_expr.st
                    arg_infos_from_call = ([a.st for a in args], kw_sts)
                    if func.st.delayed_compile_req is not None:
                        req = func.st.delayed_compile_req
                        assert isinstance(req, PFLCompileReq)
                        req = dataclasses.replace(
                            req,
                            args_from_call=arg_infos_from_call)
                        # only template and inline function allow inline expand
                        # print(req.get_func_compile_uid())
                        PFL_LOGGER.warning("%s",
                                        str(func.st.delayed_compile_req))
                        compiled_func = self.parse_compile_req_to_pfl_ast(
                            req, allow_inline_expand=True)
                        func.st = dataclasses.replace(compiled_func.st)
                    # check is compile-time function
                    if raw_func in ALL_COMPILE_TIME_FUNCS:
                        if raw_func is compiler_print_type:
                            assert len(args) == 1 and len(
                                kw_keys
                            ) == 0, "compiler_print_type only support one argument"
                            args_str = ", ".join(str(a.st) for a in args)
                            PFL_LOGGER.warning(args_str)
                            res = args[0]
                            expr = expr.args[0]
                        elif raw_func is compiler_isinstance:
                            # TODO currently int and float are treated as function.
                            if not ctx.cfg.allow_isinstance:
                                raise PFLAstParseError(
                                    "isinstance is disabled in config, you need to enable it in parse config.",
                                    expr)
                            type_to_check = args[0].st
                            type_candidates_expr = args[1]
                            if type_candidates_expr.st.type == PFLExprType.TUPLE:
                                type_candidates = type_candidates_expr.st.childs
                                assert len(
                                    type_candidates
                                ) > 0, "type_candidates must not be empty"
                            else:
                                type_candidates = [type_candidates_expr.st]
                            compare_res: list[bool] = []
                            # TODO better compare
                            if type_to_check.proxy_dcls is not None:
                                for c in type_candidates:
                                    if c.proxy_dcls is not None:
                                        compare_res.append(
                                            issubclass(type_to_check.proxy_dcls,
                                                    c.proxy_dcls))
                                    else:
                                        compare_res.append(False)
                            else:
                                for c in type_candidates:
                                    assert c.type == PFLExprType.DATACLASS_TYPE
                                    if type_to_check.type == PFLExprType.DATACLASS_OBJECT:
                                        compare_res.append(
                                            issubclass(
                                                type_to_check.
                                                get_origin_type_checked(),
                                                c.get_origin_type_checked()))
                                    else:
                                        compare_res.append(False)
                            res = PFLConstant(PFLASTType.CONSTANT,
                                            source_loc,
                                            value=any(compare_res))
                            res.check_and_infer_type()
                        elif raw_func is compiler_remove_optional:
                            assert len(args) == 1
                            res_st = dataclasses.replace(args[0].st).get_optional_undefined_removed()
                            res = dataclasses.replace(
                                args[0],
                                st=res_st)
                        else:
                            raise NotImplementedError(
                                f"compile-time function {raw_func} not implemented"
                            )
                    else:
                        parse_cfg = get_parse_context_checked().cfg
                        if not parse_cfg.allow_kw:
                            assert not expr.keywords, f"kwargs is disabled, you need to enable it in parse config."
                        res = PFLCall(PFLASTType.CALL,
                                    source_loc,
                                    func=func,
                                    args=args,
                                    keys=kw_keys if kw_keys else undefined,
                                    vals=vals if vals else undefined)
                        match_res, overload_info = res.check_and_infer_type_with_overload()
                        # handle template fn spec
                        if overload_info.has_template_fn_spec:
                            for arg_info, arg_expr in match_res.args:
                                if arg_info.template_fn_arg_spec_idx is not None:
                                    assert not is_undefined(arg_expr), "template fn spec can't use default."
                                    _, arg_spec_expr_info = match_res.args[arg_info.template_fn_arg_spec_idx]
                                    assert not is_undefined(arg_spec_expr_info), "arg spec can't use default."
                                    assert isinstance(arg_spec_expr_info, (PFLArray, PFLTuple)), "arg spec must be array or tuple"
                                    # for fn spec, we only support PFLName or list/tuple of PFLName.
                                    if isinstance(arg_expr, PFLName):
                                        elts = [arg_expr]
                                    else:
                                        assert isinstance(arg_expr, (PFLArray, PFLTuple)), "arg expr must be name, array or tuple"
                                        elts = arg_expr.elts
                                    # compile template func via args in arg spec
                                    for elt in elts:
                                        assert elt.st.type == PFLExprType.FUNCTION
                                        req = elt.st.delayed_compile_req
                                        assert isinstance(req, PFLCompileReq), f"fn {elt.st} don't have compile req (not template)"
                                        req = dataclasses.replace(
                                            req,
                                            args_from_call=((x.st for x in arg_spec_expr_info.elts), {}))
                                        compiled_func = self.parse_compile_req_to_pfl_ast(
                                            req, allow_inline_expand=True)
                                        elt.st = dataclasses.replace(compiled_func.st)
            elif isinstance(expr, ast.IfExp):
                res = PFLIfExp(
                    PFLASTType.IF_EXP,
                    source_loc,
                    test=self._parse_expr_to_pfl(expr.test, scope),
                    body=self._parse_expr_to_pfl(expr.body, scope),
                    orelse=self._parse_expr_to_pfl(expr.orelse, scope))
                res.check_and_infer_type()
            else:
                raise PFLAstParseError(f"not support {type(expr)}", expr)
            if isinstance(res, (PFLName, PFLAttribute, PFLSubscript)):
                assert isinstance(
                    expr, (ast.Name, ast.Attribute, ast.Subscript, ast.Call)
                ), f"expr must be Name, Attribute or Subscript, got {type(expr)}"
                if isinstance(expr, (ast.Name, ast.Attribute, ast.Subscript)) and isinstance(expr.ctx, ast.Store):
                    res.is_store = True
        except PFLAstParseError:
            raise
        except BaseException as e:
            raise PFLAstParseError(f"Unknown error {e}", expr) from e
        return res

    def _get_plain_attribute_chain_name(self, expr: PFLExpr) -> Optional[PFLName]:
        if isinstance(expr, PFLAttribute):
            return self._get_plain_attribute_chain_name(expr.value)
        elif isinstance(expr, PFLName):
            return expr
        else:
            return None

    def _if_test_optional_removal(self, expr: PFLExpr) -> dict[str, Union[PFLAttribute, PFLName]]:
        # TODO better match
        res: dict[str, Union[PFLAttribute, PFLName]] = {}
        if isinstance(expr, PFLBoolOp):
            if expr.op == BoolOpType.AND:
                for value in expr.values:
                    res.update(self._if_test_optional_removal(value)) 
        elif isinstance(expr, PFLCompare):
            if expr.op == CompareType.IS_NOT:
                if isinstance(expr.right, PFLConstant) and expr.right.value is None:
                    const_node = expr.right
                    tgt_node = expr.left
                elif isinstance(expr.left, PFLConstant) and expr.left.value is None:
                    const_node = expr.left
                    tgt_node = expr.right
                else:
                    const_node = None 
                    tgt_node = None
                if const_node is not None and tgt_node is not None:
                    nested_name = self._get_plain_attribute_chain_name(tgt_node)
                    if nested_name is not None:
                        attr_expr = tgt_node
                        assert isinstance(attr_expr, (PFLAttribute, PFLName))
                        res[nested_name.id] = attr_expr
        return res 

    def _parse_block_to_pfl_ast(
            self, body: list[ast.stmt],
            scope: dict[str,
                        PFLExprInfo]) -> tuple[list[PFLAstStmt], ReturnInfo]:
        # TODO add return type support
        block: list[PFLAstStmt] = []
        # block = PFLFunc(PFLASTType.BLOCK, -1, -1, "", [], [])
        return_info = ReturnInfo(complete=False, all_return_stmts=[])
        req = get_parse_context_checked().compile_req
        for stmt in body:
            source_loc = (stmt.lineno, stmt.col_offset, stmt.end_lineno,
                          stmt.end_col_offset)
            try:
                if not isinstance(stmt, tuple(_ALL_SUPPORTED_AST_TYPES)):
                    raise PFLAstParseError(f"not support {type(stmt)}", stmt)
                if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                    if isinstance(stmt, ast.AnnAssign):
                        assert stmt.simple == 1, "only support simple ann assign"
                    if isinstance(stmt, ast.Assign):
                        if len(stmt.targets) != 1:
                            raise PFLAstParseError(
                                "only support single assign", stmt)
                    if stmt.value is not None:
                        value = self._parse_expr_to_pfl(stmt.value, scope)
                    else:
                        value = None
                    if value is not None and req is not None and req.dcls_infer_field_type:
                        attr_target: Optional[ast.Attribute] = None
                        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Attribute):
                            attr_target = stmt.target

                        elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                            tgt_temp = stmt.targets[0]
                            if isinstance(tgt_temp, ast.Attribute):
                                attr_target = tgt_temp

                        if attr_target is not None and isinstance(attr_target.value, ast.Name):
                            val_in_scope = scope[attr_target.value.id]
                            if val_in_scope.type == PFLExprType.DATACLASS_OBJECT:
                                dcls_info = val_in_scope.get_dcls_info_checked()
                                dcls_info.delayed_init_set_field_type(attr_target.attr, value.st)

                    anno_st: Optional[PFLExprInfo] = None
                    if isinstance(stmt, ast.AnnAssign):
                        assert value is not None 
                        anno_in_ast = evaluate_annotation_expr(stmt.annotation)
                        assert anno_in_ast is not None, "annotation must be evaluated to a valid type"
                        if self._anno_transform is not None:
                            anno_st = self._anno_transform(
                                value.st, anno_in_ast, undefined)
                        else:
                            anno_st = PFLExprInfo.from_annotype(
                                parse_type_may_optional_undefined(anno_in_ast))
                        # TODO better unknown type handling
                        # a: list[int] = [], value type is list[unknown], so we always assign value.st to anno_st
                        # TODO may need to check if value.st is compatible with anno_st
                        value.st = dataclasses.replace(anno_st)
                    if isinstance(stmt, ast.Assign):
                        tgt = stmt.targets[0]
                    else:
                        tgt = stmt.target
                    if isinstance(tgt, ast.Name) and value is not None:
                        ctx = get_parse_context_checked()

                        is_new_var = False
                        if tgt.id not in scope:
                            is_new_var = True
                        elif not ctx.cfg.allow_var_type_override:
                            value.st.check_convertable(scope[tgt.id],
                                                       "assign value")
                        scope[tgt.id] = value.st
                        target = self._parse_expr_to_pfl(tgt, scope)

                        assert isinstance(target, PFLName)
                        target.is_new = is_new_var
                    elif isinstance(tgt, ast.Tuple) and value is not None:
                        ctx = get_parse_context_checked()
                        assert value.st.type == PFLExprType.TUPLE, "value type must be tuple"
                        assert len(tgt.elts) == len(
                            value.st.childs), "tuple length must be same"
                        is_new_vars: list[bool] = []
                        for i, elt in enumerate(tgt.elts):
                            is_new_var = False
                            assert isinstance(
                                elt,
                                ast.Name), "assign tuple item must be Name"
                            if elt.id not in scope:
                                is_new_var = True
                            elif not ctx.cfg.allow_var_type_override:
                                value.st.childs[i].check_convertable(
                                    scope[elt.id], "assign value")
                            scope[elt.id] = value.st.childs[i]
                            is_new_vars.append(is_new_var)
                        target = self._parse_expr_to_pfl(tgt, scope)
                        assert isinstance(target, PFLTuple)
                        if ctx.cfg.tuple_assign_must_be_homogeneous:
                            assert all(is_new_vars) or not any(is_new_vars), \
                                "tuple assign must be homogeneous, all new or all old"
                        for i, elt in enumerate(tgt.elts):
                            tgt_pfl_name = target.elts[i]
                            assert isinstance(tgt_pfl_name, PFLName)
                            tgt_pfl_name.is_new = is_new_vars[i]
                    else:
                        target = self._parse_expr_to_pfl(tgt, scope)

                    if isinstance(stmt, ast.Assign):
                        assert value is not None
                        target.st = dataclasses.replace(value.st)
                        node = PFLAssign(PFLASTType.ASSIGN,
                                         source_loc,
                                         target=target,
                                         value=value)
                        node.check_and_infer_type()
                        block.append(node)
                    else:
                        assert anno_st is not None
                        target.st = anno_st
                        node = PFLAnnAssign(PFLASTType.ANN_ASSIGN,
                                            source_loc,
                                            target=target,
                                            annotation=ast.unparse(
                                                stmt.annotation),
                                            value=value)
                        node.check_and_infer_type()
                        block.append(node)
                    # if isinstance(target, PFLAttribute) and value is not None and req is not None and req.dcls_infer_field_type:
                    #     maybe_template_dcls = target.value
                    #     if maybe_template_dcls.st.type == PFLExprType.DATACLASS_OBJECT:
                    #         dcls_info = maybe_template_dcls.st.get_dcls_info_checked()
                    #         dcls_info.delayed_init_set_field_type(target.attr, value.st)
                elif isinstance(stmt, ast.AugAssign):
                    target = self._parse_expr_to_pfl(stmt.target, scope)
                    op = _AST_BINOP_TO_PFL_BINOP[type(stmt.op)]
                    value = self._parse_expr_to_pfl(stmt.value, scope)
                    node = PFLAugAssign(PFLASTType.AUG_ASSIGN,
                                        source_loc,
                                        target=target,
                                        op=op,
                                        value=value)
                    node.check_and_infer_type()
                    block.append(node)
                    if isinstance(target, PFLName):
                        scope[target.id] = target.st
                elif isinstance(stmt, ast.If):
                    ctx = get_parse_context_checked()
                    test = self._parse_expr_to_pfl(stmt.test, scope)
                    # TODO if some special condition (e.g. obj.field is not None),
                    # create tmp type with optional removed for this object
                    # in if body scope.
                    if test.is_const and ctx.cfg.inline_constexpr_if:
                        # TODO we can't do constexpr inline when a dependency
                        # isn't a template function.
                        const_data = test.st.constexpr_data_checked
                        if bool(const_data):
                            res_nodes, rinfo = self._parse_block_to_pfl_ast(
                                stmt.body, scope)
                        else:
                            res_nodes, rinfo = self._parse_block_to_pfl_ast(
                                stmt.orelse, scope)
                        return_info.all_return_stmts.extend(
                            rinfo.all_return_stmts)
                        block.extend(res_nodes)
                        if rinfo.complete:
                            return_info.complete = True
                        continue
                    private_scope_if = scope.copy()
                    if ctx.cfg.allow_remove_optional_based_on_cond:
                        test_optional_removals = self._if_test_optional_removal(test)
                        for var_name, attr_expr in test_optional_removals.items():
                            # TODO support dataclass neste fields
                            if isinstance(attr_expr, PFLName) and var_name in private_scope_if:
                                private_scope_if[var_name] = private_scope_if[var_name]._remove_optional()
                    ifbody, if_rinfo = self._parse_block_to_pfl_ast(
                        stmt.body, private_scope_if)
                    private_scope_else = scope.copy()
                    # TODO support remove-optional in else scope
                    orelse, orelse_rinfo = self._parse_block_to_pfl_ast(
                        stmt.orelse, private_scope_else)
                    if if_rinfo.complete and orelse_rinfo.complete:
                        return_info.complete = True
                    return_info.all_return_stmts.extend(
                        if_rinfo.all_return_stmts)
                    return_info.all_return_stmts.extend(
                        orelse_rinfo.all_return_stmts)
                    common_vars = undefined
                    common_vars_st: Union[dict[str, PFLExprInfo],
                                          Undefined] = undefined
                    if get_parse_context_checked().cfg.allow_new_var_after_if:
                        # compare and merge scopes
                        # 1. get new variables in each scope
                        new_vars_if = set(private_scope_if.keys()) - set(
                            scope.keys())
                        new_vars_else = set(private_scope_else.keys()) - set(
                            scope.keys())
                        # 2. get common variables in both scopes, common vars must have same type.
                        common_vars = list(new_vars_if & new_vars_else)
                        common_vars_st = {}
                        for common_var in common_vars:
                            var_in_if = private_scope_if[common_var]
                            var_in_else = private_scope_else[common_var]
                            merged = var_in_if.try_merge_two_info(var_in_else)
                            scope[common_var] = merged
                            common_vars_st[common_var] = merged

                    node = PFLIf(PFLASTType.IF,
                                 source_loc,
                                 test=test,
                                 body=ifbody,
                                 orelse=orelse,
                                 _new=common_vars_st)
                    node.check_and_infer_type()
                    block.append(node)
                elif isinstance(stmt, ast.For):
                    # variable created in for/while scope won't leaked to parent scope.
                    private_scope = scope.copy()
                    value = self._parse_expr_to_pfl(stmt.iter,
                                                       private_scope)
                    tgt = stmt.target
                    if isinstance(tgt, ast.Tuple):
                        assert value.st.type == PFLExprType.ARRAY, "value type must be array (for x in list/dict.items)"
                        assert value.st.childs[0].type == PFLExprType.TUPLE
                        assert len(
                            tgt.elts) == len(value.st.childs[0].childs
                                             ), "tuple length must be same"
                        is_new_vars: list[bool] = []
                        value_tuple_sts = value.st.childs[0].childs
                        for i, elt in enumerate(tgt.elts):
                            is_new_var = False
                            assert isinstance(
                                elt,
                                ast.Name), "assign tuple item must be Name"
                            if elt.id not in private_scope:
                                is_new_var = True
                            else:
                                value_tuple_sts[i].check_convertable(
                                    private_scope[elt.id], "assign value")
                            private_scope[elt.id] = value_tuple_sts[i]
                            is_new_vars.append(is_new_var)
                        ctx = get_parse_context_checked()
                        if ctx.cfg.tuple_assign_must_be_homogeneous:
                            assert all(is_new_vars) or not any(is_new_vars), \
                                "tuple assign must be homogeneous, all new or all old"
                        target = self._parse_expr_to_pfl(tgt, private_scope)
                        assert isinstance(target, PFLTuple)
                        for i, elt in enumerate(tgt.elts):
                            tgt_pfl_name = target.elts[i]
                            assert isinstance(tgt_pfl_name, PFLName)
                            tgt_pfl_name.is_new = is_new_vars[i]
                    else:
                        assert isinstance(tgt, ast.Name)
                        if value.st.type == PFLExprType.ARRAY:
                            target_st = value.st.childs[0]
                        elif value.st.type == PFLExprType.RANGE:
                            # TODO js requires value must be function call (e.g. must be range(...))
                            target_st = PFLExprInfo(
                                PFLExprType.NUMBER,
                                annotype=parse_type_may_optional_undefined(
                                    int))
                        else:
                            raise NotImplementedError(
                                "for loop iter type must be array or range object"
                            )
                        is_new_var = False
                        if isinstance(tgt, ast.Name):
                            if tgt.id not in private_scope:
                                is_new_var = True
                            else:
                                target_st.check_convertable(
                                    private_scope[tgt.id], "assign value")
                            private_scope[tgt.id] = target_st
                        target = self._parse_expr_to_pfl(
                            stmt.target, private_scope)
                        assert isinstance(target, PFLName)
                        target.is_new = is_new_var
                    forbody, rinfo = self._parse_block_to_pfl_ast(
                        stmt.body, private_scope)
                    return_info.all_return_stmts.extend(rinfo.all_return_stmts)
                    res_node = PFLFor(PFLASTType.FOR,
                                      source_loc,
                                      target=target,
                                      iter=value,
                                      body=forbody)
                    res_node.check_and_infer_type()
                    block.append(res_node)
                elif isinstance(stmt, ast.While):
                    private_scope = scope.copy()
                    test = self._parse_expr_to_pfl(stmt.test, private_scope)
                    forbody, rinfo = self._parse_block_to_pfl_ast(
                        stmt.body, private_scope)
                    return_info.all_return_stmts.extend(rinfo.all_return_stmts)
                    node = PFLWhile(PFLASTType.WHILE,
                                    source_loc,
                                    test=test,
                                    body=forbody)
                    node.check_and_infer_type()

                    block.append(node)
                elif isinstance(stmt, ast.Expr):
                    node = PFLExprStmt(PFLASTType.EXPR_STMT,
                                       source_loc,
                                       value=self._parse_expr_to_pfl(
                                           stmt.value, scope))
                    block.append(node)
                elif isinstance(stmt, ast.Return):
                    value = None
                    if stmt.value is not None:
                        value = self._parse_expr_to_pfl(stmt.value, scope)
                    ret_stmt = PFLReturn(PFLASTType.RETURN,
                                         source_loc,
                                         value=value)

                    block.append(ret_stmt)
                    return_info.all_return_stmts.append(ret_stmt)
                    return_info.complete = True
                    # for return/break/continue, ignore all following statements
                    break
                elif isinstance(stmt, ast.Break):
                    block.append(PFLBreak(PFLASTType.BREAK, source_loc))
                    break
                elif isinstance(stmt, ast.Continue):
                    block.append(PFLContinue(PFLASTType.CONTINUE, source_loc))
                    break
                elif isinstance(stmt, ast.FunctionDef):
                    func_node = self._parse_funcdef_to_pfl_ast(stmt, scope)
                    return_info.all_return_stmts.clear()
                    # func_node.end_scope = private_scope.copy()
                    block.append(func_node)
                else:
                    raise PFLAstParseError(f"not support {type(stmt)}", stmt)
            except PFLAstParseError:
                raise
            except BaseException as e:
                raise PFLAstParseError(f"Unknown error {e}", stmt) from e
        return block, return_info

    def _map_call_args_to_func_def(self, args: list[PFLExprInfo],
                                   kwargs: dict[str, PFLExprInfo],
                                   func_node: ast.FunctionDef,
                                   is_bound_method: bool):
        # only handle args and kwargs here.
        res: dict[str, PFLExprInfo] = {}
        func_node_args = func_node.args.args if not is_bound_method else func_node.args.args[
            1:]
        arg_name_to_ast_arg: dict[str, ast.arg] = {
            arg.arg: arg
            for arg in func_node_args
        }
        for j in range(len(args)):
            ast_arg = func_node_args[j]
            assert ast_arg.arg not in res
            res[ast_arg.arg] = args[j]
        for k, v in kwargs.items():
            assert k in arg_name_to_ast_arg
            if k in res:
                raise ValueError(f"kwarg {k} already used in pos args")
            res[k] = v
        return res

    def _parse_funcdef_to_pfl_ast(
            self,
            stmt: ast.FunctionDef,
            scope: dict[str, PFLExprInfo],
            external_annos: Optional[tuple[dict[str, Any], Any]] = None,
            arg_infos_from_call: Optional[tuple[list[PFLExprInfo],
                                                dict[str,
                                                     PFLExprInfo]]] = None,
            self_type: Optional[PFLExprInfo] = None,
            is_method: bool = False,
            constexpr_args: Optional[dict[str, Any]] = None) -> PFLFunc:
        # external_annos only exists for main functions, depended functions won't use this.
        source_loc = (stmt.lineno, stmt.col_offset, stmt.end_lineno,
                      stmt.end_col_offset)
        assert not stmt.args.posonlyargs, "posonlyargs is not supported in PFL"
        assert not stmt.args.vararg, "vararg is not supported in PFL"
        assert not stmt.args.kwonlyargs, "kwonlyargs is not supported in PFL"
        assert not stmt.args.kwarg, "kwarg is not supported in PFL"
        args: list[PFLArg] = []
        private_scope = scope.copy()
        external_arg_annos: Optional[dict[str, Any]] = None
        external_ret_anno: Optional[Any] = None
        if external_annos is not None:
            external_arg_annos, external_ret_anno = external_annos
        num_arg_no_default = len(stmt.args.args) - len(stmt.args.defaults)
        arg_dict_from_call: Optional[dict[str, PFLExprInfo]] = None
        parse_ctx = get_parse_context_checked()
        compile_req = parse_ctx.get_compile_req_checked()
        if arg_infos_from_call is not None:
            arg_dict_from_call = self._map_call_args_to_func_def(
                arg_infos_from_call[0], arg_infos_from_call[1], stmt,
                compile_req.is_bound_method())
        meta = compile_req.meta
        # print(arg_infos_from_call)
        # if arg_dict_from_call is not None:
        #     for k, v in arg_dict_from_call.items():
        #         print(k, v._constexpr_data)
        for i, arg in enumerate(stmt.args.args):
            if i == 0 and compile_req.is_bound_method():
                continue
            constexpr_data = undefined
            if constexpr_args is not None and arg.arg in constexpr_args:
                # if constexpr_args is provided, we should set the constexpr data
                # for this arg.
                constexpr_data = constexpr_args[arg.arg]
            arg_loc = (arg.lineno, arg.col_offset, arg.end_lineno,
                       arg.end_col_offset)
            anno_in_ast = None
            if arg.annotation is not None:
                anno_in_ast = evaluate_annotation_expr(arg.annotation)
            # print("anno_in_ast", anno_in_ast)
            anno_st = None
            anno_in_ast_st = None
            if i == 0 and compile_req.is_method_def:
                assert self_type is not None, "self_type must be provided for method definition"
                anno_st = dataclasses.replace(self_type)
            default = undefined
            if i >= num_arg_no_default:
                default_pfl = self._parse_expr_to_pfl(stmt.args.defaults[i - num_arg_no_default], scope)
                assert default_pfl.is_const, \
                    f"default value of arg {arg.arg} must be constexpr, but got {default_pfl}"
                default = default_pfl
            if anno_st is None:
                if external_arg_annos is not None:
                    if arg.arg in external_arg_annos:
                        ext_anno = PFLExprInfo.from_annotype(
                            parse_type_may_optional_undefined(
                                external_arg_annos[arg.arg]))
                        ext_anno._constexpr_data = constexpr_data
                        if self._anno_transform is not None and anno_in_ast is not None:
                            anno_st = self._anno_transform(
                                ext_anno, anno_in_ast, default)
                        else:
                            anno_st = ext_anno
                elif arg_dict_from_call is not None:
                    if arg.arg in arg_dict_from_call:
                        anno_st_from_call = dataclasses.replace(
                            arg_dict_from_call[arg.arg],
                            )
                        if anno_in_ast is not None:
                            if self._anno_transform is not None:
                                # user should validate in anno_transform
                                anno_st = self._anno_transform(
                                    anno_st_from_call, anno_in_ast, default)
                            else:
                                # if _anno_transform is None, we 
                                # prefer annotation defined in function def
                                # than anno from args.
                                anno_in_ast_st = PFLExprInfo.from_annotype(
                                    parse_type_may_optional_undefined(
                                        anno_in_ast))
                                assert anno_st_from_call.is_convertable(
                                    anno_in_ast_st)
                                # use type from annotation
                                anno_in_ast_st._constexpr_data = anno_st_from_call._constexpr_data
                                anno_st = anno_in_ast_st
                        else:
                            anno_st = anno_st_from_call
                        if not is_undefined(constexpr_data):
                            anno_st._constexpr_data = constexpr_data

            if anno_st is None and anno_in_ast is not None:
                anno_st = PFLExprInfo.from_annotype(
                    parse_type_may_optional_undefined(anno_in_ast))
                anno_st._constexpr_data = constexpr_data
                if self._anno_transform is not None:
                    # user should validate in anno_transform
                    anno_st = self._anno_transform(anno_st, anno_in_ast, default)

            assert anno_st is not None, f"can't get annotation of arg {arg.arg} from both func def and external."
            st = anno_st
            # reset constexpr to undefined if arg not in constexpr_args set in meta
            if meta.constexpr_args and arg.arg not in meta.constexpr_args:
                st._constexpr_data = undefined

            arg_obj = PFLArg(PFLASTType.ARG, arg_loc, arg=arg.arg, st=st)
            if arg.annotation is not None:
                arg_obj.annotation = ast.unparse(arg.annotation)
            if not is_undefined(default):
                arg_obj.default = default
            args.append(arg_obj)
            private_scope[arg_obj.arg] = arg_obj.st
        # for pfl_arg, default in zip(args[num_arg_no_default:],
        #                             stmt.args.defaults):
        #     default_pfl = self._parse_expr_to_pfl(default, scope)
        #     assert default_pfl.is_const, \
        #         f"default value of arg {pfl_arg.arg} must be constant, but got {default_pfl}"
        #     pfl_arg.default = default_pfl
        funbody, rinfo = self._parse_block_to_pfl_ast(stmt.body, private_scope)
        
        if not rinfo.complete:
            ret_none_value = PFLConstant(PFLASTType.CONSTANT,
                                         (-1, -1, None, None),
                                         value=None)
            ret_none_value.check_and_infer_type()
            ret = PFLReturn(PFLASTType.RETURN, (-1, -1, None, None),
                            value=ret_none_value)
            rinfo.all_return_stmts.append(ret)

        ret_sts: list[PFLExprInfo] = []
        if rinfo.all_return_stmts:
            first_rstmt = rinfo.all_return_stmts[0]
            first_rstmt_st = PFLExprInfo(
                PFLExprType.NONE_TYPE
            ) if first_rstmt.value is None else first_rstmt.value.st
            ret_sts.append(first_rstmt_st)
            for rstmt in rinfo.all_return_stmts[1:]:
                rstmt_st = PFLExprInfo(
                    PFLExprType.NONE_TYPE
                ) if rstmt.value is None else rstmt.value.st
                if stmt.returns is None:
                    assert rstmt_st.is_equal_type(first_rstmt_st), \
                        f"all return stmts must have same type, but got {rstmt_st} and {first_rstmt_st}"
                ret_sts.append(rstmt_st)
        finfo_args: list[PFLExprFuncArgInfo] = []
        for a in args:
            arg_info = PFLExprFuncArgInfo(a.arg, a.st)
            if a.default is not None:
                arg_info.default = a.default.st._constexpr_data if a.default.is_const else undefined
                arg_info.default_type = a.default.st
            finfo_args.append(arg_info)
        func_node_finfo = PFLExprFuncInfo(
            stmt.name,
            ret_sts[0],
            finfo_args,
        )
        func_node_finfo.is_method = is_method
        func_node_st = PFLExprInfo(type=PFLExprType.FUNCTION,
                                   func_info=func_node_finfo)
        func_node = PFLFunc(PFLASTType.FUNC,
                            source_loc,
                            name=stmt.name,
                            args=args,
                            st=func_node_st,
                            body=funbody)
        if meta.userdata is not None:
            func_node.userdata = meta.userdata
        if stmt.returns is not None:
            ann_res = evaluate_annotation_expr(stmt.returns)
            st = PFLExprInfo.from_annotype(
                parse_type_may_optional_undefined(ann_res))
            if ret_sts:
                for ret_st in ret_sts:
                    assert ret_st.is_convertable(st), f"{st} vs {ret_st}"
            func_node.ret_st = st
        if func_node.ret_st is None and external_ret_anno is not None:
            st = PFLExprInfo.from_annotype(
                parse_type_may_optional_undefined(external_ret_anno))
            if ret_sts:
                for ret_st in ret_sts:
                    assert ret_st.is_convertable(st), f"{st} vs {ret_st}"
            func_node.ret_st = st
        if ret_sts and func_node.ret_st is None:
            func_node.ret_st = ret_sts[0]
        # if stmt.decorator_list:
        #     # ctx = get_parse_context_checked()
        #     # ctx._disable_type_check = True
        #     func_node.decorator_list = [self._parse_expr_to_pfl(e, scope) for e in stmt.decorator_list]
        #     # ctx._disable_type_check = False

        # TODO disable nested func support
        # always clear return info when func is end
        # TODO add function to scope
        return func_node

    def parse_expr_string_to_pfl_ast(self,
            expr_str: str, constants: dict[str, Any],
            init_scope_types: dict[str, Any],
            partial_type_infer: bool = False):
        expr_str_ast = ast.parse(expr_str)
        assert len(expr_str_ast.body) == 1, "only expr is allowed."
        stmt = expr_str_ast.body[0]
        assert isinstance(stmt, ast.Expr)
        expr_node = stmt.value
        func_code_lines = expr_str.splitlines()
        transformer = MapAstNodeToConstant(
            constants,
            set(constants.keys()),
            PFLErrorFormatContext(func_code_lines),
            backend=self._backend,
            var_preproc=self._var_preproc)
        # use MapAstNodeToConstant to preprocess scope.
        transformer.visit(expr_node)
        parse_cfg = self._parse_cfg
        backend = self._backend
        if partial_type_infer:
            parse_cfg = dataclasses.replace(parse_cfg, 
                allow_partial_type_infer=True)
        parse_ctx = PFLParseContext(
            func_code_lines,
            {},
            self._var_preproc,
            cfg=parse_cfg,
            backend=backend,
            node_to_constants=transformer._node_to_compile_constant,
            compile_req=None,
            allow_inline_expand=False)
        with enter_parse_context(parse_ctx) as ctx:
            # scope is used as constant.
            init_scope = {}
            for k, v in init_scope_types.items():
                init_scope[k] = PFLExprInfo.from_annotype(
                    parse_type_may_optional_undefined(v), is_type=True)

            pfl_node = self._parse_expr_to_pfl(expr_node, init_scope)
        return pfl_node

    def parse_func_compile_req_to_pfl_ast(
            self,
            req: PFLCompileReq,
            allow_inline_expand: bool = False) -> PFLFunc:
        assert not req.is_dcls
        parse_cfg = self._parse_cfg
        backend = self._backend
        func = req.func_or_dcls
        # if length of func_qname_parts > 1, get the last part class from globals
        func_uid = req.uid
        func_meta = req.meta
        outer_ctx = get_parse_context()
        # TODO should we include the decorators?
        # func_code_lines, first_lineno = self._func_code_getter(func)
        # func_code_lines = [l.rstrip() for l in func_code_lines]
        if func in self._cache_fn_precompile_info:
            cache = self._cache_fn_precompile_info[func]
            module_code = cache.module_code
            module_path = cache.module_path
            func_node = cache.tree
            func_code_lines = cache.func_lines
            code = cache.code 
            first_lineno = cache.first_lineno
            module_code_lines = cache.module_code_lines
        else:
            module_code, module_path = self._module_code_path_getter(func)
            func_code_lines, first_lineno = getsourcelinesby_lines(
                func, [line + '\n' for line in module_code.splitlines()])
            func_code_lines = [l.rstrip() for l in func_code_lines]
            # remove indents of func_code_lines
            code = "\n".join(func_code_lines)
            common_indent = determine_code_common_indent(code)
            code_no_common_indent = remove_common_indent_from_code(code)
            tree = ast.parse(code_no_common_indent)
            func_node: Optional[ast.FunctionDef] = None
            for node in tree.body:
                if isinstance(node,
                            ast.FunctionDef) and node.name == func.__name__:
                    func_node = node
            assert func_node is not None
            module_code_lines = module_code.splitlines()
            self._cache_fn_precompile_info[func] = _CompileFuncCache(
                module_code=module_code,
                module_path=module_path,
                tree=func_node,
                module_code_lines=module_code_lines,
                func_lines=func_code_lines,
                code=code,
                first_lineno=first_lineno)
            for child_node in ast.walk(func_node):
                # add first_lineno to all nodes
                if hasattr(child_node, 'lineno'):
                    child_node.lineno += first_lineno - 1  # type: ignore
                if hasattr(child_node, 'end_lineno'):
                    child_node.end_lineno += first_lineno - 1  # type: ignore
                if hasattr(child_node, 'col_offset'):
                    child_node.col_offset += common_indent  # type: ignore
                if hasattr(child_node, 'end_col_offset'):
                    child_node.end_col_offset += common_indent  # type: ignore
        closure = inspect.getclosurevars(func)
        fn_globals_base = func.__globals__ | dict(closure.nonlocals)
        global_nonlocal_names = set(closure.nonlocals.keys()) | set(closure.globals.keys())
        fn_globals = fn_globals_base.copy()
        if req.bound_self is not None:
            # add "self" to globals
            assert req.bound_self is not None, "bound_self must not be None"
            assert func_node is not None
            # get first arg name
            first_arg = func_node.args.args[0] if func_node.args.args else None
            assert first_arg is not None
            fn_globals[first_arg.arg] = req.bound_self
        # find funcdef
        transformer = MapAstNodeToConstant(
            fn_globals,
            global_nonlocal_names,
            PFLErrorFormatContext(func_code_lines),
            backend=backend,
            var_preproc=self._var_preproc)
        transformer.visit(func_node)
        anno_eval_globals = fn_globals_base.copy()
        outer_ctx = get_parse_context()
        if outer_ctx is not None:
            parse_ctx = PFLParseContext.from_outer_ctx(
                outer_ctx,
                module_code_lines,
                anno_eval_globals,
                node_to_constants=transformer._node_to_compile_constant,
                compile_req=req,
                allow_inline_expand=allow_inline_expand)
        else:
            parse_ctx = PFLParseContext(
                module_code_lines,
                anno_eval_globals,
                self._var_preproc,
                cfg=parse_cfg,
                backend=backend,
                node_to_constants=transformer._node_to_compile_constant,
                compile_req=req,
                allow_inline_expand=allow_inline_expand)
        if func_meta is not None:
            self.func_node_to_meta[func_node] = func_meta
        with enter_parse_context(parse_ctx) as ctx:
            # if is_root:
            init_scope: dict[str, PFLExprInfo] = {}
            # for k, v in STD_REGISTRY.global_dict.items():
            #     if v.backend is None or v.backend == backend:
            #         init_scope[v.mapped_name] = ctx.cache.cached_parse_std_item(v)
            #         if not v.is_func:
            #             anno_eval_globals[v.mapped_name] = v.dcls
            scope = init_scope.copy()
            try:
                self_type = None
                if req.self_type is not None:
                    if not isinstance(req.self_type, PFLExprInfo):
                        self_type = PFLExprInfo.from_annotype(req.self_type)
                    else:
                        self_type = req.self_type
                func_pfl_node = self._parse_funcdef_to_pfl_ast(
                    func_node, scope, req.external_anno, req.args_from_call,
                    self_type, req.is_method_def, req.constexpr_args)
                func_compile_uid = req.get_func_compile_uid(
                    delayed_info=func_pfl_node.st.get_func_info_checked())
                func_pfl_node.uid = func_compile_uid
                func_pfl_node.deps = ctx.depend_compilables
                func_pfl_node.backend = backend
                func_pfl_node.st.compiled_uid = func_compile_uid

                func_pfl_node.compile_info.code = code
                func_pfl_node.compile_info.first_lineno = first_lineno
                func_pfl_node.compile_info.original = func
                func_pfl_node.compile_info.meta = func_meta
                func_pfl_node.compile_info.path = module_path
                if func_compile_uid not in self._all_compiled:
                    self._all_compiled[func_compile_uid] = func_pfl_node
                else:
                    func_pfl_node = cast(PFLFunc, self._all_compiled[func_compile_uid])

            except PFLAstParseError as e:
                print(f"In parsing function {func.__name__} at {module_path}:{first_lineno}")
                error_line = get_parse_context_checked(
                ).format_error_from_lines_node(e.node)
                print(error_line)
                raise e
        return func_pfl_node


    def parse_dcls_compile_req_to_pfl_ast(
            self,
            req: PFLCompileReq,
            allow_inline_expand: bool = False) -> PFLClass:
        assert req.is_dcls
        ctx = get_parse_context_checked()
        dcls_type = req.self_type
        assert isinstance(dcls_type, PFLExprInfo)
        dcls_obj_type = dataclasses.replace(dcls_type, type=PFLExprType.DATACLASS_OBJECT)
        lineno = inspect.getsourcelines(req.func_or_dcls)[1]
        cls_node = PFLClass(PFLASTType.CLASS, (lineno, 0, -1, -1),
            name=req.func_or_dcls.__name__, st=dcls_type)
        assert inspect.isclass(req.func_or_dcls)
        has_user_init = is_dcls_init_defined_by_user(req.func_or_dcls)
        has_user_pinit = is_dcls_post_init_defined_by_user(req.func_or_dcls)
        parse_cache = get_parse_cache_checked()
        # template dataclasses info is shared for all subsequence usage.
        _, module_path = self._module_code_path_getter(req.func_or_dcls)
        assert req.info is not None
        if has_user_init:
            fn = req.func_or_dcls.__init__
            if req.meta.is_template:
                if req.args_from_call is not None:
                    args_from_call = ([dcls_obj_type] + req.args_from_call[0], req.args_from_call[1])
                else:
                    args_from_call = ([dcls_obj_type], {})
                fn_req = ctx.get_compile_req(fn, None, req.meta, is_method_def=True, self_type=dcls_obj_type,
                    args_from_call=args_from_call,
                    local_ids=req.info._locals_ids)
                fn_req.dcls_infer_field_type = True
                compiled_func = self.parse_func_compile_req_to_pfl_ast(
                    fn_req, allow_inline_expand=allow_inline_expand)
                # parse 
                dcls_info = dcls_type.get_dcls_info_checked()
                for field in dcls_info.args:
                    assert field.is_type_parsed, f"you must set all fields in init fn when your dcls is template, got {field.name} missing."
                cls_node.init_uid = compiled_func.uid
                # use func_info to store init function info
                dcls_type.func_info = compiled_func.st.get_func_info_checked()
                if req.meta.constexpr_args:
                    for field in dcls_info.args:
                        if field.name not in req.meta.constexpr_args:
                            field.type._constexpr_data = undefined
            else:
                new_st = parse_cache.cached_parse_func(fn,
                                    self_type=dcls_obj_type.annotype)
                new_finfo = new_st.get_func_info_checked()
                if new_finfo.compilable_meta is not None:
                    assert not new_finfo.compilable_meta.is_template, \
                        "non-template class should have non-template init function"
                creq = ctx.enqueue_func_compile(
                    fn,
                    new_finfo,
                    is_method_def=True,
                    self_type=dcls_obj_type,
                    is_prop=False,
                    local_ids=req.info._locals_ids)
                # parse 
                cls_node.init_uid = creq.get_func_compile_uid()
                # use func_info to store init function info
                dcls_type.func_info = new_finfo
            dcls_type.func_info.return_type = dcls_obj_type
        else:
            if req.meta.is_template:
                dcls_info = req.info
                assert dcls_info is not None and req.args_from_call is not None
                matched = dcls_info.match_args_to_sig(req.args_from_call)
                for field_info, arg_st in matched.args:
                    if not is_undefined(arg_st):
                        field_info.type = arg_st 
                        field_info.is_type_parsed = True
                    else:
                        assert not is_undefined(field_info.default_type), \
                            f"field {field_info.name} default type must be defined"
                        field_info.type = field_info.default_type
                        field_info.is_type_parsed = True
                dcls_type.func_info = dataclasses.replace(dcls_info)
            else:
                dcls_info = req.info
                assert dcls_info is not None
                dcls_type.func_info = dataclasses.replace(dcls_info)

        if has_user_pinit:
            fn = req.func_or_dcls.__post_init__
            if req.meta.is_template:

                fn_req = ctx.get_compile_req(fn, None, req.meta, is_method_def=True, self_type=dcls_obj_type,
                    args_from_call=req.args_from_call,
                    local_ids=req.info._locals_ids)
                # parse 
                compiled_func = self.parse_func_compile_req_to_pfl_ast(
                    fn_req, allow_inline_expand=allow_inline_expand)
                cls_node.post_init_uid = compiled_func.uid
            else:
                new_st = parse_cache.cached_parse_func(fn,
                                    self_type=dcls_obj_type.annotype)
                new_finfo = new_st.get_func_info_checked()
                if new_finfo.compilable_meta is not None:
                    assert not new_finfo.compilable_meta.is_template, \
                        "non-template class should have non-template post_init function"
                creq = ctx.enqueue_func_compile(
                    fn,
                    new_finfo,
                    is_method_def=True,
                    self_type=dcls_obj_type,
                    is_prop=False,
                    local_ids=req.info._locals_ids)
                cls_node.post_init_uid = creq.get_func_compile_uid()

        cls_node.st = dcls_type
        compilable_uid = req.get_func_compile_uid()
        cls_node.name = req.func_or_dcls.__name__
        cls_node.uid = compilable_uid
        cls_node.st.compiled_uid = compilable_uid
        cls_node.compile_info.original = req.func_or_dcls
        cls_node.compile_info.meta = req.meta
        cls_node.compile_info.path = module_path
        if compilable_uid not in self._all_compiled:
            self._all_compiled[compilable_uid] = cls_node
        else:
            cls_node = cast(PFLClass, self._all_compiled[compilable_uid])
        return cls_node 

    def parse_compile_req_to_pfl_ast(
            self,
            req: PFLCompileReq,
            allow_inline_expand: bool = False) -> Union[PFLClass, PFLFunc]:
        if req.is_dcls:
            return self.parse_dcls_compile_req_to_pfl_ast(req,
                                                           allow_inline_expand)
        else:
            return self.parse_func_compile_req_to_pfl_ast(req,
                                                          allow_inline_expand)

    def _get_compile_req(
            self,
            func: Callable,
            external_anno: Optional[tuple[dict[str, Any], Any]] = None,
            arg_infos_from_call: Optional[tuple[list[PFLExprInfo],
                                                dict[str,
                                                     PFLExprInfo]]] = None,
            constexpr_args: Optional[dict[str, Any]] = None) -> PFLCompileReq:
        is_bound_method = False
        bound_self = None
        if inspect.ismethod(func):
            is_bound_method = True
            bound_self = func.__self__
            # get unbound func
            func = func.__func__
        prep_res = self._var_preproc(func)
        func = prep_res.value
        func_meta = prep_res.compilable_meta
        func_qname_parts = func.__qualname__.split('.')
        # if length of func_qname_parts > 1, get the last part class from globals
        self_type = None
        if not is_bound_method and len(func_qname_parts) > 1:
            parent_from_qn = func.__globals__
            for part in func_qname_parts[:-1]:
                if isinstance(parent_from_qn, dict):
                    parent_from_qn = parent_from_qn[part]
                elif isinstance(parent_from_qn, type):
                    parent_from_qn = getattr(parent_from_qn, part)
                else:
                    raise NotImplementedError
            assert inspect.isclass(parent_from_qn) and dataclasses.is_dataclass(
                parent_from_qn
            ), f"parent {parent_from_qn} must be a dataclass type, got {type(parent_from_qn)}"
            self_type = parse_type_may_optional_undefined(parent_from_qn)
        func_uid = get_module_id_of_type(func)
        func_sig = inspect.signature(func)
        is_method_def = False
        if self_type is not None and len(func_sig.parameters) > 0 and \
            func_sig.parameters[list(func_sig.parameters.keys())[0]].name == 'self':
            is_method_def = True
        if func_meta is None:
            func_meta = PFLCompileFuncMeta(backends=[self._backend],
                                           is_template=False,
                                           always_inline=False)
        req = PFLCompileReq(func,
                            func_uid,
                            func_meta,
                            args_from_call=arg_infos_from_call,
                            external_anno=external_anno,
                            self_type=self_type,
                            bound_self=bound_self,
                            is_method_def=is_method_def,
                            constexpr_args=constexpr_args)
        return req

    def parse_func_to_pfl_ast(
            self,
            func: Callable,
            scope: Optional[dict[str, PFLExprInfo]] = None,
            external_anno: Optional[tuple[dict[str, Any], Any]] = None,
            constexpr_args: Optional[dict[str, Any]] = None) -> PFLFunc:
        req = self._get_compile_req(func,
                                    external_anno,
                                    constexpr_args=constexpr_args)
        func_need_to_compile: list[PFLCompileReq] = [req]
        outer_ctx = get_parse_context()
        assert outer_ctx is None, "must be root"
        parse_ctx_root = PFLParseContext.create_root_ctx(
            self._backend, self._var_preproc, self._parse_cfg,
            func_need_to_compile)
        with enter_parse_context(parse_ctx_root):
            while func_need_to_compile:
                compile_req = func_need_to_compile.pop()
                if compile_req.get_func_compile_uid(
                ) not in self._all_compiled:
                    PFL_LOGGER.warning("%s", str(compile_req))
                    self.parse_compile_req_to_pfl_ast(compile_req)
        return cast(PFLFunc, self._all_compiled[req.get_func_compile_uid()])

    def parse_funcs_to_pfl_ast(
            self,
            funcs: list[Callable],
            external_annos: Optional[list[tuple[dict[str, Any], Any]]] = None):

        outer_ctx = get_parse_context()
        assert outer_ctx is None, "must be root"
        func_need_to_compile: list[PFLCompileReq] = []
        for j in range(len(funcs)):
            external_anno = None if not external_annos else external_annos[
                j] if external_annos is not None else None
            req = self._get_compile_req(funcs[j], external_anno)
            func_need_to_compile.append(req)
        parse_ctx_root = PFLParseContext.create_root_ctx(
            self._backend, self._var_preproc, self._parse_cfg,
            func_need_to_compile)
        with enter_parse_context(parse_ctx_root):
            while func_need_to_compile:
                compile_req = func_need_to_compile.pop()
                if compile_req.get_func_compile_uid(
                ) not in self._all_compiled:
                    PFL_LOGGER.warning("%s", str(compile_req))
                    self.parse_compile_req_to_pfl_ast(compile_req)
        return self._all_compiled_to_pfl_library(self._all_compiled)

    def _all_compiled_to_pfl_library(
            self, all_compiled: dict[str, Union[PFLFunc, PFLClass]]) -> PFLLibrary:
        """Convert all compiled functions to a PFL library."""

        all_modules: dict[str, PFLModule] = {}
        for k, v in all_compiled.items():
            module_id = v.get_module_import_path()
            if module_id not in all_modules:
                assert v.compile_info.original is not None
                module_code, path = self._module_code_path_getter(v.compile_info.original)
                mod = PFLModule(PFLASTType.MODULE, (-1, -1, None, None),
                                uid=module_id)
                mod.compile_info.code = module_code
                mod.compile_info.path = path
                all_modules[module_id] = mod
            all_modules[module_id].body.append(v)
        return PFLLibrary(all_modules)

def parse_func_to_pfl_ast(
        func: Callable,
        scope: Optional[dict[str, PFLExprInfo]] = None,
        backend: str = "js",
        parse_cfg: Optional[PFLParseConfig] = None,
        func_code_getter: Optional[Callable[[Any], tuple[list[str],
                                                         int]]] = None,
        module_code_path_getter: Optional[Callable[[Any], tuple[str, str]]] = None,
        var_preproc: Optional[Callable[[Any], PFLProcessedVarMeta]] = None,
        anno_transform: Optional[Callable[[PFLExprInfo, Any, Union[Undefined, Any]],
                                          PFLExprInfo]] = None,
        all_compiled: Optional[dict[str, Union[PFLFunc, PFLClass]]] = None,
        external_anno: Optional[tuple[dict[str, Any], Any]] = None,
        constexpr_args: Optional[dict[str, Any]] = None) -> PFLFunc:
    parser = PFLParser(backend, parse_cfg, func_code_getter,
                       module_code_path_getter, var_preproc,
                       anno_transform)
    if all_compiled is not None:
        parser._all_compiled = all_compiled
    return parser.parse_func_to_pfl_ast(func, scope, external_anno,
                                        constexpr_args)


def parse_func_to_pfl_library(
        func: Callable,
        scope: Optional[dict[str, PFLExprInfo]] = None,
        backend: str = "js",
        parse_cfg: Optional[PFLParseConfig] = None,
        func_code_getter: Optional[Callable[[Any], tuple[list[str],
                                                         int]]] = None,
        module_code_path_getter: Optional[Callable[[Any], tuple[str, str]]] = None,
        var_preproc: Optional[Callable[[Any], PFLProcessedVarMeta]] = None,
        anno_transform: Optional[Callable[[PFLExprInfo, Any, Union[Undefined, Any]],
                                          PFLExprInfo]] = None,
        external_anno: Optional[tuple[dict[str, Any], Any]] = None,
        constexpr_args: Optional[dict[str, Any]] = None) -> PFLLibrary:
    """Parse func and its dependencies to a PFL library.
    this function will parse whole file instead of func code only to ast. 
    if your func is dynamic generated, you need to use `tempfile_in_linecache` to add your dynamic code to linecache.
    """
    parser = PFLParser(backend, parse_cfg, func_code_getter,
                       module_code_path_getter, var_preproc,
                       anno_transform)
    parser.parse_func_to_pfl_ast(func, scope, external_anno, constexpr_args)
    return parser._all_compiled_to_pfl_library(parser._all_compiled)

def parse_expr_string_to_pfl_ast(
    expr_str: str, 
    constants: dict[str, Any],
    init_scope_types: dict[str, Any],
    partial_type_infer: bool = False,
    backend: str = "js",
    parse_cfg: Optional[PFLParseConfig] = None
) -> PFLExpr:
    """Parse a expression string to PFL AST.
    """
    parser = PFLParser(backend, parse_cfg)
    return parser.parse_expr_string_to_pfl_ast(
        expr_str, constants, init_scope_types, partial_type_infer
    )

class _AstAsDict:
    def __init__(self, exclude_fields: Optional[set[str]] = None, ignore_dcls_info: bool = False, ignore_func_info: bool = False):
        if exclude_fields is None:
            exclude_fields = set()
        exclude_fields.update([
            "compile_info", "source_loc"
        ])
        self.exclude_fields = exclude_fields
        self._ignore_dcls_info = ignore_dcls_info
        self._ignore_func_info = ignore_func_info

    def _ast_as_dict(self, obj):
        if isinstance(obj, PFLAstNodeBase):
            result = []
            for f in dataclasses.fields(obj):
                if f.name in self.exclude_fields:
                    continue
                value = self._ast_as_dict(getattr(obj, f.name))
                if not isinstance(value, Undefined):
                    result.append((f.name, value))
            return dict(result)
        elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
            return type(obj)(*[self._ast_as_dict(v) for v in obj])
        elif isinstance(obj, (list, tuple)):
            # Assume we can create an object of this type by passing in a
            # generator (which is not true for namedtuples, handled
            # above).
            return type(obj)(self._ast_as_dict(v) for v in obj)
        elif isinstance(obj, dict):
            return type(obj)(
                (self._ast_as_dict(k), self._ast_as_dict(v)) for k, v in obj.items())
        else:
            if isinstance(obj, PFLExprInfo):
                res = obj.to_dict()
                if "dcls_info" in res and self._ignore_dcls_info:
                    res.pop("dcls_info")
                if "func_info" in res and self._ignore_func_info:
                    res.pop("func_info")
                return res 
            return obj


    def _ast_as_dict_for_dump(self, obj):
        if isinstance(obj, PFLAstNodeBase):
            result = []
            for f in dataclasses.fields(obj):
                # FIXME: better way to remove code field in PFLFunc
                if f.name in self.exclude_fields:
                    continue
                value = self._ast_as_dict_for_dump(getattr(obj, f.name))
                if not isinstance(value, Undefined):
                    result.append((f.name, value))
            return dict(result)
        elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
            return type(obj)(*[self._ast_as_dict_for_dump(v) for v in obj])
        elif isinstance(obj, (list, tuple)):
            # Assume we can create an object of this type by passing in a
            # generator (which is not true for namedtuples, handled
            # above).
            return type(obj)(self._ast_as_dict_for_dump(v) for v in obj)
        elif isinstance(obj, dict):
            return type(obj)((self._ast_as_dict_for_dump(k), self._ast_as_dict_for_dump(v))
                            for k, v in obj.items())
        else:
            if isinstance(obj, PFLExprInfo):
                return str(obj)
            return obj


def pfl_ast_to_dict(node: PFLAstNodeBase):
    return _AstAsDict()._ast_as_dict(node)


def ast_dump(node: PFLAstNodeBase):
    return _AstAsDict()._ast_as_dict_for_dump(node)
