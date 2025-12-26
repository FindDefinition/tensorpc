
import dataclasses
import linecache
import os
import re
import tempfile
import textwrap
from typing import (
    Iterable,
    Optional,
    Union,
    cast,
    TypeVar,
    overload,
)
import typing
from tensorpc.core.tree_id import UniqueTreeId
import inspect
import triton

import triton.language as tl
from typing import Any, Callable
import triton
import ast 
from tensorpc.apps.ttfs.aggtype import (
    TRITON_AGG_FLATTEN_META_FIELD,
    FieldMeta,
    is_aggregate_type,
    TritonAggFlatField,
)
from triton.experimental.gluon._runtime import GluonJITFunction
from triton.runtime.autotuner import Autotuner

if typing.TYPE_CHECKING:
    from _typeshed import DataclassInstance as StandardDataclass

_T = TypeVar("_T")

def _get_flatten_agg_to_kwargs_lines(obj_type, root_arg_name: str):
    assert is_aggregate_type(obj_type)
    agg_fields: dict[str, TritonAggFlatField] = getattr(obj_type, TRITON_AGG_FLATTEN_META_FIELD)
    lines: list[str] = []
    for name, field in agg_fields.items():
        if field.meta.is_kernel_argument:
            if field.meta.accessor is not None:
                accessor_lines = field.meta.accessor.get_flatten_agg_to_kwarg_lines(field, root_arg_name, f"{root_arg_name}.{name}")
                lines.extend(accessor_lines)
            else:
                lines.append(f"kwargs['{field.mangle_path(root_arg_name)}'] = {root_arg_name}.{name}")
    return lines

def get_mangled_name(dcls: "StandardDataclass", obj_name: str, field_path: str) -> str:
    """Get mangled name for a field path in an aggregate dataclass.
    e.g. for field_path = "a.b.c" and obj_name = "args", return mangled name for args.a.b.c
    assert"""
    dcls_type = type(dcls)
    assert is_aggregate_type(dcls_type)
    agg_fields: dict[str, TritonAggFlatField] = getattr(dcls_type, TRITON_AGG_FLATTEN_META_FIELD)
    field = agg_fields[field_path]
    return field.mangle_path(obj_name)
    
def _create_flatten_agg_arg_nodes(obj_type, root_arg_name: str):
    assert is_aggregate_type(obj_type)
    agg_fields: dict[str, TritonAggFlatField] = getattr(obj_type, TRITON_AGG_FLATTEN_META_FIELD)
    res: list[tuple[ast.arg, bool]] = []
    for name, field in agg_fields.items():
        if not field.meta.is_kernel_argument:
            continue
        if field.meta.accessor is not None:
            field_constexpr_tuple = field.meta.accessor.get_custom_fields()
            for field_name, is_constexpr in field_constexpr_tuple:
                name_mangled = field.mangle_path(root_arg_name, field_name)
                anno = ast.Attribute(ast.Name("tl"), "constexpr") if is_constexpr else None
                arg = ast.arg(arg=name_mangled, annotation=anno)
                res.append((arg, is_constexpr))
        else:
            name_mangled = field.mangle_path(root_arg_name)
            anno = ast.Attribute(ast.Name("tl"), "constexpr") if field.is_constexpr else None

            arg = ast.arg(arg=name_mangled, annotation=anno)
            res.append((arg, field.is_constexpr))
    return res

def _parts_to_ast_node(parts: list[str]) -> ast.expr:
    if len(parts) == 1:
        return ast.Name(id=parts[0], ctx=ast.Load())
    else:
        return ast.Attribute(
            value=_parts_to_ast_node(parts[:-1]),
            attr=parts[-1],
            ctx=ast.Load(),
        )

@dataclasses.dataclass
class _StackItem:
    node: ast.Call 
    field_meta: Optional[FieldMeta] = None

def _create_agg_from_flatten_args_ast_node(obj_type, root_cls_name: str, root_agg_name: str):
    """create nested aggregate type from flattened args
    e.g. given obj_type = A, and args = (a_b_c, a_b_d, a_e)
    return A(b=B(c=a_b_c, d=a_b_d), e=a_e)
    """
    assert is_aggregate_type(obj_type)
    agg_fields: dict[str, TritonAggFlatField] = getattr(obj_type, TRITON_AGG_FLATTEN_META_FIELD)
    cur_prefix = UniqueTreeId("")
    # (cur_dcls_field_name, )
    root_node = ast.Name(id=root_cls_name, ctx=ast.Load())
    cur_ctor_stack: list[_StackItem] = [
        _StackItem(
            ast.Call(
                func=root_node,
                args=[],
                keywords=[],
            )
        )
    ]
    for name, field in agg_fields.items():
        if not field.meta.is_kernel_argument:
            continue
        # cur_prefix: a.d
        # cur_field = a.b.c
        name_mangled = field.mangle_path(root_agg_name)
        parts = name.split('.')
        prefix = UniqueTreeId.from_parts(parts[:-1])
        field_name = parts[-1]
        common_idx = prefix.common_prefix_index(cur_prefix)
        for j in range(len(cur_prefix.parts) - 1, common_idx - 1, -1):
            # pop stack
            item = cur_ctor_stack.pop()
            cur_ctor_stack[-1].node.keywords.append(
                ast.keyword(
                    arg=cur_prefix.parts[j],
                    value=item.node,
                )
            )
        cur_prefix = prefix
        for j in range(common_idx, len(prefix.parts)):
            parts_local = prefix.parts[: j + 1]
            qname = ".".join(parts_local)
            get_field_type_parts = "ttfs.mp.get_field_type".split(".")
            fn_node = _parts_to_ast_node([*get_field_type_parts])
            call_get_type = ast.Call(
                fn_node,
                [
                    root_node,
                    ast.Constant(value=qname),
                ],
                []
            )
            call_node = ast.Call(
                func=call_get_type,
                args=[],
                keywords=[],
            )
            # push stack
            # print(qname, meta.parent_field_metas[j], j, len(meta.parent_field_metas))
            stack_item = _StackItem(call_node, field_meta=field.parent_field_metas[j])
            cur_ctor_stack.append(stack_item)
        if field.meta.accessor is not None:
            get_field_type_parts = "ttfs.mp.get_field_type".split(".")
            fn_node = _parts_to_ast_node([*get_field_type_parts])
            call_get_type = ast.Call(
                fn_node,
                [
                    root_node,
                    ast.Constant(value=name),
                ],
                []
            )
            arg_node = field.meta.accessor.get_load_call_node(field, root_agg_name, call_get_type)
        else:
            arg_node = ast.Name(id=name_mangled, ctx=ast.Load())
        cur_ctor_stack[-1].node.keywords.append(
            ast.keyword(
                arg=field_name,
                value=arg_node,
            )
        )
    for j in range(len(cur_ctor_stack) - 1, 0, -1):
        # pop stack
        item = cur_ctor_stack.pop()
        cur_ctor_stack[-1].node.keywords.append(
            ast.keyword(
                arg=cur_prefix.parts[j - 1],
                value=item.node,
            )
        )
    assert len(cur_ctor_stack) == 1
    return cur_ctor_stack[0].node

def _expand_agg_in_args_kwargs_v2(sig: inspect.Signature, flatten_fn, args, kwargs, agg_names: list[str]):
    sig_args = dict(sig.bind(*args, **kwargs).arguments)
    aggs = []
    for name in agg_names:
        assert name in sig_args
        aggs.append(sig_args.pop(name))
    flatten_fn(sig_args, *aggs)
    return sig_args

@dataclasses.dataclass
class _JitExMeta:
    agg_names: list[str]
    agg_metas: dict[str, Any]
    original_sig: inspect.Signature
    flatten_fn: Optional[Callable[[Any, dict], None]] = None

class JITFunctionEx(triton.JITFunction[_T]):
    _ttfs_jit_ex_meta: _JitExMeta

    def _ttfs_set_agg_metas(self, metas: _JitExMeta):
        self._ttfs_jit_ex_meta = metas

    def run(self, *args, grid, warmup, **kwargs):
        if self._ttfs_jit_ex_meta.flatten_fn is not None:
            sig = self._ttfs_jit_ex_meta.original_sig
            kwargs = _expand_agg_in_args_kwargs_v2(sig, self._ttfs_jit_ex_meta.flatten_fn, args, kwargs, self._ttfs_jit_ex_meta.agg_names)
            return super().run(grid=grid, warmup=warmup, **kwargs)
        else:
            return super().run(*args, grid=grid, warmup=warmup, **kwargs)

class GluonJITFunctionEx(GluonJITFunction[_T]):
    _ttfs_jit_ex_meta: _JitExMeta

    def _ttfs_set_agg_metas(self, metas: _JitExMeta):
        self._ttfs_jit_ex_meta = metas

    def run(self, *args, grid, warmup, **kwargs):
        if self._ttfs_jit_ex_meta.flatten_fn is not None:
            sig = self._ttfs_jit_ex_meta.original_sig
            kwargs = _expand_agg_in_args_kwargs_v2(sig, self._ttfs_jit_ex_meta.flatten_fn, args, kwargs, self._ttfs_jit_ex_meta.agg_names)
            return super().run(grid=grid, warmup=warmup, **kwargs)
        else:
            return super().run(*args, grid=grid, warmup=warmup, **kwargs)

# from xformers
# Hackfix to get access to get source-code for
# `exec`-created functions - see https://stackoverflow.com/a/69668999
_getlines_orig = None
_FILENAME_TO_SRC: dict[str, list[str]] = {}

# Materializing the codegen to disk can be useful for external tools, e.g. ncu
# Disabled by default because writing to disk at module import time is unexpected and error-prone.
_should_materialize_codegen = os.environ.get("TTFS_MATERIALIZE_CODEGEN") == "1"
_should_keep_materialized_source = os.environ.get("TTFS_KEEP_CODEGEN") == "1"
_tmp_dir = None


def _monkey_patched_getlines(filename, module_globals=None):
    if filename in _FILENAME_TO_SRC:
        return _FILENAME_TO_SRC[filename]
    else:
        return _getlines_orig(filename, module_globals)  # type: ignore

def _get_new_src(fn, sig, agg_types: dict[str, Any]) -> tuple[str, Optional[Callable[[Any, dict], None]]]:
    try:
        raw_src, starting_line_number = inspect.getsourcelines(fn)
    except OSError as e:
        raise ValueError("@jit functions should be defined in a Python file") from e
    # function source code (without decorators)
    src = textwrap.dedent("".join(raw_src))
    src = src[re.search(r"^def\s+\w+\s*\(", src, re.MULTILINE).start():]
    tree = ast.parse(src)
    fn_node = tree.body[0]
    assert isinstance(fn_node, ast.FunctionDef)
    agg_types_root_name: dict[str, str] = {}
    new_args: list[ast.arg] = []
    new_sig_params: list[inspect.Parameter] = []
    lines: list[str] = []
    agg_names: list[str] = []
    for arg in fn_node.args.args:
        if arg.arg in agg_types:
            agg_names.append(arg.arg)
            assert arg.annotation is not None 
            agg_types_root_name[arg.arg] = ast.unparse(arg.annotation)
            # replace arg
            agg_type = agg_types[arg.arg]
            flatted_agg_args = _create_flatten_agg_arg_nodes(agg_type, arg.arg)
            
            sig_param = sig.parameters[arg.arg]
            new_args.extend([x[0] for x in flatted_agg_args])
            for flat_arg, is_constexpr in flatted_agg_args:
                new_sig_params.append(
                    inspect.Parameter(
                        name=flat_arg.arg,
                        kind=sig_param.kind,
                        annotation=tl.constexpr if is_constexpr else inspect.Parameter.empty,
                    )
                )
            fn_node.body.insert(0, ast.Assign(
                targets=[
                    ast.Name(id=arg.arg, ctx=ast.Store()),
                ],
                value=_create_agg_from_flatten_args_ast_node(agg_type, agg_types_root_name[arg.arg], arg.arg),
            ))
            lines.extend(_get_flatten_agg_to_kwargs_lines(agg_type, arg.arg))
        else:
            new_args.append(arg)
            new_sig_params.append(sig.parameters[arg.arg])
    new_sig = sig.replace(parameters=new_sig_params)
    fn_node.args.args = new_args
    new_src = ast.unparse(ast.fix_missing_locations(tree))
    lines = [textwrap.indent(line, "    ") for line in lines]
    flatten_fn = None
    if lines:
        flatten_arg_code = f"""
def flatten_fn(kwargs, {", ".join(agg_names)}):
{'\n'.join(lines)}
        """
        # print(flatten_arg_code)
        code_obj = compile(flatten_arg_code, "<flatten_agg_to_kwargs>", "exec")
        local_vars: dict[str, Any] = {}
        import tensorpc.apps.ttfs as ttfs

        exec(code_obj, {"ttfs": ttfs}, local_vars)
        flatten_fn = local_vars["flatten_fn"]
    return new_src, flatten_fn

@overload
def triton_jitx(fn: _T) -> triton.JITFunction[_T]:
    ...


@overload
def triton_jitx(
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
    is_gluon_jit: bool = False,
) -> Callable[[_T], triton.JITFunction[_T]]:
    ...


def triton_jitx(
    fn: Optional[_T] = None,
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
    is_gluon_jit: bool = False,
) -> Union[triton.JITFunction[_T], Callable[[_T], triton.JITFunction[_T]]]:
    """
    Decorator for JIT-compiling a function using the Triton compiler with host aggregate argument support.

    You must annotate the aggregate argument type if you want to use aggregate argument.

    WARNING: we will mangle field name of aggregate type to generate flattened argument names, so
    field names in `META` of grid function and autotune configs will be mangled.
    don't pass any tuneable argument by aggregate type.

    Example:
    ```python
    @ttfs.aggregate
    class Args:
        a: tl.tensor
        b: tl.constexpr

    @ttfs.triton_jitx_kernel
    def kernel(args: Args, BLOCK_M: tl.constexpr):
        tl.static_print(args)
    a = torch.randn(16, 16).cuda()
    args = Args(
        a=a,
        b=42,
    )
    kernel[(1,)](args=args, BLOCK_M=8)

    ```

    :note: When a jit'd function is called, arguments are
        implicitly converted to pointers if they have a :code:`.data_ptr()` method
        and a `.dtype` attribute.

    :note: This function will be compiled and run on the GPU. It will only have access to:

           * python primitives,
           * builtins within the triton package,
           * arguments to this function,
           * other jit'd functions

    :param fn: the function to be jit-compiled
    :type fn: Callable
    """

    from triton import knobs
    def decorator(fn: _T) -> triton.JITFunction[_T]:
        global _FILENAME_TO_SRC, _getlines_orig, _tmp_dir

        assert callable(fn)
        if knobs.runtime.interpret or is_gluon_jit:
            raise NotImplementedError
            # from triton.runtime.interpreter import InterpretedFunction
            # return InterpretedFunction(fn, version=version, do_not_specialize=do_not_specialize,
            #                            do_not_specialize_on_alignment=do_not_specialize_on_alignment, debug=debug,
            #                            noinline=noinline, repr=repr, launch_metadata=launch_metadata)
        else:
            # parse aggregate types in function signature
            # assert "ttfs" in fn.__globals__, "you must import ttfs to global namespace before using jitx."
            sig = inspect.signature(fn)
            has_agg = False
            agg_types: dict[str, Any] = {}
            agg_args: list[tuple[int, str]] = []
            for i, param in enumerate(sig.parameters.values()):
                param_type = param.annotation
                if is_aggregate_type(param_type):
                    has_agg = True 
                    agg_types[param.name] = param_type
                    agg_args.append((i, param.name))
            if is_gluon_jit:
                jf = GluonJITFunction(
                    fn,
                    version=version,
                    do_not_specialize=do_not_specialize,
                    do_not_specialize_on_alignment=do_not_specialize_on_alignment,
                    debug=debug,
                    noinline=noinline,
                    repr=repr,
                    launch_metadata=launch_metadata,
                )
            else:
                jf = triton.JITFunction(
                    fn,
                    version=version,
                    do_not_specialize=do_not_specialize,
                    do_not_specialize_on_alignment=do_not_specialize_on_alignment,
                    debug=debug,
                    noinline=noinline,
                    repr=repr,
                    launch_metadata=launch_metadata,
                )
            if not has_agg: 
                return jf
            new_src, flatten_fn = _get_new_src(fn, sig, agg_types)
            fn_basename = f"ttfs-{fn.__name__}"
            is_monkey_patched = False
            getlines_orig = None
            if _should_materialize_codegen:
                if not _tmp_dir:
                    _tmp_dir = tempfile.TemporaryDirectory()
                fn_filename = os.path.join(_tmp_dir.name, f"{fn_basename}.py")
                if _should_keep_materialized_source:
                    # destroy the TemporaryDirectory object
                    _tmp_dir = None
                    # create path if not exists
                    os.makedirs(os.path.dirname(fn_filename), exist_ok=True)
                with open(fn_filename, "w") as f:
                    f.write(new_src)
            else:
                # Patch `getlines` only the first time
                if not _FILENAME_TO_SRC:
                    getlines_orig = linecache.getlines
                    linecache.getlines = _monkey_patched_getlines
                fn_filename = f"<{fn_basename}>"
                _FILENAME_TO_SRC[fn_filename] = new_src.splitlines(keepends=True)
                is_monkey_patched = True
            try:
                code = compile(new_src, fn_filename, "exec")

                _locals: dict[str, Any] = {}
                gbs = fn.__globals__.copy()
                import tensorpc.apps.ttfs as ttfs
                gbs["ttfs"] = ttfs
                exec(code, fn.__globals__, _locals)
                assert len(_locals) == 1, len(_locals)
                fn = next(iter(_locals.values()))
                # print(new_src)
                if is_gluon_jit:

                    jf = GluonJITFunctionEx(
                        fn,
                        version=version,
                        do_not_specialize=do_not_specialize,
                        do_not_specialize_on_alignment=do_not_specialize_on_alignment,
                        debug=debug,
                        noinline=noinline,
                        repr=repr,
                        launch_metadata=launch_metadata,
                    )
                else:
                    jf = JITFunctionEx(
                        fn,
                        version=version,
                        do_not_specialize=do_not_specialize,
                        do_not_specialize_on_alignment=do_not_specialize_on_alignment,
                        debug=debug,
                        noinline=noinline,
                        repr=repr,
                        launch_metadata=launch_metadata,
                    )
                # jf._unsafe_update_src(new_src)
                # jf.signature = new_sig
                jf._ttfs_set_agg_metas(_JitExMeta(
                    agg_names=list(agg_types.keys()),
                    agg_metas=agg_types,
                    original_sig=sig,
                    flatten_fn=flatten_fn,
                ))
            finally:
                if is_monkey_patched:
                    # Un-patch `getlines`
                    _FILENAME_TO_SRC.pop(fn_filename)
                    if not _FILENAME_TO_SRC:
                        linecache.getlines = getlines_orig  # type: ignore
            return jf

    if fn is not None:
        return decorator(fn)

    else:
        return decorator

@overload
def gluon_jitx(fn: _T) -> GluonJITFunction[_T]:
    ...


@overload
def gluon_jitx(
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Callable[[_T], GluonJITFunction[_T]]:
    ...


def gluon_jitx(
    fn: Optional[_T] = None,
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Union[GluonJITFunction[_T], Callable[[_T], GluonJITFunction[_T]]]:
    res = triton_jitx( 
        version=version,
        repr=repr,
        launch_metadata=launch_metadata,
        do_not_specialize=do_not_specialize,
        do_not_specialize_on_alignment=do_not_specialize_on_alignment,
        debug=debug,
        noinline=noinline,
        is_gluon_jit=True,
    )
    if fn is None:
        return cast(Callable[[_T], GluonJITFunction[_T]], res)
    else:
        return cast(GluonJITFunction[_T], res(fn))

class AutotunerEx(Autotuner):
    def run(self, *args, **kwargs):
        meta = self.fn._ttfs_jit_ex_meta
        if meta.flatten_fn is not None:
            sig = meta.original_sig
            kwargs = _expand_agg_in_args_kwargs_v2(sig, meta.flatten_fn, args, kwargs, self._ttfs_jit_ex_meta.agg_names)
            return super().run(**kwargs)
        else:
            return super().run(*args, **kwargs)

def autotunex(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None,
             warmup=None, rep=None, use_cuda_graph=False, do_bench=None, cache_results=False):

    def decorator(fn):
        return AutotunerEx(fn, fn.arg_names, configs, key, reset_to_zero, restore_value, pre_hook=pre_hook,
                         post_hook=post_hook, prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
                         use_cuda_graph=use_cuda_graph, do_bench=do_bench, cache_results=cache_results)

    return decorator
