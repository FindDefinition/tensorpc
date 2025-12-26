import asyncio
import concurrent.futures
import contextlib
import enum
from functools import partial
import inspect
import math
import importlib.machinery
import concurrent
import multiprocessing
import subprocess
from tempfile import NamedTemporaryFile
import time
import traceback
import types
import triton
from tensorpc.apps.mls.logger import LOGGER
from tensorpc.apps.mls.tsim.core import TensorSimConfig, TensorSimMode, get_tensorsim_context
from tensorpc.core import pfl
import numpy as np
import dataclasses
from tensorpc.core import dataclass_dispatch as dataclasses_pydantic
from typing import Annotated, Any, Callable, ClassVar, Optional, Type, TypeAlias, TypeVar, Union, cast, get_type_hints, overload
from tensorpc.apps.mls import tsim
from tensorpc.apps.mls.tsim import DTypeEnum
import triton.language as tl
from triton.compiler import CompiledKernel
from tensorpc.core.annolib import Undefined
from tensorpc.core.moduleid import get_object_type_from_module_id
from tensorpc.core.pfl.core import PFLInlineRunEnv
from tensorpc.dock.client import list_all_app_in_machine, list_all_running_apps_in_relay
from tensorpc.utils.package_finder import find_submodule_from_file

from .std import PointerScalarFloat, PointerScalarInt, PointerTensor, TensorDescriptor, Tensor, ConstExpr
import rich.progress

TRITON_VERSION_TUPLE = tuple(int(x) for x in triton.__version__.split(".")[:2])

T = TypeVar("T")

@dataclasses.dataclass
class HostTensorDescriptor:
    data: np.ndarray
    block_shape: list[int]

def _triton_anno_transform(inferred: pfl.PFLExprInfo,
                           anno_in_ast: Any,
                           default: Union[Undefined, Any]) -> pfl.PFLExprInfo:
    if anno_in_ast is tl.constexpr:
        # handle arg: tl.constexpr = xxx
        # if not isinstance(default, Undefined):
        #     pass 
        # print("_triton_anno_transform", anno_in_ast, inferred)
        inferred = dataclasses.replace(inferred)
        inferred.anno_metadatas_ext.append(ConstExpr())
    return inferred



@dataclasses_pydantic.dataclass
class TritonSimFuncMeta(pfl.PFLCompileFuncMeta):
    sim_kwargs: Optional[dict[str, Any]] = None
    real_kwargs: Optional[dict[str, Any]] = None
    raw_fn: Optional[Optional[Callable[[dict[str, Any]], float]]] = None

@overload
def mark_triton_compilable(fn: T) -> T:
    ...


@overload
def mark_triton_compilable(
        fn: None = None,
        *,
        inline_run_env_fn: Optional[Callable[..., pfl.PFLInlineRunEnv]] = None,
        is_template: bool = False,
        sim_kwargs: Optional[dict[str, Any]] = None,
        real_kwargs: Optional[dict[str, Any]] = None,
        raw_fn: Optional[Callable[[dict[str, Any]], Any]] = None) -> Callable[[T], T]:
    ...


@pfl.register_pfl_std(mapped_name="triton_compiler_mark_pfl_compilable",
                      backend=None,
                      _internal_disable_type_check=True)
def mark_triton_compilable(
        fn: Optional[T] = None,
        *,
        inline_run_env_fn: Optional[Callable[..., pfl.PFLInlineRunEnv]] = None,
        is_template: bool = False,
        sim_kwargs: Optional[dict[str, Any]] = None,
        real_kwargs: Optional[dict[str, Any]] = None,
        raw_fn: Optional[Callable[[dict[str, Any]], Any]] = None) -> Union[T, Callable[[T], T]]:
    
    def wrapper(fn_wrapped: T) -> T:
        prev_meta: Optional[TritonSimFuncMeta] = getattr(
            fn_wrapped, pfl.PFL_COMPILE_META_ATTR, None)
        inline_run_env_fn_ = inline_run_env_fn
        # get constexpr args
        sig = inspect.signature(cast(Callable, fn_wrapped))
        constexpr_args_set = set()
        for param in sig.parameters.values():
            if param.annotation is tl.constexpr:
                constexpr_args_set.add(param.name)
        if not constexpr_args_set:
            constexpr_args = None
        else:
            constexpr_args = constexpr_args_set
        if prev_meta is None:
            prev_meta = TritonSimFuncMeta(["triton"],
                                          inline_run_env_fn_,
                                          is_template=is_template,
                                          sim_kwargs=sim_kwargs,
                                          real_kwargs=real_kwargs,
                                          raw_fn=raw_fn,
                                          constexpr_args=constexpr_args)
            setattr(fn_wrapped, pfl.PFL_COMPILE_META_ATTR, prev_meta)
        else:
            prev_meta.backends = ["triton"]
            prev_meta.inline_run_env_fn = inline_run_env_fn_
            prev_meta.is_template = is_template
            prev_meta.sim_kwargs = sim_kwargs
            prev_meta.real_kwargs = real_kwargs
            prev_meta.raw_fn = raw_fn
            prev_meta.constexpr_args = constexpr_args
            prev_meta.validate()

        return cast(T, fn_wrapped)

    if fn is None:
        return wrapper
    else:
        return wrapper(fn)


def create_global_mem_from_kwargs(
        kwargs: dict[str, Any]) -> tsim.SimMemoryStorage:
    global_mem_arrays: dict[str, np.ndarray] = {}
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            global_mem_arrays[k] = v
        elif isinstance(v, HostTensorDescriptor):
            global_mem_arrays[k] = v.data
    return tsim.create_sim_memory(global_mem_arrays)

@dataclasses.dataclass
class TritonSimInfo:
    grid_size: tuple[int, int, int]
    ref_results: dict[str, Any]

    global_mem: Optional[tsim.SimMemoryStorage] = None
    # for visualization only.
    # support identifier or identifier.T (transposed) or None (empty space)
    vis_layout: Optional[list[list[Optional[str]]]] = None
    grid_size_for_triton: Optional[Callable[[Any], tuple[int, int,
                                                         int]]] = None
    postprocess_fn: Optional[Callable[[str, np.ndarray], Any]] = None
    kwargs_for_raw: Optional[dict[str, Any]] = None

class TritonSimExecType(enum.IntEnum):
    SIM = 0
    REAL_TRITON = 1
    REAL_RAW = 2

def _validate_and_convert_triton_kwargs(
    kwargs: dict[str, Any],
    global_mem: Optional[tsim.SimMemoryStorage] = None
) -> tuple[dict[str, Any], Optional[tsim.SimMemoryStorage]]:
    new_kwargs: dict[str, Any] = {}
    if global_mem is None:
        global_mem = create_global_mem_from_kwargs(kwargs)
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            # check v is contiguous
            v_dtype_tsim = DTypeEnum.from_numpy_dtype(v.dtype)
            assert global_mem is not None
            desc = global_mem.memory_blocks[k]
            ptr_float = desc.byte_offset_with_hole // desc.dtype.byte_size()
            ts = tsim.create_pointer_scalar(v_dtype_tsim, ptr_float,
                                            global_mem)
            if ts.is_floating():
                v_ptr = PointerScalarFloat(ts)
            else:
                v_ptr = PointerScalarInt(ts)
            new_kwargs[k] = v_ptr
        elif isinstance(v, HostTensorDescriptor):
            v_dtype_tsim = DTypeEnum.from_numpy_dtype(v.data.dtype)
            assert global_mem is not None
            desc = global_mem.memory_blocks[k]
            ptr_float = desc.byte_offset_with_hole // desc.dtype.byte_size()
            ts = tsim.create_pointer_scalar(v_dtype_tsim, ptr_float,
                                            global_mem)
            tsd = tsim.create_tensor_block_pointer(ts, list(
                v.data.shape), [s // v.data.itemsize for s in v.data.strides],
                                                   v.block_shape)
            new_kwargs[k] = TensorDescriptor(tsd)
        elif isinstance(v, PointerTensor):
            new_kwargs[k] = v
        elif isinstance(v, TensorDescriptor):
            new_kwargs[k] = v
        elif isinstance(v, (PointerScalarFloat, PointerScalarInt)):
            new_kwargs[k] = v
        else:
            # TODO add support for string constexpr
            assert isinstance(
                v, (int, float,
                    bool, str, types.NoneType)), f"Unsupported type {type(v)} for triton kwargs."
            new_kwargs[k] = v

    return new_kwargs, global_mem

def _convert_triton_real_kwargs_to_np(
    kwargs: dict[str, Any]
) -> dict[str, Any]:
    import torch
    from triton.tools.tensor_descriptor import TensorDescriptor as TTTensorDescriptor

    new_kwargs: dict[str, Any] = {}
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.bfloat16:
                v = v.to(torch.float32)
            new_kwargs[k] = v.cpu().numpy()
        elif isinstance(v, TTTensorDescriptor):
            data = v.base 
            if data.dtype == torch.bfloat16:
                data = data.to(torch.float32)
            new_kwargs[k] = HostTensorDescriptor(data.cpu().numpy(), v.block_shape)
        else:
            # TODO add support for string constexpr
            assert isinstance(
                v, (int, float,
                    bool, str, types.NoneType)), f"Unsupported type {type(v)} for triton kwargs."
            new_kwargs[k] = v
    return new_kwargs

def _create_metadata_from_triton_kwargs(
        kwargs: dict[str, Any]) -> dict[str, Any]:
    new_kwargs: dict[str, Any] = {}
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            # check v is contiguous
            v_dtype_tsim = DTypeEnum.from_numpy_dtype(v.dtype)
            ts = tsim.create_pointer_scalar_meta(v_dtype_tsim)
            if ts.is_floating():
                v_ptr = PointerScalarFloat(ts)
            else:
                v_ptr = PointerScalarInt(ts)
            new_kwargs[k] = v_ptr
        elif isinstance(v, HostTensorDescriptor):
            v_dtype_tsim = DTypeEnum.from_numpy_dtype(v.data.dtype)
            ts = tsim.create_pointer_scalar_meta(v_dtype_tsim)
            tsd = tsim.create_tensor_block_pointer(ts, list(
                v.data.shape), [s // v.data.itemsize for s in v.data.strides],
                                                   v.block_shape)
            new_kwargs[k] = TensorDescriptor(tsd)
        elif isinstance(v, PointerTensor):
            new_kwargs[k] = dataclasses.replace(
                v, _wrapped=v._wrapped.to_meta_tensor())
        elif isinstance(v, TensorDescriptor):
            new_kwargs[k] = dataclasses.replace(
                v, _wrapped=v._wrapped.to_meta_tensor())
        elif isinstance(v, (PointerScalarFloat, PointerScalarInt)):
            new_kwargs[k] = dataclasses.replace(
                v, _wrapped=v._wrapped.to_meta_tensor())
        else:
            # TODO add support for string constexpr
            assert isinstance(
                v, (int, float,
                    bool, str, types.NoneType)), f"Unsupported type {type(v)} for triton kwargs {k}."
            new_kwargs[k] = v

    return new_kwargs


def _handle_triton_inline_data(
        inline_run_env_fn: Callable[[], pfl.PFLInlineRunEnv]):
    env = inline_run_env_fn()
    # convert test data to triton sim
    prev_global_mem = env.get_userdata_typed(TritonSimInfo).global_mem
    new_kwargs, global_mem = _validate_and_convert_triton_kwargs(
        env.kwargs, prev_global_mem)
    env.kwargs = new_kwargs
    if prev_global_mem is None:
        env.get_userdata_typed(TritonSimInfo).global_mem = global_mem
    return env



@dataclasses_pydantic.dataclass
class TritonInlineRunEnv(PFLInlineRunEnv):
    raw_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)


class TritonKernelRunner(pfl.PFLAsyncRunner):

    def __init__(self, library: pfl.PFLLibrary, inline_env: PFLInlineRunEnv):
        super().__init__(library)
        self.triton_sim_info = inline_env.get_userdata_typed(TritonSimInfo)
        self._init_inline_env = inline_env

    def copy(self):
        return TritonKernelRunner(self._library, self._init_inline_env)

    def get_unwrapped_triton_fn(self, key: Callable) -> Callable:
        return may_triton_func(key)

    def get_triton_fn_inline_env(
            self, key: Union[str, Callable], exec_type: TritonSimExecType) -> TritonInlineRunEnv:
        if not isinstance(key, str):
            key = may_triton_func(key)
        kwargs = None 
        specs = self._library.get_compiled_unit_specs(key)
        assert len(specs) == 1 
        fn_meta = specs[0].compile_info.meta
        assert isinstance(fn_meta, TritonSimFuncMeta)
        if exec_type == TritonSimExecType.SIM:
            kwargs = fn_meta.sim_kwargs
        elif exec_type == TritonSimExecType.REAL_TRITON:
            kwargs = fn_meta.real_kwargs
        else:
            raise ValueError(f"Unsupported exec_type {exec_type}") 
        res = self._library.get_compiled_unit_inline_env(key, kwargs)
        sim_info = res.get_userdata_typed(TritonSimInfo)
        kwargs, gmem = _validate_and_convert_triton_kwargs(
            res.kwargs, sim_info.global_mem)
        sim_info = dataclasses.replace(sim_info, global_mem=gmem)
        return TritonInlineRunEnv(annotations=res.annotations,
                                  contexts=res.contexts,
                                  kwargs=kwargs,
                                  raw_kwargs=res.kwargs,
                                  userdata=sim_info)

    async def run_kernel(self, fn: Any, grid_size: tuple[int, int, int],
                         global_mem: tsim.SimMemoryStorage, kwargs: dict[str, Any], reverse_grid: bool = True) -> None:
        fn_no_jit = may_triton_func(fn)

        lib = self._library
        kwargs, _ = _validate_and_convert_triton_kwargs(kwargs, global_mem)
        jkl_arr_tuple = np.meshgrid(
            range(grid_size[0]), range(grid_size[1]), range(grid_size[2]),
            indexing="ij")
        jkl_arr = np.stack(jkl_arr_tuple, axis=-1).reshape(
            -1, 3)
        if reverse_grid:
            jkl_arr = jkl_arr[::-1]
        for jkl in jkl_arr:
            j = int(jkl[0])
            k = int(jkl[1])
            l = int(jkl[2])
            with tsim.enter_tensorsim_context([j, k, l],
                                                grid_size,
                                                global_mem=global_mem):
                kwargs_cloned = kwargs.copy()
                for key, v in kwargs_cloned.items():
                    if isinstance(
                            v, (PointerTensor, PointerScalarFloat,
                                PointerScalarInt, TensorDescriptor)):
                        kwargs_cloned[key] = v.clone()
                await self.run_func(
                    lib.get_compiled_unit_specs(fn_no_jit)[0].uid,
                    kwargs_cloned)

    def _run_kernel_in_executor(self, fn: Any, grid_size: tuple[int, int, int],
                                global_mem: tsim.SimMemoryStorage,
                                kwargs) -> None:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            self.run_kernel(fn, grid_size, global_mem, kwargs))
        loop.close()
        return

    def run_kernel_in_executor(self, fn: Any, grid_size: tuple[int, int, int],
                               global_mem: tsim.SimMemoryStorage,
                               **kwargs) -> None:
        fn_no_jit = may_triton_func(fn)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self._run_kernel_in_executor, fn_no_jit,
                                     grid_size, global_mem, kwargs)
            future.result()

    async def run_kernel_test(self, fn: Any, reverse_grid: bool = True, external_inline_env: Optional[PFLInlineRunEnv] = None) -> TritonSimInfo:
        fn_no_jit = may_triton_func(fn)
        if external_inline_env is not None:
            inline_env = external_inline_env
        else:
            inline_env = self.get_triton_fn_inline_env(fn_no_jit, TritonSimExecType.SIM)
        sim_info = inline_env.get_userdata_typed(TritonSimInfo)
        assert sim_info.global_mem is not None
        await self.run_kernel(fn_no_jit, sim_info.grid_size,
                              sim_info.global_mem, inline_env.kwargs,
                              reverse_grid)
        return sim_info

    async def validate_kernel_by_test_data(self,
                                           fn: Any,
                                           res_comparer: Optional[Callable[
                                               [Any, Any], Any]] = None,
                                           run_triton: bool = False,
                                           external_inline_env: Optional[PFLInlineRunEnv] = None) -> None:

        fn_no_jit = may_triton_func(fn)
        if external_inline_env is not None:
            inline_env = external_inline_env
        else:
            inline_env = self.get_triton_fn_inline_env(fn_no_jit, TritonSimExecType.SIM)
        sim_info = inline_env.get_userdata_typed(TritonSimInfo)
        assert sim_info.global_mem is not None
        await self.run_kernel(fn_no_jit, sim_info.grid_size,
                              sim_info.global_mem, inline_env.kwargs)
        if sim_info.global_mem is not None:
            for k, ref in sim_info.ref_results.items():
                desc = sim_info.global_mem.memory_blocks[k]
                res = desc.get_data_view_checked()
                if sim_info.postprocess_fn is not None:
                    res = sim_info.postprocess_fn(k, res)
                print(k, np.linalg.norm(res - ref))
                if res_comparer is not None:
                    res_comparer(res, ref)
        if run_triton:
            await self.validate_kernel_in_triton_process(fn)

    def _get_triton_run_args(self, fn: Any):
        fn_no_jit = may_triton_func(fn)
        path = inspect.getfile(fn_no_jit)
        module_import_path = find_submodule_from_file(path)
        is_func_id = True
        func_id_or_path = path
        if module_import_path is None:
            is_func_id = False
        else:
            func_id_local = "::".join(fn_no_jit.__qualname__.split("."))
            func_id_or_path = f"{module_import_path}::{func_id_local}"
        return func_id_or_path,  is_func_id, fn_no_jit.__name__

    async def validate_kernel_in_triton_process(self, fn: Any):
        # TODO: this don't support ptr of ptr in real triton sim currently.
        assert isinstance(
            fn, (triton.JITFunction,
                 triton.runtime.Autotuner)), "fn must be a triton JITFunction"
        func_id_or_path, is_func_id, fn_name = self._get_triton_run_args(fn)
        spawn_ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
                1, mp_context=spawn_ctx) as ex:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(ex,
                                        _run_triton_fn_for_validation,
                                        func_id_or_path,
                                        fn_name, is_func_id)

    async def bench_kernel_in_triton_process(self, fn: Any,
                                            warming_up: int = 2,
                                            run_cnt: int = 3,
                                            override_kwargs_set: Optional[dict[str, dict[str, Any]]] = None,
                                            ext_inline_env: Optional[TritonInlineRunEnv] = None):
        # TODO: this don't support ptr of ptr in real triton sim currently.
        assert isinstance(
            fn, (triton.JITFunction,
                 triton.runtime.Autotuner)), "fn must be a triton JITFunction"
        func_id_or_path, is_func_id, fn_name = self._get_triton_run_args(fn)
        spawn_ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
                1, mp_context=spawn_ctx) as ex:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(ex,
                                              _run_triton_fn_for_bench,
                                              func_id_or_path,
                                              fn_name, is_func_id,
                                              warming_up,
                                              run_cnt,
                                              override_kwargs_set,
                                              ext_inline_env)

def _get_triton_fn(func_id_or_path: str, fn_name: str, is_func_id: bool):
    if is_func_id:
        fn = get_object_type_from_module_id(func_id_or_path)
        assert fn is not None
    else:
        module = types.ModuleType(func_id_or_path)
        spec = importlib.machinery.ModuleSpec(func_id_or_path,
                                              None,
                                              origin=func_id_or_path)
        module.__spec__ = spec

        with open(func_id_or_path, "r") as f:
            code = f.read()
        code_comp = compile(code, func_id_or_path, "exec")
        module.__file__ = func_id_or_path
        exec(code_comp, module.__dict__)
        fn = module.__dict__[fn_name]
    return fn

def _get_triton_run_datas_from_path(func_id_or_path: str, fn_name: str,
                             is_func_id: bool, override_kwargs: Optional[dict[str, Any]] = None,
                             ext_inline_env: Optional[TritonInlineRunEnv] = None):
    import torch
    fn = _get_triton_fn(func_id_or_path, fn_name, is_func_id)
    fn_unwrapped = may_triton_func(fn)
    fn_sig = inspect.signature(fn_unwrapped)
    all_triton_param_names = set(fn_sig.parameters.keys())
    fn_meta = None 
    if ext_inline_env is None:
        fn_meta = pfl.get_compilable_meta(fn_unwrapped)
        assert isinstance(fn_meta, TritonSimFuncMeta), "fn must be a triton compilable function"
        assert fn_meta.inline_run_env_fn is not None, "inline_run_env_fn must be set for real run."
        if fn_meta.real_kwargs is not None:
            inline_env = fn_meta.inline_run_env_fn(**fn_meta.real_kwargs)
        else:
            inline_env = fn_meta.inline_run_env_fn()
        kwargs = inline_env.kwargs
    else:
        inline_env = ext_inline_env
        kwargs = inline_env.raw_kwargs
    kwargs = kwargs.copy()
    if override_kwargs is not None:
        kwargs.update(override_kwargs)
    if isinstance(fn, triton.runtime.Autotuner):
        for cfg in fn.configs:
            for k_to_rm in cfg.kwargs:
                kwargs.pop(k_to_rm, None)
            break
    sim_info = inline_env.get_userdata_typed(TritonSimInfo)
    # TODO support tensor descriptor and host tensor descriptor
    kwargs_triton: dict[str, torch.Tensor] = {}
    kwargs_torch: dict[str, torch.Tensor] = {}
    reserved_triton_param_names = {
        "maxnreg", "num_warps", "num_stages", "num_ctas"
    }
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            kwargs_triton[k] = torch.from_numpy(v).cuda()
            kwargs_torch[k] = kwargs_triton[k]
        elif isinstance(v, HostTensorDescriptor):
            # only support triton 3.4 and sm_90 above.
            from triton.tools.tensor_descriptor import TensorDescriptor
            v_th = torch.from_numpy(v.data).cuda()
            desc = TensorDescriptor(v_th, shape=v.data.shape, strides=v_th.stride(), block_shape=v.block_shape)
            kwargs_triton[k] = desc
            kwargs_torch[k] = v_th
        else:
            if k in all_triton_param_names or k in reserved_triton_param_names:
                kwargs_triton[k] = v

            kwargs_torch[k] = v
    if sim_info.kwargs_for_raw is not None:
        kwargs_torch.update(sim_info.kwargs_for_raw)
    return fn, fn_meta, kwargs_triton, kwargs_torch, inline_env

@dataclasses.dataclass
class TritonKernelCompileInfo:
    asm: dict[str, str]
    metadata: dict[str, Any]
    best_config: Any = None

def get_triton_compile_infos(kernel, kwargs_tt, log_ptxas_info: bool = False) -> TritonKernelCompileInfo:
    from triton.runtime.driver import driver

    device = driver.active.get_current_device()
    best_config = {}
    if isinstance(kernel, triton.runtime.Autotuner):
        best_config = kernel.best_config.all_kwargs()
        kernel = kernel.fn
    if hasattr(kernel, "device_caches"):
        caches = kernel.device_caches[device]
    else:
        caches = kernel.cache
    if len(caches) == 5:
        cache_entry, _, target, backend, binder = caches
    else:
        cache_entry, target, backend, binder = caches
    bound_args, specialization, options = binder(**kwargs_tt, **best_config, debug=False)
    # compute cache key
    key = str(specialization) + str(options)
    # Get the first cached compilation (or iterate over all)
    compiled_kernel = cache_entry[key]

    res = get_triton_compile_info_from_res(compiled_kernel, log_ptxas_info)
    res.best_config = best_config
    return res

def get_triton_compile_info_from_res(compiled_kernel: CompiledKernel, log_ptxas_info: bool = False) -> TritonKernelCompileInfo:
    # Access the 'asm' dictionary
    asm_dict = compiled_kernel.asm.copy()
    for k in list(asm_dict.keys()):
        value = asm_dict[k]
        if not isinstance(value, str):
            # e.g. cubin
            asm_dict.pop(k)
    # List all keys in the 'asm' dictionary
    metadata_dict = compiled_kernel.metadata._asdict().copy()
    metadata_dict.pop("target")
    metadata_dict["n_regs"] = compiled_kernel.n_regs
    metadata_dict["n_spills"] = compiled_kernel.n_spills
    metadata_dict["n_max_threads"] = compiled_kernel.n_max_threads
    # metadata_dict["arch_number"] = compiled_kernel.metadata.target.arch

    if log_ptxas_info:
        with NamedTemporaryFile(suffix=".ptx") as f:
            f.write(asm_dict["ptx"].encode("utf-8"))
            arch = compiled_kernel.metadata.target.arch
            if arch == 90:
                arch_str = "sm_90a"
            else:
                arch_str = f"sm_{arch}"
            out = subprocess.check_output(
                ["ptxas", f.name, "-v", "--warn-on-spills", f"-arch={arch_str}"],
                stderr=subprocess.STDOUT,
                universal_newlines=True) 
            print(out)

    return TritonKernelCompileInfo(asm_dict, metadata_dict)

def _run_triton_fn_for_validation(func_id_or_path: str, fn_name: str,
                                  is_func_id: bool):
    fn, fn_meta, kwargs_tt, kwargs_th, inline_env = _get_triton_run_datas_from_path(
        func_id_or_path, fn_name, is_func_id)
    sim_info = inline_env.get_userdata_typed(TritonSimInfo)
    if sim_info.grid_size_for_triton is not None:
        fn[sim_info.grid_size_for_triton](**kwargs_tt)
    else:
        fn[sim_info.grid_size](**kwargs_tt)
    # compare result here.
    run_info = get_triton_compile_infos(fn, kwargs_tt)
    if run_info.metadata["n_spills"] > 0:
        LOGGER.warning(f"{run_info.metadata['n_spills']} register spills detected. try eliminate it to increase performance.")
    print(run_info.best_config)
    for k, ref in sim_info.ref_results.items():
        if isinstance(ref, HostTensorDescriptor):
            ref_data = ref.data
        else:
            ref_data = ref
        res = kwargs_th[k].float().cpu().numpy()
        if sim_info.postprocess_fn is not None:
            res = sim_info.postprocess_fn(k, res)
        
        print(
            f"{k}-{ref_data.shape} triton res: {np.linalg.norm(ref_data.astype(np.float32) - res.astype(np.float32))}"
        )
        # error = np.abs(ref_data.astype(np.float32) - res.astype(np.float32))
        # print(error)


def _run_triton_fn_for_bench(func_id_or_path: str,
                             fn_name: str,
                             is_func_id: bool,
                             warming_up: int = 2,
                             run_cnt: int = 3,
                             override_kwargs_set: Optional[dict[str, dict[str, Any]]] = None,
                             ext_inline_env: Optional[TritonInlineRunEnv] = None):
    import torch
    if override_kwargs_set is None:
        override_kwargs_set = {
            "": {}
        }
    durations: list[float] = []
    raw_durations: list[dict[str, float]] = []
    run_info: Optional[TritonKernelCompileInfo] = None
    for kw_set_name, override_kwargs in override_kwargs_set.items():
        fn, fn_meta, kwargs_tt, kwargs_th, inline_env = _get_triton_run_datas_from_path(
            func_id_or_path, fn_name, is_func_id, override_kwargs, ext_inline_env)
        raw_fn = None 
        if fn_meta is not None:
            raw_fn = fn_meta.raw_fn
        sim_info = inline_env.get_userdata_typed(TritonSimInfo)
        grid_size_or_grid_size_fn = sim_info.grid_size_for_triton
        if grid_size_or_grid_size_fn is None:
            grid_size_or_grid_size_fn = sim_info.grid_size
        try:
            for j in range(warming_up):
                fn[grid_size_or_grid_size_fn](**kwargs_tt)
        except:
            traceback.print_exc()
            continue 
        stream = torch.cuda.current_stream()
        for j in range(run_cnt):
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record(stream)
            fn[grid_size_or_grid_size_fn](**kwargs_tt)
            end_ev.record(stream)
            start_ev.synchronize()
            end_ev.synchronize()
            duration = start_ev.elapsed_time(end_ev)
            durations.append(duration)
            print(f"{fn_name} {kw_set_name}(triton) dur: {duration:.3f} ms")
            time.sleep(0.01)
        if raw_fn is not None:
            kwargs_th_ = kwargs_th.copy()
            kwargs_th_["__triton_kwargs__"] = kwargs_tt
            for j in range(warming_up):
                raw_fn(kwargs_th_)
            for j in range(run_cnt):
                duration = raw_fn(kwargs_th_)
                if isinstance(duration, float):
                    duration_dict = {
                        "raw": duration,
                    }
                else:
                    assert isinstance(duration, dict), \
                        "raw_fn must return a float or dict with float values"
                    duration_dict = duration
                raw_durations.append(duration_dict)
                raw_dur_msg = ", ".join(
                    f"{k}: {v:.3f} ms" for k, v in duration_dict.items())
                print(f"{fn_name}(raw) dur: {raw_dur_msg}")
        run_info = get_triton_compile_infos(fn, kwargs_tt)
        if run_info.metadata["n_spills"] > 0:
            LOGGER.warning(f"{run_info.metadata['n_spills']} register spills detected. try eliminate it to increase performance.")
        
        print("nregs:", run_info.metadata["n_regs"], "nspills:", run_info.metadata["n_spills"], 
              "shared:", run_info.metadata["shared"], "cfg", run_info.best_config)
    assert run_info is not None
    return durations, raw_durations, run_info

def may_triton_func(fn: Any) -> Any:
    if isinstance(fn, triton.runtime.Autotuner):
        fn = fn.fn
    if isinstance(fn, triton.JITFunction):
        return fn.fn
    return fn


def _triton_var_preproc(fn: Any) -> pfl.PFLProcessedVarMeta:
    if isinstance(fn, triton.runtime.Autotuner):
        fn = fn.fn
    is_triton_jit = False
    if isinstance(fn, triton.JITFunction):
        is_triton_jit = True
        res = fn.fn
    else:
        res = fn
    proc_res = pfl.default_pfl_var_proc(res)
    fn_metadata = proc_res.compilable_meta
    if is_triton_jit and fn_metadata is None:
        meta = pfl.PFLCompileFuncMeta(is_template=True)
        proc_res.compilable_meta = meta
    return proc_res 

def _tt_assign_check(tgt_meta: Any, src_meta: Any):
    if isinstance(tgt_meta, Tensor) and isinstance(
            src_meta, Tensor):
        assert tgt_meta._wrapped.dtype == src_meta._wrapped.dtype, \
            f"Assigning {src_meta} to {tgt_meta} with different dtype {src_meta._wrapped.dtype} != {tgt_meta._wrapped.dtype}"
        assert tgt_meta._wrapped.shape == src_meta._wrapped.shape, \
            f"Assigning {src_meta} to {tgt_meta} with different shape {src_meta._wrapped.shape} != {tgt_meta._wrapped.shape}"

def parse_triton_compilable_to_runner(
    fn: Union[triton.JITFunction, triton.runtime.Autotuner],
    do_meta_eval: bool = True,
    module_code_path_getter: Optional[Callable[[Any], tuple[str, str]]] = None,
    external_inline_env: Optional[PFLInlineRunEnv] = None,
) -> TritonKernelRunner:
    if isinstance(fn, triton.runtime.Autotuner):
        fn = fn.fn
    fn_unwrapped = inspect.unwrap(fn.fn)

    fn_metadata = pfl.get_compilable_meta(fn_unwrapped)
    if external_inline_env is not None:
        env = external_inline_env
    else:
        assert isinstance(fn_metadata, TritonSimFuncMeta)
        inline_run_env_fn = fn_metadata.inline_run_env_fn
        assert inline_run_env_fn is not None
        if fn_metadata.sim_kwargs is not None:
            env = inline_run_env_fn(**fn_metadata.sim_kwargs)
        else:
            env = inline_run_env_fn()
    meta_args = _create_metadata_from_triton_kwargs(env.kwargs)
    external_annos = {k: type(v) for k, v in meta_args.items()}
    sim_info = env.get_userdata_typed(TritonSimInfo)
    kwargs, gmem = _validate_and_convert_triton_kwargs(
        env.kwargs, sim_info.global_mem)
    sim_info = dataclasses.replace(sim_info, global_mem=gmem)
    env_tt = TritonInlineRunEnv(annotations=env.annotations,
                             contexts=env.contexts,
                             kwargs=kwargs,
                             raw_kwargs=env.kwargs,
                             userdata=sim_info)
    type_hints = get_type_hints(fn_unwrapped)
    constexpr_args = {}
    for k, v in type_hints.items():
        if v is tl.constexpr:
            constexpr_args[k] = env_tt.raw_kwargs[k]
    lib = pfl.parse_func_to_pfl_library(fn.fn,
                                        backend="triton",
                                        external_anno=(external_annos, None),
                                        var_preproc=_triton_var_preproc,
                                        anno_transform=_triton_anno_transform,
                                        module_code_path_getter=module_code_path_getter,
                                        constexpr_args=constexpr_args)
    if do_meta_eval:
        evaluator = pfl.PFLStaticEvaluator.meta_evaulator(lib, assign_check=_tt_assign_check)
        evaluator.eval_total_tree(fn.fn, meta_args)
    return TritonKernelRunner(lib, env_tt)

class TritonRuntimeRunnerNested:
    def __init__(self, grid: Union[tuple[int, ...], Callable[[Any], tuple[int, ...]]], 
            fn: Union[triton.JITFunction, triton.runtime.Autotuner],
            autotune_cfg_idx: int = 0,
            module_code_path_getter: Optional[Callable[[Any], tuple[str, str]]] = None,
            mode: TensorSimMode = TensorSimMode.FULL,
            submit_to_ui: bool = False, 
            submit_ref_results: Optional[dict[str, np.ndarray]] = None,
            submit_asm_to_ui: bool = False):
        self._autotuner: Optional[triton.runtime.Autotuner] = None
        self._autotune_cfg_idx = autotune_cfg_idx
        if isinstance(fn, triton.runtime.Autotuner):
            self._autotuner = fn
            fn = fn.fn
        self.fn = fn
        self.grid = grid
        self._fn_unwrapped = inspect.unwrap(fn.fn)
        sig = inspect.signature(self._fn_unwrapped)
        self._mode = mode

        self._sig = sig
        self._module_code_path_getter = module_code_path_getter
        self._submit_to_ui = submit_to_ui
        if submit_to_ui:
            if submit_ref_results is None:
                submit_ref_results = {}
        self._submit_ref_results = submit_ref_results
        self._submit_asm_to_ui = submit_asm_to_ui

    async def _run_sim(self, *args, **kwargs):
        import torch
        from triton.tools.tensor_descriptor import TensorDescriptor as TTTensorDescriptor
        triton_reserved_param_names = {
            "maxnreg", "num_warps", "num_stages", "num_ctas"
        }
        kwargs_without_reserved = {k: v for k, v in kwargs.items() if k not in triton_reserved_param_names}
        kwargs_reserved = {k: v for k, v in kwargs.items() if k in triton_reserved_param_names}
        mapped_kwargs = self._sig.bind(*args, **kwargs_without_reserved).arguments
        mapped_kwargs.update(kwargs_reserved)
        mapped_kwargs = self._get_autotuner_run_kwargs(mapped_kwargs)
        kwargs_cpu = mapped_kwargs.copy()
        kwargs_cpu = _convert_triton_real_kwargs_to_np(kwargs_cpu)
        kwargs_for_sim, global_mem = _validate_and_convert_triton_kwargs(kwargs_cpu)
        assert global_mem is not None 
        grid_may_fn = self.grid
        if inspect.isfunction(grid_may_fn):
            grid_size = grid_may_fn(mapped_kwargs)
        else:
            grid_size = cast(tuple[int, ...], grid_may_fn)
        if len(grid_size) != 3:
            grid_size = (*grid_size, *[1] * (3 - len(grid_size)))
        if self._submit_to_ui:
            fn_path = inspect.getfile(self._fn_unwrapped)
            fn_lineno = inspect.getsourcelines(self._fn_unwrapped)[1]
            assert self._submit_ref_results is not None 
            sim_info = TritonSimInfo(
                grid_size=(grid_size[0], grid_size[1], grid_size[2]),
                ref_results=self._submit_ref_results,
                global_mem=global_mem,
            )
            inline_env = TritonInlineRunEnv(
                kwargs=kwargs_for_sim,
                userdata=sim_info,
                raw_kwargs=kwargs_cpu)
            app_metas = list_all_app_in_machine()
            for meta in app_metas:
                if meta.module_name == "TritonSim":
                    client = meta.create_client()
                    with client:
                        client.app_chunked_remote_call("launch_external_sim", fn_path, fn_lineno, inline_env)
                    break
        type_hints = get_type_hints(self._fn_unwrapped)
        constexpr_args = {}
        for k, v in type_hints.items():
            if v is tl.constexpr:
                constexpr_args[k] = kwargs_for_sim[k]
        meta_args = _create_metadata_from_triton_kwargs(kwargs_for_sim)
        external_annos = {k: type(v) for k, v in meta_args.items()}
        do_meta_eval = True
        lib = pfl.parse_func_to_pfl_library(self.fn.fn,
                                            backend="triton",
                                            external_anno=(external_annos, None),
                                            var_preproc=_triton_var_preproc,
                                            anno_transform=_triton_anno_transform,
                                            module_code_path_getter=self._module_code_path_getter,
                                            constexpr_args=constexpr_args)
        if do_meta_eval:
            evaluator = pfl.PFLStaticEvaluator.meta_evaulator(lib, assign_check=_tt_assign_check)
            evaluator.eval_total_tree(self.fn.fn, meta_args)
        runner = pfl.PFLAsyncRunner(lib)
        fn_no_jit = may_triton_func(self.fn.fn)
        lib = runner._library
        jkl_arr_tuple = np.meshgrid(
            range(grid_size[0]), range(grid_size[1]), range(grid_size[2]),
            indexing="ij")
        jkl_arr = np.stack(jkl_arr_tuple, axis=-1).reshape(
            -1, 3)
        sim_cfg = TensorSimConfig(mode=self._mode)
        jkl_arr = rich.progress.track(jkl_arr, description="Running Triton Sim...")
        
        with tsim.enter_tensorsim_context([-1, -1, -1],
                                            grid_size,
                                            global_mem=global_mem,
                                            cfg=sim_cfg) as ctx:
            for jkl in jkl_arr:
                j = int(jkl[0])
                k = int(jkl[1])
                l = int(jkl[2])
                ctx.set_grid_id([j, k, l])
                kwargs_cloned = kwargs_for_sim.copy()
                # clone pointer tensor because it may be modified in place
                for key, v in kwargs_cloned.items():
                    if isinstance(
                            v, (PointerTensor, PointerScalarFloat,
                                PointerScalarInt, TensorDescriptor)):
                        kwargs_cloned[key] = v.clone()
                await (runner.run_func(
                    lib.get_compiled_unit_specs(fn_no_jit)[0].uid,
                    kwargs_cloned))
                # import tensorpc 
                # tensorpc.dbg.breakpoint()
        need_run_ker: bool = False
        # print(kwargs_for_sim)
        if self._mode == TensorSimMode.FULL:
            for k, v in mapped_kwargs.items():
                is_written = False
                if k in global_mem.memory_blocks:
                    is_written = global_mem.memory_blocks[k].is_written()
                if is_written:
                    if isinstance(v, torch.Tensor):
                        v.copy_(torch.from_numpy(global_mem.memory_blocks[k].get_data_view_checked())) 
                    elif isinstance(v, TTTensorDescriptor):
                        v.base.copy_(torch.from_numpy(global_mem.memory_blocks[k].get_data_view_checked()))
            if self._submit_asm_to_ui:
                need_run_ker = True 
        else:
            assert self._mode == TensorSimMode.LOGIC_ONLY
            need_run_ker = True
            # for logic only, we run real triton kernel instead of sim.
            # logic only is used to check memory access and detect invalid
            # reduce on empty value.
        if need_run_ker:
            if self._autotuner is not None:
                compiled_ker = self._autotuner[self.grid](*args, **kwargs)
            else:
                compiled_ker = self.fn[self.grid](*args, **kwargs)
            if self._submit_asm_to_ui:
                submit_compiled_kernel_to_ui(compiled_ker)

    def __call__(self, *args, **kwargs):
        try:
            asyncio.run(self._run_sim(*args, **kwargs))
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, exiting.")
            raise 

    def _get_autotuner_run_kwargs(self, mapped_kwargs):
        if self._autotuner is None:
            return mapped_kwargs
        if hasattr(self._autotuner, "best_config"):
            config = self._autotuner.best_config
        else:
            config = self._autotuner.configs[self._autotune_cfg_idx]
        full_nargs = {**mapped_kwargs, **config.all_kwargs()}
        if config.pre_hook is not None:
            config.pre_hook(full_nargs)
        return full_nargs

def submit_compiled_kernel_to_ui(kernel: CompiledKernel, via_relay: bool = True, log_ptxas_info: bool = False):
    info = get_triton_compile_info_from_res(kernel, log_ptxas_info=log_ptxas_info)
    if via_relay:
        app_metas = list_all_running_apps_in_relay()
    else:
        app_metas = list_all_app_in_machine()
    for meta in app_metas:
        if "TritonSim" in meta.module_name:
            client = meta.create_client()
            with client:
                fn_name = "compiled_kernel"
                client.app_chunked_remote_call("set_triton_compile_info", fn_name, info)
            break

class TritonRuntimeRunner:
    def __init__(self, fn: Union[triton.JITFunction, triton.runtime.Autotuner], 
            autotune_cfg_idx: int = 0,
            mode: TensorSimMode = TensorSimMode.FULL,
            submit_to_ui: bool = False, 
            submit_asm_to_ui: bool = False,
            submit_ref_results: Optional[dict[str, np.ndarray]] = None):
        self.fn = fn
        self._autotune_cfg_idx = autotune_cfg_idx
        self._mode = mode
        self._submit_to_ui = submit_to_ui
        self._submit_ref_results = submit_ref_results
        self._submit_asm_to_ui = submit_asm_to_ui

    def __getitem__(self, grid):
        return TritonRuntimeRunnerNested(grid, self.fn, mode=self._mode,
                                         autotune_cfg_idx=self._autotune_cfg_idx,
                                         submit_to_ui=self._submit_to_ui,
                                         submit_ref_results=self._submit_ref_results,
                                         submit_asm_to_ui=self._submit_asm_to_ui) 

@dataclasses.dataclass
class Duration:
    val: float = 0.0

@contextlib.contextmanager
def measure_duration_torch(*,
                            stream: Optional[Any] = None,
                            enable: bool = True):
    import torch 
    if not enable:
        yield Duration()
    else:
        if stream is None:
            stream = torch.cuda.current_stream()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record(stream)
        dur = Duration()
        yield dur
        end_ev.record(stream)
        start_ev.synchronize()
        end_ev.synchronize()
        duration = start_ev.elapsed_time(end_ev)
        dur.val = duration
