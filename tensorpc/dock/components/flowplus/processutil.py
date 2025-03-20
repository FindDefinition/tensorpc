import asyncio
from concurrent.futures import ProcessPoolExecutor as _SysProcessPoolExecutor
from functools import partial
from typing import Tuple, Any, Optional
from tensorpc.utils.typeutils import take_annotation_from
from .compute import get_compute_flow_context, ComputeNodeWrapper
from .customnode import CustomNode
import inspect


def _node_function_wrapped_process_target(
        *args, __tensorpc_node_code: str, __tensorpc_fn_qname: str,
        __tensorpc_fn_partial_params: Optional[Tuple[Any, Any]], **kwargs):
    mod_dict = {}
    if __tensorpc_fn_partial_params is not None:
        pargs = __tensorpc_fn_partial_params[0]
        pkwargs = __tensorpc_fn_partial_params[1]
    else:
        pargs = ()
        pkwargs = {}
    exec(__tensorpc_node_code, mod_dict)
    parts = __tensorpc_fn_qname.split('.')
    obj = mod_dict[parts[0]]
    for part in parts[1:]:
        obj = getattr(obj, part)
    return obj(*pargs, *args, **pkwargs, **kwargs)

class ProcessPoolExecutor(_SysProcessPoolExecutor):
    """This should only be used inside node compute."""
    def __init__(self,
                 node_id: str,
                 max_workers=None,
                 mp_context=None,
                 initializer=None,
                 initargs=()):
        ctx = get_compute_flow_context()
        assert ctx is not None
        node = ctx.cflow.graph.get_node_by_id(node_id)
        wrapper = node.get_component_checked(ComputeNodeWrapper)
        cnode = wrapper.cnode
        assert isinstance(cnode, CustomNode)
        node_module = cnode._module
        assert node_module is not None
        super().__init__(max_workers, mp_context, initializer, initargs)
        self._node_module_name = node_module.module.__name__
        self._node_code = node_module.code

    def _get_runnable_fn_in_node(self, fn):
        if isinstance(fn, partial):
            fn_partial = fn
            fn = fn.func
        else:
            fn_partial = None
        fn_mod = fn.__module__
        if fn_mod is None:
            return fn
        if fn_mod == self._node_module_name:
            assert not inspect.ismethod(
                fn), "you can't use node method in process."
            fn_name = fn.__qualname__
            if fn_partial is not None:
                partial_params = (fn_partial.args, fn_partial.keywords)
            else:
                partial_params = None
            fn = partial(_node_function_wrapped_process_target,
                         __tensorpc_node_code=self._node_code,
                         __tensorpc_fn_qname=fn_name,
                         __tensorpc_fn_partial_params=partial_params)
        return fn 

    @take_annotation_from(_SysProcessPoolExecutor.submit)
    def submit(self, fn, *args, **kwargs):
        fn_runnable = self._get_runnable_fn_in_node(fn)
        return super().submit(fn_runnable, *args, **kwargs)

    @take_annotation_from(_SysProcessPoolExecutor.map)
    def map(self, fn, *args, **kwargs):
        fn_runnable = self._get_runnable_fn_in_node(fn)
        return super().map(fn_runnable, *args, **kwargs)


async def run_in_node_executor(exc: ProcessPoolExecutor, fn, *args, **kwargs):
    assert isinstance(exc, ProcessPoolExecutor)
    return await asyncio.get_running_loop().run_in_executor(
        exc, fn, *args, **kwargs)
