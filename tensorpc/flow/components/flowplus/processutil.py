import asyncio
from concurrent.futures import ProcessPoolExecutor as _SysProcessPoolExecutor
from functools import partial

from tensorpc.utils.typeutils import take_annotation_from
from .compute import get_compute_flow_context, ComputeNodeWrapper
from .customnode import CustomNode
import inspect

def _node_function_wrapped_process_target(*args, __tensorpc_node_code: str, __tensorpc_fn_qname: str, **kwargs):
    mod_dict = {}
    exec(__tensorpc_node_code, mod_dict)
    parts = __tensorpc_fn_qname.split('.')
    obj = mod_dict[parts[0]]
    for part in parts[1:]:
        obj = getattr(obj, part)
    return obj(*args, **kwargs)

class ProcessPoolExecutor(_SysProcessPoolExecutor):
    """This should only be used inside node compute."""
    def __init__(self, node_id: str, max_workers=None, mp_context=None,
                 initializer=None, initargs=()):
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

    @take_annotation_from(_SysProcessPoolExecutor.submit)
    def submit(self, fn, *args, **kwargs):
        fn_mod = fn.__module__
        if fn_mod is None:
            return super().submit(fn, *args, **kwargs)
        if fn_mod == self._node_module_name:
            assert not inspect.ismethod(fn), "you can't use node method in process."
            fn_name = fn.__qualname__
            fn = partial(_node_function_wrapped_process_target, __tensorpc_node_code=self._node_code, __tensorpc_fn_qname=fn_name)
        return super().submit(fn, *args, **kwargs)

    @take_annotation_from(_SysProcessPoolExecutor.map)
    def map(self, fn, *args, **kwargs):
        fn_mod = fn.__module__
        if fn_mod is None:
            return super().map(fn, *args, **kwargs)
        if fn_mod == self._node_module_name:
            assert not inspect.ismethod(fn), "you can't use node method in process."
            fn_name = fn.__qualname__
            fn = partial(_node_function_wrapped_process_target, __tensorpc_node_code=self._node_code, __tensorpc_fn_qname=fn_name)
        return super().map(fn, *args, **kwargs)

async def run_in_node_executor(exc: ProcessPoolExecutor, fn, *args, **kwargs):
    assert isinstance(exc, ProcessPoolExecutor)
    return await asyncio.get_running_loop().run_in_executor(exc, fn, *args, **kwargs)