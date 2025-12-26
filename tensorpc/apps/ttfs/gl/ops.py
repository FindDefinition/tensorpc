from typing import Callable, TypeVar, cast
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr
from triton.experimental.gluon.language._layouts import DotOperandLayout, NVMMADistributedLayout
from triton._C.libtriton import ir
import triton.language.core as tl_core

def _check(cond: bool, msg_fn: Callable[[], str], category=ValueError):
    if not cond:
        raise category(msg_fn())

@builtin
def async_copy_global_to_shared(smem, pointer, mask=None, other=None, cache_modifier="", eviction_policy="", volatile=False,
                                _semantic=None):
    """
    Asynchronously copy elements from global memory to shared memory.

    Args:
        smem (shared_memory_descriptor): Destination shared memory descriptor.
        pointer (tensor): Source pointer tensor.
        mask (tensor, optional): Mask tensor for predicated loads. Defaults to None.
        other (tensor, optional): Tensor to be used if mask is False. Defaults to None.
        cache_modifier (str): Cache modifier specifier. Defaults to "".
        eviction_policy (str): Eviction policy specifier. Defaults to "".
        volatile (bool): Whether the load is volatile. Defaults to False.
    """
    mask = _unwrap_if_constexpr(mask)
    other = _unwrap_if_constexpr(other)

    cache_modifier = _semantic._str_to_load_cache_modifier(cache_modifier)
    eviction_policy = _semantic._str_to_eviction_policy(eviction_policy)
    volatile = _unwrap_if_constexpr(volatile)
    if mask is not None:
        pointer, mask = _semantic.broadcast_impl_value(pointer, mask)
    if other is not None:
        other = _semantic.to_tensor(other)
        other = _semantic.cast(other, pointer.dtype.element_ty)
        pointer, other = _semantic.broadcast_impl_value(pointer, other)

    _check(
        smem.shape == pointer.shape, lambda:
        f"expected smem shape to match pointer shape but got smem.shape = {smem.shape}, pointer.shape = {pointer.shape}"
    )
    mask_handle = mask.handle if mask is not None else ir.value()
    other_handle = other.handle if other is not None else ir.value()
    _semantic.builder.create_async_copy_global_to_local(smem.handle, pointer.handle, mask_handle, other_handle,
                                                        cache_modifier, eviction_policy, volatile)

