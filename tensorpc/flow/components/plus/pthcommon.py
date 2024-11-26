from typing import Any, Callable, Coroutine, Dict, List, Optional, Type, Union

import numpy as np
from tensorpc.core.tree_id import UniqueTreeIdForTree
from tensorpc.flow.components import mui, flowui
import humanize

from tensorpc.flow.jsonlike import IconButtonData, TreeItem

_TORCH_MODULE_KEYS = [
    "register_load_state_dict_post_hook",
    "register_module",
    "state_dict",
    "forward",
    "named_parameters",
]


def check_type_is_torch_module(type: Type) -> bool:
    for key in _TORCH_MODULE_KEYS:
        if not hasattr(type, key):
            return False
    return True


class PytorchModuleTreeItem(TreeItem):

    def __init__(
        self,
        mod: Any,
        module_id: str = "",
        btns: Union[List[IconButtonData], mui.Undefined] = mui.undefined,
        on_lazy_expand: Optional[Callable[[str], Coroutine]] = None,
        on_button_click: Optional[Callable[[str, str],
                                           Coroutine]] = None) -> None:
        self._mod = mod
        self._module_id = module_id
        self._num_submodules = len(list(mod.children()))
        self._on_lazy_expand = on_lazy_expand
        self._btns = btns
        self._on_button_click = on_button_click

    async def get_child_desps(
            self,
            parent_ns: UniqueTreeIdForTree) -> Dict[str, mui.JsonLikeNode]:
        res = {}
        import torch
        assert isinstance(self._mod, torch.nn.Module)
        self._mod.modules
        for name, child_mod in self._mod.named_children():
            fake_node = self.__class__(child_mod, "", self._btns)
            res[name] = fake_node.get_json_like_node(
                parent_ns.append_part(name))
        return res

    def _get_tensor_meta(self, ten: Any):
        try:
            from torch.distributed.tensor import DTensor  # type: ignore
            is_dtensor = isinstance(ten, DTensor)
        except ImportError:
            is_dtensor = False
        shape_str = ",".join(map(str, ten.shape))
        placements_str = None
        if is_dtensor:
            from torch.distributed.tensor import Shard, Partial, Replicate  # type: ignore
            p_strs = []
            for p in ten.placements:
                if isinstance(p, Shard):
                    p_strs.append(f"S({p.dim})")
                elif isinstance(p, Partial):
                    p_strs.append(f"P")
                elif isinstance(p, Replicate):
                    p_strs.append(f"R")
            placements_str = ",".join(p_strs)
            return f"[{shape_str}]:[{placements_str}]"
        return f"[{shape_str}]"

    def _reduce_list_tuple_to_single_if_same(self, tup: Any):
        if isinstance(tup, (list, tuple)):
            if len(tup) == 1:
                return tup[0]
            val = tup[0]
            for v in tup[1:]:
                if v != val:
                    return tup
            return val
        return tup

    def _get_pytorch_module_str(self, mod: Any) -> str:
        total_size = 0
        import torch
        for name, param in mod.named_parameters():
            total_size += param.numel() * param.element_size()
        res = humanize.naturalsize(total_size)
        if isinstance(mod, torch.nn.Linear):
            w_str = f"{self._get_tensor_meta(mod.weight)}"
            if mod.bias is not None:
                res = f"{w_str}+B {res}"
            else:
                res = f"{w_str} {res}"

        if isinstance(mod,
                      (torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Conv1d)):
            w_str = f"{self._get_tensor_meta(mod.weight)}"
            ks = self._reduce_list_tuple_to_single_if_same(mod.kernel_size)
            s = self._reduce_list_tuple_to_single_if_same(mod.stride)
            p = self._reduce_list_tuple_to_single_if_same(mod.padding)
            d = self._reduce_list_tuple_to_single_if_same(mod.dilation)
            kspd_str = f"K={ks}"
            if s != 1:
                kspd_str += f" S={s}"
            if p != 0:
                kspd_str += f" P={p}"
            if d != 1:
                kspd_str += f" D={d}"
            if mod.bias is not None:
                res = f"{w_str}+B {kspd_str} {res}"
            else:
                res = f"{w_str} {kspd_str} {res}"

        elif isinstance(mod, torch.nn.Embedding):
            w_str = f"{self._get_tensor_meta(mod.weight)}"
            res = f"{w_str} {res}"
        return res

    def get_json_like_node(self, id: UniqueTreeIdForTree) -> mui.JsonLikeNode:
        return mui.JsonLikeNode(id,
                                id.parts[-1],
                                mui.JsonLikeType.Object.value,
                                typeStr=type(self._mod).__name__,
                                cnt=self._num_submodules,
                                drag=False,
                                iconBtns=self._btns,
                                value=self._get_pytorch_module_str(self._mod))

    async def handle_lazy_expand(self) -> Any:
        if self._on_lazy_expand is not None:
            return await self._on_lazy_expand(self._module_id)

    async def handle_button(self, button_key: str) -> Optional[bool]:
        if self._on_button_click is not None:
            return await self._on_button_click(self._module_id, button_key)

    async def get_child(self, key: str) -> Any:
        if self._module_id == "":
            module_id = key
        else:
            module_id = f"{self._module_id}.{key}"
        return self.__class__(self._mod.get_submodule(key), module_id,
                              self._btns, self._on_lazy_expand,
                              self._on_button_click)
