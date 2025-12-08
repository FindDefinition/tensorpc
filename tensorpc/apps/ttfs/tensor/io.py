from typing import Any, Self
from tensorpc.apps.ttfs.aggtype import (
    aggregate,
    aggregate_replace_gluon,
    aggregate_replace_triton,
    constexpr_function,
    triton_jit,
    gluon_jit_kernel,
    triton_jit,
    triton_jit_kernel,
)
from tensorpc.apps.ttfs.mp import (
    get_field,
    get_field_triton,
    is_constexpr,
    is_gluon,
)

from tensorpc.apps.ttfs.tensor.base import BlockedTensorDesc
from tensorpc.apps.ttfs.mp import TritonConstexprField

import triton.language as tl
from triton.experimental.gluon import language as gl
from tensorpc.apps.ttfs.gl.ops import async_copy_global_to_shared


@constexpr_function
def _get_tensor_idx(names, name):
    names = [gl._unwrap_if_constexpr(n) for n in names]
    return names.index(name)


@constexpr_function
def _tuple_range(length):
    return tuple(i for i in range(length))


@aggregate
class GroupedPointerIO:
    pointers: Any
    blocked_descs: Any
    names: tl.constexpr
    main_axis_name: tl.constexpr
    inc_strides: tl.tuple
    apply_mask: tl.constexpr

    @triton_jit
    def get_mask(self, name: tl.constexpr):
        return None

    @triton_jit
    def _get_mask_final(
        self, name: tl.constexpr, mask=None, apply_mask: tl.constexpr = True
    ):
        if apply_mask:
            if mask is not None:
                mask_ = mask
            else:
                if self.apply_mask:
                    mask_ = self.get_mask(name)
                else:
                    mask_ = None
        else:
            mask_ = None
        return mask_

    @triton_jit
    def load_base(
        self,
        pointers,
        name: tl.constexpr,
        mask=None,
        other=None,
        apply_mask: tl.constexpr = True,
    ) -> tl.tensor:
        mask_ = self._get_mask_final(name, mask, apply_mask)
        if is_gluon():
            return gl.load(pointers, mask_, other)
        else:
            return tl.load(pointers, mask_, other)

    @triton_jit
    def store_base(
        self,
        pointers,
        value,
        name: tl.constexpr,
        mask=None,
        apply_mask: tl.constexpr = True,
    ):
        mask_ = self._get_mask_final(name, mask, apply_mask)
        if is_gluon():
            gl.store(pointers, value, mask_)
        else:
            tl.store(pointers, value, mask_)

    @triton_jit
    def cp_load_base(
        self,
        pointers,
        name: tl.constexpr,
        smem,
        mask=None,
        other=None,
        apply_mask: tl.constexpr = True,
    ):
        if is_gluon():
            mask_ = self._get_mask_final(name, mask, apply_mask)
            async_copy_global_to_shared(smem, pointers, mask=mask_, other=other)
        else:
            raise NotImplementedError(
                "cp_load_base is not implemented for triton backend."
            )

    @triton_jit
    def load(
        self, name: tl.constexpr, mask=None, other=None, apply_mask: tl.constexpr = True
    ) -> tl.tensor:
        pointer = self.pointers[_get_tensor_idx(self.names, name)]
        return self.load_base(
            pointer, name, mask=mask, other=other, apply_mask=apply_mask
        )

    @triton_jit
    def cp_load(
        self,
        name: tl.constexpr,
        smem,
        mask=None,
        other=None,
        apply_mask: tl.constexpr = True,
    ):
        pointer = self.pointers[_get_tensor_idx(self.names, name)]
        return self.cp_load_base(
            pointer, name, smem, mask=mask, other=other, apply_mask=apply_mask
        )

    @triton_jit
    def store(
        self, name: tl.constexpr, value, mask=None, apply_mask: tl.constexpr = True
    ):
        pointer = self.pointers[_get_tensor_idx(self.names, name)]
        return self.store_base(pointer, value, name, mask=mask, apply_mask=apply_mask)

    @triton_jit
    def increment(self, inc_block) -> Self:
        if is_gluon():
            return aggregate_replace_gluon(
                self,
                pointers=[
                    self.pointers[i] + inc_block * self.inc_strides[i]
                    for i in tl.tuple(_tuple_range(len(self.pointers)))
                ],
            )
        else:
            return aggregate_replace_triton(
                self,
                pointers=[
                    self.pointers[i] + inc_block * self.inc_strides[i]
                    for i in tl.tuple(_tuple_range(len(self.pointers)))
                ],
            )


@aggregate
class GroupedPointerWithSingleMaskIO(GroupedPointerIO):
    # only mask in main axis is applied.
    offsets_main: tl.tensor

    @triton_jit
    def get_mask(self, name: tl.constexpr):
        desc = self.blocked_descs[_get_tensor_idx(self.names, name)]
        if is_gluon():
            fake_offset = desc.get_blocked_offset(
                self.main_axis_name, expand_dims=False
            )

            real_offset = gl.convert_layout(self.offsets_main, fake_offset.type.layout)
        else:
            real_offset = self.offsets_main
        boundry = desc.get_dim_size_by_name(self.main_axis_name)
        return desc.unslice_1d_offset(real_offset, self.main_axis_name) < boundry.value

    @triton_jit
    def increment(self, inc_block) -> Self:
        if is_gluon():
            return aggregate_replace_gluon(
                self,
                pointers=[
                    self.pointers[i]
                    + inc_block
                    * self.blocked_descs[i]
                    .get_block_size_by_name(self.main_axis_name)
                    .value
                    * self.inc_strides[i]
                    for i in tl.tuple(_tuple_range(len(self.pointers)))
                ],
                offsets_main=self.offsets_main
                + inc_block
                * self.blocked_descs[0]
                .get_block_size_by_name(self.main_axis_name)
                .value,
            )
        else:
            return aggregate_replace_triton(
                self,
                pointers=[
                    self.pointers[i]
                    + inc_block
                    * self.blocked_descs[i]
                    .get_block_size_by_name(self.main_axis_name)
                    .value
                    * self.inc_strides[i]
                    for i in tl.tuple(_tuple_range(len(self.pointers)))
                ],
                offsets_main=self.offsets_main
                + inc_block
                * self.blocked_descs[0]
                .get_block_size_by_name(self.main_axis_name)
                .value,
            )


@aggregate
class GroupedPointerWithFullMaskIO(GroupedPointerIO):
    offsets_main: tl.tensor
    remain_mask: tl.tensor

    @triton_jit
    def get_mask(self, name: tl.constexpr):
        desc = self.blocked_descs[_get_tensor_idx(self.names, name)]
        if is_gluon():
            fake_offset = desc.get_blocked_offset(
                self.main_axis_name, expand_dims=False
            )

            real_offset = gl.convert_layout(self.offsets_main, fake_offset.type.layout)
        else:
            real_offset = self.offsets_main
        boundry = desc.get_dim_size_by_name(self.main_axis_name)
        return (
            desc.unslice_1d_offset(real_offset, self.main_axis_name) < boundry.value
        ) & self.remain_mask

    @triton_jit
    def increment(self, inc_block) -> Self:
        if is_gluon():
            return aggregate_replace_gluon(
                self,
                pointers=[
                    self.pointers[i]
                    + inc_block
                    * self.blocked_descs[i]
                    .get_block_size_by_name(self.main_axis_name)
                    .value
                    * self.inc_strides[i]
                    for i in tl.tuple(_tuple_range(len(self.pointers)))
                ],
                offsets_main=self.offsets_main
                + inc_block
                * self.blocked_descs[0]
                .get_block_size_by_name(self.main_axis_name)
                .value,
            )
        else:
            return aggregate_replace_triton(
                self,
                pointers=[
                    self.pointers[i]
                    + inc_block
                    * self.blocked_descs[i]
                    .get_block_size_by_name(self.main_axis_name)
                    .value
                    * self.inc_strides[i]
                    for i in tl.tuple(_tuple_range(len(self.pointers)))
                ],
                offsets_main=self.offsets_main
                + inc_block
                * self.blocked_descs[0]
                .get_block_size_by_name(self.main_axis_name)
                .value,
            )


@aggregate
class GroupedScatterIO(GroupedPointerIO):
    idx_main: tl.tensor

    @triton_jit
    def _get_scatter_pointers(self, name: tl.constexpr):
        desc = self.blocked_descs[_get_tensor_idx(self.names, name)]
        block_size = desc.get_block_size_by_name(self.main_axis_name)
        stride = desc.get_stride_by_name(self.main_axis_name)
        pointer = self.pointers[_get_tensor_idx(self.names, name)]
        return pointer + self.idx_main * block_size.value * stride.value

    @triton_jit
    def get_mask(self, name: tl.constexpr):
        desc = self.blocked_descs[_get_tensor_idx(self.names, name)]
        offset = desc.get_blocked_offset(self.main_axis_name, expand_dims=True)
        block_size = desc.get_block_size_by_name(self.main_axis_name)
        boundry = desc.get_dim_size_by_name(self.main_axis_name)
        return offset + self.idx_main * block_size.value < boundry.value

    @triton_jit
    def load(
        self, name: tl.constexpr, mask=None, other=None, apply_mask: tl.constexpr = True
    ) -> tl.tensor:
        pointers = self._get_scatter_pointers(name)
        return self.load_base(
            pointers, name, mask=mask, other=other, apply_mask=apply_mask
        )

    @triton_jit
    def cp_load(
        self,
        name: tl.constexpr,
        smem,
        mask=None,
        other=None,
        apply_mask: tl.constexpr = True,
    ):
        pointers = self._get_scatter_pointers(name)
        self.cp_load_base(
            pointers, name, smem, mask=mask, other=other, apply_mask=apply_mask
        )

    @triton_jit
    def store(
        self, name: tl.constexpr, value, mask=None, apply_mask: tl.constexpr = True
    ):
        pointers = self._get_scatter_pointers(name)
        self.store_base(pointers, value, name, mask=mask, apply_mask=apply_mask)

    @triton_jit
    def increment(self, inc_block) -> Self:
        return self

    @triton_jit
    def replace_offset(self, new_idx) -> Self:
        if is_gluon():
            return aggregate_replace_gluon(
                self,
                idx_main=new_idx,
            )
        else:
            return aggregate_replace_triton(
                self,
                idx_main=new_idx,
            )


@constexpr_function
def _tuple_constexpr_to_list(t):
    # triton has bug when unwrap constexpr with tl.tuple. so we convert nested constexpr to a constexpr[list] here.
    return [tl.core._unwrap_if_constexpr(v.value) for v in t]


@triton_jit
def create_grouped_io_iter(
    blocked_descs: tl.tuple,
    main_axis_name: tl.constexpr,
    main_mask: tl.constexpr = False,
    full_mask: tl.constexpr = False,
    out_of_range: tl.constexpr = 0,
):
    pointers = [
        desc.get_blocked_ptrs(out_of_range=out_of_range) for desc in blocked_descs
    ]
    names: tl.constexpr = _tuple_constexpr_to_list(
        [TritonConstexprField(desc.name) for desc in blocked_descs]
    )

    if not main_mask:
        return GroupedPointerIO(
            pointers=pointers,
            blocked_descs=blocked_descs,
            names=names,
            main_axis_name=tl.constexpr(main_axis_name),
            inc_strides=[
                desc.get_stride_by_name(main_axis_name).value
                * desc.get_block_size_by_name(main_axis_name).value
                for desc in blocked_descs
            ],
            apply_mask=False,
        )
    else:
        main_axis_offset = blocked_descs[0].get_blocked_offset(
            main_axis_name, expand_dims=False, out_of_range=out_of_range
        )
        if full_mask:
            tl.static_assert(
                len(blocked_descs) == 1,
                "Currently only support single blocked desc for full_mask=True",
            )

            mask_remain = blocked_descs[0].get_blocked_masks_except_dim_name(
                main_axis_name
            )
            return GroupedPointerWithFullMaskIO(
                pointers=pointers,
                blocked_descs=blocked_descs,
                names=names,
                main_axis_name=main_axis_name,
                inc_strides=[
                    desc.get_stride_by_name(main_axis_name).value
                    for desc in blocked_descs
                ],
                offsets_main=main_axis_offset,
                remain_mask=mask_remain,
                apply_mask=True,
            )
        else:
            return GroupedPointerWithSingleMaskIO(
                pointers=pointers,
                blocked_descs=blocked_descs,
                names=names,
                main_axis_name=main_axis_name,
                inc_strides=[
                    desc.get_stride_by_name(main_axis_name).value
                    for desc in blocked_descs
                ],
                offsets_main=main_axis_offset,
                apply_mask=True,
            )


@triton_jit
def create_io_iter(
    blocked_desc: BlockedTensorDesc,
    main_axis_name: tl.constexpr,
    main_mask: tl.constexpr = False,
    full_mask: tl.constexpr = False,
):
    return create_grouped_io_iter(
        blocked_descs=(blocked_desc,),
        main_axis_name=main_axis_name,
        main_mask=main_mask,
        full_mask=full_mask,
    )


@triton_jit
def create_grouped_scatter_io_iter(
    blocked_descs: tl.tuple,
    main_axis_name: tl.constexpr,
    main_mask: tl.constexpr = False,
):
    pointers = [desc.get_blocked_ptrs() for desc in blocked_descs]
    names: tl.constexpr = _tuple_constexpr_to_list(
        [TritonConstexprField(desc.name) for desc in blocked_descs]
    )

    return GroupedScatterIO(
        pointers=pointers,
        blocked_descs=blocked_descs,
        names=names,
        main_axis_name=main_axis_name,
        inc_strides=[
            desc.get_stride_by_name(main_axis_name).value
            * desc.get_block_size_by_name(main_axis_name).value
            for desc in blocked_descs
        ],
        apply_mask=main_mask,
        idx_main=gl.to_tensor(0),
    )


@triton_jit
def create_scatter_io_iter(
    blocked_desc: BlockedTensorDesc,
    main_axis_name: tl.constexpr,
    main_mask: tl.constexpr = False,
):
    return create_grouped_scatter_io_iter(
        blocked_descs=(blocked_desc,),
        main_axis_name=main_axis_name,
        main_mask=main_mask,
    )
