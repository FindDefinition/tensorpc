import ast
import dataclasses
from typing import Any

import torch

import triton.language as tl
from triton.experimental.gluon import language as gl
from typing_extensions import Annotated

from tensorpc.apps.ttfs.aggtype import (
    FieldMeta,
    TritonAggFieldAccessor,
    TritonAggFlatField,
    aggregate,
    aggregate_replace_gluon,
    aggregate_replace_triton,
    aggregate_super_method_gluon,
    aggregate_super_method_triton,
    constexpr_function,
    triton_jit,
    triton_jit_kernel,
)
from tensorpc.apps.ttfs.mp import (
    TritonConstexprField,
    TritonField,
    all_fields_is_constexpr,
    filter_constexpr_fields,
    filter_non_constexpr_fields,
    get_all_field_keys_with_type,
    get_all_fields_with_type_gluon,
    get_field,
    get_field_triton,
    get_num_warps,
    has_field,
    is_constexpr,
    is_gluon,
    replace_all_fields_with_type_gluon,
    replace_all_fields_with_type_triton,
)
from tensorpc.apps.ttfs.tensor import sym
from tensorpc.apps.ttfs.tensor.core_ops import (
    merge_tuple_dict_gluon,
    merge_tuple_dict_key,
    merge_tuple_dict_triton,
)
from tensorpc.apps.ttfs.tensor.layout import (
    get_default_block_io_layout,
    get_default_swizzed_shared_layout_nvidia,
)


@aggregate
class StrideWithMultipleOf:
    stride: Any
    multiple_of: tl.constexpr
    need_multiple_of: tl.constexpr


@aggregate
class TensorDesc:
    ptr: Any
    shape_sym: Annotated[str, tl.constexpr]
    # -1 or (-1, -1, 64, -1)
    strides: Annotated[Any, FieldMeta(is_kernel_arg=False)] = None
    stride_multiple_of: tl.constexpr = dataclasses.field(default=tl.constexpr(1))
    _ttfs_pv_stride: Annotated[str, tl.constexpr, FieldMeta(is_kernel_arg=False)] = ""
    def __post_init__(self):
        if isinstance(self.ptr, torch.Tensor):
            # init from host code
            msg = (f"strides must be None in host side init, get {self.strides}. use "
                    "Annotated[ttfs.TensorDesc, ttfs.tensor_desc_meta(\"B,H\")] "
                    "to indicate which dims to pass strides for. we will fill strides from torch tensor.")
            assert self.strides is None, msg
        ndim: tl.constexpr = len(sym.get_shape_names_from_sym(self.shape_sym))
        if self.strides is None:
            self.strides = tl.tuple([tl.constexpr(-1)] * ndim)
        assert isinstance(
            self.strides, tl.tuple
        ), f"Strides must be a tl.tuple, but got {type(self.strides)}."
        pv_stride_sym = tl.core._unwrap_if_constexpr(self._ttfs_pv_stride)
        
        if pv_stride_sym != "":
            value_axes = sym.get_sym_axes(self.shape_sym, pv_stride_sym)
            new_strides = list(tl.constexpr(-1) for _ in range(ndim))
            for j, axis in enumerate(value_axes): 
                assert axis < ndim, f"passed stride axis {axis} out of range for shape with ndim {ndim}."
                new_strides[axis] = self.strides[j]
            self.strides = tl.tuple(new_strides) 
        if isinstance(self.strides, tl.tuple):
            assert (
                len(self.strides) == ndim
            ), f"Strides length {len(self.strides)} doesn't match shape length {ndim}."
        if isinstance(self.stride_multiple_of, (tl.tuple, tuple, list)):
            assert (
                len(self.stride_multiple_of) == ndim
            ), f"stride_multiple_of length {len(self.stride_multiple_of)} doesn't match shape length {ndim}."
        else:
            self.stride_multiple_of = tuple([1] * ndim)
        # print("ndim", ndim)

    @staticmethod
    @triton_jit
    def empty():
        return TensorDesc(
            ptr=None,
            shape_sym="B",
        )

    @triton_jit
    def get_dtype(self):
        dtype: tl.constexpr = self.ptr.dtype.element_ty
        return TritonConstexprField(dtype)

    @triton_jit
    def get_ndim(self):
        ndim: tl.constexpr = len(sym.get_shape_names_from_sym(self.shape_sym))
        return TritonConstexprField(ndim)

    @triton_jit
    def is_tma_desc(self):
        if is_gluon():
            return TritonConstexprField(
                isinstance(self.ptr, gl.nvidia.hopper.tma.tensor_descriptor)
            )
        else:
            return TritonConstexprField(isinstance(self.ptr, tl.tensor_descriptor))

    @triton_jit
    def is_valid_ptr(self):
        # ptr may be none.
        return TritonConstexprField(self.ptr is not None)

    @triton_jit
    def assert_is_valid(self):
        tl.static_assert(self.is_valid_ptr().value, "desc pointer must not be None.")

    @triton_jit
    def assert_is_pointer(self):
        tl.static_assert(
            not self.is_tma_desc().value, "Only support pointer tensor desc."
        )

    @triton_jit
    def assert_is_tma_desc(self):
        tl.static_assert(self.is_tma_desc().value, "Only support tma tensor desc.")

    @triton_jit
    def get_default_blocked_io_layout_from_block(
        self, block_shape_sym: tl.constexpr, block_shape: tl.constexpr
    ):
        tl.static_assert(is_gluon(), "layout is only supported in gluon backend.")
        num_warps: tl.constexpr = get_num_warps()
        order: tl.constexpr = sym.get_blocked_layout_order(
            self.shape_sym, block_shape_sym
        )
        tl.static_print(block_shape_sym, block_shape, order)

        layout: tl.constexpr = get_default_block_io_layout(
            shape=block_shape,
            bitwidth=self.get_dtype().value.primitive_bitwidth,
            order=order,
            num_warps=num_warps,
            num_threads_per_warp=32,
        )
        return TritonConstexprField(layout)

class TensorDescFieldAccessor(TritonAggFieldAccessor):
    def __init__(self, ndim: int, passed_strides: str = "", constexpr_strides: str = ""):
        # currently ndim is only used in validation
        self._ndim = ndim
        self._passed_strides = passed_strides
        if passed_strides != "":
            self._num_value_stride = len(passed_strides.split(","))
        else:
            self._num_value_stride = 0
        passed_strides_parts = [x.strip() for x in passed_strides.split(",")]
        constexpr_stride_idxes: list[int] = []
        if constexpr_strides:
            constexpr_stride_parts = [x.strip() for x in constexpr_strides.split(",")]
            for i, part in enumerate(passed_strides_parts):
                if part in constexpr_stride_parts:
                    constexpr_stride_idxes.append(i)
        self._constexpr_stride_idxes = constexpr_stride_idxes
        self._stride_multiple_of_short_name = "stride_mul"
        if ndim > 0:
            assert self._num_value_stride <= ndim, f"passed_strides {passed_strides} has more dims than ndim {ndim}."
    
    def get_custom_fields(self) -> list[tuple[str, bool]]: 
        res: list[tuple[str, bool]] = [
            ("ptr", False),
            ("shape_sym", True),
            (self._stride_multiple_of_short_name, True),
        ]
        for i in range(self._num_value_stride): 
            is_constexpr = i in self._constexpr_stride_idxes
            res.append((f"s{i}", is_constexpr))
        return res

    def get_load_call_node(self, meta: TritonAggFlatField, root_arg_name: str, fn_node: ast.Call) -> ast.Call:
        new_stride_tuples = []
        for i in range(self._num_value_stride): 
            arg_name = meta.mangle_path(root_arg_name, f"s{i}")
            new_stride_tuples.append(ast.Name(id=arg_name, ctx=ast.Load()))
        
        node = ast.Call(
            func=fn_node,
            args=[],
            keywords=[
                ast.keyword(
                    arg="ptr",
                    value=ast.Name(
                        id=meta.mangle_path(root_arg_name, "ptr"), ctx=ast.Load()
                    ),
                ),
                ast.keyword(
                    arg="shape_sym",
                    value=ast.Name(
                        id=meta.mangle_path(root_arg_name, "shape_sym"),
                        ctx=ast.Load(),
                    ),
                ),
                ast.keyword(
                    arg="stride_multiple_of",
                    value=ast.Name(
                        id=meta.mangle_path(root_arg_name, self._stride_multiple_of_short_name),
                        ctx=ast.Load(),
                    ),
                ),

            ],
        )
        if self._passed_strides != "":
            node.keywords.append(
                ast.keyword(
                    arg="strides",
                    value=ast.Tuple(elts=new_stride_tuples, ctx=ast.Load())
                ),
            )
            node.keywords.append(
                ast.keyword(
                    arg="_ttfs_pv_stride",
                    value=ast.Constant(value=self._passed_strides),
                )
            )
        return node

    def get_flatten_agg_to_kwarg_lines(self, meta: TritonAggFlatField, root_arg_name: str, desc_path: str) -> list[str]:
        
        lines = [
            f"kwargs['{meta.mangle_path(root_arg_name, 'ptr')}'] = {desc_path}.ptr",
            f"kwargs['{meta.mangle_path(root_arg_name, 'shape_sym')}'] = {desc_path}.shape_sym",
            f"kwargs['{meta.mangle_path(root_arg_name, self._stride_multiple_of_short_name)}'] = {desc_path}.stride_multiple_of",
        ]
        if self._ndim > 0:
            lines.append(
                f"assert({desc_path}.ptr is not None and {desc_path}.ptr.ndim == {self._ndim}, 'TensorDesc ndim mismatch, expected {self._ndim}, get ' + str({desc_path}.ptr.ndim))"
            )
        if self._num_value_stride > 0:
            axes_var_name = meta.mangle_path(root_arg_name, 'axes')
            # shape_sym is constexpr in host aggregate
            lines.append(f"{axes_var_name} = ttfs.tensor.sym.cached_get_sym_axes_host({desc_path}.shape_sym.value, '{self._passed_strides}')")
            lines.append(f"if {desc_path}.ptr is not None:")
            for i in range(self._num_value_stride): 
                lines.append(
                    f"    kwargs['{meta.mangle_path(root_arg_name, f's{i}')}'] = {desc_path}.ptr.stride({axes_var_name}[{i}])"
                )
        return lines

def tensor_desc_meta(passed_strides: str = "", constexpr_strides: str = "", ndim = -1) -> FieldMeta:
    return FieldMeta(accessor=TensorDescFieldAccessor(ndim, passed_strides, constexpr_strides))

@aggregate
class _TensorDimMultipleOf:
    pass

@aggregate(kw_only=True)
class TensorManagerBase:
    ttfs_dim_multiple_of: Any = dataclasses.field(
        default_factory=lambda: _TensorDimMultipleOf()
    )
    _ttfs_internal_offset_names: Annotated[Any, tl.constexpr, FieldMeta(is_kernel_arg=False)] = dataclasses.field(
        default_factory=lambda: tl.constexpr([])
    )
    _ttfs_internal_offsets: Annotated[tl.tuple, FieldMeta(is_kernel_arg=False)] = dataclasses.field(
        default_factory=lambda: tl.tuple([])
    )

    def __post_init__(self):
        # validate all TensorDesc fields
        fields = dataclasses.fields(self)
        all_fields_desc = []
        all_fields_not_desc = []
        for f in fields:
            key = f.name
            value = getattr(self, key)
            if isinstance(value, TensorDesc):
                all_fields_desc.append(key)
            else:
                all_fields_not_desc.append(key)
        for f in dataclasses.fields(TensorManagerBase):
            all_fields_not_desc.remove(f.name)
        for f in fields:
            key = f.name
            value = getattr(self, key)
            if isinstance(value, TensorDesc):
                shape_sym = gl._unwrap_if_constexpr(value.shape_sym)
                shape_names, _ = sym.parse_shape_sym(shape_sym)
                for dim_name in shape_names:
                    assert (
                        dim_name in all_fields_not_desc
                    ), f"Dimension {dim_name} doesn't exist in TensorManager. available dims: {all_fields_not_desc}"
                assert len(shape_names) == len(
                    value.strides
                ), f"Strides length {len(value.strides)} doesn't match shape length {len(shape_names)} for tensor {key}."

    @triton_jit
    def get_tensor_desc(self, name: tl.constexpr):
        desc = get_field(self, name).value
        return desc

    @triton_jit
    def get_dim_multiple_of(self, dim_name: tl.constexpr):
        if has_field(self.ttfs_dim_multiple_of, dim_name):
            dim_multiple_of_field = get_field(self.ttfs_dim_multiple_of, dim_name)
            tl.static_assert(
                dim_multiple_of_field.is_constexpr,
                "all fields in `ttfs_dim_multiple_of` must be constexpr.",
            )
            return dim_multiple_of_field
        else:
            return TritonConstexprField(1)

    @triton_jit
    def get_tensor_dim_size(self, name: tl.constexpr, axis: tl.constexpr):
        desc = get_field(self, name).value
        dim_name: tl.constexpr = sym.get_name_by_dim(desc.shape_sym, axis)
        return get_field(self, dim_name)

    @triton_jit
    def get_tensor_dim_size_by_name(self, name: tl.constexpr, dim_name: tl.constexpr):
        desc = get_field(self, name).value
        dim_name_unified: tl.constexpr = sym.local_name_to_global(
            desc.shape_sym, dim_name
        )
        return get_field(self, dim_name_unified)

    @triton_jit
    def get_tensor_stride(self, name: tl.constexpr, axis: tl.constexpr):
        desc = get_field(self, name).value
        if is_constexpr(desc.strides[axis]):
            if desc.strides[axis] < 0:
                # calc stride from dims, assume contiguous
                stride_names: tl.constexpr = sym.get_stride_names(desc.shape_sym, axis)
                constexpr_fields: tl.constexpr = filter_constexpr_fields(
                    self, stride_names
                )
                non_constexpr_fields: tl.constexpr = filter_non_constexpr_fields(
                    self, stride_names
                )
                # aggregate can keep constexpr info
                stride_c = TritonConstexprField(1)
                for i in gl.static_range(len(constexpr_fields)):
                    field = get_field(self, tl.constexpr(constexpr_fields[i]))
                    stride_c = TritonConstexprField(stride_c.value * field.value)
                if all_fields_is_constexpr(self, stride_names):
                    return stride_c
                else:
                    # for non-constexpr dims, we may need to use multiple_of
                    stride = 1
                    for i in gl.static_range(len(non_constexpr_fields)):
                        field = get_field(self, tl.constexpr(non_constexpr_fields[i]))
                        stride *= field.value
                    return TritonField(stride * stride_c.value)
            else:
                return TritonConstexprField(desc.strides[axis])
        else:
            return TritonField(desc.strides[axis])

    @triton_jit
    def get_tensor_stride_with_multiple_of(
        self, name: tl.constexpr, axis: tl.constexpr
    ):
        desc = get_field(self, name).value
        stride_multiple_of_in_desc: tl.constexpr = desc.stride_multiple_of[axis]
        if is_constexpr(desc.strides[axis]):
            if desc.strides[axis] < 0:
                # if stride is unset, stride_multiple_of defined in desc will be ignored
                # calc stride from dims, assume contiguous
                stride_names: tl.constexpr = sym.get_stride_names(desc.shape_sym, axis)
                constexpr_fields: tl.constexpr = filter_constexpr_fields(
                    self, stride_names
                )
                non_constexpr_fields: tl.constexpr = filter_non_constexpr_fields(
                    self, stride_names
                )
                # aggregate can keep constexpr info
                stride_c = TritonConstexprField(1)
                stride_multiple_of_c = TritonConstexprField(1)
                for i in gl.static_range(len(constexpr_fields)):
                    field = get_field(self, tl.constexpr(constexpr_fields[i]))
                    stride_c = TritonConstexprField(stride_c.value * field.value)
                    stride_multiple_of_c = TritonConstexprField(
                        stride_multiple_of_c.value
                        * self.get_dim_multiple_of(
                            tl.constexpr(constexpr_fields[i])
                        ).value
                    )
                if all_fields_is_constexpr(self, stride_names):
                    return StrideWithMultipleOf(
                        stride_c,
                        stride_multiple_of_c.value,
                        False,
                    )
                else:
                    # for non-constexpr dims, we may need to use multiple_of
                    stride = 1
                    for i in gl.static_range(len(non_constexpr_fields)):
                        field = get_field(self, tl.constexpr(non_constexpr_fields[i]))
                        stride *= field.value
                        stride_multiple_of_c = TritonConstexprField(
                            stride_multiple_of_c.value
                            * self.get_dim_multiple_of(
                                tl.constexpr(non_constexpr_fields[i])
                            ).value
                        )

                    return StrideWithMultipleOf(
                        TritonField(stride * stride_c.value),
                        stride_multiple_of_c.value,
                        stride_multiple_of_c.value != 1,
                    )
            else:
                return StrideWithMultipleOf(
                    TritonConstexprField(desc.strides[axis]),
                    stride_multiple_of_in_desc,
                    stride_multiple_of_in_desc != 1,
                )
        else:
            return StrideWithMultipleOf(
                TritonField(desc.strides[axis]),
                stride_multiple_of_in_desc,
                stride_multiple_of_in_desc != 1,
            )

    @triton_jit
    def get_tensor_stride_by_name(self, name: tl.constexpr, dim_name: tl.constexpr):
        desc = get_field(self, name).value
        dim_name_unified: tl.constexpr = sym.local_name_to_global(
            desc.shape_sym, dim_name
        )
        axis: tl.constexpr = sym.get_dim_by_name(desc.shape_sym, dim_name_unified)
        return self.get_tensor_stride(name, axis)

    @triton_jit
    def get_tensor_stride_with_multiple_of_by_name(
        self, name: tl.constexpr, dim_name: tl.constexpr
    ):
        desc = get_field(self, name).value
        dim_name_unified: tl.constexpr = sym.local_name_to_global(
            desc.shape_sym, dim_name
        )
        axis: tl.constexpr = sym.get_dim_by_name(desc.shape_sym, dim_name_unified)
        return self.get_tensor_stride_with_multiple_of(name, axis)

    @triton_jit
    def get_dim_offset_by_name(self, dim_name: tl.constexpr):
        if sym.is_name_in_names(dim_name, self._ttfs_internal_offset_names):
            return self._ttfs_internal_offsets[
                sym.get_name_in_names_idx(dim_name, self._ttfs_internal_offset_names)
            ]
        else:
            return 0

    @triton_jit
    def get_dim_offset_tuple_by_names(self, dim_names: tl.constexpr):
        dim_name_lst: tl.constexpr = sym.unify_offset_names(dim_names)
        return [
            self.get_dim_offset_by_name(tl.constexpr(dim_name_lst[i]))
            for i in tl.tuple(sym.tuple_range(len(dim_name_lst)))
        ]

    @triton_jit
    def get_tensor_dim_is_divisible(
        self, name: tl.constexpr, axis: tl.constexpr, divisor: tl.constexpr
    ):
        desc = get_field(self, name).value
        dim_size = self.get_tensor_dim_size(name, axis)
        if dim_size.is_constexpr:
            return TritonConstexprField(dim_size.value % divisor == 0)
        else:
            dim_multiple_of: tl.constexpr = self.get_dim_multiple_of(
                sym.get_name_by_dim(desc.shape_sym, axis)
            ).value
            return TritonConstexprField(dim_multiple_of % divisor == 0)

    @triton_jit
    def get_ptr_offset(
        self,
        name: tl.constexpr,
        off_names: tl.constexpr,
        offsets: tl.tuple,
        offset_can_invalid: tl.constexpr = False,
    ):
        # off_names can be constexpr(["M", "N"]) or "M,N"
        off_names_u: tl.constexpr = sym.unify_offset_names(off_names)
        desc = get_field(self, name).value
        off_names_u_filtered: tl.constexpr = sym.filter_sym(
            desc.shape_sym, off_names_u, not offset_can_invalid
        )
        if len(off_names_u_filtered) == 0:
            return 0
        else:
            offset = 0
            for i in gl.static_range(len(off_names_u_filtered)):
                offset = (
                    offset
                    + self.get_tensor_stride_by_name(
                        name, tl.constexpr(off_names_u_filtered[i])
                    ).value
                    * offsets[i]
                )
            return offset

    @triton_jit
    def get_offseted_ptr(
        self,
        name: tl.constexpr,
        off_names: tl.constexpr,
        offsets: tl.tuple,
        offset_can_invalid: tl.constexpr = False,
    ):
        desc = get_field(self, name).value
        tl.static_assert(
            not desc.is_tma_desc().value, "Only support pointer tensor desc."
        )
        tl.static_print(desc)
        tl.static_print(
            self.get_ptr_offset(name, off_names, offsets, offset_can_invalid)
        )

        return desc.ptr + self.get_ptr_offset(
            name, off_names, offsets, offset_can_invalid
        )

    @triton_jit
    def get_tensor_dim_sizes_by_sym(self, name: tl.constexpr, shape_sym: tl.constexpr):
        shape_names: tl.constexpr = sym.get_shape_names_from_sym(shape_sym)
        shape = [
            self.get_tensor_dim_size_by_name(name, tl.constexpr(shape_names[i]))
            for i in tl.tuple(sym.tuple_range(len(shape_names)))
        ]
        return shape

    @triton_jit
    def get_tensor_strides_by_sym(self, name: tl.constexpr, shape_sym: tl.constexpr):
        shape_names: tl.constexpr = sym.get_shape_names_from_sym(shape_sym)
        shape = [
            self.get_tensor_stride_by_name(name, tl.constexpr(shape_names[i]))
            for i in tl.tuple(sym.tuple_range(len(shape_names)))
        ]
        return shape

    @triton_jit
    def get_block_desc(
        self,
        name: tl.constexpr,
        block_shape_sym: tl.constexpr,
        block_sizes: tl.constexpr,
        block_layout: tl.constexpr = None,
    ):
        # TODO if desc.ptr is tma desc?
        desc = get_field(self, name).value
        desc.assert_is_pointer()
        desc.assert_is_valid()
        if is_gluon():
            if block_layout is None:
                block_layout_: tl.constexpr = (
                    desc.get_default_blocked_io_layout_from_block(
                        block_shape_sym, block_sizes
                    ).value
                )
            else:
                block_layout_: tl.constexpr = block_layout
        else:
            block_layout_: tl.constexpr = None
        # block_sym_unified: tl.constexpr = sym.local_sym_to_global(desc.shape_sym, block_shape_sym)
        # if sym.is_any_sym_in_names(block_sym_unified, self._ttfs_internal_offset_names):
        #     return BlockedTensorDescWithOffset(
        #         ptr=desc.ptr,
        #         shape_sym=desc.shape_sym,
        #         strides=desc.strides,
        #         name=name,
        #         mgr=self,
        #         block_shape_sym=block_shape_sym,
        #         block_shape=block_sizes,
        #         layout=block_layout_,
        #         offset_tuple=self.get_dim_offset_tuple_by_names(block_sym_unified),
        #     )
        # else:
        return BlockedTensorDesc(
            ptr=desc.ptr,
            shape_sym=desc.shape_sym,
            strides=desc.strides,
            name=name,
            mgr=self,
            layout=block_layout_,
            block_shape_sym=block_shape_sym,
            block_shape=block_sizes,
        )

    @triton_jit
    def get_external_offset_block_desc(
        self,
        name: tl.constexpr,
        block_shape_sym: tl.constexpr,
        block_sizes: tl.constexpr,
        offset_tuple: tl.tuple,
        block_layout: tl.constexpr = None,
    ):
        # TODO if desc.ptr is tma desc?
        desc = get_field(self, name).value
        desc.assert_is_pointer()
        desc.assert_is_valid()
        if is_gluon():
            if block_layout is None:
                block_layout_: tl.constexpr = (
                    desc.get_default_blocked_io_layout_from_block(
                        block_shape_sym, block_sizes
                    ).value
                )
            else:
                block_layout_: tl.constexpr = block_layout
        else:
            block_layout_: tl.constexpr = None
        return BlockedTensorDescWithOffset(
            ptr=desc.ptr,
            shape_sym=desc.shape_sym,
            strides=desc.strides,
            name=name,
            mgr=self,
            layout=block_layout_,
            block_shape_sym=block_shape_sym,
            block_shape=block_sizes,
            offset_tuple=offset_tuple,
        )

    @triton_jit
    def get_tma_block_desc(
        self,
        name: tl.constexpr,
        block_shape_sym: tl.constexpr,
        block_sizes: tl.constexpr,
        offset_tuple=None,
    ):
        # TODO currently we assume user offset pointer with all remain dims except block dims.
        # e.g. for [B, S, H, D], we assume user offset B and H when get block desc with block shape [S, D]

        # if user use `offset_all_desc`, we get b and h offset and original pointer.
        # it's ok because B, H, D is always divisible, we can use [B * S, H * D] as blocked tma desc.

        # however, if tensor shape is [B, H, S, D], we must use [B * H * S, D] as blocked tma desc,
        # if S isn't divisible, this tma will load/store wrong data.
        desc = get_field(self, name).value
        desc.assert_is_valid()
        tl.static_assert(not desc.is_tma_desc().value, "Not implemented for now")
        shape = self.get_tensor_dim_sizes_by_sym(name, block_shape_sym)
        strides = self.get_tensor_strides_by_sym(name, block_shape_sym)
        if desc.is_tma_desc().value:
            if is_gluon():
                smem_layout: tl.constexpr = desc.ptr.layout
            else:
                smem_layout: tl.constexpr = None
            tma_desc = desc.ptr
        else:
            if is_gluon():
                smem_layout: tl.constexpr = gl.NVMMASharedLayout.get_default_for(
                    block_sizes,
                    desc.get_dtype().value,
                    # TODO transposed only support matrix.
                    transposed=sym.check_matrix_is_transposed(
                        desc.shape_sym, block_shape_sym
                    ),
                )
                # WARNING: this requires triton 3.6+
                tma_desc = gl.nvidia.hopper.tma.make_tensor_descriptor(
                    desc.ptr, shape, strides, block_sizes, smem_layout
                )
            else:
                tma_desc = tl.make_tensor_descriptor(
                    desc.ptr, shape, strides, block_sizes
                )
                smem_layout: tl.constexpr = None
        block_sym_unified: tl.constexpr = sym.local_sym_to_global(
            desc.shape_sym, block_shape_sym
        )

        return BlockedTensorDescWithOffset(
            ptr=tma_desc,
            shape_sym=desc.shape_sym,
            strides=desc.strides,
            name=name,
            mgr=self,
            block_shape_sym=block_shape_sym,
            block_shape=block_sizes,
            layout=smem_layout,
            offset_tuple=(
                offset_tuple
                if offset_tuple is not None
                else self.get_dim_offset_tuple_by_names(block_sym_unified)
            ),
        )

    @triton_jit
    def get_block_desc_by_ptr(
        self,
        ptr,
        name: tl.constexpr,
        block_shape_sym: tl.constexpr,
        block_sizes: tl.constexpr,
        block_layout: tl.constexpr = None,
    ):
        desc = get_field(self, name).value
        desc.assert_is_valid()
        if is_gluon():
            if block_layout is None:
                block_layout_: tl.constexpr = (
                    desc.get_default_blocked_io_layout_from_block(
                        block_shape_sym, block_sizes
                    ).value
                )
            else:
                block_layout_: tl.constexpr = block_layout
        else:
            block_layout_: tl.constexpr = None

        return BlockedTensorDesc(
            ptr=ptr,
            shape_sym=desc.shape_sym,
            strides=desc.strides,
            name=name,
            mgr=self,
            block_shape_sym=block_shape_sym,
            block_shape=block_sizes,
            layout=block_layout_,
        )

    @triton_jit
    def offset_all_desc_ptr(self, off_names: tl.constexpr, offsets: tl.tuple):
        """offset all tensor desc ptrs that contain dimensions in off_names.
        e.g. for tensor manager with q(B, S, H, D), m(B, H, S) and c(B, S),
        tm.offset_all_desc_ptr(["B", "H"], (1, 2)) will offset q and m ptrs by b and h.
        c will be offseted only by b.

        WARNING: you should use offset_all_desc for dimensions that need to check boundry.
        """
        all_field_keys: tl.constexpr = get_all_field_keys_with_type(self, TensorDesc)

        if is_gluon():
            all_fields_tuple = get_all_fields_with_type_gluon(self, TensorDesc)
            all_fields_tuple_offseted = [
                aggregate_replace_gluon(
                    all_fields_tuple[i],
                    # pointer check (not tma) is done in get_offseted_ptr
                    ptr=(
                        self.get_offseted_ptr(
                            all_field_keys[i],
                            off_names,
                            offsets,
                            offset_can_invalid=True,
                        )
                        if all_fields_tuple[i].is_valid_ptr().value
                        else None
                    ),
                )
                for i in tl.tuple(sym.tuple_range(len(all_field_keys)))
            ]
            return replace_all_fields_with_type_gluon(
                self, TensorDesc, all_fields_tuple_offseted
            )
        else:
            all_fields_tuple_offseted = [
                aggregate_replace_triton(
                    all_fields_tuple[i],
                    ptr=(
                        all_fields_tuple[i].ptr
                        + self.get_ptr_offset(
                            all_field_keys[i],
                            off_names,
                            offsets,
                            offset_can_invalid=True,
                        )
                        if all_fields_tuple[i].is_valid_ptr().value
                        else None
                    ),
                )
                for i in tl.tuple(sym.tuple_range(len(all_field_keys)))
            ]
            return replace_all_fields_with_type_triton(
                self, TensorDesc, all_fields_tuple_offseted
            )

    @triton_jit
    def offset_all_desc(self, off_names: tl.constexpr, offsets: tl.tuple):
        """Offset tensor descriptors without change ptrs.
        usually used in dimensions that may be out of boundry, e.g. sequence dimension.

        WARNING: you can't use this in a loop because it will change static type of this aggregate object.
        """

        # store offsets in _ttfs_internal_offset_names and _ttfs_internal_offsets
        off_names_u: tl.constexpr = sym.unify_offset_names(off_names)
        if is_gluon():
            new_names: gl.constexpr = merge_tuple_dict_key(
                self._ttfs_internal_offset_names, off_names_u
            )
            new_offsets = merge_tuple_dict_gluon(
                self._ttfs_internal_offset_names,
                self._ttfs_internal_offsets,
                off_names_u,
                offsets,
            )
            return aggregate_replace_gluon(
                self,
                _ttfs_internal_offset_names=new_names,
                _ttfs_internal_offsets=new_offsets,
            )
        else:
            new_names: gl.constexpr = merge_tuple_dict_key(
                self._ttfs_internal_offset_names, off_names_u
            )

            new_offsets = merge_tuple_dict_triton(
                self._ttfs_internal_offset_names,
                self._ttfs_internal_offsets,
                off_names_u,
                offsets,
            )
            return aggregate_replace_triton(
                self,
                _ttfs_internal_offset_names=new_names,
                _ttfs_internal_offsets=new_offsets,
            )


@constexpr_function
def _insert_front(val, vec):
    # print(val, vec, tl.core._unwrap_if_constexpr(vec))
    return [val] + [tl.core._unwrap_if_constexpr(v) for v in vec]


@aggregate(kw_only=True)
class BlockedTensorDesc(TensorDesc):
    name: tl.constexpr
    mgr: TensorManagerBase
    block_shape_sym: tl.constexpr  # e.g. "S,H", can be alias name
    block_shape: tl.constexpr  # e.g. (32, 32)
    layout: tl.constexpr

    @triton_jit
    def offset_desc_ptr(
        self,
        off_names: tl.constexpr,
        offsets: tl.tuple,
        offset_can_invalid: tl.constexpr = True,
    ):
        self.assert_is_pointer()
        self.assert_is_valid()

        return aggregate_replace_gluon(
            self,
            # pointer check (not tma) is done in get_offseted_ptr
            ptr=self.mgr.get_offseted_ptr(
                self.name,
                off_names,
                offsets,
                offset_can_invalid=offset_can_invalid,
            ),
        )

    @triton_jit
    def get_dim_size(self, axis: tl.constexpr):
        return self.mgr.get_tensor_dim_size(self.name, axis)

    @triton_jit
    def get_dim_size_by_block_dim(self, block_dim: tl.constexpr):
        block_name: tl.constexpr = sym.get_block_dim_name_by_idx(
            self.block_shape_sym, block_dim
        )
        return self.get_dim_size_by_name(block_name)

    @triton_jit
    def get_dim_size_by_name(self, dim_name: tl.constexpr):
        return self.mgr.get_tensor_dim_size_by_name(self.name, dim_name)

    @triton_jit
    def get_block_size_by_name(self, block_dim_name: tl.constexpr):
        dim: tl.constexpr = sym.get_dim_by_name(self.block_shape_sym, block_dim_name)
        return TritonConstexprField(self.block_shape[dim])

    @triton_jit
    def get_stride(self, axis: tl.constexpr):
        return self.mgr.get_tensor_stride(self.name, axis)

    @triton_jit
    def get_stride_by_name(self, dim_name: tl.constexpr):
        return self.mgr.get_tensor_stride_by_name(self.name, dim_name)

    @triton_jit
    def get_blocked_masks_except_axis(self, axis: tl.constexpr):
        return self.get_blocked_masks(except_axis=axis)

    @triton_jit
    def get_blocked_masks_except_dim_name(self, block_dim_name: tl.constexpr):
        dim: tl.constexpr = sym.get_dim_by_name(self.block_shape_sym, block_dim_name)
        return self.get_blocked_masks_except_axis(dim)

    @triton_jit
    def _get_block_dim_is_divisible(self, axis: tl.constexpr):
        block_size: tl.constexpr = self.block_shape[axis]
        dim = self.get_dim_size_by_block_dim(axis)
        if dim.is_constexpr:
            return TritonConstexprField(dim.value % block_size == 0)
        else:
            dim_multiple_of: tl.constexpr = self.mgr.get_dim_multiple_of(
                sym.get_block_dim_name_by_idx(self.block_shape_sym, axis)
            ).value
            return TritonConstexprField(dim_multiple_of % block_size == 0)

    @triton_jit
    def _get_block_axes_is_divisible(self, except_axis: tl.constexpr = -1):
        block_dim_is_divisible: tl.constexpr = [
            self._get_block_dim_is_divisible(i).value
            for i in tl.tuple(sym.tuple_range(len(self.block_shape)))
        ]
        return TritonConstexprField(
            sym.get_false_inds_of_bool_tuple(block_dim_is_divisible, except_axis)
        )

    @triton_jit
    def is_need_block_mask(self, except_axis: tl.constexpr = -1):
        block_dims_is_divisible = self._get_block_axes_is_divisible(except_axis)
        return TritonConstexprField(len(block_dims_is_divisible.value) > 0)

    @triton_jit
    def get_blocked_masks(self, except_axis: tl.constexpr = -1):
        block_dims_is_divisible = self._get_block_axes_is_divisible(except_axis)
        gl.static_assert(
            len(block_dims_is_divisible.value) > 0,
            "use `is_need_block_mask` to check before get mask.",
        )
        masks = [
            (
                self.get_blocked_offset_by_idx(
                    block_dims_is_divisible.value[i], expand_dims=True
                )
                < self.get_dim_size_by_block_dim(block_dims_is_divisible.value[i]).value
            )
            for i in tl.tuple(sym.tuple_range(len(block_dims_is_divisible.value)))
        ]

        mask = masks[0]
        for i in tl.static_range(1, len(masks)):
            mask = mask & masks[i]
        return mask

    @triton_jit
    def get_default_blocked_io_layout(self):
        return self.get_default_blocked_io_layout_from_block(
            self.block_shape_sym, self.block_shape
        )

    @triton_jit
    def get_blocked_offset_by_idx(
        self,
        idx: tl.constexpr,
        expand_dims: tl.constexpr = False,
        out_of_range: tl.constexpr = 0,
    ):
        block_size: tl.constexpr = self.block_shape[idx]
        if is_gluon():
            res_layout: tl.constexpr = sym.get_sliced_layout(
                self.layout, len(self.block_shape), idx
            )
            res = gl.arange(0, block_size, layout=res_layout)
        else:
            res = tl.arange(0, block_size)
        dim_name: tl.constexpr = sym.get_block_dim_name_by_idx(
            self.block_shape_sym, idx
        )
        dim_name_global: tl.constexpr = sym.local_name_to_global(
            self.shape_sym, dim_name
        )

        if sym.is_name_in_names(dim_name_global, self.mgr._ttfs_internal_offset_names):
            res = (
                res
                + self.mgr._ttfs_internal_offsets[
                    sym.get_name_in_names_idx(
                        dim_name_global, self.mgr._ttfs_internal_offset_names
                    )
                ]
            )
        if out_of_range == 0:
            res_oor = res
        elif out_of_range == 1:
            boundry = self.get_dim_size_by_block_dim(idx)
            res_oor = gl.where(res < boundry.value, res, 0)
        else:
            boundry = self.get_dim_size_by_block_dim(idx)
            # TODO should we use multiple_of in ttfs_dim_multiple_of here if boundry isn't constexpr?
            res_oor = res % boundry.value

        if not expand_dims:
            return res_oor
        else:
            return self.unslice_1d_offset(res_oor, dim_name)

    @triton_jit
    def _get_dim_by_block_dim_name(self, dim_name: tl.constexpr):
        dim_name_unified: tl.constexpr = sym.local_name_to_global(
            self.shape_sym, dim_name
        )
        block_sym_unified: tl.constexpr = sym.local_sym_to_global(
            self.shape_sym, self.block_shape_sym
        )
        block_idx: tl.constexpr = sym.get_dim_by_name(
            block_sym_unified, dim_name_unified
        )
        return TritonConstexprField(block_idx)

    @triton_jit
    def unslice_1d_offset(self, offset_1d, dim_name: tl.constexpr):
        block_idx: tl.constexpr = self._get_dim_by_block_dim_name(dim_name).value
        axes: tl.constexpr = sym.get_sliced_axes(len(self.block_shape), block_idx)
        for i in tl.static_range(len(axes)):
            offset_1d = offset_1d.expand_dims(axes[i])
        return offset_1d

    @triton_jit
    def get_blocked_offset(
        self,
        dim_name: tl.constexpr,
        expand_dims: tl.constexpr = False,
        out_of_range: tl.constexpr = 0,
    ):
        dim: tl.constexpr = self._get_dim_by_block_dim_name(dim_name).value
        return self.get_blocked_offset_by_idx(dim, expand_dims, out_of_range)

    @triton_jit
    def get_blocked_ptrs(self, out_of_range: tl.constexpr = 0):
        strides = [
            self.mgr.get_tensor_stride_with_multiple_of_by_name(
                self.name, sym.get_block_dim_name_by_idx(self.block_shape_sym, i)
            )
            for i in tl.tuple(sym.tuple_range(len(self.block_shape)))
        ]
        offsets = [
            self.get_blocked_offset_by_idx(
                i, expand_dims=True, out_of_range=out_of_range
            )
            for i in tl.tuple(sym.tuple_range(len(self.block_shape)))
        ]

        # triton/gluon analysizer perfer ptr + offset pattern instead of ptr + some_func_or_op(), so we have to inline them here.
        # if use ptr + self.get_blocked_offsets() or use gl.static_range to sum offsets first, triton kernel will use more registers.

        # multiple_of: if last dim isn't constexpr, it must be multiple_of aligned to allow vector access.
        # user can specify it by `stride_multiple_of` in TensorDesc or `ttfs_dim_multiple_of` in TensorManager (for auto inferenced stride).
        if len(strides) == 1:
            return self.ptr + offsets[0] * (
                tl.multiple_of(strides[0].stride.value, strides[0].multiple_of)
                if strides[0].need_multiple_of
                else strides[0].stride.value
            )
        elif len(strides) == 2:
            return (
                self.ptr
                + offsets[0]
                * (
                    tl.multiple_of(strides[0].stride.value, strides[0].multiple_of)
                    if strides[0].need_multiple_of
                    else strides[0].stride.value
                )
                + offsets[1]
                * (
                    tl.multiple_of(strides[1].stride.value, strides[1].multiple_of)
                    if strides[1].need_multiple_of
                    else strides[1].stride.value
                )
            )
        elif len(strides) == 3:
            return (
                self.ptr
                + offsets[0]
                * (
                    tl.multiple_of(strides[0].stride.value, strides[0].multiple_of)
                    if strides[0].need_multiple_of
                    else strides[0].stride.value
                )
                + offsets[1]
                * (
                    tl.multiple_of(strides[1].stride.value, strides[1].multiple_of)
                    if strides[1].need_multiple_of
                    else strides[1].stride.value
                )
                + offsets[2]
                * (
                    tl.multiple_of(strides[2].stride.value, strides[2].multiple_of)
                    if strides[2].need_multiple_of
                    else strides[2].stride.value
                )
            )
        elif len(strides) == 4:
            return (
                self.ptr
                + offsets[0]
                * (
                    tl.multiple_of(strides[0].stride.value, strides[0].multiple_of)
                    if strides[0].need_multiple_of
                    else strides[0].stride.value
                )
                + offsets[1]
                * (
                    tl.multiple_of(strides[1].stride.value, strides[1].multiple_of)
                    if strides[1].need_multiple_of
                    else strides[1].stride.value
                )
                + offsets[2]
                * (
                    tl.multiple_of(strides[2].stride.value, strides[2].multiple_of)
                    if strides[2].need_multiple_of
                    else strides[2].stride.value
                )
                + offsets[3]
                * (
                    tl.multiple_of(strides[3].stride.value, strides[3].multiple_of)
                    if strides[3].need_multiple_of
                    else strides[3].stride.value
                )
            )
        else:
            # general case, but use more registers.
            res = offsets[0]
            for i in gl.static_range(1, len(strides)):
                res += offsets[i] * (
                    tl.multiple_of(strides[i].stride.value, strides[i].multiple_of)
                    if strides[i].need_multiple_of
                    else strides[i].stride.value
                )
            return self.ptr + res

    @triton_jit
    def get_blocked_offsets(self, out_of_range: tl.constexpr = 0):
        strides = [
            self.mgr.get_tensor_stride_with_multiple_of_by_name(
                self.name, sym.get_block_dim_name_by_idx(self.block_shape_sym, i)
            )
            for i in tl.tuple(sym.tuple_range(len(self.block_shape)))
        ]
        offsets = [
            self.get_blocked_offset_by_idx(
                i, expand_dims=True, out_of_range=out_of_range
            )
            for i in tl.tuple(sym.tuple_range(len(self.block_shape)))
        ]
        res = offsets[0]
        for i in gl.static_range(1, len(strides)):
            res += offsets[i] * (
                tl.multiple_of(strides[i].stride.value, strides[i].multiple_of)
                if strides[i].need_multiple_of
                else strides[i].stride.value
            )
        return res

    @triton_jit
    def get_default_smem_layout(self):
        tl.static_assert(is_gluon(), "layout is only supported in gluon backend.")
        gl.static_assert(
            len(self.block_shape) <= 2, "Only support 1D/2D blocked tensor for now."
        )
        return TritonConstexprField(
            gl.NVMMASharedLayout.get_default_for(
                self.block_shape,
                self.get_dtype().value,
                transposed=sym.check_matrix_is_transposed(
                    self.shape_sym, self.block_shape_sym
                ),
            )
        )

    @triton_jit
    def get_default_swizzled_smem_layout(self, op_idx: tl.constexpr):
        tl.static_assert(is_gluon(), "layout is only supported in gluon backend.")
        gl.static_assert(
            len(self.block_shape) == 2, "Only support 2D blocked tensor for now."
        )
        kwidth: tl.constexpr = 32 // self.get_dtype().value.primitive_bitwidth
        is_transposed = sym.check_matrix_is_transposed(
            self.shape_sym, self.block_shape_sym
        )
        return TritonConstexprField(
            get_default_swizzed_shared_layout_nvidia(
                op_idx,
                kwidth,
                self.block_shape,
                [0, 1] if is_transposed else [1, 0],
                self.get_dtype().value,
            )
        )

    @triton_jit
    def allocate_shared_memory(self, num_stages: tl.constexpr):
        tl.static_assert(
            is_gluon(), "allocate_shared_memory is only supported in gluon backend."
        )
        dtype: gl.constexpr = self.get_dtype().value
        # tl.static_print(self.name, "allocate_shared_memory", num_stages, self.block_shape)
        return gl.allocate_shared_memory(
            dtype,
            _insert_front(num_stages, self.block_shape),
            layout=self.get_default_smem_layout().value,
        )

    @triton_jit
    def allocate_single_shared_memory(self):
        tl.static_assert(
            is_gluon(), "allocate_shared_memory is only supported in gluon backend."
        )
        dtype: gl.constexpr = self.get_dtype().value
        return gl.allocate_shared_memory(
            dtype, self.block_shape, layout=self.get_default_smem_layout().value
        )

    @triton_jit
    def simple_load(self, apply_mask: tl.constexpr = True, other=None):
        """Load current block immediately. mask is applied by default.
        Usually used in triton kernels without loop.

        Don't use this function in any loop, use iterator in io.py instead.
        """
        self.assert_is_pointer()
        if apply_mask and self.is_need_block_mask().value:
            mask = self.get_blocked_masks()
        else:
            mask = None
        pointers = self.get_blocked_ptrs()
        if is_gluon():
            return gl.load(pointers, mask, other)
        else:
            return tl.load(pointers, mask, other)

    @triton_jit
    def simple_store(self, value, apply_mask: tl.constexpr = True):
        """Store current block immediately. mask is applied by default.
        Usually used in triton kernels without loop.

        Don't use this function in any loop, use iterator in io.py instead.
        """
        self.assert_is_pointer()
        if apply_mask and self.is_need_block_mask().value:
            mask = self.get_blocked_masks()
        else:
            mask = None
        pointers = self.get_blocked_ptrs()
        if is_gluon():
            gl.store(pointers, value, mask)
        else:
            tl.store(pointers, value, mask)


@aggregate(kw_only=True)
class BlockedTensorDescWithOffset(BlockedTensorDesc):
    offset_tuple: tl.tuple

    @triton_jit
    def get_blocked_offset_by_idx(
        self,
        idx: tl.constexpr,
        expand_dims: tl.constexpr = False,
        out_of_range: tl.constexpr = 0,
    ):
        # triton don't support string constexpr in argument, so we have to use int.
        # out_of_range: 0("none") | 1("zero") | 2("circular")
        block_size: tl.constexpr = self.block_shape[idx]
        if is_gluon():
            res_layout: tl.constexpr = sym.get_sliced_layout(
                self.layout, len(self.block_shape), idx
            )
            res = gl.arange(0, block_size, layout=res_layout) + self.offset_tuple[idx]
        else:
            res = tl.arange(0, block_size) + self.offset_tuple[idx]
        if out_of_range == 0:
            res_oor = res
        elif out_of_range == 1:
            boundry = self.get_dim_size_by_block_dim(idx)
            res_oor = gl.where(res < boundry.value, res, 0)
        else:
            boundry = self.get_dim_size_by_block_dim(idx)
            res_oor = res % boundry.value
        if not expand_dims:
            return res_oor
        else:
            axes: tl.constexpr = sym.get_sliced_axes(len(self.block_shape), idx)
            for i in tl.static_range(len(axes)):
                res_oor = res_oor.expand_dims(axes[i])
            return res_oor

    @triton_jit
    def replace_offset(self, new_offset_tuple):
        gl.static_assert(
            len(new_offset_tuple) == len(self.block_shape),
            "New offset tuple length must match block shape length.",
        )
        if is_gluon():
            return aggregate_replace_gluon(
                self,
                offset_tuple=new_offset_tuple,
            )
        else:
            return aggregate_replace_triton(
                self,
                offset_tuple=new_offset_tuple,
            )

    @triton_jit
    def replace_block_offset(self, new_block_offset_tuple):
        gl.static_assert(
            len(new_block_offset_tuple) == len(self.block_shape),
            "New offset tuple length must match block shape length.",
        )
        if is_gluon():
            return aggregate_replace_gluon(
                self,
                offset_tuple=[
                    new_block_offset_tuple[i] * self.block_shape[i]
                    for i in tl.tuple(sym.tuple_range(len(self.block_shape)))
                ],
            )
        else:
            return aggregate_replace_triton(
                self,
                offset_tuple=[
                    new_block_offset_tuple[i] * self.block_shape[i]
                    for i in tl.tuple(sym.tuple_range(len(self.block_shape)))
                ],
            )
