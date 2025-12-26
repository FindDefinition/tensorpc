from typing import Callable, Optional, TypeVar, cast
import triton
from triton.experimental.gluon import language as gl

from ..aggtype import constexpr_function


@constexpr_function
def _get_shape_per_cta(shape, cta_split_num):
    shape_per_cta = shape
    if cta_split_num is not None:
        assert len(cta_split_num) == len(shape)
        for dim in range(len(shape_per_cta)):
            shape_per_cta[dim] /= cta_split_num[dim]
    return shape_per_cta

@constexpr_function
def get_default_block_layout(shape, size_per_thread, order, num_warps, num_threads_per_warp, cta_split_num=None):
    # from triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td
    rank = len(shape)
    threads_per_warp = [1 for _ in range(rank)]
    warps_per_cta = [1 for _ in range(rank)]
    shape_per_cta = _get_shape_per_cta(shape, cta_split_num)

    remaining_lanes = num_threads_per_warp
    remaining_threads = num_warps * num_threads_per_warp
    remaining_warps = num_warps
    prev_lanes = 1
    prev_warps = 1

    for d in range(rank - 1):
        i = order[d]
        threads_per_cta = min(max(1, shape_per_cta[i] // size_per_thread[i]), remaining_threads)
        threads_per_warp[i] = min(threads_per_cta, remaining_lanes)
        warps_per_cta[i] = min(max(1, threads_per_cta // threads_per_warp[i]), remaining_warps)
        remaining_warps //= warps_per_cta[i]
        remaining_lanes //= threads_per_warp[i]
        remaining_threads //= threads_per_cta
        prev_lanes *= threads_per_warp[i]
        prev_warps *= warps_per_cta[i]
    
    threads_per_warp[order[rank - 1]] = num_threads_per_warp // prev_lanes
    warps_per_cta[order[rank - 1]] = num_warps // prev_warps

    return gl.BlockedLayout(size_per_thread, threads_per_warp, warps_per_cta, order, cta_split_num=cta_split_num)

@constexpr_function
def get_default_block_io_layout(shape, bitwidth, order, num_warps, num_threads_per_warp, cta_split_num=None):
    # from triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td
    size_per_thread = [128 // bitwidth, *[1] * (len(shape) -1)]
    size_per_thread = [size_per_thread[order[i]] for i in range(len(shape))]
    rank = len(shape)
    threads_per_warp = [1 for _ in range(rank)]
    warps_per_cta = [1 for _ in range(rank)]
    shape_per_cta = _get_shape_per_cta(shape, cta_split_num)

    remaining_lanes = num_threads_per_warp
    remaining_threads = num_warps * num_threads_per_warp
    remaining_warps = num_warps
    prev_lanes = 1
    prev_warps = 1

    for d in range(rank - 1):
        i = order[d]
        threads_per_cta = min(max(1, shape_per_cta[i] // size_per_thread[i]), remaining_threads)
        threads_per_warp[i] = min(threads_per_cta, remaining_lanes)
        warps_per_cta[i] = min(max(1, threads_per_cta // threads_per_warp[i]), remaining_warps)
        remaining_warps //= warps_per_cta[i]
        remaining_lanes //= threads_per_warp[i]
        remaining_threads //= threads_per_cta
        prev_lanes *= threads_per_warp[i]
        prev_warps *= warps_per_cta[i]
    
    threads_per_warp[order[rank - 1]] = num_threads_per_warp // prev_lanes
    warps_per_cta[order[rank - 1]] = num_warps // prev_warps

    return gl.BlockedLayout(size_per_thread, threads_per_warp, warps_per_cta, order, cta_split_num=cta_split_num)


@constexpr_function
def get_default_swizzed_shared_layout_nvidia(op_idx: int, k_width: int, block_shape, order, dtype, transposed=False, ctas_per_cga=None, cta_split_num=None,
                        cta_order=None):
    # from triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td
    # TODO check swizzleDotOperandLike
    shape_per_cta = _get_shape_per_cta(block_shape, cta_split_num)
    bitwidth = dtype.primitive_bitwidth
    K = shape_per_cta[order[0]]
    perPhase = max(1024 // (K * bitwidth), 1)
    mmaStride = 8 
    vec = 4 * k_width
    if transposed:
        vec, mmaStride = mmaStride, vec
    rank = len(block_shape)
    kDim = rank - 1 if op_idx == 0 else rank - 2
    if order[0] != kDim:
        vec, mmaStride = mmaStride, vec
    maxPhase = max(min(mmaStride, 1024 // (vec * bitwidth)), 1)
    maxPhase = max(maxPhase // perPhase, 1)
    return gl.SwizzledSharedLayout(vec, perPhase, maxPhase, order, ctas_per_cga=ctas_per_cga,
            cta_split_num=cta_split_num,
            cta_order=cta_order,)

@constexpr_function
def _get_warps_per_cta(version: int, ret_shape: list[int], numWarps: int, instr_shape: list[int], is_chained: bool = False):
    """
    is_chained: 
    // Contains a chained dot. We prefer to assign warps to one axis
    // to facilitate use cases like flash attention, allowing reductions within
    // the same warp.
    """
    if version == 2:
        shapePerWarp = [16, 8]
        warps = [1, 1]
        reps = [triton.cdiv(ret_shape[0], shapePerWarp[0]), triton.cdiv(ret_shape[1], shapePerWarp[1])]
        while warps[0] * warps[1] < numWarps:
            if reps[0] >= reps[1]:
                warps[0] *= 2
                if reps[0] != 1:
                    reps[0] //= 2
            else:
                warps[1] *= 2
                reps[1] //= 2
        # print(ret_shape, reps, warps)
        if is_chained:
            if ret_shape[0] >= ret_shape[1]:
                warps_per_cta = [numWarps, 1]
            else:
                warps_per_cta = [1, numWarps]
            return warps_per_cta
        return warps
    elif version == 3:
        if is_chained:
            return [numWarps, 1]
        ret = [4, 1]
        shapePerWarp = [16, instr_shape[1]]
        while ret[0] * ret[1] < numWarps:
            if ret_shape[0] > shapePerWarp[0] * ret[0]:
                ret[0] *= 2
            else:
                ret[1] *= 2
        return ret
    else:
        raise NotImplementedError

@constexpr_function
def _get_nvidia_mma_version_to_instr_shape(version: int, ret_shape: list[int], dtype, num_warps):
    if version == 1:
        return [16, 16]
    elif version == 2:
        rank = len(ret_shape)
        ret = [1 for _ in range(rank)]
        ret[rank - 1] = 8
        ret[rank - 2] = 16
        return ret
    elif version == 3:
        if ret_shape[0] % 64 != 0 or ret_shape[1] % 8 != 0:
            raise NotImplementedError("type not supported")
        validN = []
        # fn and fnuz: see https://onnx.ai/onnx/technical/float8.html
        # only nan values and no infinite values (FN), no negative zero (UZ)
        # fp8e4nv: Float8E4M3+FN
        # fp8e4b8: Float8E4M3+FNUZ
        # fp8e5: Float8E5M2
        # fp8e5b16: Float8E5M2+FNUZ

        if dtype in [gl.float16, gl.bfloat16, gl.float32, gl.float8e5, gl.float8e4nv, gl.float8e4b8]:
            validN = [256, 248, 240, 232, 224, 216, 208, 200, 192, 184, 176,
                     168, 160, 152, 144, 136, 128, 120, 112, 104, 96,  88,
                     80,  72,  64,  56,  48,  40,  32,  24,  16,  8]
        elif dtype in [gl.int8, gl.float8e4b15]: # see get_fp8e4b15_ty in src/ir.cc 
            validN = [224, 208, 192, 176, 160, 144, 128, 112, 96, 80, 64, 48, 32,
                     24, 16, 8]
        else:
            raise NotImplementedError("type not supported")
        m = 16
        m_warps = max(ret_shape[0] // m, 1)
        n_warps = max(num_warps // m_warps, 1)
        maxN = max(ret_shape[1] // n_warps, 8)
        for n in validN:
            if ret_shape[1] % n == 0 and n <= maxN:
                return [m, n]
        raise NotImplementedError("type not supported")
    elif version == 5:
        m = 128 if ret_shape[0] >= 128 else 64
        n = 256 if ret_shape[1] >= 256 else ret_shape[1]
        k = 256 // dtype.primitive_bitwidth
        return [m, n, k]

@constexpr_function
def get_default_mma(version: int, dtype, ret_shape, numWarps: int, inst_shape= None, is_chained: bool = False,
        warps_per_cta=None):
    # from warpsPerTile in lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp
    if inst_shape is None:
        inst_shape = _get_nvidia_mma_version_to_instr_shape(version, ret_shape, dtype, numWarps)
    if warps_per_cta is None:
        warps_per_cta = _get_warps_per_cta(version, ret_shape, numWarps, inst_shape, is_chained)
    res = gl.NVMMADistributedLayout(
        version=[version, 0],
        warps_per_cta=warps_per_cta,
        instr_shape=inst_shape,
    )
    return res 
