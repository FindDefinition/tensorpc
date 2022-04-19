import json
import pickle
from collections import abc
from enum import Enum
from functools import reduce
from typing import Any, Dict, Hashable, List, Tuple, Union

import msgpack
import numpy as np
from distflow.protos import arraybuf_pb2, remote_object_pb2

JSON_INDEX_KEY = "__cumm_json_index"


class EncodeMethod(Enum):
    Json = 0
    Pickle = 1
    MessagePack = 2
    JsonArray = 3
    PickleArray = 4
    MessagePackArray = 5


class Placeholder(object):
    def __init__(self, index: int, nbytes: int):
        self.index = index
        self.nbytes = nbytes

    def __add__(self, other):
        assert self.index == other.index
        return Placeholder(self.index, self.nbytes + other.nbytes)

    def __repr__(self):
        return "Placeholder[{},{}]".format(self.index, self.nbytes)

    def __eq__(self, other):
        return self.index == other.index and self.nbytes == other.nbytes


def _inv_map(dict_map: Dict[Hashable, Hashable]) -> Dict[Hashable, Hashable]:
    return {v: k for k, v in dict_map.items()}


def byte_size(obj: Union[bytes, np.ndarray]) -> int:
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    elif isinstance(obj, bytes):
        return len(obj)
    else:
        raise NotImplementedError


NPDTYPE_TO_PB_MAP = {
    np.dtype(np.uint64): arraybuf_pb2.dtype.DataType.uint64,
    np.dtype(np.uint32): arraybuf_pb2.dtype.DataType.uint32,
    np.dtype(np.uint16): arraybuf_pb2.dtype.DataType.uint16,
    np.dtype(np.uint8): arraybuf_pb2.dtype.DataType.uint8,
    np.dtype(np.int64): arraybuf_pb2.dtype.DataType.int64,
    np.dtype(np.int32): arraybuf_pb2.dtype.DataType.int32,
    np.dtype(np.int16): arraybuf_pb2.dtype.DataType.int16,
    np.dtype(np.int8): arraybuf_pb2.dtype.DataType.int8,
    np.dtype(np.float64): arraybuf_pb2.dtype.DataType.float64,
    np.dtype(np.float32): arraybuf_pb2.dtype.DataType.float32,
    np.dtype(np.float16): arraybuf_pb2.dtype.DataType.float16,
    "custom_bytes": arraybuf_pb2.dtype.DataType.CustomBytes,
}

NPDTYPE_TO_JSONARRAY_MAP = {
    np.dtype(np.uint64): 1,
    np.dtype(np.uint32): 2,
    np.dtype(np.uint16): 3,
    np.dtype(np.uint8): 4,
    np.dtype(np.int64): 5,
    np.dtype(np.int32): 6,
    np.dtype(np.int16): 7,
    np.dtype(np.int8): 8,
    np.dtype(np.float64): 9,
    np.dtype(np.float32): 10,
    np.dtype(np.float16): 11,
    np.dtype(np.bool_): 12,
}

INV_NPDTYPE_TO_PB_MAP = _inv_map(NPDTYPE_TO_PB_MAP)
INV_NPDTYPE_TO_JSONARRAY_MAP = _inv_map(NPDTYPE_TO_JSONARRAY_MAP)

NPBYTEORDER_TO_PB_MAP = {
    "=": arraybuf_pb2.dtype.ByteOrder.native,
    "<": arraybuf_pb2.dtype.ByteOrder.littleEndian,
    ">": arraybuf_pb2.dtype.ByteOrder.bigEndian,
    "|": arraybuf_pb2.dtype.ByteOrder.na,
}
INV_NPBYTEORDER_TO_PB_MAP = _inv_map(NPBYTEORDER_TO_PB_MAP)


def bytes2pb(data: bytes, send_data=True) -> arraybuf_pb2.ndarray:
    dtype = arraybuf_pb2.dtype.DataType.CustomBytes
    pb = arraybuf_pb2.ndarray(
        dtype=arraybuf_pb2.dtype(type=dtype),
        shape=[len(data)],
    )
    if send_data:
        pb.data = data
    return pb


def array2pb(array: np.ndarray, send_data=True) -> arraybuf_pb2.ndarray:
    if array.ndim > 0 and send_data:
        array = np.ascontiguousarray(array)
    dtype = NPDTYPE_TO_PB_MAP[array.dtype]
    order = NPBYTEORDER_TO_PB_MAP[array.dtype.byteorder]
    pb_dtype = arraybuf_pb2.dtype(
        type=dtype,
        byte_order=order,
    )
    pb = arraybuf_pb2.ndarray(
        shape=list(array.shape),
        dtype=pb_dtype,
    )
    if send_data:
        pb.data = array.tobytes()
    return pb


def pb2data(buf: arraybuf_pb2.ndarray) -> np.ndarray:
    if buf.dtype.type == arraybuf_pb2.dtype.DataType.CustomBytes:
        return buf.data
    byte_order = INV_NPBYTEORDER_TO_PB_MAP[buf.dtype.byte_order]
    dtype = INV_NPDTYPE_TO_PB_MAP[buf.dtype.type].newbyteorder(byte_order)
    res = np.frombuffer(buf.data, dtype).reshape(list(buf.shape))
    return res


def pb2meta(buf: arraybuf_pb2.ndarray) -> Tuple[List[int], int]:
    if buf.dtype.type == arraybuf_pb2.dtype.DataType.CustomBytes:
        return list(buf.shape), None
    byte_order = INV_NPBYTEORDER_TO_PB_MAP[buf.dtype.byte_order]
    dtype = INV_NPDTYPE_TO_PB_MAP[buf.dtype.type].newbyteorder(byte_order)
    shape = list(buf.shape)
    return (shape, dtype)


def data2pb(array_or_bytes: Union[bytes, np.ndarray],
            send_data=True) -> arraybuf_pb2.ndarray:
    if isinstance(array_or_bytes, np.ndarray):
        return array2pb(array_or_bytes, send_data)
    elif isinstance(array_or_bytes, bytes):
        return bytes2pb(array_or_bytes, send_data)
    else:
        raise NotImplementedError("only support ndarray/bytes.")


class FromBufferStream(object):
    def __init__(self):
        self.current_buf_idx = -1
        self.num_args = -1
        self.current_buf_length = -1
        self.current_buf_shape = None
        self.current_dtype = None
        self.func_key = None
        self.current_datas = []
        self.args = []

    def clear(self):
        self.current_buf_idx = -1
        self.num_args = -1
        self.current_buf_length = -1
        self.current_buf_shape = None
        self.current_dtype = None
        self.func_key = None
        self.current_datas = []
        self.args = []

    def __call__(self, buf):
        if buf.arg_id == 0:
            self.num_args = buf.num_args
        if buf.chunk_id == 0:
            self.current_buf_shape = list(buf.shape)
            self.current_buf_length = buf.num_chunk
            self.current_dtype = buf.dtype
            self.func_key = buf.func_key
            self.current_datas = []
        self.current_datas.append(buf.chunked_data)
        if buf.chunk_id == buf.num_chunk - 1:
            # single arg end, get array
            data = b"".join(self.current_datas)
            assert len(self.current_datas) > 0
            single_buf = arraybuf_pb2.ndarray(
                shape=self.current_buf_shape,
                dtype=self.current_dtype,
                data=data,
            )
            self.args.append(pb2data(single_buf))
            self.current_datas = []
            if buf.arg_id == buf.num_args - 1:
                # end. return args
                assert len(self.args) > 0
                res = self.args
                self.args = []
                return res, self.func_key
        return None


def _div_up(a, b):
    return (a + b - 1) // b


def to_protobuf_stream(data_list: List[Any],
                       func_key,
                       flags: int,
                       chunk_size=32 * 1024):
    if not isinstance(data_list, list):
        raise ValueError("input must be a list")
    streams = []
    arg_ids = list(range(len(data_list)))
    arg_ids[-1] = -1
    num_args = len(data_list)
    for arg_idx, arg in enumerate(data_list):
        if isinstance(arg, np.ndarray):
            ref_buf = array2pb(arg)
            shape = arg.shape
        elif isinstance(arg, bytes):
            ref_buf = bytes2pb(arg)
            shape = []
        else:
            raise NotImplementedError
        data = ref_buf.data
        num_chunk = _div_up(len(data), chunk_size)
        if num_chunk == 0:
            num_chunk = 1  # avoid empty string raise error
        bufs = []
        for i in range(num_chunk):
            buf = remote_object_pb2.RemoteCallStream(
                num_chunk=num_chunk,
                chunk_id=i,
                num_args=num_args,
                arg_id=arg_idx,
                dtype=ref_buf.dtype,
                func_key=func_key,
                chunked_data=data[i * chunk_size:(i + 1) * chunk_size],
                shape=[],
                flags=flags,
            )
            bufs.append(buf)
        assert len(bufs) > 0
        bufs[0].shape[:] = shape
        streams += bufs
    return streams


def is_json_index(data):
    return isinstance(data, dict) and JSON_INDEX_KEY in data


def _extract_arrays_from_data(arrays,
                              data,
                              object_classes=(np.ndarray, bytes),
                              json_index=False):
    # can't use abc.Sequence because string is sequence too.
    if isinstance(data, (list, tuple)):
        data_skeleton = [None] * len(data)
        for i in range(len(data)):
            e = data[i]
            if isinstance(e, object_classes):
                if json_index:
                    data_skeleton[i] = {JSON_INDEX_KEY: len(arrays)}
                else:
                    data_skeleton[i] = Placeholder(len(arrays), byte_size(e))
                arrays.append(e)
            else:
                data_skeleton[i] = _extract_arrays_from_data(
                    arrays, e, object_classes, json_index)
        if isinstance(data, tuple):
            data_skeleton = tuple(data_skeleton)
        return data_skeleton
    elif isinstance(data, abc.Mapping):
        data_skeleton = {}
        for k, v in data.items():
            if isinstance(v, object_classes):
                if json_index:
                    data_skeleton[k] = {JSON_INDEX_KEY: len(arrays)}
                else:
                    data_skeleton[k] = Placeholder(len(arrays), byte_size(v))
                arrays.append(v)
            else:
                data_skeleton[k] = _extract_arrays_from_data(
                    arrays, v, object_classes, json_index)
        return data_skeleton
    else:
        data_skeleton = None
        if isinstance(data, object_classes):
            if json_index:
                data_skeleton = {JSON_INDEX_KEY: len(arrays)}
            else:
                data_skeleton = Placeholder(len(arrays), byte_size(data))
            arrays.append(data)
        else:
            data_skeleton = data
        return data_skeleton


def extract_arrays_from_data(data,
                             object_classes=(np.ndarray, bytes),
                             json_index=False):
    arrays = []
    data_skeleton = _extract_arrays_from_data(arrays,
                                              data,
                                              object_classes=object_classes,
                                              json_index=json_index)
    return arrays, data_skeleton


def put_arrays_to_data(arrays, data_skeleton, json_index=False) -> Any:
    if not arrays:
        return data_skeleton
    return _put_arrays_to_data(arrays, data_skeleton, json_index)


def _put_arrays_to_data(arrays, data_skeleton, json_index=False):
    if isinstance(data_skeleton, (list, tuple)):
        length = len(data_skeleton)
        data = [None] * length
        for i in range(length):
            e = data_skeleton[i]
            if isinstance(e, Placeholder):
                data[i] = arrays[e.index]
            elif is_json_index(e):
                data[i] = arrays[e[JSON_INDEX_KEY]]
            else:
                data[i] = _put_arrays_to_data(arrays, e, json_index)
        if isinstance(data_skeleton, tuple):
            data = tuple(data)
        return data
    elif isinstance(data_skeleton, abc.Mapping):
        data = {}
        for k, v in data_skeleton.items():
            if isinstance(v, Placeholder):
                data[k] = arrays[v.index]
            elif is_json_index(v):
                data[k] = arrays[v[JSON_INDEX_KEY]]
            else:
                data[k] = _put_arrays_to_data(arrays, v, json_index)
        return data
    else:
        if isinstance(data_skeleton, Placeholder):
            data = arrays[data_skeleton.index]
        elif is_json_index(data_skeleton):
            data = arrays[data_skeleton[JSON_INDEX_KEY]]
        else:
            data = data_skeleton
        return data


def _json_dumps_to_binary(obj):
    return json.dumps(obj).encode("ascii")


_METHOD_TO_DUMP = {
    remote_object_pb2.EncodeMethod.Json: _json_dumps_to_binary,
    remote_object_pb2.EncodeMethod.JsonArray: _json_dumps_to_binary,
    remote_object_pb2.EncodeMethod.MessagePack: msgpack.dumps,
    remote_object_pb2.EncodeMethod.MessagePackArray: msgpack.dumps,
    remote_object_pb2.EncodeMethod.Pickle: pickle.dumps,
    remote_object_pb2.EncodeMethod.PickleArray: pickle.dumps,
}

_METHOD_TO_LOAD = {
    remote_object_pb2.EncodeMethod.Json: json.loads,
    remote_object_pb2.EncodeMethod.JsonArray: json.loads,
    remote_object_pb2.EncodeMethod.MessagePack: msgpack.loads,
    remote_object_pb2.EncodeMethod.MessagePackArray: msgpack.loads,
    remote_object_pb2.EncodeMethod.Pickle: pickle.loads,
    remote_object_pb2.EncodeMethod.PickleArray: pickle.loads,
}


def dumps_method(x, method: int):
    method &= _ENCODE_METHOD_MASK
    return _METHOD_TO_DUMP[method](x)


def loads_method(x, method: int):
    method &= _ENCODE_METHOD_MASK
    return _METHOD_TO_LOAD[method](x)


def _enable_json_index(method):
    return method == remote_object_pb2.EncodeMethod.JsonArray or method == remote_object_pb2.EncodeMethod.MessagePackArray


_ENCODE_METHOD_MASK = remote_object_pb2.EncodeMethod.Mask
_ENCODE_METHOD_ARRAY_MASK = remote_object_pb2.EncodeMethod.ArrayMask


def data_to_pb(data, method: int):
    method &= _ENCODE_METHOD_MASK
    if method & _ENCODE_METHOD_ARRAY_MASK:
        arrays, data_skeleton = extract_arrays_from_data(
            data, json_index=_enable_json_index(method))
        data_to_be_send = arrays + [_METHOD_TO_DUMP[method](data_skeleton)]
        data_to_be_send = [data2pb(a) for a in data_to_be_send]
    else:
        data_to_be_send = [data2pb(_METHOD_TO_DUMP[method](data))]
    return data_to_be_send


def data_from_pb(bufs, method: int):
    method &= _ENCODE_METHOD_MASK
    if method & _ENCODE_METHOD_ARRAY_MASK:
        results_raw = [pb2data(b) for b in bufs]
        results_array = results_raw[:-1]
        data_skeleton_bytes = results_raw[-1]
        data_skeleton = _METHOD_TO_LOAD[method](data_skeleton_bytes)
        results = put_arrays_to_data(results_array, data_skeleton,
                                     _enable_json_index(method))
    else:
        results_raw = [pb2data(b) for b in bufs]
        results = _METHOD_TO_LOAD[method](results_raw[-1])
    return results


def data_to_json(data, method: int) -> Tuple[List[arraybuf_pb2.ndarray], str]:
    method &= _ENCODE_METHOD_MASK
    if method == remote_object_pb2.EncodeMethod.JsonArray:
        arrays, decoupled = extract_arrays_from_data(data, json_index=True)
        arrays = [data2pb(a) for a in arrays]
    else:
        arrays = []
        decoupled = data
    return arrays, json.dumps(decoupled)


def data_from_json(bufs: List[arraybuf_pb2.ndarray], data: str, method: int):
    arrays = [pb2data(b) for b in bufs]
    data_skeleton = json.loads(data)
    method &= _ENCODE_METHOD_MASK
    if method == remote_object_pb2.EncodeMethod.JsonArray:
        res = put_arrays_to_data(arrays, data_skeleton, json_index=True)
    else:
        res = data_skeleton
    return res


def align_offset(offset, n):
    """given a byte offset, align it and return an aligned offset
    """
    if n <= 0:
        return offset
    return n * ((offset + n - 1) // n)


def data_to_pb_shmem(data, shared_mem, multi_thread=False, align_nbit=0):
    if not isinstance(shared_mem, np.ndarray):
        raise ValueError("you must provide a np.ndarray")
    arrays, data_skeleton = extract_arrays_from_data(data)
    data_skeleton_bytes = pickle.dumps(data_skeleton)
    data_to_be_send = arrays + [data_skeleton_bytes]
    data_to_be_send = [data2pb(a, send_data=False) for a in data_to_be_send]
    sum_array_nbytes = 0
    array_buffers = []
    for i in range(len(arrays)):
        if isinstance(arrays[i], bytes):
            sum_array_nbytes += len(arrays[i])
            array_buffers.append((arrays[i], len(arrays[i])))
        else:
            arrays[i] = np.ascontiguousarray(arrays[i])
            sum_array_nbytes += arrays[i].nbytes
            array_buffers.append((arrays[i].view(np.uint8), arrays[i].nbytes))
    if sum_array_nbytes + len(data_skeleton_bytes) > shared_mem.nbytes:
        x, y = sum_array_nbytes + len(data_skeleton_bytes), shared_mem.nbytes
        raise ValueError("your shared mem is too small: {} vs {}.".format(
            x, y))
    # assign datas to shared mem
    start = 0
    for a_buf, size in array_buffers:
        start = align_offset(start, align_nbit)
        shared_mem_view = memoryview(shared_mem[start:start + size])
        if not isinstance(a_buf, bytes):
            buf_mem_view = memoryview(a_buf.reshape(-1))
            if multi_thread:  # slow when multi_thread copy in worker
                shared_mem[start:start + size] = a_buf.reshape(-1)
            else:
                shared_mem_view[:] = buf_mem_view
        else:
            shared_mem_view[:] = a_buf
        start += size

    shared_mem[start:start + len(data_skeleton_bytes)] = np.frombuffer(
        data_skeleton_bytes, dtype=np.uint8)
    return data_to_be_send


def data_from_pb_shmem(bufs, shared_mem, copy=True, align_nbit=0):
    results_metas = [pb2meta(b) for b in bufs]
    results_array_metas = results_metas[:-1]
    skeleton_bytes_meta = results_metas[-1]
    results_array = []
    start = 0
    for shape, dtype in results_array_metas:
        start = align_offset(start, align_nbit)
        if dtype is not None:
            length = np.prod(shape, dtype=np.int64,
                             initial=1) * np.dtype(dtype).itemsize
            arr = np.frombuffer(memoryview(shared_mem[start:start + length]),
                                dtype=dtype).reshape(shape)
            if copy:
                arr = arr.copy()
            results_array.append(arr)
        else:
            length = shape[0]
            results_array.append(bytes(shared_mem[start:start + length]))
        start += int(length)
    data_skeleton_bytes = shared_mem[start:start + skeleton_bytes_meta[0][0]]
    data_skeleton = pickle.loads(data_skeleton_bytes)
    results = put_arrays_to_data(results_array, data_skeleton)
    return results


def dumps(obj, multi_thread=False, buffer=None, use_bytearray=False):
    """
    layout:
    +--------------+------------+---------------------------------+--------------+
    |meta_start_pos|meta_end_pos|      array/bytes content        |     meta     |
    +--------------+------------+---------------------------------+--------------+
    data without array/bytes will be saved as bytes in content.
    """
    arrays, data_skeleton = extract_arrays_from_data(obj)
    data_skeleton_bytes = pickle.dumps(data_skeleton)
    data_to_be_send = arrays + [data_skeleton_bytes]
    data_to_be_send = [data2pb(a, send_data=False) for a in data_to_be_send]
    protobuf = remote_object_pb2.RemoteCallReply(arrays=data_to_be_send)
    protobuf_bytes = protobuf.SerializeToString()
    meta_length = len(protobuf_bytes)
    sum_array_nbytes = 0
    array_buffers = []
    for i in range(len(arrays)):
        if isinstance(arrays[i], bytes):
            sum_array_nbytes += len(arrays[i])
            array_buffers.append((arrays[i], len(arrays[i])))
        else:
            # ascontiguous will convert scalar to 1-D array. be careful.
            arrays[i] = np.ascontiguousarray(arrays[i])
            sum_array_nbytes += arrays[i].nbytes
            array_buffers.append((arrays[i].view(np.uint8), arrays[i].nbytes))

    total_length = sum_array_nbytes + len(data_skeleton_bytes) + meta_length
    if buffer is None:
        if not use_bytearray:
            buffer = np.empty(total_length + 16, dtype=np.uint8)
        else:
            buffer = bytearray(total_length + 16)
    else:
        assert len(buffer) >= total_length + 16
    buffer_view = memoryview(buffer)
    content_end_offset = 16 + sum_array_nbytes + len(data_skeleton_bytes)
    meta_end_offset = content_end_offset + meta_length
    buffer_view[:8] = np.array(content_end_offset, dtype=np.int64).tobytes()
    buffer_view[8:16] = np.array(meta_end_offset, dtype=np.int64).tobytes()
    buffer_view[content_end_offset:meta_end_offset] = protobuf_bytes
    shared_mem = np.frombuffer(buffer_view[16:content_end_offset],
                               dtype=np.uint8)
    start = 0
    for a_buf, size in array_buffers:
        shared_mem_view = memoryview(shared_mem[start:start + size])
        if not isinstance(a_buf, bytes):
            buf_mem_view = memoryview(a_buf.reshape(-1))
            if multi_thread:  # slow when multi_thread copy in worker
                shared_mem[start:start + size] = a_buf.reshape(-1)
            else:
                shared_mem_view[:] = buf_mem_view
        else:
            shared_mem_view[:] = a_buf
        start += size

    shared_mem[start:start + len(data_skeleton_bytes)] = np.frombuffer(
        data_skeleton_bytes, dtype=np.uint8)
    return buffer


def loads(binary, copy=False):
    buffer_view = memoryview(binary)
    content_end_offset = np.frombuffer(buffer_view[:8], dtype=np.int64).item()
    meta_end_offset = np.frombuffer(buffer_view[8:16], dtype=np.int64).item()
    pb_bytes = buffer_view[content_end_offset:meta_end_offset]
    shared_mem = buffer_view[16:]
    pb = remote_object_pb2.RemoteCallReply()
    pb.ParseFromString(pb_bytes)

    results_metas = [pb2meta(b) for b in pb.arrays]

    results_array_metas = results_metas[:-1]
    skeleton_bytes_meta = results_metas[-1]
    results_array = []
    start = 0
    for shape, dtype in results_array_metas:
        if dtype is not None:
            length = reduce(lambda x, y: x * y,
                            shape) * np.dtype(dtype).itemsize
            arr = np.frombuffer(memoryview(shared_mem[start:start + length]),
                                dtype=dtype).reshape(shape)
            if copy:
                arr = arr.copy()
            results_array.append(arr)
        else:
            length = shape[0]
            results_array.append(bytes(shared_mem[start:start + length]))
        start += int(length)
    data_skeleton_bytes = shared_mem[start:start + skeleton_bytes_meta[0][0]]
    data_skeleton = pickle.loads(data_skeleton_bytes)
    results = put_arrays_to_data(results_array, data_skeleton)
    return results


def dumps_arraybuf(obj):
    arrays, data_skeleton = extract_arrays_from_data(obj, json_index=True)
    arrays_pb = [data2pb(a) for a in arrays]
    pb = arraybuf_pb2.arrayjson(data=json.dumps(data_skeleton),
                                arrays=arrays_pb)
    return pb.SerializeToString()


def loads_arraybuf(binary: bytes):
    pb = arraybuf_pb2.arrayjson()
    pb.ParseFromString(binary)
    arrays_pb = pb.arrays
    data_skeleton = json.loads(pb.data)
    arrays = [pb2data(a) for a in arrays_pb]
    obj = put_arrays_to_data(arrays, data_skeleton, json_index=True)
    return obj
