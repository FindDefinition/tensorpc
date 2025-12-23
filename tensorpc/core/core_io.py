import json
import pickle
from collections import abc
from enum import Enum, IntEnum
from functools import reduce
import time
from typing import Any, AsyncIterable, AsyncIterator, Callable, Dict, Hashable, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union
from typing_extensions import Literal
from tensorpc.core.client import RemoteException
import msgpack
import numpy as np
from tensorpc.core.tree_id import UniqueTreeIdForComp
from tensorpc.protos_export import arraybuf_pb2, rpc_message_pb2, wsdef_pb2
import traceback
import numpy.typing as npt

try:
    import optree 
    HAS_OPTREE = True
except ImportError:
    HAS_OPTREE = False

JSON_INDEX_KEY = "__jsonarray_index"


class EncodeMethod(Enum):
    Json = 0
    Pickle = 1
    MessagePack = 2
    JsonArray = 3
    PickleArray = 4
    MessagePackArray = 5

class JsonNodeSpecialFlags(IntEnum):
    ARRAY = 0x1
    JSON_ONLY = 0x2
    FREEZE = 0x4

    MASK_ARRAY_AND_JSON_ONLY = ARRAY | JSON_ONLY

class FrontendMsgDecodeFlag(IntEnum):
    NEED_SCAN = 0x1


class Placeholder(object):

    def __init__(self, index: int, nbytes: int, flag: int = int(JsonNodeSpecialFlags.ARRAY), data: Optional[Any] = None):
        self.index = index
        self.nbytes = nbytes
        self.flag = flag
        self.data = data

    def __add__(self, other):
        assert self.index == other.index
        return Placeholder(self.index, self.nbytes + other.nbytes, self.flag, self.data)

    def __repr__(self):
        return "Placeholder[{},{}]".format(self.index, self.nbytes)

    def __eq__(self, other):
        return self.index == other.index and self.nbytes == other.nbytes and self.flag == other.flag


KT = TypeVar("KT")
VT = TypeVar("VT")


def _inv_map(dict_map: Dict[KT, VT]) -> Dict[VT, KT]:
    return {v: k for k, v in dict_map.items()}


def byte_size(obj: Union[bytes, np.ndarray]) -> int:
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    elif isinstance(obj, bytes):
        return len(obj)
    elif isinstance(obj, JSArrayBuffer):
        return len(obj.data)
    else:
        raise NotImplementedError


NPDTYPE_TO_PB_MAP: Dict[np.dtype, "arraybuf_pb2.dtype.DataType"] = {
    np.dtype(np.uint64): arraybuf_pb2.dtype.uint64,
    np.dtype(np.uint32): arraybuf_pb2.dtype.uint32,
    np.dtype(np.uint16): arraybuf_pb2.dtype.uint16,
    np.dtype(np.uint8): arraybuf_pb2.dtype.uint8,
    np.dtype(np.int64): arraybuf_pb2.dtype.int64,
    np.dtype(np.int32): arraybuf_pb2.dtype.int32,
    np.dtype(np.int16): arraybuf_pb2.dtype.int16,
    np.dtype(np.int8): arraybuf_pb2.dtype.int8,
    np.dtype(np.float64): arraybuf_pb2.dtype.float64,
    np.dtype(np.float32): arraybuf_pb2.dtype.float32,
    np.dtype(np.float16): arraybuf_pb2.dtype.float16,
}

NPDTYPE_TO_JSONARRAY_MAP: Dict[np.dtype, int] = {
    np.dtype(np.bool_): 5,
    np.dtype(np.float16): 7,
    np.dtype(np.float32): 0,
    np.dtype(np.float64): 4,
    np.dtype(np.int8): 3,
    np.dtype(np.int16): 2,
    np.dtype(np.int32): 1,
    np.dtype(np.int64): 8,
    np.dtype(np.uint8): 6,
    np.dtype(np.uint16): 9,
    np.dtype(np.uint32): 10,
    np.dtype(np.uint64): 11,
}

BYTES_JSONARRAY_CODE = 100
BYTES_SKELETON_CODE = 101
BYTES_JSONARRAY_ARRAYBUFFER_CODE = 102

INV_NPDTYPE_TO_PB_MAP = _inv_map(NPDTYPE_TO_PB_MAP)
INV_NPDTYPE_TO_JSONARRAY_MAP = _inv_map(NPDTYPE_TO_JSONARRAY_MAP)

NPBYTEORDER_TO_PB_MAP: Dict[Literal["=", "<", ">", "|"],
                            "arraybuf_pb2.dtype.ByteOrder"] = {
                                "=": arraybuf_pb2.dtype.native,
                                "<": arraybuf_pb2.dtype.littleEndian,
                                ">": arraybuf_pb2.dtype.bigEndian,
                                "|": arraybuf_pb2.dtype.na,
                            }
INV_NPBYTEORDER_TO_PB_MAP: Dict["arraybuf_pb2.dtype.ByteOrder",
                                Literal["=", "<", ">",
                                        "|"]] = _inv_map(NPBYTEORDER_TO_PB_MAP)

class JSArrayBuffer:
    def __init__(self, data: bytes, to_uint8array: bool = False) -> None:
        assert isinstance(data, bytes)
        self.data = data
        self.dtype_code = BYTES_JSONARRAY_CODE if to_uint8array else BYTES_JSONARRAY_ARRAYBUFFER_CODE

def bytes2pb(data: bytes, send_data=True) -> arraybuf_pb2.ndarray:
    dtype = arraybuf_pb2.dtype.CustomBytes
    pb = arraybuf_pb2.ndarray(
        dtype=arraybuf_pb2.dtype(type=dtype),
        shape=[len(data)],
    )
    if send_data:
        pb.data = data
    return pb


def array2pb(array: npt.NDArray, send_data=True) -> arraybuf_pb2.ndarray:
    if array.ndim > 0 and send_data:
        if not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array)
    assert isinstance(array, np.ndarray)
    dtype = NPDTYPE_TO_PB_MAP[array.dtype]
    assert array.dtype.byteorder in ("=", "<", ">", "|")
    order = NPBYTEORDER_TO_PB_MAP[array.dtype.byteorder]  # type: ignore
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


def pb2data(buf: arraybuf_pb2.ndarray) -> Union[np.ndarray, bytes, JSArrayBuffer]:
    if buf.dtype.type == arraybuf_pb2.dtype.CustomBytes:
        return buf.data
    elif buf.dtype.type == arraybuf_pb2.dtype.Base64: # TODO change name
        return JSArrayBuffer(buf.data)
    byte_order = INV_NPBYTEORDER_TO_PB_MAP[buf.dtype.byte_order]
    dtype = INV_NPDTYPE_TO_PB_MAP[buf.dtype.type].newbyteorder(byte_order)
    res = np.frombuffer(buf.data, dtype).reshape(list(buf.shape))
    return res


def pb2meta(buf: arraybuf_pb2.ndarray) -> Tuple[List[int], np.dtype]:
    if buf.dtype.type == arraybuf_pb2.dtype.CustomBytes:
        return list(buf.shape), np.dtype(np.uint8)
    byte_order = INV_NPBYTEORDER_TO_PB_MAP[buf.dtype.byte_order]
    dtype = INV_NPDTYPE_TO_PB_MAP[buf.dtype.type].newbyteorder(byte_order)
    shape = list(buf.shape)
    return (shape, dtype)


def data2pb(array_or_bytes: Union[bytes, np.ndarray, JSArrayBuffer],
            send_data=True) -> arraybuf_pb2.ndarray:
    if isinstance(array_or_bytes, np.ndarray):
        return array2pb(array_or_bytes, send_data)
    elif isinstance(array_or_bytes, bytes):
        return bytes2pb(array_or_bytes, send_data)
    elif isinstance(array_or_bytes, JSArrayBuffer):
        return bytes2pb(array_or_bytes.data, send_data)
    else:
        raise NotImplementedError("only support ndarray/bytes.")


class JsonSpecialData:

    def __init__(self, data, flag: int) -> None:
        self.data = data
        self.flag = flag

    def __repr__(self):
        return "JsonSpecialData[{},{}]".format(type(self.data), self.flag)

    @classmethod
    def from_option(cls, data, is_json_only: bool, need_freeze: bool):
        flag = 0
        if is_json_only:
            flag |= JsonNodeSpecialFlags.JSON_ONLY
        if need_freeze:
            flag |= JsonNodeSpecialFlags.FREEZE
        return cls(data, flag)

    def replace_data(self, new_data: Any):
        return JsonSpecialData(new_data, self.flag)

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
        self.accum_buf_length = -1

        self.current_buf_shape = None
        self.current_dtype = None
        self.current_is_np_arr = False
        self.func_key = None
        self.current_datas = []
        self.args = []

    def __call__(self, buf: rpc_message_pb2.RemoteCallStream):
        if buf.arg_id == 0:
            self.num_args = buf.num_args
        if buf.chunk_id == 0:
            self.current_buf_shape = list(buf.shape)
            self.current_buf_length = buf.num_chunk
            self.current_dtype = buf.dtype
            self.func_key = buf.func_key
            self.current_is_np_arr = buf.dtype.type != arraybuf_pb2.dtype.CustomBytes and buf.dtype.type != arraybuf_pb2.dtype.Base64
            self.current_datas = []
            self.accum_buf_length = 0
        if self.current_is_np_arr:
            if buf.chunk_id == 0:
                assert self.current_dtype is not None 
                byte_order = INV_NPBYTEORDER_TO_PB_MAP[self.current_dtype.byte_order]
                dtype = INV_NPDTYPE_TO_PB_MAP[self.current_dtype.type].newbyteorder(byte_order)
                assert self.current_buf_shape is not None 
                res = np.empty(self.current_buf_shape, dtype=dtype)
                self.args.append(res)
            # res_u8_view = memoryview(self.args[-1].view(np.uint8))
            res_u8_view = self.args[-1].reshape(-1).view(np.uint8)
            res_u8_view[self.accum_buf_length:self.accum_buf_length + len(buf.chunked_data)] = np.frombuffer(buf.chunked_data, dtype=np.uint8)
        else:
            self.current_datas.append(buf.chunked_data)
        if buf.chunk_id == buf.num_chunk - 1:
            # single arg end, get array
            if not self.current_is_np_arr:
                data = b"".join(self.current_datas)
                assert len(self.current_datas) > 0
                single_buf = arraybuf_pb2.ndarray(
                    shape=self.current_buf_shape,
                    dtype=self.current_dtype,
                    data=data,
                )
                self.args.append(pb2data(single_buf))
            self.current_datas = []
            self.accum_buf_length = 0
            self.current_is_np_arr = False
            if buf.arg_id == buf.num_args - 1:
                # end. return args
                assert len(self.args) > 0
                res = self.args
                self.args = []
                return res, self.func_key
        self.accum_buf_length += len(buf.chunked_data)
        return None

    def _check_remote_exception(self, exception_bytes: Union[bytes, str]):
        if not exception_bytes:
            return
        exc_dict = json.loads(exception_bytes)
        raise RemoteException(exc_dict["detail"])

    async def generator_async(self, stream_iter: AsyncIterator[rpc_message_pb2.RemoteCallStream]):
        async for request in stream_iter:
            self._check_remote_exception(request.exception)
            res = self(request)
            if res is not None:
                self.clear()
                incoming, _ = res
                arrays = incoming[:-1]
                data_skeleton_bytes = incoming[-1]
                data_skeleton = loads_method(data_skeleton_bytes,
                                                    request.flags)
                data = put_arrays_to_data(
                    arrays, data_skeleton)
                yield request, data

    def generator(self, stream_iter: Iterator[rpc_message_pb2.RemoteCallStream]):
        for request in stream_iter:
            self._check_remote_exception(request.exception)
            res = self(request)
            if res is not None:
                self.clear()
                incoming, _ = res
                arrays = incoming[:-1]
                data_skeleton_bytes = incoming[-1]
                data_skeleton = loads_method(data_skeleton_bytes,
                                                    request.flags)
                data = put_arrays_to_data(
                    arrays, data_skeleton)
                yield request, data

def _div_up(a, b):
    return (a + b - 1) // b


def to_protobuf_stream(data_list: List[Any],
                       func_key,
                       flags: int,
                       chunk_size=256 * 1024):
    return list(to_protobuf_stream_gen(data_list, func_key, flags, chunk_size))

def to_protobuf_stream_gen(data_list: List[Any],
                       func_key,
                       flags: int,
                       chunk_size=256 * 1024):
    if not isinstance(data_list, list):
        raise ValueError("input must be a list")
    arg_ids = list(range(len(data_list)))
    arg_ids[-1] = -1
    num_args = len(data_list)
    for arg_idx, arg in enumerate(data_list):
        if isinstance(arg, np.ndarray):
            data_bytes = None
            order = NPBYTEORDER_TO_PB_MAP[arg.dtype.byteorder]
            data_dtype = arraybuf_pb2.dtype(
                type=NPDTYPE_TO_PB_MAP[arg.dtype],
                byte_order=order,
            )
            # ref_buf = array2pb(arg)
            shape = arg.shape
            length = arg.nbytes
        elif isinstance(arg, bytes):
            data_dtype = arraybuf_pb2.dtype(
                type=arraybuf_pb2.dtype.CustomBytes)
            # ref_buf = bytes2pb(arg)
            data_bytes = arg
            shape = ()
            length = len(data_bytes)
        elif isinstance(arg, JSArrayBuffer):
            data_dtype = arraybuf_pb2.dtype(
                type=arraybuf_pb2.dtype.Base64)
            data_bytes = arg.data
            shape = ()
            length = len(data_bytes)
        else:
            raise NotImplementedError
        # data = ref_buf.data
        num_chunk = _div_up(length, chunk_size)
        if num_chunk == 0:
            num_chunk = 1  # avoid empty string raise error
        if isinstance(arg, np.ndarray):
            if not arg.flags['C_CONTIGUOUS']:
                arg = np.ascontiguousarray(arg)
            arg_view = arg.view(np.uint8).reshape(-1)
            for i in range(num_chunk):
                buf = rpc_message_pb2.RemoteCallStream(
                    num_chunk=num_chunk,
                    chunk_id=i,
                    num_args=num_args,
                    arg_id=arg_idx,
                    dtype=data_dtype,
                    func_key=func_key,
                    chunked_data=arg_view[i * chunk_size:(i + 1) *
                                          chunk_size].tobytes(),
                    shape=[],
                    flags=flags,
                )
                if i == 0:
                    buf.shape[:] = shape
                yield buf
        else:
            assert data_bytes is not None
            for i in range(num_chunk):
                buf = rpc_message_pb2.RemoteCallStream(
                    num_chunk=num_chunk,
                    chunk_id=i,
                    num_args=num_args,
                    arg_id=arg_idx,
                    dtype=data_dtype,
                    func_key=func_key,
                    chunked_data=data_bytes[i * chunk_size:(i + 1) *
                                            chunk_size],
                    shape=[],
                    flags=flags,
                )
                if i == 0:
                    buf.shape[:] = shape
                yield buf

def is_json_index(data, json_idx_key=JSON_INDEX_KEY):
    return isinstance(data, dict) and json_idx_key in data

class _ExtractContext:
    def __init__(self, object_classes: Tuple[Any, ...], json_index: str):
        self.arrays = []
        self.object_classes = object_classes
        self.json_index = json_index
        self.flags = 0

def _extract_arrays_from_data(data,
                              ctx: _ExtractContext):
    # can't use abc.Sequence because string is sequence too.
    # TODO use pytorch optree if available
    object_classes = ctx.object_classes
    json_index = ctx.json_index
    arrays = ctx.arrays
    data_skeleton: Optional[Union[List[Any], Dict[str, Any], Placeholder]]
    if isinstance(data, (list, tuple)):
        data_skeleton = [None] * len(data)
        for i in range(len(data)):
            e = data[i]
            if isinstance(e, object_classes):
                if json_index:
                    data_skeleton[i] = {json_index: [len(arrays), 1]}
                else:
                    data_skeleton[i] = Placeholder(len(arrays), byte_size(e))
                arrays.append(e)
            else:
                data_skeleton[i] = _extract_arrays_from_data(
                    e, ctx)
        data_skeleton_res = data_skeleton
        if isinstance(data, tuple):
            data_skeleton_res = tuple(data_skeleton)
        return data_skeleton_res
    elif isinstance(data, abc.Mapping):
        data_skeleton = {}
        for k, v in data.items():
            if isinstance(v, object_classes):
                if json_index:
                    data_skeleton[k] = {json_index: [len(arrays), 1]}
                else:
                    data_skeleton[k] = Placeholder(len(arrays), byte_size(v))
                arrays.append(v)
            else:
                data_skeleton[k] = _extract_arrays_from_data(
                    v, ctx)
        return data_skeleton
    elif isinstance(data, JsonSpecialData):
        ctx.flags |= FrontendMsgDecodeFlag.NEED_SCAN
        if json_index:
            data_skeleton = {json_index: [data.data, data.flag]}
        else:
            data_skeleton = Placeholder(0, 0, data.flag, data.data)
        return data_skeleton
    elif isinstance(data, UniqueTreeIdForComp):
        # we delay UniqueTreeIdForComp conversion here to allow modify uid
        return data.uid_encoded
    else:
        data_skeleton = None
        if isinstance(data, object_classes):
            if json_index:
                data_skeleton = {json_index: [len(arrays), 1]}
            else:
                data_skeleton = Placeholder(len(arrays), byte_size(data))
            arrays.append(data)
        else:
            data_skeleton = data
        return data_skeleton

def _extract_arrays_from_data_no_unique_id(data, ctx: _ExtractContext):
    # can't use abc.Sequence because string is sequence too.
    # TODO use pytorch optree if available
    object_classes = ctx.object_classes
    json_index = ctx.json_index
    arrays = ctx.arrays
    data_skeleton: Optional[Union[List[Any], Dict[str, Any], Placeholder]]
    if isinstance(data, (list, tuple)):
        data_skeleton = [None] * len(data)
        for i in range(len(data)):
            e = data[i]
            if isinstance(e, object_classes):
                if json_index:
                    data_skeleton[i] = {json_index: [len(arrays), 1]}
                else:
                    data_skeleton[i] = Placeholder(len(arrays), byte_size(e))
                arrays.append(e)
            else:
                data_skeleton[i] = _extract_arrays_from_data_no_unique_id(
                    e, ctx)
        data_skeleton_res = data_skeleton
        if isinstance(data, tuple):
            data_skeleton_res = tuple(data_skeleton)
        return data_skeleton_res
    elif isinstance(data, abc.Mapping):
        data_skeleton = {}
        for k, v in data.items():
            if isinstance(v, object_classes):
                if json_index:
                    data_skeleton[k] = {json_index: [len(arrays), 1]}
                else:
                    data_skeleton[k] = Placeholder(len(arrays), byte_size(v))
                arrays.append(v)
            else:
                data_skeleton[k] = _extract_arrays_from_data_no_unique_id(
                    v, ctx)
        return data_skeleton
    elif isinstance(data, JsonSpecialData):
        ctx.flags |= FrontendMsgDecodeFlag.NEED_SCAN
        if json_index:
            data_skeleton = {json_index: [data.data, data.flag]}
        else:
            data_skeleton = Placeholder(0, 0, data.flag, data.data)
        return data_skeleton
    else:
        data_skeleton = None
        if isinstance(data, object_classes):
            if json_index:
                data_skeleton = {json_index: [len(arrays), 1]}
            else:
                data_skeleton = Placeholder(len(arrays), byte_size(data))
            arrays.append(data)
        else:
            data_skeleton = data
        return data_skeleton


def extract_object_from_data(data,
                             object_classes):
    arrays: List[Any] = []
    ctx = _ExtractContext(object_classes, JSON_INDEX_KEY)
    data_skeleton = _extract_arrays_from_data(data, ctx)
    return arrays, data_skeleton


def extract_arrays_from_data(data,
                             object_classes=(np.ndarray, bytes, JSArrayBuffer),
                             json_index="",
                             handle_unique_tree_id: bool = False,
                             external_ctx: Optional[_ExtractContext] = None):
    if external_ctx is None:
        ctx = _ExtractContext(object_classes, json_index)
    else:
        ctx = external_ctx
    arrays = ctx.arrays
    if HAS_OPTREE:
        variables, structure = optree.tree_flatten(data)
        new_vars = []
        for v in variables:
            if isinstance(v, object_classes):
                if json_index:
                    new_vars.append({json_index: [len(arrays), 1]})
                else:
                    new_vars.append(Placeholder(len(arrays), byte_size(v)))
                arrays.append(v)
            elif isinstance(v, JsonSpecialData):
                ctx.flags |= FrontendMsgDecodeFlag.NEED_SCAN
                if json_index:
                    special = {json_index: [v.data, v.flag]}
                else:
                    special = Placeholder(0, 0, v.flag, v.data)
                new_vars.append(special)
            elif isinstance(v, UniqueTreeIdForComp) and handle_unique_tree_id:
                # we delay UniqueTreeIdForComp conversion here to allow modify uid
                new_vars.append(v.uid_encoded)
            else:
                new_vars.append(v)
        # currently no way to convert structure to json, so we have to build json skeleton manually
        data_skeleton = optree.tree_unflatten(structure, new_vars)
        if arrays:
            ctx.flags |= FrontendMsgDecodeFlag.NEED_SCAN
        return arrays, data_skeleton
    if handle_unique_tree_id:
        data_skeleton = _extract_arrays_from_data(data, ctx)
    else:
        data_skeleton = _extract_arrays_from_data_no_unique_id(data, ctx)
    if arrays:
        ctx.flags |= FrontendMsgDecodeFlag.NEED_SCAN
    return arrays, data_skeleton


def put_arrays_to_data(arrays, data_skeleton, json_index=JSON_INDEX_KEY) -> Any:
    # if not arrays:
    #     return data_skeleton
    return _put_arrays_to_data(arrays, data_skeleton, json_index)


def _put_arrays_to_data(arrays, data_skeleton, json_index=JSON_INDEX_KEY) -> Any:
    if isinstance(data_skeleton, (list, tuple)):
        length = len(data_skeleton)
        data_arr: list[Any] = [None] * length
        for i in range(length):
            e = data_skeleton[i]
            if isinstance(e, Placeholder):
                if (e.flag & JsonNodeSpecialFlags.ARRAY):
                    data_arr[i] = arrays[e.index]
                elif e.flag & JsonNodeSpecialFlags.FREEZE or e.flag & JsonNodeSpecialFlags.JSON_ONLY:
                    data_arr[i] = JsonSpecialData(e.data, e.flag)
                else:
                    data_arr[i] = _put_arrays_to_data(arrays, e.data, json_index)
            elif is_json_index(e, json_index):
                flag = e[json_index][1]
                if flag & JsonNodeSpecialFlags.ARRAY:
                    data_arr[i] = arrays[e[json_index][0]]
                elif flag & JsonNodeSpecialFlags.FREEZE or flag & JsonNodeSpecialFlags.JSON_ONLY:
                    data_arr[i] = JsonSpecialData(e[json_index][0], flag)
                else:
                    data_arr[i] = _put_arrays_to_data(arrays, e[json_index][0], json_index)
            else:
                data_arr[i] = _put_arrays_to_data(arrays, e, json_index)
        if isinstance(data_skeleton, tuple):
            data_arr = tuple(data_arr)
        return data_arr
    elif isinstance(data_skeleton, abc.Mapping):
        data = {}
        for k, v in data_skeleton.items():
            if isinstance(v, Placeholder):
                if (v.flag & JsonNodeSpecialFlags.ARRAY):
                    data[k] = arrays[v.index]
                elif v.flag & JsonNodeSpecialFlags.FREEZE or v.flag & JsonNodeSpecialFlags.JSON_ONLY:
                    data[k] = JsonSpecialData(v.data, v.flag)
                else:
                    data[k] = _put_arrays_to_data(arrays, v.data, json_index)
            elif is_json_index(v, json_index):
                flag = v[json_index][1]
                if flag & JsonNodeSpecialFlags.ARRAY:
                    data[k] = arrays[v[json_index][0]]
                elif flag & JsonNodeSpecialFlags.FREEZE or flag & JsonNodeSpecialFlags.JSON_ONLY:
                    data[k] = JsonSpecialData(v[json_index][0], flag)
                else:
                    data[k] = _put_arrays_to_data(arrays, v[json_index][0], json_index)
            else:
                data[k] = _put_arrays_to_data(arrays, v, json_index)
        return data
    else:
        if isinstance(data_skeleton, Placeholder):
            if (data_skeleton.flag & JsonNodeSpecialFlags.ARRAY):
                data = arrays[data_skeleton.index]
            elif data_skeleton.flag & JsonNodeSpecialFlags.FREEZE or data_skeleton.flag & JsonNodeSpecialFlags.JSON_ONLY:
                data = JsonSpecialData(data_skeleton.data, data_skeleton.flag)
            else:
                data = data_skeleton.data
        elif is_json_index(data_skeleton, json_index):
            flag = data_skeleton[json_index][1]
            if flag & JsonNodeSpecialFlags.ARRAY:
                data = arrays[data_skeleton[json_index][0]]
            elif flag & JsonNodeSpecialFlags.FREEZE or flag & JsonNodeSpecialFlags.JSON_ONLY:
                data = JsonSpecialData(data_skeleton[json_index][0], flag)
            else:
                data = _put_arrays_to_data(arrays, data_skeleton[json_index][0], json_index)
        else:
            data = data_skeleton
        return data


def _json_dumps_to_binary(obj):
    return json.dumps(obj).encode("ascii")


_METHOD_TO_DUMP: Dict[int, Callable] = {
    rpc_message_pb2.Json: _json_dumps_to_binary,
    rpc_message_pb2.JsonArray: _json_dumps_to_binary,
    rpc_message_pb2.MessagePack: msgpack.dumps,
    rpc_message_pb2.MessagePackArray: msgpack.dumps,
    rpc_message_pb2.Pickle: pickle.dumps,
    rpc_message_pb2.PickleArray: pickle.dumps,
}

_METHOD_TO_LOAD: Dict[int, Callable] = {
    rpc_message_pb2.Json: json.loads,
    rpc_message_pb2.JsonArray: json.loads,
    rpc_message_pb2.MessagePack: msgpack.loads,
    rpc_message_pb2.MessagePackArray: msgpack.loads,
    rpc_message_pb2.Pickle: pickle.loads,
    rpc_message_pb2.PickleArray: pickle.loads,
}


def dumps_method(x, method: int):
    method &= _ENCODE_METHOD_MASK
    return _METHOD_TO_DUMP[method](x)


def loads_method(x, method: int):
    method &= _ENCODE_METHOD_MASK
    return _METHOD_TO_LOAD[method](x)


def _enable_json_index(method):
    return method == rpc_message_pb2.JsonArray or method == rpc_message_pb2.MessagePackArray


_ENCODE_METHOD_MASK = rpc_message_pb2.Mask
_ENCODE_METHOD_ARRAY_MASK = rpc_message_pb2.ArrayMask


def data_to_pb(data, method: int):
    method &= _ENCODE_METHOD_MASK
    if method & _ENCODE_METHOD_ARRAY_MASK:
        arrays, data_skeleton = extract_arrays_from_data(
            data, json_index=JSON_INDEX_KEY if _enable_json_index(method) else "")
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
                                     JSON_INDEX_KEY if _enable_json_index(method) else "")
    else:
        results_raw = [pb2data(b) for b in bufs]
        results = _METHOD_TO_LOAD[method](results_raw[-1])
    return results


def data_to_json(data, method: int) -> Tuple[List[arraybuf_pb2.ndarray], str]:
    method &= _ENCODE_METHOD_MASK
    if method == rpc_message_pb2.JsonArray:
        arrays, decoupled = extract_arrays_from_data(data, json_index=JSON_INDEX_KEY)
        arrays = [data2pb(a) for a in arrays]
    else:
        arrays = []
        decoupled = data
    return arrays, json.dumps(decoupled)


def data_from_json(bufs: Sequence[arraybuf_pb2.ndarray], data: str, method: int):
    arrays = [pb2data(b) for b in bufs]
    data_skeleton = json.loads(data)
    method &= _ENCODE_METHOD_MASK
    if method == rpc_message_pb2.JsonArray:
        res = put_arrays_to_data(arrays, data_skeleton, json_index=JSON_INDEX_KEY)
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
        arr = arrays[i]
        if isinstance(arr, (bytes, memoryview, bytearray)):
            sum_array_nbytes += len(arr)
            array_buffers.append((arr, len(arr)))
        elif isinstance(arr, JSArrayBuffer):
            sum_array_nbytes += len(arr.data)
            array_buffers.append((arr.data, len(arr.data)))
        else:
            if not arr.flags['C_CONTIGUOUS']:
                arrays[i] = np.ascontiguousarray(arr)
            sum_array_nbytes += arr.nbytes
            array_buffers.append((arr.view(np.uint8), arr.nbytes))
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
    protobuf = rpc_message_pb2.RemoteCallReply(arrays=data_to_be_send)
    protobuf_bytes = protobuf.SerializeToString()
    meta_length = len(protobuf_bytes)
    sum_array_nbytes = 0
    array_buffers = []
    for i in range(len(arrays)):
        arr = arrays[i]
        if isinstance(arr, (bytes, memoryview, bytearray)):
            sum_array_nbytes += len(arr)
            array_buffers.append((arr, len(arr)))
        elif isinstance(arr, JSArrayBuffer):
            sum_array_nbytes += len(arr.data)
            array_buffers.append((arr.data, len(arr.data)))
        else:
            # ascontiguous will convert scalar to 1-D array. be careful.
            if not arr.flags['C_CONTIGUOUS']:
                arrays[i] = np.ascontiguousarray(arr)

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
    pb = rpc_message_pb2.RemoteCallReply()
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
    arrays, data_skeleton = extract_arrays_from_data(obj, json_index=JSON_INDEX_KEY)
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
    obj = put_arrays_to_data(arrays, data_skeleton, json_index=JSON_INDEX_KEY)
    return obj


class SocketMsgType(Enum):
    Subscribe = 0x01
    UnSubscribe = 0x02
    RPC = 0x03
    Event = 0x04
    Chunk = 0x05
    QueryServiceIds = 0x06
    Notification = 0x07
    EventChunk = 0x08
    HeaderChunk = 0x09

    Ping = 0x0a
    Pong = 0x0b
    ResetLargeDataClient = 0x0c

    EventError = 0x10
    RPCError = 0x20
    UserError = 0x30
    SubscribeError = 0x40
    OnConnectError = 0x50

    ErrorMask = 0xF0


def encode_protobuf_uint(val: int):
    """this function encode protobuf fised uint to make sure
    message size is stable.
    """
    assert val >= 0
    return val + 1


def decode_protobuf_uint(val: int):
    return val - 1


def json_only_encode(data, type: SocketMsgType, req: wsdef_pb2.Header):
    req.data = json.dumps(data)
    req_msg_size = req.ByteSize()
    final_size = 5 + req_msg_size
    cnt_arr = np.array([0], np.int32)
    binary = bytearray(final_size)
    binary_view = memoryview(binary)
    binary_view[0] = type.value
    cnt_arr[0] = req_msg_size
    binary_view[1:5] = cnt_arr.tobytes()
    binary_view[5:req_msg_size + 5] = req.SerializeToString()
    return binary


class SocketMessageEncoder:
    """
    tensorpc socket message format

    0-1: msg type, can be rpc/event/error/raw

    if type is raw, following bytes are raw byte message.

    if not:

    1~5: header length
    5~X: header 
    X~Y: array data

    """

    def __init__(
        self, data, skeleton_size_limit: int = int(1024 * 1024 * 3.6)) -> None:
        # unique tree id obj will only be handled in websocket.
        ctx = _ExtractContext(
            (np.ndarray, bytes, JSArrayBuffer), JSON_INDEX_KEY)
        arrays, data_skeleton = extract_arrays_from_data(data, json_index=JSON_INDEX_KEY, handle_unique_tree_id=True, external_ctx=ctx)
        self.arrays: List[Union[np.ndarray, bytes, JSArrayBuffer]] = arrays
        self.data_skeleton = data_skeleton
        self._flags = ctx.flags
        self._total_size = 0
        self._arr_metadata: List[Tuple[int, List[int]]] = []
        for arr in self.arrays:
            if isinstance(arr, np.ndarray):
                self._total_size += arr.nbytes
                self._arr_metadata.append(
                    (NPDTYPE_TO_JSONARRAY_MAP[arr.dtype], list(arr.shape)))
            elif isinstance(arr, JSArrayBuffer):
                self._total_size += len(arr.data)
                self._arr_metadata.append((arr.dtype_code, [len(arr.data)]))
            else:
                self._total_size += len(arr)
                self._arr_metadata.append((BYTES_JSONARRAY_CODE, [len(arr)]))
        self._ser_skeleton = json.dumps(self.get_skeleton())
        # print(self._ser_skeleton)
        if len(self._ser_skeleton) > skeleton_size_limit:
            data_skeleton_pack = msgpack.packb(self.data_skeleton)
            assert data_skeleton_pack is not None
            self.arrays.append(data_skeleton_pack)
            self._total_size += len(data_skeleton_pack)
            self._arr_metadata.append(
                (BYTES_SKELETON_CODE, [len(data_skeleton_pack)]))
            self.data_skeleton = {}
            self._ser_skeleton = json.dumps(self.get_skeleton())

    def get_total_array_binary_size(self):
        return self._total_size

    def get_skeleton(self):
        return [self._flags, self._arr_metadata, self.data_skeleton]

    def get_message_chunks(self, type: SocketMsgType, req: wsdef_pb2.Header,
                           chunk_size: int):
        req.data = self._ser_skeleton
        req_msg_size = req.ByteSize()
        if req_msg_size + 5 > chunk_size:
            print(req_msg_size, self._ser_skeleton)

        final_size = 5 + req_msg_size + self.get_total_array_binary_size()
        cnt_arr = np.array([0], np.int32)
        if final_size < chunk_size:
            binary = bytearray(final_size)
            binary_view = memoryview(binary)
            binary_view[0] = type.value
            cnt_arr[0] = req_msg_size
            binary_view[1:5] = cnt_arr.tobytes()
            binary_view[5:req_msg_size + 5] = req.SerializeToString()
            start = req_msg_size + 5
    
            for arr in self.arrays:
                if isinstance(arr, np.ndarray):
                    buff2 = arr.reshape(-1).view(np.uint8).data
                    binary_view[start:start + arr.nbytes] = buff2
                    start += arr.nbytes
                elif isinstance(arr, JSArrayBuffer):
                    binary_view[start:start + len(arr.data)] = arr.data
                    start += len(arr.data)
                else:
                    # bytes
                    binary_view[start:start + len(arr)] = arr
                    start += len(arr)
            yield binary
            return
        assert req_msg_size + 5 <= chunk_size, "req size must smaller than chunk size"
        # if field of fixedXX is zero, it will be ignored. so all value of protobuf MUST LARGER THAN ZERO here.
        chunk_header = wsdef_pb2.Header(
            service_id=encode_protobuf_uint(req.service_id),
            chunk_index=encode_protobuf_uint(0),
            rpc_id=encode_protobuf_uint(req.rpc_id),
            data="")

        header_msg_size = chunk_header.ByteSize()
        chunk_size_for_arr = chunk_size - header_msg_size - 5
        assert chunk_size_for_arr > 0
        num_chunks = _div_up(self._total_size, chunk_size_for_arr)
        req.chunk_index = num_chunks

        # req msg size will change if value changed.
        req_msg_size = req.ByteSize()

        res_header_binary = bytearray(req_msg_size + 5)
        res_header_binary[0] = type.value
        cnt_arr[0] = req_msg_size
        res_header_binary[1:5] = cnt_arr.tobytes()
        res_header_binary[5:req_msg_size + 5] = req.SerializeToString()
        yield res_header_binary
        # breakpoint()

        # req2 = wsdef_pb2.Header()
        # req2.ParseFromString(res_header_binary[5:req_msg_size + 5])
        chunk = bytearray(chunk_size)
        if type == SocketMsgType.Event:
            chunk[0] = SocketMsgType.EventChunk.value
        else:
            chunk[0] = SocketMsgType.Chunk.value
        cnt_arr[0] = header_msg_size
        chunk[1:5] = cnt_arr.tobytes()
        chunk[5:header_msg_size + 5] = chunk_header.SerializeToString()
        chunk_idx = 0
        start = header_msg_size + 5
        remain_msg_size = num_chunks * (
            header_msg_size + 5) + self.get_total_array_binary_size()
        chunk_remain_size = min(remain_msg_size,
                                chunk_size) - header_msg_size - 5
        for arr in self.arrays:
            if isinstance(arr, np.ndarray):
                size = arr.nbytes
                memview = arr.reshape(-1).view(np.uint8).data
            elif isinstance(arr, JSArrayBuffer):
                size = len(arr.data)
                memview = arr.data
            else:
                size = len(arr)
                memview = memoryview(arr)
            arr_start = 0
            while size > 0:
                ser_size = min(size, chunk_remain_size)
                chunk[start:start + ser_size] = memview[arr_start:arr_start +
                                                        ser_size]
                arr_start += ser_size
                start += ser_size
                chunk_remain_size -= ser_size
                size -= ser_size
                if chunk_remain_size == 0:
                    yield chunk
                    chunk_idx += 1
                    if chunk_idx != num_chunks:
                        remain_msg_size -= chunk_size
                        chunk = bytearray(min(remain_msg_size, chunk_size))
                        chunk_remain_size = len(chunk) - header_msg_size - 5
                        if type == SocketMsgType.Event:
                            chunk[0] = SocketMsgType.EventChunk.value
                        else:
                            chunk[0] = SocketMsgType.Chunk.value
                        cnt_arr[0] = header_msg_size

                        chunk[1:5] = cnt_arr.tobytes()
                        chunk_header.chunk_index = encode_protobuf_uint(
                            chunk_idx)

                        chunk[5:header_msg_size +
                              5] = chunk_header.SerializeToString()
                        start = header_msg_size + 5


class TensoRPCHeader:

    def __init__(self, binary) -> None:
        self.req_length = np.frombuffer(binary[1:5], dtype=np.int32)[0]
        req_arr = binary[5:self.req_length + 5]
        req = wsdef_pb2.Header()
        req.ParseFromString(req_arr)

        self.type = SocketMsgType(binary[0])
        self.req = req


def parse_array_of_chunked_message(req: wsdef_pb2.Header, chunks: List[bytes]):
    assert req.chunk_index > 0, "chunked message req must have chunk_index > 0"
    meta, skeleton = json.loads(req.data)
    # chunked
    num_chunks = req.chunk_index
    # print(num_chunks, len(chunks) - 1)
    assert num_chunks == len(chunks)
    cur_chunk = chunks[0]
    chunk_idx = 0
    chunk_header_length = np.frombuffer(cur_chunk[1:5], dtype=np.int32)[0]
    chunk_size = len(cur_chunk) - 5 - chunk_header_length
    cur_chunk_start = 5 + chunk_header_length
    # print("START", chunk_header_length)
    arrs: List[Union[bytes, np.ndarray]] = []
    for dtype_jarr, shape in meta:
        if dtype_jarr == BYTES_JSONARRAY_CODE or dtype_jarr == BYTES_JSONARRAY_ARRAYBUFFER_CODE:
            data = np.empty(shape, np.uint8)
            data_buffer = data.reshape(-1).view(np.uint8).data
            size = shape[0]
        else:
            dtype_np = INV_NPDTYPE_TO_JSONARRAY_MAP[dtype_jarr]
            size = shape[0] * dtype_np.itemsize
            for s in shape[1:]:
                size *= s
            data = np.empty(shape, dtype_np)
            data_buffer = data.reshape(-1).view(np.uint8).data
        arr_start = 0
        while size > 0:
            ser_size = min(size, chunk_size)
            data_buffer[arr_start:arr_start +
                        ser_size] = cur_chunk[cur_chunk_start:cur_chunk_start +
                                              ser_size]
            # print(len(cur_chunk), ser_size, arr_start, len(data_buffer), chunk_size)
            size -= ser_size
            chunk_size -= ser_size
            cur_chunk_start += ser_size
            arr_start += ser_size
            if chunk_size == 0 and chunk_idx != num_chunks - 1:
                chunk_idx += 1
                cur_chunk = chunks[chunk_idx]
                chunk_header_length = np.frombuffer(cur_chunk[1:5],
                                                    dtype=np.int32)[0]
                chunk_size = len(cur_chunk) - 5 - chunk_header_length
                cur_chunk_start = 5 + chunk_header_length
        if dtype_jarr == BYTES_JSONARRAY_CODE or dtype_jarr == BYTES_JSONARRAY_ARRAYBUFFER_CODE:
            arrs.append(data.tobytes())
        else:
            arrs.append(data)

    return put_arrays_to_data(arrs, skeleton)


def parse_message_chunks(header: TensoRPCHeader, chunks: List[bytes]):
    req = header.req
    meta, skeleton = json.loads(req.data)
    if req.chunk_index == 0:
        # not chunked
        data_arr = chunks[0][header.req_length + 5:]
        start = 0
        arrs: List[Union[npt.NDArray, bytes]] = []
        for dtype_jarr, shape in meta:
            if dtype_jarr == BYTES_JSONARRAY_CODE or dtype_jarr == BYTES_JSONARRAY_ARRAYBUFFER_CODE:
                arrs.append(data_arr[start:start + shape[0]])
                start += shape[0]
            else:
                dtype_np = INV_NPDTYPE_TO_JSONARRAY_MAP[dtype_jarr]

                size = shape[0] * dtype_np.itemsize
                for s in shape[1:]:
                    size *= s
                arrs.append(
                    np.frombuffer(data_arr[start:start + size],
                                  dtype=dtype_np).reshape(shape))
                start += size
        data = put_arrays_to_data(arrs, skeleton, JSON_INDEX_KEY)
        return data
    res = parse_array_of_chunked_message(req, chunks)
    return res


def get_error_json(type: str, detail: str):
    return {"error": type, "detail": detail}


def get_exception_json(exc: BaseException):
    detail = traceback.format_exc()
    exception_json = {"error": str(exc), "detail": detail}
    return exception_json

def _main():
    data = {'': [['position-x', '65.490|(minimap.scrollValueX - 0.5) * (1 - minimap.layout.scrollFactorX).{"type":34,"st":{"type":-1},"left":{"type":34,"st":{"type":-1},"left":{"type":43,"st":{"type":-1},"value":{"type":39,"st":{"type":-1},"id":"minimap"},"attr":"scrollValueX"},"right":{"type":40,"st":{"type":0},"value":0.5},"op":1},"right":{"type":34,"st":{"type":-1},"left":{"type":40,"st":{"type":0},"value":1},"right":{"type":43,"st":{"type":-1},"value":{"type":43,"st":{"type":-1},"value":{"type":39,"st":{"type":-1},"id":"minimap"},"attr":"layout"},"attr":"scrollFactorX"},"op":1},"op":2}'], ['position-y', '66.536|-(minimap.scrollValueY - 0.5) * (1 - minimap.layout.scrollFactorY).{"type":34,"st":{"type":-1},"left":{"type":35,"st":{"type":-1},"op":3,"operand":{"type":34,"st":{"type":-1},"left":{"type":43,"st":{"type":-1},"value":{"type":39,"st":{"type":-1},"id":"minimap"},"attr":"scrollValueY"},"right":{"type":40,"st":{"type":0},"value":0.5},"op":1}},"right":{"type":34,"st":{"type":-1},"left":{"type":40,"st":{"type":0},"value":1},"right":{"type":43,"st":{"type":-1},"value":{"type":43,"st":{"type":-1},"value":{"type":39,"st":{"type":-1},"id":"minimap"},"attr":"layout"},"attr":"scrollFactorY"},"op":1},"op":2}'], ['scale-x', '28.156|minimap.layout.scrollFactorX.{"type":43,"st":{"type":-1},"value":{"type":43,"st":{"type":-1},"value":{"type":39,"st":{"type":-1},"id":"minimap"},"attr":"layout"},"attr":"scrollFactorX"}'], ['scale-y', '28.156|minimap.layout.scrollFactorY.{"type":43,"st":{"type":-1},"value":{"type":43,"st":{"type":-1},"value":{"type":39,"st":{"type":-1},"id":"minimap"},"attr":"layout"},"attr":"scrollFactorY"}']]}
    arrs, sk = extract_arrays_from_data(data, json_index=JSON_INDEX_KEY)
    data_recover = put_arrays_to_data(arrs, sk, json_index=JSON_INDEX_KEY)

if __name__ == "__main__":
    _main()