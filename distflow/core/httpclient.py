import asyncio
import json

import aiohttp

from distflow.core import core_io as core_io
from distflow.protos import remote_object_pb2 as remote_object_pb2


async def http_remote_call(sess: aiohttp.ClientSession, url: str, key: str,
                           *args, **kwargs):
    arrays, decoupled = core_io.extract_arrays_from_data((args, kwargs),
                                                         json_index=True)
    arrays = [core_io.data2pb(a) for a in arrays]
    request = remote_object_pb2.RemoteJsonCallRequest(
        service_key=key,
        arrays=arrays,
        data=json.dumps(decoupled),
        callback="")
    async with sess.post(url, data=request.SerializeToString()) as resp:
        data = await resp.read()
    resp_pb = remote_object_pb2.RemoteJsonCallReply()
    resp_pb.ParseFromString(data)
    arrays = [core_io.pb2data(b) for b in resp_pb.arrays]
    data_skeleton = json.loads(resp_pb.data)
    results = core_io.put_arrays_to_data(arrays,
                                         data_skeleton,
                                         json_index=True)
    results = results[0]
    return results


async def main():
    url = "http://localhost:50053/api/jsonrpc"
    async with aiohttp.ClientSession() as session:
        print(await http_remote_call(session, url, "codeai.benchmark.echo", 5))


if __name__ == "__main__":
    asyncio.run(main())
