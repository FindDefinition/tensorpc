import asyncio
import json
import traceback
from typing import Any, Dict, List, Optional

import aiohttp
import requests
from tensorpc.core import core_io as core_io
from tensorpc.core.client import RemoteManager
from tensorpc.protos_export import rpc_message_pb2, wsdef_pb2
from tensorpc.core.client import RemoteException
import time


async def http_remote_call(sess: aiohttp.ClientSession, url: str, key: str,
                           *args, **kwargs):
    arrays, decoupled = core_io.extract_arrays_from_data((args, kwargs),
                                                         json_index=True)
    arrays = [core_io.data2pb(a) for a in arrays]
    request = rpc_message_pb2.RemoteJsonCallRequest(service_key=key,
                                                    arrays=arrays,
                                                    data=json.dumps(decoupled),
                                                    callback="")
    async with sess.post(url, data=request.SerializeToString()) as resp:
        data = await resp.read()
    if resp.status != 200:
        raise ValueError(
            f"Http Post {url} {key} Failed with Status {resp.status}")
    resp_pb = rpc_message_pb2.RemoteJsonCallReply()
    resp_pb.ParseFromString(data)
    if resp_pb.exception != "":
        exc_dict = json.loads(resp_pb.exception)
        raise RemoteException(exc_dict["detail"])
    arrays = [core_io.pb2data(b) for b in resp_pb.arrays]
    data_skeleton = json.loads(resp_pb.data)
    results = core_io.put_arrays_to_data(arrays,
                                         data_skeleton,
                                         json_index=True)
    results = results[0]
    return results


def http_remote_call_request(url: str, key: str, *args, **kwargs):
    arrays, decoupled = core_io.extract_arrays_from_data((args, kwargs),
                                                         json_index=True)
    arrays = [core_io.data2pb(a) for a in arrays]
    request = rpc_message_pb2.RemoteJsonCallRequest(service_key=key,
                                                    arrays=arrays,
                                                    data=json.dumps(decoupled),
                                                    callback="")
    res = requests.post(url, data=request.SerializeToString())
    if res.status_code != 200:
        raise ValueError(
            f"Http Post {url} {key} Failed with Status {res.status_code}")
    resp_pb = rpc_message_pb2.RemoteJsonCallReply()
    resp_pb.ParseFromString(res.content)
    if resp_pb.exception != "":
        exc_dict = json.loads(resp_pb.exception)
        raise RemoteException(exc_dict["detail"])
    arrays = [core_io.pb2data(b) for b in resp_pb.arrays]
    data_skeleton = json.loads(resp_pb.data)
    results = core_io.put_arrays_to_data(arrays,
                                         data_skeleton,
                                         json_index=True)
    results = results[0]
    return results


class PendingReply:

    def __init__(self,
                 rpc_id: int,
                 service_id: int,
                 is_event: bool,
                 header: Optional[wsdef_pb2.Header] = None) -> None:
        self.num_chunk = -1
        self.chunks: List[Optional[Any]] = []
        self.avail_cnt = 0
        self.rpc_id = rpc_id
        self.service_id = service_id

        self.is_event = is_event

        self.future: asyncio.Future[Any] = asyncio.Future()

        self.header = header

    def insert_chunk(self, data, chunk_idx: int):
        assert self.chunks[chunk_idx] is None
        self.chunks[chunk_idx] = data
        self.avail_cnt += 1
        return self.avail_cnt == self.num_chunk

    def init_chunks(self, num_chunk: int):
        self.num_chunk = num_chunk
        self.chunks = [None] * num_chunk

    def get_chunks(self):
        return self.chunks


class WebsocketClient:

    def __init__(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        self._ws = ws
        self._name_to_serv_id: Dict[str, int] = {}
        self._rpc_id_to_pending: Dict[int, PendingReply] = {}
        self._event_id_to_pending: Dict[int, PendingReply] = {}

        self._event_id_to_handler: Dict[int, Any] = {}

        self._msg_max_size = 1 << 20
        self._loop_task = asyncio.create_task(self._loop())
        self._loop_inited = asyncio.Event()
        # self._shutdown_ev = asyncio.Event()
        # self._shutdown_task = asyncio.create_task(self._shutdown_ev.wait())

    async def _loop(self):
        enc = core_io.SocketMessageEncoder([])
        msg = list(
            enc.get_message_chunks(core_io.SocketMsgType.QueryServiceIds,
                                   wsdef_pb2.Header(), 1048576))
        # recv_task = asyncio.create_task(self._ws.receive())
        try:
            await self._ws.send_bytes(msg[0])
            # while True:
            #     await asyncio.wait([recv_task, self._shutdown_task], return_when=asyncio.FIRST_COMPLETED)
            #     if self._shutdown_task.done():
            #         break
            #     msg = recv_task.result()
            #     recv_task = asyncio.create_task(self._ws.receive())
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    data = msg.data
                    msg_header = core_io.TensoRPCHeader(data)
                    msg_type = msg_header.type
                    req = msg_header.req
                    # serv_key: str = self._serv_id_to_name[req.service_id]
                    if msg_type.value & core_io.SocketMsgType.ErrorMask.value:
                        data = json.loads(req.data)
                        exc = RemoteException(data["error"], data["detail"])
                        if req.rpc_id in self._rpc_id_to_pending:
                            self._rpc_id_to_pending[
                                req.rpc_id].future.set_exception(exc)
                        else:
                            print(json.dumps(data, indent=2))
                    elif msg_type == core_io.SocketMsgType.QueryServiceIds:
                        self._name_to_serv_id = json.loads(msg_header.req.data)
                        self._loop_inited.set()
                    elif msg_type == core_io.SocketMsgType.Event:
                        rpc_id = req.rpc_id
                        pending = PendingReply(rpc_id, req.service_id, True,
                                               req)
                        if req.chunk_index == 0:
                            if req.service_id in self._event_id_to_handler:
                                res = core_io.parse_message_chunks(
                                    msg_header, [data])
                                try:
                                    self._event_id_to_handler[
                                        pending.service_id](res[0])
                                except:
                                    traceback.print_exc()
                        else:
                            pending.header = req
                            pending.init_chunks(req.chunk_index)
                            self._event_id_to_pending[rpc_id] = pending
                    elif msg_type == core_io.SocketMsgType.RPC:
                        # fill pending reply. if finished, call resolve
                        rpc_id = req.rpc_id
                        if rpc_id not in self._rpc_id_to_pending:
                            # may be removed if timeout
                            print(
                                f"{rpc_id} not found, may be removed when timeout."
                            )
                            continue
                        num_chunk = req.chunk_index
                        pending = self._rpc_id_to_pending[rpc_id]
                        if num_chunk > 0:
                            pending.header = req
                            pending.init_chunks(num_chunk)
                        else:
                            res = core_io.parse_message_chunks(
                                msg_header, [data])
                            pending.future.set_result(res[0])
                            self._rpc_id_to_pending.pop(rpc_id)
                    elif msg_type == core_io.SocketMsgType.Chunk or msg_type == core_io.SocketMsgType.EventChunk:
                        # fill pending reply. if finished, call resolve

                        is_ev = msg_type == core_io.SocketMsgType.EventChunk
                        rpc_pendings = self._rpc_id_to_pending if not is_ev else self._event_id_to_pending
                        rpc_id = core_io.decode_protobuf_uint(req.rpc_id)
                        if rpc_id not in rpc_pendings:
                            # may be removed if timeout
                            print(
                                f"CHUNK {rpc_id} not found, may be removed when timeout."
                            )
                            continue
                        assert rpc_id in rpc_pendings
                        pending = rpc_pendings[rpc_id]
                        try:
                            if pending.insert_chunk(
                                    data,
                                    core_io.decode_protobuf_uint(
                                        req.chunk_index)):
                                rpc_pendings.pop(rpc_id)
                                # finished
                                if pending.is_event:
                                    if pending.service_id in self._event_id_to_handler:
                                        assert pending.header is not None
                                        res = core_io.parse_array_of_chunked_message(
                                            pending.header,
                                            pending.get_chunks())
                                        try:
                                            self._event_id_to_handler[
                                                pending.service_id](res[0])
                                        except:
                                            traceback.print_exc()
                                else:
                                    assert pending.header is not None
                                    res = core_io.parse_array_of_chunked_message(
                                        pending.header, pending.get_chunks())
                                    pending.future.set_result(res[0])
                        except:
                            traceback.print_exc()
                            raise
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print("ERROR")
                    break
        finally:
            # clear all pending features
            for pending in self._rpc_id_to_pending.values():
                if not pending.future.done():
                    pending.future.set_exception(ValueError("Unknown"))

    async def on(self, event: str, handler):
        await self._loop_inited.wait()
        event_id = self._name_to_serv_id[event]
        self._event_id_to_handler[event_id] = handler

    def _add_pending(self, pending: PendingReply):
        self._rpc_id_to_pending[pending.rpc_id] = pending

    def _delete_pending(self, pending: PendingReply):
        if pending.rpc_id in self._rpc_id_to_pending:
            self._rpc_id_to_pending.pop(pending.rpc_id)

    async def subscribe(self, event: str):
        await self._send_simple(core_io.SocketMsgType.Subscribe, event,
                                time.time_ns(), [])

    def _rpc_timeout_callback(self, pending: PendingReply, key: str):
        if not pending.future.done():
            pending.future.set_exception(
                TimeoutError(f"rpc {key} with id {pending.rpc_id} timeout."))
            self._delete_pending(pending)

    async def remote_json_call(self,
                               key: str,
                               *args,
                               timeout: Optional[int] = None,
                               **kwargs):
        await self._loop_inited.wait()
        serv_id = self._name_to_serv_id[key]
        data = [args, kwargs]
        rpc_id = time.time_ns()
        pending = PendingReply(rpc_id, serv_id, False)
        self._add_pending(pending)
        if timeout is not None:
            assert timeout > 0
            loop = asyncio.get_running_loop()
            loop.call_later(timeout, self._rpc_timeout_callback, pending, key)
        asyncio.create_task(
            self._send_simple(core_io.SocketMsgType.RPC, key, rpc_id, data))
        return await pending.future

    async def _send_simple(self, msg_type: core_io.SocketMsgType, serv: str,
                           rpc_id: int, data):
        await self._loop_inited.wait()
        serv_id = 0
        if serv:
            serv_id = self._name_to_serv_id[serv]
        enc = core_io.SocketMessageEncoder(data)
        header = wsdef_pb2.Header(service_id=serv_id, rpc_id=rpc_id)
        chunks = list(
            enc.get_message_chunks(msg_type, header, self._msg_max_size))
        await self._ws.send_bytes(chunks[0])


async def main():
    url = "http://localhost:51052/api/rpc"
    wsurl = "http://localhost:51052/api/ws"
    import numpy as np
    async with aiohttp.ClientSession() as session:

        print(await http_remote_call(session, url, "Test.echo", 5))
        async with session.ws_connect(wsurl) as ws:
            client = WebsocketClient(ws)
            print("??")

            await client.subscribe("Test.event")
            await client.on("Test.event", lambda x: print(x))
            await client.subscribe(
                "tensorpc.services.vis::VisService.new_vis_message")

            def onevent(x):
                hasnan = np.isnan(x[0]["base"]["data"]).any()
                print("hasnan", hasnan, x[0]["base"]["data"][0])

            await client.on(
                "tensorpc.services.vis::VisService.new_vis_message", onevent)
            print("?")
            print(client._name_to_serv_id)
            res = await client.remote_json_call("Test.echo", 5)
            # res2 = await client.remote_json_call("Test.large_return")
            # print(res, np.linalg.norm(res2 - np.arange(1000000)))
            await asyncio.sleep(5)
    # with RemoteManager("localhost:51051") as robj:
    #     robj.shutdown()


if __name__ == "__main__":
    # asyncio.run(main())
    data = http_remote_call_request(
        "https://localhost:51052/api/rpc",
        "tensorpc.services.collection:Simple.echo", 5)
    print(data, type(data))
