# Copyright 2024 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import subprocess
import threading
import time
from pathlib import Path
import asyncio
import numpy as np
import pytest

from tensorpc.core import asyncclient
from tensorpc.utils.wait_tools import get_free_loopback_tcp_port
from tensorpc import PACKAGE_ROOT
from tensorpc.services.for_test import Service2
import pytest_asyncio


@pytest_asyncio.fixture
async def server_client():
    with get_free_loopback_tcp_port() as port:
        # request a free port
        pass
    serv_def = PACKAGE_ROOT / "services/serv_def.yaml"
    proc = subprocess.Popen(
        f"python -m tensorpc.serve --port {port} --serv_def_file {serv_def}",
        shell=True)
    try:
        async with asyncclient.AsyncRemoteManager(
                "localhost:{}".format(port)) as robj:
            await robj.wait_for_remote_ready(10, 30)
            yield robj
    finally:
        async with asyncclient.AsyncRemoteManager(
                "localhost:{}".format(port)) as robj:
            await robj.shutdown()
        proc.wait()


@contextlib.asynccontextmanager
async def server_client_local():
    with get_free_loopback_tcp_port() as port:
        # request a free port
        pass
    serv_def = PACKAGE_ROOT / "services/serv_def.yaml"
    proc = subprocess.Popen(
        f"python -m tensorpc.serve --port {port} --serv_def_file {serv_def}",
        shell=True)
    try:
        async with asyncclient.AsyncRemoteManager(
                "localhost:{}".format(port)) as robj:
            await robj.wait_for_remote_ready(10, 30)
            yield robj
    finally:
        async with asyncclient.AsyncRemoteManager(
                "localhost:{}".format(port)) as robj:
            await robj.shutdown()
        print(proc.communicate())


def local_func(a, b):
    return a + b


def func_np(a, b, e=False):
    if e:
        raise ValueError("error")
    from codeai.distributed import prim
    assert prim.get_server_context() is not None
    return np.add(a, 2 * b)


def gen_func(a):
    for i in range(10):
        yield a + i


def func_np_with_stdout(a, b):
    print("helloworld")
    return np.add(a, 2 * b)


def code():
    return ("import numpy as np"
            "def remote_func(a, b):"
            "  return np.add(a, b)")


@pytest.mark.asyncio
async def test_query_meta(server_client: asyncclient.AsyncRemoteManager):
    robj = server_client
    meta = await robj.query_service_meta("Test3Async.add")
    print(meta)
    assert meta["is_gen"] == False
    assert len(meta["args"]) == 2


@pytest.mark.asyncio
async def test_remote_call(server_client: asyncclient.AsyncRemoteManager):
    robj = server_client
    datas_a = np.random.uniform(size=(30))
    datas_b = np.random.uniform(size=(30))
    expected = datas_a + datas_b
    res = await robj.remote_call("Test3.add", 1, 2)
    assert res == 3
    res = await robj.remote_call("Test3.add", 1, 2)
    assert res == 3

    res = await robj.remote_call("Test3Async.add", datas_a, datas_b)
    assert np.allclose(res, expected)
    res = await robj.remote_json_call("Test3Async.add", b=datas_b, a=datas_a)
    assert np.allclose(res, expected)
    res = await robj.chunked_remote_call("Test3Async.add",
                                         b=datas_b,
                                         a=datas_a)
    assert np.allclose(res, expected)

    with pytest.raises(asyncclient.RemoteException):
        await robj.remote_call("Test3Async.add", b=datas_b, a=datas_a, e=True)

    def generator():
        for a, b in zip(datas_a, datas_b):
            yield [a], {"b": b}  # args, kwargs, args must be a list

    res = []
    async for r in robj.stream_remote_call("Test3Async.add", generator()):
        res.append(r)
    assert np.allclose(np.array(res), expected)


@pytest.mark.asyncio
async def test_remote_generator(server_client: asyncclient.AsyncRemoteManager):
    robj = server_client
    res = []
    async for x in robj.remote_generator("Test3Async.gen_func", a=4):
        res.append(x)
    expected = [4 + i for i in range(10)]
    assert np.allclose(res, expected)
    res = []
    async for x in robj.chunked_remote_generator("Test3Async.gen_func", a=4):
        res.append(x)
    expected = [4 + i for i in range(10)]
    assert np.allclose(res, expected)

def gen_2():
    for i in range(10):
        yield i

async def async_gen_2():
    for i in range(10):
        yield i
        await asyncio.sleep(0.01)


def gen_2_large():
    for i in range(2):
        yield np.zeros(2000000, dtype=np.float64) + i


@pytest.mark.asyncio
async def test_stream(server_client: asyncclient.AsyncRemoteManager):
    robj = server_client
    serv = Service2(1)
    res = await robj.client_stream("Test3Async.client_stream",
                                   gen_2(),
                                   a=4,
                                   b=6)
    res2 = await robj.chunked_client_stream("Test3Async.client_stream",
                                   async_gen_2(),
                                   a=4,
                                   b=6)

    expected = serv.client_stream(gen_2(), 4, 6)
    assert np.allclose(res, expected)
    assert np.allclose(res2, expected)

    res = []
    async for x in robj.bi_stream("Test3Async.bi_stream", gen_2(), a=4, b=6):
        res.append(x)
    expected = list(serv.bi_stream(gen_2(), 4, 6))
    assert np.allclose(res, expected)

    res = []
    async for x in robj.chunked_bi_stream("Test3Async.bi_stream", gen_2_large(), a=np.full([2000000], 4.0), b=6):
        res.append(x)
    expected = list(serv.bi_stream(gen_2_large(), np.full([2000000], 4.0), 6))
    assert np.allclose(res, expected)


async def main_async():
    async with server_client_local() as robj:
        await robj.health_check()
        await test_query_meta(robj)
        await test_remote_call(robj)
        await test_remote_generator(robj)
        await test_stream(robj)


if __name__ == "__main__":
    asyncio.run(main_async())
