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

import numpy as np
import pytest

from tensorpc.core import client
from tensorpc.utils.wait_tools import get_free_loopback_tcp_port
from tensorpc import PACKAGE_ROOT
from tensorpc.services.for_test import Service2


@pytest.fixture
def server_client():
    with get_free_loopback_tcp_port() as port:
        # request a free port
        pass
    serv_def = PACKAGE_ROOT / "services/serv_def.yaml"
    proc = subprocess.Popen(
        f"python -m tensorpc.serve_sync --port {port} --serv_def_file {serv_def}",
        shell=True)
    try:
        with client.simple_client("localhost:{}".format(port)) as robj:
            robj.wait_for_remote_ready(10, 30)
            yield robj
    finally:
        with client.simple_client("localhost:{}".format(port)) as robj:
            robj.shutdown()
        proc.wait()


@contextlib.contextmanager
def server_client_local():
    with get_free_loopback_tcp_port() as port:
        # request a free port
        pass
    serv_def = PACKAGE_ROOT / "services/serv_def.yaml"
    proc = subprocess.Popen(
        f"python -m tensorpc.serve_sync --port {port} --serv_def_file {serv_def}",
        shell=True)
    try:
        with client.simple_client("localhost:{}".format(port)) as robj:
            robj.wait_for_channel_ready(10)
            yield robj
    finally:
        with client.simple_client("localhost:{}".format(port)) as robj:
            robj.wait_for_channel_ready(10)
            robj.shutdown()
        proc.wait()


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


def test_query_meta(server_client: client.RemoteManager):
    robj = server_client
    meta = robj.query_service_meta("Test3.add")
    print(meta)
    assert meta["is_gen"] == False
    assert len(meta["args"]) == 2


def test_remote_call(server_client: client.RemoteManager):
    robj = server_client
    datas_a = np.random.uniform(size=(30))
    datas_b = np.random.uniform(size=(30))
    expected = datas_a + datas_b
    res = robj.remote_call("Test2.add", 1, 2)
    assert res == 3
    res = robj.remote_call("Test3.add", datas_a, datas_b)
    assert np.allclose(res, expected)
    res = robj.remote_json_call("Test3.add", b=datas_b, a=datas_a)
    assert np.allclose(res, expected)
    res = robj.chunked_remote_call("Test3.add", b=datas_b, a=datas_a)
    assert np.allclose(res, expected)

    with pytest.raises(client.RemoteException):
        robj.remote_call("Test3.add", b=datas_b, a=datas_a, e=True)

    def generator():
        for a, b in zip(datas_a, datas_b):
            yield [a], {"b": b}  # args, kwargs, args must be a list

    res = []
    for r in robj.stream_remote_call("Test3.add", generator()):
        res.append(r)
    assert np.allclose(np.array(res), expected)


def test_remote_generator(server_client):
    robj = server_client
    res = list(robj.remote_generator("Test3.gen_func", a=4))
    expected = [4 + i for i in range(10)]
    assert np.allclose(res, expected)


def gen_2():
    for i in range(10):
        yield i


def test_stream(server_client):
    robj = server_client
    serv = Service2(1)
    res = robj.client_stream("Test3.client_stream", gen_2(), a=4, b=6)
    expected = serv.client_stream(gen_2(), 4, 6)
    assert np.allclose(res, expected)
    res = list(robj.bi_stream("Test3.bi_stream", gen_2(), a=4, b=6))
    expected = list(serv.bi_stream(gen_2(), 4, 6))
    assert np.allclose(res, expected)


if __name__ == "__main__":
    with server_client_local() as robj:
        robj.health_check()
        test_query_meta(robj)
        test_remote_call(robj)
        test_remote_generator(robj)
        test_stream(robj)
