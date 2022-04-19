# Copyright 2022 Yan Yan
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
from typing import Tuple
import numpy as np
import pytest
import aiohttp 
from distflow.core import asyncclient
from distflow.utils.wait_tools import get_free_loopback_tcp_port
from distflow import PACKAGE_ROOT
from distflow.services.for_test import Service2
from distflow.core.httpclient import http_remote_call
import pytest_asyncio

@pytest_asyncio.fixture
async def sess_url():
    with get_free_loopback_tcp_port() as port:
        # request a free port
        pass
    with get_free_loopback_tcp_port() as port2:
        # request a free port
        pass

    serv_def = PACKAGE_ROOT / "services/serv_def.yaml"
    proc = subprocess.Popen(
        f"python -m distflow.serve --port {port} --http_port={port2} --serv_def_file {serv_def}",
        shell=True)
    url = f"http://localhost:{port2}/api/jsonrpc"
    try:
        async with asyncclient.AsyncRemoteManager(
                "localhost:{}".format(port)) as robj:
            await robj.wait_for_remote_ready()
        async with aiohttp.ClientSession() as session:
            yield session, url
    finally:
        async with asyncclient.AsyncRemoteManager(
                "localhost:{}".format(port)) as robj:
            await robj.shutdown()
        proc.wait()

@contextlib.asynccontextmanager
async def sess_url_local():
    with get_free_loopback_tcp_port() as port:
        # request a free port
        pass
    with get_free_loopback_tcp_port() as port2:
        # request a free port
        pass

    serv_def = PACKAGE_ROOT / "services/serv_def.yaml"
    proc = subprocess.Popen(
        f"python -m distflow.serve --port {port} --http_port={port2} --serv_def_file {serv_def}",
        shell=True)
    url = f"http://localhost:{port2}/api/jsonrpc"
    try:
        async with asyncclient.AsyncRemoteManager(
                "localhost:{}".format(port)) as robj:
            await robj.wait_for_remote_ready()
        async with aiohttp.ClientSession() as session:
            yield session, url
    finally:
        async with asyncclient.AsyncRemoteManager(
                "localhost:{}".format(port)) as robj:
            await robj.shutdown()
        proc.wait()


@pytest.mark.asyncio
async def test_remote_call(sess_url: Tuple[aiohttp.ClientSession, str]):
    sess = sess_url[0]
    url = sess_url[1]
    datas_a = np.random.uniform(size=(30))
    datas_b = np.random.uniform(size=(30))
    expected = datas_a + datas_b
    res = await http_remote_call(sess, url, "Test2.add", 1, 2)
    assert res == 3
    res = await http_remote_call(sess, url, "Test3Async.add", datas_a, datas_b)
    assert np.allclose(res, expected)

async def main_async():
    async with sess_url_local() as robj:
        await test_remote_call(robj)

if __name__ == "__main__":
    asyncio.run(main_async())