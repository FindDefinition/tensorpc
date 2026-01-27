"""
use subprocess to check language server stdio.
references:
https://github.com/windmill-labs/windmill/blob/v1.101.1/lsp/pyls_launcher.py
https://github.com/python-lsp/python-lsp-jsonrpc/blob/v1.0.0/pylsp_jsonrpc/streams.py
"""

import asyncio
import json

import logging
import signal
import subprocess
import threading
import os
from typing import List, Optional

import aiohttp
from aiohttp import web
import ssl

from tensorpc.core.asynctools import cancel_task
from ..logger import LOGGER

def _patch_uri(uri: str, prefix: str):
    assert uri.startswith("file://")
    return "file://" + prefix + uri[len("file://"):]

class AsyncJsonRpcStreamReader:

    def __init__(self, reader: asyncio.StreamReader, need_prefix_dict: dict[str, str], prefix: Optional[str] = None):
        self._rfile = reader
        self._prefix = prefix
        self._need_prefix_dict = need_prefix_dict

    async def listen(self, message_consumer):
        """Blocking call to listen for messages on the rfile.

        Args:
            message_consumer (fn): function that is passed each message as it is read off the socket.
        """
        async for line in self._rfile:
            content_length = self._content_length(line)
            while line and line.strip():
                line = await self._rfile.readline()
            if line == b"" or content_length is None:
                break
            request_str = await self._rfile.readexactly(content_length)
            data = json.loads(request_str.decode('utf-8'))

            try:
                if self._prefix is not None:
                    if "params" in data:
                        params = data["params"]
                        if "uri" in params and params['uri'] in self._need_prefix_dict:
                            params["uri"] = self._need_prefix_dict[params['uri']]
                    if "result" in data and isinstance(data["result"], list):
                        for item in data["result"]:
                            # if isinstance(item, dict) and "uri" in item and item["uri"] in self._need_prefix_dict:
                            #     item["uri"] = self._need_prefix_dict[item["uri"]]
                            if isinstance(item, dict) and "uri" in item:
                                item["uri"] = _patch_uri(item["uri"], self._prefix)
                # print("[JSONRPC OUT]", data)

                await message_consumer(data)
            except ValueError:
                LOGGER.exception("Failed to parse JSON message %s", request_str)
                continue
            except:
                LOGGER.exception("Failed to process JSON message %s", data)
                continue

    @staticmethod
    def _content_length(line):
        """Extract the content length from an input line."""
        if line.startswith(b'Content-Length: '):
            _, value = line.split(b'Content-Length: ')
            value = value.strip()
            try:
                return int(value)
            except ValueError as e:
                raise ValueError(
                    "Invalid Content-Length header: {}".format(value)) from e

        return None


class AsyncJsonRpcStreamWriter:

    def __init__(self, wfile: asyncio.StreamWriter, **json_dumps_args):
        self._wfile = wfile
        self._wfile_lock = asyncio.Lock()
        self._json_dumps_args = json_dumps_args

    async def close(self):
        async with self._wfile_lock:
            self._wfile.close()

    async def write(self, message):
        async with self._wfile_lock:
            if self._wfile.is_closing():
                return
            try:
                body = json.dumps(message, **self._json_dumps_args)

                # Ensure we get the byte length, not the character length
                content_length = len(body) if isinstance(body, bytes) else len(
                    body.encode('utf-8'))
                response = (
                    "Content-Length: {}\r\n"
                    "Content-Type: application/vscode-jsonrpc; charset=utf8\r\n\r\n"
                    "{}".format(content_length, body))
                self._wfile.write(response.encode('utf-8'))
                await self._wfile.drain()
            except Exception:  # pylint: disable=broad-except
                LOGGER.exception("Failed to write message to output file %s",
                              message)


class LanguageServerHandler:
    def __init__(self):
        self._prefix: Optional[str] = None 

    def set_prefix(self, prefix: Optional[str]):
        self._prefix = prefix

    async def handle_ls_open(self, request):
        # graph_id = os.getenv("TENSORPC_FLOW_GRAPH_ID")
        # node_id = os.getenv("TENSORPC_FLOW_NODE_ID")
        prefix = self._prefix 
        # if graph_id is not None and node_id is not None:
        #     prefix = self._get_lsp_prefix(graph_id, node_id)
        # prefix = None
        ls_type = request.match_info.get('type')
        LOGGER.warning("New %s language server request", ls_type)
        assert ls_type in ["pyright"]
        if ls_type == "pyright":
            ls_cmd = ["python", "-m", "tensorpc.cli.pyright_launch"]
        else:
            raise NotImplementedError
        task: Optional[asyncio.Task] = None
        aproc: Optional[asyncio.subprocess.Process] = None
        try:
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            aproc = await asyncio.create_subprocess_exec(
                *ls_cmd,
                env=os.environ,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE)
            assert aproc.stdout is not None
            assert aproc.stdin is not None
            # Create a writer that formats json messages with the correct LSP headers
            need_prefix_dict: dict[str, str] = {}
            writer = AsyncJsonRpcStreamWriter(aproc.stdin)
            reader = AsyncJsonRpcStreamReader(aproc.stdout, need_prefix_dict, prefix=prefix)

            async def cosumer(msg):
                await ws.send_json(msg)
            task = asyncio.create_task(reader.listen(cosumer))
            # Create a reader for consuming stdout of the language server. We need to
            # consume this in another thread
            async for ws_msg in ws:
                if ws_msg.type == aiohttp.WSMsgType.TEXT:
                    ws_data = json.loads(ws_msg.data)
                    # print("[JSONRPC IN]", ws_data)

                    if prefix is not None:
                        # we patch path in frontend to support multiple-app-one-page,
                        # so we may need to remove prefix before sending to language server
                        if "params" in ws_data:
                            params = ws_data["params"]
                            for k, v in params.items():
                                if isinstance(v, dict) and "uri" in v and prefix in v["uri"]:
                                    path_remove_prefix = v["uri"].replace(prefix, "")
                                    need_prefix_dict[path_remove_prefix] = v["uri"]
                                    v["uri"] = path_remove_prefix

                    await writer.write(ws_data)
                elif ws_msg.type == aiohttp.WSMsgType.ERROR:
                    LOGGER.error(ws_msg)
                else:
                    raise NotImplementedError
        finally:
            if task is not None:
                await cancel_task(task)
            if aproc is not None:
                # TODO does this work on windows?
                aproc.send_signal(signal=signal.SIGINT)
                timeout = 5.0
                try:
                    await asyncio.wait_for(aproc.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    LOGGER.warning(
                        "Language server did not exit within %.1f seconds, terminating",
                        timeout)
                    aproc.terminate()
        return ws
