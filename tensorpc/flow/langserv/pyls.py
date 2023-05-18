"""
use subprocess to check language server stdio.
references:
https://github.com/windmill-labs/windmill/blob/v1.101.1/lsp/pyls_launcher.py
https://github.com/python-lsp/python-lsp-jsonrpc/blob/v1.0.0/pylsp_jsonrpc/streams.py
"""


import asyncio
import json

import logging
import subprocess
import threading
import os
from typing import List

import aiohttp
from aiohttp import web
import ssl


log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

class AsyncJsonRpcStreamReader:
    def __init__(self, reader: asyncio.StreamReader):
        self._rfile = reader

    # def close(self):
    #     self._rfile.close()

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
            try:
                await message_consumer(json.loads(request_str.decode('utf-8')))
            except ValueError:
                log.exception("Failed to parse JSON message %s", request_str)
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
                raise ValueError("Invalid Content-Length header: {}".format(value)) from e

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
                content_length = len(body) if isinstance(body, bytes) else len(body.encode('utf-8'))
                response = (
                    "Content-Length: {}\r\n"
                    "Content-Type: application/vscode-jsonrpc; charset=utf8\r\n\r\n"
                    "{}".format(content_length, body)
                )
                self._wfile.write(response.encode('utf-8'))
                await self._wfile.drain()
            except Exception:  # pylint: disable=broad-except
                log.exception("Failed to write message to output file %s", message)


class LanguageServerHandler:
    def __init__(self, ls_cmd: List[str]) -> None:
        self.ls_cmd = ls_cmd

    async def handle_ls_open(self, request):
        print("NEW CONN", request)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        aproc = await asyncio.create_subprocess_exec(*self.ls_cmd, env=os.environ,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE)
        assert aproc.stdout is not None 
        assert aproc.stdin is not None 

        # proc = subprocess.Popen(["python", "-m", "tensorpc.cli.pyright_launch", "--stdio"], env=os.environ,
        #     stdin=subprocess.PIPE,
        #     stdout=subprocess.PIPE)
        # Create a writer that formats json messages with the correct LSP headers
        writer = AsyncJsonRpcStreamWriter(aproc.stdin)
        reader = AsyncJsonRpcStreamReader(aproc.stdout)
        async def cosumer(msg):
            # print("OUTPUT", type(msg), msg)
            await ws.send_json(msg)
        task = asyncio.create_task(reader.listen(cosumer))
        # Create a reader for consuming stdout of the language server. We need to
        # consume this in another thread
        async for ws_msg in ws:
            if ws_msg.type == aiohttp.WSMsgType.TEXT:
                # print("INPUT", ws_msg.json())  
                await writer.write(ws_msg.json())
        
            elif ws_msg.type == aiohttp.WSMsgType.ERROR:
                print("ERROR", ws_msg)
                print("ERROR", ws_msg.data)
            else:
                raise NotImplementedError
        return ws

async def serve_app(app,
                    port,
                    async_shutdown_ev: asyncio.Event,
                    url=None,
                    ssl_context=None):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=url, port=port, ssl_context=ssl_context)
    await site.start()
    await async_shutdown_ev.wait()
    await runner.cleanup()

async def serve_ls(port: int,
                   ls_cmd: List[str],
                    ssl_key_path: str = "",
                    ssl_crt_path: str = ""):
    async_shutdown_ev = asyncio.Event()
    app = web.Application()
    handler = LanguageServerHandler(ls_cmd)
    app.router.add_get("", handler.handle_ls_open)
    ssl_context = None
    if ssl_key_path != "" and ssl_key_path != "":
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(ssl_crt_path, ssl_key_path)
    return await serve_app(app,
                    port,
                    async_shutdown_ev,
                    ssl_context=ssl_context)

