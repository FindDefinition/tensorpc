import asyncio
from tensorpc.core import marker
import time
import numpy as np


class Service1:

    def add(self, a, b):
        return a + b

    def echo(self, x):
        return x

    def large_return(self):
        return np.arange(1000000)

    @marker.mark_websocket_event
    async def event(self):
        await asyncio.sleep(0.5)
        return time.time()


class Service2:

    def __init__(self, x) -> None:
        self.x = x

    def mul(self, a):
        return self.x * a

    def increment(self):
        self.x += 1

    def get_x(self):
        return self.x

    @staticmethod
    def add(a, b):
        return a + b

    def gen_func(self, a):
        for i in range(10):
            yield a + i

    @marker.mark_client_stream
    @staticmethod
    def client_stream(stream_iter, a, b):
        res = a
        for data in stream_iter:
            res += data - b
        return res

    @marker.mark_bidirectional_stream
    def bi_stream(self, stream_iter, a, b):
        res = a
        for data in stream_iter:
            res += data - b
            yield res


class Service2Async:

    def __init__(self, x) -> None:
        self.x = x

    @staticmethod
    async def add(a, b):
        return a + b

    @staticmethod
    async def sum(a):
        print(a.shape)
        return a.sum()

    def gen_func(self, a):
        for i in range(10):
            yield a + i

    def gen_func_async(self, a):
        for i in range(10):
            yield a + i

    @marker.mark_client_stream
    async def client_stream(self, stream_iter, a, b):
        res = a
        async for data in stream_iter:
            res += data - b
        return res

    @marker.mark_bidirectional_stream
    @staticmethod
    async def bi_stream(stream_iter, a, b):
        res = a
        async for data in stream_iter:
            res += data - b
            yield res
