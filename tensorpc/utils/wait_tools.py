import contextlib
import socket
import time
import asyncio
from typing import List
from async_timeout import timeout

import threading
from typing import Any, Callable, Optional, TypeVar, cast


def wait_until(func,
               max_retries: int = 200,
               check_interval: float = 1,
               check_func=None):
    while max_retries > 0:
        res = func()
        if res:
            return res
        if check_func is not None:
            check_func()
        time.sleep(check_interval)
        max_retries -= 1
    raise TimeoutError


async def wait_until_async(func,
                           max_retries: int = 200,
                           check_interval: float = 1,
                           check_func=None):
    while max_retries > 0:
        res = await func()
        if res:
            return res
        if check_func is not None:
            check_func()
        await asyncio.sleep(check_interval)
        max_retries -= 1
    raise TimeoutError


async def wait_blocking_async(blocking_func,
                              max_retries: int = 200,
                              check_interval: float = 1,
                              check_func=None):
    while max_retries > 0:
        async with timeout(check_interval) as status:
            await blocking_func()
        if not status.expired:
            return
        max_retries -= 1
    raise TimeoutError


def wait_until_noexcept_call(func,
                             *args,
                             max_retries: int = 200,
                             check_interval: float = 1,
                             **kw):
    while max_retries > 0:
        try:
            return func(*args, **kw)
        except Exception as e:
            print("func fail with Exception {}, wait...".format(e))
        time.sleep(check_interval)
        max_retries -= 1
    raise TimeoutError


def wait_until_call(func, max_retries=200, check_interval=1):
    while max_retries > 0:
        is_valid, res = func()
        if is_valid:
            return res
        time.sleep(check_interval)
        max_retries -= 1
    raise TimeoutError


@contextlib.contextmanager
def get_free_loopback_tcp_port():
    if socket.has_ipv6:
        tcp_socket = socket.socket(socket.AF_INET6)
    else:
        tcp_socket = socket.socket(socket.AF_INET)
    tcp_socket.bind(('', 0))
    address_tuple = tcp_socket.getsockname()
    try:
        yield address_tuple[1]
    finally:
        tcp_socket.close()


def get_free_ports(count: int):
    ports: List[int] = []
    for i in range(count):
        with get_free_loopback_tcp_port() as port:
            ports.append(port)
    return ports

class Debouncer:
    def __init__(self, f: Callable[..., Any], interval: float):
        self.f = f
        self.interval = interval
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def __call__(self, *args, **kwargs) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.interval, self.f, args, kwargs)
            self._timer.start()


VoidFunction = TypeVar("VoidFunction", bound=Callable[..., None])


def debounce(interval: float):
    """
    Wait `interval` seconds before calling `f`, and cancel if called again.
    The decorated function will return None immediately,
    ignoring the delayed return value of `f`.
    """

    def decorator(f: VoidFunction) -> VoidFunction:
        if interval <= 0:
            return f
        return cast(VoidFunction, Debouncer(f, interval))

    return decorator