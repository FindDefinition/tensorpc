import contextlib
import socket
import time
import asyncio
from typing import Coroutine, List
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

def get_primary_ip():
    # https://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        # doesn't even have to be reachable
        s.connect(('10.254.254.254', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

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

async def _period_loop(duration: float, shutdown_ev: asyncio.Event, user_callback: Callable[[], Coroutine[None, None, Any]], is_pre: bool = True):
    shutdown_task = asyncio.create_task(shutdown_ev.wait())
    sleep_task = asyncio.create_task(asyncio.sleep(duration))
    wait_tasks = [shutdown_task, sleep_task]
    while True:
        if is_pre:
            await user_callback()
        done, pending = await asyncio.wait(
            wait_tasks, return_when=asyncio.FIRST_COMPLETED)
        if shutdown_task in done:
            break
        if sleep_task in done:
            wait_tasks.remove(sleep_task)
            sleep_task = asyncio.create_task(
                asyncio.sleep(duration))
            wait_tasks.append(sleep_task)
            if not is_pre:
                await user_callback()

class PeriodicTask:
    def __init__(self, duration: float, user_callback: Callable[[], Coroutine[None, None, Any]], is_pre: bool = True):
        self.duration = duration
        self.user_callback = user_callback
        self.is_pre = is_pre
        self.shutdown_ev = asyncio.Event()
        self.task = asyncio.create_task(
            _period_loop(duration, self.shutdown_ev, user_callback, is_pre))

    async def close(self):
        self.shutdown_ev.set()
        await self.task
