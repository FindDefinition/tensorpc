import contextlib
import time
import asyncio
from typing import Coroutine, List
from async_timeout import timeout

import threading
from typing import Any, Callable, Optional, TypeVar, cast
import socket
import psutil

from contextlib import suppress

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


def get_all_ip_addresses(family: socket.AddressFamily):
    for interface, snics in psutil.net_if_addrs().items():
        for snic in snics:
            if snic.family == family:
                yield (interface, snic.address)


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


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b

async def _period_loop(duration: float, shutdown_ev: asyncio.Event, user_callback: Callable[[], Coroutine[None, None, Any]], is_pre: bool = True, align_ts: bool = False):
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
            cur_ts = time.time_ns()
            real_duration = duration
            if align_ts:
                # align to next duration
                duration_ns = int(duration * 1_000_000_000)
                next_ts = _div_up(cur_ts, duration_ns) * duration_ns
                real_duration = (next_ts - cur_ts) / 1_000_000_000
            sleep_task = asyncio.create_task(
                asyncio.sleep(real_duration))
            wait_tasks.append(sleep_task)
            if not is_pre:
                await user_callback()

class PeriodicTask:
    def __init__(self, duration: float, user_callback: Callable[[], Coroutine[None, None, Any]], is_pre: bool = True, align_ts: bool = False):
        self.duration = duration
        self.user_callback = user_callback
        self.is_pre = is_pre
        self.shutdown_ev = asyncio.Event()
        self.task = asyncio.create_task(
            _period_loop(duration, self.shutdown_ev, user_callback, is_pre, align_ts))

    async def close(self):
        self.shutdown_ev.set()
        await self.task

async def _cancel(task):
    # more info: https://stackoverflow.com/a/43810272/1113207
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task

async def wait_queue_until_event(handler: Callable[[Any], None],
                                 q: asyncio.Queue, ev: asyncio.Event):
    q_get_task = asyncio.create_task(q.get())
    shut_task = asyncio.create_task(ev.wait())
    wait_tasks: List[asyncio.Task] = [q_get_task, shut_task]
    while True:
        (done,
         pending) = await asyncio.wait(wait_tasks,
                                       return_when=asyncio.FIRST_COMPLETED)
        if ev.is_set():
            for task in pending:
                await _cancel(task)
            break
        if q_get_task in done:
            handler(q_get_task.result())
            q_get_task = asyncio.create_task(q.get())
        wait_tasks = [q_get_task, shut_task]

