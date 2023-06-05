
import contextlib
from functools import partial
from typing import (TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, Coroutine, Dict,
                    Iterable, List, Optional, Set, Tuple, Type, TypeVar, Union)

from typing_extensions import ParamSpec
import asyncio
P = ParamSpec('P')

T = TypeVar('T',)
T2 = TypeVar('T2')


# simple example
def func_forward(func: Callable[..., T], *args,
                       **kwargs) -> T:
    return func(*args, **kwargs)

def func_forward2(func: Callable[..., T],
                            *args,
                            **kwargs) -> T:
    res = func_forward(
        func_forward, func, *args, **kwargs)
    return res

def func_forward2_problem(func: Callable[..., T],
                            *args,
                            **kwargs) -> T:
    res = func_forward(
        partial(func_forward, func), *args, **kwargs)
    return res


def add(a: int, b: int) -> int:
    print(a + b)
    return a + b

def main():
    res = func_forward(add, 1, 2)
    res2 = func_forward2(add, 1, 2)

# real world example
class App:
    pass 

def get_app():
    return App()

@contextlib.contextmanager
def enter_app_conetxt(app: "App"):
    yield app 

def _run_func_with_app(app: "App", func: Callable[..., T], *args,
                       **kwargs) -> T:
    with enter_app_conetxt(app):
        return func(*args, **kwargs)

async def run_in_executor(func: Callable[..., T],
                            *args,
                            **kwargs) -> T:
    ft = asyncio.get_running_loop().run_in_executor(
        None, _run_func_with_app, get_app(), func, *args, **kwargs)
    return await ft

async def run_in_executor_problem(func: Callable[..., T],
                            *args,
                            **kwargs) -> T:
    ft = asyncio.get_running_loop().run_in_executor(
        None, partial(_run_func_with_app, get_app(), func) , *args, **kwargs)
    return await ft

if __name__ == '__main__':
    func_forward2_problem(add, 1, 2)
    asyncio.run(run_in_executor_problem(add, 1, 2))