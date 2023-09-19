from .base import EventEmitter, KT, VTs, ExceptionParam
from typing import (Any, Callable, Dict, Generic, List, Mapping, Optional, Set,
                    Tuple, TypeVar, Union, cast)
from asyncio import AbstractEventLoop, ensure_future, Future, iscoroutine

from typing_extensions import TypeVarTuple, Unpack

class AsyncIOEventEmitter(EventEmitter[KT, Unpack[VTs]]):
    """An event emitter class which can run asyncio coroutines in addition to
    synchronous blocking functions. For example:

    ```py
    @ee.on('event')
    async def async_handler(*args, **kwargs):
        await returns_a_future()
    ```

    On emit, the event emitter  will automatically schedule the coroutine using
    `asyncio.ensure_future` and the configured event loop (defaults to
    `asyncio.get_event_loop()`).

    Unlike the case with the EventEmitter, all exceptions raised by
    event handlers are automatically emitted on the `error` event. This is
    important for asyncio coroutines specifically but is also handled for
    synchronous functions for consistency.

    When `loop` is specified, the supplied event loop will be used when
    scheduling work with `ensure_future`. Otherwise, the default asyncio
    event loop is used.

    For asyncio coroutine event handlers, calling emit is non-blocking.
    In other words, you do not have to await any results from emit, and the
    coroutine is scheduled in a fire-and-forget fashion.
    """

    def __init__(self, loop: Optional[AbstractEventLoop] = None):
        super(AsyncIOEventEmitter, self).__init__()
        self._loop: Optional[AbstractEventLoop] = loop
        self._waiting: Set[Future] = set()

    def _emit_exc_run(
        self,
        f: Callable[[ExceptionParam], Any],
        arg: ExceptionParam,
    ) -> None:
        coro = f(arg)
        if iscoroutine(coro):
            if self._loop:
                # ensure_future is *extremely* cranky about the types here,
                # but this is relatively well-tested and I think the types
                # are more strict than they should be
                fut: Any = ensure_future(cast(Any, coro), loop=self._loop)
            else:
                fut = ensure_future(cast(Any, coro))

    def _emit_run(
        self,
        f: Callable,
        args: Tuple[Unpack[VTs]],
    ):
        try:
            coro: Any = f(*args)
        except Exception as exc:
            self.emit_exception(ExceptionParam(exc))
        else:
            if iscoroutine(coro):
                if self._loop:
                    # ensure_future is *extremely* cranky about the types here,
                    # but this is relatively well-tested and I think the types
                    # are more strict than they should be
                    fut: Any = ensure_future(cast(Any, coro), loop=self._loop)
                else:
                    fut = ensure_future(cast(Any, coro))

            elif isinstance(coro, Future):
                fut = cast(Any, coro)
            else:
                return

            def callback(f):
                self._waiting.remove(f)

                if f.cancelled():
                    return

                exc: Exception = f.exception()
                if exc:
                    self.emit_exception(ExceptionParam(exc))

            fut.add_done_callback(callback)
            self._waiting.add(fut)

    async def _emit_run_async(
        self,
        f: Callable,
        args: Tuple[Unpack[VTs]],
    ):
        try:
            coro: Any = f(*args)
        except Exception as exc:
            self.emit_exception(ExceptionParam(exc))
        else:
            if iscoroutine(coro):
                await coro
            else:
                return
