import threading
import dataclasses 
from typing import Callable, Generic, Optional, TypeVar
from typing_extensions import ParamSpec
import asyncio
import inspect

_T_func = TypeVar("_T_func", bound=Callable)

@dataclasses.dataclass
class HandlerItem(Generic[_T_func]):
    type: str
    handler: _T_func
    once: bool = False
    loop: Optional[asyncio.AbstractEventLoop] = None


class SimpleRPCHandler(Generic[_T_func]):
    """usually used for event that only have one handler and need return value.
    """
    def __init__(self) -> None:
        self._msg_single_handlers: dict[str, HandlerItem[_T_func]] = {}
        self._lock = threading.Lock()

    def on(self, event: str, f: _T_func, force_replace: bool = False, once: bool = False, loop: Optional[asyncio.AbstractEventLoop] = None):
        with self._lock:
            if not force_replace:
                assert event not in self._msg_single_handlers, f"event {event} already registered."
            self._msg_single_handlers[event] = HandlerItem(event, f, once, loop)

    def once(self, event: str, f: _T_func, force_replace: bool = False, loop: Optional[asyncio.AbstractEventLoop] = None):
        return self.on(event, f, force_replace, True, loop)

    def off(self, event: str):
        with self._lock:
            return self._msg_single_handlers.pop(event, None)

    def has_event_handler(self, event: str):
        with self._lock:
            return event in self._msg_single_handlers

    def get_event_handler(self, event: str):
        with self._lock:
            return self._msg_single_handlers[event]

    async def call_event(self, event: str, *args, **kwargs):
        with self._lock:
            handler_item = self._msg_single_handlers.get(event)
            assert handler_item is not None, f"event {event} not registered."
            if handler_item.once:
                self._msg_single_handlers.pop(event)

        res = handler_item.handler(*args, **kwargs)
        res_handler = res
        if inspect.iscoroutine(res):
            if handler_item.loop is not None:
                res_handler = asyncio.run_coroutine_threadsafe(res, handler_item.loop).result()
            else:
                res_handler = await res
        return res_handler

