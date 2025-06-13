import inspect
from typing import (Any, Callable, Generic)

from typing_extensions import TypeVarTuple, Unpack

VTs = TypeVarTuple(name="VTs")

class SingleAsyncEventEmitter(Generic[Unpack[VTs]]):
    def __init__(self) -> None:
        self._handlers: dict[Callable[[Unpack[VTs]], Any], Callable[[Unpack[VTs]], Any]] = {}

    def on(self, handler: Callable[[Unpack[VTs]], Any]):
        self._handlers[handler] = handler
        return self

    def off(self, handler: Callable[[Unpack[VTs]], Any]):
        self._handlers.pop(handler)
        return self

    def is_empty(self) -> bool:
        return not self._handlers

    async def emit_async(self, *args: Unpack[VTs]) -> None:
        if self._handlers:
            for handler in self._handlers.values():
                coro = handler(*args)
                if inspect.iscoroutine(coro):
                    await coro
