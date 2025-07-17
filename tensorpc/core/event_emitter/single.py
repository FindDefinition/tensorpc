import inspect
from typing import (Any, Callable, Generic)
import dataclasses
from typing_extensions import TypeVarTuple, Unpack

VTs = TypeVarTuple(name="VTs")

@dataclasses.dataclass
class _Handler(Generic[Unpack[VTs]]):
    handler: Callable[[Unpack[VTs]], Any]
    once: bool = False

class SingleAsyncEventEmitter(Generic[Unpack[VTs]]):
    def __init__(self) -> None:
        self._handlers: dict[Callable[[Unpack[VTs]], Any], _Handler] = {}

    def on(self, handler: Callable[[Unpack[VTs]], Any]):
        self._handlers[handler] = _Handler(handler=handler, once=False)
        return self

    def off(self, handler: Callable[[Unpack[VTs]], Any]):
        self._handlers.pop(handler)
        return self

    def once(self, handler: Callable[[Unpack[VTs]], Any]):
        self._handlers[handler] = _Handler(handler=handler, once=True)
        return self

    def is_empty(self) -> bool:
        return not self._handlers

    async def emit_async(self, *args: Unpack[VTs]) -> None:
        if self._handlers:
            handler_to_remove: list[Any] = []
            for handler in self._handlers.values():
                coro = handler.handler(*args)
                if handler.once:
                    handler_to_remove.append(handler.handler)
                if inspect.iscoroutine(coro):
                    await coro
            for handler in handler_to_remove:
                self._handlers.pop(handler)