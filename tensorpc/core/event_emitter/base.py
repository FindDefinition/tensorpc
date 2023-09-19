"""
based on pyee implementation: https://github.com/jfhbrook/pyee
type-safe version with limited feature.
"""
from asyncio import iscoroutine
from collections import OrderedDict
import dataclasses
import inspect
from threading import Lock
from typing import (Any, Callable, Dict, Generic, List, Mapping, Optional, Set,
                    Tuple, TypeVar, Union, cast)

from typing_extensions import TypeVarTuple, Unpack


class PyeeException(Exception):
    """An exception internal to pyee."""


KT = TypeVar(name="KT")
VTs = TypeVarTuple(name="VTs")
Handler = TypeVar(name="Handler", bound=Callable)


@dataclasses.dataclass
class ExceptionParam:
    exc: Exception


class EventEmitter(Generic[KT, Unpack[VTs]]):
    """The base event emitter class. All other event emitters inherit from
    this class.

    Most events are registered with an emitter via the `on` and `once`
    methods, and fired with the `emit` method. 

    All callbacks are handled in a synchronous, blocking manner. As in node.js,
    raised exceptions are not automatically handled for you---you must catch
    your own exceptions, and treat them accordingly.
    """

    def __init__(self) -> None:
        self._events: Dict[
            KT,
            "OrderedDict[Callable[[Unpack[VTs]], Any], Callable[[Unpack[VTs]], Any]]",
        ] = dict()
        self._exc_event_handlers: OrderedDict[Callable[[ExceptionParam], Any],
                                              Callable[[ExceptionParam],
                                                       Any]] = OrderedDict()
        self._lock: Lock = Lock()

    def __getstate__(self) -> Mapping[str, Any]:
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = Lock()

    def on(
        self,
        event: KT,
        f: Optional[Callable[[Unpack[VTs]], Any]] = None
    ) -> Union[Callable[[Unpack[VTs]], Any], Callable[
        [Callable[[Unpack[VTs]], Any]], Callable[[Unpack[VTs]], Any]]]:
        """Registers the function `f` to the event name `event`, if provided.

        If `f` isn't provided, this method calls `EventEmitter#listens_to`, and
        otherwise calls `EventEmitter#add_listener`. In other words, you may either
        use it as a decorator:

        ```py
        @ee.on('data')
        def data_handler(data):
            print(data)
        ```

        Or directly:

        ```py
        ee.on('data', data_handler)
        ```

        In both the decorated and undecorated forms, the event handler is
        returned. The upshot of this is that you can call decorated handlers
        directly, as well as use them in remove_listener calls.

        Note that this method's return type is a union type. If you are using
        mypy or pyright, you will probably want to use either
        `EventEmitter#listens_to` or `EventEmitter#add_listener`.
        """
        if f is None:
            return self.listens_to(event)
        else:
            return self.add_listener(event, f)

    def listens_to(
        self, event: KT
    ) -> Callable[[Callable[[Unpack[VTs]], Any]], Callable[[Unpack[VTs]],
                                                           Any]]:
        """Returns a decorator which will register the decorated function to
        the event name `event`:

        ```py
        @ee.listens_to("event")
        def data_handler(data):
            print(data)
        ```

        By only supporting the decorator use case, this method has improved
        type safety over `EventEmitter#on`.
        """

        def on(f: Callable[[Unpack[VTs]],
                           Any]) -> Callable[[Unpack[VTs]], Any]:
            self._add_event_handler(event, f, f)
            return f

        return on

    def add_listener(
            self, event: KT, f: Callable[[Unpack[VTs]],
                                         Any]) -> Callable[[Unpack[VTs]], Any]:
        """Register the function `f` to the event name `event`:

        ```
        def data_handler(data):
            print(data)

        h = ee.add_listener("event", data_handler)
        ```

        By not supporting the decorator use case, this method has improved
        type safety over `EventEmitter#on`.
        """
        self._add_event_handler(event, f, f)
        return f

    def listens_to_exception(
        self
    ) -> Callable[[Callable[[ExceptionParam], Any]], Callable[[ExceptionParam],
                                                              Any]]:
        """Returns a decorator which will register the decorated function to
        the event name `event`:

        ```py
        @ee.listens_to("event")
        def data_handler(data):
            print(data)
        ```

        By only supporting the decorator use case, this method has improved
        type safety over `EventEmitter#on`.
        """

        def on(
            f: Callable[[ExceptionParam],
                        Any]) -> Callable[[ExceptionParam], Any]:
            self._add_exception_event_handler(f, f)
            return f

        return on

    def add_exception_listener(
            self, f: Callable[[ExceptionParam],
                              Any]) -> Callable[[ExceptionParam], Any]:
        """Register the function `f` to the event name `event`:

        ```
        def data_handler(data):
            print(data)

        h = ee.add_listener("event", data_handler)
        ```

        By not supporting the decorator use case, this method has improved
        type safety over `EventEmitter#on`.
        """
        self._add_exception_event_handler(f, f)
        return f

    def _add_event_handler(self, event: KT, k: Callable[[Unpack[VTs]], Any],
                           v: Callable[[Unpack[VTs]], Any]):
        # Fire 'new_listener' *before* adding the new listener!
        # self.emit("new_listener", event, k)

        # Add the necessary function
        # Note that k and v are the same for `on` handlers, but
        # different for `once` handlers, where v is a wrapped version
        # of k which removes itself before calling k
        with self._lock:
            if event not in self._events:
                self._events[event] = OrderedDict()
            self._events[event][k] = v

    def _add_exception_event_handler(self, k: Callable[[ExceptionParam], Any],
                                     v: Callable[[ExceptionParam], Any]):
        # Add the necessary function
        # Note that k and v are the same for `on` handlers, but
        # different for `once` handlers, where v is a wrapped version
        # of k which removes itself before calling k
        with self._lock:
            self._exc_event_handlers[k] = v

    def _emit_run(
        self,
        f: Callable[[Unpack[VTs]], Any],
        args: Tuple[Unpack[VTs]],
    ) -> None:
        f(*args)

    async def _emit_run_async(
        self,
        f: Callable[[Unpack[VTs]], Any],
        args: Tuple[Unpack[VTs]],
    ) -> None:
        coro = f(*args)
        if inspect.iscoroutine(coro):
            return await coro 
        else:
            return coro

    def event_names(self) -> Set[KT]:
        """Get a set of events that this emitter is listening to."""
        return set(self._events.keys())

    def _emit_handle_potential_error(self, event: KT, error: Any) -> None:
        if event == "error":
            if isinstance(error, Exception):
                raise error
            else:
                raise PyeeException(
                    f"Uncaught, unspecified 'error' event: {error}")

    def _call_handlers(
        self,
        event: KT,
        *args: Unpack[VTs],
    ) -> bool:
        handled = False

        with self._lock:
            funcs = list(self._events.get(event, OrderedDict()).values())
        for f in funcs:
            self._emit_run(f, args)
            handled = True

        return handled

    async def _call_handlers_async(
        self,
        event: KT,
        *args: Unpack[VTs],
    ) -> bool:
        handled = False

        with self._lock:
            funcs = list(self._events.get(event, OrderedDict()).values())
        for f in funcs:
            await self._emit_run_async(f, args)
            handled = True

        return handled

    def _emit_exc_run(
        self,
        f: Callable[[ExceptionParam], Any],
        arg: ExceptionParam,
    ) -> None:
        f(arg)

    def _call_exc_handlers(
        self,
        args: ExceptionParam,
    ) -> bool:
        handled = False

        with self._lock:
            funcs = list(self._exc_event_handlers.values())
        for f in funcs:
            self._emit_exc_run(f, args)
            handled = True

        return handled

    def emit(
        self,
        event: KT,
        *args: Unpack[VTs],
    ) -> bool:
        """Emit `event`, passing `*args` and `**kwargs` to each attached
        function. Returns `True` if any functions are attached to `event`;
        otherwise returns `False`.

        Example:

        ```py
        ee.emit('data', '00101001')
        ```

        Assuming `data` is an attached function, this will call
        `data('00101001')'`.
        """
        handled = self._call_handlers(event, *args)

        if not handled:
            self._emit_handle_potential_error(event, args[0] if args else None)

        return handled

    async def emit_async(
        self,
        event: KT,
        *args: Unpack[VTs],
    ) -> bool:
        """Emit `event` , passing `*args` and `**kwargs` to each attached
        function. Returns `True` if any functions are attached to `event`;
        otherwise returns `False`.

        this function run coroutines in direct way instead of wait them in futures.

        Example:

        ```py
        await ee.emit_async('data', '00101001')
        ```

        Assuming `data` is an attached function, this will call
        `data('00101001')'`.
        """
        handled = await self._call_handlers_async(event, *args)

        if not handled:
            self._emit_handle_potential_error(event, args[0] if args else None)

        return handled

    def emit_exception(
        self,
        args: ExceptionParam,
    ) -> bool:
        """Emit `event`, passing `*args` and `**kwargs` to each attached
        function. Returns `True` if any functions are attached to `event`;
        otherwise returns `False`.

        Example:

        ```py
        ee.emit('data', '00101001')
        ```

        Assuming `data` is an attached function, this will call
        `data('00101001')'`.
        """
        handled = self._call_exc_handlers(args)
        return handled

    def once(
        self,
        event: KT,
        f: Optional[Callable[[Unpack[VTs]], Any]] = None,
    ) -> Callable:
        """The same as `ee.on`, except that the listener is automatically
        removed after being called.
        """

        def _wrapper(
                f: Callable[[Unpack[VTs]],
                            Any]) -> Callable[[Unpack[VTs]], Any]:

            def g(*args: Unpack[VTs], ) -> Any:
                with self._lock:
                    # Check that the event wasn't removed already right
                    # before the lock
                    if event in self._events and f in self._events[event]:
                        self._remove_listener(event, f)
                    else:
                        return None
                # f may return a coroutine, so we need to return that
                # result here so that emit can schedule it
                return f(*args)

            self._add_event_handler(event, f, g)
            return f

        if f is None:
            return _wrapper
        else:
            return _wrapper(f)

    def _remove_listener(self, event: KT, f: Callable[[Unpack[VTs]],
                                                      Any]) -> None:
        """Naked unprotected removal."""
        self._events[event].pop(f)
        if not len(self._events[event]):
            del self._events[event]

    def remove_listener(self, event: KT, f: Callable[[Unpack[VTs]],
                                                     Any]) -> None:
        """Removes the function `f` from `event`."""
        with self._lock:
            self._remove_listener(event, f)

    def remove_all_listeners(self, event: Optional[KT] = None) -> None:
        """Remove all listeners attached to `event`.
        If `event` is `None`, remove all listeners on all events.
        """
        with self._lock:
            if event is not None:
                self._events[event] = OrderedDict()
            else:
                self._events = dict()

    def listeners(self, event: KT) -> List[Callable[[Unpack[VTs]], Any]]:
        """Returns a list of all listeners registered to the `event`."""
        return list(self._events.get(event, OrderedDict()).keys())


