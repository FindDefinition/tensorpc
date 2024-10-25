import asyncio
import contextlib
import contextvars
import dataclasses
import enum
import inspect
import threading
import traceback
from functools import partial
from pathlib import Path
from typing import (TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable,
                    Coroutine, Dict, Generic, Iterable, List, Optional, Set,
                    Tuple, Type, TypeVar, Union)

from typing_extensions import (Concatenate, Literal, ParamSpec, Protocol, Self,
                               TypeAlias)

from tensorpc.core.moduleid import (get_qualname_of_type, is_lambda,
                                    is_valid_function)
from tensorpc.core.tree_id import UniqueTreeId

from tensorpc.flow.core.context import ALL_APP_CONTEXT_GETTERS
from tensorpc.flow.jsonlike import Undefined, BackendOnlyProp, undefined
from tensorpc.core.serviceunit import ObservedFunction, ObservedFunctionRegistry, ObservedFunctionRegistryProtocol
from tensorpc.flow.client import is_inside_app_session

if TYPE_CHECKING:
    from ..flowapp.app import App, EditableApp
    from .component import Component

CORO_NONE = Union[Coroutine[None, None, None], None]
CORO_ANY: TypeAlias = Union[Coroutine[None, None, Any], Any]

ValueType: TypeAlias = Union[int, float, str]
EventDataType: TypeAlias = Union[int, str]
NumberType: TypeAlias = Union[int, float]

SimpleEventType: TypeAlias = Union[Tuple[EventDataType, Any], Tuple[EventDataType, Any, Optional[str]]]
T = TypeVar("T")

T_comp = TypeVar("T_comp")


@dataclasses.dataclass
class Event:
    type: EventDataType
    data: Any
    # only used for template component such as table.
    # key indicates the id of template item.
    keys: Union[Undefined, List[str]] = undefined
    # for template control components.
    indexes: Union[Undefined, List[int]] = undefined


class EventHandler:

    def __init__(self, cb: Callable, simple_event: bool = True, converter: Optional[Callable[[Any], Any]] = None) -> None:
        self.cb = cb
        self.simple_event = simple_event
        self.converter = converter

    def run_event(self, event: Event) -> CORO_ANY:
        if self.simple_event:
            if self.converter is not None:
                return self.cb(self.converter(event.data))
            return self.cb(event.data)
        else:
            return self.cb(event)

    async def run_event_async(self, event: Event):
        if self.simple_event:
            if self.converter is not None:
                coro = self.cb(self.converter(event.data))
            coro = self.cb(event.data)
        else:
            coro = self.cb(event)
        if inspect.iscoroutine(coro):
            res = await coro
        else:
            res = coro
        return res

    def run_noarg_event(self, event: Event) -> CORO_ANY:
        if self.simple_event:
            return self.cb()
        else:
            return self.cb(event)


class EventHandlers:

    def __init__(self,
                 handlers: List[EventHandler],
                 stop_propagation: bool = False,
                 throttle: Optional[NumberType] = None,
                 debounce: Optional[NumberType] = None,
                 backend_only: bool = False,
                 simple_event: bool = True,
                 dont_send_to_backend: bool = False) -> None:

        self.handlers = handlers
        self.stop_propagation = stop_propagation
        self.debounce = debounce
        self.throttle = throttle
        self.backend_only = backend_only
        self.simple_event = simple_event
        self.dont_send_to_backend = dont_send_to_backend

    def to_dict(self):
        res: Dict[str, Any] = {
            "stopPropagation": self.stop_propagation,
        }
        if self.debounce is not None:
            res["debounce"] = self.debounce
        if self.throttle is not None:
            res["throttle"] = self.throttle
        if self.dont_send_to_backend:
            res["dontSendToBackend"] = True
        return res

    def get_bind_event_handlers(self, event: Event):
        return [
            partial(handler.run_event, event=event)
            for handler in self.handlers
        ]

    def get_bind_event_handlers_noarg(self, event: Event):
        return [
            partial(handler.run_noarg_event, event=event)
            for handler in self.handlers
        ]

    def remove_handler(self, handler: Callable):
        self.handlers = [h for h in self.handlers if h.cb != handler]
        return


class AppContext:

    def __init__(self, app: "App", is_remote: bool = False) -> None:
        self.app = app
        self.is_remote = is_remote

    def is_editable_app(self):
        return self.app._is_editable_app()

class EventHandlingContext:

    def __init__(self, uid: UniqueTreeId) -> None:
        self.comp_uid = uid
        self.delayed_callbacks: List[Callable[[], CORO_ANY]] = []


APP_CONTEXT_VAR: contextvars.ContextVar[
    Optional[AppContext]] = contextvars.ContextVar("flowapp_context",
                                                   default=None)

EVENT_HANDLING_CONTEXT_VAR: contextvars.ContextVar[
    Optional[EventHandlingContext]] = contextvars.ContextVar("flowapp_event_context",
                                                   default=None)


def get_app_context() -> Optional[AppContext]:
    return APP_CONTEXT_VAR.get()

def get_event_handling_context() -> Optional[EventHandlingContext]:
    return EVENT_HANDLING_CONTEXT_VAR.get()

def is_inside_app():
    return is_inside_app_session() and get_app_context() is not None


def get_editable_app() -> "EditableApp":
    ctx = get_app_context()
    assert ctx is not None and ctx.is_editable_app()
    return ctx.app  # type: ignore


def get_app() -> "App":
    ctx = get_app_context()
    assert ctx is not None
    return ctx.app

def get_app_storage():
    ctx = get_app_context()
    assert ctx is not None
    return ctx.app.app_storage

def app_is_remote_comp() -> bool:
    ctx = get_app_context()
    assert ctx is not None
    return ctx.app._is_remote_component

@contextlib.contextmanager
def enter_app_context(app: "App"):
    ctx = AppContext(app)
    token = APP_CONTEXT_VAR.set(ctx)
    try:
        yield ctx
    finally:
        APP_CONTEXT_VAR.reset(token)

@contextlib.contextmanager
def enter_app_conetxt_obj(ctx: AppContext):
    token = APP_CONTEXT_VAR.set(ctx)
    try:
        yield ctx
    finally:
        APP_CONTEXT_VAR.reset(token)

ALL_APP_CONTEXT_GETTERS.add((get_app_context, enter_app_conetxt_obj))

@contextlib.contextmanager
def enter_event_handling_conetxt(uid: UniqueTreeId):
    ctx = EventHandlingContext(uid)
    token = EVENT_HANDLING_CONTEXT_VAR.set(ctx)
    try:
        yield ctx
    finally:
        EVENT_HANDLING_CONTEXT_VAR.reset(token)

def get_app_persist_storage():
    ctx = get_app_context()
    assert ctx is not None
    return ctx.app.get_persist_storage()

def enqueue_delayed_callback(cb: Callable[[], CORO_ANY]):
    ctx = get_event_handling_context()
    assert ctx is not None
    ctx.delayed_callbacks.append(cb)

def find_component(
        type: Type[T_comp],
        validator: Optional[Callable[[T_comp],
                                     bool]] = None) -> Optional[T_comp]:
    appctx = get_app_context()
    assert appctx is not None, "you must use this function in app"
    return appctx.app.find_component(type, validator)


def find_all_components(
        type: Type[T_comp],
        check_nested: bool = False,
        validator: Optional[Callable[[T_comp], bool]] = None) -> List[T_comp]:
    appctx = get_app_context()
    assert appctx is not None, "you must use this function in app"
    return appctx.app.find_all_components(type, check_nested, validator)


def find_component_by_uid(uid: str) -> Optional["Component"]:
    appctx = get_app_context()
    assert appctx is not None, "you must use this function in app"
    try:
        return appctx.app.root._get_comp_by_uid(uid)
    except KeyError:
        return None


def find_component_by_uid_with_type_check(
        uid: str, type: Type[T_comp]) -> Optional[T_comp]:
    appctx = get_app_context()
    assert appctx is not None, "you must use this function in app"
    try:
        res = appctx.app.root._get_comp_by_uid(uid)
        assert isinstance(res, type)
        return res
    except KeyError:
        return None


def get_reload_manager():
    appctx = get_app_context()
    assert appctx is not None, "you must use this function in app"
    return appctx.app._flow_reload_manager


class AppSpecialEventType(enum.Enum):
    EnterHoldContext = "EnterHoldContext"
    ExitHoldContext = "ExitHoldContext"
    # emitted when a flow event comes
    FlowForward = "FlowForward"
    Initialize = "Initialize"
    Exit = "Exit"
    # emitted after autorun exit
    AutoRunEnd = "AutoRunEnd"

    CodeEditorSave = "CodeEditorSave"
    WatchDogChange = "WatchDogChange"

    ObservedFunctionChange = "ObservedFunctionChange"
    # emitted when layout update is sent to frontend.
    LayoutChange = "LayoutChange"
    VscodeTensorpcMessage = "VscodeTensorpcMessage"
    VscodeBreakpointChange = "VscodeBreakpointChange"

    # emitted in remote comp server only
    RemoteCompMount = "RemoteCompMount"
    RemoteCompUnmount = "RemoteCompUnmount"


@dataclasses.dataclass
class _CompReloadMeta:
    uid: str
    handler: EventHandler
    cb_file: str
    cb_real: Callable
    cb_qualname: str


def create_reload_metas(uid_to_comp: Dict[str, "Component"], path: str):
    path_resolve = str(Path(path).resolve())
    metas: List[_CompReloadMeta] = []
    for k, v in uid_to_comp.items():
        # try:
        #     # if comp is inside tensorpc official, ignore it.
        #     Path(v_file).relative_to(PACKAGE_ROOT)
        #     continue
        # except:
        #     pass
        for handler_type, handlers in v._flow_event_handlers.items():
            for handler in handlers.handlers:
                cb = handler.cb
                is_valid_func = is_valid_function(cb)
                cb_real = cb
                if isinstance(cb, partial):
                    is_valid_func = is_valid_function(cb.func)
                    cb_real = cb.func

                if not is_valid_func or is_lambda(cb):
                    continue
                cb_file = str(Path(inspect.getfile(cb_real)).resolve())
                if cb_file != path_resolve:
                    continue
                # code, _ = inspect.getsourcelines(cb_real)
                metas.append(
                    _CompReloadMeta(k, handler, cb_file, cb_real,
                                    cb_real.__qualname__))
    return metas


class AppObservedFunctionRegistry(ObservedFunctionRegistry):

    def is_enabled(self):
        return not self.is_frozen and is_inside_app_session()


ALL_OBSERVED_FUNCTIONS: ObservedFunctionRegistryProtocol = AppObservedFunctionRegistry(
)


def observe_function(func: Callable):
    return ALL_OBSERVED_FUNCTIONS.register(func)


def observe_autorun_function(func: Callable):
    assert isinstance(ALL_OBSERVED_FUNCTIONS, AppObservedFunctionRegistry)
    return ALL_OBSERVED_FUNCTIONS.register(func, autorun_when_changed=True)


def observe_autorun_script(func: Callable):
    assert isinstance(ALL_OBSERVED_FUNCTIONS, AppObservedFunctionRegistry)
    return ALL_OBSERVED_FUNCTIONS.register(func,
                                           autorun_when_changed=True,
                                           autorun_block_symbol=r"#%%")



def run_coro_sync(coro: Coroutine, allow_current_thread: bool = True) -> Any:
    loop = get_app()._loop
    assert loop is not None
    if get_app()._flowapp_thread_id == threading.get_ident():
        if not allow_current_thread:
            raise RuntimeError("can't use run_coro_sync in current thread")
        # we can't wait fut here
        task = asyncio.create_task(coro)
        # we can't wait fut here
        return task
        # return fut
    else:
        # we can wait fut here.
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        return fut.result()
