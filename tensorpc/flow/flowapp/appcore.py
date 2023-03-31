import contextlib
import contextvars
import dataclasses
import enum
import inspect
import traceback
from functools import partial
from pathlib import Path
from typing import (TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable,
                    Coroutine, Dict, Generic, Iterable, List, Optional, Set, Tuple,
                    Type, TypeVar, Union)

from typing_extensions import (Concatenate, Literal, ParamSpec, Protocol, Self,
                               TypeAlias)

from tensorpc.core.moduleid import (get_qualname_of_type, is_lambda,
                                    is_valid_function)
from tensorpc.flow.jsonlike import Undefined, BackendOnlyProp, undefined
from tensorpc.core.serviceunit import ObservedFunctionRegistry
from tensorpc.flow.client import is_inside_app

if TYPE_CHECKING:
    from .app import App, EditableApp
    from .core import Component

CORO_NONE = Union[Coroutine[None, None, None], None]
CORO_ANY: TypeAlias = Union[Coroutine[Any, None, None], Any]

ValueType: TypeAlias = Union[int, float, str]

NumberType: TypeAlias = Union[int, float]

EventType: TypeAlias = Tuple[ValueType, Any]
T = TypeVar("T")

T_comp = TypeVar("T_comp")


class EventHandler:

    def __init__(self,
                 cb: Callable,
                 stop_propagation: bool = False,
                 throttle: Optional[NumberType] = None,
                 debounce: Optional[NumberType] = None,
                 backend_only: bool = False) -> None:
        self.cb = cb
        self.stop_propagation = stop_propagation
        self.debounce = debounce
        self.throttle = throttle
        self.backend_only = backend_only

    def to_dict(self):
        res: Dict[str, Any] = {
            "stopPropagation": self.stop_propagation,
        }
        if self.debounce is not None:
            res["debounce"] = self.debounce
        if self.throttle is not None:
            res["throttle"] = self.throttle
        return res


class AppContext:

    def __init__(self, app: "App") -> None:
        self.app = app

    def is_editable_app(self):
        return self.app._is_editable_app()


APP_CONTEXT_VAR: contextvars.ContextVar[
    Optional[AppContext]] = contextvars.ContextVar("flowapp_context",
                                                   default=None)


def get_app_context() -> Optional[AppContext]:
    return APP_CONTEXT_VAR.get()


def get_editable_app() -> "EditableApp":
    ctx = get_app_context()
    assert ctx is not None and ctx.is_editable_app()
    return ctx.app # type: ignore

def get_app() -> "App":
    ctx = get_app_context()
    assert ctx is not None
    return ctx.app

@contextlib.contextmanager
def enter_app_conetxt(app: "App"):
    ctx = AppContext(app)
    token = APP_CONTEXT_VAR.set(ctx)
    try:
        yield ctx
    finally:
        APP_CONTEXT_VAR.reset(token)


def get_app_storage():
    ctx = get_app_context()
    assert ctx is not None
    return ctx.app.get_persist_storage()


def find_component(type: Type[T_comp]) -> Optional[T_comp]:
    appctx = get_app_context()
    assert appctx is not None, "you must use this function in app"
    return appctx.app.find_component(type)

def find_all_components(type: Type[T_comp]) -> List[T_comp]:
    appctx = get_app_context()
    assert appctx is not None, "you must use this function in app"
    return appctx.app.find_all_components(type)

def find_component_by_uid(uid: str) -> Optional["Component"]:
    appctx = get_app_context()
    assert appctx is not None, "you must use this function in app"
    try:
        return appctx.app.root._get_comp_by_uid(uid)
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
        for handler_type, handler in v._flow_event_handlers.items():
            if not isinstance(handler, Undefined):
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
                metas.append(_CompReloadMeta(k, handler, cb_file, cb_real, cb_real.__qualname__))
    return metas


async def _run_zeroarg_func(cb: Callable):
    try:
        coro = cb()
        if inspect.iscoroutine(coro):
            await coro
    except:
        traceback.print_exc()


class AppObservedFunctionRegistry(ObservedFunctionRegistry):
    def handle_record(self, sig: inspect.Signature, args, kwargs):
        bargs = sig.bind(*args, **kwargs)
        return 
    
    def is_enabled(self):
        return not self.is_frozen and is_inside_app()

_ALL_OBSERVED_FUNCTIONS = AppObservedFunctionRegistry()

def observe_function(func: Callable):
    return _ALL_OBSERVED_FUNCTIONS.register(func)