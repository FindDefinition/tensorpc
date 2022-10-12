# Copyright 2022 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import enum
import io
from typing import (Any, AsyncGenerator, Awaitable, Callable, Coroutine, Dict,
                    Generic, Iterable, List, Optional, Tuple, Type, TypeVar,
                    Union)
import asyncio
import traceback
import inspect
from tensorpc.utils.registry import HashableRegistry
from tensorpc.utils.uniquename import UniqueNamePool
import dataclasses
import re
import sys
from typing_extensions import Literal, ParamSpec, Concatenate, Self, TypeAlias, Protocol
from tensorpc.apps.flow.coretypes import MessageLevel
ALL_APP_EVENTS = HashableRegistry()

_CORO_NONE = Union[Coroutine[None, None, None], None]
_CORO_ANY: TypeAlias = Union[Coroutine[Any, None, None], Any]

ValueType: TypeAlias = Union[int, float, str]

NumberType: TypeAlias = Union[int, float]

class Undefined:
    def __repr__(self) -> str:
        return "undefined"


class NoDefault:
    pass


# DON'T MODIFY THIS VALUE!!!
undefined = Undefined()
nodefault = NoDefault()


class UIType(enum.Enum):
    # controls
    ButtonGroup = 0x0
    Input = 0x1
    Switch = 0x2
    Select = 0x3
    Slider = 0x4
    RadioGroup = 0x5
    CodeEditor = 0x6
    Button = 0x7
    ListItemButton = 0x8
    ListItemText = 0x9
    Image = 0xa
    Text = 0xb
    Plotly = 0xc
    ChartJSLine = 0xd
    MultipleSelect = 0xe
    Paper = 0xf
    Typography = 0x10
    Collapse = 0x11
    Card = 0x12
    Chip = 0x13
    Accordion = 0x14
    Alert = 0x15
    AccordionSummary = 0x16
    AccordionDetail = 0x17
    TabContext = 0x18
    Tab = 0x19
    TabPanel = 0x1a
    TabList = 0x1b
    Checkbox = 0x1c
    AppBar = 0x1d
    Toolbar = 0x1e
    Drawer = 0x1f
    CircularProgress = 0x20
    LinearProgress = 0x21

    # special
    TaskLoop = 0x100
    FlexBox = 0x101
    MUIList = 0x102
    Divider = 0x103
    AppTerminal = 0x104
    ThemeProvider = 0x105

    # react fragment
    Fragment = 0x200

    MASK_THREE = 0x1000
    MASK_THREE_GEOMETRY = 0x0100
    
    ThreeCanvas = 0x1000
    ThreePoints = 0x1001

    ThreePerspectiveCamera = 0x1002
    ThreeGroup = 0x1003
    ThreeOrthographicCamera = 0x1004

    ThreeFlex = 0x1005
    ThreeFlexItemBox = 0x1006
    ThreeHtml = 0x1007
    
    ThreeHud = 0x1008

    ThreeMapControl = 0x1010
    ThreeOrbitControl = 0x1011
    ThreePointerLockControl = 0x1012
    ThreeFirstPersonControl = 0x1013
    ThreeTransformControl = 0x1014

    ThreeBoundingBox = 0x1020
    ThreeAxesHelper = 0x1021
    ThreeInfiniteGridHelper = 0x1022
    ThreeSegments = 0x1023
    ThreeImage = 0x1024
    ThreeBoxes2D = 0x1025

    ThreeText = 0x1026
    ThreeMeshMaterial = 0x1028
    ThreeMesh = 0x1029
    ThreeBufferGeometry = 0x102a
    ThreeFlexAutoReflow = 0x102b
    ThreeLine = 0x102c
    ThreeFlexManualReflow = 0x102d

    ThreeSimpleGeometry = 0x1101
    ThreeShape = 0x1102

    MASK_LEAFLET = 0x2000
    LeafletMapContainer = 0x2000
    LeafletTileLayer = 0x2001
    LeafletMarker = 0x2002
    LeafletPopup = 0x2003
    LeafletPolyline = 0x2004
    LeafletPolygon = 0x2005
    LeafletCircle = 0x2006
    LeafletRectangle = 0x2007
    LeafletTooltip = 0x2008
    LeafletCircleMarker = 0x2009

    # composite elements
    # a poly line and lots of circle markers/tooltips (typo) in single flowapp element.
    LeafletTracklet = 0x2100 



class AppEventType(enum.Enum):
    # layout events
    UpdateLayout = 0
    UpdateComponents = 1
    DeleteComponents = 2

    # ui event
    UIEvent = 10
    UIUpdateEvent = 11
    UISaveStateEvent = 12
    Notify = 13
    UIUpdatePropsEvent = 14
    UIException = 15
    # clipboard
    CopyToClipboard = 20
    # schedule event, won't be sent to frontend.
    ScheduleNext = 100
    # special UI event
    AppEditor = 200
    # for components that don't support contolled mode.
    # for example, editor and leaflet map
    UnControlledEvent = 300


class UIRunStatus(enum.Enum):
    Stop = 0
    Running = 1
    Pause = 2


class TaskLoopEvent(enum.Enum):
    Start = 0
    Stop = 1
    Pause = 2


class AppEditorEventType(enum.Enum):
    SetValue = 0
    RevealLine = 1


class AppEditorFrontendEventType(enum.Enum):
    Save = 0
    Change = 1
    SaveEditorState = 2

@dataclasses.dataclass
class UserMessage:
    uid: str
    error: str 
    level: MessageLevel
    detail: str

    def to_dict(self):
        return {
            "uid": self.uid,
            "error": self.error,
            "level": self.level.value,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, dc: Dict[str, str]):
        return cls(uid=dc["uid"], error=dc["error"], 
                detail=dc["detail"], level=MessageLevel(dc["level"]))

    @classmethod
    def create_error(cls, uid: str, error: str, detail: str):
        return cls(uid, error, MessageLevel.Error, detail)

    @classmethod
    def create_warning(cls, uid: str, error: str, detail: str):
        return cls(uid, error, MessageLevel.Warning, detail)

    @classmethod
    def createinfo(cls, uid: str, error: str, detail: str):
        return cls(uid, error, MessageLevel.Info, detail)

class AppEditorFrontendEvent:
    def __init__(self, type: AppEditorFrontendEventType, data: Any) -> None:
        self.type = type
        self.data = data

    def to_dict(self):
        return {
            "type": self.type.value,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(AppEditorFrontendEventType(data["type"]), data["data"])


@ALL_APP_EVENTS.register(key=AppEventType.UIEvent.value)
class UIEvent:
    def __init__(self, uid_to_data: Dict[str, Any]) -> None:
        self.uid_to_data = uid_to_data

    def to_dict(self):
        return self.uid_to_data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(data)

    def merge_new(self, new):
        return new


class NotifyType(enum.Enum):
    AppStart = 0
    AppStop = 1
    Reflow = 2


@ALL_APP_EVENTS.register(key=AppEventType.Notify.value)
class NotifyEvent:
    def __init__(self, type: NotifyType) -> None:
        self.type = type

    def to_dict(self):
        return self.type.value

    @classmethod
    def from_dict(cls, data: int):
        return cls(NotifyType(data))

    def merge_new(self, new):
        assert isinstance(new, NotifyEvent)
        return new

@ALL_APP_EVENTS.register(key=AppEventType.UnControlledEvent.value)
class UncontrolledEvent:
    def __init__(self, uid_to_data: Dict[str, Any]) -> None:
        self.uid_to_data = uid_to_data

    def to_dict(self):
        return self.uid_to_data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(data)

    def merge_new(self, new):
        assert isinstance(new, UncontrolledEvent)
        return UncontrolledEvent({
            **new.uid_to_data,
            **self.uid_to_data,
        })

@ALL_APP_EVENTS.register(key=AppEventType.UISaveStateEvent.value)
class UISaveStateEvent:
    def __init__(self, uid_to_data: Dict[str, Any]) -> None:
        self.uid_to_data = uid_to_data

    def to_dict(self):
        return self.uid_to_data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(data)

    def merge_new(self, new):
        assert isinstance(new, UISaveStateEvent)
        return UISaveStateEvent({
            **new.uid_to_data,
            **self.uid_to_data,
        })


@ALL_APP_EVENTS.register(key=AppEventType.UIUpdateEvent.value)
class UIUpdateEvent:
    def __init__(
        self, uid_to_data_undefined: Dict[str, Tuple[Dict[str, Any],
                                                     List[str]]]
    ) -> None:
        self.uid_to_data_undefined = uid_to_data_undefined

    def to_dict(self):
        return self.uid_to_data_undefined

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(data)

    def merge_new(self, new):
        assert isinstance(new, UIUpdateEvent)
        res_uid_to_data: Dict[str, Any] = self.uid_to_data_undefined.copy()
        for k, v in new.uid_to_data_undefined.items():
            if k in self.uid_to_data_undefined:
                res_uid_to_data[k] = ({
                    **v[0],
                    **self.uid_to_data_undefined[k][0]
                }, [*v[1], *self.uid_to_data_undefined[k][1]])
            else:
                res_uid_to_data[k] = v
        return UIUpdateEvent(res_uid_to_data)

@ALL_APP_EVENTS.register(key=AppEventType.UIException.value)
class UIExceptionEvent:
    def __init__(
        self, user_excs: List[UserMessage]) -> None:
        self.user_excs = user_excs

    def to_dict(self):
        return [v.to_dict() for v in self.user_excs]

    @classmethod
    def from_dict(cls, data: List[Any]):

        return cls([UserMessage.from_dict(v) for v in data])

    def merge_new(self, new):
        assert isinstance(new, UIExceptionEvent)
        return UIExceptionEvent(self.user_excs + new.user_excs)


@ALL_APP_EVENTS.register(key=AppEventType.AppEditor.value)
class AppEditorEvent:
    def __init__(self, type: AppEditorEventType, data) -> None:
        self.data = data
        self.type = type

    def to_dict(self):
        return {
            "type": self.type.value,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(AppEditorEventType(data["type"]), data["data"])

    def merge_new(self, new):
        assert isinstance(new, AppEditorEvent)
        return new


@ALL_APP_EVENTS.register(key=AppEventType.UpdateLayout.value)
class LayoutEvent:
    def __init__(self, data) -> None:
        self.data = data

    def to_dict(self):
        return self.data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(data)

    def merge_new(self, new):
        assert isinstance(new, LayoutEvent)

        return new


@ALL_APP_EVENTS.register(key=AppEventType.UpdateComponents.value)
class UpdateComponentsEvent:
    def __init__(self, data: Dict[str, Any]) -> None:
        self.data = data

    def to_dict(self):
        return self.data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(data)

    def merge_new(self, new):
        assert isinstance(new, UpdateComponentsEvent)
        return UpdateComponentsEvent({
            **new.data,
            **self.data,
        })


@ALL_APP_EVENTS.register(key=AppEventType.DeleteComponents.value)
class DeleteComponentsEvent:
    def __init__(self, data: List[str]) -> None:
        self.data = data

    def to_dict(self):
        return self.data

    @classmethod
    def from_dict(cls, data: List[str]):
        return cls(data)

    def merge_new(self, new):
        assert isinstance(new, DeleteComponentsEvent)
        return DeleteComponentsEvent(list(set(self.data + new.data)))


@ALL_APP_EVENTS.register(key=AppEventType.CopyToClipboard.value)
class CopyToClipboardEvent:
    def __init__(self, text: str) -> None:
        self.text = text

    def to_dict(self):
        return {
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(data["text"])

    def merge_new(self, new):
        assert isinstance(new, CopyToClipboardEvent)
        return new


@ALL_APP_EVENTS.register(key=AppEventType.ScheduleNext.value)
class ScheduleNextForApp:
    def __init__(self, data) -> None:
        self.data = data

    def to_dict(self):
        return self.data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(data)

    def merge_new(self, new):
        assert isinstance(new, ScheduleNextForApp)
        return new


APP_EVENT_TYPES = Union[UIEvent, LayoutEvent, CopyToClipboardEvent,
                        UpdateComponentsEvent, DeleteComponentsEvent,
                        ScheduleNextForApp, AppEditorEvent, UIUpdateEvent,
                        UISaveStateEvent, NotifyEvent, UIExceptionEvent,
                        UncontrolledEvent]


def app_event_from_data(data: Dict[str, Any]) -> "AppEvent":
    type_to_event: Dict[AppEventType, APP_EVENT_TYPES] = {}
    for evtypeval, evdata in data["typeToEvents"]:
        found = False
        for k, v in ALL_APP_EVENTS.items():
            if k == evtypeval:
                type_to_event[AppEventType(k)] = v.from_dict(evdata)
                found = True
                break
        if not found:
            raise ValueError("not found", evtypeval)
    return AppEvent(data["uid"], type_to_event)


# class ControlEvent:
#     def __init__(self, uid: str, data: Any) -> None:
#         self.uid = uid
#         self.data = data

#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]):
#         return cls(data["uid"], data["data"])


class AppEvent:
    def __init__(self,
                 uid: str,
                 type_to_event: Dict[AppEventType, APP_EVENT_TYPES],
                 sent_event: Optional[asyncio.Event] = None) -> None:
        self.uid = uid
        self.type_to_event = type_to_event
        # event that indicate this app event is sent
        # used for callback
        self.sent_event = sent_event

    def to_dict(self):
        # here we don't use dict for typeToEvents because key in js must be string.
        t2e = [(k.value, v.to_dict()) for k, v in self.type_to_event.items()]
        # make sure layout is proceed firstly.
        t2e.sort(key=lambda x: x[0])
        return {"uid": self.uid, "typeToEvents": t2e}

    def merge_new(self, new: "AppEvent") -> "AppEvent":
        new_type_to_event: Dict[AppEventType,
                                APP_EVENT_TYPES] = new.type_to_event.copy()
        if self.sent_event is not None:
            assert new.sent_event is None, "sent event of new must be None"
            sent_event = self.sent_event
        else:
            sent_event = new.sent_event
        for k, v in self.type_to_event.items():
            if k in new.type_to_event:
                new_type_to_event[k] = v.merge_new(new.type_to_event[k])
            else:
                new_type_to_event[k] = v
        return AppEvent(self.uid, new_type_to_event, sent_event)

    def __add__(self, other: "AppEvent"):
        return self.merge_new(other)

    def __iadd__(self, other: "AppEvent"):
        ret = self.merge_new(other)
        self.type_to_event = ret.type_to_event
        self.sent_event = ret.sent_event
        return self

def camel_to_snake(name: str):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('__([A-Z])', r'_\1', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower()


def snake_to_camel(name: str):
    res = ''.join(word.title() for word in name.split('_'))
    res = res[0].lower() + res[1:]
    return res



def _split_props_to_undefined(props: Dict[str, Any]):
        res = {}
        res_und = []
        for res_camel, val in props.items():
            if isinstance(val, Undefined):
                res_und.append(res_camel)
            else:
                res[res_camel] = val
        return res, res_und


@dataclasses.dataclass
class BasicProps:
    status: int = UIRunStatus.Stop.value
    def get_dict_and_undefined(self, state: Dict[str, Any]):
        this_type = type(self)
        res = {}
        ref_dict = dataclasses.asdict(self)
        res_und = []
        for field in dataclasses.fields(this_type):
            if field.name in state:
                continue
            res_camel = snake_to_camel(field.name)
            val = ref_dict[field.name]
            if isinstance(val, Undefined):
                res_und.append(res_camel)
            else:
                res[res_camel] = val
        return res, res_und

    def get_dict(self):
        this_type = type(self)
        res = {}
        ref_dict = dataclasses.asdict(self)
        for field in dataclasses.fields(this_type):
            res_camel = snake_to_camel(field.name)
            val = ref_dict[field.name]
            res[res_camel] = val
        return res

@dataclasses.dataclass
class ContainerBaseProps(BasicProps):
    childs: List[str] = dataclasses.field(default_factory=list)

T_base_props = TypeVar("T_base_props", bound=BasicProps)
T_container_props = TypeVar("T_container_props", bound=ContainerBaseProps)


P = ParamSpec('P')
T3 = TypeVar('T3')


def init_anno_fwd(
        this: Callable[P, Any],
        val: Optional[T3] = None) -> Callable[[Callable], Callable[P, T3]]:
    def decorator(real_function: Callable) -> Callable[P, T3]:
        def new_function(*args: P.args, **kwargs: P.kwargs) -> T3:
            return real_function(*args, **kwargs)

        return new_function

    return decorator

# TProp = TypeVar('TProp', covariant=True)

T_child = TypeVar("T_child")


class Component(Generic[T_base_props, T_child]):
    def __init__(self,
                 uid: str,
                 type: UIType,
                 prop_cls: Type[T_base_props],
                 queue: Optional[asyncio.Queue] = None) -> None:
        self._queue: Optional[asyncio.Queue] = queue
        self._flow_uid = uid
        self._flow_comp_type = type
        # self._status = UIRunStatus.Stop
        # task for callback of controls
        # if previous control callback hasn't finished yet,
        # the new control event will be IGNORED
        self._task: Optional[asyncio.Task] = None
        self._parent = ""
        self.__props = prop_cls()
        self.__prop_cls = prop_cls
        self._mounted_override = False
        self._flow_event_handlers: Dict[str, Union[EventHandler, Undefined]] = {}
        self.__sx_props: Dict[str, Any] = {}

    @property
    def props(self) -> T_base_props:
        return self.__props

    @property
    def propcls(self) -> Type[T_base_props]:
        return self.__prop_cls

    def is_mounted(self):
        return self._queue is not None

    def _prop_base(self, prop: Callable[P, Any], this: T3) -> Callable[P, T3]:
        def wrapper(*args: P.args, **kwargs: P.kwargs):
            for k, v in kwargs.items():
                setattr(self.__props, k, v)
            return this
        return wrapper 

    def _update_props_base(self, prop: Callable[P, Any]):
        def wrapper(*args: P.args, **kwargs: P.kwargs):
            for k, v in kwargs.items():
                setattr(self.__props, k, v)
            return self.create_update_event(kwargs)
        return wrapper 

    def get_callback(self) -> Optional[Callable]:
        return None

    def set_callback(self, val: Any):
        return

    async def handle_event(self, ev: Any):
        pass

    async def _clear(self):
        # self.uid = ""
        # self._queue = None
        # ignore all task error here.
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except:
                traceback.print_exc()

            # await _cancel(self._task)
            self._task = None
        self._parent = ""

    async def _cancel_task(self):
        # ignore all task error here.
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except:
                traceback.print_exc()
            self._task = None

    def set_sx_props(self, sx_props: Dict[str, Any]):
        self.__sx_props = sx_props

    def to_dict(self):
        """undefined will be removed here.
        if you reimplement to_dict, you need to use 
        camel name, no conversion provided.
        """
        props = self.get_props()
        props, und = _split_props_to_undefined(props)
        props.update(self.__sx_props)
        # state = self.get_state()
        # newstate = {}
        # for k, v in state.items():
        #     if not isinstance(v, Undefined):
        #         newstate[snake_to_camel(k)] = v
        # static, _ = self.__props.get_dict_and_undefined(state)  # type: ignore
        res = {
            "uid": self._flow_uid,
            "type": self._flow_comp_type.value,
            "props": props,
            # "status": self._status.value,
        }
        evs = []
        for k, v in self._flow_event_handlers.items():
            if not isinstance(v, Undefined):
                d = v.to_dict()
                d["type"] = k
                evs.append(d)
        if evs:
            res["usedEvents"] = evs
        return res

    def _to_dict_with_sync_props(self):
        props = self.get_sync_props()
        props, und = _split_props_to_undefined(props)
        res = {
            "uid": self._flow_uid,
            "type": self._flow_comp_type.value,
            "props": props,
            # "status": self._status.value,
        }
        return res

    def get_sync_props(self) -> Dict[str, Any]:
        """this function is used for props you want to kept when app
        shutdown or layout updated.
        1. app shutdown: only limited component support props recover.
        2. update layout: all component will override props
        by previous sync props
        """
        return {"status": self.props.status}

    def get_persist_props(self) -> Optional[Dict[str, Any]]:
        return None

    async def set_persist_props_async(self, state: Dict[str, Any]) -> None:
        return

    def get_props(self) -> Dict[str, Any]:
        return self.__props.get_dict()  # type: ignore

    def validate_props(self, props: Dict[str, Any]):
        """use this function to validate props before call
        set props.
        """
        return True

    def set_props(self, props: Dict[str, Any]):
        if self.validate_props(props):
            fields = dataclasses.fields(self.__props)
            name_to_fields = {f.name: f for f in fields}
            for name, value in props.items():
                if name in name_to_fields:
                    setattr(self.__props, name, value)

    async def put_app_event(self, ev: AppEvent):
        if self.is_mounted():
            return await self.queue.put(ev)

    @property
    def queue(self):
        assert self._queue is not None, f"you must add ui by flexbox.add_xxx"
        return self._queue

    def state_change_callback(self, data: Any):
        pass

    def create_update_event(self, data: Dict[str, Union[Any, Undefined]]):
        data_no_und = {}
        data_unds = []
        for k, v in data.items():
            k = snake_to_camel(k)
            if isinstance(v, Undefined):
                data_unds.append(k)
            else:
                data_no_und[k] = v
        ev = UIUpdateEvent({self._flow_uid: (data_no_und, data_unds)})
        # uid is set in flowapp service later.
        return AppEvent("", {AppEventType.UIUpdateEvent: ev})

    def create_update_prop_event(self, data: Dict[str, Union[Any, Undefined]]):
        data_no_und = {}
        data_unds = []
        for k, v in data.items():
            k = snake_to_camel(k)
            if isinstance(v, Undefined):
                data_unds.append(k)
            else:
                data_no_und[k] = v
        ev = UIUpdateEvent({self._flow_uid: (data_no_und, data_unds)})
        # uid is set in flowapp service later.
        return AppEvent("", {AppEventType.UIUpdatePropsEvent: ev})

    def create_uncontrolled_comp_event(self, data: Dict[str, Any]):
        ev = UncontrolledEvent({self._flow_uid: data})
        # uid is set in flowapp service later.
        return AppEvent("", {AppEventType.UnControlledEvent: ev})

    async def send_app_event_and_wait(self, ev: AppEvent):
        if ev.sent_event is None:
            ev.sent_event = asyncio.Event()
        await self.put_app_event(ev)
        if self.is_mounted():
            await ev.sent_event.wait()

    def create_update_comp_event(self, updates: Dict[str, Any]):
        ev = UpdateComponentsEvent(updates)
        # uid is set in flowapp service later.
        return AppEvent("", {AppEventType.UpdateComponents: ev})

    def create_delete_comp_event(self, deletes: List[str]):
        ev = DeleteComponentsEvent(deletes)
        # uid is set in flowapp service later.
        return AppEvent("", {AppEventType.DeleteComponents: ev})

    def create_user_msg_event(self, exc: UserMessage):
        ev = UIExceptionEvent([exc])
        # uid is set in flowapp service later.
        return AppEvent("", {AppEventType.UIException: ev})

    def create_editor_event(self, type: AppEditorEventType, data: Any):
        # uid is set in flowapp service later.
        ev = AppEditorEvent(type, data)
        return AppEvent("", {AppEventType.AppEditor: ev})

    async def run_callback(self,
                           cb: Callable[[], _CORO_NONE],
                           sync_state: bool = False,
                           sync_first: bool = True):
        self.props.status = UIRunStatus.Running.value
        # only ui with loading support need sync first.
        # otherwise don't use this because slow
        if sync_first:
            ev = asyncio.Event()
            await self.sync_status(sync_state, ev)
            await ev.wait()
        try:
            coro = cb()
            if inspect.iscoroutine(coro):
                await coro
        except Exception as e:
            traceback.print_exc()
            ss = io.StringIO()
            traceback.print_exc(file=ss)
            user_exc = UserMessage.create_error(self._flow_uid, repr(e), ss.getvalue())
            await self.put_app_event(self.create_user_msg_event(user_exc))
        finally:
            self.props.status = UIRunStatus.Stop.value 
            await self.sync_status(sync_state)


    async def sync_status(self,
                          sync_state: bool = False,
                          sent_event: Optional[asyncio.Event] = None):
        if sync_state:
            ev = self.create_update_event(self.get_sync_props())
            ev.sent_event = sent_event
            await self.put_app_event(ev)
        else:
            ev = self.create_update_event({"status": self.props.status})
            ev.sent_event = sent_event
            await self.put_app_event(ev)

    async def sync_state(self, sent_event: Optional[asyncio.Event] = None):
        return await self.sync_status(True, sent_event)

    def get_sync_event(self, sync_state: bool = False):
        if sync_state:
            return self.create_update_event(self.get_sync_props())
        else:
            return self.create_update_event({"status": self.props.status})


class ContainerBase(Component[T_container_props, T_child]):
    def __init__(self,
                 base_type: UIType,
                 prop_cls: Type[T_container_props],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 _children: Optional[Dict[str, T_child]] = None,
                 inited: bool = False) -> None:
        super().__init__(uid, base_type, prop_cls, queue)
        if inited:
            assert queue is not None and uid_to_comp is not None
        if uid_to_comp is None:
            uid_to_comp = {}
        self._pool = UniqueNamePool()
        self._uid_to_comp = uid_to_comp
        self._children = _children

        # self.props.childs: List[str] = []

        self.inited = inited
        self._prevent_add_layout = False


    async def _clear(self):
        for c in self.props.childs:
            cc = self[c]
            await cc._clear()
        self.props.childs.clear()
        await super()._clear()
        self._pool.unique_set.clear()

    def _attach_to_app(self, queue: asyncio.Queue):
        # update queue of this and childs
        if not self.inited:
            self._queue = queue
            for k, v in self._uid_to_comp.items():
                v._queue = queue
                if isinstance(v, ContainerBase):
                    v.inited = True
            self.inited = True

    def _update_child_ns_and_comp_dict(self, ns: str,
                                       uid_to_comp: Dict[str, Component]):
        prev_uid_to_comp = self._uid_to_comp
        self._uid_to_comp = uid_to_comp
        for k, v in prev_uid_to_comp.items():
            new_name = f"{ns}.{k}"
            v._flow_uid = new_name
            v._parent = ns
            uid_to_comp[v._flow_uid] = v
            if isinstance(v, ContainerBase):
                v._uid_to_comp = uid_to_comp

    def __getitem__(self, key: str):
        assert key in self.props.childs, f"{key}, {self.props.childs}"
        # print(key, self, self.uid, self._get_uid_with_ns(key), len(self._uid_to_comp))
        return self._uid_to_comp[self._get_uid_with_ns(key)]

    def _get_all_nested_child_recursive(self, name: str, res: List[Component]):
        comp = self[name]
        res.append(comp)
        if isinstance(comp, ContainerBase):
            for child in comp.props.childs:
                comp._get_all_nested_child_recursive(child, res)

    def _get_all_nested_child(self, name: str):
        res: List[Component] = []
        self._get_all_nested_child_recursive(name, res)
        return res

    def _get_all_nested_childs(self):
        comps: List[Component] = []
        for c in self.props.childs:
            comps.extend(self._get_all_nested_child(c))
        return comps

    def _get_uid_with_ns(self, name: str):
        if self._flow_uid == "":
            return (f"{name}")
        return (f"{self._flow_uid}.{name}")

    def _add_prop_to_ui(self, ui: Component):
        if self.inited:
            ui._queue = self.queue
        ui._parent = self._flow_uid
        if isinstance(ui, ContainerBase):
            ui._pool = self._pool
            ui._uid_to_comp = self._uid_to_comp

    def add_component(self,
                      name: str,
                      comp: Component,
                      add_to_state: bool = True,
                      anonymous: bool = False):
        uid = self._get_uid_with_ns(name)
        if anonymous:
            uid = self._pool(uid)
        comp._flow_uid = uid
        if add_to_state:
            assert uid not in self._uid_to_comp
        assert comp._parent == "", "this component must not be added before."
        self._add_prop_to_ui(comp)
        if add_to_state:
            self._uid_to_comp[comp._flow_uid] = comp
            self.props.childs.append(name)
        return comp

    def add_component_v2(self,
                      name: str,
                      comp: Component):
        """ 
        we assume self uid is ready.
        if non-container
            1. set uid to ns + component name
            2. set queue, 
            done
        if container
            2.1 set uid_to_comp
            3. if has init dict, consume it
            4. change uid/parent of all nested childs with new namespace
        """
        # consume children  before assign uid to 
        # ensure comp is a child-inited standclone component.
        if isinstance(comp, ContainerBase):
            if comp._children is not None:
                # consume this _children
                comp.add_layout(comp._children)
                comp._children = None

        namespace = self._flow_uid
        if namespace == "":
            comp._flow_uid = name
        else:
            comp._flow_uid = namespace + "." + name
        comp_child_ns = comp._flow_uid
        # print("WTF", comp_child_ns, comp.uid)
        comp._queue = self._queue
        self._uid_to_comp[comp._flow_uid] = comp
        comps_added: List[Component] = [comp]
        if isinstance(comp, ContainerBase):
            comp_uid_to_comp = comp._uid_to_comp
            # set all uid/parent of child with new uid
            for k, v in comp_uid_to_comp.items():
                new_name = f"{comp_child_ns}.{k}"
                new_parent = f"{comp_child_ns}.{v._parent}"
                v._flow_uid = new_name
                v._parent = new_parent
                v._queue = self._queue
                # add standclone comp map to main map
                self._uid_to_comp[v._flow_uid] = v
                # print("WTF2", comp_child_ns, k)
                comps_added.append(v)
                if isinstance(v, ContainerBase):
                    v._uid_to_comp = self._uid_to_comp
            comp._uid_to_comp = self._uid_to_comp
        self.props.childs.append(name)
        return comp, comps_added

    def add_layout(self, layout: Dict[str, Component]):
        """ {
            btn0: Button(...),
            box0: VBox({
                btn1: Button(...),
                ...
            }, flex...),
        }
        """
        if self._prevent_add_layout:
            raise ValueError("you must init layout in app_create_layout")
        comps_added: List[Component] = []
        # we assume layout is a ordered dict (python >= 3.7)
        for k, v in layout.items():
            comps_added.extend(self.add_component_v2(k, v)[1])
        return comps_added

    def get_props(self):
        state = super().get_props()
        state["childs"] = [self[n]._flow_uid for n in self.props.childs]
        return state

    def remove_child(self, name: str):
        """if child isn't a container, just stop callback task, reset uid/_parent and remove from 
        self child.
        if child is a container, we need to reset namespace/parent to 
        standalone component.
        """
        # TODO we may need a global aio lock here.
        comp = self[name]
        self.props.childs.remove(name)
        return _detach_component(comp)

    async def set_new_layout(self, layout: Dict[str, Component]):
        # for k, v in layout.items():
        #     print("-------------", k, v._flow_uid, type(v), len(v._uid_to_comp))
        #     for vv in v._uid_to_comp.values():
        #         print(vv._flow_uid)
        # print("===================")
        # TODO we need to remove from parent when we detach a component.
        for k, v in layout.items():
            _detach_component(v)
        #     _detach_component(v)

        # for k, v in layout.items():
        #     print("-------------", k, type(v), len(v._uid_to_comp))
        #     for vv in v._uid_to_comp.values():
        #         print(vv._flow_uid)

        # remove all first
        # TODO we may need to stop task of a comp
        comps_to_remove: List[Component] = []
        comp_uids_to_remove: List[str] = []
        for c in self.props.childs.copy():
            comp, comp_to_remove_this, comp_to_remove_uids = self.remove_child(c)
            comps_to_remove.extend(comp_to_remove_this)
            comp_uids_to_remove.extend(comp_to_remove_uids)
        for comp in comps_to_remove:
            await comp._cancel_task()
        # TODO should we merge two events to one?
        await self.put_app_event(
            self.create_delete_comp_event(comp_uids_to_remove))
        self.add_layout(layout)
        comps_frontend = {
            c._flow_uid: c.to_dict()
            for c in self._get_all_nested_childs()
        }
        # make sure all child of this box is rerendered.
        # TODO merge events
        comps_frontend[self._flow_uid] = self.to_dict()
        await self.put_app_event(self.create_update_comp_event(comps_frontend))
        child_uids = [self[c]._flow_uid for c in self.props.childs]
        await self.put_app_event(self.create_update_event({"childs": child_uids}))

    async def remove_childs_by_keys(self, keys: List[str]):
        comps_to_remove: List[Component] = []
        comp_uids_to_remove: List[str] = []
        for c in keys:
            if c not in self.props.childs:
                continue
            comp, comp_to_remove_this, comp_to_remove_uids = self.remove_child(c)
            comps_to_remove.extend(comp_to_remove_this)
            comp_uids_to_remove.extend(comp_to_remove_uids)
        for comp in comps_to_remove:
            await comp._cancel_task()
        if not comps_to_remove:
            return
        await self.put_app_event(
            self.create_delete_comp_event(comp_uids_to_remove))
        child_uids = [self[c]._flow_uid for c in self.props.childs]
        await self.put_app_event(self.create_update_event({"childs": child_uids}))

    async def update_childs(self, layout: Dict[str, Component]):
        for k, v in layout.items():
            if k not in self.props.childs:
                _detach_component(v)
        # remove replaced components first.
        comps_to_remove: List[Component] = []
        comp_uids_to_remove: List[str] = []
        for c in self.props.childs:
            comp = self[c]
            if c in layout:
                comp_detached, comp_to_remove_this, comp_to_remove_uids = self.remove_child(c)
                comps_to_remove.extend(comp_to_remove_this)
                comp_uids_to_remove.extend(comp_to_remove_uids)
        for comp in comps_to_remove:
            await comp._cancel_task()

        comp_added = self.add_layout(layout)
        update_comps_frontend = {c._flow_uid: c.to_dict() for c in comp_added}
        # make sure all child of this box is rerendered.
        update_comps_frontend[self._flow_uid] = self.to_dict()
        await self.put_app_event(
            self.create_update_comp_event(update_comps_frontend))

    async def replace_childs(self, layout: Dict[str, Component]):
        for k in layout.keys():
            assert k in self.props.childs
        return await self.update_childs(layout)


def _detach_component(comp: Component):
    """detach a component from app.
    """
    # print("-----DETACH------", comp.uid, comp)
    standalone_map: Dict[str, Component] = {}
    all_removed_prev_uids = [comp._flow_uid]
    all_removed = [comp]

    if isinstance(comp, ContainerBase):
        all_child = comp._get_all_nested_childs()
        for c in all_child:
            all_removed_prev_uids.append(c._flow_uid)
            c._queue = None
            if comp._flow_uid:
                c._parent = c._parent[len(comp._flow_uid) + 1:]
                c._flow_uid = c._flow_uid[len(comp._flow_uid) + 1:]
            if isinstance(c, ContainerBase):
                c._uid_to_comp = standalone_map
            standalone_map[c._flow_uid] = c
        all_removed.extend(all_removed)
        comp._uid_to_comp = standalone_map

    comp._flow_uid = ""
    comp._parent = ""
    comp._queue = None
    return comp, all_removed, all_removed_prev_uids

@dataclasses.dataclass
class FragmentProps(ContainerBaseProps):
    disabled: Union[Undefined, bool] = undefined

class Fragment(ContainerBase[FragmentProps, Component]):
    def __init__(self,
                 children: Dict[str, Component],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(UIType.Fragment, FragmentProps, uid, queue,
                         uid_to_comp, children, inited)
    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def set_disabled(self, disabled: bool):
        await self.send_app_event_and_wait(self.update_event(disabled=disabled))

class EventHandler:

    def __init__(self,
                 cb: Callable[[Any], _CORO_ANY],
                 stop_propagation: bool = False,
                 throttle: Optional[NumberType] = None,
                 debounce: Optional[NumberType] = None) -> None:
        self.cb = cb
        self.stop_propagation = stop_propagation
        self.debounce = debounce
        self.throttle = throttle

    def to_dict(self):
        res: Dict[str, Any] = {
            "stopPropagation": self.stop_propagation,
        }
        if self.debounce is not None:
            res["debounce"] = self.debounce
        if self.throttle is not None:
            res["throttle"] = self.throttle
        return res 


def create_ignore_usr_msg(comp: Component):
    msg = comp.create_user_msg_event((UserMessage.create_warning(comp._flow_uid, "UI Running", 
        f"UI {comp._flow_uid}@{str(type(comp).__name__)} is still running, so ignore your control")))
    return msg

