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
import builtins
from pathlib import Path
import threading
from typing import (Any, AsyncGenerator, Awaitable, Callable, Coroutine, Dict,
                    Generic, Iterable, List, Optional, Tuple, Type, TypeVar,
                    Union, Set)
import asyncio
import traceback
import inspect

import cachetools
from tensorpc.core.serviceunit import AppFuncType, ReloadableDynamicClass, ServFunctionMeta
from tensorpc.utils.registry import HashableRegistry
from tensorpc.utils.uniquename import UniqueNamePool
from tensorpc.flow.constants import TENSORPC_ANYLAYOUT_FUNC_NAME, TENSORPC_LEGACY_LAYOUT_FUNC_NAME
import dataclasses
import re
import sys
from typing_extensions import Literal, ParamSpec, Concatenate, Self, TypeAlias, Protocol
from tensorpc.flow.coretypes import MessageLevel

ALL_APP_EVENTS = HashableRegistry()

_CORO_NONE = Union[Coroutine[None, None, None], None]
_CORO_ANY: TypeAlias = Union[Coroutine[Any, None, None], Any]

ValueType: TypeAlias = Union[int, float, str]

NumberType: TypeAlias = Union[int, float]

EventType: TypeAlias = Tuple[ValueType, Any]


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
    Dialog = 0xb
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
    ToggleButton = 0x22
    ToggleButtonGroup = 0x23
    AutoComplete = 0x24
    MultipleAutoComplete = 0x25
    IconButton = 0x26
    JsonLikeTreeView = 0x27
    Allotment = 0x28
    AllotmentPane = 0x29
    FlexLayout = 0x30
    MosaicLayout = 0x31

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
    ThreeCameraControl = 0x1015

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
    FrontendUIEvent = 16
    # clipboard
    CopyToClipboard = 20
    # schedule event, won't be sent to frontend.
    ScheduleNext = 100
    # special UI event
    AppEditor = 200
    # send event to component, for append update
    # and uncontrolled component.
    ComponentEvent = 300


class FrontendEventType(enum.Enum):
    # only used on backend
    # if user don't define DragCollect handler, Drop won't be scheduled.
    DragCollect = -1

    Click = 0
    DoubleClick = 1
    Enter = 2
    Leave = 3
    Over = 4
    Out = 5
    Up = 6
    Down = 7
    ContextMenu = 8
    Move = 9
    Change = 20
    Delete = 21
    InputChange = 22
    DialogClose = 23
    Drag = 24
    Drop = 25

    TreeLazyExpand = 30
    TreeItemSelect = 31
    TreeItemToggle = 32
    TreeItemFocus = 33

    ComplexLayoutCloseTab = 40
    ComplexLayoutSelectTab = 41

    # leaflet events
    MapZoom = 60
    MapMove = 61

    PlotlyClickData = 100
    PlotlyClickAnnotation = 101


ALL_POINTER_EVENTS = [
    FrontendEventType.Down.value,
    FrontendEventType.Up.value,
    FrontendEventType.Move.value,
    FrontendEventType.Enter.value,
    FrontendEventType.Leave.value,
    FrontendEventType.Over.value,
    FrontendEventType.Out.value,
]


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
        return cls(uid=dc["uid"],
                   error=dc["error"],
                   detail=dc["detail"],
                   level=MessageLevel(dc["level"]))

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

    def __init__(self, uid_to_data: Dict[str, EventType]) -> None:
        self.uid_to_data = uid_to_data

    def to_dict(self):
        return self.uid_to_data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(data)

    def merge_new(self, new):
        return new


@ALL_APP_EVENTS.register(key=AppEventType.FrontendUIEvent.value)
class FrontendUIEvent:

    def __init__(
            self, uid_to_data: Dict[str, Tuple[Union[NumberType, str],
                                               Any]]) -> None:
        self.uid_to_data = uid_to_data

    def to_dict(self):
        return self.uid_to_data

    @classmethod
    def from_dict(cls, data: Dict[str, Tuple[Union[NumberType, str], Any]]):
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


@ALL_APP_EVENTS.register(key=AppEventType.ComponentEvent.value)
class ComponentEvent:

    def __init__(self, uid_to_data: Dict[str, Any]) -> None:
        self.uid_to_data = uid_to_data

    def to_dict(self):
        return self.uid_to_data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(data)

    def merge_new(self, new):
        assert isinstance(new, ComponentEvent)
        return ComponentEvent({
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

    def __init__(self, user_excs: List[UserMessage]) -> None:
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

    def __init__(self,
                 data: Dict[str, Any],
                 deleted: Optional[List[str]] = None) -> None:
        self.data = data
        if deleted is None:
            deleted = []
        self.deleted = deleted

    def to_dict(self):
        return {
            "new": self.data,
            "del": self.deleted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(data["new"], data["del"])

    def merge_new(self, new):
        assert isinstance(new, UpdateComponentsEvent)
        return UpdateComponentsEvent({
            **new.data,
            **self.data,
        }, list(set(self.deleted + new.deleted)))


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
                        ComponentEvent, FrontendUIEvent]


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
                 sent_event: Optional[asyncio.Event] = None,
                 event_id: str = "") -> None:
        self.uid = uid
        self.type_to_event = type_to_event
        # event that indicate this app event is sent
        # used for callback
        self.sent_event = sent_event
        self.event_id = event_id

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

    def get_event_uid(self):
        if self.event_id:
            return self.uid + "-" + self.event_id
        return self.uid

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


def _undefined_dict_factory(x: List[Tuple[str, Any]]):
    res: Dict[str, Any] = {}
    for k, v in x:
        if not isinstance(v, Undefined):
            res[k] = v
    return res


@dataclasses.dataclass
class _DataclassSer:
    obj: Any


def as_dict_no_undefined(obj: Any):
    return dataclasses.asdict(_DataclassSer(obj),
                              dict_factory=_undefined_dict_factory)["obj"]


@dataclasses.dataclass
class DataClassWithUndefined:

    def get_dict_and_undefined(self, state: Dict[str, Any]):
        this_type = type(self)
        res = {}
        # we only support update in first-level dict,
        # so we ignore all undefined in childs.
        ref_dict = dataclasses.asdict(self,
                                      dict_factory=_undefined_dict_factory)
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
        ref_dict = dataclasses.asdict(self,
                                      dict_factory=_undefined_dict_factory)
        for field in dataclasses.fields(this_type):
            res_camel = snake_to_camel(field.name)
            if field.name not in ref_dict:
                val = undefined
            else:
                val = ref_dict[field.name]
            res[res_camel] = val
        return res


@dataclasses.dataclass
class BasicProps(DataClassWithUndefined):
    status: int = UIRunStatus.Stop.value


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


def _get_obj_def_path(obj):
    try:
        _flow_comp_def_path = str(
            Path(inspect.getfile(builtins.type(obj))).resolve())
    except:
        traceback.print_exc()
        _flow_comp_def_path = ""
    path = Path(_flow_comp_def_path)
    if not path.exists() or path.suffix != ".py":
        _flow_comp_def_path = ""
    return _flow_comp_def_path


class FlowSpecialMethods:

    def __init__(self, metas: List[ServFunctionMeta]) -> None:
        self.create_layout: Optional[ServFunctionMeta] = None
        self.auto_run: Optional[ServFunctionMeta] = None
        self.did_mount: Optional[ServFunctionMeta] = None
        self.will_unmount: Optional[ServFunctionMeta] = None
        self.create_object: Optional[ServFunctionMeta] = None

        self.metas = metas
        for m in self.metas:
            # assert m.is_binded, "metas must be binded before this class"
            if m.name == TENSORPC_ANYLAYOUT_FUNC_NAME:
                self.create_layout = m
            elif m.name == TENSORPC_LEGACY_LAYOUT_FUNC_NAME:
                self.create_layout = m
            elif m.user_app_meta is not None:
                if m.user_app_meta.type == AppFuncType.CreateLayout:
                    self.create_layout = m
                elif m.user_app_meta.type == AppFuncType.ComponentDidMount:
                    self.did_mount = m
                elif m.user_app_meta.type == AppFuncType.ComponentWillUnmount:
                    self.will_unmount = m
                elif m.user_app_meta.type == AppFuncType.CreateObject:
                    self.create_object = m
                elif m.user_app_meta.type == AppFuncType.AutoRun:
                    self.auto_run = m

    def bind(self, obj):
        if self.create_layout is not None:
            self.create_layout.bind(obj)
        if self.auto_run is not None:
            self.auto_run.bind(obj)
        if self.did_mount is not None:
            self.did_mount.bind(obj)
        if self.will_unmount is not None:
            self.will_unmount.bind(obj)
        if self.create_object is not None:
            self.create_object.bind(obj)



_TYPE_METHOD_CACHE = {}


@cachetools.cached(_TYPE_METHOD_CACHE, lock=threading.Lock())
def get_special_methods(obj_type):
    new_metas = ReloadableDynamicClass.get_metas_of_regular_methods(
        obj_type, include_base=True)
    flow_special = FlowSpecialMethods(new_metas)
    return flow_special


class Component(Generic[T_base_props, T_child]):

    def __init__(self,
                 type: UIType,
                 prop_cls: Type[T_base_props],
                 allowed_events: Optional[Iterable[ValueType]] = None,
                 uid: str = "",
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
        self._flow_event_handlers: Dict[ValueType, Union[EventHandler,
                                                         Undefined]] = {}
        self.__sx_props: Dict[str, Any] = {}
        self._flow_allowed_events: Set[ValueType] = set()
        if allowed_events is not None:
            self._flow_allowed_events = set(allowed_events)
        self._flow_user_data: Any = None
        self._flow_comp_def_path = _get_obj_def_path(self)
        self._flow_reference_count = 0

    def get_special_methods(self):
        res = get_special_methods(type(self))
        res.bind(self)
        return res 

    @property
    def props(self) -> T_base_props:
        return self.__props

    @property
    def propcls(self) -> Type[T_base_props]:
        return self.__prop_cls

    def _attach(self, uid: str, queue: asyncio.Queue) -> dict:
        if self._flow_reference_count == 0:
            self._flow_uid = uid
            self._queue = queue
            self._flow_reference_count += 1
            return {uid: self}
        self._flow_reference_count += 1
        return {}

    def _detach(self) -> dict:
        self._flow_reference_count -= 1
        if self._flow_reference_count == 0:
            res_uid = self._flow_uid
            self._flow_uid = ""
            self._queue = None
            return {res_uid: self}
        return {}

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

    async def handle_event(self, ev: EventType):
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
            if not isinstance(v, Undefined) and not v.backend_only:
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

    def register_event_handler(self,
                               type: ValueType,
                               cb: Callable,
                               stop_propagation: bool = False,
                               throttle: Optional[NumberType] = None,
                               debounce: Optional[NumberType] = None,
                               backend_only: bool = False):
        if self._flow_allowed_events:
            if not backend_only:
                assert type in self._flow_allowed_events, f"only support events: {self._flow_allowed_events}"
        evh = EventHandler(cb, stop_propagation, throttle, debounce,
                           backend_only)
        self._flow_event_handlers[type] = evh
        return evh

    def clear_event_handlers(self):
        self._flow_event_handlers.clear()

    def get_event_handler(self, type: ValueType):
        res = self._flow_event_handlers.get(type)
        if isinstance(res, Undefined):
            res = None
        return res

    def state_change_callback(self,
                              data: Any,
                              type: ValueType = FrontendEventType.Change.value
                              ):
        pass

    def create_update_event(self, data: Dict[str, Union[Any, Undefined]]):
        data_no_und = {}
        data_unds = []
        for k, v in data.items():
            k = snake_to_camel(k)
            if isinstance(v, Undefined):
                data_unds.append(k)
            else:
                data_no_und[k] = as_dict_no_undefined(v)
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

    def create_comp_event(self, data: Dict[str, Any]):
        ev = ComponentEvent({self._flow_uid: data})
        # uid is set in flowapp service later.
        return AppEvent("", {AppEventType.ComponentEvent: ev})

    async def send_and_wait(self, ev: AppEvent):
        if ev.sent_event is None:
            ev.sent_event = asyncio.Event()
        await self.put_app_event(ev)
        if self.is_mounted():
            await ev.sent_event.wait()

    def create_update_comp_event(self, updates: Dict[str, Any],
                                 deleted: Optional[List[str]]):
        ev = UpdateComponentsEvent(updates, deleted)
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
                           sync_first: bool = True,
                           res_callback: Optional[Callable[[Any],
                                                           _CORO_NONE]] = None,
                           change_status: bool = True):
        if change_status:
            self.props.status = UIRunStatus.Running.value
        # only ui with loading support need sync first.
        # otherwise don't use this because slow
        if sync_first:
            ev = asyncio.Event()
            await self.sync_status(sync_state, ev)
            await ev.wait()
        res = None
        try:
            coro = cb()
            if inspect.iscoroutine(coro):
                res = await coro
            else:
                res = coro
            if res_callback is not None:
                res_coro = res_callback(res)
                if inspect.iscoroutine(res_coro):
                    await res_coro

        except Exception as e:
            traceback.print_exc()
            ss = io.StringIO()
            traceback.print_exc(file=ss)
            user_exc = UserMessage.create_error(self._flow_uid, repr(e),
                                                ss.getvalue())
            await self.put_app_event(self.create_user_msg_event(user_exc))
        finally:
            if change_status:
                self.props.status = UIRunStatus.Stop.value
                await self.sync_status(sync_state)
        return res

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
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 _children: Optional[Dict[str, T_child]] = None,
                 inited: bool = False,
                 allowed_events: Optional[Iterable[ValueType]] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(base_type, prop_cls, allowed_events, uid, queue)
        if inited:
            assert queue is not None  # and uid_to_comp is not None
        if uid_to_comp is None:
            uid_to_comp = {}
        self._pool = UniqueNamePool()
        self._uid_to_comp = uid_to_comp
        if _children is None:
            _children = {}
        # self._children = _children
        self._child_comps: Dict[str, Component] = {}
        for k, v in _children.items():
            assert isinstance(v, Component)
            self._child_comps[k] = v
        # self.props.childs: List[str] = []
        self.inited = inited
        self._prevent_add_layout = False

    def _get_comp_by_uid(self, uid: str):
        parts = uid.split(".")
        # uid contains root, remove it at first.
        return self._get_comp_by_uid_resursive(parts[1:])

    def _get_comp_by_uid_resursive(self, parts: List[str]) -> Component:
        key = parts[0]
        assert key in self._child_comps
        child_comp = self._child_comps[key]
        if len(parts) == 1:
            return self._child_comps[key]
        else:
            assert isinstance(child_comp, ContainerBase)
            return child_comp._get_comp_by_uid_resursive(parts[1:])

    def _foreach_comp_recursive(self, child_ns: str,
                                handler: Callable[[str, Component], None]):
        for k, v in self._child_comps.items():
            child_uid = f"{child_ns}.{k}"
            if isinstance(v, ContainerBase):
                handler(child_uid, v)
                v._foreach_comp_recursive(child_uid, handler)
            else:
                handler(child_uid, v)

    def _foreach_comp(self, handler: Callable[[str, Component], None]):
        assert self._flow_uid != "", "_flow_uid must be set before modify_comp"
        handler(self._flow_uid, self)
        self._foreach_comp_recursive(self._flow_uid, handler)

    def _update_uid(self):

        def handler(uid, v: Component):
            v._flow_uid = uid

        self._foreach_comp(handler)

    def _detach(self):
        disposed_uids: Dict[str, Component] = super()._detach()
        for v in self._child_comps.values():
            disposed_uids.update(v._detach())
        return disposed_uids

    def _detach_child(self, childs: Optional[List[str]] = None):
        disposed_uids: Dict[str, Component] = {}
        if childs is None:
            childs = list(self._child_comps.keys())
        for k in childs:
            v = self._child_comps[k]
            disposed_uids.update(v._detach())
            print(k, v, v._flow_reference_count, disposed_uids)

        return disposed_uids

    def _attach_child(self,
                      queue: asyncio.Queue,
                      childs: Optional[List[str]] = None):
        atached_uids: Dict[str, Component] = {}
        if childs is None:
            childs = list(self._child_comps.keys())
        for k in childs:
            v = self._child_comps[k]
            atached_uids.update(v._attach(f"{self._flow_uid}.{k}", queue))
        return atached_uids
    
    def _attach(self, uid: str, queue: asyncio.Queue):
        attached: Dict[str, Component] = super()._attach(uid, queue)
        for k, v in self._child_comps.items():
            attached.update(v._attach(f"{uid}.{k}", queue))
        return attached
    
    def _get_uid_to_comp_dict(self):
        res: Dict[str, Component] = {}

        def handler(uid, v: Component):
            res[uid] = v

        self._foreach_comp(handler)
        return res

    async def _clear(self):
        for c in self._child_comps:
            cc = self[c]
            await cc._clear()
        self._child_comps.clear()
        await super()._clear()
        self._pool.unique_set.clear()

    def __getitem__(self, key: str):
        assert key in self._child_comps, f"{key}, {self._child_comps.keys()}"
        return self._child_comps[key]

    def _get_all_nested_child_recursive(self, name: str, res: List[Component]):
        comp = self[name]
        res.append(comp)
        if isinstance(comp, ContainerBase):
            for child in comp._child_comps:
                comp._get_all_nested_child_recursive(child, res)

    def _get_all_nested_child(self, name: str):
        res: List[Component] = []
        self._get_all_nested_child_recursive(name, res)
        return res

    def _get_all_nested_childs(self, names: Optional[List[str]] = None):
        if names is None:
            names = list(self._child_comps.keys())
        comps: List[Component] = []
        for c in names:
            comps.extend(self._get_all_nested_child(c))
        return comps

    def _get_uid_with_ns(self, name: str):
        if self._flow_uid == "":
            return (f"{name}")
        return (f"{self._flow_uid}.{name}")

    def add_layout(self, layout: Union[Dict[str, Component], List[Component]]):
        return self.init_add_layout(layout)

    def init_add_layout(self, layout: Union[Dict[str, Component],
                                            List[Component]]):
        # TODO prevent call this in layout function
        """ {
            btn0: Button(...),
            box0: VBox({
                btn1: Button(...),
                ...
            }, flex...),
        }
        """
        if isinstance(layout, list):
            layout = {str(i): v for i, v in enumerate(layout)}
        # for k, v in layout.items():
        #     v._flow_name = k
        if self._prevent_add_layout:
            raise ValueError("you must init layout in app_create_layout")
        self._child_comps.update(layout)

    def get_props(self):
        state = super().get_props()
        state["childs"] = [self[n]._flow_uid for n in self._child_comps]
        return state
    
    async def __run_special_methods(self, attached: List[Component],
                                    detached: List[Component]):
        for attach in attached:
            special_methods = attach.get_special_methods()

            if special_methods.did_mount is not None:
                await self.run_callback(
                    special_methods.did_mount.get_binded_fn(),
                    sync_first=False,
                    change_status=False)
        for deleted in detached:
            special_methods = deleted.get_special_methods()
            if special_methods.will_unmount is not None:
                await self.run_callback(
                    special_methods.will_unmount.get_binded_fn(),
                    sync_first=False,
                    change_status=False)

    def set_new_layout_locally(self, layout: Dict[str, Component]):
        detached_uid_to_comp = self._detach_child()
        self._child_comps = layout
        attached = self._attach_child(self.queue)
        # update all childs of this component
        comps_frontend = {
            c._flow_uid: c
            for c in self._get_all_nested_childs()
        }
        comps_frontend[self._flow_uid] = self
        comps_frontend_dict = {
            k: v.to_dict()
            for k, v in comps_frontend.items()
        }
        return self.create_update_comp_event(
            comps_frontend_dict, list(detached_uid_to_comp.keys())), list(
                attached.values()), list(detached_uid_to_comp.values())

    async def set_new_layout(self, layout: Union[Dict[str, Component],
                                                 List[Component]]):
        if isinstance(layout, list):
            layout = {str(i): v for i, v in enumerate(layout)}
        new_ev, attached, removed = self.set_new_layout_locally(layout)
        for deleted in removed:
            await deleted._cancel_task()
        await self.__run_special_methods(attached, removed)
        await self.put_app_event(new_ev)

    async def remove_childs_by_keys(self, keys: List[str]):
        detached_uid_to_comp = self._detach_child(keys)
        for comp in detached_uid_to_comp.values():
            await comp._cancel_task()
        if not detached_uid_to_comp:
            return
        await self.__run_special_methods([],
                                         list(detached_uid_to_comp.values()))
        await self.put_app_event(
            self.create_delete_comp_event(list(detached_uid_to_comp.keys())))
        child_uids = [self[c]._flow_uid for c in self._child_comps]
        await self.put_app_event(
            self.create_update_event({"childs": child_uids}))

    def update_childs_locally(self, layout: Dict[str, Component]):
        intersect = set(layout.keys()).intersection(self._child_comps.keys())
        detached = self._detach_child(list(intersect))
        self._child_comps.update(layout)
        attached = self._attach_child(self.queue, list(layout.keys()))
        # remove replaced components first.
        comps_frontend = {
            c._flow_uid: c
            for c in self._get_all_nested_childs()
        }
        comps_frontend[self._flow_uid] = self
        comps_frontend_dict = {
            k: v.to_dict()
            for k, v in comps_frontend.items()
        }
        return self.create_update_comp_event(
            comps_frontend_dict, list(detached.keys())), list(
                attached.values()), list(detached.values())

    async def update_childs(self, layout: Dict[str, Component]):
        new_ev, attached, removed = self.update_childs_locally(layout)
        for deleted in removed:
            await deleted._cancel_task()
        await self.__run_special_methods(attached, removed)
        await self.put_app_event(new_ev)

    async def replace_childs(self, layout: Dict[str, Component]):
        for k in layout.keys():
            assert k in self._child_comps
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
                 children: Union[List[Component], Dict[str, Component]],
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.Fragment, FragmentProps, uid_to_comp, children,
                         inited)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def set_disabled(self, disabled: bool):
        await self.send_and_wait(self.update_event(disabled=disabled))


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


def create_ignore_usr_msg(comp: Component):
    msg = comp.create_user_msg_event((UserMessage.create_warning(
        comp._flow_uid, "UI Running",
        f"UI {comp._flow_uid}@{str(type(comp).__name__)} is still running, so ignore your control"
    )))
    return msg
