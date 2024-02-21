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
import asyncio
import builtins
import copy
import dataclasses
import enum
from functools import partial
import inspect
import io
import re
import sys
import threading
import traceback
from pathlib import Path
from typing import (Any, AsyncGenerator, Awaitable, Callable, Coroutine, Dict,
                    Generic, Iterable, List, Optional, Set, Tuple, Type,
                    Union, ClassVar, Dict, Protocol)

from typing_extensions import (Concatenate, ContextManager, Literal, ParamSpec,
                               Protocol, Self, TypeAlias, TypeVar)

from tensorpc.core.event_emitter.aio import AsyncIOEventEmitter
from pydantic import (
    BaseModel,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    TypeAdapter,
    ValidationError,
)
from pydantic_core import PydanticCustomError, core_schema

from tensorpc.core.core_io import JsonOnlyData
from tensorpc.core.event_emitter.base import ExceptionParam
from tensorpc.core.moduleid import is_tensorpc_dynamic_path
from tensorpc.core.serviceunit import (AppFuncType, ReloadableDynamicClass,
                                       ServFunctionMeta)
from tensorpc.core.tree_id import UniqueTreeId
from tensorpc.flow.coretypes import MessageLevel

from tensorpc.flow.flowapp.appcore import EventHandler, EventHandlers
from tensorpc.flow.flowapp.reload import AppReloadManager, FlowSpecialMethods
from tensorpc.utils.registry import HashableRegistry
from tensorpc.utils.uniquename import UniqueNamePool

from tensorpc.core import dataclass_dispatch as dataclasses_strict

from ..jsonlike import (BackendOnlyProp, DataClassWithUndefined, Undefined,
                        as_dict_no_undefined, asdict_no_deepcopy, camel_to_snake, snake_to_camel,
                        split_props_to_undefined, undefined, undefined_dict_factory)
from .appcore import SimpleEventType, NumberType, ValueType, get_app, Event, EventDataType
from tensorpc.flow.constants import TENSORPC_FLOW_COMP_UID_STRUCTURE_SPLIT


class DataclassType(Protocol):
    # as already noted in comments, checking for this attribute is currently
    # the most reliable way to ascertain that something is a dataclass
    __dataclass_fields__: ClassVar[Dict] 


ALL_APP_EVENTS = HashableRegistry()

_CORO_NONE = Union[Coroutine[None, None, None], None]

_CORO_ANY: TypeAlias = Union[Coroutine[None, None, Any], Any]


class NoDefault:
    pass


class AppComponentCore:

    def __init__(self, queue: asyncio.Queue,
                 reload_mgr: AppReloadManager) -> None:
        self.queue = queue
        self.reload_mgr = reload_mgr


# DON'T MODIFY THIS VALUE!!!
nodefault = NoDefault()


class UIType(enum.IntEnum):
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
    # TabContext = 0x18
    # Tab = 0x19
    # TabPanel = 0x1a
    # TabList = 0x1b
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
    # AllotmentPane = 0x29
    FlexLayout = 0x2a
    DynamicControls = 0x2b
    MonacoEditor = 0x2c
    Icon = 0x2d
    Markdown = 0x2e
    TextField = 0x2f
    DataGrid = 0x30
    Tabs = 0x31
    VirtualizedBox = 0x32
    DataFlexBox = 0x33
    JsonViewer = 0x34
    ListItemIcon = 0x35
    Link = 0x36
    BlenderSlider = 0x37
    SimpleControls = 0x38
    # this component have different state structure.
    TanstackJsonLikeTreeView = 0x39
    MenuList = 0x3a
    MatrixDataGrid = 0x3b

    GridLayout = 0x40

    # special
    TaskLoop = 0x100
    FlexBox = 0x101
    MUIList = 0x102
    Divider = 0x103
    AppTerminal = 0x104
    ThemeProvider = 0x105
    Handle = 0x106

    # special containers
    # react fragment
    Fragment = 0x200
    MatchCase = 0x201

    MASK_THREE = 0x1000
    MASK_THREE_GEOMETRY = 0x0100
    MASK_THREE_POST_PROCESS = 0x0200

    ThreeCanvas = 0x1000
    ThreePoints = 0x1001

    ThreePerspectiveCamera = 0x1002
    ThreeGroup = 0x1003
    ThreeOrthographicCamera = 0x1004

    ThreeFlex = 0x1005
    ThreeFlexItemBox = 0x1006
    ThreeHtml = 0x1007

    ThreeHud = 0x1008
    ThreeView = 0x1009

    ThreeMapControl = 0x1010
    ThreeOrbitControl = 0x1011
    ThreePointerLockControl = 0x1012
    ThreeFirstPersonControl = 0x1013
    ThreeTransformControl = 0x1014
    ThreeCameraControl = 0x1015
    ThreePivotControl = 0x1016

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

    ThreeScreenShot = 0x102f
    ThreePointLight = 0x1030
    ThreeDirectionalLight = 0x1031
    ThreeSpotLight = 0x1032

    ThreeAmbientLight = 0x1033
    ThreeHemisphereLight = 0x1034

    ThreePrimitiveMesh = 0x1035
    ThreeEdges = 0x1036
    ThreeBufferMesh = 0x1037
    ThreeVoxelMesh = 0x1038
    ThreeInstancedMesh = 0x1039
    ThreeSky = 0x103a
    ThreeEnvironment = 0x103b
    ThreeWireframe = 0x103c
    ThreeLightFormer = 0x103d
    ThreeAccumulativeShadows = 0x103e
    ThreeRandomizedLight = 0x103f
    ThreeURILoaderContext = 0x1040
    ThreeCubeCamera = 0x1041
    ThreeContactShadows = 0x1042
    ThreeGizmoHelper = 0x1043
    ThreeSelectionContext = 0x1044
    ThreeOutlines = 0x1045

    ThreeMeshBasicMaterial = 0x1050
    ThreeMeshStandardMaterial = 0x1051
    ThreeMeshLambertMaterial = 0x1052
    ThreeMeshMatcapMaterial = 0x1053
    ThreeMeshNormalMaterial = 0x1054
    ThreeMeshPhongMaterial = 0x1055
    ThreeMeshPhysicalMaterial = 0x1056
    ThreeMeshToonMaterial = 0x1057
    ThreeMeshDepthMaterial = 0x1058
    ThreeRawShaderMaterial = 0x1059
    ThreeMeshTransmissionMaterial = 0x105a
    ThreeMeshDiscardMaterial = 0x105b

    ThreeMeshShaderMaterial = 0x105c

    ThreeSimpleGeometry = 0x1101
    ThreeShape = 0x1102

    ThreeEffectComposer = 0x1200
    ThreeEffectOutline = 0x1201
    ThreeEffectBloom = 0x1202
    ThreeEffectDepthOfField = 0x1203
    ThreeEffectToneMapping = 0x1204

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


class AppEventType(enum.IntEnum):
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
    UIUpdateUsedEvents = 17
    # clipboard
    CopyToClipboard = 20
    InitLSPClient = 21
    # schedule event, won't be sent to frontend.
    ScheduleNext = 100
    # special UI event
    AppEditor = 200
    # send event to component, for append update
    # and uncontrolled component.
    ComponentEvent = 300


class FrontendEventType(enum.IntEnum):
    """type for all component events.
    
    event handled in handle_event use FrontendEventType.EventName.value,
    
    event handled in event_emitter use FrontendEventType.EventName.name,
    """
    # only used on backend
    # if user don't define DragCollect handler, Drop won't be scheduled.
    DragCollect = -1
    # file drop use special path to handle
    FileDrop = -2
    # emitted by event_emitter
    BeforeMount = -3
    # emitted by event_emitter
    BeforeUnmount = -4
    # emitted by DataGrid when data change. user can use this to save data item.
    DataItemChange = -5

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
    # modal close: include dialog and drawer (modal based).
    ModalClose = 23
    Drag = 24
    Drop = 25
    SelectNewItem = 26
    ContextMenuSelect = 28


    TreeLazyExpand = 30
    TreeItemSelectChange = 31

    TreeItemToggle = 32
    TreeItemFocus = 33
    TreeItemButton = 34
    TreeItemRename = 36
    TreeItemExpandChange = 37

    ComplexLayoutCloseTab = 40
    ComplexLayoutSelectTab = 41
    ComplexLayoutTabReload = 42
    ComplexLayoutSelectTabSet = 43
    ComplexLayoutStoreModel = 44

    EditorSave = 50
    EditorChange = 51
    EditorQueryState = 52
    EditorSaveState = 53
    EditorReady = 54

    # leaflet events
    MapZoom = 60
    MapMove = 61

    # data grid events
    DataGridRowSelection = 70
    DataGridFetchDetail = 71
    DataGridFetchInf = 72
    DataGridRowRangeChanged = 73
    DataGridProxyLazyLoadRange = 74

    PlotlyClickData = 100
    PlotlyClickAnnotation = 101


UI_TYPES_SUPPORT_DATACLASS: Set[UIType] = {
    UIType.DataGrid, UIType.MatchCase,
    UIType.DataFlexBox,
    UIType.Tabs, UIType.Allotment,
    UIType.GridLayout,
    UIType.MenuList,
    UIType.MatrixDataGrid
}

class AppDraggableType(enum.Enum):
    JsonLikeTreeItem = "JsonLikeTreeItem"


ALL_POINTER_EVENTS = [
    FrontendEventType.Down.value,
    FrontendEventType.Up.value,
    FrontendEventType.Move.value,
    FrontendEventType.Enter.value,
    FrontendEventType.Leave.value,
    FrontendEventType.Over.value,
    FrontendEventType.Out.value,
    FrontendEventType.Click.value,
    FrontendEventType.DoubleClick.value,
    FrontendEventType.ContextMenu.value,

]


class UIRunStatus(enum.IntEnum):
    Stop = 0
    Running = 1
    Pause = 2


class TaskLoopEvent(enum.IntEnum):
    Start = 0
    Stop = 1
    Pause = 2


class AppEditorEventType(enum.IntEnum):
    SetValue = 0
    RevealLine = 1


class AppEditorFrontendEventType(enum.IntEnum):
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

    def __init__(self, uid_to_data: Dict[str, SimpleEventType]) -> None:
        self.uid_to_data = uid_to_data

    def to_dict(self):
        return self.uid_to_data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(data)

    def merge_new(self, new):
        return new


@ALL_APP_EVENTS.register(key=AppEventType.UIUpdateUsedEvents.value)
class UpdateUsedEventsEvent:

    def __init__(self, uid_to_data: Dict[str, Any]) -> None:
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

    def __init__(self,
                 uid_to_data_undefined: Dict[str, Tuple[Dict[str, Any],
                                                        List[str]]],
                 json_only: bool = False) -> None:
        self.uid_to_data_undefined = uid_to_data_undefined
        self.json_only = json_only

    def as_json_only(self):
        return UIUpdateEvent(self.uid_to_data_undefined, True)

    def to_dict(self):
        if self.json_only:
            return JsonOnlyData(self.uid_to_data_undefined)
        else:
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

@ALL_APP_EVENTS.register(key=AppEventType.InitLSPClient.value)
class InitLSPClientEvent:

    def __init__(self, port: int, init_cfg: dict) -> None:
        self.port = port
        self.init_cfg = init_cfg

    def to_dict(self):
        return {
            "port": self.port,
            "initConfig": self.init_cfg
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(data["port"], data["initConfig"])

    def merge_new(self, new):
        assert isinstance(new, InitLSPClientEvent)
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
                        ComponentEvent, FrontendUIEvent, UpdateUsedEventsEvent,
                        InitLSPClientEvent]


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
                 event_id: str = "",
                 is_loopback: bool = False) -> None:
        self.uid = uid
        self.type_to_event = type_to_event
        # event that indicate this app event is sent
        # used for callback
        self.sent_event = sent_event
        self.event_id = event_id
        self.is_loopback = is_loopback

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


@dataclasses_strict.dataclass
class BasicProps(DataClassWithUndefined):
    status: int = UIRunStatus.Stop.value
    tensorpc_dynamic_eval: Union[Undefined, Dict[str, Any]] = undefined
    # used for template component
    override_props: Union[Dict[str, str], Undefined] = undefined

@dataclasses_strict.dataclass
class ContainerBaseProps(BasicProps):
    childs: List[str] = dataclasses_strict.field(default_factory=list)


T_base_props = TypeVar("T_base_props", bound=BasicProps)
T_container_props = TypeVar("T_container_props", bound=ContainerBaseProps)
T = TypeVar("T")
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
    is_dynamic_path = False
    try:
        path = inspect.getfile(builtins.type(obj))
        is_dynamic_path = is_tensorpc_dynamic_path(path)
        if is_dynamic_path:
            _flow_comp_def_path = path 
        else:
            _flow_comp_def_path = str(
                Path(inspect.getfile(builtins.type(obj))).resolve())
    except:
        traceback.print_exc()
        _flow_comp_def_path = ""
    if is_dynamic_path:
        return _flow_comp_def_path
    path = Path(_flow_comp_def_path)
    if not path.exists() or path.suffix != ".py":
        _flow_comp_def_path = ""
    return _flow_comp_def_path

class _EventSlot:
    def __init__(self, event_type: EventDataType, comp: "Component"):
        self.event_type = event_type
        self.comp = comp

    def on_standard(self, handler: Callable[[Event], Any]) -> "_EventSlot":
        """standard event means the handler must be a function with one argument of Event.
        this must be used to get template key
        if you use template layout such as table column def.
        """
        self.comp.register_event_handler(self.event_type, handler, simple_event=False)
        return self 
    
    def on(self, handler: Callable) -> "_EventSlot":
        """simple event means the event data isn't Event, but the data of Event, or none for no-arg event
        such as click.
        """
        self.comp.register_event_handler(self.event_type, handler, simple_event=True)
        return self 

    def off(self, handler: Callable) -> "_EventSlot":
        self.comp.remove_event_handler(self.event_type, handler)
        return self 

    def configure(self, stop_propagation: bool = False,
            throttle: Optional[NumberType] = None,
            debounce: Optional[NumberType] = None) -> "_EventSlot":
        self.comp.configure_event_handlers(self.event_type, stop_propagation,
                throttle, debounce)
        return self

class _EmitterEventSlot:
    # TODO remove this
    def __init__(self, event_type: EventDataType, emitter: "AsyncIOEventEmitter[EventDataType, Event]"):
        self.event_type = event_type
        self.emitter = emitter

    def on(self, handler: Callable[[Event], Any]) -> "_EmitterEventSlot":
        """simple event means the event data isn't Event, but the data of Event, or none for no-arg event
        such as click.
        """
        self.emitter.on(self.event_type, handler)
        return self 

    def off(self, handler: Callable) -> "_EmitterEventSlot":
        self.emitter.remove_listener(self.event_type, handler)
        return self 
    
T_child_structure = TypeVar("T_child_structure", default=Any, bound=DataclassType)

class _ComponentEffects:
    def __init__(self) -> None:
        self._flow_effects: Dict[str, List[Callable[[], Union[Callable[[], Any], None, Coroutine[None, None, Union[Callable[[], Any], None]]]]]] = {}
        self._flow_unmounted_effects: Dict[str, List[Callable[[], _CORO_NONE]]] = {}

    def use_effect(self, effect: Callable[[], Union[Optional[Callable[[], Any]], Coroutine[None, None, Optional[Callable[[], Any]]]]], key: str = ""):
        if key not in self._flow_effects:
            self._flow_effects[key] = []
            self._flow_unmounted_effects[key] = []

        self._flow_effects[key].append(effect)

    def has_effect_key(self, key: str):
        return key in self._flow_effects
    
    def remove_effect_key(self, key: str):
        self._flow_effects.pop(key)
        self._flow_unmounted_effects.pop(key)


class Component(Generic[T_base_props, T_child]):

    def __init__(self,
                 type: UIType,
                 prop_cls: Type[T_base_props],
                 allowed_events: Optional[Iterable[EventDataType]] = None,
                 uid: Optional[UniqueTreeId] = None,
                 json_only: bool = False) -> None:
        self._flow_comp_core: Optional[AppComponentCore] = None
        self._flow_uid: Optional[UniqueTreeId] = uid
        self._flow_comp_type = type
        # self._status = UIRunStatus.Stop
        # task for callback of controls
        # if previous control callback hasn't finished yet,
        # the new control event will be IGNORED
        self._task: Optional[asyncio.Task] = None
        self._parent = ""
        self.__props = prop_cls()
        self.__prop_cls = prop_cls
        self._prop_validator = TypeAdapter(self.__prop_cls)
        self._prop_field_names: Set[str] = set([x.name for x in dataclasses.fields(prop_cls)])
        self._mounted_override = False
        self.__sx_props: Dict[str, Any] = {}
        self._flow_allowed_events: Set[EventDataType] = set([FrontendEventType.BeforeMount.value, FrontendEventType.BeforeUnmount.value])
        if allowed_events is not None:
            self._flow_allowed_events.update(allowed_events)
        self._flow_user_datas: List[Any] = []
        self._flow_comp_def_path = _get_obj_def_path(self)
        self._flow_reference_count = 0
        # tensorpc will scan your prop dict to find
        # np.ndarray and bytes by default.
        # this will cost time in deep and large json, if you use
        # json_only, this scan will be skipped.
        self._flow_json_only = json_only

        self.effects = _ComponentEffects()
        self._flow_unmount_effect_objects: List[Callable[[], _CORO_NONE]] = []

        self._flow_event_context_creator: Optional[Callable[
            [], ContextManager]] = None
        # flow event handlers is used for frontend events
        self._flow_event_handlers: Dict[EventDataType, EventHandlers] = {}
        # event emitter is used for backend events, e.g. mount, unmount
        self._flow_event_emitter: AsyncIOEventEmitter[EventDataType, Event] = AsyncIOEventEmitter()
        self._flow_event_emitter.add_exception_listener(self.__event_emitter_on_exc)
        self.event_before_mount = self._create_emitter_event_slot(FrontendEventType.BeforeMount)
        self.event_before_unmount = self._create_emitter_event_slot(FrontendEventType.BeforeUnmount)
    
    def use_effect(self, effect: Callable[[], Union[Optional[Callable[[], Any]], Coroutine[None, None, Optional[Callable[[], Any]]]]], key: str = ""):
        return self.effects.use_effect(effect, key)

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any, _handler: GetCoreSchemaHandler):
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.any_schema(),
        )

    @classmethod
    def validate(cls, v):
        if not isinstance(v, Component):
            raise ValueError('Component required')
        return v

    def _create_event_slot(self, event_type: Union[FrontendEventType, EventDataType]):
        if isinstance(event_type, FrontendEventType):
            event_type_value = event_type.value
        else:
            event_type_value = event_type
        return _EventSlot(event_type_value, self)
    
    def _create_emitter_event_slot(self, event_type: Union[FrontendEventType, EventDataType]):
        if isinstance(event_type, FrontendEventType):
            event_type_value = event_type.value
            assert event_type.value < 0, "only support backend events"
            return _EmitterEventSlot(event_type_value, self._flow_event_emitter)
        else:
            event_type_value = event_type
        return _EmitterEventSlot(event_type_value, self._flow_event_emitter)

    @property
    def flow_event_emitter(self) -> AsyncIOEventEmitter[EventDataType, Event]:
        return self._flow_event_emitter

    def get_special_methods(self, reload_mgr: AppReloadManager):
        metas = reload_mgr.query_type_method_meta(type(self), no_code=True)
        # copy here to avoid different obj bind same meta.
        metas = [x.copy() for x in metas]
        res = FlowSpecialMethods(metas)
        res.bind(self)
        return res

    def set_flow_event_context_creator(
            self, context_creator: Optional[Callable[[], ContextManager]]):
        """set a context which will be entered before event handler is called
        """
        self._flow_event_context_creator = context_creator

    @property
    def props(self) -> T_base_props:
        return self.__props

    @property
    def _flow_uid_encoded(self) -> str:
        assert self._flow_uid is not None 
        return self._flow_uid.uid_encoded

    @property
    def propcls(self) -> Type[T_base_props]:
        return self.__prop_cls

    def merge_prop(self, prop: T_base_props):
        assert type(prop) == type(self.__props)
        prop_dict = prop.get_dict()
        for k, v in prop_dict.items():
            setattr(self.__props, k, v)

    def _attach(self, uid: UniqueTreeId, comp_core: AppComponentCore) -> dict:
        if self._flow_reference_count == 0:
            self._flow_uid = uid
            self._flow_comp_core = comp_core
            self._flow_reference_count += 1
            self.flow_event_emitter.emit(FrontendEventType.BeforeMount.value, Event(FrontendEventType.BeforeMount.value, None))
            return {uid: self}
        self._flow_reference_count += 1
        return {}

    def _detach(self) -> Dict[UniqueTreeId, "Component"]:
        self._flow_reference_count -= 1
        if self._flow_reference_count == 0:
            self.flow_event_emitter.emit(FrontendEventType.BeforeUnmount.value, Event(FrontendEventType.BeforeUnmount.value, None))
            res_uid = self._flow_uid
            assert res_uid is not None 
            self._flow_uid = None
            self._flow_comp_core = None
            return {res_uid: self}
        return {}

    def is_mounted(self):
        return self._flow_comp_core is not None

    def _prop_base(self, prop: Callable[P, Any], this: T3) -> Callable[P, T3]:
        """set prop by keyword arguments
        this function is used to provide intellisense result for all props.
        """
        def wrapper(*args: P.args, **kwargs: P.kwargs):
            # do validation on changed props only
            # self.__prop_cls(**kwargs)
            # TypeAdapter(self.__prop_cls).validate_python(kwargs)
            for k, v in kwargs.items():
                setattr(self.__props, k, v)
            # do validation for all props (call model validator)
            self._prop_validator.validate_python(self.__props)
            return this
        return wrapper

    def _update_props_base(self,
                           prop: Callable[P, Any],
                           json_only: bool = False):
        """create prop update event by keyword arguments
        this function is used to provide intellisense result for all props.
        """
        def wrapper(*args: P.args, **kwargs: P.kwargs):
            # do validation on changed props only
            # self.__prop_cls(**kwargs)
            # TypeAdapter(self.__prop_cls).validate_python(kwargs)
            for k, v in kwargs.items():
                setattr(self.__props, k, v)
            # do validation for all props (call model validator)
            self._prop_validator.validate_python(self.__props)
            return self.create_update_event(kwargs, json_only)

        return wrapper

    async def handle_event(self, ev: Event, is_sync: bool = False):
        pass
    
    def __repr__(self):
        res = f"{self.__class__.__name__}({self._flow_uid_encoded})"
        # if self._flow_user_data is not None:
        #     res += f"({self._flow_user_data})"
        return res 

    def find_user_meta_by_type(self, type: Type[T]) -> Optional[T]:
        for x in self._flow_user_datas:
            if isinstance(x, type):
                return x
        return None 
    
    def set_user_meta_by_type(self, obj: Any):
        obj_type = type(obj)
        for i, x in self._flow_user_datas:
            if isinstance(x, obj_type):
                self._flow_user_datas[i] = obj
                return self
        self._flow_user_datas.append(obj)
        return self

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

    def update_sx_props(self, sx_props: Dict[str, Any]):
        self.__sx_props.update(sx_props)
        return self
    
    def get_sx_props(self):
        return self.__sx_props

    def to_dict(self):
        """undefined will be removed here.
        if you reimplement to_dict, you need to use 
        camel name, no conversion provided.
        """
        props = self.get_props()
        props, und = split_props_to_undefined(props)
        props.update(as_dict_no_undefined(self.__sx_props))
        res = {
            "uid": self._flow_uid_encoded,
            "type": self._flow_comp_type.value,
            "props": props,
        }
        evs = self._get_used_events_dict()
        if evs:
            res["props"]["usedEvents"] = evs
        if self._flow_json_only:
            res["props"] = JsonOnlyData(props)
        return res

    def _get_used_events_dict(self):
        evs = []
        for k, v in self._flow_event_handlers.items():
            if not isinstance(v, Undefined) and not v.backend_only and v.handlers:
                d = v.to_dict()
                d["type"] = k
                evs.append(d)
        return evs

    def _to_dict_with_sync_props(self):
        props = self.get_sync_props()
        props, und = split_props_to_undefined(props)
        res = {
            "uid": self._flow_uid_encoded,
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
        return self.__props.get_dict(dict_factory=_undefined_comp_dict_factory, obj_factory=_undefined_comp_obj_factory)  # type: ignore

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

    async def put_loopback_ui_event(self, ev: SimpleEventType):
        if self.is_mounted():
            assert self._flow_uid is not None 
            return await self.queue.put(
                AppEvent("",
                         {AppEventType.UIEvent: UIEvent({self._flow_uid.uid_encoded: ev})},
                         is_loopback=True))

    async def put_app_event(self, ev: AppEvent):
        if self.is_mounted():
            return await self.queue.put(ev)

    def set_override_props(self, **kwargs: str):
        for k in kwargs.keys():
            assert k in self._prop_field_names, f"overrided prop must be defined in props class, {k}"
        new_kwargs: Dict[str, str] = {}
        for k, v in kwargs.items():
            new_kwargs[snake_to_camel(k)] = v
        if isinstance(self.props.override_props, Undefined):
            self.props.override_props = new_kwargs
        else:
            self.props.override_props.update(new_kwargs)
        return self

    def set_override_props_unchecked(self, **kwargs: str):
        new_kwargs: Dict[str, str] = {}
        for k, v in kwargs.items():
            new_kwargs[snake_to_camel(k)] = v
        if isinstance(self.props.override_props, Undefined):
            self.props.override_props = new_kwargs
        else:
            self.props.override_props.update(new_kwargs)
        return self
    
    def set_override_props_unchecked_dict(self, kwargs: Dict[str, str]):
        new_kwargs: Dict[str, str] = {}
        for k, v in kwargs.items():
            new_kwargs[snake_to_camel(k)] = v
        if isinstance(self.props.override_props, Undefined):
            self.props.override_props = new_kwargs
        else:
            self.props.override_props.update(new_kwargs)
        return self

    @property
    def queue(self):
        assert self._flow_comp_core is not None, f"you must add ui by flexbox.add_xxx"
        return self._flow_comp_core.queue

    @property
    def flow_app_comp_core(self):
        assert self._flow_comp_core is not None, f"you must add ui by flexbox.add_xxx"
        return self._flow_comp_core

    def configure_event_handlers(self, 
            type: Union[FrontendEventType, EventDataType],
            stop_propagation: bool = False,
            throttle: Optional[NumberType] = None,
            debounce: Optional[NumberType] = None,
            backend_only: bool = False):
        if isinstance(type, FrontendEventType):
            type_value = type.value
        else:
            type_value = type
        if type_value not in self._flow_event_handlers:
            self._flow_event_handlers[type_value] = EventHandlers([])
        handlers = self._flow_event_handlers[type_value]
        handlers.stop_propagation = stop_propagation
        handlers.throttle = throttle
        handlers.debounce = debounce
        handlers.backend_only = backend_only
        return 

    
    def register_event_handler(self,
                               type: Union[FrontendEventType, EventDataType],
                               cb: Callable,
                               stop_propagation: bool = False,
                               throttle: Optional[NumberType] = None,
                               debounce: Optional[NumberType] = None,
                               backend_only: bool = False,
                               simple_event: bool = True):
        if self._flow_allowed_events:
            if not backend_only:
                assert type in self._flow_allowed_events, f"only support events: {self._flow_allowed_events}"
        
        evh = EventHandler(cb, simple_event)
        if isinstance(type, FrontendEventType):
            type_value = type.value
        else:
            type_value = type
        if type_value not in self._flow_event_handlers:
            self._flow_event_handlers[type_value] = EventHandlers([])
        handlers = self._flow_event_handlers[type_value]
        if type == FrontendEventType.DragCollect:
            assert len(handlers.handlers) == 0, "DragCollect only support one handler"
        self.configure_event_handlers(type_value, stop_propagation, throttle, debounce, backend_only)
        handlers.handlers.append(evh)
        # self._flow_event_handlers[type_value] = evh
        # if once:
        #     self._flow_event_emitter.once(type_value, self.handle_event)
        # else:
        #     self._flow_event_emitter.once(type_value, self.handle_event)
        return evh
    
    def remove_event_handler(self, type: EventDataType, handler: Callable):
        if type in self._flow_event_handlers:
            return self._flow_event_handlers[type].remove_handler(handler)
        return False

    def remove_event_handlers(self, type: EventDataType):
        if type in self._flow_event_handlers:
            del self._flow_event_handlers[type]
            return True
        return False

    def clear_event_handlers(self):
        self._flow_event_handlers.clear()

    def get_event_handlers(self, type: EventDataType):
        res = self._flow_event_handlers.get(type)
        if isinstance(res, Undefined):
            res = None
        return res

    def state_change_callback(self,
                              data: Any,
                              type: ValueType = FrontendEventType.Change.value
                              ):
        pass

    def create_update_event(self,
                            data: Dict[str, Union[Any, Undefined]],
                            json_only: bool = False,
                            validate: bool = False):
        if validate:
            self.__prop_cls(**data)
        data_no_und = {}
        data_unds = []
        for k, v in data.items():
            # k = snake_to_camel(k)
            if isinstance(v, Undefined):
                data_unds.append(k)
            else:
                data_no_und[k] = as_dict_no_undefined(v)
        assert self._flow_uid is not None 
        ev = UIUpdateEvent({self._flow_uid.uid_encoded: (data_no_und, data_unds)},
                           json_only)
        # uid is set in flowapp service later.
        return AppEvent("", {AppEventType.UIUpdateEvent: ev})

    def create_update_used_events_event(self):
        used_events = self._get_used_events_dict()
        return self.create_update_event({"usedEvents": used_events}, True)

    async def sync_used_events(self):
        return await self.put_app_event(self.create_update_used_events_event())

    def create_update_prop_event(self, data: Dict[str, Union[Any, Undefined]]):
        data_no_und = {}
        data_unds = []
        for k, v in data.items():
            k = snake_to_camel(k)
            if isinstance(v, Undefined):
                data_unds.append(k)
            else:
                data_no_und[k] = v
        assert self._flow_uid is not None 
        ev = UIUpdateEvent({self._flow_uid.uid_encoded: (data_no_und, data_unds)})
        # uid is set in flowapp service later.
        return AppEvent("", {AppEventType.UIUpdatePropsEvent: ev})

    def create_comp_event(self, data: Dict[str, Any]):
        """create component control event for
        backend -> frontend direct communication
        """
        assert self._flow_uid is not None 
        ev = ComponentEvent({self._flow_uid.uid_encoded: as_dict_no_undefined(data)})
        # uid is set in flowapp service later.
        return AppEvent("", {AppEventType.ComponentEvent: ev})

    async def send_and_wait(self, ev: AppEvent, wait: bool = True):
        if ev.sent_event is None:
            ev.sent_event = asyncio.Event()
        await self.put_app_event(ev)
        if self.is_mounted():
            if wait:
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

    async def __event_emitter_on_exc(self, exc_param: ExceptionParam):
        traceback.print_exc()
        e = exc_param.exc
        ss = io.StringIO()
        traceback.print_exc(file=ss)
        assert self._flow_uid is not None 
        user_exc = UserMessage.create_error(self._flow_uid.uid_encoded, repr(e),
                                            ss.getvalue())
        await self.put_app_event(self.create_user_msg_event(user_exc))
        app = get_app()
        if app._flowapp_enable_exception_inspect:
            await app._inspect_exception()


    async def run_callback(self,
                           cb: Callable[[], _CORO_ANY],
                           sync_state: bool = False,
                           sync_status_first: bool = False,
                           res_callback: Optional[Callable[[Any],
                                                           _CORO_ANY]] = None,
                           change_status: bool = True) -> Optional[Any]:
        """
        Runs the given callback function and handles its result and potential exceptions.

        Args:
            cb: The callback function to run.
            sync_state: Whether to synchronize the component's state before and after running the callback.
                this is required for components which can change state
                in frontend, e.g. switch, slider, etc. for components that
                won't interact with user in frontend, this can be set to False.
            sync_status_first: Whether to wait for the component's state to be synchronized before running the callback.
                should be used for components with loading support. e.g. buttons
            res_callback: An optional callback function to run with the result of the main callback.
            change_status: Whether to change the component's status to "Running" before running the callback and to "Stop" after.

        Returns:
            The result of the main callback function.

        Raises:
            Any exception raised by the main callback function.
        """

        if change_status:
            self.props.status = UIRunStatus.Running.value
        # only ui with loading support need sync first.
        # otherwise don't use this because slow
        if sync_status_first:
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
            assert self._flow_uid is not None 

            user_exc = UserMessage.create_error(self._flow_uid.uid_encoded, repr(e),
                                                ss.getvalue())
            await self.put_app_event(self.create_user_msg_event(user_exc))
            app = get_app()
            if app._flowapp_enable_exception_inspect:
                await app._inspect_exception()
        finally:
            if change_status:
                self.props.status = UIRunStatus.Stop.value
                await self.sync_status(sync_state)
        return res
    
    async def run_callbacks(self,
                           cbs: List[Callable[[], _CORO_NONE]],
                           sync_state: bool = False,
                           sync_status_first: bool = False,
                           res_callback: Optional[Callable[[Any],
                                                           _CORO_NONE]] = None,
                           change_status: bool = True):
        """
        Runs the given callback function and handles its result and potential exceptions.

        Args:
            cbs: The callback functions to run.
            sync_state: Whether to synchronize the component's state before and after running the callback.
                this is required for components which can change state
                in frontend, e.g. switch, slider, etc. for components that
                won't interact with user in frontend, this can be set to False.
            sync_status_first: Whether to wait for the component's state to be synchronized before running the callback.
                should be used for components with loading support. e.g. buttons
            res_callback: An optional callback function to run with the result of the main callback.
            change_status: Whether to change the component's status to "Running" before running the callback and to "Stop" after.

        Returns:
            The result of the main callback function.

        Raises:
            Any exception raised by the main callback function.
        """

        if change_status:
            self.props.status = UIRunStatus.Running.value
        # only ui with loading support need sync first.
        # otherwise don't use this because slow
        if sync_status_first:
            ev = asyncio.Event()
            await self.sync_status(sync_state, ev)
            await ev.wait()
        res = None
        for cb in cbs:
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
                assert self._flow_uid is not None 
                user_exc = UserMessage.create_error(self._flow_uid.uid_encoded, repr(e),
                                                    ss.getvalue())
                await self.put_app_event(self.create_user_msg_event(user_exc))
                app = get_app()
                if app._flowapp_enable_exception_inspect:
                    await app._inspect_exception()
        # finally:
        if change_status:
            self.props.status = UIRunStatus.Stop.value
            await self.sync_status(sync_state)
        else:
            if sync_state:
                await self.sync_state()
        return res

    async def sync_status(self,
                          sync_state: bool = False,
                          sent_event: Optional[asyncio.Event] = None):
        if sync_state:
            sync_props = self.get_sync_props()
            if sync_props:
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

class ForEachResult(enum.Enum):
    Continue = 0
    Return = 1

def _find_comps_in_dc(obj):
    "(list[tuple[str, Any]]) -> dict[str, Any]"
    
    """same as dataclasses.asdict except that this function
    won't recurse into nested container.
    """
    res_comp_localids: List[Tuple[Component, str]] = []
    if not dataclasses.is_dataclass(obj):
        raise TypeError("asdict() should be called on dataclass instances")
    _find_comps_in_dc_inner(obj, res_comp_localids, "")
    return res_comp_localids

def _find_comps_in_dc_inner(obj, res_comp_localids: List[Tuple[Component, str]], comp_local_id: str):
    if comp_local_id == "":
        local_id_prefix = ""
    else:
        local_id_prefix = f"{comp_local_id}{TENSORPC_FLOW_COMP_UID_STRUCTURE_SPLIT}"
    if isinstance(obj, Component):
        res_comp_localids.append((obj, comp_local_id))
        return
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            local_id = local_id_prefix + f.name
            _find_comps_in_dc_inner(getattr(obj, f.name), res_comp_localids, local_id)
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
        return type(obj)(*[_find_comps_in_dc_inner(v, res_comp_localids, local_id_prefix + str(i)) for i, v in enumerate(obj)])
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        return type(obj)(_find_comps_in_dc_inner(v, res_comp_localids, local_id_prefix + str(i)) for i, v in enumerate(obj))
    elif isinstance(obj, dict):
        # TODO validate that all keys are number or letters
        for k in obj.keys():
            assert isinstance(k, str) and k.isalnum(), f"key {k} must be string and alphanumeric"
        return type(obj)((k,
                          _find_comps_in_dc_inner(v, res_comp_localids, local_id_prefix + k))
                         for k, v in obj.items())

def _undefined_comp_dict_factory(x: List[Tuple[str, Any]]):
    res: Dict[str, Any] = {}
    for k, v in x:
        if isinstance(v, Component):
            assert v.is_mounted(), f"you must ensure component is inside comp tree if you add it to props, {k}, {type(v)}"
            res[k] = v._flow_uid_encoded
        elif isinstance(v, UniqueTreeId):
            res[k] = v.uid_encoded
        elif not isinstance(v, (Undefined, BackendOnlyProp)):
            res[k] = v
    return res

def _undefined_comp_obj_factory(x: Any):
    if isinstance(x, Component):
        assert x.is_mounted(), f"you must ensure component is inside comp tree if you add it to props, {type(x)}"
        return x._flow_uid_encoded
    return x

class ContainerBase(Component[T_container_props, T_child]):

    def __init__(self,
                 base_type: UIType,
                 prop_cls: Type[T_container_props],
                 _children: Optional[Union[Dict[str, T_child], DataclassType]] = None,
                 inited: bool = False,
                 allowed_events: Optional[Iterable[EventDataType]] = None,
                 uid: Optional[UniqueTreeId] = None,
                 app_comp_core: Optional[AppComponentCore] = None) -> None:
        super().__init__(base_type, prop_cls, allowed_events, uid)
        self._flow_comp_core = app_comp_core
        if inited:
            assert app_comp_core is not None  # and uid_to_comp is not None
        self._pool = UniqueNamePool()
        if _children is None:
            _children = {}
        # self._children = _children
        self._child_comps: Dict[str, Component] = {}
        self._child_structure: Optional[DataclassType] = None

        if isinstance(_children, dict):
            for k, v in _children.items():
                # assert k.isalnum(), "child key must be alphanumeric"
                # TODO check uid key is valid, can only contains number and letter
                assert isinstance(v, Component)
                self._child_comps[k] = v
        else:
            assert base_type in UI_TYPES_SUPPORT_DATACLASS
            assert dataclasses.is_dataclass(_children)
            # parse dataclass, get components, save structure
            self._child_structure = _children
            children_dict = _find_comps_in_dc(_children)
            for comp, local_id in children_dict:
                self._child_comps[local_id] = comp

        # self.props.childs: List[str] = []
        self.inited = inited
        self._prevent_add_layout = False
        
    def __repr__(self):
        res = super().__repr__()
        if self._child_comps:
            res += f"({','.join(self._child_comps.keys())})"
        return res 

    def _get_comp_by_uid(self, uid: str):
        uid_obj = UniqueTreeId(uid)
        parts = uid_obj.parts
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

    def _get_comps_by_uid(self, uid: str):
        parts = uid.split(".")
        # uid contains root, remove it at first.
        return [self] + self._get_comps_by_uid_resursive(parts[1:])

    def _get_comps_by_uid_resursive(self, parts: List[str]) -> List[Component]:
        key = parts[0]
        assert key in self._child_comps
        child_comp = self._child_comps[key]
        if len(parts) == 1:
            return [child_comp]
        else:
            assert isinstance(child_comp, ContainerBase)
            return [child_comp] + child_comp._get_comps_by_uid_resursive(
                parts[1:])

    def _foreach_comp_recursive(self, child_ns: UniqueTreeId,
                                handler: Callable[[UniqueTreeId, Component],
                                                  Union[ForEachResult, None]]):
        res_foreach: List[Tuple[UniqueTreeId, ContainerBase]] = []
        for k, v in self._child_comps.items():
            child_uid = child_ns.append_part(k)
            if isinstance(v, ContainerBase):
                res = handler(child_uid, v)
                if res is None:
                    res_foreach.append((child_uid, v))
                elif res == ForEachResult.Continue:
                    continue 
                elif res == ForEachResult.Return:
                    return 
            else:
                res = handler(child_uid, v)
                if res == ForEachResult.Continue:
                    continue 
                elif res == ForEachResult.Return:
                    return 
        for child_uid, v in res_foreach:
            v._foreach_comp_recursive(child_uid, handler)

    def _foreach_comp(self, handler: Callable[[UniqueTreeId, Component], Union[ForEachResult,
                                                                      None]]):
        assert self._flow_uid is not None, f"_flow_uid must be set before modify_comp, {type(self)}, {self._flow_reference_count}, {id(self)}"
        handler(self._flow_uid, self)
        self._foreach_comp_recursive(self._flow_uid, handler)

    def _update_uid(self):

        def handler(uid: UniqueTreeId, v: Component):
            v._flow_uid = uid

        self._foreach_comp(handler)

    def _detach(self):
        disposed_uids: Dict[UniqueTreeId, Component] = super()._detach()
        for v in self._child_comps.values():
            disposed_uids.update(v._detach())
        return disposed_uids

    def _detach_child(self, childs: Optional[List[str]] = None):
        disposed_uids: Dict[UniqueTreeId, Component] = {}
        if childs is None:
            childs = list(self._child_comps.keys())
        for k in childs:
            v = self._child_comps[k]
            disposed_uids.update(v._detach())
        return disposed_uids

    def _attach_child(self,
                      comp_core: AppComponentCore,
                      childs: Optional[List[str]] = None):
        atached_uids: Dict[str, Component] = {}
        assert self._flow_uid is not None 
        if childs is None:
            childs = list(self._child_comps.keys())
        for k in childs:
            v = self._child_comps[k]
            atached_uids.update(v._attach(self._flow_uid.append_part(k), comp_core))
        return atached_uids

    def _attach(self, uid: UniqueTreeId, comp_core: AppComponentCore):
        attached: Dict[str, Component] = super()._attach(uid, comp_core)
        assert self._flow_uid is not None 
        for k, v in self._child_comps.items():
            attached.update(v._attach(self._flow_uid.append_part(k), comp_core))
        return attached

    def _get_uid_encoded_to_comp_dict(self):
        res: Dict[str, Component] = {}

        def handler(uid: UniqueTreeId, v: Component):
            res[uid.uid_encoded] = v

        self._foreach_comp(handler)
        return res

    def _get_uid_to_comp_dict(self):
        res: Dict[UniqueTreeId, Component] = {}

        def handler(uid: UniqueTreeId, v: Component):
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

    def __contains__(self, key: str):
        return key in self._child_comps

    def __len__(self):
        return len(self._child_comps)

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

    def add_layout(self, layout: Union[Dict[str, Component], List[Component]]):
        return self.init_add_layout(layout)

    def __check_child_structure_is_none(self):
        assert self._child_structure is None, "you can only use set_layout or init to specify child with structure"

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
        self.__check_child_structure_is_none()
        if isinstance(layout, list):
            layout = {str(i): v for i, v in enumerate(layout)}
        # for k, v in layout.items():
        #     v._flow_name = k
        if self._prevent_add_layout:
            raise ValueError("you must init layout in app_create_layout")
        self._child_comps.update(layout)

    def get_props(self):
        state = super().get_props()
        state["childs"] = [self[n]._flow_uid_encoded for n in self._child_comps]
        if self._child_structure is not None:
            state["childsComplex"] = asdict_no_deepcopy(self._child_structure, dict_factory=_undefined_comp_dict_factory, obj_factory=_undefined_comp_obj_factory)
        return state

    async def _run_special_methods(
            self,
            attached: List[Component],
            detached: List[Component],
            reload_mgr: Optional[AppReloadManager] = None):
        if reload_mgr is None:
            reload_mgr = self.flow_app_comp_core.reload_mgr

        for deleted in detached:
            special_methods = deleted.get_special_methods(reload_mgr)
            if special_methods.will_unmount is not None:
                await self.run_callback(
                    special_methods.will_unmount.get_binded_fn(),
                    sync_status_first=False,
                    change_status=False)
            for k, unmount_effects in deleted.effects._flow_unmounted_effects.items():
                for unmount_effect in unmount_effects:
                    await self.run_callback(
                        unmount_effect,
                        sync_status_first=False,
                        change_status=False)
                unmount_effects.clear()
        for attach in attached:
            special_methods = attach.get_special_methods(reload_mgr)
            if special_methods.did_mount is not None:
                await self.run_callback(
                    special_methods.did_mount.get_binded_fn(),
                    sync_status_first=False,
                    change_status=False)
            # run effects 
            for k, effects in attach.effects._flow_effects.items():
                for effect in effects:
                    res = await self.run_callback(
                        effect,
                        sync_status_first=False,
                        change_status=False)
                    if res is not None:
                        # res is effect
                        attach.effects._flow_unmounted_effects[k].append(res)

    def set_new_layout_locally(self, layout: Union[Dict[str, Component], T_child_structure]):
        detached_uid_to_comp = self._detach_child()
        if isinstance(layout, dict):
            self._child_comps = layout
        else:
            assert dataclasses.is_dataclass(layout)
            assert type(layout) == type(self._child_structure), f"{type(layout)}, {type(self._child_structure)}"
            self._child_comps.clear()
            # parse dataclass, get components, save structure
            self._child_structure = layout
            children_dict = _find_comps_in_dc(layout)
            for comp, local_id in children_dict:
                self._child_comps[local_id] = comp
        attached = self._attach_child(self.flow_app_comp_core)
        # update all childs of this component
        comps_frontend = {
            c._flow_uid_encoded: c
            for c in self._get_all_nested_childs()
        }
        comps_frontend_dict = {
            k: v.to_dict()
            for k, v in comps_frontend.items()
        }
        child_uids = [self[c]._flow_uid_encoded for c in self._child_comps]
        update_msg: Dict[str, Any] = {
            "childs": child_uids
        }
        if self._child_structure is not None:
            update_msg["childsComplex"] = asdict_no_deepcopy(self._child_structure, dict_factory=_undefined_comp_dict_factory, obj_factory=_undefined_comp_obj_factory)
        update_ev = self.create_update_event(update_msg)
        deleted = [x.uid_encoded for x in detached_uid_to_comp.keys()]
        return update_ev + self.create_update_comp_event(
            comps_frontend_dict, deleted), list(
                attached.values()), list(detached_uid_to_comp.values())

    async def set_new_layout(self, layout: Union[Dict[str, Component],
                                                 List[Component],
                                                 T_child_structure]):
        if isinstance(layout, list):
            layout = {str(i): v for i, v in enumerate(layout)}
        new_ev, attached, removed = self.set_new_layout_locally(layout)
        for deleted in removed:
            await deleted._cancel_task()
        await self.put_app_event(new_ev)
        await self._run_special_methods(attached, removed)

    async def remove_childs_by_keys(self, keys: List[str]):
        self.__check_child_structure_is_none()
        detached_uid_to_comp = self._detach_child(keys)
        for k, comp in detached_uid_to_comp.items():
            await comp._cancel_task()
        for k in keys:
            self._child_comps.pop(k)
        if not detached_uid_to_comp:
            return
        deleted = [x.uid_encoded for x in detached_uid_to_comp.keys()]

        await self.put_app_event(
            self.create_delete_comp_event(deleted))
        await self._run_special_methods([],
                                        list(detached_uid_to_comp.values()))

    def update_childs_complex_event(self):
        update_msg: Dict[str, Any] = {}
        update_msg["childsComplex"] = asdict_no_deepcopy(self._child_structure, dict_factory=_undefined_comp_dict_factory, obj_factory=_undefined_comp_obj_factory)
        update_ev = self.create_update_event(update_msg)
        return update_ev
    
    async def update_childs_complex(self):
        await self.send_and_wait(self.update_childs_complex_event())

    def update_childs_locally(self, layout: Dict[str, Component]):
        self.__check_child_structure_is_none()
        intersect = set(layout.keys()).intersection(self._child_comps.keys())
        detached = self._detach_child(list(intersect))
        self._child_comps.update(layout)
        attached = self._attach_child(self.flow_app_comp_core,
                                      list(layout.keys()))
        # remove replaced components first.
        comps_frontend = {
            c._flow_uid_encoded: c
            for c in self._get_all_nested_childs(list(layout.keys()))
        }
        comps_frontend_dict = {
            k: v.to_dict()
            for k, v in comps_frontend.items()
        }
        child_uids = [self[c]._flow_uid_encoded for c in self._child_comps]
        update_msg: Dict[str, Any] = {
            "childs": child_uids
        }
        if self._child_structure is not None:
            update_msg["childsComplex"] = asdict_no_deepcopy(self._child_structure, dict_factory=_undefined_comp_dict_factory, obj_factory=_undefined_comp_obj_factory)
        update_ev = self.create_update_event(update_msg)
        deleted = [x.uid_encoded for x in detached.keys()]

        return update_ev + self.create_update_comp_event(
            comps_frontend_dict, deleted), list(
                attached.values()), list(detached.values())

    async def update_childs(self, layout: Union[Dict[str, Component],
                                                 List[Component]]):
        if isinstance(layout, list):
            layout = {str(i): v for i, v in enumerate(layout)}
        self.__check_child_structure_is_none()
        new_ev, attached, removed = self.update_childs_locally(layout)
        for deleted in removed:
            await deleted._cancel_task()
        await self.put_app_event(new_ev)
        await self._run_special_methods(attached, removed)

    async def replace_childs(self, layout: Dict[str, Component]):
        self.__check_child_structure_is_none()
        for k in layout.keys():
            assert k in self._child_comps
        return await self.update_childs(layout)


@dataclasses_strict.dataclass
class FragmentProps(ContainerBaseProps):
    disabled: Union[Undefined, bool] = undefined


class Fragment(ContainerBase[FragmentProps, Component]):

    def __init__(self,
                 children: Union[List[Component], Dict[str, Component]],
                 inited: bool = False) -> None:
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.Fragment, FragmentProps, children,
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

# FIXME use dataclasses_strict
@dataclasses.dataclass
class MatchCaseProps(ContainerBaseProps):
    condition: Union[Undefined, ValueType] = undefined

@dataclasses.dataclass
class MatchCaseItem:
    # if value is undefined, it is default case
    value: Union[ValueType, Undefined]
    child: Component
    isExpr: Union[bool, Undefined] = undefined

@dataclasses.dataclass
class ExprCaseItem:
    value: str
    child: Component
    isExpr: bool = True

@dataclasses.dataclass
class MatchCaseChildDef:
    items: List["Union[MatchCaseItem, ExprCaseItem]"]

class MatchCase(ContainerBase[MatchCaseProps, Component]):
    """special container for extended switch case. (implemented by if/else)
    It is not a real container, but a component with children. 
    It is used to implement switch case in frontend.
    this can be used to implement tab.

    when you use ExprCaseItem, you need to specify a filter expr with "x"
    instead of provide a single value, check [filtrex](https://github.com/m93a/filtrex)
    for more details.

    Example:
    ```Python
    mc = MatchCase([
        MatchCase.Case("some_value", mui.LayoutA(...)),
        MatchCase.Case("other_value", mui.LayoutB(...)),
        MatchCase.ExprCase('\"value\" in x', mui.LayoutC(...)),
        MatchCase.Case(undefined, mui.LayoutD(...)),
    ]) # here condition is undefined, will use default case
    ```

    is equivalent to following javascript code:
    ```javascript
    if (condition === "some_value"){
        return mui.LayoutA(...)
    } else if (condition === "other_value"){
        return mui.LayoutB(...)
    } else if ("value" in condition){
        return mui.LayoutC(...)
    }
    return mui.LayoutD(...)
    ```

    """
    Case = MatchCaseItem
    ExprCase = ExprCaseItem
    ChildDef = MatchCaseChildDef

    def __init__(self,
                 children: List[Union[MatchCaseItem, ExprCaseItem]], init_value: Union[ValueType, Undefined] = undefined) -> None:
        super().__init__(UIType.MatchCase, MatchCaseProps, MatchCaseChildDef(items=children))
        self.props.condition = init_value

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def set_condition(self, condition: Union[ValueType, Undefined]):
        assert isinstance(self._child_structure, MatchCaseChildDef)
        if isinstance(condition, Undefined):
            return await self.send_and_wait(self.update_event(condition=condition))
        has_expr_case = False
        for item in self._child_structure.items:
            if item.value == condition:
                await self.send_and_wait(self.update_event(condition=condition))
                return
            if item.isExpr:
                has_expr_case = True 
        if not has_expr_case:
            raise ValueError(f"Condition {condition} not found in MatchCase")
        else:
            await self.send_and_wait(self.update_event(condition=condition))



def create_ignore_usr_msg(comp: Component):
    msg = comp.create_user_msg_event((UserMessage.create_warning(
        comp._flow_uid_encoded, "UI Running",
        f"UI {comp._flow_uid_encoded}@{str(type(comp).__name__)} is still running, so ignore your control"
    )))
    return msg


if __name__ == "__main__":
    print(snake_to_camel("sizeAttention"))
