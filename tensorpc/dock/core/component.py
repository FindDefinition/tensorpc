# Copyright 2024 Yan Yan
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
import collections.abc
import copy
import dataclasses
import enum
import inspect
import io
import json
import re
import sys
import threading
import time
import traceback
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import (TYPE_CHECKING, Any, AsyncGenerator, AsyncIterator,
                    Awaitable, Callable, Coroutine, Generator, Generic,
                    Iterable, Mapping, Optional, Sequence, Set, Type, Union,
                    cast)

import grpc
from pydantic import (BaseModel, GetCoreSchemaHandler, GetJsonSchemaHandler,
                      TypeAdapter, ValidationError)
from pydantic_core import PydanticCustomError, core_schema
from typing_extensions import (Concatenate, ContextManager, Literal, ParamSpec,
                               Self, TypeAlias, TypeVar)

import tensorpc.core.datamodel.jmes as jmespath
from tensorpc.core import dataclass_dispatch as dataclasses_strict
from tensorpc.core import pfl, prim
from tensorpc.core.annolib import DataclassType
from tensorpc.core.asynctools import cancel_task
from tensorpc.core.core_io import JsonSpecialData
from tensorpc.core.datamodel.asdict import (DataClassWithUndefined,
                                            as_dict_no_undefined_with_exclude,
                                            asdict_no_deepcopy,
                                            asdict_no_deepcopy_with_field)
from tensorpc.core.datamodel.draft import (DraftASTType, DraftBase,
                                           DraftObject, DraftUpdateOp,
                                           capture_draft_update, create_draft,
                                           get_draft_jmespath,
                                           get_draft_pflpath,
                                           insert_assign_draft_op)
from tensorpc.core.datamodel.events import DraftChangeEvent
from tensorpc.core.event_emitter.aio import AsyncIOEventEmitter
from tensorpc.core.event_emitter.base import ExceptionParam
from tensorpc.core.moduleid import is_tensorpc_dynamic_path
from tensorpc.core.tree_id import (UniqueTreeId, UniqueTreeIdForComp,
                                   UniqueTreeIdForTree)
from tensorpc.dock import appctx, marker
from tensorpc.dock.constants import (TENSORPC_APP_ROOT_COMP,
                                     TENSORPC_FLOW_COMP_UID_STRUCTURE_SPLIT,
                                     TENSORPC_FLOW_COMP_UID_TEMPLATE_SPLIT)
from tensorpc.dock.core.appcore import (EventHandler, EventHandlers,
                                        enter_batch_event_context)
from tensorpc.dock.core.reload import AppReloadManager, FlowSpecialMethods
from tensorpc.dock.core.uitypes import ALL_KEY_CODES
from tensorpc.dock.coretypes import MessageLevel, get_unique_node_id
from tensorpc.dock.serv_names import serv_names
from tensorpc.utils.registry import HashableRegistry
from tensorpc.utils.rich_logging import get_logger
from tensorpc.utils.uniquename import UniqueNamePool
from tensorpc.constants import TENSORPC_DEV_USE_PFL_PATH

from ..jsonlike import (BackendOnlyProp, Undefined, as_dict_no_undefined,
                        camel_to_snake, snake_to_camel,
                        split_props_to_undefined, undefined)
from .appcore import (Event, EventDataType, NumberType, RemoteCompEvent,
                      SimpleEventType, ValueType, enter_event_handling_conetxt,
                      get_app, get_event_handling_context)

if TYPE_CHECKING:
    from .datamodel import DataModel
LOGGER = get_logger("tensorpc.ui", log_time_format="%X|")


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
    MASK_DATA_MODEL = 0x10000

    # # placeholder
    # RemoteComponent = -1
    # controls
    ButtonGroup = 0x0
    Input = 0x1
    Switch = 0x2
    Select = 0x3
    Slider = 0x4
    RadioGroup = 0x5
    FlexBox = 0x6
    Button = 0x7
    ListItemButton = 0x8
    ListItemText = 0x9
    Image = 0xa
    Dialog = 0xb
    TooltipFlexBox = 0xc
    # ChartJSLine = 0xd
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
    MUIList = 0x18
    Divider = 0x19
    AppTerminal = 0x1a
    ThemeProvider = 0x1b
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
    Breadcrumbs = 0x30
    Tabs = 0x31
    VirtualizedBox = 0x32
    Terminal = 0x33
    JsonViewer = 0x34
    ListItemIcon = 0x35
    Link = 0x36
    BlenderSlider = 0x37
    SimpleControls = 0x38
    # this component have different state structure.
    TanstackJsonLikeTreeView = 0x39
    MenuList = 0x3a
    SimpleEditor = 0x3c
    IFrame = 0x3d
    Pagination = 0x3e
    VideoPlayer = 0x3f
    GridLayout = 0x40
    VideoBasicStream = 0x41
    TaskLoop = 0x42
    VideoRTCStream = 0x43
    JsonEditor = 0x44
    JsonFastViewer = 0x45

    RANGE_CHART_START = 0x50
    Plotly = 0x51

    MUIBarChart = 0x52
    MUILineChart = 0x53
    MUIScatterChart = 0x54
    MUISparkLineChart = 0x55
    
    RANGE_CHART_END = 0x90
    # special containers
    # react fragment
    Fragment = 0x200
    MatchCase = 0x201

    # data model components
    DataModel = 0x10000
    DataGrid = 0x10001
    DataFlexBox = 0x10002
    DataPortal = 0x10003
    DataSubQuery = 0x10004
    MatrixDataGrid = 0x10005

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

    ThreeBVH = 0x1041
    ThreeURILoaderContext = 0x11040
    ThreeCubeCamera = 0x11041
    ThreeContactShadows = 0x1042
    ThreeGizmoHelper = 0x1043
    ThreeSelectionContext = 0x1044
    ThreeOutlines = 0x1045
    ThreeInstancedBufferMesh = 0x1046
    ThreeDataListGroup = 0x11047
    ThreeHudGroup = 0x1048

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
    ThreeMeshPortalMaterial = 0x105d

    ThreeSimpleGeometry = 0x1101
    ThreeShape = 0x1102
    ThreeLineShape = 0x1103

    ThreeEffectComposer = 0x1200
    ThreeEffectOutline = 0x1201
    ThreeEffectBloom = 0x1202
    ThreeEffectDepthOfField = 0x1203
    ThreeEffectToneMapping = 0x1204

    RANGE_UIKIT_START = 0x1300

    UIKitRoot = 0x1301
    UIKitContainer = 0x1302
    UIKitContent = 0x1303
    UIKitFullscreen = 0x1304
    UIKitText = 0x1305
    UIKitInput = 0x1306
    UIKitImage = 0x1307
    UIKitButton = 0x1308
    UIKitSlider = 0x1309
    UIKitToggleButton = 0x130a
    UIKitToggleButtonGroup = 0x130b
    UIKitCheckbox = 0x130c
    UIKitSwitch = 0x130d

    UIKitDialog = 0x130e
    UIKitTooltip = 0x130f
    UIKitTabs = 0x1310
    UIKitSeparator = 0x1311
    UIKitProgress = 0x1312
    UIKitIconButton = 0x1313
    UIKitBadge = 0x1314


    RANGE_UIKIT_END = 0x1400

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

    MASK_FLOW_COMPONENTS = 0x8000
    Flow = 0x8001
    DataFlow = 0x18001
    FlowMiniMap = 0x8002
    FlowControls = 0x8003
    FlowNodeResizer = 0x8004
    FlowNodeToolBar = 0x8005
    FlowBackground = 0x8006
    # special
    FlowHandle = 0x8007


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
    UIUpdateBasePropsEvent = 14
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
    # emitted when a drop comes from remote component.
    DropFromRemoteComp = -6
    # emitted by event_emitter
    AfterMount = -7
    # emitted by event_emitter
    AfterUnmount = -8

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
    Missed = 10
    Wheel = 11

    KeyHold = 12
    KeyDown = 13
    KeyUp = 14
    CanvasViewportChange = 15
    HudGroupLayoutChange = 16
    MeshPoseChange = 17
    PointerLockReleased = 18

    Change = 20
    Delete = 21
    InputChange = 22
    # modal close: include dialog and drawer (modal based).
    ModalClose = 23
    Drag = 24
    Drop = 25
    SelectNewItem = 26
    Error = 27
    ContextMenuSelect = 28
    ComponentReady = 29

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
    EditorDecorationsChange = 54
    EditorAction = 55
    EditorCursorSelection = 56
    EditorInlayHintsQuery = 57
    EditorHoverQuery = 58
    EditorCodelensQuery = 59
    EditorBreakpointChange = 60

    # data grid events
    DataGridRowSelection = 70
    DataGridFetchDetail = 71
    DataGridFetchInf = 72
    DataGridRowRangeChanged = 73
    DataGridProxyLazyLoadRange = 74

    # flow events
    FlowSelectionChange = 80
    FlowNodesInitialized = 81
    FlowEdgeConnection = 82
    FlowEdgeDelete = 83
    FlowNodeDelete = 84
    FlowNodeContextMenu = 85
    FlowPaneContextMenu = 86
    FlowNodeLogicChange = 87
    FlowEdgeLogicChange = 88
    # visualization change such as position and size.
    FlowVisChange = 89


    PlotlyClickData = 100
    PlotlyClickAnnotation = 101

    # Terminal Events
    TerminalInput = 110
    TerminalResize = 111
    TerminalFrontendUnmount = 112
    TerminalFrontendMount = 113

    # data box events
    DataBoxSecondaryActionClick = 120

    # leaflet events
    MapZoom = 150
    MapMove = 151

    # chart events 
    # used by bar and scatter
    ChartItemClick = 160
    # used by line/area line
    ChartAreaClick = 161
    ChartLineClick = 162
    ChartMarkClick = 163
    ChartAxisClick = 164

    VideoStreamReady = 170
    RTCSdpRequest = 171

UI_TYPES_SUPPORT_DATACLASS: Set[UIType] = {
    UIType.DataGrid, UIType.MatchCase, UIType.DataFlexBox, UIType.Tabs,
    UIType.Allotment, UIType.GridLayout, UIType.MenuList,
    UIType.MatrixDataGrid, UIType.Flow, UIType.Markdown,
    UIType.MonacoEditor, UIType.ThreeDataListGroup
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
    FrontendEventType.Wheel.value,
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
    def from_dict(cls, dc: dict[str, str]):
        return cls(uid=dc["uid"],
                   error=dc["error"],
                   detail=dc["detail"],
                   level=MessageLevel(dc["level"]))

    @classmethod
    def create_error(cls, uid: str, error: str, detail: str):
        return cls(uid, error, MessageLevel.Error, detail)

    @classmethod
    def from_exception(cls, uid: str, exc: BaseException, tb=None):
        lines = traceback.format_exception(None, value=exc, tb=tb)
        return cls(uid, str(exc), MessageLevel.Error, "\n".join(lines))

    @classmethod
    def create_warning(cls, uid: str, error: str, detail: str):
        return cls(uid, error, MessageLevel.Warning, detail)

    @classmethod
    def createinfo(cls, uid: str, error: str, detail: str):
        return cls(uid, error, MessageLevel.Info, detail)


def patch_uid_keys_with_prefix(data: dict[str, Any], prefixes: list[str]):
    new_data = {}
    for k, v in data.items():
        k_uid = UniqueTreeIdForComp(k)
        new_uid = UniqueTreeIdForComp.from_parts(prefixes + k_uid.parts)
        new_data[new_uid.uid_encoded] = v
    return new_data


def patch_uid_list_with_prefix(data: list[str], prefixes: list[str]):
    new_data = []
    for k in data:
        k_uid = UniqueTreeIdForComp(k)
        new_uid = UniqueTreeIdForComp.from_parts(prefixes + k_uid.parts)
        new_data.append(new_uid.uid_encoded)
    return new_data

def unpatch_uid(uid_encoded: str, prefixes: list[str]):
    len_prefix = len(prefixes)
    temp_index = uid_encoded.find(TENSORPC_FLOW_COMP_UID_TEMPLATE_SPLIT)
    if temp_index != -1:
        k_no_template = uid_encoded[:temp_index]
        k_uid = UniqueTreeIdForComp(k_no_template)
        new_uid = UniqueTreeIdForComp.from_parts(k_uid.parts[len_prefix:])
        new_uid_encoded = new_uid.uid_encoded + uid_encoded[temp_index:]
    else:
        k_uid = UniqueTreeIdForComp(uid_encoded)
        new_uid = UniqueTreeIdForComp.from_parts(k_uid.parts[len_prefix:])
        new_uid_encoded = new_uid.uid_encoded
    return new_uid_encoded

def unpatch_uid_keys_with_prefix(data: dict[str, Any], prefixes: list[str]):
    new_data = {}
    for k, v in data.items():
        new_uid_encoded = unpatch_uid(k, prefixes)
        new_data[new_uid_encoded] = v
    return new_data

def patch_unique_id(data: Any, prefixes: list[str]):
    # can't use abc.Sequence because string is sequence too.
    if isinstance(data, list):
        new_data = []
        for i in range(len(data)):
            d = data[i]
            if isinstance(d, UniqueTreeIdForComp):
                d = d.copy()
                d.set_parts_inplace(prefixes + d.parts)
            else:
                d = patch_unique_id(d, prefixes)
            new_data.append(d)
        return new_data
    elif isinstance(data, tuple):
        new_data = []
        for i in range(len(data)):
            d = data[i]
            if isinstance(d, UniqueTreeIdForComp):
                d = d.copy()
                d.set_parts_inplace(prefixes + d.parts)
            else:
                d = patch_unique_id(d, prefixes)
            new_data.append(d)
        return tuple(new_data)
        
    elif isinstance(data, collections.abc.Mapping):
        new_data = {}
        for k, d in data.items():
            if isinstance(d, UniqueTreeIdForComp):
                d = d.copy()
                d.set_parts_inplace(prefixes + d.parts)
            else:
                d = patch_unique_id(d, prefixes)
            new_data[k] = d
        return new_data
    elif isinstance(data, UniqueTreeIdForComp):
        # data.parts[1:]: remote the ROOT part
        data = data.copy()
        data.set_parts_inplace(prefixes + data.parts)
        return data
    else:
        return data

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
    def from_dict(cls, data: dict[str, Any]):
        return cls(AppEditorFrontendEventType(data["type"]), data["data"])


@ALL_APP_EVENTS.register(key=AppEventType.UIEvent.value)
class UIEvent:

    def __init__(self, uid_to_data: dict[str, SimpleEventType]) -> None:
        self.uid_to_data = uid_to_data

    def to_dict(self):
        return self.uid_to_data

    def unpatch_keys_prefix_inplace(self, prefixes: list[str]):
        self.uid_to_data = unpatch_uid_keys_with_prefix(
            self.uid_to_data, prefixes)
        for uid, data in self.uid_to_data.items():
            ev_type = data[0]
            if ev_type == FrontendEventType.Drop.value:
                original_uid = UniqueTreeIdForComp(data[1]["uid"])
                data[1]["uid"] = UniqueTreeIdForComp.from_parts(original_uid.parts[len(prefixes):]).uid_encoded


    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(data)

    def merge_new(self, new):
        return new


@ALL_APP_EVENTS.register(key=AppEventType.UIUpdateUsedEvents.value)
class UpdateUsedEventsEvent:

    def __init__(self, uid_to_data: dict[str, Any]) -> None:
        self.uid_to_data = uid_to_data

    def to_dict(self):
        return self.uid_to_data

    def patch_keys_prefix_inplace(self, prefixes: list[str]):
        self.uid_to_data = patch_uid_keys_with_prefix(self.uid_to_data,
                                                      prefixes)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(data)

    def merge_new(self, new):
        return new


@ALL_APP_EVENTS.register(key=AppEventType.FrontendUIEvent.value)
class FrontendUIEvent:

    def __init__(
            self, uid_to_data: dict[str, tuple[Union[NumberType, str],
                                               Any]]) -> None:
        self.uid_to_data = uid_to_data

    def to_dict(self):
        return self.uid_to_data

    @classmethod
    def from_dict(cls, data: dict[str, tuple[Union[NumberType, str], Any]]):
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

    def __init__(self, uid_to_data: dict[str, Any]) -> None:
        self.uid_to_data = uid_to_data

    def to_dict(self):
        return self.uid_to_data

    def patch_keys_prefix_inplace(self, prefixes: list[str]):
        self.uid_to_data = patch_uid_keys_with_prefix(self.uid_to_data,
                                                      prefixes)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(data)

    def merge_new(self, new):
        assert isinstance(new, ComponentEvent)
        return ComponentEvent({
            **new.uid_to_data,
            **self.uid_to_data,
        })


@ALL_APP_EVENTS.register(key=AppEventType.UISaveStateEvent.value)
class UISaveStateEvent:

    def __init__(self, uid_to_data: dict[str, Any]) -> None:
        self.uid_to_data = uid_to_data

    def to_dict(self):
        return self.uid_to_data

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(data)

    def merge_new(self, new):
        assert isinstance(new, UISaveStateEvent)
        return UISaveStateEvent({
            **new.uid_to_data,
            **self.uid_to_data,
        })


@ALL_APP_EVENTS.register(key=AppEventType.UIUpdateEvent.value)
@ALL_APP_EVENTS.register(key=AppEventType.UIUpdateBasePropsEvent.value)
class UIUpdateEvent:

    def __init__(self,
                 uid_to_data_undefined: dict[str, tuple[dict[str, Any],
                                                        list[str]]],
                 json_only: bool = False) -> None:
        self.uid_to_data_undefined = uid_to_data_undefined
        self.json_only = json_only

    def as_json_only(self):
        return UIUpdateEvent(self.uid_to_data_undefined, True)

    def to_dict(self):
        if self.json_only:
            return JsonSpecialData.from_option(self.uid_to_data_undefined, is_json_only=True, need_freeze=False)
        else:
            return self.uid_to_data_undefined

    def patch_keys_prefix_inplace(self, prefixes: list[str]):
        self.uid_to_data_undefined = patch_uid_keys_with_prefix(
            self.uid_to_data_undefined, prefixes)

    @classmethod
    def from_dict(cls, data: Union[dict[str, Any], JsonSpecialData]):
        json_only = False
        if isinstance(data, JsonSpecialData):
            json_only = True
            data_dict = data.data
        else:
            data_dict = data
        return cls(data_dict, json_only)

    def merge_new(self, new):
        assert isinstance(new, UIUpdateEvent)
        res_uid_to_data: dict[str, Any] = self.uid_to_data_undefined.copy()
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

    def __init__(self, user_excs: list[UserMessage]) -> None:
        self.user_excs = user_excs

    def to_dict(self):
        return [v.to_dict() for v in self.user_excs]

    @classmethod
    def from_dict(cls, data: list[Any]):

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
    def from_dict(cls, data: dict[str, Any]):
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
    def from_dict(cls, data: dict[str, Any]):
        return cls(data)

    def merge_new(self, new):
        assert isinstance(new, LayoutEvent)

        return new


@ALL_APP_EVENTS.register(key=AppEventType.UpdateComponents.value)
class UpdateComponentsEvent:

    def __init__(self,
                 data: dict[str, Any],
                 deleted: Optional[list[str]] = None) -> None:
        self.data = data
        if deleted is None:
            deleted = []
        self.deleted = deleted
        # only added before sent to connected remote app.
        self.remote_component_all_childs: Optional[list[str]] = None

    def patch_keys_prefix_inplace(self, prefixes: list[str]):
        self.data = patch_uid_keys_with_prefix(self.data, prefixes)
        self.deleted = patch_uid_list_with_prefix(self.deleted, prefixes)

    def to_dict(self):
        res = {
            "new": self.data,
            "del": self.deleted,
        }
        if self.remote_component_all_childs is not None:
            res["remoteComponentAllChilds"] = self.remote_component_all_childs
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        remote_component_all_childs = data.get("remoteComponentAllChilds",
                                               None)
        res = cls(data["new"], data["del"])
        res.remote_component_all_childs = remote_component_all_childs
        return res

    def merge_new(self, new):
        assert isinstance(new, UpdateComponentsEvent)
        res = UpdateComponentsEvent({
            **new.data,
            **self.data,
        }, list(set(self.deleted + new.deleted)))
        res.remote_component_all_childs = self.remote_component_all_childs
        return res


@ALL_APP_EVENTS.register(key=AppEventType.CopyToClipboard.value)
class CopyToClipboardEvent:

    def __init__(self, text: str) -> None:
        self.text = text

    def to_dict(self):
        return {
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(data["text"])

    def merge_new(self, new):
        assert isinstance(new, CopyToClipboardEvent)
        return new


@ALL_APP_EVENTS.register(key=AppEventType.InitLSPClient.value)
class InitLSPClientEvent:

    def __init__(self, port: Union[int, str], init_cfg: dict) -> None:
        self.port = port
        self.init_cfg = init_cfg

    def to_dict(self):
        return {"port": self.port, "initConfig": self.init_cfg}

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
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
    def from_dict(cls, data: dict[str, Any]):
        return cls(data)

    def merge_new(self, new):
        assert isinstance(new, ScheduleNextForApp)
        return new


APP_EVENT_TYPES = Union[UIEvent, LayoutEvent, CopyToClipboardEvent,
                        UpdateComponentsEvent, ScheduleNextForApp,
                        AppEditorEvent, UIUpdateEvent, UISaveStateEvent,
                        NotifyEvent, UIExceptionEvent, ComponentEvent,
                        FrontendUIEvent, UpdateUsedEventsEvent,
                        InitLSPClientEvent]


def app_event_from_data(data: dict[str, Any]) -> "AppEvent":
    type_event_tuple: list[tuple[AppEventType, APP_EVENT_TYPES]] = []
    for evtypeval, evdata in data["typeToEvents"]:
        found = False
        for k, v in ALL_APP_EVENTS.items():
            if k == evtypeval:
                type_event_tuple.append(
                    (AppEventType(k), v.from_dict(evdata)))
                found = True
                break
        if not found:
            raise ValueError("not found", evtypeval)
    return AppEvent(data["uid"],
                    type_event_tuple,
                    remote_prefixes=data.get("remotePrefixes", None))


# class ControlEvent:
#     def __init__(self, uid: str, data: Any) -> None:
#         self.uid = uid
#         self.data = data

#     @classmethod
#     def from_dict(cls, data: dict[str, Any]):
#         return cls(data["uid"], data["data"])


class AppEvent:

    def __init__(self,
                 uid: str,
                 type_event_tuple: list[tuple[AppEventType, APP_EVENT_TYPES]],
                 sent_event: Optional[asyncio.Event] = None,
                 event_id: str = "",
                 is_loopback: bool = False,
                 remote_prefixes: Optional[list[str]] = None,
                 after_send_callback: Optional[Callable[[], Awaitable[None]]] = None) -> None:
        # node uid, not component uid
        self.uid = uid
        self.type_event_tuple = type_event_tuple
        # event that indicate this app event is sent
        # used for callback
        self.sent_event = sent_event
        self.event_id = event_id
        self.is_loopback = is_loopback
        self._remote_prefixes = remote_prefixes
        # for additional events, such as remote component events.
        # RemoteCompEvent: only available in remote component.
        self._additional_events: list[RemoteCompEvent] = []
        # currently only used for app init
        self._after_send_callback = after_send_callback

    def is_empty(self):
        return not self.type_event_tuple and not self._additional_events

    def replace_type_event_tuple(self, new_type_event_tuple: list[tuple[AppEventType, APP_EVENT_TYPES]]):
        new_app_event = AppEvent(
            self.uid,
            new_type_event_tuple,
            self.sent_event,
            self.event_id,
            self.is_loopback,
            self._remote_prefixes,
            self._after_send_callback,
        )
        new_app_event._additional_events = self._additional_events
        return new_app_event

    def to_dict(self):
        # here we don't use dict for typeToEvents because key in js must be string.
        t2e = [(k.value, v.to_dict()) for k, v in self.type_event_tuple]
        # make sure layout is proceed firstly.
        t2e.sort(key=lambda x: x[0])
        res = {"uid": self.uid, "typeToEvents": t2e}
        if self._remote_prefixes is not None:
            res["remotePrefixes"] = self._remote_prefixes
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return app_event_from_data(data)

    def merge_new(self, new: "AppEvent") -> "AppEvent":
        new_type_ev_tuple = [e for e in new.type_event_tuple]
        if self.sent_event is not None:
            assert new.sent_event is None, "sent event of new must be None"
            sent_event = self.sent_event
        else:
            sent_event = new.sent_event
        res = AppEvent(self.uid, self.type_event_tuple + new_type_ev_tuple, sent_event)
        res._additional_events = self._additional_events + new._additional_events
        return res 

    def get_event_uid(self):
        if self.event_id:
            return self.uid + "-" + self.event_id
        return self.uid

    def __add__(self, other: "AppEvent"):
        return self.merge_new(other)

    def __iadd__(self, other: "AppEvent"):
        ret = self.merge_new(other)
        self.type_event_tuple = ret.type_event_tuple
        self.sent_event = ret.sent_event
        return self

    def patch_keys_prefix_inplace(self, prefixes: list[str]):
        for _, v in self.type_event_tuple:
            if isinstance(v, (UpdateUsedEventsEvent, UIUpdateEvent,
                              ComponentEvent, UpdateComponentsEvent)):
                v.patch_keys_prefix_inplace(prefixes)

@dataclasses_strict.dataclass
class _DataclassHelper:
    obj: Any


@dataclasses_strict.dataclass
class BasicProps(DataClassWithUndefined):
    # status: int = UIRunStatus.Stop.value
    pass


@dataclasses_strict.dataclass
class ContainerBaseProps(BasicProps):
    pass 
    # childs: list[str] = dataclasses_strict.field(default_factory=list)

@dataclasses_strict.dataclass
class _DataModelPFLQueryDesc:
    dm: Any
    func_uid: str 
    key: str
    query_desc: Any

def _preprocess_pfl_func(func: Callable, num_fixed_args: int = 2) -> tuple[Callable, Optional[list[Any]]]:
    tail_kws = None
    
    if isinstance(func, partial):
        assert not func.args, "args isn't supported in partial, use keywords instead."
        tail_kws = func.keywords
        func = func.func
    assert not inspect.ismethod(func), "use Class.method instead of obj.method"
    fing_sig = inspect.signature(func)
    for k, p in fing_sig.parameters.items():
        assert p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD, "pfl func only support positional or keyword arguments."
    tail_args = None
    if tail_kws is not None:
        bind_args = fing_sig.bind_partial(**tail_kws)
        bind_args.apply_defaults()
        cnt = 0
        tail_args = []
        for k, p in fing_sig.parameters.items():
            if cnt >= num_fixed_args:
                assert k in bind_args.arguments, "you must use partial to set tail keywords."
                tail_args.append(bind_args.arguments[k])
            else:
                assert k not in tail_kws, "you can't use partial on first and second argument."
            cnt += 1
    else:
        assert len(fing_sig.parameters) == num_fixed_args, "func must have two arguments, first is self, second is event data."
    return func, tail_args

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
        # traceback.print_exc()
        _flow_comp_def_path = ""
    if is_dynamic_path:
        return _flow_comp_def_path
    path = Path(_flow_comp_def_path)
    if not path.exists() or path.suffix != ".py":
        _flow_comp_def_path = ""
    return _flow_comp_def_path


@dataclasses.dataclass
class DraftOpUserData:
    component: "Component"
    disabled_handlers: list[Callable] = dataclasses.field(
        default_factory=list)

TEventData = TypeVar("TEventData")

def _draft_expr_or_str_to_str(draft_expr: Any) -> str:
    if isinstance(draft_expr, DraftBase):
        draft_expr_str = get_draft_pflpath(draft_expr)
    else:
        draft_expr_str = draft_expr
    return pfl.compile_pflpath_to_compact_str(draft_expr_str)

class _EventSlotBase(Generic[TEventData]):

    def __init__(self, event_type: EventDataType, comp: "Component", converter: Optional[Callable[[Any], TEventData]] = None):
        self.event_type = event_type
        self.comp = comp
        self.converter = converter

    def on_standard(self, handler: Callable[[Event], Any]) -> Self:
        """standard event means the handler must be a function with one argument of Event.
        this must be used to get template key
        if you use template layout such as table column def.
        """
        self.comp.register_event_handler(self.event_type,
                                         handler,
                                         simple_event=False,
                                         converter=self.converter)
        return self

    def configure(self,
                  stop_propagation: Optional[bool] = None,
                  throttle: Optional[NumberType] = None,
                  debounce: Optional[NumberType] = None,
                  key_codes: Optional[list[str]] = None,
                  set_pointer_capture: bool = False,
                  release_pointer_capture: bool = False,
                  key_hold_interval_delay: Optional[NumberType] = None) -> Self:
        """configure event handlers.
        Args:
            stop_propagation: whether to stop propagation of the event.
            throttle: throttle time in milliseconds.
            debounce: debounce time in milliseconds.
            key_codes: list of key codes to filter the event.
            set_pointer_capture: whether to set pointer capture on pointer down event.
            release_pointer_capture: whether to release pointer capture on pointer up event.
            key_hold_interval_delay: interval delay for key hold event in milliseconds.
        Returns:
            Self: the event slot itself.
        """
        self.comp.configure_event_handlers(
            self.event_type,
            stop_propagation,
            throttle,
            debounce,
            key_codes=key_codes,
            set_pointer_capture=set_pointer_capture,
            release_pointer_capture=release_pointer_capture,
            key_hold_interval_delay=key_hold_interval_delay)
        return self

    def set_frontend_draft_change(self, update_ops: list["EventFrontendUpdateOp"]) -> Self:
        """Set draft exprs to change frontend datamodel directly.
        """
        self.comp.configure_event_handlers(
            self.event_type,
            update_ops=update_ops)
        return self

    def add_frontend_draft_change(self, target_draft: Any, attr: Union[str, int], src_draft: Optional[Any] = None, target_comp: Union["Component", Undefined] = undefined) -> Self:
        """Set draft exprs to change frontend datamodel directly.
        """
        target_draft_str = _draft_expr_or_str_to_str(target_draft)
        if src_draft is not None:
            src_draft_str = _draft_expr_or_str_to_str(src_draft)
        else:
            src_draft_str = undefined
        self.comp._append_event_handler_update_op(
            self.event_type,
            update_op=EventFrontendUpdateOp(
                attr=attr,
                targetPath=target_draft_str,
                targetComp=target_comp,
                srcPath=src_draft_str,
            ))
        return self

    def add_frontend_handler(self, dm: "DataModel", func: Callable[[Any, TEventData], None], use_immer: bool = True, targetPath: str = "") -> Self:
        """use Python Frontend Language (subset of python) to handle event in frontend directly.

        the func must be a function defined in DataModel, and the first argument must be self, the second argument is event data.
        we only use func as a key here (and tail arg calculation), the whole library is already compiled in DataModel.
        """
        # TODO use string targetPath can cause unexpected bugs, consider use draft expr and type check.
        assert isinstance(dm, Component) and dm._flow_comp_type == UIType.DataModel, "dm must be DataModel type."
        assert dm._pfl_library is not None, "your datamodel must define pfl marked functions."
        func, tail_args = _preprocess_pfl_func(func, num_fixed_args=2)
        func_specs = dm._pfl_library.get_compiled_unit_specs(func)
        assert len(func_specs) == 1, "func can't be template"
        if targetPath != "":
            targetPath = _draft_expr_or_str_to_str(targetPath)

        op = EventFrontendUpdateOp(
            attr="",
            targetPath=undefined if targetPath == "" else targetPath,
            partialTailArgs=tail_args if tail_args is not None else undefined,
            pflFuncUid=func_specs[0].uid,
        )
        if not use_immer:
            op.dontUseImmer = True
        self.comp._append_event_handler_update_op(
            self.event_type,
            update_op=op)
        return self

    def add_frontend_draft_set_none(self, target_draft: Any, attr: Union[str, int], target_comp: Union["Component", Undefined] = undefined) -> Self:
        """Set draft exprs to change frontend datamodel directly.
        """
        target_draft_str = _draft_expr_or_str_to_str(target_draft)
        self.comp._append_event_handler_update_op(
            self.event_type,
            update_op=EventFrontendUpdateOp(
                attr=attr,
                targetPath=target_draft_str,
                targetComp=target_comp,
                srcPath=None,
            ))
        return self

    def disable_and_stop_propagation(self) -> Self:
        self.comp.configure_event_handlers(self.event_type,
                                           stop_propagation=True)
        return self

    def clear(self):
        self.comp.remove_event_handlers(self.event_type)
        return self


class EventSlot(_EventSlotBase[TEventData]):

    def on(self, handler: Callable[[TEventData], Any]):
        """simple event means the event data isn't Event, but the data of Event, or none for no-arg event
        such as click.
        """
        self.comp.register_event_handler(self.event_type,
                                         handler,
                                         simple_event=True,
                                         converter=self.converter)
        return self

    def off(self, handler: Callable[[TEventData], Any]):
        self.comp.remove_event_handler(self.event_type, handler)
        return self


class EventSlotZeroArg(_EventSlotBase):

    def on(self, handler: Callable[[], Any]):
        """simple event means the event data isn't Event, but the data of Event, or none for no-arg event
        such as click.
        """
        self.comp.register_event_handler(self.event_type,
                                         handler,
                                         simple_event=True,
                                         converter=self.converter)
        return self

    def off(self, handler: Callable[[], Any]):
        self.comp.remove_event_handler(self.event_type, handler)
        return self


class _EventSlotEmitterBase:

    def __init__(self, event_type: EventDataType,
                 emitter: "AsyncIOEventEmitter[EventDataType, Event]"):
        self.event_type = event_type
        self.emitter = emitter

    def on_standard(self, handler: Callable[[Event], Any]) -> Self:
        """standard event means the handler must be a function with one argument of Event.
        this must be used to get template key
        if you use template layout such as table column def.
        """
        self.emitter.on(self.event_type, handler)
        return self

    def clear(self):
        self.emitter.remove_all_listeners(self.event_type)
        return self

class EventSlotEmitter(_EventSlotEmitterBase, Generic[TEventData]):
    # TODO remove this
    def __init__(self,
                 event_type: EventDataType,
                 emitter: "AsyncIOEventEmitter[EventDataType, Event]",
                 converter: Optional[Callable[[Any], TEventData]] = None):
        self.event_type = event_type
        self.emitter = emitter
        self.converter = converter

    def on(self, handler: Callable[[TEventData], Any]) -> "EventSlotEmitter":
        """simple event means the event data isn't Event, but the data of Event, or none for no-arg event
        such as click.
        """
        # use f_key as correct key instead of partial.
        self.emitter.on(self.event_type,
                        partial(self._handle_event, handler=handler),
                        f_key=handler)
        return self

    def _handle_event(self, event: Event, handler: Callable[[TEventData],
                                                            Any]):
        if self.converter is not None:
            return handler(self.converter(event.data))
        return handler(event.data)

    def off(self, handler: Callable) -> "EventSlotEmitter":
        self.emitter.remove_listener(self.event_type, handler)
        return self


class EventSlotNoArgEmitter(_EventSlotEmitterBase):
    # TODO remove this
    def __init__(self, event_type: EventDataType,
                 emitter: "AsyncIOEventEmitter[EventDataType, Event]"):
        self.event_type = event_type
        self.emitter = emitter

    def on(self, handler: Callable[[], Any]) -> Self:
        """simple event means the event data isn't Event, but the data of Event, or none for no-arg event
        such as click.
        """
        self.emitter.on(self.event_type,
                        partial(self._handle_event, handler=handler),
                        f_key=handler)
        return self

    def _handle_event(self, event: Event, handler: Callable[[], Any]):
        return handler()

    def off(self, handler: Callable) -> Self:
        self.emitter.remove_listener(self.event_type, handler)
        return self

@dataclasses.dataclass
class EventFrontendUpdateOp:
    attr: Union[str, int]
    targetPath: Union[Undefined, str] = undefined 
    targetComp: Union[Undefined, "Component"] = undefined
    srcPath: Optional[Union[Undefined, str]] = undefined
    pflAstJson: Union[Undefined, str] = undefined
    pflCode: Union[Undefined, str] = undefined
    dontUseImmer: Union[Undefined, bool] = undefined
    partialTailArgs: Union[Undefined, list[Any]] = undefined
    pflFuncUid: Union[Undefined, str] = undefined
    isPFLPath: Union[Undefined, bool] = undefined

T_child_structure = TypeVar("T_child_structure",
                            default=Any,
                            bound=DataclassType)


class _ComponentEffects:

    def __init__(self) -> None:
        self._flow_effects: dict[str, list[Callable[[], Union[Callable[
            [], Any], None, Coroutine[None, None, Union[Callable[[], Any],
                                                        None]]]]]] = {}
        self._flow_unmounted_effects: dict[str,
                                           list[tuple[Any, Callable[[],
                                                         _CORO_NONE]]]] = {}

    def use_effect(self,
                   effect: Callable[[],
                                    Union[Optional[Callable[[], Any]],
                                          Coroutine[None, None,
                                                    Optional[Callable[[],
                                                                      Any]]]]],
                   key: str = ""):
        if key not in self._flow_effects:
            self._flow_effects[key] = []
            self._flow_unmounted_effects[key] = []

        self._flow_effects[key].append(effect)

    def has_effect_key(self, key: str):
        return key in self._flow_effects

    def remove_effect_key(self, key: str):
        self._flow_effects.pop(key)
        self._flow_unmounted_effects.pop(key)

    def remove_effect(self, effect: Callable[[], Any], key: str = ""):
        if key in self._flow_effects:
            self._flow_effects[key].remove(effect)
            for i, (e, _) in enumerate(self._flow_unmounted_effects[key]):
                if e is effect:
                    self._flow_unmounted_effects[key].pop(i)
                    break

class Component(Generic[T_base_props, T_child]):

    def __init__(self,
                 type: UIType,
                 prop_cls: Type[T_base_props],
                 allowed_events: Optional[Iterable[EventDataType]] = None,
                 uid: Optional[UniqueTreeIdForComp] = None,
                 json_only: bool = False) -> None:
        self._flow_comp_core: Optional[AppComponentCore] = None
        self._flow_uid: Optional[UniqueTreeIdForComp] = uid
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
        self._prop_field_names: Set[str] = set(
            [x.name for x in dataclasses.fields(prop_cls)])
        self._mounted_override = False
        self.__raw_props: dict[str, Any] = {}
        self._flow_allowed_events: Set[EventDataType] = set([
            FrontendEventType.BeforeMount.value,
            FrontendEventType.BeforeUnmount.value
        ])
        if allowed_events is not None:
            self._flow_allowed_events.update(allowed_events)
        self._flow_user_datas: list[Any] = []
        self._flow_comp_def_path = _get_obj_def_path(self)
        self._flow_reference_count = 0

        self._flow_pfl_library: Optional[bytes] = None
        self._flow_data_model_paths: dict[str, Union[str, tuple[Component, str], _DataModelPFLQueryDesc]] = {}
        self._flow_exclude_field_ids: set[int] = set()

        # tensorpc will scan your prop dict to find
        # np.ndarray and bytes by default.
        # this will cost time in deep and large json, if you use
        # json_only, this scan will be skipped.
        # WARNING: you shouldn't use json_only when prop contains another component.
        self._flow_json_only = json_only
        self._flow_comp_status = UIRunStatus.Stop.value
        self.effects = _ComponentEffects()
        self._flow_unmount_effect_objects: list[Callable[[], _CORO_NONE]] = []

        self._flow_event_context_creator: Optional[Callable[
            [], ContextManager]] = None
        # flow event handlers is used for frontend events
        self._flow_event_handlers: dict[EventDataType, EventHandlers] = {}
        # event emitter is used for backend events, e.g. mount, unmount
        self._flow_event_emitter: AsyncIOEventEmitter[
            EventDataType, Event] = AsyncIOEventEmitter()
        self._flow_event_emitter.add_exception_listener(
            self.__event_emitter_on_exc)
        # can be used to init data before dict send to frontend.
        self.event_before_mount = self._create_emitter_event_slot_noarg(
            FrontendEventType.BeforeMount)
        self.event_before_unmount = self._create_emitter_event_slot_noarg(
            FrontendEventType.BeforeUnmount)
        self.event_after_mount = self._create_emitter_event_slot_noarg(
            FrontendEventType.AfterMount)
        # we don't need after_unmount because there is no interaction between
        # before and after unmount.
        # self.event_after_unmount = self._create_emitter_event_slot_noarg(
        #     FrontendEventType.AfterUnmount)

    def use_effect(self,
                   effect: Callable[[],
                                    Union[Optional[Callable[[], Any]],
                                          Coroutine[None, None,
                                                    Optional[Callable[[],
                                                                      Any]]]]],
                   key: str = ""):
        return self.effects.use_effect(effect, key)

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any,
                                     _handler: GetCoreSchemaHandler):
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.any_schema(),
        )

    @classmethod
    def validate(cls, v):
        if not isinstance(v, Component):
            raise ValueError('Component required')
        return v

    def _create_event_slot(self,
                           event_type: Union[FrontendEventType, EventDataType],
                           converter: Optional[Callable[[Any],
                                                        TEventData]] = None):
        if isinstance(event_type, FrontendEventType):
            event_type_value = event_type.value
        else:
            event_type_value = event_type
        return EventSlot(event_type_value, self, converter)

    def _create_event_slot_noarg(self, event_type: Union[FrontendEventType,
                                                         EventDataType],
                                converter: Optional[Callable[[Any],
                                                        TEventData]] = None):
        if isinstance(event_type, FrontendEventType):
            event_type_value = event_type.value
        else:
            event_type_value = event_type
        return EventSlotZeroArg(event_type_value, self, converter)

    def _create_emitter_event_slot(
            self,
            event_type: Union[FrontendEventType, EventDataType],
            converter: Optional[Callable[[Any], TEventData]] = None):
        if isinstance(event_type, FrontendEventType):
            event_type_value = event_type.value
            assert event_type.value < 0, "only support backend events"
            return EventSlotEmitter(event_type_value, self._flow_event_emitter,
                                    converter)
        else:
            event_type_value = event_type
        return EventSlotEmitter(event_type_value, self._flow_event_emitter,
                                converter)

    def _create_emitter_event_slot_noarg(self,
                                         event_type: Union[FrontendEventType,
                                                           EventDataType]):
        if isinstance(event_type, FrontendEventType):
            event_type_value = event_type.value
            assert event_type.value < 0, "only support backend events"
            return EventSlotNoArgEmitter(event_type_value,
                                         self._flow_event_emitter)
        else:
            event_type_value = event_type
        return EventSlotNoArgEmitter(event_type_value,
                                     self._flow_event_emitter)

    @property
    def flow_event_emitter(self) -> AsyncIOEventEmitter[EventDataType, Event]:
        return self._flow_event_emitter

    def get_special_methods(self, reload_mgr: AppReloadManager):
        metas = reload_mgr.query_type_method_meta(type(self),
                                                  no_code=True,
                                                  include_base=True)
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

    def _attach(
            self, uid: UniqueTreeIdForComp,
            comp_core: AppComponentCore) -> dict[UniqueTreeIdForComp, "Component"]:
        if self._flow_reference_count == 0:
            self._flow_uid = uid
            self._flow_comp_core = comp_core
            self._flow_reference_count += 1
            self.flow_event_emitter.emit(
                FrontendEventType.BeforeMount.value,
                Event(FrontendEventType.BeforeMount.value, None))
            return {uid: self}
        self._flow_reference_count += 1
        return {}

    def _finish_detach(self):
        self._flow_uid = None
        self._flow_comp_core = None

    def _prepare_detach(self) -> dict[UniqueTreeIdForComp, "Component"]:
        self._flow_reference_count -= 1
        if self._flow_reference_count == 0:
            # self.flow_event_emitter.emit(
            #     FrontendEventType.BeforeUnmount.value,
            #     Event(FrontendEventType.BeforeUnmount.value, None))
            res_uid = self._flow_uid
            assert res_uid is not None
            # self._flow_uid = None
            # self._flow_comp_core = None
            # delay _flow_uid and _flow_comp_core clean in _finish_detach
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
                           json_only: bool = False,
                           ensure_json_keys: Optional[list[str]] = None,
                           exclude_field_ids: set[int] | None = None):
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
            return self.create_update_event(kwargs, json_only, ensure_json_keys=ensure_json_keys, exclude_field_ids=self._flow_exclude_field_ids)

        return wrapper

    async def handle_event(self, ev: Event, is_sync: bool = False) -> Any:
        ev_type = FrontendEventType(ev.type)
        LOGGER.warning(f"Receive event {ev_type.name}, but `handle_event` not implemented for `{self.__class__.__name__}`")

    def __repr__(self):
        if self._flow_uid is None:
            return f"{self.__class__.__name__}()"
        res = f"{self.__class__.__name__}({self._flow_uid_encoded})"
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

    async def _cancel_task(self, source: str = ""):
        # ignore all task error here.
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except:
                traceback.print_exc()
            self._task = None

    def update_raw_props(self, sx_props: dict[str, Any]):
        self.__raw_props.update(sx_props)
        return self

    def get_raw_props(self):
        return self.__raw_props

    def _get_dm_props_for_frontend(self, model_paths: dict[str, Union[str, tuple["Component", str], _DataModelPFLQueryDesc]]):
        dm_paths_new = {}
        for k, v in model_paths.items():
            if isinstance(v, _DataModelPFLQueryDesc):
                dm_paths_new[k] = (v.dm._flow_uid, (v.func_uid, v.key, v.query_desc))

            elif not isinstance(v, str):
                dm_paths_new[k] = (v[0]._flow_uid, v[1])
            else:
                dm_paths_new[k] = v
        # group dm props by container to speed up query in frontend.
        dm_paths_new_grouped = {}
        id_to_containers = {}
        for k, v in model_paths.items():
            if isinstance(v, _DataModelPFLQueryDesc):
                container = id(v.dm)
                id_to_containers[container] = v.dm

            elif not isinstance(v, str):
                container = id(v[0])
                id_to_containers[id(v[0])] = v[0]
            else:
                container = None
            if container not in dm_paths_new_grouped:
                dm_paths_new_grouped[container] = []
            if isinstance(v, _DataModelPFLQueryDesc):
                dm_paths_new_grouped[container].append((k, (v.func_uid, v.key, v.query_desc)))
            elif not isinstance(v, str):
                dm_paths_new_grouped[container].append((k, v[1]))
            else:
                dm_paths_new_grouped[container].append((k, v))
        grouped_res = []
        for container_id, paths in dm_paths_new_grouped.items():
            if container_id is not None:
                container = id_to_containers[container_id]
                grouped_res.append(
                    (container._flow_uid, paths))
            else:
                grouped_res.append(
                    ("", paths))
        return dm_paths_new, grouped_res

    def to_dict(self):
        """undefined will be removed here.
        if you reimplement to_dict, you need to use 
        camel name, no conversion provided.
        """
        props = self.get_props_dict()
        props["status"] = self._flow_comp_status
        props, und = split_props_to_undefined(props)
        props.update(as_dict_no_undefined(self.__raw_props))
        res = {
            "uid": self._flow_uid,
            "type": self._flow_comp_type.value,
            "props": props,
        }
        if self._flow_data_model_paths:
            dm_paths_new, dm_paths_new_grouped = self._get_dm_props_for_frontend(self._flow_data_model_paths)
            res["dmProps"] = dm_paths_new
            res["dmPropsGrouped"] = dm_paths_new_grouped
        if self._flow_pfl_library is not None:
            res["pflLibraryBin"] = self._flow_pfl_library
        evs = self._get_used_events_dict()
        if evs:
            res["usedEvents"] = evs
        if self._flow_json_only:
            res["props"] = JsonSpecialData.from_option(props, is_json_only=True, need_freeze=False)
        return res

    def _get_used_events_dict(self):
        evs = []
        for k, v in self._flow_event_handlers.items():
            if not isinstance(v, Undefined) and not v.backend_only:
                disable_and_stop = (v.stop_propagation)
                if v.handlers or disable_and_stop or v.update_ops:
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

    def get_sync_props(self) -> dict[str, Any]:
        """this function is used for props you want to kept when app
        shutdown or layout updated.
        1. app shutdown: only limited component support props recover.
        2. update layout: all component will override props
        by previous sync props
        """
        return {"status": self._flow_comp_status}

    def get_persist_props(self) -> Optional[dict[str, Any]]:
        return None

    async def set_persist_props_async(self, state: dict[str, Any]) -> None:
        return

    def get_props_dict(self) -> dict[str, Any]:
        if self._flow_exclude_field_ids:
            res = self.props.get_dict_with_fields(
                dict_factory=partial(undefined_comp_dict_factory_with_exclude, exclude_field_ids=self._flow_exclude_field_ids),
                obj_factory=undefined_comp_obj_factory)  # type: ignore
        else:
            res = self.__props.get_dict(
                dict_factory=undefined_comp_dict_factory,
                obj_factory=undefined_comp_obj_factory)  # type: ignore
        return res

    def validate_props(self, props: dict[str, Any]) -> bool:
        """use this function to validate props before call
        set props.
        """
        return True

    def set_props(self, props: dict[str, Any]):
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
                AppEvent("", [(
                    AppEventType.UIEvent,
                    UIEvent({self._flow_uid.uid_encoded: ev})
                )],
                         is_loopback=True))

    async def put_app_event(self, ev: AppEvent):
        if self.is_mounted():
            return await self.queue.put(ev)

    def bind_fields(self, **kwargs: Union[str, tuple["Component", Union[str, Any]], Any]):
        for k in kwargs.keys():
            assert k in self._prop_field_names, f"overrided prop must be defined in props class, {k}"
        return self.bind_fields_unchecked(**kwargs)

    def bind_fields_unchecked_dict(self, kwargs: dict[str, Union[str, tuple["Component", Union[str, Any]], Any]]):
        new_kwargs: dict[str, Union[str, tuple["Component", str]]] = {}
        for k, v_may_draft in kwargs.items():
            if isinstance(v_may_draft, DraftBase):
                v = get_draft_pflpath(v_may_draft)
            else:
                v = v_may_draft
            if isinstance(v, str):
                # print("PFL", v)
                new_kwargs[k] = pfl.compile_pflpath_to_compact_str(v)
            else:
                assert isinstance(v, tuple) and len(v) == 2
                assert isinstance(v[0], Component)
                vv = v[1] 
                if isinstance(vv, DraftBase):
                    vp = get_draft_pflpath(vv)
                    vp = pfl.compile_pflpath_to_compact_str(vp)
                    new_kwargs[k] =  (v[0], vp)
                else:
                    vp = pfl.compile_pflpath_to_compact_str(vv)
                    new_kwargs[k] = (v[0], vp)

        self._flow_data_model_paths.update(new_kwargs)
        return self

    def bind_pfl_query(self, dm: "DataModel", **kwargs: tuple[Callable, str]):
        for k in kwargs.keys():
            assert k in self._prop_field_names, f"overrided prop must be defined in props class {self.__prop_cls}, {k}"
        return self.bind_pfl_query_unchecked_dict(dm, kwargs)

    def bind_pfl_query_unchecked_dict(self, dm: "DataModel", kwargs: Mapping[str, tuple[Callable, str]]):
        assert isinstance(dm, Component) and dm._flow_comp_type == UIType.DataModel, "dm must be DataModel type."
        assert dm._pfl_library is not None, "your datamodel must define pfl marked functions."
        for prop, (func, key) in kwargs.items():
            if isinstance(func, partial):
                func_real = func.func 
            else:
                func_real = func 
            meta = pfl.get_compilable_meta(func_real)
            assert meta is not None 
            assert meta.userdata is not None 
            fn_type = meta.userdata["dmFuncType"]
            assert fn_type in [1, 2], "only support mark_pfl_query_func/mark_pfl_query_nested_func"
            if fn_type == 1:
                num_fixed_args = 1
            else:
                num_fixed_args = 2

            func, tail_args = _preprocess_pfl_func(func, num_fixed_args=num_fixed_args)
            if tail_args is not None:
                for arg in tail_args:
                    assert isinstance(arg, (str, int, float, bool)), "only support primitive tail args in pfl query"
            func_specs = dm._pfl_library.get_compiled_unit_specs(func)
            assert len(func_specs) == 1, "func can't be template"
            pfl_func_id = func_specs[0].uid
            fn_cache_key = pfl_func_id
            if tail_args is not None:
                fn_cache_key_parts = [pfl_func_id]
                for arg in tail_args:
                    if isinstance(arg, float):
                        prefix = "f"
                    elif isinstance(arg, bool):
                        prefix = "b"
                    elif isinstance(arg, int):
                        prefix = "i"
                    else:
                        prefix = "s"
                    fn_cache_key_parts.append(f"{prefix}:{arg}")
                fn_cache_key = UniqueTreeId.from_parts(fn_cache_key_parts).uid_encoded
            query_desc: dict[str, Any] = {
                "tailQueryUid": fn_cache_key,
                "dmFuncType": fn_type,
            }
            if tail_args is not None:
                query_desc["tailArgs"] = tail_args
            self._flow_data_model_paths[prop] = _DataModelPFLQueryDesc(dm, pfl_func_id, key, query_desc)
        return self 
        
    def bind_fields_unchecked(self, **kwargs: Union[str, tuple["Component", Union[str, Any]], Any]):
        return self.bind_fields_unchecked_dict(kwargs)

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
                                 stop_propagation: Optional[bool] = False,
                                 throttle: Optional[NumberType] = None,
                                 debounce: Optional[NumberType] = None,
                                 backend_only: Optional[bool] = False,
                                 update_ops: Optional[list[EventFrontendUpdateOp]] = None,
                                 key_codes: Optional[list[str]] = None,
                                set_pointer_capture: bool = False,
                                release_pointer_capture: bool = False,
                                key_hold_interval_delay: Optional[NumberType] = None):
        if isinstance(type, FrontendEventType):
            type_value = type.value
        else:
            type_value = type
        if type_value not in self._flow_event_handlers:
            self._flow_event_handlers[type_value] = EventHandlers([])
        handlers = self._flow_event_handlers[type_value]
        if stop_propagation is not None:
            handlers.stop_propagation = stop_propagation
        handlers.throttle = throttle
        handlers.debounce = debounce
        handlers.set_pointer_capture = set_pointer_capture
        handlers.release_pointer_capture = release_pointer_capture
        handlers.key_hold_interval_delay = key_hold_interval_delay
        if backend_only is not None:
            handlers.backend_only = backend_only
        if update_ops is not None:
            handlers.update_ops = update_ops
        if key_codes is not None:
            for code in key_codes:
                assert code in ALL_KEY_CODES
            handlers.key_codes = key_codes
        return

    def _append_event_handler_update_op(self,
                                 type: Union[FrontendEventType, EventDataType],
                                 update_op: EventFrontendUpdateOp):
        if isinstance(type, FrontendEventType):
            type_value = type.value
        else:
            type_value = type
        if type_value not in self._flow_event_handlers:
            self._flow_event_handlers[type_value] = EventHandlers([])
        handlers = self._flow_event_handlers[type_value]
        handlers.update_ops.append(update_op)
        return

    def register_event_handler(self,
                               type: Union[FrontendEventType, EventDataType],
                               cb: Callable,
                               stop_propagation: bool = False,
                               throttle: Optional[NumberType] = None,
                               debounce: Optional[NumberType] = None,
                               backend_only: bool = False,
                               simple_event: bool = True,
                               converter: Optional[Callable[[Any],
                                                            Any]] = None,
                               update_ops: Optional[list[EventFrontendUpdateOp]] = None):
        if self._flow_allowed_events:
            if not backend_only:
                assert type in self._flow_allowed_events, f"only support events: {self._flow_allowed_events}, but got {type}"

        evh = EventHandler(cb, simple_event, converter=converter)
        if isinstance(type, FrontendEventType):
            type_value = type.value
        else:
            type_value = type
        if type_value not in self._flow_event_handlers:
            self._flow_event_handlers[type_value] = EventHandlers([])
        handlers = self._flow_event_handlers[type_value]
        if type == FrontendEventType.DragCollect:
            assert len(
                handlers.handlers) == 0, "DragCollect only support one handler"
        self.configure_event_handlers(type_value, stop_propagation, throttle,
                                      debounce, backend_only,
                                      update_ops)
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

    def state_change_callback(
            self,
            value: Any,
            type: ValueType = FrontendEventType.Change.value):
        pass

    def create_update_event(self,
                            data: dict[str, Union[Any, Undefined]],
                            json_only: bool = False,
                            validate: bool = False,
                            ensure_json_keys: Optional[list[str]] = None,
                            exclude_field_ids: Optional[set[int]] = None):
        if validate:
            self.__prop_cls(**data)  # type: ignore
        data_no_und = {}
        data_unds = []
        ensure_json_keys_set = set(ensure_json_keys) if ensure_json_keys else set()
        for k, v in data.items():
            # k = snake_to_camel(k)
            if isinstance(v, Undefined):
                data_unds.append(k)
            else:
                if k in ensure_json_keys_set:
                    data_no_und[k] = v
                else:
                    if exclude_field_ids is not None:
                        data_no_und[k] = as_dict_no_undefined_with_exclude(v, exclude_field_ids)
                    else:
                        data_no_und[k] = as_dict_no_undefined(v)
        assert self._flow_uid is not None
        ev = UIUpdateEvent(
            {self._flow_uid.uid_encoded: (data_no_und, data_unds)}, json_only)
        # uid is set in flowapp service later.
        return AppEvent("", [(AppEventType.UIUpdateEvent, ev)])

    def _create_update_base_props_event(self, dm_props: Optional[Union[dict[str, Any], Undefined]] = None, 
                used_events: Optional[Union[list[Any], Undefined]] = None,
                pfl_library: Optional[Union[bytes, Undefined]] = None):
        data_no_und = {}
        data_unds = []
        if isinstance(dm_props, Undefined):
            data_unds.append("dmProps")
            data_unds.append("dmPropsGrouped")
        else:
            if dm_props is not None:
                dm_paths_new, dm_paths_new_grouped = self._get_dm_props_for_frontend(dm_props)
                data_no_und["dmProps"] = dm_paths_new
                data_no_und["dmPropsGrouped"] = dm_paths_new_grouped
        if isinstance(used_events, Undefined):
            data_unds.append("usedEvents")
        else:
            if used_events is not None:
                data_no_und["usedEvents"] = used_events
        if isinstance(pfl_library, Undefined):
            data_unds.append("pflLibraryBin")
        else:
            if used_events is not None:
                data_no_und["pflLibraryBin"] = pfl_library

        assert self._flow_uid is not None
        ev = UIUpdateEvent(
            {self._flow_uid.uid_encoded: (data_no_und, data_unds)}, False)
        # uid is set in flowapp service later.
        return AppEvent("", [(AppEventType.UIUpdateBasePropsEvent, ev)])

    def create_update_used_events_event(self):
        used_events = self._get_used_events_dict()
        return self._create_update_base_props_event(used_events=used_events)

    async def sync_used_events(self):
        return await self.put_app_event(self.create_update_used_events_event())

    def create_comp_event(self, data: dict[str, Any]):
        """create component control event for
        backend -> frontend direct communication
        """
        assert self._flow_uid is not None
        ev = ComponentEvent(
            {self._flow_uid.uid_encoded: as_dict_no_undefined(data)})
        # uid is set in flowapp service later.
        return AppEvent("", [(AppEventType.ComponentEvent, ev)])

    def create_comp_raw_event(self, data: Any):
        """create component control event for
        backend -> frontend direct communication
        """
        assert self._flow_uid is not None
        ev = ComponentEvent(
            {self._flow_uid.uid_encoded: data})
        # uid is set in flowapp service later.
        return AppEvent("", [(AppEventType.ComponentEvent, ev)])

    async def send_and_wait(self, ev: AppEvent, wait: bool = True):
        if ev.sent_event is None:
            ev.sent_event = asyncio.Event()
        await self.put_app_event(ev)
        if self.is_mounted():
            if wait:
                await ev.sent_event.wait()

    def create_update_comp_event(self, updates: dict[str, Any],
                                 deleted: Optional[list[str]]):
        ev = UpdateComponentsEvent(updates, deleted)
        # uid is set in flowapp service later.
        return AppEvent("", [(AppEventType.UpdateComponents, ev)])

    def create_delete_comp_event(self, deletes: list[str]):
        ev = UpdateComponentsEvent({}, deletes)
        # uid is set in flowapp service later.
        return AppEvent("", [(AppEventType.UpdateComponents, ev)])

    def create_remote_comp_event(self, key: str, data: Any):
        """create event for remote comp to send to mounted master app.
        """
        res = AppEvent("", [])
        res._additional_events = [RemoteCompEvent(key, data)]
        return res

    def create_user_msg_event(self, exc: UserMessage):
        ev = UIExceptionEvent([exc])
        # uid is set in flowapp service later.
        return AppEvent("", [(AppEventType.UIException, ev)])

    def create_editor_event(self, type: AppEditorEventType, data: Any):
        # uid is set in flowapp service later.
        ev = AppEditorEvent(type, data)
        return AppEvent("", [(AppEventType.AppEditor, ev)])

    def send_error(self, title: str, detail: str):
        assert self._flow_uid is not None
        user_exc = UserMessage.create_error(self._flow_uid.uid_encoded, title,
                                            detail)
        return self.put_app_event(self.create_user_msg_event(user_exc))

    def send_exception(self,
                       e: BaseException,
                       tb=None,
                       tb_from_sys: bool = True):
        ss = io.StringIO()
        traceback.print_exc(file=ss)
        if self._flow_uid is None:
            uid = "UNKNOWN"
        else:
            uid = self._flow_uid.uid_encoded
        if tb_from_sys:
            tb = sys.exc_info()[2]
        return self.put_app_event(
            self.create_user_msg_event(
                UserMessage.from_exception(uid, e, tb)))

    async def __event_emitter_on_exc(self, exc_param: ExceptionParam):
        traceback.print_exception(exc_param.exc)
        e = exc_param.exc
        ss = io.StringIO()
        traceback.print_exception(exc_param.exc, file=ss)
        assert self._flow_uid is not None
        user_exc = UserMessage.create_error(self._flow_uid.uid_encoded,
                                            repr(e), ss.getvalue())
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
                           change_status: bool = True,
                           capture_draft: bool = False) -> Optional[Any]:
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
            self._flow_comp_status = UIRunStatus.Running.value
        # only ui with loading support need sync first.
        # otherwise don't use this because slow
        if sync_status_first:
            ev = asyncio.Event()
            await self.sync_status(sync_state, ev)
            await ev.wait()
        res = None
        assert self._flow_uid is not None
        datamodel_ctx = nullcontext()
        if capture_draft:
            datamodel_ctx = capture_draft_update()
        with enter_event_handling_conetxt(self._flow_uid) as evctx:
            try:
                batch_ev = AppEvent("", [])
                with enter_batch_event_context(batch_ev):
                    with datamodel_ctx as ctx:
                        coro = cb()
                        if inspect.iscoroutine(coro):
                            res = await coro
                        else:
                            res = coro
                        if res_callback is not None:
                            res_coro = res_callback(res)
                            if inspect.iscoroutine(res_coro):
                                await res_coro
                    if ctx is not None:
                        await self._run_draft_update(ctx._ops)
                if not batch_ev.is_empty():
                    await self.put_app_event(batch_ev)
            except BaseException as e:
                traceback.print_exc()
                ss = io.StringIO()
                traceback.print_exc(file=ss)
                user_exc = UserMessage.create_error(
                    self._flow_uid.uid_encoded
                    if self._flow_uid is not None else "", repr(e),
                    ss.getvalue())
                await self.put_app_event(self.create_user_msg_event(user_exc))
                app = get_app()
                if app._flowapp_enable_exception_inspect:
                    await app._inspect_exception()
            finally:
                if change_status:
                    self._flow_comp_status = UIRunStatus.Stop.value
                    await self.sync_status(sync_state)
                if evctx.delayed_callbacks:
                    for cb in evctx.delayed_callbacks:
                        coro = cb()
                        if inspect.iscoroutine(coro):
                            await coro
        return res

    async def run_callbacks(
            self,
            cbs: list[Callable[[], _CORO_NONE]],
            sync_state: bool = False,
            sync_status_first: bool = False,
            res_callback: Optional[Callable[[Any], _CORO_NONE]] = None,
            change_status: bool = True,
            capture_draft: bool = False,
            finish_callback: Optional[Callable[[], _CORO_NONE]] = None):
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
            self._flow_comp_status = UIRunStatus.Running.value
        # only ui with loading support need sync first.
        # otherwise don't use this because slow
        if sync_status_first:
            ev = asyncio.Event()
            await self.sync_status(False, ev)
            await ev.wait()
        res_list :list[Any] = []
        assert self._flow_uid is not None
        with enter_event_handling_conetxt(self._flow_uid) as evctx:
            batch_ev = AppEvent("", [])
            with enter_batch_event_context(batch_ev):

                for cb in cbs:
                    datamodel_ctx = nullcontext()
                    if capture_draft:
                        datamodel_ctx = capture_draft_update()
                    try:
                        # we shouldn't batch draft update here.
                        with datamodel_ctx as ctx:
                            coro = cb()
                            if inspect.iscoroutine(coro):
                                res = await coro
                            else:
                                res = coro
                            res_list.append(res)
                            if res_callback is not None:
                                res_coro = res_callback(res)
                                if inspect.iscoroutine(res_coro):
                                    await res_coro
                        if ctx is not None and ctx._ops:
                            await self._run_draft_update(ctx._ops)
                    except BaseException as e:
                        traceback.print_exc()
                        ss = io.StringIO()
                        traceback.print_exc(file=ss)
                        assert self._flow_uid is not None
                        user_exc = UserMessage.create_error(
                            self._flow_uid.uid_encoded, repr(e), ss.getvalue())
                        await self.put_app_event(
                            self.create_user_msg_event(user_exc))
                        res_list.append(None)
                        # app = get_app()
                        # if app._flowapp_enable_exception_inspect:
                        #     await app._inspect_exception()
            # finally:
            if not batch_ev.is_empty():
                await self.put_app_event(batch_ev)
            if finish_callback is not None:
                coro = finish_callback()
                if inspect.iscoroutine(coro):
                    await coro
            if change_status:
                self._flow_comp_status = UIRunStatus.Stop.value
                await self.sync_status(sync_state)
            else:
                if sync_state:
                    await self.sync_state()
            if evctx.delayed_callbacks:
                for cb in evctx.delayed_callbacks:
                    coro = cb()
                    if inspect.iscoroutine(coro):
                        await coro
        return res_list

    async def _run_draft_update(self, ops: list[DraftUpdateOp]):
        if not ops:
            return 
        comp_uid_to_ops = {}
        for op in ops:
            userdata = op.get_userdata_typed(DraftOpUserData)
            if userdata is not None:
                comp = userdata.component
                if comp._flow_comp_type == UIType.DataModel and comp.is_mounted():
                    if comp._flow_uid not in comp_uid_to_ops:
                        comp_uid_to_ops[comp._flow_uid] = (comp, [])
                    comp_uid_to_ops[comp._flow_uid][1].append(op)
        for uid, (comp, ops) in comp_uid_to_ops.items():
            if ops:
                ev_or_none = await comp._internal_update_with_jmes_ops_event(ops)
                if ev_or_none is not None:
                    await self.put_app_event(ev_or_none)

    def __data_model_auto_event_handler(self, value: Any, draft: Any):
        assert isinstance(draft, DraftBase)
        insert_assign_draft_op(draft, value)

    async def __data_model_auto_event_handler_sync(self, value: Any, draft: Any, comp: Any):
        assert isinstance(draft, DraftBase)
        async with comp.draft_update():
            insert_assign_draft_op(draft, value)

    async def __uncontrolled_draft_change_handler(self, ev: DraftChangeEvent, prop_name: str, value_prep: Optional[Callable[[Any], Any]] = None):
        new_value = ev.new_value
        if value_prep is not None:
            new_value = value_prep(new_value)
        await self.put_app_event(self._update_props_base(self.propcls)(**{
            prop_name: new_value
        }))


    def _bind_field_with_change_event(self, field_name: str, draft: Any, sync_update: bool = False, uncontrolled: bool = False, uncontrolled_prep: Optional[Callable[[Any], Any]] = None):
        """Bind a draft with change event. bind_fields is called automatically.
        Equal to following code:

        ```Python
        origin_draft = ...
        async def handle_change(self, value):
            origin_draft.a.b = value
        ```
        """
        assert not self.is_mounted(), "you can't bind field after component mounted"
        assert isinstance(draft, DraftBase)
        assert isinstance(draft._tensorpc_draft_attr_userdata, DraftOpUserData), "you must use comp.get_draft_target() to get draft"
        comp: DataModel = cast(Any, draft._tensorpc_draft_attr_userdata.component)
        if draft._tensorpc_draft_attr_cur_node.type == DraftASTType.FUNC_CALL:
            raise ValueError("can't bind field with getItem or getAttr result")
        assert FrontendEventType.Change.value in self._flow_allowed_events
        if not uncontrolled:
            self.bind_fields(**{field_name: draft})
        if sync_update:
            self.register_event_handler(FrontendEventType.Change, 
                partial(self.__data_model_auto_event_handler_sync, draft=draft, comp=comp), simple_event=True)
        else:
            self.register_event_handler(FrontendEventType.Change, 
                partial(self.__data_model_auto_event_handler, draft=draft), simple_event=True)
        if uncontrolled:
            comp.install_draft_change_handler(draft, partial(self.__uncontrolled_draft_change_handler, 
                prop_name=field_name, value_prep=uncontrolled_prep), installed_comp=self)
        return self 

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
            ev = self.create_update_event({"status": self._flow_comp_status})
            ev.sent_event = sent_event
            await self.put_app_event(ev)

    async def sync_state(self, sent_event: Optional[asyncio.Event] = None):
        return await self.sync_status(True, sent_event)

    def get_sync_event(self, sync_state: bool = False):
        if sync_state:
            return self.create_update_event(self.get_sync_props())
        else:
            return self.create_update_event({"status": self._flow_comp_status})

    async def _run_mount_special_methods(self, container_comp: "Component", reload_mgr: Optional[AppReloadManager] = None):
        if reload_mgr is None:
            reload_mgr = container_comp.flow_app_comp_core.reload_mgr

        special_methods = self.get_special_methods(reload_mgr)
        if special_methods.did_mount is not None:
            # run callback in container comp.
            await container_comp.run_callback(
                special_methods.did_mount.get_binded_fn(),
                sync_status_first=False,
                change_status=False,
                capture_draft=True)
        with capture_draft_update() as ctx:
            await self._flow_event_emitter.emit_async(FrontendEventType.AfterMount.value,
                                                    Event(FrontendEventType.AfterMount.value, None)) 
        if ctx._ops:
            await self._run_draft_update(ctx._ops)

        # run effects
        for k, effects in self.effects._flow_effects.items():
            for effect in effects:
                res = await container_comp.run_callback(effect,
                                                sync_status_first=False,
                                                change_status=False,
                                                capture_draft=True)
                if res is not None:
                    # res is effect
                    self.effects._flow_unmounted_effects[k].append(
                        (effect, res))

    async def _run_unmount_special_methods(self, container_comp: "Component", reload_mgr: Optional[AppReloadManager] = None):
        if reload_mgr is None:
            reload_mgr = container_comp.flow_app_comp_core.reload_mgr
        special_methods = self.get_special_methods(reload_mgr)
        if special_methods.will_unmount is not None:
            # run callback in container comp because self component is unmounted,
            # so app queue are already removed.
            await container_comp.run_callback(
                special_methods.will_unmount.get_binded_fn(),
                sync_status_first=False,
                change_status=False)
        await self._flow_event_emitter.emit_async(FrontendEventType.BeforeUnmount.value,
                                                  Event(FrontendEventType.BeforeUnmount.value, None)) 
        for k, unmount_effects in self.effects._flow_unmounted_effects.items(
        ):
            for _, unmount_effect in unmount_effects:
                await container_comp.run_callback(unmount_effect,
                                        sync_status_first=False,
                                        change_status=False)
            unmount_effects.clear()

class ForEachResult(enum.Enum):
    Continue = 0
    Return = 1


def _find_comps_in_dc(obj):
    "(list[tuple[str, Any]]) -> dict[str, Any]"
    """same as dataclasses.asdict except that this function
    won't recurse into nested container.
    """
    res_comp_localids: list[tuple[Component, str]] = []
    if not dataclasses.is_dataclass(obj):
        raise TypeError("asdict() should be called on dataclass instances")
    _find_comps_in_dc_inner(obj, res_comp_localids, "")
    return res_comp_localids


def _find_comps_in_dc_inner(obj, res_comp_localids: list[tuple[Component,
                                                               str]],
                            comp_local_id: str):
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
            _find_comps_in_dc_inner(getattr(obj, f.name), res_comp_localids,
                                    local_id)
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
        for i, v in enumerate(obj):
            _find_comps_in_dc_inner(v, res_comp_localids, local_id_prefix +
                                    str(i))
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        for i, v in enumerate(obj):
            _find_comps_in_dc_inner(v, res_comp_localids, local_id_prefix +
                                        str(i))
    elif isinstance(obj, dict):
        # TODO validate that all keys are number or letters
        for k in obj.keys():
            assert isinstance(k,
                              str), f"key {k} must be string and alphanumeric"
        for k, v in obj.items():
            _find_comps_in_dc_inner(v, res_comp_localids, local_id_prefix +
                                     k)


def undefined_comp_dict_factory(x: list[tuple[str, Any]]):
    res: dict[str, Any] = {}
    for k, v in x:
        if isinstance(v, Component):
            assert v.is_mounted(
            ), f"you must ensure component is inside comp tree if you add it to props, {k}, {type(v)}"
            # res[k] = v._flow_uid_encoded
            res[k] = v._flow_uid
        elif isinstance(v, UniqueTreeIdForComp):
            # delay convert to string for remote component uid patch
            res[k] = v
        elif isinstance(v, UniqueTreeIdForTree):
            res[k] = v.uid_encoded
        elif not isinstance(v, (Undefined, BackendOnlyProp)):
            res[k] = v
    return res

def undefined_comp_dict_factory_with_exclude(x: list[tuple[str, Any, Any]], exclude_field_ids: set[int]):
    res: dict[str, Any] = {}
    for k, v, f in x:
        if id(f) in exclude_field_ids:
            continue
        if isinstance(v, Component):
            assert v.is_mounted(
            ), f"you must ensure component is inside comp tree if you add it to props, {k}, {type(v)}"
            # res[k] = v._flow_uid_encoded
            res[k] = v._flow_uid
        elif isinstance(v, UniqueTreeIdForComp):
            # delay convert to string for remote component uid patch
            res[k] = v
        elif isinstance(v, UniqueTreeIdForTree):
            res[k] = v.uid_encoded
        elif not isinstance(v, (Undefined, BackendOnlyProp)):
            res[k] = v
    return res

def undefined_comp_obj_factory(x: Any):
    if isinstance(x, Component):
        assert x.is_mounted(
        ), f"you must ensure component is inside comp tree if you add it to props, {type(x)}"
        # return x._flow_uid_encoded
        return x._flow_uid
    return x


def component_dict_to_serializable_dict(x: dict[str, Component]):
    layout_dict = {u: c.to_dict() for u, c in x.items()}
    # print("LOCAL LAYOUT", layout_dict)
    for u, v in x.items():
        if isinstance(v, RemoteComponentBase) and v.is_remote_mounted:
            ll, root_uid_remote = v.get_layout_dict_sync()
            # if get_layout_dict_sync fail, root_uid_remote will be empty.
            if root_uid_remote != "":
                layout_dict.update(ll)
                # patch childs of remote component container
                layout_dict[u]["props"]["childs"] = [UniqueTreeIdForComp(root_uid_remote)]
    return layout_dict


async def component_dict_to_serializable_dict_async(x: dict[str, Component]):
    layout_dict = {u: c.to_dict() for u, c in x.items()}
    for u, v in x.items():
        if isinstance(v, RemoteComponentBase) and v.is_remote_mounted:
            ll, root_uid_remote = v.get_layout_dict_sync()
            # if get_layout_dict_sync fail, root_uid_remote will be empty.
            if root_uid_remote != "":
                layout_dict.update(ll)
                # patch childs of remote component container
                layout_dict[u]["props"]["childs"] = [UniqueTreeIdForComp(root_uid_remote)]
    return layout_dict


class ContainerBase(Component[T_container_props, T_child]):

    def __init__(self,
                 base_type: UIType,
                 prop_cls: Type[T_container_props],
                 _children: Optional[Union[Mapping[str, T_child],
                                           DataclassType]] = None,
                 inited: bool = False,
                 allowed_events: Optional[Iterable[EventDataType]] = None,
                 uid: Optional[UniqueTreeIdForComp] = None,
                 app_comp_core: Optional[AppComponentCore] = None,
                 json_only: bool = False) -> None:
        super().__init__(base_type, prop_cls, allowed_events, uid, json_only)
        self._flow_comp_core = app_comp_core
        if inited:
            assert app_comp_core is not None  # and uid_to_comp is not None
        self._pool = UniqueNamePool()
        if _children is None:
            _children = {}
        # self._children = _children
        self._child_comps: dict[str, Component] = {}
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
            children_dict = self._find_comps_in_dataclass(_children)
            for comp, local_id in children_dict:
                self._child_comps[local_id] = comp

        # self.props.childs: list[str] = []
        self.inited = inited
        self._prevent_add_layout = False

    def __repr__(self):
        res = super().__repr__()
        if self._child_comps:
            res += f"({','.join(self._child_comps.keys())})"
        return res

    def _find_comps_in_dataclass(self, _children: DataclassType):
        return _find_comps_in_dc(_children)

    def _get_comp_by_uid(self, uid: str):
        uid_obj = UniqueTreeIdForComp(uid)
        parts = uid_obj.parts
        # RemoteComponentBase shouldn't handle any event and shouldn't
        # be used as regular container, so we always return remote comp
        # to indicate it's remote component.
        if isinstance(self, RemoteComponentBase) and self.is_remote_mounted:
            return self
        # uid contains root, remove it at first.
        return self._get_comp_by_uid_resursive(parts[1:])

    def _get_comp_by_uid_resursive(self, parts: list[str]) -> Component:
        key = parts[0]
        child_comp = self._child_comps[key]
        if isinstance(child_comp,
                      RemoteComponentBase) and child_comp.is_remote_mounted:
            return child_comp
        if len(parts) == 1:
            return self._child_comps[key]
        else:
            assert isinstance(child_comp, ContainerBase)
            return child_comp._get_comp_by_uid_resursive(parts[1:])

    def _get_comps_by_uid(self, uid: str):
        parts = UniqueTreeIdForComp(uid).parts
        if isinstance(self, RemoteComponentBase) and self.is_remote_mounted:
            return [self]
        # uid contains root, remove it at first.
        return [self] + self._get_comps_by_uid_resursive(parts[1:])

    def _get_comps_by_uid_resursive(self, parts: list[str]) -> list[Component]:
        key = parts[0]
        try:
            child_comp = self._child_comps[key]
        except KeyError:
            traceback.print_exc()
            LOGGER.error("can't find child comp %s by uid %s, ava: %s", str(type(self)), key, str(list(self._child_comps.keys())))
            raise
        if isinstance(child_comp,
                      RemoteComponentBase) and child_comp.is_remote_mounted:
            return [child_comp]
        if len(parts) == 1:
            return [child_comp]
        else:
            assert isinstance(child_comp, ContainerBase)
            return [child_comp] + child_comp._get_comps_by_uid_resursive(
                parts[1:])

    def _foreach_comp_recursive(self, child_ns: UniqueTreeIdForComp,
                                handler: Callable[[UniqueTreeIdForComp, Component],
                                                  Union[ForEachResult, None]]):
        res_foreach: list[tuple[UniqueTreeIdForComp, ContainerBase]] = []
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

    def _foreach_comp(self, handler: Callable[[UniqueTreeIdForComp, Component],
                                              Union[ForEachResult, None]]):
        assert self._flow_uid is not None, f"_flow_uid must be set before modify_comp, {type(self)}, {self._flow_reference_count}, {id(self)}"
        handler(self._flow_uid, self)
        self._foreach_comp_recursive(self._flow_uid, handler)

    def _update_uid(self):

        def handler(uid: UniqueTreeIdForComp, v: Component):
            v._flow_uid = uid

        self._foreach_comp(handler)

    def _prepare_detach(self):
        disposed_uids: dict[UniqueTreeIdForComp, Component] = super()._prepare_detach()
        for v in self._child_comps.values():
            disposed_uids.update(v._prepare_detach())
        return disposed_uids

    def _prepare_detach_child(self, childs: Optional[list[str]] = None):
        disposed_uids: dict[UniqueTreeIdForComp, Component] = {}
        if childs is None:
            childs = list(self._child_comps.keys())
        for k in childs:
            v = self._child_comps[k]
            disposed_uids.update(v._prepare_detach())
        return disposed_uids

    def _attach_child(self,
                      comp_core: AppComponentCore,
                      childs: Optional[list[str]] = None):
        atached_uids: dict[UniqueTreeIdForComp, Component] = {}
        assert self._flow_uid is not None
        if childs is None:
            childs = list(self._child_comps.keys())
        for k in childs:
            v = self._child_comps[k]
            atached_uids.update(
                v._attach(self._flow_uid.append_part(k), comp_core))
        return atached_uids

    def _attach(self, uid: UniqueTreeIdForComp, comp_core: AppComponentCore):
        attached: dict[UniqueTreeIdForComp,
                       Component] = super()._attach(uid, comp_core)
        assert self._flow_uid is not None
        for k, v in self._child_comps.items():
            attached.update(v._attach(self._flow_uid.append_part(k),
                                      comp_core))
        return attached

    def _get_uid_encoded_to_comp_dict(self):
        res: dict[str, Component] = {}

        def handler(uid: UniqueTreeIdForComp, v: Component):
            res[uid.uid_encoded] = v

        self._foreach_comp(handler)
        return res

    def _get_uid_to_comp_dict(self):
        res: dict[UniqueTreeIdForComp, Component] = {}

        def handler(uid: UniqueTreeIdForComp, v: Component):
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

    def get_item_type_checked(self, key: str, check_type: Type[T]) -> T:
        assert key in self._child_comps, f"{key}, {self._child_comps.keys()}"
        res = self._child_comps[key]
        assert isinstance(res, check_type)
        return res

    def __getitem__(self, key: str):
        assert key in self._child_comps, f"{key}, {self._child_comps.keys()}"
        return self._child_comps[key]

    def __contains__(self, key: str):
        return key in self._child_comps

    def __len__(self):
        return len(self._child_comps)

    def _get_all_nested_child_recursive(self, name: str, res: list[Component]):
        comp = self[name]
        res.append(comp)
        if isinstance(comp, ContainerBase):
            for child in comp._child_comps:
                comp._get_all_nested_child_recursive(child, res)

    def _get_all_nested_child(self, name: str):
        res: list[Component] = []
        self._get_all_nested_child_recursive(name, res)
        return res

    def _get_all_nested_childs(self, names: Optional[list[str]] = None):
        if names is None:
            names = list(self._child_comps.keys())
        comps: list[Component] = []
        for c in names:
            comps.extend(self._get_all_nested_child(c))
        return comps

    def add_layout(self, layout: Union[Mapping[str, Component],
                                       Sequence[Component]]):
        return self.init_add_layout(layout)

    def __check_child_structure_is_none(self):
        assert self._child_structure is None, "you can only use set_layout or init to specify child with structure"

    def init_add_layout(self, layout: Union[Mapping[str, Component],
                                            Sequence[Component]]):
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
        num_exists = len(self._child_comps)
        if isinstance(layout, Sequence):
            layout = {str(i + num_exists): v for i, v in enumerate(layout)}
        # for k, v in layout.items():
        #     v._flow_name = k
        if self._prevent_add_layout:
            raise ValueError("you must init layout in app_create_layout")
        self._child_comps.update(layout)

    def get_props_dict(self):
        state = super().get_props_dict()
        state["childs"] = [self[n]._flow_uid for n in self._child_comps]
        if self._child_structure is not None:
            if self._flow_exclude_field_ids:
                state["childsComplex"] = asdict_no_deepcopy_with_field(
                    self._child_structure,
                    dict_factory_with_field=partial(undefined_comp_dict_factory_with_exclude, exclude_field_ids=self._flow_exclude_field_ids),
                    obj_factory=undefined_comp_obj_factory)
            else:
                state["childsComplex"] = asdict_no_deepcopy(
                    self._child_structure,
                    dict_factory=undefined_comp_dict_factory,
                    obj_factory=undefined_comp_obj_factory)
        return state

    async def _run_special_methods(
            self,
            attached: list[Component],
            detached: dict[UniqueTreeIdForComp, Component],
            reload_mgr: Optional[AppReloadManager] = None,
            center_callback: Optional[Callable[[], Awaitable[None]]] = None):
        """Run lifecycle methods.
        All methods must run in leaf-to-root order, include attach
        and detach.
        """
        if reload_mgr is None:
            reload_mgr = self.flow_app_comp_core.reload_mgr
        # all lifecycle method must run in leaf-to-root order.
        # this can resolve nested layout-change problem.
        # when we run lifecycle methods of a component,
        # we can ensure layout change process of all childs are finished.
        # so it's safe to call any nested layout change func in this component.
        # NOTE: we assume all lifecycle method only handle child components,
        # don't use functions such as `find_component` to modify other
        # component, which will cause undefined behavior.
        sort_items: list[tuple[tuple[str, ...], int, Component]] = []
        for att in attached:
            uid = att._flow_uid
            assert uid is not None
            sort_items.append((tuple(uid.parts), False, att))
        for uid, det in detached.items():
            sort_items.append((tuple(uid.parts), True, det))

        sort_items.sort(reverse=True)
        for _, is_detach, comp in sort_items:
            if is_detach:
                deleted = comp
                await deleted._run_unmount_special_methods(self, reload_mgr)
        if center_callback is not None:
            await center_callback()
        for _, is_detach, comp in sort_items:
            if not is_detach:
                attach = comp
                await attach._run_mount_special_methods(self, reload_mgr)

    async def set_new_layout_locally(self, layout: Union[dict[str, Component],
                                                         T_child_structure],
                                    update_child_complex: bool = True):
        detached_uid_to_comp = self._prepare_detach_child()
        if isinstance(layout, dict):
            self._child_comps = layout
        else:
            assert dataclasses.is_dataclass(layout)
            assert type(layout) == type(
                self._child_structure
            ), f"{type(layout)}, {type(self._child_structure)}"
            self._child_comps.clear()
            # parse dataclass, get components, save structure
            self._child_structure = layout
            children_dict = self._find_comps_in_dataclass(layout)
            for comp, local_id in children_dict:
                self._child_comps[local_id] = comp
        attached = self._attach_child(self.flow_app_comp_core)
        new_detached_uid_to_comp = detached_uid_to_comp.copy()
        for k, v in detached_uid_to_comp.items():
            if v._flow_reference_count > 0:
                assert k in attached
                new_detached_uid_to_comp.pop(k)
                attached.pop(k)
        detached_uid_to_comp = new_detached_uid_to_comp
        # update all childs of this component
        comps_frontend = {
            c._flow_uid_encoded: c
            for c in self._get_all_nested_childs()
        }
        comps_frontend_dict = await component_dict_to_serializable_dict_async(
            comps_frontend)
        # child_uids = [self[c]._flow_uid_encoded for c in self._child_comps]
        child_uids = [self[c]._flow_uid for c in self._child_comps]

        update_msg: dict[str, Any] = {"childs": child_uids}
        if self._child_structure is not None and update_child_complex:
            update_msg["childsComplex"] = asdict_no_deepcopy(
                self._child_structure,
                dict_factory=undefined_comp_dict_factory,
                obj_factory=undefined_comp_obj_factory)
        update_ev = self.create_update_event(update_msg)
        deleted = [x.uid_encoded for x in detached_uid_to_comp.keys()]
        return update_ev + self.create_update_comp_event(
            comps_frontend_dict, deleted), list(
                attached.values()), detached_uid_to_comp

    async def set_new_layout(
            self,
            layout: Union[dict[str, Component], list[Component],
                          T_child_structure],
            post_ev_creator: Optional[Callable[[], AppEvent]] = None,
            disable_delay: bool = False,
            update_child_complex: bool = True):
        if isinstance(layout, list):
            layout = {str(i): v for i, v in enumerate(layout)}

        self_to_be_removed = self._check_ctx_contains_self(
            list(self._child_comps.keys()))
        evctx = get_event_handling_context()
        if evctx is not None and self_to_be_removed and not disable_delay:
            evctx.delayed_callbacks.append(lambda: self._set_new_layout_delay(
                layout,
                comp_dont_need_cancel=evctx.comp_uid,
                post_ev_creator=post_ev_creator,
                update_child_complex=update_child_complex))
        else:
            await self._set_new_layout_delay(
                layout, post_ev_creator=post_ev_creator,
                update_child_complex=update_child_complex)

    async def _lifecycle_center_cb(self, new_ev: AppEvent, post_ev_creator: Optional[Callable[[], AppEvent]] = None):
        if post_ev_creator is not None:
            await self.put_app_event(new_ev + post_ev_creator())
        else:
            await self.put_app_event(new_ev)

    async def _set_new_layout_delay(
            self,
            layout: Union[dict[str, Component], T_child_structure],
            comp_dont_need_cancel: Optional[UniqueTreeIdForComp] = None,
            post_ev_creator: Optional[Callable[[], AppEvent]] = None,
            update_child_complex: bool = True):
        new_ev, attached, removed_dict = await self.set_new_layout_locally(
            layout, update_child_complex=update_child_complex)
        for deleted_uid, deleted in removed_dict.items():
            if comp_dont_need_cancel is not None and comp_dont_need_cancel == deleted_uid:
                continue
            await deleted._cancel_task(f"set_new_layout-{deleted_uid}-{type(deleted)}")
        # await self.put_app_event(new_ev)
        # if post_ev_creator is not None:
        #     await self.put_app_event(post_ev_creator())
        await self._run_special_methods(attached, removed_dict, center_callback=partial(self._lifecycle_center_cb, new_ev, post_ev_creator))
        for v in removed_dict.values():
            v._finish_detach()

    def _check_ctx_contains_self(self, keys: Union[list[str], Set[str]]):
        evctx = get_event_handling_context()
        self_to_be_removed = False
        if evctx is not None:
            for k in keys:
                if k in self._child_comps:
                    comp = self._child_comps[k]
                    if comp._flow_uid is not None and evctx.comp_uid.startswith(
                            comp._flow_uid):
                        self_to_be_removed = True
                        break
        return self_to_be_removed

    async def remove_childs_by_keys(
            self,
            keys: list[str],
            update_child_complex: bool = True,
            post_ev_creator: Optional[Callable[[], AppEvent]] = None):
        if update_child_complex:
            self.__check_child_structure_is_none()
        self_to_be_removed = self._check_ctx_contains_self(keys)
        evctx = get_event_handling_context()
        if evctx is not None and self_to_be_removed:
            evctx.delayed_callbacks.append(
                lambda: self._remove_childs_by_keys_delay(
                    keys,
                    post_ev_creator,
                    comp_dont_need_cancel=evctx.comp_uid))
        else:
            await self._remove_childs_by_keys_delay(keys,
                                                    post_ev_creator)

    async def _remove_childs_by_keys_delay(
            self,
            keys: list[str],
            post_ev_creator: Optional[Callable[[], AppEvent]] = None,
            comp_dont_need_cancel: Optional[UniqueTreeIdForComp] = None):
        detached_uid_to_comp = self._prepare_detach_child(keys)
        for k, comp in detached_uid_to_comp.items():
            if comp_dont_need_cancel is not None and comp_dont_need_cancel == k:
                continue
            await comp._cancel_task("remove_childs_by_keys")
        for k in keys:
            self._child_comps.pop(k)
        if not detached_uid_to_comp:
            return
        deleted: list[str] = []
        for k, v in detached_uid_to_comp.items():
            deleted.append(k.uid_encoded)
            if isinstance(v, RemoteComponentBase):
                deleted.extend(v._cur_child_uids)
        ev = self.create_delete_comp_event(deleted)
        # if post_ev_creator is not None:
        #     ev = ev + post_ev_creator()
        # await self.put_app_event(ev)
        # if post_ev_creator is not None:
        #     await self.put_app_event(post_ev_creator())
        await self._run_special_methods([], detached_uid_to_comp, center_callback=partial(
            self._lifecycle_center_cb, ev, post_ev_creator)
        )
        for v in detached_uid_to_comp.values():
            v._finish_detach()

    def update_childs_complex_event(self):
        update_msg: dict[str, Any] = {}
        update_msg["childsComplex"] = asdict_no_deepcopy(
            self._child_structure,
            dict_factory=undefined_comp_dict_factory,
            obj_factory=undefined_comp_obj_factory)
        update_ev = self.create_update_event(update_msg)
        return update_ev

    async def update_childs_complex(self):
        """TODO: this function assume the child components isn't changed.
        if you need to update child comps, you must use set_new_layout.
        """
        await self.send_and_wait(self.update_childs_complex_event())

    async def update_childs_locally(self,
                                    layout: dict[str, Component],
                                    update_child_complex: bool = True):
        """update child components locally, without sending event to frontend.
        
        Args:
            layout: new layout
            update_child_complex: whether to update child complex structure. only 
                for advanced usage.
        """
        if update_child_complex:
            self.__check_child_structure_is_none()
        intersect = set(layout.keys()).intersection(self._child_comps.keys())
        detached = self._prepare_detach_child(list(intersect))
        self._child_comps.update(layout)
        attached = self._attach_child(self.flow_app_comp_core,
                                      list(layout.keys()))
        new_detached_uid_to_comp = detached.copy()
        for k, v in detached.items():
            if v._flow_reference_count > 0:
                assert k in attached
                new_detached_uid_to_comp.pop(k)
                attached.pop(k)
        detached = new_detached_uid_to_comp

        # remove replaced components first.
        comps_frontend = {
            c._flow_uid_encoded: c
            for c in self._get_all_nested_childs(list(layout.keys()))
        }
        comps_frontend_dict = await component_dict_to_serializable_dict_async(
            comps_frontend)
        child_uids = [self[c]._flow_uid for c in self._child_comps]
        update_msg: dict[str, Any] = {"childs": child_uids}
        if update_child_complex and self._child_structure is not None:
            update_msg["childsComplex"] = asdict_no_deepcopy(
                self._child_structure,
                dict_factory=undefined_comp_dict_factory,
                obj_factory=undefined_comp_obj_factory)
        update_ev = self.create_update_event(update_msg)
        deleted = [x.uid_encoded for x in detached.keys()]
        deleted: list[str] = []
        for k, v in detached.items():
            deleted.append(k.uid_encoded)
            if isinstance(v, RemoteComponentBase):
                deleted.extend(v._cur_child_uids)
        return update_ev + self.create_update_comp_event(
            comps_frontend_dict, deleted), list(attached.values()), detached

    async def update_childs(
            self,
            layout: Union[dict[str, Component], list[Component]],
            update_child_complex: bool = True,
            post_ev_creator: Optional[Callable[[], AppEvent]] = None):
        """update child components locally, without sending event to frontend.
        
        Args:
            layout: new layout
            update_child_complex: whether to update child complex structure. only 
                for advanced usage.
            additional_ev: additional event to send
        """
        if isinstance(layout, list):
            layout = {str(i): v for i, v in enumerate(layout)}
        if update_child_complex:
            self.__check_child_structure_is_none()
        intersect = set(layout.keys()).intersection(self._child_comps.keys())
        evctx = get_event_handling_context()
        self_to_be_removed = self._check_ctx_contains_self(intersect)
        if evctx is not None and self_to_be_removed:
            evctx.delayed_callbacks.append(lambda: self._update_childs_delay(
                layout, update_child_complex, post_ev_creator, evctx.
                comp_uid))
        else:
            await self._update_childs_delay(layout, update_child_complex,
                                            post_ev_creator)

    async def _update_childs_delay(
            self,
            layout: dict[str, Component],
            update_child_complex: bool = True,
            post_ev_creator: Optional[Callable[[], AppEvent]] = None,
            comp_dont_need_cancel: Optional[UniqueTreeIdForComp] = None):
        new_ev, attached, removed_dict = await self.update_childs_locally(
            layout, update_child_complex)
        for deleted_uid, deleted in removed_dict.items():
            if comp_dont_need_cancel == deleted_uid:
                continue
            await deleted._cancel_task("update_childs")
        # if post_ev_creator is not None:
        #     new_ev = new_ev + post_ev_creator()
        # await self.put_app_event(new_ev)
        # if post_ev_creator is not None:
        #     await self.put_app_event(post_ev_creator())
        await self._run_special_methods(attached, removed_dict, center_callback=partial(
            self._lifecycle_center_cb, new_ev, post_ev_creator)
        )
        for v in removed_dict.values():
            v._finish_detach()

    async def replace_childs(self, layout: dict[str, Component]):
        self.__check_child_structure_is_none()
        for k in layout.keys():
            assert k in self._child_comps
        return await self.update_childs(layout)

    def create_comp_event_raw(self, raw_ev_data: dict):
        assert self._flow_uid is not None
        ev = ComponentEvent(
                {self._flow_uid.uid_encoded: raw_ev_data})
        return AppEvent("", [(AppEventType.ComponentEvent, ev)])

    def create_comp_event(self, data: dict[str, Any]):
        """create component control event for
        backend -> frontend direct communication
        """
        assert self._flow_uid is not None
        if self._child_structure is not None:
            ev_data = asdict_no_deepcopy(
                _DataclassHelper(data),
                dict_factory=undefined_comp_dict_factory,
                obj_factory=undefined_comp_obj_factory)
            assert isinstance(ev_data, dict)
            ev = ComponentEvent({self._flow_uid.uid_encoded: ev_data["obj"]})
        else:
            ev = ComponentEvent(
                {self._flow_uid.uid_encoded: as_dict_no_undefined(data)})
        # uid is set in flowapp service later.
        return AppEvent("", [(AppEventType.ComponentEvent, ev)])


class RemoteComponentBase(ContainerBase[T_container_props, T_child], abc.ABC):

    def __init__(self, url: str, port: int, key: str, base_type: UIType,
                 prop_cls: Type[T_container_props],
                 fail_callback: Optional[Callable[[], Coroutine[None, None, Any]]] = None,
                 enable_fallback_layout: bool = True,
                 fastrpc_timeout: int = 5) -> None:
        super().__init__(base_type, prop_cls)
        self._url = url
        self._port = port
        self._key = key
        self._cur_child_uids: list[str] = []

        self._is_remote_mounted: bool = False

        self._mount_lock = asyncio.Lock()
        self._enable_fallback_layout = enable_fallback_layout

        self._fail_callback = fail_callback

        self._shutdown_ev: asyncio.Event = asyncio.Event()
        self._remote_task: Optional[asyncio.Task] = None

        self._cur_ts = 0

        self._fastrpc_timeout = fastrpc_timeout

    @property
    def is_remote_mounted(self):
        return self._is_remote_mounted

    def set_fall_callback(self, fail_callback: Optional[Callable[[], Coroutine[None, None, Any]]]):
        self._fail_callback = fail_callback

    def set_enable_fallback_layout(self, enable_fallback_layout: bool):
        self._enable_fallback_layout = enable_fallback_layout

    @abc.abstractmethod
    async def setup_remote_object(self):
        ...

    @abc.abstractmethod
    async def shutdown_remote_object(self):
        ...

    @abc.abstractmethod
    async def health_check(self, timeout: Optional[int] = None) -> Any:
        ...

    @abc.abstractmethod
    async def remote_call(self, service_key: str, timeout: Optional[int], /, *args, **kwargs) -> Any:
        ...

    @abc.abstractmethod
    async def remote_generator(self, service_key: str, timeout: Optional[int], /, *args, **kwargs) -> AsyncGenerator[Any, None]:
        yield

    @abc.abstractmethod
    def remote_call_sync(self, service_key: str, timeout: Optional[int], /, *args, **kwargs) -> Any:
        ...

    @abc.abstractmethod
    async def set_fallback_layout(self):
        ...

    def get_url_and_port(self):
        return self._url, self._port

    def set_cur_child_uids(self, cur_child_uids: list[str]):
        self._cur_child_uids = cur_child_uids

    @marker.mark_did_mount
    async def mount_handler(self):
        async with self._mount_lock:
            # avoid unmount during mount.
            await self._reconnect_to_remote_comp()

    @marker.mark_will_unmount
    async def unmount_handler(self):
        async with self._mount_lock:
            if self._is_remote_mounted:
                self._is_remote_mounted = False
                try:
                    # await self.remote_call(serv_names.REMOTE_COMP_UNMOUNT_APP,
                    #                        2, self._key)
                    await self._close_remote_loop()
                finally:
                    await self.shutdown_remote_object()

    async def _remote_msg_handle_loop(self, prefixes: list[str], url: str = "", port: int = -1):
        try:
            aiter_remote = self.remote_generator(serv_names.REMOTE_COMP_MOUNT_APP_GENERATOR, None, 
                        self._key, prefixes, url, port)
            res = await aiter_remote.__anext__()
            layout, root_comp_uid_before = res["layout"], res["remoteRootUid"]
            root_comp_uid = root_comp_uid_before
            layout, root_comp_uid = self._patch_layout_dict(layout, root_comp_uid_before, prefixes)
            self.set_cur_child_uids(list(layout.keys()))
            for k, v in layout.items():
                assert isinstance(v["uid"], UniqueTreeId), f"{layout}"
            # first event: update layout from remote
            update_comp_ev = self.create_update_comp_event(layout, [])
            # second event: update childs prop in remote container
            prop_upd_ev = self.create_update_event({"childs": [UniqueTreeIdForComp(root_comp_uid)]})
            # WARNING: connect button remove container that contains itself,
            # so the actual set_new_layout is delayed after current callback finish.
            # so we can't update stuff here, we must ensure
            # layout update done after set_new_layout.
            # so we use post_ev_creator here.
            await self.set_new_layout({}, post_ev_creator=lambda: update_comp_ev + prop_upd_ev, disable_delay=True)
            self._is_remote_mounted = True
            next_item_task = asyncio.create_task(aiter_remote.__anext__(), name="rcomp-loop-next")
            shutdown_task = asyncio.create_task(self._shutdown_ev.wait(), name="rcomp-shutdown-wait")
            while True:
                try:
                    done, pending = await asyncio.wait(
                        [next_item_task, shutdown_task],
                        return_when=asyncio.FIRST_COMPLETED)
                except asyncio.CancelledError:
                    print("CANCEL")
                    await cancel_task(shutdown_task)
                    await cancel_task(next_item_task)
                    break
                if shutdown_task in done:
                    # shutdown
                    for task in pending:
                        task.cancel()
                    # close aiter_remote
                    # await aiter_remote.aclose()
                    break
                if next_item_task in done:
                    try:
                        ev = next_item_task.result()
                        self._cur_ts = time.time_ns()
                        app = get_app()
                        if isinstance(ev, RemoteCompEvent):
                            key = ev.key
                            await self.run_callback(partial(app.handle_msg_from_remote_comp, key, ev))
                        else:
                            app_event_dict = ev
                            app_event = AppEvent.from_dict(app_event_dict)
                            # ev._remote_prefixes = prefixes
                            app_event.patch_keys_prefix_inplace(prefixes)
                            for _, ui_ev in app_event.type_event_tuple:
                                if isinstance(ui_ev, UpdateComponentsEvent):
                                    assert ui_ev.remote_component_all_childs is not None
                                    ui_ev.remote_component_all_childs = patch_uid_list_with_prefix(
                                        ui_ev.remote_component_all_childs, prefixes)
                                    self.set_cur_child_uids(ui_ev.remote_component_all_childs)
                            app_event_dict = app_event.to_dict()
                            app_event_dict = patch_unique_id(app_event_dict, prefixes)

                            app_ev = AppEvent.from_dict(cast(dict, app_event_dict))
                            await self.put_app_event(app_ev)
                        next_item_task = asyncio.create_task(aiter_remote.__anext__(), name="rcomp-loop-next")
                    except StopAsyncIteration:
                        break

        except:
            traceback.print_exc()
            raise 
        finally:
            # await self.remote_call(serv_names.REMOTE_COMP_UNMOUNT_APP,
            #                         2, self._key)
            await self.disconnect(close_remote_loop=False)
            self._is_remote_mounted = False
            self._remote_task = None
            self._cur_ts = 0

    async def _reconnect_to_remote_comp(self):
        try:
            await self.shutdown_remote_object()
            await self._close_remote_loop()
            assert self._flow_uid is not None, "shouldn't happen"

            # node_uid = get_unique_node_id(master_meta.graph_id,
            #                     master_meta.node_id)
            await self.setup_remote_object()
            prefixes = self._flow_uid.parts
            self._shutdown_ev.clear()
            await self.health_check(1)
            self._remote_task = asyncio.create_task(
                self._remote_msg_handle_loop(prefixes, "", -1), name="remote_msg_handle_loop")
            self._cur_ts = 0
        except BaseException as e:
            await self.send_exception(e)
            await self.disconnect()

    async def _close_remote_loop(self):
        self._shutdown_ev.set()
        if self._remote_task is not None:
            await self._remote_task
            self._remote_task = None

    async def disconnect(self, close_remote_loop: bool = True):
        # try:
        #     await self.remote_call(serv_names.REMOTE_COMP_UNMOUNT_APP,
        #                             2, self._key)
        # except:
        #     traceback.print_exc()
        self._cur_ts = 0
        if close_remote_loop:
            await self._close_remote_loop()
        await self.shutdown_remote_object()
        self._is_remote_mounted = False 
        if self._fail_callback is not None:
            await self._fail_callback()
        if self._enable_fallback_layout:
            await self.set_fallback_layout()

    def get_layout_dict_sync(self) -> tuple[dict[str, Any], str]:
        assert self._flow_uid is not None, "shouldn't happen"
        prefixes = self._flow_uid.parts
        try:
            res = self.remote_call_sync(serv_names.REMOTE_COMP_GET_LAYOUT,
                                        None, self._key, prefixes)
        except:
            traceback.print_exc()
            return {}, ""
        return self._patch_layout_dict(res["layout"],
                                        res["remoteRootUid"], prefixes)

    def _patch_layout_dict(self, layout, remote_root_uid: str, prefixes: list[str]):
        layout_dict = layout
        layout_dict = patch_uid_keys_with_prefix(layout_dict, prefixes)
        for k, v in layout_dict.items():
            layout_dict[k] = patch_unique_id(v, prefixes)
        root_uid = UniqueTreeIdForComp(remote_root_uid)
        root_comp_uid = UniqueTreeIdForComp.from_parts(prefixes + root_uid.parts).uid_encoded
        return layout_dict, root_comp_uid


    async def get_layout_dict(self) -> tuple[dict[str, Any], str]:
        assert self._flow_uid is not None, "shouldn't happen"
        prefixes = self._flow_uid.parts
        try:
            res = await self.remote_call(serv_names.REMOTE_COMP_GET_LAYOUT,
                                         None, self._key, prefixes)
        except:
            traceback.print_exc()
            return {}, ""
        return self._patch_layout_dict(res["layout"],
                                        res["remoteRootUid"], prefixes)

    async def handle_remote_event(self,
                                  ev_data: tuple[str, Any],
                                  is_sync: bool = False):
        # print(ev_data)
        assert self._flow_uid is not None, "shouldn't happen"
        prefixes = self._flow_uid.parts
        ev_data_dict = {
            ev_data[0]: ev_data[1]
        }
        uiev = UIEvent.from_dict(ev_data_dict)
        uiev.unpatch_keys_prefix_inplace(prefixes)
        ev_data_dict = uiev.to_dict()
        while True:
            try:
                return await self.remote_call(
                    serv_names.REMOTE_COMP_RUN_SINGLE_EVENT, self._fastrpc_timeout, self._key,
                    AppEventType.UIEvent.value, ev_data_dict, is_sync)
            except grpc.aio.AioRpcError as e:
                if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    cur_ts = time.time_ns()
                    duration = cur_ts - self._cur_ts
                    duration_second = duration / 1e9
                    if duration_second < self._fastrpc_timeout:
                        LOGGER.warning("remote comp loop is busy, retry...")
                        continue
                await self.send_exception(e)
                await self.disconnect()
                break
            except BaseException as e:
                await self.send_exception(e)
                break

    async def collect_drag_source_data(self,
                                  ev: UIEvent):
        # print(ev_data)
        try:
            # result is returned iff is_sync is True
            res = await self.remote_call(
                serv_names.REMOTE_COMP_RUN_SINGLE_EVENT, self._fastrpc_timeout, self._key,
                AppEventType.UIEvent.value, ev.to_dict(), is_sync=True)
            return list(res.values())[0]
        except grpc.aio.AioRpcError as e:
            await self.send_exception(e)
            await self.disconnect()
        except BaseException as e:
            await self.send_exception(e)
        return None

    async def get_file(self, file_key: str, offset: int, count: Optional[int] = None, chunk_size=2**16):
        async for x in self.remote_generator(serv_names.REMOTE_COMP_GET_FILE, 10, self._key, file_key, offset, count, chunk_size):
            yield x
 
    async def get_file_metadata(self, file_key: str, comp_uid: Optional[str] = None):
        assert self._flow_uid is not None, "shouldn't happen"
        prefixes = self._flow_uid.parts
        if comp_uid is not None:
            comp_uid = unpatch_uid(comp_uid, prefixes)
        return await self.remote_call(serv_names.REMOTE_COMP_GET_FILE_METADATA, self._fastrpc_timeout, self._key, file_key, comp_uid)

    async def send_remote_comp_event(self, key: str, event: RemoteCompEvent):
        return await self.remote_call(serv_names.REMOTE_COMP_RUN_REMOTE_COMP_EVENT, self._fastrpc_timeout, self._key, key, event)

@dataclasses_strict.dataclass
class FragmentProps(ContainerBaseProps):
    disabled: Union[Undefined, bool] = undefined


class Fragment(ContainerBase[FragmentProps, Component]):

    def __init__(self,
                 children: Union[Sequence[Component], Mapping[str, Component]],
                 inited: bool = False) -> None:
        if isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.Fragment, FragmentProps, children, inited)

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


@dataclasses.dataclass
class MatchCaseProps(ContainerBaseProps):
    condition: Union[Undefined, ValueType] = undefined


@dataclasses.dataclass
class MatchCaseItem:
    # if value is undefined, it is default case
    value: Union[ValueType, bool, Undefined]
    child: Component
    isExpr: Union[bool, Undefined] = undefined


@dataclasses.dataclass
class ExprCaseItem:
    value: str
    child: Component
    isExpr: bool = True


@dataclasses.dataclass
class MatchCaseChildDef:
    items: list["Union[MatchCaseItem, ExprCaseItem]"]


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
                 children: Sequence[Union[MatchCaseItem, ExprCaseItem]],
                 init_value: Union[ValueType, Undefined] = undefined) -> None:
        super().__init__(UIType.MatchCase, MatchCaseProps,
                         MatchCaseChildDef(items=[*children]))
        self.props.condition = init_value

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    @property
    def childs_complex(self):
        assert isinstance(self._child_structure, MatchCaseChildDef)
        return self._child_structure

    @staticmethod 
    def binary_selection(success_val: Union[ValueType, bool], success: Component, fail: Optional[Component] = None):
        """Create a simple `MatchCase` that show `success` component when condition is strictly equal to `success_val`, 
        otherwise show `fail`.

        strictly equal in javascript means `===`, so keep in mind that `1` is not equal to `True`. 
        """
        cases: list[MatchCase.Case] = [MatchCase.Case(success_val, success)]
        if fail is not None:
            cases.append(MatchCase.Case(undefined, fail))
        return MatchCase(cases)

    async def set_condition(self, condition: Union[ValueType, Undefined]):
        assert isinstance(self._child_structure, MatchCaseChildDef)
        if isinstance(condition, Undefined):
            return await self.send_and_wait(
                self.update_event(condition=condition))
        has_expr_case = False
        for item in self._child_structure.items:
            if item.value == condition:
                await self.send_and_wait(self.update_event(condition=condition)
                                         )
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
