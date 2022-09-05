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
from typing import (Any, AsyncGenerator, Awaitable, Callable, Coroutine, Dict,
                    Generic, Iterable, List, Optional, Tuple, Type, TypeVar,
                    Union)
import asyncio
import traceback
import inspect
from ....autossh.core import _cancel
from tensorpc.utils.registry import HashableRegistry
from tensorpc.utils.uniquename import UniqueNamePool
import dataclasses
import re
from tensorpc.compat import Python3_10AndLater
import sys
from typing_extensions import Literal, ParamSpec, Concatenate, Self, TypeAlias, Protocol

ALL_APP_EVENTS = HashableRegistry()

_CORO_NONE = Union[Coroutine[None, None, None], None]

ValueType: TypeAlias = Union[int, float, str]

NumberType: TypeAlias = Union[int, float]

class Undefined:
    pass


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


    # special
    TaskLoop = 0x100
    FlexBox = 0x101
    MUIList = 0x102
    Divider = 0x103

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

    ThreeSimpleGeometry = 0x1101
    ThreeShape = 0x1102

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
    # clipboard
    CopyToClipboard = 20
    # schedule event, won't be sent to frontend.
    ScheduleNext = 100
    # special UI event
    AppEditor = 200


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
                        UISaveStateEvent, NotifyEvent]


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
        # here we don't use dict for typeToEvents due to frontend key type limit.
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
        return

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


# @init_anno_fwd(ComponentBaseProps)
# def open_for_writing(*args, **kwargs):
#     kwargs['mode'] = 'w'
#     return open(*args, **kwargs)

# open_for_writing()
# open_for_writing(file='')

P2 = ParamSpec('P2')
# TProp = TypeVar('TProp', covariant=True)

# class IComponent(Protocol[TProp]):
#     @property
#     def props(self) -> TProp: ...
#     # @property
#     # def prop(self) -> Type[TProp]: ...
#     def get_callback(self) -> Optional[Callable]: ...
#     def set_callback(self, val: Any): ...
#     async def _clear(self): ...
#     def to_dict(self) -> Dict[str, Any]: ...
#     def get_state(self) -> Dict[str, Any]: ...
#     def set_state(self, state: Dict[str, Any]): ...
#     @property
#     def queue(self) -> asyncio.Queue[AppEvent]: ...
#     def state_change_callback(self, data: Any): ...
#     def create_update_event(self, data: Dict[str, Union[Any, Undefined]]): ...
#     async def send_app_event_and_wait(self, ev: AppEvent): ...
#     def create_update_comp_event(self, updates: Dict[str, Any]): ...
#     def create_delete_comp_event(self, deletes: List[str]): ...
#     async def run_callback(self,
#                            cb: Callable[[], _CORO_NONE],
#                            sync_state: bool = False): ...
#     async def sync_status(self,
#                           sync_state: bool = False,
#                           sent_event: Optional[asyncio.Event] = None): ...
#     async def sync_state(self, sent_event: Optional[asyncio.Event] = None): ...
#     def get_sync_event(self, sync_state: bool = False): ...


T_child = TypeVar("T_child")


class Component(Generic[T_base_props, T_child]):
    def __init__(self,
                 uid: str,
                 type: UIType,
                 prop_cls: Type[T_base_props],
                 queue: Optional[asyncio.Queue] = None) -> None:
        self._queue = queue
        self.uid = uid
        self.type = type
        # self._status = UIRunStatus.Stop
        # task for callback of controls
        # if previous control callback hasn't finished yet,
        # the new control event will be IGNORED
        self._task: Optional[asyncio.Task] = None
        self.parent = ""
        self.__props = prop_cls()
        self.__prop_cls = prop_cls

    @property
    def props(self) -> T_base_props:
        return self.__props

    @property
    def propcls(self) -> Type[T_base_props]:
        return self.__prop_cls

    # @property
    # def prop(self) -> Type[T_base_props]:
    #     """set some prop of this component"""
    #     @init_anno_fwd(self.__prop_cls)
    #     def wrapper(**kwargs):
    #         self.__props._tensorpc_component_special_key = self  # type: ignore
    #         for k, v in kwargs.items():
    #             setattr(self.__props, k, v)
    #         return self.__props

    #     return wrapper  # type: ignore

    def _prop_base(self, prop: Callable[P, Any], this: T3) -> Callable[P, T3]:
        def wrapper(*args: P.args, **kwargs: P.kwargs):
            for k, v in kwargs.items():
                setattr(self.__props, k, v)
            return this
        return wrapper 

    def _update_props_base(self, prop: Callable[P, Any]):
        def wrapper(*args: P.args, **kwargs: P.kwargs):
            return self.create_update_event(kwargs)
        return wrapper 


    # def prop3(self):
    #     propcls: Type[T_base_props] = self.propcls
    #     return self.prop2(propcls)

    # @property
    # def prop2(self):
    #     """set some prop of this component"""
    #     @init_anno_fwd(self.__prop_cls, self)
    #     def wrapper(*args, **kwargs):
    #         for k, v in kwargs.items():
    #             setattr(self.__props, k, v)
    #         return self
    #     return wrapper

    def get_callback(self) -> Optional[Callable]:
        return None

    def set_callback(self, val: Any):
        return

    async def handle_event(self, ev: Any):
        pass

    async def _clear(self):
        self.uid = ""
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
        self.parent = ""

    def to_dict(self):
        """undefined will be removed here.
        if you reimplement to_dict, you need to use 
        camel name, no conversion provided.
        """
        props = self.get_props()
        props, und = _split_props_to_undefined(props)
        # state = self.get_state()
        # newstate = {}
        # for k, v in state.items():
        #     if not isinstance(v, Undefined):
        #         newstate[snake_to_camel(k)] = v
        # # TODO better way to resolve type anno problem
        # static, _ = self.__props.get_dict_and_undefined(state)  # type: ignore
        res = {
            "uid": self.uid,
            "type": self.type.value,
            "props": props,
            # "status": self._status.value,
        }
        # static["state"] = newstate
        # static["uid"] = self.uid
        # static["type"] = self.type.value
        return res

    def get_sync_props(self) -> Dict[str, Any]:
        """this function is used for props you want to kept when app
        shutdown or layout updated.
        1. app shutdown: only limited component support props recover.
        2. update layout: all component will override props
        by previous sync props
        """
        return {"status": self.props.status}

    # def set_sync_state(self, state: Dict[str, Any]):
    #     if "status" in state:
    #         self.props.status = state["status"]

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

    @property
    def queue(self):
        assert self._queue is not None, "you must add ui by flexbox.add_xxx"
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
        ev = UIUpdateEvent({self.uid: (data_no_und, data_unds)})
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
        ev = UIUpdateEvent({self.uid: (data_no_und, data_unds)})
        # uid is set in flowapp service later.
        return AppEvent("", {AppEventType.UIUpdatePropsEvent: ev})

    async def send_app_event_and_wait(self, ev: AppEvent):
        if ev.sent_event is None:
            ev.sent_event = asyncio.Event()
        await self.queue.put(ev)
        await ev.sent_event.wait()

    def create_update_comp_event(self, updates: Dict[str, Any]):
        ev = UpdateComponentsEvent(updates)
        # uid is set in flowapp service later.
        return AppEvent("", {AppEventType.UpdateComponents: ev})

    def create_delete_comp_event(self, deletes: List[str]):
        ev = DeleteComponentsEvent(deletes)
        # uid is set in flowapp service later.
        return AppEvent("", {AppEventType.DeleteComponents: ev})

    async def run_callback(self,
                           cb: Callable[[], _CORO_NONE],
                           sync_state: bool = False):
        self.props.status = UIRunStatus.Running.value
        ev = asyncio.Event()
        await self.sync_status(sync_state, ev)
        await ev.wait()
        try:
            coro = cb()
            if inspect.iscoroutine(coro):
                await coro
        except:
            traceback.print_exc()
            # TODO send exception to frontend
            
            raise
        finally:
            self.props.status = UIRunStatus.Stop.value 
            await self.sync_status(sync_state)

    async def sync_status(self,
                          sync_state: bool = False,
                          sent_event: Optional[asyncio.Event] = None):
        if sync_state:
            ev = self.create_update_event(self.get_sync_props())
            ev.sent_event = sent_event
            await self.queue.put(ev)
        else:
            ev = self.create_update_event({"status": self.props.status})
            ev.sent_event = sent_event
            await self.queue.put(ev)

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
                 _init_dict: Optional[Dict[str, T_child]] = None,
                 inited: bool = False) -> None:
        super().__init__(uid, base_type, prop_cls, queue)
        if inited:
            assert queue is not None and uid_to_comp is not None
        if uid_to_comp is None:
            uid_to_comp = {}
        self._pool = UniqueNamePool()
        self._uid_to_comp = uid_to_comp
        self._init_dict = _init_dict

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
            v.uid = new_name
            uid_to_comp[v.uid] = v
            if isinstance(v, ContainerBase):
                v._uid_to_comp = uid_to_comp

    def __getitem__(self, key: str):
        assert key in self.props.childs, f"{key}, {self.props.childs}"
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

    def _get_all_child_comp_uids(self):
        return [self[n].uid for n in self.props.childs]

    def _get_uid_with_ns(self, name: str):
        if not self.inited:
            return (f"{name}")
        return (f"{self.uid}.{name}")

    def _add_prop_to_ui(self, ui: Component):
        if self.inited:
            ui._queue = self.queue
        ui.parent = self.uid
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
        comp.uid = uid
        if add_to_state:
            assert uid not in self._uid_to_comp
        assert comp.parent == "", "this component must not be added before."
        self._add_prop_to_ui(comp)
        if add_to_state:
            self._uid_to_comp[comp.uid] = comp
            self.props.childs.append(name)
        return comp

    def _extract_layout(self, layout: Dict[str, Component]):
        """ get all components without add to app state.
        """
        # we assume layout is a ordered dict (python >= 3.7)
        comps: List[Component] = []
        for k, v1 in layout.items():
            v = v1
            comps.append(self.add_component(k, v, False, anonymous=False))
            if isinstance(v, ContainerBase):
                msg = "you must use VBox/HBox/Box instead in dict layout"
                assert v._init_dict is not None, msg
                comps.extend(v._extract_layout(v._init_dict))
        return comps

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
        # we assume layout is a ordered dict (python >= 3.7)
        for k, v1 in layout.items():
            v = v1
            if not isinstance(v, ContainerBase):
                self.add_component(k, v, anonymous=False)
            else:
                if v._init_dict is not None:
                    # consume this _init_dict
                    v.add_layout(v._init_dict)
                    v._init_dict = None
                # update uids of childs of v,
                ns = f"{k}"
                if self.uid != "":
                    ns = f"{self.uid}.{k}"
                v._update_child_ns_and_comp_dict(ns, self._uid_to_comp)
                if not v.inited and self.inited:
                    v._attach_to_app(self._queue)
                self.add_component(k, v, anonymous=False)

    def get_props(self):
        state = super().get_props()
        state["childs"] = self._get_all_child_comp_uids()
        return state

    async def set_new_layout(self, layout: Dict[str, Component]):
        # remove all first
        # TODO we may need to stop task of a comp
        comps_to_remove: List[Component] = []
        for c in self.props.childs:
            comps_to_remove.extend(self._get_all_nested_child(c))
        for c in comps_to_remove:
            await c._clear()
            self._uid_to_comp.pop(c.uid)
        self.props.childs.clear()
        # TODO should we merge two events to one?
        await self.queue.put(
            self.create_delete_comp_event([c.uid for c in comps_to_remove]))
        self.add_layout(layout)
        comps_frontend = {
            c.uid: c.to_dict()
            for c in self._get_all_nested_childs()
        }
        # make sure all child of this box is rerendered.
        # TODO merge events
        comps_frontend[self.uid] = self.to_dict()
        await self.queue.put(self.create_update_comp_event(comps_frontend))
        await self.queue.put(self.create_update_event({"childs": self.props.childs}))

    async def remove_childs_by_keys(self, keys: List[str]):
        comps_to_remove: List[Component] = []
        for c in keys:
            comps_to_remove.extend(self._get_all_nested_child(c))
        for c in comps_to_remove:
            await c._clear()
            self._uid_to_comp.pop(c.uid)
        for k in keys:
            self.props.childs.remove(k)
        # make sure all child of this box is rerendered.
        await self.queue.put(
            self.create_delete_comp_event([c.uid for c in comps_to_remove]))
        # TODO combine two event to one
        await self.queue.put(
            self.create_update_comp_event({self.uid: self.to_dict()}))
        await self.queue.put(self.create_update_event({"childs": self.props.childs}))

    async def update_childs(self, layout: Dict[str, Component]):
        new_comps = self._extract_layout(layout)
        update_comps_frontend = {c.uid: c.to_dict() for c in new_comps}
        for c in new_comps:
            if c.uid in self._uid_to_comp:
                item = self._uid_to_comp.pop(c.uid)
                item.parent = ""  # mark invalid
            self._uid_to_comp[c.uid] = c
        for n in layout.keys():
            if n not in self.props.childs:
                self.props.childs.append(n)
        # make sure all child of this box is rerendered.
        update_comps_frontend[self.uid] = self.to_dict()
        await self.queue.put(
            self.create_update_comp_event(update_comps_frontend))
        await self.queue.put(self.create_update_event({"childs": self.props.childs}))
