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

import dataclasses
import enum
from functools import partial
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union, Generic
import operator
from typing_extensions import Literal

from tensorpc.flow.flowapp.core import AppEvent
from .. import mui
import inspect

T = TypeVar("T")

_CONFIG_META_KEY = "_tensorpc_config_panel_meta"

_BASE_TYPES = (
    int,
    float,
    bool,
    str,
)


def get_args(t: Any) -> Tuple[Any, ...]:
    return getattr(t, "__args__", None) or ()


def get_origin(tp):
    if tp is Generic:
        return Generic
    return getattr(tp, "__origin__", None)


_BASE_TYPES = (int, float, bool, str)


def _check_is_basic_type(tp):
    origin = get_origin(tp)
    if origin is not None:
        if origin in (list, tuple, dict):
            args = get_args(tp)
            return all(_check_is_basic_type(a) for a in args)
        else:
            return origin in _BASE_TYPES or origin is Literal
    else:
        if inspect.isclass(tp):
            return tp in _BASE_TYPES or issubclass(tp,
                                                   (enum.Enum, enum.IntEnum))
        else:
            return False


@dataclasses.dataclass
class ConfigMeta:
    alias: Optional[str]


@dataclasses.dataclass
class SwitchMeta(ConfigMeta):
    pass


@dataclasses.dataclass
class InputMeta(ConfigMeta):
    multiline: bool
    rows: int
    font_size: mui.ValueType
    font_family: str


@dataclasses.dataclass
class SliderMeta(ConfigMeta):
    begin: mui.NumberType
    end: mui.NumberType
    step: Union[mui.NumberType, mui.Undefined] = mui.undefined


@dataclasses.dataclass
class ControlItemMeta:
    getter: Callable[[], Any]
    setter: Callable[[Any], None]
    compare: Callable[[Any], bool]


_BUILTIN_DCLS_TYPE = set(
    [mui.ControlColorRGB, mui.ControlColorRGBA, mui.ControlVector2, mui.ControlVectorN])


def setattr_single(val, obj, name, mapper: Optional[Callable] = None):
    if mapper is not None:
        setattr(obj, name, mapper(val))
    else:
        setattr(obj, name, val)

def setattr_vector_n(val, obj, name):
    # val: [axis, value]
    val_prev = getattr(obj, name).data.copy()
    val_prev[val[0]] = val[1]
    setattr(obj, name, mui.ControlVectorN(val_prev))

def getattr_single(obj, name):
    return getattr(obj, name)


def compare_single(value, obj, name, mapper: Optional[Callable] = None):
    if mapper is not None:
        return getattr(obj, name) == mapper(value)
    else:
        return getattr(obj, name) == value

def compare_vector_n(value, obj, name):
    # val: [axis, value]
    return getattr(obj, name).data[value[0]] == value[1]

def parse_to_control_nodes(origin_obj, current_obj, current_name: str,
                           obj_uid_to_meta: Dict[str, ControlItemMeta]):
    res_node = mui.ControlNode(id=current_name,
                               name=current_name.split(".")[-1],
                               type=mui.ControlNodeType.Folder.value)
    for f in dataclasses.fields(current_obj):
        next_name = f.name
        if current_name:
            next_name = current_name + "." + f.name
        child_node = mui.ControlNode(id=next_name,
                                     name=f.name,
                                     type=mui.ControlNodeType.Folder.value)
        meta: Optional[ConfigMeta] = None
        if _CONFIG_META_KEY in f.metadata:
            meta = f.metadata[_CONFIG_META_KEY]
        ty = f.type
        if dataclasses.is_dataclass(ty) and ty not in _BUILTIN_DCLS_TYPE:
            res = parse_to_control_nodes(origin_obj,
                                         getattr(current_obj, f.name),
                                         next_name, obj_uid_to_meta)
            res_node.children.append(res)

            continue
        if not _check_is_basic_type(ty) and ty not in _BUILTIN_DCLS_TYPE:
            continue  # TODO add support for simple complex type
        # we support int/float/bool/str
        getter = partial(getattr_single, obj=current_obj, name=f.name)
        comparer = partial(compare_single, obj=current_obj, name=f.name)
        if meta is not None and meta.alias is not None:
            child_node.alias = meta.alias

        if ty is bool:
            # use switch
            # if meta is not None:
            #     assert isinstance(meta, SwitchMeta)
            child_node.type = mui.ControlNodeType.Bool.value
            child_node.initValue = getattr(current_obj, f.name)
            setter = partial(setattr_single,
                             obj=current_obj,
                             name=f.name,
                             mapper=bool)
            # setter = lambda x: setattrV2(current_obj, f.name, bool(x))

        elif ty is int or ty is float:
            # use textfield with number type
            if ty is int:
                setter = partial(setattr_single,
                                 obj=current_obj,
                                 name=f.name,
                                 mapper=int)

                # setter = lambda x: setattrV2(current_obj, f.name, int(x))
            else:
                setter = partial(setattr_single,
                                 obj=current_obj,
                                 name=f.name,
                                 mapper=float)

                # setter = lambda x: setattrV2(current_obj, f.name, float(x))
            if isinstance(meta, SliderMeta):
                child_node.type = mui.ControlNodeType.RangeNumber.value
                child_node.initValue = getattr(current_obj, f.name)
                child_node.min = meta.begin
                child_node.max = meta.end
                child_node.step = meta.step
                if isinstance(meta.step, mui.Undefined) and ty is int:
                    child_node.step = 1
            else:
                child_node.type = mui.ControlNodeType.Number.value
                child_node.initValue = getattr(current_obj, f.name)
                if ty is int:
                    child_node.step = 1
            #     raise NotImplementedError
        elif ty is str:
            # use textfield
            setter = partial(setattr_single,
                             obj=current_obj,
                             name=f.name,
                             mapper=str)

            # setter = lambda x: setattr(current_obj, f.name, str(x))
            child_node.type = mui.ControlNodeType.String.value
            child_node.initValue = getattr(current_obj, f.name)
            if meta is not None and isinstance(meta, InputMeta):
                child_node.rows = meta.multiline

        elif ty is mui.ControlColorRGB or ty is mui.ControlColorRGBA:
            child_node.type = mui.ControlNodeType.Color.value
            child_node.initValue = getattr(current_obj, f.name)
            if ty is mui.ControlColorRGB:
                mapper = lambda x: mui.ControlColorRGB(x["r"], x["g"], x["b"])
            else:
                mapper = lambda x: mui.ControlColorRGBA(
                    x["r"], x["g"], x["b"], x["a"])
            setter = partial(setattr_single,
                             obj=current_obj,
                             name=f.name,
                             mapper=mapper)
            comparer = partial(compare_single,
                               obj=current_obj,
                               name=f.name,
                               mapper=mapper)

        elif ty is mui.ControlVector2:
            child_node.type = mui.ControlNodeType.Vector2.value
            child_node.initValue = getattr(current_obj, f.name)
            mapper = lambda x: mui.ControlVector2(x["x"], x["y"])
            setter = partial(setattr_single,
                             obj=current_obj,
                             name=f.name,
                             mapper=mapper)
            comparer = partial(compare_single,
                               obj=current_obj,
                               name=f.name,
                               mapper=mapper)
        elif ty is mui.ControlVectorN:
            child_node.type = mui.ControlNodeType.VectorN.value
            child_node.count = len(getattr(current_obj, f.name).data)
            child_node.initValue = getattr(current_obj, f.name)
            if isinstance(meta, SliderMeta):
                child_node.min = meta.begin
                child_node.max = meta.end
                child_node.step = meta.step
            else:
                child_node.type = mui.ControlNodeType.Number.value
                child_node.initValue = getattr(current_obj, f.name)
            setter = partial(setattr_vector_n,
                             obj=current_obj,
                             name=f.name)
            comparer = partial(compare_vector_n,
                               obj=current_obj,
                               name=f.name)
        else:
            ty_origin = get_origin(ty)
            # print(ty, ty_origin, type(ty), type(ty_origin))
            if ty_origin is Literal:
                child_node.type = mui.ControlNodeType.Select.value
                child_node.initValue = getattr(current_obj, f.name)
                child_node.selects = list(get_args(ty))
                # setter = lambda x: setattr(current_obj, f.name, x)
                setter = partial(setattr_single, obj=current_obj, name=f.name)

            elif ty_origin is None and issubclass(ty,
                                                  (enum.Enum, enum.IntEnum)):
                child_node.type = mui.ControlNodeType.Select.value
                child_node.initValue = getattr(current_obj, f.name).value
                child_node.selects = list(x.value for x in ty)
                # setter = lambda x: setattr(current_obj, f.name, ty(x))
                setter = partial(setattr_single,
                                 obj=current_obj,
                                 name=f.name,
                                 mapper=ty)

            else:
                continue
                # use textfield with json
                child_node.type = mui.ControlNodeType.String.value
                try:
                    child_node.initValue = json.dumps(
                        getattr(current_obj, f.name))
                except:
                    # ignore field that can't be dumped to json.
                    continue
                if meta is not None and isinstance(meta, InputMeta):
                    child_node.rows = meta.multiline

                setter = partial(setattr_single,
                                 obj=current_obj,
                                 name=f.name,
                                 mapper=json.loads)
        res_node.children.append(child_node)
        obj_uid_to_meta[child_node.id] = ControlItemMeta(
            getter, setter, comparer)
    return res_node


def control_nodes_v1_to_v2(ctrl_node_v1: mui.ControlNode) -> mui.JsonLikeNode:
    childs: List[mui.JsonLikeNode] = [
        control_nodes_v1_to_v2(c) for c in ctrl_node_v1.children
    ]
    # print(ctrl_node_v1.initValue, type(ctrl_node_v1.initValue))
    ctrl_desp = mui.ControlDesp(
        type=ctrl_node_v1.type,
        initValue=ctrl_node_v1.initValue,
        min=ctrl_node_v1.min,
        max=ctrl_node_v1.max,
        step=ctrl_node_v1.step,
        selects=ctrl_node_v1.selects,
        rows=ctrl_node_v1.rows,
        count=ctrl_node_v1.count)
    node = mui.JsonLikeNode(id=ctrl_node_v1.id,
                            name=ctrl_node_v1.name if isinstance(ctrl_node_v1.alias, mui.Undefined) else ctrl_node_v1.alias,
                            type=mui.JsonLikeType.Object.value,
                            typeStr="",
                            children=childs,
                            userdata=ctrl_desp)
    return node


class ConfigPanel(mui.DynamicControls):
    def __init__(self,
                 config_obj: Any,
                 callback: Optional[Callable[[str, Any],
                                             mui._CORO_NONE]] = None):
        assert dataclasses.is_dataclass(config_obj)
        # parse config dataclass.
        self._obj_to_ctrl_meta: Dict[str, ControlItemMeta] = {}
        node = parse_to_control_nodes(config_obj, config_obj, "",
                                      self._obj_to_ctrl_meta)
        super().__init__(init=node.children, callback=self.callback)
        self.__config_obj = config_obj
        self.__callback_key = "config_panel_v2_handler"
        if callback is not None:
            self.register_event_handler(self.__callback_key,
                                        callback,
                                        backend_only=True)

    async def callback(self, value: Tuple[str, Any]):
        print(value)
        uid = value[0]
        cmeta = self._obj_to_ctrl_meta[uid]
        compare_res = cmeta.compare(value[1])
        if not compare_res:
            # here we need to compare value, emit event iff
            # the value is changed.
            # TODO this is due to limitation of leva control.
            # we may need to write own dynamic control
            # based on tanstack table.
            cmeta.setter(value[1])
            handlers = self.get_event_handlers(self.__callback_key)
            if handlers is not None:
                for handler in handlers.handlers:
                    coro = handler.cb(uid, cmeta.getter())
                    if inspect.iscoroutine(coro):
                        await coro

    @property
    def config(self):
        return self.__config_obj

    @staticmethod
    def base_meta(alias: Optional[str] = None):
        return {_CONFIG_META_KEY: ConfigMeta(alias)}

    @staticmethod
    def switch_meta(alias: Optional[str] = None):
        return {_CONFIG_META_KEY: SwitchMeta(alias)}

    @staticmethod
    def input_meta(multiline: bool,
                   rows: int,
                   font_size: mui.ValueType,
                   font_family: str,
                   alias: Optional[str] = None):
        return {
            _CONFIG_META_KEY:
            InputMeta(alias=alias,
                      multiline=multiline,
                      rows=rows,
                      font_size=font_size,
                      font_family=font_family)
        }

    @staticmethod
    def slider_meta(begin: mui.NumberType,
                    end: mui.NumberType,
                    alias: Optional[str] = None):
        return {
            _CONFIG_META_KEY: SliderMeta(begin=begin, end=end, alias=alias)
        }


class ConfigPanelV2(mui.SimpleControls):
    def __init__(self,
                 config_obj: Any,
                 callback: Optional[Callable[[str, Any],
                                             mui._CORO_NONE]] = None):
        assert dataclasses.is_dataclass(config_obj)
        # parse config dataclass.
        self._obj_to_ctrl_meta: Dict[str, ControlItemMeta] = {}
        node = parse_to_control_nodes(config_obj, config_obj, "",
                                      self._obj_to_ctrl_meta)
        super().__init__(init=control_nodes_v1_to_v2(node).children,
                         callback=self.callback)
        self.__config_obj = config_obj
        self.__callback_key = "config_panel_v3_handler"
        if callback is not None:
            self.register_event_handler(self.__callback_key,
                                        callback,
                                        backend_only=True)

    async def callback(self, value: Tuple[str, Any]):
        uid = value[0]
        cmeta = self._obj_to_ctrl_meta[uid]
        compare_res = cmeta.compare(value[1])
        if not compare_res:
            # here we need to compare value, emit event iff
            # the value is changed.
            # TODO this is due to limitation of leva control.
            # we may need to write own dynamic control
            # based on tanstack table.
            cmeta.setter(value[1])
            handlers = self.get_event_handlers(self.__callback_key)
            if handlers is not None:
                for handler in handlers.handlers:
                    coro = handler.cb(uid, cmeta.getter())
                    if inspect.iscoroutine(coro):
                        await coro

    @property
    def config(self):
        return self.__config_obj

    @staticmethod
    def base_meta(alias: Optional[str] = None):
        return {_CONFIG_META_KEY: ConfigMeta(alias)}

    @staticmethod
    def switch_meta(alias: Optional[str] = None):
        return {_CONFIG_META_KEY: SwitchMeta(alias)}

    @staticmethod
    def input_meta(multiline: bool,
                   rows: int,
                   font_size: mui.ValueType,
                   font_family: str,
                   alias: Optional[str] = None):
        return {
            _CONFIG_META_KEY:
            InputMeta(alias=alias,
                      multiline=multiline,
                      rows=rows,
                      font_size=font_size,
                      font_family=font_family)
        }

    @staticmethod
    def slider_meta(begin: mui.NumberType,
                    end: mui.NumberType,
                    step: Union[mui.NumberType, mui.Undefined] = mui.undefined,
                    alias: Optional[str] = None):
        return {
            _CONFIG_META_KEY: SliderMeta(begin=begin, end=end, step=step, alias=alias)
        }
