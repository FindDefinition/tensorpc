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
from functools import partial
import json
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, Generic
import operator
from .. import mui

T = TypeVar("T")

_CONFIG_META_KEY = "_tensorpc_config_panel_meta"

_BASE_TYPES = (int, float, bool, str, )

def get_args(t: Any) -> Tuple[Any, ...]:
    return getattr(t, "__args__", None) or ()

def get_origin(tp):
    if tp is Generic:
        return Generic
    return getattr(tp, "__origin__", None)

_BASE_TYPES = (int, float, bool, str, )

def _check_is_basic_type(tp):
    origin = get_origin(tp)
    if origin is not None:
        if origin in (list, tuple, dict):
            args = get_args(tp)
            return all(_check_is_basic_type(a) for a in args)
        else:
            return tp in _BASE_TYPES
    else:
        return tp in _BASE_TYPES


@dataclasses.dataclass
class ConfigMeta:
    pass


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
    step: mui.NumberType


class ConfigPanel(mui.FlexBox):
    """a composited component that parse dataclasses to config panel.
    supported field: common value type and SINGLE nested dataclass,
    don't support structured nested dataclass member, e.g.
    List[OtherDataClass].
    don't support optional/union.
    WARNING: config object must be singleton. 
    TODO add support for optional (add a checkbox/switch)
    TODO add support for simple structure (List[some_dataclass] or Dict[some_dataclass])
    TODO add support for select (Literal[some_string])

    """
    def __init__(self, config_obj: Any, max_input_rows: int = 3, append_childs: Optional[Dict[str, mui.Component]] = None):
        assert dataclasses.is_dataclass(config_obj)
        # parse config dataclass.
        self.max_input_rows = max_input_rows
        layout_type, nfield_to_comp = self._parse_dataclass_and_bind(
            config_obj, config_obj, "")
        layout: Dict[str, mui.Component] = {**layout_type}
        super().__init__()
        if append_childs is not None:
            layout.update(append_childs)
        self.add_layout(layout)
        self.props.flex_flow = "column nowrap"
        self.__config_obj = config_obj
        self.props.overflow_y = "auto"
        self.props.min_height = 0
        self._nfield_to_comp = nfield_to_comp

    @property 
    def config(self):
        return self.__config_obj

    @staticmethod
    def switch_meta():
        return {
            _CONFIG_META_KEY: SwitchMeta()
        }

    @staticmethod
    def input_meta(multiline: bool, rows: int, font_size: mui.ValueType,
                   font_family: str):
        return {
            _CONFIG_META_KEY: InputMeta(multiline=multiline,
                         rows=rows,
                         font_size=font_size,
                         font_family=font_family)
        }

    @staticmethod
    def slider_meta(begin: mui.NumberType, end: mui.NumberType,
                    step: mui.NumberType):
        return {
            _CONFIG_META_KEY: SliderMeta(begin=begin, end=end, step=step)
        }

    def _sync_config_event(self, current_obj: Any, current_name: str):
        uievent = mui.AppEvent("", {})
        for f in dataclasses.fields(current_obj):
            next_name = f.name
            if current_name:
                next_name = current_name + "." + f.name
            ty = f.type
            if dataclasses.is_dataclass(ty):
                upd = self._sync_config_event(getattr(current_obj, f.name), next_name)
                continue
            if not _check_is_basic_type(ty):
                continue # TODO add support for simple complex type
            comp = self._nfield_to_comp[next_name]
            if ty is bool:
                # use switch
                assert isinstance(comp, mui.SwitchBase)
                upd = comp.update_event(checked=getattr(current_obj, f.name))
            elif ty is int or ty is float or ty is str:
                # use textfield with number type
                assert isinstance(comp, mui.Input)
                upd = comp.update_event(value=getattr(current_obj, f.name))
            else:
                # use textfield with json
                assert isinstance(comp, mui.Input)
                upd = comp.update_event(
                    value=json.dumps(getattr(current_obj, f.name)))
            uievent += upd
        return uievent

    async def sync_config(self):
        return await self.send_app_event_and_wait(self._sync_config_event(self.__config_obj, ""))

    def _parse_dataclass_and_bind(self, origin_obj, current_obj,
                                  current_name: str):
        layout: Dict[str, mui.MUIComponentType] = {}
        nfield_to_comp: Dict[str, mui.MUIComponentType] = {}
        for f in dataclasses.fields(current_obj):
            next_name = f.name
            if current_name:
                next_name = current_name + "." + f.name
            ty = f.type
            if dataclasses.is_dataclass(ty):
                # nested parse
                summary = mui.AccordionSummary(
                    {"t": mui.Typography(f"{f.name}")})
                res = self._parse_dataclass_and_bind(origin_obj, ty, next_name)
                detail = mui.AccordionDetails(res[0])
                nfield_to_comp.update(res[1])
                layout[f.name] = mui.Accordion(
                    summary, detail.prop(padding_left=1, padding_right=1))
                continue
            if not _check_is_basic_type(ty):
                continue # TODO add support for simple complex type
            # we support int/float/bool/str
            meta: Optional[ConfigMeta] = None
            if _CONFIG_META_KEY in f.metadata:
                meta = f.metadata[_CONFIG_META_KEY]

            cb = partial(self._callback,
                         cur_name=current_name,
                         field_name=f.name,
                         origin_obj=origin_obj,
                         type=ty)
            if ty is bool:
                # use switch
                if meta is not None:
                    assert isinstance(meta, SwitchMeta)
                comp = mui.Switch(f.name, cb)
                if f.default != dataclasses.MISSING:
                    comp.props.checked = f.default
            elif ty is int or ty is float:
                # use textfield with number type
                if meta is not None:
                    assert isinstance(meta, (InputMeta, SliderMeta))
                if meta is None or isinstance(meta, InputMeta):
                    if meta is None:
                        comp = mui.Input(f.name, multiline=False,
                                         callback=cb).prop(mui_margin="dense",
                                                           type="number")
                    else:
                        comp = mui.Input(f.name,
                                         multiline=meta.multiline,
                                         callback=cb)
                        comp.prop(mui_margin="dense",
                                  type="number",
                                  rows=meta.rows,
                                  font_family=meta.font_family,
                                  font_size=meta.font_size)
                    if f.default != dataclasses.MISSING:
                        comp.props.value = str(f.default)

                elif isinstance(meta, SliderMeta):
                    comp = mui.Slider(f.name,
                                      meta.begin,
                                      meta.end,
                                      meta.step,
                                      callback=cb)
                    if f.default != dataclasses.MISSING:
                        comp.props.value = f.default
                else:
                    raise NotImplementedError
            elif ty is str:
                # use textfield
                if meta is not None:
                    assert isinstance(meta, InputMeta)

                if meta is None:
                    comp = mui.Input(f.name, multiline=False,
                                     callback=cb).prop(mui_margin="dense",
                                                       type="number")
                else:
                    comp = mui.Input(f.name,
                                     multiline=meta.multiline,
                                     callback=cb)
                    comp.prop(mui_margin="dense",
                              type="number",
                              rows=meta.rows,
                              font_family=meta.font_family,
                              font_size=meta.font_size)

                if f.default != dataclasses.MISSING:
                    comp.props.value = f.default
            else:
                # use textfield with json
                if meta is not None:
                    assert isinstance(meta, InputMeta)
                if meta is None:
                    comp = mui.Input(f.name, multiline=False,
                                     callback=cb).prop(mui_margin="dense",
                                                       type="number")
                else:
                    comp = mui.Input(f.name,
                                     multiline=meta.multiline,
                                     callback=cb)
                    comp.prop(mui_margin="dense",
                              type="number",
                              rows=meta.rows,
                              font_family=meta.font_family,
                              font_size=meta.font_size)
                if f.default != dataclasses.MISSING:
                    comp.props.value = json.dumps(f.default)

            layout[f.name] = comp
            nfield_to_comp[next_name] = comp
        return layout, nfield_to_comp

    @staticmethod
    def _callback(value, origin_obj, cur_name, field_name, type):
        if type is bool:
            value = value
        elif type is int:
            value = int(value)
        elif type is float:
            value = float(value)
        elif type is str:
            value = value
        else:
            value = json.loads(value)
        if cur_name:
            getter = operator.attrgetter(cur_name)
            setattr(getter(origin_obj), field_name, value)
        else:
            setattr(origin_obj, field_name, value)
