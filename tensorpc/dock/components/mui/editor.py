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

from typing import cast
from functools import partial

import tensorpc.core.dataclass_dispatch as dataclasses
import enum
from tensorpc.core.datamodel.events import DraftChangeEvent, DraftChangeEventHandler, DraftEventType

from typing import (TYPE_CHECKING, Any,
                    Awaitable, Callable, Coroutine, Dict, Iterable, List,
                    Optional, Set, Tuple, Type, TypeVar, Union)

from typing_extensions import Literal, TypeAlias, TypedDict, Self
from pydantic import field_validator, model_validator

from tensorpc.core.datamodel.draft import DraftBase, insert_assign_draft_op
from tensorpc.dock.core.appcore import Event, get_batch_app_event
from tensorpc.dock.core.common import (handle_standard_event)
from .core import FlexComponentBaseProps, MUIComponentType, MUIContainerBase
from ...core.component import (
    Component, ContainerBaseProps, DraftOpUserData, 
    FrontendEventType, NumberType, UIType,
    Undefined, ValueType, undefined)
from ...core.datamodel import DataModel

class MonacoKeyMod(enum.IntEnum):
    CtrlCmd = 0
    Alt = 1
    Shift = 2
    WinCtrl = 3


@dataclasses.dataclass
class MonacoEditorAction:
    id: str
    label: str
    keybindings: Optional[List[Tuple[List[MonacoKeyMod], int]]] = None
    precondition: Optional[str] = None
    keybindingContext: Optional[str] = None
    contextMenuGroupId: Optional[str] = None
    contextMenuOrder: Optional[NumberType] = None
    userdata: Optional[Any] = None

@dataclasses.dataclass
class MonacoBreakpoint:
    enabled: bool 
    lineNumber: int

    @model_validator(mode="after")
    def _check_bkpt_valid(self) -> Self:
        assert self.lineNumber > 0, "lineNumber must be greater than 0"
        return self

@dataclasses.dataclass
class MonacoConstrainedRange:
    range: tuple[int, int, int, int]
    label: str
    allowMultiline: Union[bool, Undefined] = undefined
    decorationOptions: Union[Undefined, "MonacoModelDecoration"] = undefined


@dataclasses.dataclass
class MonacoEditorOptions:
    # you need to enable glyphMargin to use breakpoints.
    glyphMargin: Union[bool, Undefined] = undefined
    wordWrap: Union[Literal["off", "on", "wordWrapColumn", "bounded"], Undefined] = undefined
    wrappingIndent: Union[Literal["none", "same", "indent", "deepIndent"], Undefined] = undefined
    wrappingStrategy: Union[Literal["simple", "advanced"], Undefined] = undefined
    wordWrapColumn: Union[int, Undefined] = undefined

@dataclasses.dataclass
class MonacoEditorProps(FlexComponentBaseProps, ContainerBaseProps):
    value: Union[str, Undefined] = undefined
    language: Union[str, Undefined] = undefined
    path: Union[str, Undefined] = undefined
    debounce: Union[NumberType, Undefined] = undefined
    readOnly: Union[bool, Undefined] = undefined

    actions: Union[list[MonacoEditorAction], Undefined] = undefined
    options: Union[MonacoEditorOptions, Undefined] = undefined
    bkpts: Union[list[MonacoBreakpoint], Undefined] = undefined
    line: Union[int, Undefined] = undefined
    enableConstrainedEditing: Union[bool, Undefined] = undefined
    constrainedRanges: Union[list[MonacoConstrainedRange], Undefined] = undefined

class _MonacoEditorControlType(enum.IntEnum):
    SetLineNumber = 0
    Save = 1
    SetValue = 2
    SetDecoration = 3
    # only valid for constrained model.
    ToggleHighlightOfEditableArea = 4

@dataclasses.dataclass
class MonacoPosition:
    lineNumber: int
    column: int

@dataclasses.dataclass
class MonacoRange:
    startLineNumber: int
    startColumn: int
    endLineNumber: int
    endColumn: int


@dataclasses.dataclass
class MonacoSaveEvent:
    value: str
    saveVersionId: int
    viewState: Any
    userdata: Optional[Any] = None
    # let user know which path/lang is saved
    lang: Optional[str] = None 
    path: Optional[str] = None 
    decorationsRanges: Optional[dict[str, List[MonacoRange]]] = None
    constrainedValues: Optional[dict[str, str]] = None

@dataclasses.dataclass
class MonacoSelection(MonacoRange):
    selectionStartLineNumber: int
    selectionStartColumn: int
    positionLineNumber: int
    positionColumn: int

@dataclasses.dataclass
class MonacoHoverQueryEvent:
    position: MonacoPosition

@dataclasses.dataclass
class MonacoInlayHintQueryEvent:
    range: MonacoRange
    value: Optional[str] = None

@dataclasses.dataclass
class MonacoMarkdownString:
    value: str 

@dataclasses.dataclass
class MonacoHover:
    contents: list[MonacoMarkdownString]
    range: Union[Undefined, MonacoRange] = undefined

@dataclasses.dataclass
class MonacoInlayHint:
    label: str
    position: MonacoPosition
    tooltip: Union[Undefined, str] = undefined
    # 1: Type, 2: Parameter
    kind: Union[Undefined, Literal[1, 2]] = undefined 
    paddingLeft: Union[Undefined, bool] = undefined
    paddingRight: Union[Undefined, bool] = undefined

@dataclasses.dataclass
class MonacoInlayHintList:
    hints: list[MonacoInlayHint]

class MonacoMinimapPosition(enum.IntEnum):
    Inline = 1
    Gutter = 2

class MonacoMinimapSectionHeaderStyle(enum.IntEnum):
    Normal = 1
    Underlined = 2


@dataclasses.dataclass
class MonacoModelDecorationMinimapOptions:
    position: MonacoMinimapPosition
    sectionHeaderStyle: Union[MonacoMinimapSectionHeaderStyle, Undefined] = undefined
    sectionHeaderText: Union[str, Undefined] = undefined


@dataclasses.dataclass
class MonacoModelDecoration:
    className: Union[Undefined, str] = undefined 
    glyphMarginClassName: Union[Undefined, str] = undefined 
    inlineClassName: Union[Undefined, str] = undefined 
    isWholeLine: Union[Undefined, bool] = undefined
    glyphMarginHoverMessage: Union[Undefined, MonacoMarkdownString, list[MonacoMarkdownString]] = undefined
    hoverMessage: Union[Undefined, MonacoMarkdownString, list[MonacoMarkdownString]] = undefined
    lineNumberHoverMessage: Union[Undefined, MonacoMarkdownString, list[MonacoMarkdownString]] = undefined
    zIndex: Union[Undefined, int] = undefined
    linesDecorationsClassName: Union[Undefined, str] = undefined
    marginClassName: Union[Undefined, str] = undefined
    minimap: Union[Undefined, MonacoModelDecorationMinimapOptions] = undefined

@dataclasses.dataclass
class MonacoModelDeltaDecoration:
    range: MonacoRange
    options: MonacoModelDecoration

@dataclasses.dataclass
class MonacoSelectionEvent:
    selections: List[MonacoSelection]
    selectedCode: str
    source: str

@dataclasses.dataclass
class _MonacoDecorationChangeEventRaw:
    affectsMinimap: bool 
    affectsGlyphMargin: bool
    affectsOverviewRuler: bool
    affectsLineNumber: bool

@dataclasses.dataclass
class MonacoDecorationChangeEvent:
    ranges: list[MonacoRange]
    collection: str 
    raw: _MonacoDecorationChangeEventRaw

@dataclasses.dataclass
class MonacoActionEvent:
    action: str
    selection: Optional[MonacoSelectionEvent]
    userdata: Optional[Any] = None


_T = TypeVar("_T")

class MonacoEditor(MUIContainerBase[MonacoEditorProps, MUIComponentType]):
    @dataclasses.dataclass
    class InlineComponent:
        comp: Component
        afterLineNumber: int 
        afterColumn: Union[int, Undefined] = undefined 
        # TODO add enum for this
        afterColumnAffinity: Union[int, Undefined] = undefined 
        showInHiddenAreas: Union[bool, Undefined] = undefined 
        ordinal: Union[int, Undefined] = undefined 
        suppressMouseDown: Union[bool, Undefined] = undefined 
        heightInLines: Union[int, Undefined] = undefined 
        heightInPx: Union[int, Undefined] = undefined 
        minWidthInPx: Union[int, Undefined] = undefined 

    @dataclasses.dataclass
    class ChildDef:
        icomps: dict[str, "MonacoEditor.InlineComponent"]

        def get_child_component_checked(self, key: str, type: type[_T]) -> _T:
            """Get child component by key, and check its type.
            If the type is not matched, raise TypeError.
            """
            if key not in self.icomps:
                raise KeyError(f"Child component {key} not found")
            comp = self.icomps[key]
            if not isinstance(comp.comp, type):
                raise TypeError(
                    f"Child component {key} is not of type {type.__name__}")
            return comp.comp

    def __init__(self, value: str, language: str, path: str, icomps: Optional[dict[str, "MonacoEditor.InlineComponent"]] = None) -> None:
        all_evs = [
            FrontendEventType.Change.value,
            FrontendEventType.EditorQueryState.value,
            FrontendEventType.EditorSave.value,
            FrontendEventType.EditorSaveState.value,
            FrontendEventType.ComponentReady.value,
            FrontendEventType.EditorAction.value,
            FrontendEventType.EditorCursorSelection.value,
            FrontendEventType.EditorInlayHintsQuery.value,
            FrontendEventType.EditorHoverQuery.value,
            FrontendEventType.EditorCodelensQuery.value,
            FrontendEventType.EditorBreakpointChange.value,
        ]
        super().__init__(UIType.MonacoEditor, MonacoEditorProps, MonacoEditor.ChildDef(icomps or {}),
            allowed_events=all_evs)
        self.props.language = language
        self.props.path = path
        self.props.value = value
        self._init_value = value
        self._init_language = language
        self._init_path = path
        self._view_state = None
        self._save_version_id: Optional[int] = None
        self.register_event_handler(FrontendEventType.EditorSaveState.value,
                                    self._default_on_save_state)
        self.register_event_handler(FrontendEventType.EditorQueryState.value,
                                    self._default_on_query_state)
        self.register_event_handler(FrontendEventType.EditorBreakpointChange.value,
                                    self._default_on_bkpt_change)

        self.event_change = self._create_event_slot(FrontendEventType.Change)
        self.event_editor_save = self._create_event_slot(
            FrontendEventType.EditorSave,
            converter=lambda x: MonacoSaveEvent(**x))
        self.event_component_ready = self._create_event_slot_noarg(
            FrontendEventType.ComponentReady)
        self.event_editor_action = self._create_event_slot(
            FrontendEventType.EditorAction,
            converter=lambda x: MonacoActionEvent(**x))
        self.event_editor_save.on(self._default_on_editor_save)
        self.event_editor_cursor_selection = self._create_event_slot(
            FrontendEventType.EditorCursorSelection,
            converter=lambda x: MonacoSelectionEvent(**x))
        self.event_editor_inlay_hints_query = self._create_event_slot(
            FrontendEventType.EditorInlayHintsQuery,
            lambda x: MonacoInlayHintQueryEvent(**x))
        self.event_editor_hover_query = self._create_event_slot(
            FrontendEventType.EditorHoverQuery,
            lambda x: MonacoHoverQueryEvent(**x))
        self.event_editor_codelens_query = self._create_event_slot(
            FrontendEventType.EditorCodelensQuery)
        self.event_editor_decoration_change = self._create_event_slot(
            FrontendEventType.EditorDecorationsChange,
            converter=lambda x: MonacoDecorationChangeEvent(**x))
        self.event_editor_breakpoint_change = self._create_event_slot(
            FrontendEventType.EditorBreakpointChange,
            converter=lambda x: [MonacoBreakpoint(**b) for b in x])

    @property
    def childs_complex(self):
        assert isinstance(self._child_structure, MonacoEditor.ChildDef)
        return self._child_structure

    def state_change_callback(
            self,
            value: dict[str, Any],
            type: ValueType = FrontendEventType.Change.value):
        self.props.value = value["value"]
        self._view_state = value["viewState"]

    def _default_on_save_state(self, state):
        self._view_state = state["viewState"]
        self._save_version_id = state.get("saveVersionId", None)

    def _default_on_editor_save(self, ev: MonacoSaveEvent):
        self._save_version_id = ev.saveVersionId
        self._view_state = ev.viewState

    def _default_on_query_state(self):
        res = {}
        if self._view_state is not None:
            res["viewState"] = self._view_state
        if self._save_version_id is not None:
            res["saveVersionId"] = self._save_version_id
        return res

    def _default_on_bkpt_change(self, ev: list[MonacoBreakpoint]):
        # don't send to frontend here, frontend prop only used
        # to sync bkpts from backend to frontend.
        self.props.bkpts = ev

    async def handle_event(self, ev: Event, is_sync: bool = False):
        if ev.type == FrontendEventType.EditorChange.value:
            sync_state_after_change = True 
        else:
            sync_state_after_change = False
        return await handle_standard_event(self, ev, is_sync=is_sync, sync_state_after_change=sync_state_after_change, change_status=False)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def set_line_number(self, lineno: int, select_line: Optional[bool] = None):
        ev_dict = {
            "type":
            int(_MonacoEditorControlType.SetLineNumber),
            "value":
            lineno,
        }
        if select_line is not None:
            ev_dict["selectLine"] = select_line
        ev = self.create_comp_event(ev_dict)
        await self.send_and_wait(ev)

    async def save(self, userdata: Optional[Any] = None):
        """Tell Editor in frontend to save.
        userdata will be passed back in the save event.
        You can use userdata to implement logic such as 
        save and run.
        """
        data = {
            "type": int(_MonacoEditorControlType.Save),
        }
        if userdata is not None:
            data["userdata"] = userdata
        ev = self.create_comp_event(data)
        await self.send_and_wait(ev)

    async def set_decorations(self, coll: Literal["common", "debug", "breakpoint", "git"], decorations: List[MonacoModelDeltaDecoration]):
        assert coll in ("common", "debug", "breakpoint", "git"), "collection must be one of common, debug, breakpoint, git"
        ev_dict = {
            "type":
            int(_MonacoEditorControlType.SetDecoration),
            "collection": coll,
            "decorations": decorations,
        }
        ev = self.create_comp_event(ev_dict)
        await self.send_and_wait(ev)

    async def toggle_editable_areas(self, css_single_line: Optional[str] = None, css_multi_line: Optional[str] = None):
        ev_dict: dict[str, Any] = {
            "type":
            int(_MonacoEditorControlType.ToggleHighlightOfEditableArea),
        }
        if css_single_line is not None:
            ev_dict["cssClassForSingleLine"] = css_single_line
        if css_multi_line is not None:
            ev_dict["cssClassForMultiLine"] = css_multi_line
        ev = self.create_comp_event(ev_dict)
        await self.send_and_wait(ev)

    async def set_breakpoints(self, bkpts: list[MonacoBreakpoint]):
        """set bkpt from backend.
        this method won't trigger frontend event EditorBreakpointChange.
        """
        self.prop(bkpts=bkpts)
        await self.send_and_wait(self.update_event(bkpts=bkpts))

    async def write(self,
                    content: str,
                    path: Optional[str] = None,
                    language: Optional[str] = None,
                    line: Optional[int] = None,
                    constrained_ranges: Optional[list[MonacoConstrainedRange]] = None,
                    use_comp_event: bool = True):
        if not use_comp_event:
            line_val = undefined if line is None else line
            await self.send_and_wait(
                self.update_event(value=content,
                                  path=path or undefined,
                                  language=language or undefined,
                                  line=line_val))
        else:
            data = {
                "type": int(_MonacoEditorControlType.SetValue),
                "value": content,
            }
            self.prop(value=content)
            if path is not None:
                self.prop(path=path)
                data["path"] = path
            if language is not None:
                self.prop(language=language)
                data["language"] = language
            if line is not None:
                self.prop(line=line)
                data["line"] = line
            if constrained_ranges is not None:
                self.prop(constrainedRanges=constrained_ranges)
                data["constrainedRanges"] = constrained_ranges
            ev = self.create_comp_event(data)
            await self.send_and_wait(ev)

    def _handle_draft_change(self,
                             draft_ev: DraftChangeEvent,
                             lang_modifier: Optional[Callable[[str],
                                                              str]] = None,
                             path_modifier: Optional[Callable[[str],
                                                              str]] = None):
        batch_ev = get_batch_app_event()
        modified_props = {}
        if draft_ev.is_item_changed("language"):
            language = draft_ev.new_value_dict["language"]
            if language is None:
                # evaluate failed, use a default value
                language = self._init_language
            else:
                if lang_modifier is not None:
                    language = lang_modifier(language)
            modified_props["language"] = language
        if draft_ev.is_item_changed("path"):
            path = draft_ev.new_value_dict["path"]
            if path is None:
                # evaluate failed, use a default value
                path = self._init_path
            else:
                if path_modifier is not None:
                    path = path_modifier(path)
            modified_props["path"] = path
        if draft_ev.is_item_changed("value"):
            value = draft_ev.new_value_dict["value"]
            if value is None:
                # evaluate failed, use a default value
                value = self._init_value
            modified_props["value"] = value
        if draft_ev.is_item_changed("line"):
            value = draft_ev.new_value_dict["line"]
            if value is None:
                # evaluate failed, use a default value
                value = self._init_value
            modified_props["line"] = value

        batch_ev += (self.update_event(**modified_props))

    def _handle_editor_save_for_draft(self, ev: MonacoSaveEvent,
                                      draft: Any,
                                      handler: DraftChangeEventHandler,
                                      save_event_prep: Optional[Callable[[MonacoSaveEvent], None]] = None):
        # we shouldn't trigger draft change handler when we save value directly from editor.
        with DataModel.add_disabled_handler_ctx([handler]):
            if save_event_prep is not None:
                save_event_prep(ev)
            insert_assign_draft_op(draft, ev.value)

    def bind_draft_change_uncontrolled(
            self,
            draft: Any,
            path_draft: Optional[Any] = None,
            lang_draft: Optional[Any] = None,
            path_modifier: Optional[Callable[[str], str]] = None,
            lang_modifier: Optional[Callable[[str], str]] = None,
            line_draft: Optional[Any] = None,
            save_event_prep: Optional[Callable[[MonacoSaveEvent], None]] = None):
        assert not self.is_mounted(), "must be called when unmount"
        assert isinstance(draft, DraftBase)
        assert isinstance(draft._tensorpc_draft_attr_userdata, DraftOpUserData), "you must use comp.get_draft_target() to get draft"
        model_comp: DataModel = cast(Any, draft._tensorpc_draft_attr_userdata.component)
        assert isinstance(model_comp, DataModel)
        draft_dict: Dict[str, Any] = {"value": draft}
        if path_draft is not None:
            assert isinstance(path_draft, DraftBase)
            draft_dict["path"] = path_draft
        if lang_draft is not None:
            assert isinstance(lang_draft, DraftBase)
            draft_dict["language"] = lang_draft
        if line_draft is not None:
            assert isinstance(line_draft, DraftBase)
            draft_dict["line"] = line_draft
        handler, _ = model_comp.install_draft_change_handler(
            draft_dict,
            partial(self._handle_draft_change,
                    lang_modifier=lang_modifier,
                    path_modifier=path_modifier),
            installed_comp=self)
        self.event_editor_save.on(
            partial(self._handle_editor_save_for_draft,
                    draft=draft,
                    handler=handler,
                    save_event_prep=save_event_prep))

