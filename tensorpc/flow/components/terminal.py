import asyncio
import enum
from typing import Any, Optional, Union
from typing_extensions import Literal
from tensorpc.autossh.core import EofEvent, ExceptionEvent, RawEvent, SSHClient, SSHRequest, SSHRequestType
from tensorpc.autossh.core import Event as SSHEvent

import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.flow.core.common import handle_standard_event
from tensorpc.flow.core.component import FrontendEventType, UIType
from tensorpc.flow.jsonlike import Undefined, undefined

from .mui import (MUIBasicProps, MUIComponentBase, FlexBoxProps, NumberType, Event)


@dataclasses.dataclass
class TerminalProps(MUIBasicProps):
    initData: Union[str, bytes, Undefined] = undefined
    boxProps: Union[FlexBoxProps, Undefined] = undefined
    theme: Union[Literal["light", "dark"], Undefined] = undefined

    allowProposedApi: Union[bool, Undefined] = undefined
    allowTransparency: Union[bool, Undefined] = undefined
    altClickMovesCursor: Union[bool, Undefined] = undefined
    convertEol: Union[bool, Undefined] = undefined
    cursorBlink: Union[bool, Undefined] = undefined
    cursorStyle: Union[Literal["block", "underline", "bar"], Undefined] = undefined

    cursorWidth: Union[NumberType, Undefined] = undefined
    cursorInactiveStyle: Union[Literal["block", "underline", "bar"], Undefined] = undefined
    customGlyphs: Union[bool, Undefined] = undefined
    disableStdin: Union[bool, Undefined] = undefined
    drawBoldTextInBrightColors: Union[bool, Undefined] = undefined
    fastScrollModifier: Union[Literal["alt", "ctrl", "shift", "none"], Undefined] = undefined
    fastScrollSensitivity: Union[NumberType, Undefined] = undefined
    fontSize: Union[NumberType, Undefined] = undefined
    fontFamily: Union[str, Undefined] = undefined
    fontWeight: Union[Literal["normal", "bold"], NumberType, Undefined] = undefined
    fontWeightBold: Union[Literal["normal", "bold"], NumberType, Undefined] = undefined
    ignoreBracketedPasteMode: Union[bool, Undefined] = undefined
    letterSpacing: Union[NumberType, Undefined] = undefined
    lineHeight: Union[NumberType, Undefined] = undefined
    macOptionIsMeta: Union[bool, Undefined] = undefined
    macOptionClickForcesSelection: Union[bool, Undefined] = undefined
    minimumContrastRatio: Union[NumberType, Undefined] = undefined
    rescaleOverlappingGlyphs: Union[bool, Undefined] = undefined
    rightClickSelectsWord: Union[bool, Undefined] = undefined
    screenReaderMode: Union[bool, Undefined] = undefined
    scrollback: Union[NumberType, Undefined] = undefined
    scrollOnUserInput: Union[bool, Undefined] = undefined
    scrollSensitivity: Union[NumberType, Undefined] = undefined
    smoothScrollDuration: Union[NumberType, Undefined] = undefined
    tabStopWidth: Union[NumberType, Undefined] = undefined
    wordSeparator: Union[str, Undefined] = undefined
    overviewRulerWidth: Union[NumberType, Undefined] = undefined


class TerminalEventType(enum.IntEnum):
    Raw = 0
    Eof = 1


class Terminal(MUIComponentBase[TerminalProps]):
    def __init__(self, init_data: Optional[Union[bytes, str]] = None) -> None:
        super().__init__(UIType.Terminal, TerminalProps, allowed_events=[
            FrontendEventType.TerminalInput.value,
            FrontendEventType.TerminalResize.value,
            FrontendEventType.TerminalSaveState.value,

        ])
        if init_data is not None:
            self.prop(initData=init_data)
        self.event_terminal_input = self._create_event_slot(FrontendEventType.TerminalInput)
        self.event_terminal_resize = self._create_event_slot(FrontendEventType.TerminalResize)
        self.event_terminal_save_state = self._create_event_slot(FrontendEventType.TerminalSaveState)
        self.event_terminal_save_state.on(self._default_on_save_state)

    def _default_on_save_state(self, state):
        self.props.initData = state

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync=is_sync)

class AsyncSSHTerminal(Terminal):
    def __init__(self, url: str, username: str, password: str, init_data: Optional[Union[bytes, str]] = None, 
                connect_when_mount: bool = True) -> None:
        super().__init__(init_data)
        self._shutdown_ev = asyncio.Event()
        self._client = SSHClient(url, username, password)
        self._cur_inp_queue = asyncio.Queue()
        self.event_after_mount.on(self._on_mount)
        self.event_after_unmount.on(self._on_unmount)
        self._connect_when_mount = connect_when_mount
        self.event_terminal_input.on(self._on_input)
        self.event_terminal_resize.on(self._on_resize)
        self._ssh_task = None

    async def _on_mount(self):
        if self._connect_when_mount:
            await self.connect()

    async def connect(self):
        self._shutdown_ev.clear()
        sd_task = asyncio.create_task(self._shutdown_ev.wait())
        self._cur_inp_queue = asyncio.Queue()
        self._ssh_task = asyncio.create_task(
            self._client.connect_queue(self._cur_inp_queue,
                                      self._handle_ssh_queue,
                                      shutdown_task=sd_task,
                                      request_pty=True,
                                      enable_raw_event=True))

    async def disconnect(self):
        self._shutdown_ev.set()
        if self._ssh_task is not None:
            await self._ssh_task
            self._ssh_task = None

    async def _on_unmount(self):        
        await self.disconnect()

    async def _on_input(self, data):
        if self._ssh_task is not None:
            await self._cur_inp_queue.put(data)

    async def _on_resize(self, data):
        if self._ssh_task is not None:
            await self._cur_inp_queue.put(SSHRequest(SSHRequestType.ChangeSize, [data["width"], data["height"]]))

    async def _handle_ssh_queue(self, event: SSHEvent):
        assert self._cur_inp_queue is not None 
        if isinstance(event, RawEvent):
            await self.put_app_event(self.create_comp_event({
                "type": TerminalEventType.Raw,
                "data": event.raw
            }))
        elif isinstance(event, (EofEvent, ExceptionEvent)):
            await self.put_app_event(self.create_comp_event({
                "type": TerminalEventType.Eof,
                "data": ""
            }))
