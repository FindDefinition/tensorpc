import asyncio
import enum
from functools import partial
import inspect
import traceback
from typing import Any, Callable, Optional, Union
from typing_extensions import Literal
from tensorpc.autossh.core import EofEvent, ExceptionEvent, RawEvent, SSHClient, SSHRequest, SSHRequestType, LOGGER
from tensorpc.autossh.core import Event as SSHEvent

import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.flow.core.common import handle_standard_event
from tensorpc.flow.core.component import FrontendEventType, UIType
from tensorpc.flow.jsonlike import Undefined, undefined

from .mui import (MUIBasicProps, MUIComponentBase, FlexBoxProps, NumberType,
                  Event)


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
    cursorStyle: Union[Literal["block", "underline", "bar"],
                       Undefined] = undefined

    cursorWidth: Union[NumberType, Undefined] = undefined
    cursorInactiveStyle: Union[Literal["block", "underline", "bar"],
                               Undefined] = undefined
    customGlyphs: Union[bool, Undefined] = undefined
    disableStdin: Union[bool, Undefined] = undefined
    drawBoldTextInBrightColors: Union[bool, Undefined] = undefined
    fastScrollModifier: Union[Literal["alt", "ctrl", "shift", "none"],
                              Undefined] = undefined
    fastScrollSensitivity: Union[NumberType, Undefined] = undefined
    fontSize: Union[NumberType, Undefined] = undefined
    fontFamily: Union[str, Undefined] = undefined
    fontWeight: Union[Literal["normal", "bold"], NumberType,
                      Undefined] = undefined
    fontWeightBold: Union[Literal["normal", "bold"], NumberType,
                          Undefined] = undefined
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
    ClearAndWrite = 2

@dataclasses.dataclass
class TerminalResizeEvent:
    width: int
    height: int

class Terminal(MUIComponentBase[TerminalProps]):

    def __init__(self, init_data: Optional[Union[bytes, str]] = None, callback: Optional[Callable[[Union[str, bytes]], Any]] = None) -> None:
        super().__init__(UIType.Terminal,
                         TerminalProps,
                         allowed_events=[
                             FrontendEventType.TerminalInput.value,
                             FrontendEventType.TerminalResize.value,
                             FrontendEventType.TerminalSaveState.value,
                         ])
        if init_data is not None:
            self.prop(initData=init_data)
        self.event_terminal_input = self._create_event_slot(
            FrontendEventType.TerminalInput)
        self.event_terminal_resize = self._create_event_slot(
            FrontendEventType.TerminalResize, lambda x: TerminalResizeEvent(**x))
        self.event_terminal_save_state = self._create_event_slot(
            FrontendEventType.TerminalSaveState)
        self.event_terminal_save_state.on(self._default_on_save_state)

        if callback is not None:
            self.event_terminal_input.on(callback)

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

    async def clear(self):
        await self.put_app_event(
            self.create_comp_event({
                "type": TerminalEventType.ClearAndWrite.value,
                "data": ""
            }))

    async def clear_and_write(self, content: Union[str, bytes]):
        await self.put_app_event(
            self.create_comp_event({
                "type": TerminalEventType.ClearAndWrite.value,
                "data": content
            }))

    async def send_raw(self, data: bytes):
        await self.put_app_event(
            self.create_comp_event({
                "type": TerminalEventType.Raw.value,
                "data": data
            }))
    
    async def send_eof(self):
        await self.put_app_event(
            self.create_comp_event({
                "type": TerminalEventType.Eof.value,
                "data": ""
            }))


class AsyncSSHTerminal(Terminal):

    def __init__(self,
                 init_data: Optional[Union[bytes, str]] = None,
                 connect_when_mount: bool = False,
                 url: str = "",
                 username: str = "",
                 password: str = "") -> None:
        super().__init__(init_data)
        if not url or not username or not password:
            assert not connect_when_mount, "Cannot connect when mount without url, username, and password."
        self._shutdown_ev = asyncio.Event()
        self._client = SSHClient(url, username, password)
        self._cur_inp_queue = asyncio.Queue()
        self.event_after_mount.on(self._on_mount)
        self.event_after_unmount.on(self._on_unmount)
        self._connect_when_mount = connect_when_mount
        self.event_terminal_input.on(self._on_input)
        self.event_terminal_resize.on(self._on_resize)
        self._ssh_task = None

        self._init_size: Optional[TerminalResizeEvent] = None

    async def connect(self,
                      event_callback: Optional[Callable[[SSHEvent],
                                                        None]] = None):
        assert self._client.url and self._client.username and self._client.password, "Cannot connect without url, username, and password."
        await self.connect_with_new_info(self._client.url,
                                         self._client.username,
                                         self._client.password, event_callback)

    async def _on_exit(self):
        if self._ssh_task is not None:
            # we can't await task here because it will cause deadlock
            self._ssh_task = None

    async def connect_with_new_info(
            self,
            url: str,
            username: str,
            password: str,
            event_callback: Optional[Callable[[SSHEvent], None]] = None):
        assert self._ssh_task is None, "Cannot connect with new info while the current connection is still active."
        self._client = SSHClient(url, username, password)
        await self.clear()
        self._shutdown_ev.clear()
        sd_task = asyncio.create_task(self._shutdown_ev.wait())
        self._cur_inp_queue = asyncio.Queue()
        if self._init_size is not None:
            await self._cur_inp_queue.put(
                SSHRequest(
                    SSHRequestType.ChangeSize,
                    [self._init_size.width, self._init_size.height]))
        self._ssh_task = asyncio.create_task(
            self._client.connect_queue(self._cur_inp_queue,
                                       partial(self._handle_ssh_queue,
                                               user_callback=event_callback),
                                       shutdown_task=sd_task,
                                       request_pty=True,
                                       exit_callback=self._on_exit,
                                       term_type="xterm-256color",
                                       enable_raw_event=True))

    async def disconnect(self):
        self._shutdown_ev.set()
        if self._ssh_task is not None:
            await self._ssh_task
            self._ssh_task = None

    async def _on_mount(self):
        if self._connect_when_mount:
            await self.connect()
        else:
            await self.clear_and_write("disconnected.")

    async def _on_unmount(self):
        await self.disconnect()
        await self.clear_and_write("disconnected.")
        self._init_size = None

    async def _on_input(self, data: Union[str, bytes]):
        if self._ssh_task is not None:
            await self._cur_inp_queue.put(data)

    async def _on_resize(self, data: TerminalResizeEvent):
        self._init_size = data
        if self._ssh_task is not None:
            await self._cur_inp_queue.put(
                SSHRequest(SSHRequestType.ChangeSize,
                           [data.width, data.height]))

    async def _handle_ssh_queue(self,
                                event: SSHEvent,
                                user_callback: Optional[Callable[[SSHEvent],
                                                                 Any]] = None):
        assert self._cur_inp_queue is not None
        if not isinstance(event, RawEvent) and user_callback is not None:
            try:
                res = user_callback(event)
                if inspect.iscoroutine(res):
                    await res
            except:
                traceback.print_exc()
        if isinstance(event, RawEvent):
            await self.send_raw(event.raw)
        elif isinstance(event, (EofEvent, ExceptionEvent)):
            if isinstance(event, ExceptionEvent):
                LOGGER.error(event.traceback_str)
            else:
                LOGGER.warning(event)
            if self.is_mounted():
                await self.send_eof()
