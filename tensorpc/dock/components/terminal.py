import asyncio
from collections import deque
import enum
from functools import partial
import inspect
import traceback
from typing import Any, Callable, Optional, Union
from typing_extensions import Literal
from tensorpc.autossh.constants import TENSORPC_ASYNCSSH_INIT_SUCCESS
from tensorpc.autossh.core import CommandEvent, CommandEventType, EofEvent, ExceptionEvent, LineEvent, LineEventType, LineRawEvent, RawEvent, SSHClient, SSHRequest, SSHRequestType, LOGGER, remove_trivial_r_lines
from tensorpc.autossh.core import Event as SSHEvent
import dataclasses as dataclasses_plain
import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.dock.core.common import handle_standard_event
from tensorpc.dock.core.component import EventSlotEmitter, FrontendEventType, UIType
from tensorpc.dock.jsonlike import Undefined, undefined
from tensorpc.autossh.core import LOGGER as SSH_LOGGER
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

    async def send_raw_may_unmounted(self, data: bytes):
        if self.is_mounted():
            await self.send_raw(data)
        pass 
    
    async def send_eof(self):
        await self.put_app_event(
            self.create_comp_event({
                "type": TerminalEventType.Eof.value,
                "data": ""
            }))


@dataclasses_plain.dataclass
class _AsyncSSHTerminalState:
    inp_queue: asyncio.Queue
    task: asyncio.Task 
    inited: bool 
    pid: int
    size: TerminalResizeEvent
    current_cmd: str = ""

@dataclasses_plain.dataclass
class TerminalCmdCompleteEvent:
    cmd: str 
    return_code: Optional[int] = None

class AsyncSSHTerminal(Terminal):

    def __init__(self,
                 url: str = "",
                 username: str = "",
                 password: str = "",
                 init_data: Optional[Union[bytes, str]] = None,
                 manual_connect: bool = True,
                 manual_disconnect: bool = False,
                 line_raw_ev_max_length: int = 10000,
                 init_size: Optional[tuple[int, int]] = None) -> None:
        super().__init__(init_data)
        if not url or not username or not password:
            assert manual_connect, "Cannot auto connect/disconnect when mount without url, username, and password."
        self._shutdown_ev = asyncio.Event()
        self._client = SSHClient(url, username, password)
        self.event_after_mount.on(self._on_mount)
        self.event_after_unmount.on(self._on_unmount)
        self._manual_connect = manual_connect
        self._manual_disconnect = manual_disconnect
        self.event_terminal_input.on(self._on_input)
        self.event_terminal_resize.on(self._on_resize)

        self._backend_ssh_conn_inited_event_key = "__backend_ssh_conn_inited"
        self._backend_ssh_conn_close_event_key = "__backend_ssh_conn_close"
        self._backend_ssh_cmd_complete_event_key = "__backend_ssh_cmd_complete"

        self.event_ssh_conn_close = self._create_emitter_event_slot_noarg(
            self._backend_ssh_conn_close_event_key)
        self.event_ssh_conn_inited = self._create_emitter_event_slot_noarg(
            self._backend_ssh_conn_inited_event_key)
        self.event_ssh_cmd_complete: EventSlotEmitter[TerminalCmdCompleteEvent] = self._create_emitter_event_slot(
            self._backend_ssh_cmd_complete_event_key)

        self._ssh_state: Optional[_AsyncSSHTerminalState] = None
        self._init_size: Optional[TerminalResizeEvent] = None
        if init_size is not None:
            self._init_size = TerminalResizeEvent(init_size[0], init_size[1])
        self._raw_data_buffer: Optional[bytes] = None
        self._line_raw_event_buffer: deque[bytes] = deque(maxlen=line_raw_ev_max_length)

    async def connect(self,
                      event_callback: Optional[Callable[[SSHEvent],
                                                        None]] = None):
        assert self._client.url and self._client.username and self._client.password, "Cannot connect without url, username, and password."
        await self.connect_with_new_info(self._client.url,
                                         self._client.username,
                                         self._client.password, event_callback)

    async def _on_exit(self):
        if self._ssh_state is not None:
            # we can't await task here because it will cause deadlock
            self._ssh_state = None
        SSH_LOGGER.warning("SSH Exit.")

        await self.flow_event_emitter.emit_async(
            self._backend_ssh_conn_close_event_key,
            Event(self._backend_ssh_conn_close_event_key, None))

    async def connect_with_new_info(
            self,
            url: str,
            username: str,
            password: str,
            event_callback: Optional[Callable[[SSHEvent], None]] = None,
            init_cmds: Optional[list[str]] = None):
        assert self._ssh_state is None, "Cannot connect with new info while the current connection is still active."
        self._client = SSHClient(url, username, password)
        await self.clear()
        self._shutdown_ev.clear()
        sd_task = asyncio.create_task(self._shutdown_ev.wait())
        cur_inp_queue = asyncio.Queue()
        size_state = TerminalResizeEvent(-1, -1)
        if self._init_size is not None:
            size_state = self._init_size
            await cur_inp_queue.put(
                SSHRequest(
                    SSHRequestType.ChangeSize,
                    [self._init_size.width, self._init_size.height]))
        if init_cmds:
            for cmd in init_cmds:
                await cur_inp_queue.put(cmd)
        await cur_inp_queue.put(
            f" echo \"{TENSORPC_ASYNCSSH_INIT_SUCCESS}|PID=$TENSORPC_SSH_CURRENT_PID\"\n"
        )

        ssh_task = asyncio.create_task(
            self._client.connect_queue(cur_inp_queue,
                                       partial(self._handle_ssh_queue,
                                               user_callback=event_callback),
                                       shutdown_task=sd_task,
                                       request_pty=True,
                                       exit_callback=self._on_exit,
                                       term_type="xterm-256color",
                                       enable_raw_event=True))
        self._ssh_state = _AsyncSSHTerminalState(cur_inp_queue, ssh_task, False, -1, size_state)
        return self._ssh_state.inp_queue

    async def disconnect(self):
        self._shutdown_ev.set()
        if self._ssh_state is not None:
            await self._ssh_state.task
            self._ssh_state = None

    async def _on_mount(self):
        if self._line_raw_event_buffer:
            self.props.initData = b"".join(self._line_raw_event_buffer)
            self._line_raw_event_buffer.clear()
        if not self._manual_connect:
            await self.connect()
        # else:
        #     await self.clear_and_write("disconnected.")

    async def _on_unmount(self):
        if not self._manual_disconnect:
            await self.disconnect()
            await self.clear_and_write("disconnected.")

    async def _on_input(self, data: Union[str, bytes]):
        if self._ssh_state is not None:
            assert self._ssh_state.inp_queue is not None 
            await self._ssh_state.inp_queue.put(data)

    async def _on_resize(self, data: TerminalResizeEvent):
        self._size_state = data
        if self._ssh_state is not None:
            assert self._ssh_state.inp_queue is not None 
            await self._ssh_state.inp_queue.put(
                SSHRequest(SSHRequestType.ChangeSize,
                           [data.width, data.height]))

    async def _handle_line_raw_event(self,
                                event: LineRawEvent):
        if event.line_ev_type != LineEventType.EXCEPTION:
            if event.line_ev_type == LineEventType.RBUF_OVERFLOW or event.line_ev_type == LineEventType.INCOMPLETE_START:
                if self._raw_data_buffer is not None:
                    rbuf = self._raw_data_buffer
                    rbuf += event.data
                else:
                    rbuf = event.data
                self._raw_data_buffer = remove_trivial_r_lines(rbuf)
            else:
                if self._raw_data_buffer is not None:
                    self._line_raw_event_buffer.append(self._raw_data_buffer)
                    self._raw_data_buffer = None
                data = event.data
                self._line_raw_event_buffer.append(data)

    async def _handle_ssh_queue(self,
                                event: SSHEvent,
                                user_callback: Optional[Callable[[SSHEvent],
                                                                 Any]] = None):
        assert self._ssh_state is not None
        if not isinstance(event, RawEvent):
            if isinstance(event, LineEvent):
                if not self._ssh_state.inited:
                    text = event.line.decode("utf-8").strip()
                    if text.startswith(f"{TENSORPC_ASYNCSSH_INIT_SUCCESS}|PID="):
                        self._ssh_state.inited = True
                        pid = int(text.split("=")[1])
                        self._ssh_state.pid = pid
                        SSH_LOGGER.warning("SSH init success, pid: %d", pid)
                        await self.flow_event_emitter.emit_async(
                            self._backend_ssh_conn_inited_event_key,
                            Event(self._backend_ssh_conn_inited_event_key, None))
            elif isinstance(event, (CommandEvent)):
                if event.type == CommandEventType.CURRENT_COMMAND:
                    if event.arg is not None:
                        parts = event.arg.decode("utf-8").split(";")
                        self._ssh_state.current_cmd = ";".join(parts[:-1])
                elif event.type == CommandEventType.COMMAND_COMPLETE:
                    return_code = 0
                    if event.arg is not None:
                        return_code = int(event.arg)
                    if self._ssh_state.inited and TENSORPC_ASYNCSSH_INIT_SUCCESS not in self._ssh_state.current_cmd:
                        ev = TerminalCmdCompleteEvent(self._ssh_state.current_cmd, return_code)
                        SSH_LOGGER.warning("Command \"%s\" succeed, retcode: %d", self._ssh_state.current_cmd, return_code)

                        await self.flow_event_emitter.emit_async(
                            self._backend_ssh_cmd_complete_event_key,
                            Event(self._backend_ssh_cmd_complete_event_key, ev))
            if user_callback is not None:
                try:
                    res = user_callback(event)
                    if inspect.iscoroutine(res):
                        await res
                except:
                    traceback.print_exc()
        if isinstance(event, RawEvent):
            if self.is_mounted():
                await self.send_raw(event.raw)
        elif isinstance(event, (EofEvent, ExceptionEvent)):
            if isinstance(event, ExceptionEvent):
                LOGGER.error(event.traceback_str)
            else:
                LOGGER.warning(event)
            if self.is_mounted():
                await self.send_eof()

    def get_current_state(self):
        return self._ssh_state