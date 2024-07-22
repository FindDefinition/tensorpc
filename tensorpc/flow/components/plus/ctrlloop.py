import asyncio
import contextvars
import dataclasses
import enum
from functools import partial
import gc
import inspect
import time
from types import FrameType
from typing import AsyncGenerator, Awaitable, Callable, ContextManager, Coroutine, Iterable, Optional, List, Any, TypeVar
import contextlib
from typing_extensions import ParamSpec
from tensorpc import compat
from tensorpc.core import prim
from tensorpc.core.event_emitter.aio import AsyncIOEventEmitter
from tensorpc.core.inspecttools import get_co_qualname_from_frame
from tensorpc.flow import marker
from tensorpc.flow.appctx.core import run_in_executor
from tensorpc.flow.components import mui
import collections.abc

from tensorpc.flow.core.appcore import run_coro_sync
from tensorpc.flow.core.component import EventSlotEmitter
from tensorpc.flow.core.context import ALL_APP_CONTEXT_GETTERS


@dataclasses.dataclass
class LoopEvent:
    index: int
    level: int
    caller_frame: FrameType
    key: Optional[str] = None

    def copy(self):
        return LoopEvent(self.index, self.level, self.caller_frame, self.key)


class ControlledLoopContext:
    def __init__(self, loop_comp: "ControlledLoop",
                 shutdown_ev: asyncio.Event) -> None:
        self.loop_comp = loop_comp
        self.shutdown_ev = shutdown_ev


CONTROLLED_LOOP_CONTEXT_VAR: contextvars.ContextVar[
    Optional[ControlledLoopContext]] = contextvars.ContextVar(
        "computeflow_node_context", default=None)


def get_controlled_loop_context() -> Optional[ControlledLoopContext]:
    return CONTROLLED_LOOP_CONTEXT_VAR.get()


@contextlib.contextmanager
def enter_controlled_loop_context_object(ctx: ControlledLoopContext):
    token = CONTROLLED_LOOP_CONTEXT_VAR.set(ctx)
    try:
        yield ctx
    finally:
        CONTROLLED_LOOP_CONTEXT_VAR.reset(token)


ALL_APP_CONTEXT_GETTERS.add(
    (get_controlled_loop_context, enter_controlled_loop_context_object))


class LoopState(enum.IntEnum):
    Idle = 0
    Running = 1
    Paused = 2


P = ParamSpec('P')

T = TypeVar('T')


class ControlledEventType(enum.Enum):
    Start = "ControlledEventStart"
    Stop = "ControlledEventStop"
    Paused = "ControlledEventPaused"
    IterationEnd = "ControlledEventIterationEnd"


class ControlledLoopItem(mui.FlexBox):
    def __init__(self,
                 level: int,
                 inc_numbers: Optional[List[int]] = None,
                 key: Optional[str] = None,
                 event_emitter: Optional[AsyncIOEventEmitter] = None):
        if inc_numbers is None:
            inc_numbers = [1, 10, 100]
        self.level = level
        self._btn = mui.IconButton(mui.IconType.PlayArrow,
                                   self._ctrl_btn_cb).prop(size="small",
                                                           disabled=True,
                                                           iconSize="small")
        self.prog = mui.CircularProgress(0).prop(size=30,
                                                 variant="determinate")

        self.prog.prop(position="absolute", top=0, left=0)
        self.prog_container = mui.HBox([
            self.prog,
            self._btn,
        ])
        self._detail_info = mui.Typography().prop(variant="caption", enableTooltipWhenOverflow=True)
        self.detail_info_container = mui.HBox([
            self._detail_info
        ]).prop(alignItems="center", maxWidth="200px")
        self.prog_container.prop(position="relative")
        self._stop_btm = mui.IconButton(mui.IconType.Stop,
                                        self._stop_cb).prop(size="small",
                                                            disabled=True,
                                                            iconSize="small")
        inc_btns: List[mui.Button] = []
        for inc in inc_numbers:
            assert inc > 0
            btn = mui.Button(f"+{inc}",
                             partial(self._inc_cb,
                                     inc_num=inc)).prop(size="small",
                                                        variant="outlined",
                                                        disabled=True)
            inc_btns.append(btn)
        self._loop_state = LoopState.Idle
        self._btn_group = mui.ButtonGroup(inc_btns)
        super().__init__(
            [
                mui.HBox([self.prog_container, self._stop_btm, self._btn_group]).prop(alignItems="center"),
                self.detail_info_container,
            ])
        self.prop(flexDirection="column")
        self._pause_event = asyncio.Event()
        self._cur_inc_remain = -1
        self._event_emitter = self.flow_event_emitter
        if event_emitter is not None:
            self._event_emitter = event_emitter
        self.event_loop_start: EventSlotEmitter[
            LoopEvent] = self._create_emitter_event_slot(
                ControlledEventType.Start.value)
        self.event_loop_stop: EventSlotEmitter[
            LoopEvent] = self._create_emitter_event_slot(
                ControlledEventType.Stop.value)
        self.event_loop_iter_end: EventSlotEmitter[
            LoopEvent] = self._create_emitter_event_slot(
                ControlledEventType.IterationEnd.value)
        self.event_loop_paused: EventSlotEmitter[
            LoopEvent] = self._create_emitter_event_slot(
                ControlledEventType.Paused.value)

        self._lock = asyncio.Lock()

        self._need_to_stop = False
        self.key = key

    async def _inc_cb(self, inc_num: int):
        if self._loop_state == LoopState.Paused:
            self._cur_inc_remain = inc_num
            # trigger running
            await self._ctrl_btn_cb()

    async def _stop_cb(self):
        self._need_to_stop = True
        self._pause_event.set()

    async def _ctrl_btn_cb(self):
        async with self._lock:
            if self._loop_state == LoopState.Idle:
                await self._set_ui_based_on_state(LoopState.Idle)
            elif self._loop_state == LoopState.Running:
                self._pause_event.clear()
                await self._set_ui_based_on_state(LoopState.Paused)
                self._loop_state = LoopState.Paused
            elif self._loop_state == LoopState.Paused:
                self._pause_event.set()
                await self._set_ui_based_on_state(LoopState.Running)
                self._loop_state = LoopState.Running

    async def _set_ui_based_on_state(self,
                                     state: LoopState,
                                     clear_prog_when_idle: bool = True):
        if state == LoopState.Idle:
            ev = self._btn.update_event(icon=mui.IconType.PlayArrow,
                                        disabled=True)
            ev += (self._stop_btm.update_event(disabled=True))
            for btn in self._btn_group._child_comps.values():
                assert isinstance(btn, mui.Button)
                ev += (btn.update_event(disabled=True))
            if clear_prog_when_idle:
                ev += self.prog.update_event(value=0)
            await self.send_and_wait(ev)
        elif state == LoopState.Running:
            ev = self._btn.update_event(icon=mui.IconType.Pause,
                                        disabled=False)
            ev += (self._stop_btm.update_event(disabled=False))
            for btn in self._btn_group._child_comps.values():
                assert isinstance(btn, mui.Button)
                ev += (btn.update_event(disabled=True))
            ev += self.prog.update_event(value=0)
            await self.send_and_wait(ev)
        elif state == LoopState.Paused:
            ev = self._btn.update_event(icon=mui.IconType.PlayArrow,
                                        disabled=False)
            ev += (self._stop_btm.update_event(disabled=False))
            for btn in self._btn_group._child_comps.values():
                assert isinstance(btn, mui.Button)
                ev += (btn.update_event(disabled=False))
            await self.send_and_wait(ev)

    async def _ctrl_loop(
        self,
        iterator: Iterable[T],
        shutdown_ev: asyncio.Event,
        caller_frame: FrameType,
        total: Optional[int] = None,
        default_pause: bool = False,
        report_duration: float = 0.2,
    ) -> AsyncGenerator[T, None]:
        size = -1
        if total is not None:
            size = total
        elif isinstance(iterator, collections.abc.Sized):
            size = len(iterator)
        shutdown_ev_task = asyncio.create_task(shutdown_ev.wait())
        loop_ev = LoopEvent(index=0,
                            level=self.level,
                            caller_frame=caller_frame,
                            key=self.key)
        if default_pause:
            await self._set_ui_based_on_state(LoopState.Paused)
            self._loop_state = LoopState.Paused
            self._pause_event.clear()
        else:
            await self._set_ui_based_on_state(LoopState.Running)
            self._loop_state = LoopState.Running
            self._pause_event.set()
        try:
            cnt = 0
            last_report_ts = time.time()
            caller_qname = get_co_qualname_from_frame(caller_frame)
            if self.key is None:
                await self._detail_info.write(caller_qname)
            else:
                await self._detail_info.write(f"{self.key}|{caller_qname}")
            await self._event_emitter.emit_async(
                ControlledEventType.Start.value,
                mui.Event(ControlledEventType.Start.value, loop_ev))
            for data in iterator:
                if self._cur_inc_remain > 0:
                    self._cur_inc_remain -= 1
                if self._cur_inc_remain == 0:
                    self._cur_inc_remain = -1
                    # trigger pause
                    await self._ctrl_btn_cb()
                yield data
                loop_ev = loop_ev.copy()
                loop_ev.index = cnt

                if self._event_emitter.has_event_handlers(
                        ControlledEventType.IterationEnd.value):
                    await self._event_emitter.emit_async(
                        ControlledEventType.IterationEnd.value,
                        mui.Event(ControlledEventType.IterationEnd.value,
                                  loop_ev))

                # await asyncio.sleep(0)
                if not self._pause_event.is_set(): 
                    if self._event_emitter.has_event_handlers(
                        ControlledEventType.Paused.value):
                        await self._event_emitter.emit_async(
                            ControlledEventType.Paused.value,
                            mui.Event(ControlledEventType.Paused.value,
                                    loop_ev))
                done, pending = await asyncio.wait(
                    [
                        shutdown_ev_task,
                        asyncio.create_task(self._pause_event.wait())
                    ],
                    return_when=asyncio.FIRST_COMPLETED)
                if shutdown_ev_task in done:
                    raise ValueError("Stop from app shutdown.")
                if self._need_to_stop:
                    self._need_to_stop = False
                    # user stop imdicates the whole loop should stop.
                    raise ValueError("Stop from User.")
                cnt += 1
                # state sync
                if last_report_ts + report_duration < time.time():
                    # do report
                    last_report_ts = time.time()

                    if size > 0:
                        await self.prog.update_value(value=cnt / size * 100)
            if size > 0:
                await self.prog.update_value(value=100)
            loop_ev = loop_ev.copy()
            loop_ev.index = cnt
            await self._event_emitter.emit_async(
                ControlledEventType.Stop.value,
                mui.Event(ControlledEventType.Stop.value, loop_ev))
        finally:
            shutdown_ev_task.cancel()
            self._loop_state = LoopState.Idle
            # remain progress unclear to indicate user the last state of this task.
            if self.is_mounted():
                # due to unspecific order of generator free,
                # the ui may be unmounted when enter finally.
                # so we need to check it.
                await self._set_ui_based_on_state(LoopState.Idle,
                                                clear_prog_when_idle=False)
            self._cur_inc_remain = -1
            self._need_to_stop = False


class ControlledLoop(mui.FlexBox):
    Event = LoopEvent

    def __init__(self, inc_numbers: Optional[List[int]] = None):
        first_item = ControlledLoopItem(0, inc_numbers)
        super().__init__({
            "0": first_item,
        })
        first_item._event_emitter = self.flow_event_emitter
        self._init_inc_numbers = inc_numbers
        self.prop(flexDirection="column")
        self._item_lock = asyncio.Lock()
        self._cur_loop_count = 0
        self.event_loop_start: EventSlotEmitter[
            LoopEvent] = self._create_emitter_event_slot(
                ControlledEventType.Start.value)
        self.event_loop_stop: EventSlotEmitter[
            LoopEvent] = self._create_emitter_event_slot(
                ControlledEventType.Stop.value)
        self.event_loop_iter_end: EventSlotEmitter[
            LoopEvent] = self._create_emitter_event_slot(
                ControlledEventType.IterationEnd.value)

    @contextlib.contextmanager
    def enter_controlled_loop_ctx(self, shutdown_ev: asyncio.Event):
        ctx = ControlledLoopContext(self, shutdown_ev)
        token = CONTROLLED_LOOP_CONTEXT_VAR.set(ctx)
        try:
            yield ctx
        finally:
            CONTROLLED_LOOP_CONTEXT_VAR.reset(token)

    async def run_in_executor(self, func: Callable[P, T], *args: P.args,
                              **kwargs: P.kwargs) -> T:
        """run a sync function in executor.
        """
        stev = prim.get_async_shutdown_event()
        try:
            with self.enter_controlled_loop_ctx(stev):
                return await run_in_executor(func, *args, **kwargs)
        finally:
            # if caller of ctrl loop raises, we can't capture
            # exception until the generator instance free,
            # so we run gc collect to force generator free here.
            gc.collect()

    async def _ctrl_loop(
        self,
        iterator: Iterable[T],
        shutdown_ev: asyncio.Event,
        caller_frame: FrameType,
        total: Optional[int] = None,
        default_pause: bool = False,
        report_duration: float = 0.4,
        key: Optional[str] = None,
    ) -> AsyncGenerator[T, None]:
        if self._cur_loop_count == 0:
            child_item = self._child_comps["0"]
            assert isinstance(child_item, ControlledLoopItem)
        else:
            child_item = ControlledLoopItem(
                self._cur_loop_count,
                self._init_inc_numbers,
                key=key,
                event_emitter=self.flow_event_emitter)
            async with self._item_lock:
                await self.update_childs(
                    {str(self._cur_loop_count): child_item})
        try:
            self._cur_loop_count += 1
            async for item in child_item._ctrl_loop(iterator, shutdown_ev,
                                                    caller_frame, total,
                                                    default_pause,
                                                    report_duration):
                yield item
        finally:
            if self._cur_loop_count > 1:
                async with self._item_lock:
                    await self.remove_childs_by_keys(
                        [str(self._cur_loop_count - 1)])

            self._cur_loop_count -= 1


async def _awaitable(aw: Awaitable):
    return await aw


def controlled_loop(iterator: Iterable[T],
                    key: Optional[str] = None,
                    total: Optional[int] = None,
                    default_pause: bool = False) -> Iterable[T]:
    """iterable wrapper that make loop controllable in app.
    WARNING: you must run sync function that contains this wrapper
    in ControlledLoop.run_in_executor.
    Args:
        iterator: the iterator to be wrapped.
        total: the total number of items in the iterator.
    """
    ctx = get_controlled_loop_context()
    if ctx is not None and ctx.loop_comp.is_mounted():
        cur_frame = inspect.currentframe()
        assert cur_frame is not None, "shouldn't happen"

        back_frame = cur_frame.f_back
        assert back_frame is not None, "shouldn't happen"
        aiter_obj = ctx.loop_comp._ctrl_loop(iterator,
                                            ctx.shutdown_ev,
                                            back_frame,
                                            total,
                                            default_pause,
                                            key=key)
        while True:
            try:
                if compat.Python3_10AndLater:
                    yield run_coro_sync(_awaitable(anext(aiter_obj)),
                                        allow_current_thread=False)
                else:
                    yield run_coro_sync(_awaitable(aiter_obj.__anext__()),
                                        allow_current_thread=False)
            except StopAsyncIteration:
                break
    else:
        for x in iterator:
            yield x
        return
