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
import tempfile
import asyncio
import dataclasses
import enum
import inspect
import os
import time
import traceback
from typing import Any, Callable, Coroutine, Dict, Iterable, List, Optional, Set, Tuple, Union
from typing_extensions import Literal

import numpy as np
from tensorpc.core.tracers.codefragtracer import CursorFuncTracer, CodeFragTracerResult
from tensorpc.flow.components import mui
from tensorpc.flow import appctx
from tensorpc.flow.core.appcore import AppSpecialEventType
from tensorpc.flow.core.core import EventSlotEmitter
from tensorpc.flow.coretypes import VscodeTensorpcMessage, VscodeTensorpcMessageType


@dataclasses.dataclass
class VscodeTracerData:
    func: Callable
    args: Tuple
    kwargs: Dict
    run_in_executor: bool = False


class VscodeTracerBox(mui.FlexBox):
    class EventType(enum.Enum):
        TraceStart = "vscode_tracer_box_trace_start"
        TraceEnd = "vscode_tracer_box_trace_end"

    def __init__(self, children: Optional[mui.LayoutType] = None):
        super().__init__(children)

        self.event_trace_start = self._create_emitter_event_slot_noarg(
            VscodeTracerBox.EventType.TraceStart.value)
        self.event_trace_end: EventSlotEmitter[
            Optional[CodeFragTracerResult]] = self._create_emitter_event_slot(
                VscodeTracerBox.EventType.TraceEnd.value)

        self._vscode_handler_registered = False
        self._tracer = CursorFuncTracer()
        self._trace_data: Optional[VscodeTracerData] = None
        self._is_tracing = False
        self.event_before_unmount.on(self.unset_trace_data)

    async def _handle_vscode_message(self, data: VscodeTensorpcMessage):
        if data.type == VscodeTensorpcMessageType.UpdateCursorPosition:
            if self._trace_data is None or self._is_tracing:
                return
            if data.selections is not None and len(
                    data.selections) > 0 and data.currentUri.startswith("file://"):
                path = data.currentUri[7:]
                sel = data.selections[0]
                lineno = sel.start.line + 1
                col = sel.start.character
                end_lineno = sel.end.line + 1
                end_col = sel.end.character
                code_range = (lineno, col, end_lineno, end_col)
                try:
                    self._is_tracing = True
                    await self.flow_event_emitter.emit_async(
                        VscodeTracerBox.EventType.TraceStart.value,
                        mui.Event(VscodeTracerBox.EventType.TraceStart.value,
                                None))
                    if self._trace_data.run_in_executor:
                        res = await asyncio.get_running_loop().run_in_executor(
                            None, self._tracer.run_trace_from_code_range,
                            self._trace_data.func, self._trace_data.args,
                            self._trace_data.kwargs, path, code_range)
                    else:
                        res = self._tracer.run_trace_from_code_range(
                            self._trace_data.func, self._trace_data.args,
                            self._trace_data.kwargs, path, code_range)
                    await self.flow_event_emitter.emit_async(
                        VscodeTracerBox.EventType.TraceEnd.value,
                        mui.Event(VscodeTracerBox.EventType.TraceEnd.value, res))
                except BaseException as exc:
                    traceback.print_exc()
                    await self.send_exception(exc)
                finally:
                    self._is_tracing = False

    async def prepare_trace(self,
                       func: Callable,
                       args: Tuple,
                       kwargs: Dict,
                       run_in_executor: bool = False):
        try:
            self._is_tracing = True
            await self.flow_event_emitter.emit_async(
                VscodeTracerBox.EventType.TraceStart.value,
                mui.Event(VscodeTracerBox.EventType.TraceStart.value,
                            None))
            if run_in_executor:
                await asyncio.get_running_loop().run_in_executor(
                    None, self._tracer.prepare_func_trace, func, args, kwargs)
            else:
                self._tracer.prepare_func_trace(func, args, kwargs)
            await self.flow_event_emitter.emit_async(
                VscodeTracerBox.EventType.TraceEnd.value,
                mui.Event(VscodeTracerBox.EventType.TraceEnd.value, None))
        except BaseException as exc:
            traceback.print_exc()
            await self.send_exception(exc)
        finally:
            self._is_tracing = False
        self._trace_data = VscodeTracerData(func, args, kwargs,
                                            run_in_executor)
        self._register_vscode_handler()

    def unset_trace_data(self):
        self._trace_data = None
        self._unregister_vscode_handler()

    def _register_vscode_handler(self):
        if self._vscode_handler_registered:
            return
        appctx.register_app_special_event_handler(
            AppSpecialEventType.VscodeTensorpcMessage,
            self._handle_vscode_message)
        self._vscode_handler_registered = True

    def _unregister_vscode_handler(self):
        if not self._vscode_handler_registered:
            return
        appctx.unregister_app_special_event_handler(
            AppSpecialEventType.VscodeTensorpcMessage,
            self._handle_vscode_message)
        self._vscode_handler_registered = False
