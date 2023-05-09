# Copyright 2023 Yan Yan
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

import asyncio
import dataclasses
import enum
import inspect
import time
from typing import Any, Callable, Coroutine, Dict, Iterable, List, Optional, Set, Tuple, Union
from typing_extensions import Literal

import numpy as np
from tensorpc.autossh.scheduler.core import Task, TaskStatus
from tensorpc.flow.flowapp import appctx

from tensorpc.flow import marker
from tensorpc.flow.flowapp.components import mui, three
from tensorpc.flow.flowapp.components.mui import LayoutType
from tensorpc.flow.flowapp.core import AppComponentCore, Component, FrontendEventType, UIType
from .options import CommonOptions


from tensorpc.autossh.scheduler import SchedulerClient

_TASK_STATUS_TO_UI_TEXT_AND_COLOR: Dict[TaskStatus, Tuple[str, mui._StdColorNoDefault]] = {
    TaskStatus.Pending: ("Pending", "secondary"),

    TaskStatus.Running: ("Running", "primary"),
    TaskStatus.AlmostFinished: ("Almost Finished", "primary"),
    TaskStatus.AlmostCanceled: ("Almost Cancelled", "primary"),
    TaskStatus.NeedToCancel: ("Want To Cancel", "primary"),

    TaskStatus.Canceled: ("Canceled", "warning"),
    TaskStatus.Failed: ("Failed", "error"),
    TaskStatus.Finished: ("Finished", "success"),

}

class TaskCard(mui.Paper):
    def __init__(self, client: SchedulerClient, task: Task) -> None:
        self.task_id = task.id
        self.task = task
        self.client = client
        self.name = mui.Typography(task.name if task.name else task.id)
        self.status = mui.Typography("unknown")
        self.progress = mui.CircularProgress()
        layout = [
            mui.VBox([
                self.name,
                mui.Divider("horizontal"),
                self.status,
            ]),
            self.progress,
            mui.IconButton(mui.IconType.PlayArrow, self._on_schedule_task).prop(tooltip="Schedule Task"),
            mui.IconButton(mui.IconType.Stop, self._on_soft_cancel_task).prop(tooltip="Soft Cancel Task"),
            mui.IconButton(mui.IconType.Cancel, self._on_cancel_task).prop(tooltip="Cancel Task ⌃C"),
            mui.IconButton(mui.IconType.Delete, self._on_kill_task).prop(tooltip="Kill Task"),

        ]
        super().__init__(layout)
        self.prop(flex_flow="row nowrap")

    async def _on_schedule_task(self):
        await self.client.submit_task(self.task)

    async def _on_soft_cancel_task(self):
        await self.client.soft_cancel_task(self.task_id)

    async def _on_cancel_task(self):
        await self.client.cancel_task(self.task_id)

    async def _on_kill_task(self):
        await self.client.kill_task(self.task_id)

    async def update_task_data(self, task: Task):
        self.task = task
        status_name_color = _TASK_STATUS_TO_UI_TEXT_AND_COLOR[task.state.status]
        ev = self.status.update_event(mui_color=status_name_color[1], value=status_name_color[0])
        ev += self.progress.update_event(value=task.state.progress)
        pass 

class TmuxScheduler(mui.FlexBox):
    pass 