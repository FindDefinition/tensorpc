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
import traceback
from typing import Any, Callable, Coroutine, Dict, Iterable, List, Optional, Set, Tuple, Union
from typing_extensions import Literal

import numpy as np
from tensorpc.autossh.coretypes import SSHTarget
from tensorpc.autossh.scheduler.constants import TMUX_SESSION_NAME_SPLIT, TMUX_SESSION_PREFIX
from tensorpc.autossh.scheduler.core import ResourceType, Task, TaskStatus, TaskType
from tensorpc.flow.flowapp import appctx

from tensorpc.flow import marker
from tensorpc.flow.flowapp.components import mui, three
from tensorpc.flow.flowapp.components.mui import LayoutType
from tensorpc.flow.flowapp.core import AppComponentCore, Component, FrontendEventType, UIType
from .options import CommonOptions

from tensorpc.autossh.scheduler import SchedulerClient

_TASK_STATUS_TO_UI_TEXT_AND_COLOR: Dict[TaskStatus,
                                        Tuple[str, mui._StdColorNoDefault]] = {
                                            TaskStatus.Pending:
                                            ("Pending", "secondary"),
                                            TaskStatus.Running: ("Running",
                                                                 "primary"),
                                            TaskStatus.AlmostFinished:
                                            ("Exiting", "primary"),
                                            TaskStatus.AlmostCanceled:
                                            ("Canceling", "primary"),
                                            TaskStatus.NeedToCancel:
                                            ("Want To Cancel", "primary"),
                                            TaskStatus.Canceled: ("Canceled",
                                                                  "warning"),
                                            TaskStatus.Failed: ("Failed",
                                                                "error"),
                                            TaskStatus.Finished: ("Finished",
                                                                  "success"),
                                        }


class TaskCard(mui.FlexBox):
    def __init__(self, client: SchedulerClient, task: Task) -> None:
        self.task_id = task.id
        self.task = task
        self.client = client
        self.name = mui.Typography(task.name if task.name else task.id)
        self.status = mui.Typography("unknown")
        status_name_color = _TASK_STATUS_TO_UI_TEXT_AND_COLOR[
            task.state.status]
        self.status.prop(mui_color=status_name_color[1],
                         value=status_name_color[0])
        progress_0 = task.state.progress == 0
        self.progress = mui.CircularProgress(task.state.progress * 100).prop(
            variant="indeterminate" if progress_0 else "determinate")
        self.collapse_btn = mui.IconButton(mui.IconType.ExpandMore,
                                           self._on_expand_more).prop(
                                               tooltip="Show Detail",
                                               size="small")
        self.command = mui.Typography(task.command).prop(
            font_size="14px", font_family="monospace", word_break="break-word")

        self.detail = mui.Collapse([
            mui.VBox([self.command]),
        ]).prop(timeout="auto", unmount_on_exit=True)
        self._expanded = False
        layout = [
            mui.VBox([
                mui.HBox([
                    mui.FlexBox([
                        mui.Icon(mui.IconType.DragIndicator).prop(),
                    ]).prop(take_drag_ref=True, cursor="move"),
                    self.name,
                    mui.Chip("copy tmux cmd",
                             self._on_tmux_chip).prop(color="blue",
                                                      size="small",
                                                      margin="0 3px 0 3px"),
                    mui.Chip(f"{task.num_gpu_used} gpus",
                             self._on_tmux_chip).prop(color="green",
                                                      size="small",
                                                      margin="0 3px 0 3px",
                                                      clickable=False),
                ]),
                mui.Divider("horizontal").prop(margin="5px 0 5px 0"),
                self.status,
            ]).prop(flex=1),
            mui.HBox([
                self.progress,
                mui.IconButton(mui.IconType.PlayArrow,
                               self._on_schedule_task).prop(
                                   tooltip="Schedule Task", size="small"),
                mui.IconButton(mui.IconType.Stop,
                               self._on_soft_cancel_task).prop(
                                   tooltip="Soft Cancel Task", size="small"),
                mui.IconButton(mui.IconType.Cancel, self._on_cancel_task).prop(
                    tooltip="Cancel Task ⌃C", size="small"),
                mui.IconButton(mui.IconType.Delete, self._on_kill_task).prop(
                    tooltip="Kill Task",
                    size="small",
                    confirm_message="Are You Sure to Kill This Task?"),
                self.collapse_btn,
            ]).prop(margin="0 5px 0 5px", flex=0),
        ]
        super().__init__([
            mui.Paper([
                mui.VBox([
                    *layout,
                ]).prop(
                    flex_flow="row wrap",
                    align_items="center",
                ),
                self.detail,
            ]).prop(flex_flow="column",
                    padding="5px",
                    margin="5px",
                    elevation=4,
                    flex=1)
        ])
        self.prop(draggable=True,
                  drag_type="TaskCard",
                  drag_in_child=True,
                  sx_over_drop={
                      "opacity": "0.5",
                  })

    async def _on_expand_more(self):
        self._expanded = not self._expanded
        icon = mui.IconType.ExpandLess if self._expanded else mui.IconType.ExpandMore
        await self.send_and_wait(
            self.collapse_btn.update_event(icon=icon.value) +
            self.detail.update_event(triggered=self._expanded))

    async def _on_schedule_task(self):
        await self.client.submit_task(self.task)

    async def _on_tmux_chip(self):
        await appctx.get_app().copy_text_to_clipboard(
            f"tmux attach -t {self.task.get_tmux_session_name()}")

    async def _on_soft_cancel_task(self):
        await self.client.soft_cancel_task(self.task_id)

    async def _on_cancel_task(self):
        await self.client.cancel_task(self.task_id)

    async def _on_kill_task(self):
        await self.client.kill_task(self.task_id)

    def update_task_data_event(self, task: Task):
        self.task = task
        is_running = task.state.status == TaskStatus.Running
        status_name_color = _TASK_STATUS_TO_UI_TEXT_AND_COLOR[
            task.state.status]
        ev = self.status.update_event(mui_color=status_name_color[1],
                                      value=status_name_color[0])
        if is_running:
            progress_0 = task.state.progress == 0
            ev += self.progress.update_event(
                value=task.state.progress * 100,
                variant="indeterminate" if progress_0 else "determinate")
        else:
            ev += self.progress.update_event(value=task.state.progress * 100,
                                             variant="determinate")
        return ev

    async def update_task_data(self, task: Task):
        await self.send_and_wait(self.update_task_data_event(task))


class TmuxScheduler(mui.FlexBox):
    def __init__(
        self, ssh_target: Union[SSHTarget, Callable[[], Coroutine[None, None,
                                                                  SSHTarget]]]
    ) -> None:
        ssh_target_creator: Optional[Callable[[], Coroutine[None, None,
                                                            SSHTarget]]] = None
        if isinstance(ssh_target, SSHTarget):
            self.info = mui.Typography(
                f"SSH: {ssh_target.username}@{ssh_target.hostname}:{ssh_target.port}"
            ).prop(margin="5px", font_size="14px", font_family="monospace")
        else:
            ssh_target_creator = ssh_target
            ssh_target = SSHTarget.create_fake_target()

            self.info = mui.Typography(f"SSH: ").prop(margin="5px",
                                                      font_size="14px",
                                                      font_family="monospace")
        self._ssh_target_creator = ssh_target_creator
        self.tasks = mui.VBox([]).prop(flex=1)
        super().__init__([
            mui.HBox([
                self.info.prop(flex=1),
                mui.Chip("copy tmux cmd",
                         self._on_tmux_chip).prop(color="blue", size="small")
            ]).prop(align_items="center"),
            self.tasks,
        ])
        self.prop(flex_flow="column",
                  overflow="auto",
                  width="100%",
                  height="100%")
        self.client = SchedulerClient(ssh_target)
        self.task_cards: Dict[str, TaskCard] = {}

    async def _on_tmux_chip(self):
        await appctx.get_app().copy_text_to_clipboard(
            f"tmux attach -t {self.client.schr_session_name}")

    def _get_info(self, scheduler_info: str):
        tgt = self.client.ssh_target
        return f"SSH: {tgt.username}@{tgt.hostname}:{tgt.port}, {scheduler_info}"

    async def _get_resource_info(self):
        idle, occupied = await self.client.get_resource_usage()
        num_cpu_idle = len(idle[ResourceType.CPU])
        num_cpu = num_cpu_idle + len(occupied[ResourceType.CPU])
        num_gpu_idle = len(idle[ResourceType.GPU])
        num_gpu = num_gpu_idle + len(occupied[ResourceType.GPU])
        scheduler_info = f"CPU: {num_cpu_idle}/{num_cpu}, GPU: {num_gpu_idle}/{num_gpu}"
        return scheduler_info

    @marker.mark_did_mount
    async def _on_mount(self):
        if self._ssh_target_creator is not None:
            tgt = await self._ssh_target_creator()
            self.client = SchedulerClient(tgt)
        await self.client.async_init()
        tasks = self.client.tasks.values()
        self.task_cards = {
            task.id: TaskCard(self.client, task)
            for task in tasks
        }
        await self.tasks.set_new_layout({**self.task_cards})
        await self.info.write(self._get_info(await self._get_resource_info()))
        self.period_check_task = asyncio.create_task(self._period_check_task())

    @marker.mark_will_unmount
    async def _on_unmount(self):
        await self.client.shutdown_scheduler()
        # self.period_check_task.cancel()
        pass

    async def _period_check_task(self):
        try:
            await asyncio.sleep(1)
            updated, deleted = await self.client.update_tasks()
            await self.info.write(
                self._get_info(await self._get_resource_info()))
            ev = mui.AppEvent("", {})
            new_task_cards = {}
            for updated_task in updated:
                # print(updated_task.id, updated_task.state.status)
                if updated_task.id not in self.task_cards:
                    new_task_cards[updated_task.id] = TaskCard(
                        self.client, updated_task)
                else:
                    # await self.task_cards[updated_task.id].update_task_data(updated_task)
                    ev += self.task_cards[
                        updated_task.id].update_task_data_event(updated_task)
            if updated:
                await self.send_and_wait(ev)
                await self.tasks.update_childs(new_task_cards)
                self.task_cards.update(new_task_cards)
            if deleted:
                await self.tasks.remove_childs_by_keys(deleted)
                for delete in deleted:
                    if delete in self.task_cards:
                        self.task_cards.pop(delete)
        except:
            traceback.print_exc()
            raise
        # tasks = list(self.client.tasks.values())
        # for t in tasks:
        #     print(t.id, t.state.status)
        # await self.set_new_layout([TaskCard(self.client, task) for task in tasks])
        self.period_check_task = asyncio.create_task(self._period_check_task())

    async def submit_task(self, task: Task):
        await self.client.submit_task(task)

    async def submit_func_id_task(self,
                                  func_id: str,
                                  task_id: str = "",
                                  kwargs: Optional[dict] = None,
                                  keep_tmux_session: bool = True):
        if kwargs is None:
            kwargs = {}
        task = Task(TaskType.FunctionId,
                    func_id, [kwargs],
                    id=task_id,
                    keep_tmux_session=keep_tmux_session)
        await self.client.submit_task(task)
