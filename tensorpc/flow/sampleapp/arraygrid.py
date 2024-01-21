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

import asyncio
import base64
import dataclasses
import enum
import io
import random
import sys
import time
import traceback
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import faker
from tensorpc import PACKAGE_ROOT

import cv2
import imageio
import numpy as np
from faker import Faker
from typing_extensions import Annotated, Literal

import tqdm
import tensorpc
from tensorpc.autossh.scheduler.core import TaskType
from tensorpc.core import prim
from tensorpc.core.asynctools import cancel_task
from tensorpc.core.inspecttools import get_all_members_by_type
from tensorpc.flow import (App, EditableApp, EditableLayoutApp, leaflet,
                           mark_autorun, mark_create_layout, marker, mui,
                           plotly, plus, three, UserObjTree, appctx, V)
from tensorpc.flow.client import AppClient, AsyncAppClient, add_message
from tensorpc.flow.coretypes import MessageLevel, ScheduleEvent
from tensorpc.flow.flowapp.appcore import observe_autorun_function, observe_function, observe_autorun_script
from tensorpc.flow.flowapp.components.mui import (Button, HBox, ListItemButton,
                                                  ListItemText,
                                                  MUIComponentType, VBox,
                                                  VList)
from tensorpc.flow.flowapp.components import typemetas
from tensorpc.flow.flowapp.components.plus.config import ConfigPanel
from tensorpc.flow.sampleapp.sample_reload_fn import func_support_reload
from tensorpc.flow.flowapp.objtree import get_objtree_context
from tensorpc.flow.sampleapp.sample_preview import TestPreview0
from tensorpc.flow.flowapp.components.plus.arraygrid import NumpyArrayGrid

class MatrixDataGridAppV1:    
    @marker.mark_create_layout
    def my_layout(self):
        arr = np.random.uniform(0, 1, size=[100, 3])
        arr2 = np.random.randint(0, 100, size=[100, 1]).astype(np.int64)
        column_def = mui.DataGrid.ColumnDef(id=f"unused", specialType=mui.DataGridColumnSpecialType.Number, width=80, specialProps=mui.DataGridColumnSpecialProps(mui.DataGridNumberCell(fixed=8)))
        custom_footers = [
            mui.MatchCase([
                mui.MatchCase.Case("index", mui.Typography("Max")),
                mui.MatchCase.Case(mui.undefined, mui.Typography().set_override_props(value="data").prop(enableTooltipWhenOverflow=True, tooltipEnterDelay=400, fontSize="12px")),
            ]).set_override_props(condition="condition")
        ]
        custom_footer_datas = [{
            "a-0": str(arr.max(0)[0]),
            "a-1": str(arr.max(0)[1]),
            "a-2": str(arr.max(0)[2]),
            "b-0": str(arr2.max(0)[0]),
        }]
        dgrid = mui.MatrixDataGrid(column_def, {"a": arr, "b": arr2}, 
            customFooters=custom_footers, 
            customFooterDatas=custom_footer_datas,
        )
        dgrid.prop(rowHover=True, virtualized=True, enableFilter=True, tableLayout="fixed")
        dgrid.prop(tableSxProps={
            '& .MuiTableCell-sizeSmall': {
                "padding": '2px 2px',
            },
        })
        return mui.VBox([
            dgrid.prop(stickyHeader=False, virtualized=True, size="small"),
        ]).prop(width="100%", height="100%", overflow="hidden")

class MatrixDataGridApp:    
    @marker.mark_create_layout
    def my_layout(self):
        arr = np.random.uniform(0, 1, size=[1, 3, 20000, 3])
        arr2 = np.random.randint(0, 1000, size=[1, 3, 20000, 4])
        arr3 = np.random.randint(0, 254, size=[1, 3, 20000, 1]).astype(np.uint8)
        arr.reshape(-1)[-2] = np.nan
        arr.reshape(-1)[-1] = np.inf

        grid = NumpyArrayGrid({
            "a": arr,
            "b": arr2,
            "c": arr3,
        })
        return mui.VBox([
            grid.prop(flex=1),
        ]).prop(width="100%", height="100%", overflow="hidden")


class MatrixDataGridContainerApp:    
    @marker.mark_create_layout
    def my_layout(self):
        btn = mui.Button("Edit").prop(loading=False)
        self.grid_container = mui.HBox([])
        dialog = mui.Dialog([
            self.grid_container.prop(flex=1, height="70vh", width="70vw")
        ]).prop(title="Array Viewer", maxWidth="xl")
        self.dialog = dialog
        btn.event_click.on_standard(self._on_btn_select).configure(stop_propagation=True)
        column_defs = [
            mui.DataGrid.ColumnDef("name", accessorKey="name"),
            mui.DataGrid.ColumnDef("triggerDialogBtn", cell=btn),
        ]

        arr = np.random.uniform(0, 1, size=[1, 3, 20000, 3])
        arr2 = np.random.randint(0, 1000, size=[1, 3, 20000, 4])
        arr3 = np.random.randint(0, 254, size=[1, 3, 20000, 1]).astype(np.uint8)
        arr.reshape(-1)[-2] = np.nan
        arr.reshape(-1)[-1] = np.inf
        arr_item1 = {
            "a": arr,
            "b": arr2,
            "c": arr3,
        }
        arr_item2 = arr3
        self.arr_items = [arr_item1, arr_item2]
        dgrid = mui.DataGrid(column_defs, [{
            "id": "0", 
            "index": 0,
            "name": "item1",
        }, {
            "id": "1", 
            "index": 1,
            "name": "item2",
        }]).prop(idKey="id", rowHover=True, virtualized=True, enableFilter=True)

        return mui.VBox([
            dgrid.prop(flex=1),
            dialog,
        ]).prop(width="100%", height="100%", overflow="hidden")

    async def _on_btn_select(self, event: mui.Event):
        indexes = event.indexes
        idx = indexes[0]
        item = self.arr_items[idx]
        await self.grid_container.set_new_layout([
             NumpyArrayGrid(item).prop(width="100%", height="100%", overflow="hidden")
        ])
        await self.dialog.set_open(True)
