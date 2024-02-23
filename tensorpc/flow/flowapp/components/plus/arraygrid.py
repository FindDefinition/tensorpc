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
import bisect
import dataclasses
from math import isinf, isnan

from typing import Any, Callable, Coroutine, Dict, Hashable, Iterable, List, Literal, Optional, Set, Tuple, Type, Union

import numpy as np
from tensorpc.core.core_io import _div_up
from tensorpc.core.moduleid import get_qualname_of_type
from tensorpc.flow.flowapp.components import mui, three


def _get_slice_width_by_dim_base_10(dim: int) -> int:
    # base 25px, for every digit, add 10px.
    return 30 + 8 * len(str(dim))


_SLICE_THEME = mui.Theme(typography={"fontSize": 14},
                         components={
                             "MuiInput": {
                                 "defaultProps": {
                                     "type": "number",
                                 },
                                 "styleOverrides": {
                                     "input": {
                                         "fontSize": "12px",
                                     },
                                     "root": {
                                         "padding": "0 2px 0 2px"
                                     }
                                 }
                             }
                         })


def _get_slice_inputs_from_shape(
    shape: List[int], callback: Callable[[str], Any]
) -> Tuple[mui.Component, List[mui.Input], mui.Typography]:
    slice_inputs = []
    slice_inputs.append(mui.Typography(f"shape = {shape}, slice = ["))
    slices: List[mui.Input] = []
    for i in range(len(shape) - 2):
        inp = mui.Input("", init="0", callback=callback).prop(
            width=f"{_get_slice_width_by_dim_base_10(shape[i])}px",
            disabled=shape[i] == 1)
        slice_inputs.append(inp)
        slices.append(inp)
        if i != len(shape) - 1:
            slice_inputs.append(mui.Typography(","))
    typo_last_matrix = mui.Typography(" :, :]")
    slice_inputs.append(typo_last_matrix)

    # slice_inputs.append(mui.Typography("]"))
    return mui.ThemeProvider(
        [mui.HBox(slice_inputs).prop(alignItems="center")],
        _SLICE_THEME), slices, typo_last_matrix


@dataclasses.dataclass
class ArrayMeta:
    min: np.ndarray
    max: np.ndarray
    nanIndices: Optional[np.ndarray]
    infIndices: Optional[np.ndarray]


class NumpyArrayGrid(mui.FlexBox):
    """display last two dims as matrix
    """

    def __init__(self,
                 obj: Union[Dict[str, np.ndarray], np.ndarray],
                 max_columns: int = 25,
                 max_size_row_split: int = 1000000):
        super().__init__()
        self.max_columns = max_columns
        self.max_size_row_split = max_size_row_split
        self.init_add_layout(self.update_numpy_grid(obj))
        self.prop(flexFlow="column nowrap", padding="5px")

    def update_numpy_grid(self, obj: Union[Dict[str, np.ndarray], np.ndarray]):
        if isinstance(obj, np.ndarray):
            obj = {"array": obj}
        obj_atleast_2d: Dict[str, np.ndarray] = {}
        for k, v in obj.items():
            if v.ndim < 2:
                v = v.reshape(-1, 1)
            obj_atleast_2d[k] = v
        mui.MatrixDataGrid._check_data_np_dict(obj_atleast_2d)
        first_obj = next(iter(obj_atleast_2d.values()))
        ndim = first_obj.ndim
        shape = [
            *first_obj.shape[:-1],
            sum([o.shape[-1] for o in obj_atleast_2d.values()])
        ]
        num_slice_inputs = ndim - 2
        slice_inputs_comp, slices, matrix_typo = _get_slice_inputs_from_shape(
            shape, lambda x: self._on_split_slider_change(x))
        self.matrix_typo = matrix_typo
        num_columns_split = _div_up(shape[-1], self.max_columns)
        real_display_columns = min(shape[-1], self.max_columns)
        max_rows_split = _div_up(self.max_size_row_split, real_display_columns)
        num_rows_split = _div_up(shape[-2], max_rows_split)
        real_display_rows = min(shape[-2], max_rows_split)
        self.real_display_columns = real_display_columns
        self.real_display_rows = real_display_rows
        self.column_split_slider = mui.Slider(
            0,
            num_columns_split - 1,
            callback=self._on_split_slider_change,
            label="col: ").prop(
                width="100%",
                valueInput=True,
                marks=True if num_columns_split < 50 else False)
        self.row_split_slider = mui.Slider(
            0,
            num_rows_split - 1,
            callback=self._on_split_slider_change,
            label="row: ").prop(width="100%",
                                valueInput=True,
                                marks=True if num_rows_split < 50 else False)
        self.slider_base_shape = shape[:-2]
        self.obj_all_shape = shape
        self.slices = slices
        self.obj_flatted_3d = {
            k: v.reshape(-1, *v.shape[-2:])
            for k, v in obj_atleast_2d.items()
        }
        self.obj_flatted_min_maxs = self._get_array_min_maxs(
            self.obj_flatted_3d)

        self.num_columns = [o.shape[-1] for o in obj_atleast_2d.values()]
        self.num_columns_cumsum = np.cumsum(self.num_columns).tolist()
        matrix_typo.prop(
            value=f" {0}:{real_display_rows}, {0}:{real_display_columns}]")

        self.matrix_typo = matrix_typo
        init_data_split = self._get_array_dict_from_offset(
            0, 0, real_display_rows, 0, real_display_columns)
        init_footer_data = self._get_footer_data(0)
        custom_footers = [
            mui.MatchCase([
                mui.MatchCase.Case("index", mui.Typography("min")),
                mui.MatchCase.Case(
                    mui.undefined,
                    mui.Typography().set_override_props(value="data").prop(
                        enableTooltipWhenOverflow=True,
                        tooltipEnterDelay=400,
                        fontSize="12px")),
            ]).set_override_props(condition="condition"),
            mui.MatchCase([
                mui.MatchCase.Case("index", mui.Typography("max")),
                mui.MatchCase.Case(
                    mui.undefined,
                    mui.Typography().set_override_props(value="data").prop(
                        enableTooltipWhenOverflow=True,
                        tooltipEnterDelay=400,
                        fontSize="12px")),
            ]).set_override_props(condition="condition"),
        ]

        column_def = mui.DataGrid.ColumnDef(
            id=f"unused",
            specialType=mui.DataGridColumnSpecialType.Number,
            width=75,
            specialProps=mui.DataGridColumnSpecialProps(
                mui.DataGridNumberCell(precision=6)))
        dgrid = mui.MatrixDataGrid(
            column_def,
            {**init_data_split},
            customFooters=[*custom_footers],
            customFooterDatas=init_footer_data,
        )
        dgrid.prop(rowHover=True,
                   stickyHeader=False,
                   virtualized=True,
                   size="small",
                   enableFilter=True,
                   tableLayout="fixed")
        dgrid.prop(tableSxProps={
            '& .MuiTableCell-sizeSmall': {
                "padding": '2px 2px',
            },
        })
        self.dgrid = dgrid
        init_layout = {}
        init_layout["slice_inputs"] = slice_inputs_comp
        if num_rows_split > 1:
            init_layout["row_split_slider"] = self.row_split_slider
        if num_columns_split > 1:
            init_layout["column_split_slider"] = self.column_split_slider
        self.scroll_to_index_input = mui.Input("index", init="0")
        init_layout["toolbar"] = mui.HBox([
            self.scroll_to_index_input,
            mui.Button("scroll to row", callback=self._on_scroll_to_index),
            mui.Button("scroll to first nan",
                       callback=self._on_scroll_to_first_nan),
        ])
        init_layout["grid"] = dgrid

        return init_layout

    def _get_array_min_maxs(self, obj_flatted_3d: Dict[str, np.ndarray]):
        obj_flatted_min_maxs: Dict[str, ArrayMeta] = {}
        for k, v in obj_flatted_3d.items():
            # 0 is flatted index, 1 is row, 2 is column
            min_rows = v.min(1)
            max_rows = v.max(1)
            # calc nan/inf for float array
            if v.dtype.kind == "f":
                nan_indices = np.stack(np.nonzero(np.isnan(v)), 1)
                inf_indices = np.stack(np.nonzero(np.isinf(v)), 1)
            else:
                nan_indices = None
                inf_indices = None
            obj_flatted_min_maxs[k] = ArrayMeta(min_rows, max_rows,
                                                nan_indices, inf_indices)
        return obj_flatted_min_maxs

    def _get_footer_data(self, flatted_index: int):
        footer_min_datas = {}
        footer_max_datas = {}

        for k, v in self.obj_flatted_min_maxs.items():
            for i in range(v.min.shape[-1]):
                col_key = mui.MatrixDataGrid.get_column_id(k, i)
                min_val = v.min[flatted_index, i].item()
                max_val = v.max[flatted_index, i].item()
                # json in js does not support inf or nan
                # so we convert them to string
                if isnan(min_val) or isinf(min_val):
                    min_val = str(min_val)
                if isnan(max_val) or isinf(max_val):
                    max_val = str(max_val)
                footer_min_datas[col_key] = min_val
                footer_max_datas[col_key] = max_val
        return [footer_min_datas, footer_max_datas]

    async def scroll_to_index(self, index: int):
        row_value, col_value, _ = self._get_current_row_column_flatted_index()
        if index < 0:
            index += self.obj_all_shape[-2]
        assert index >= 0 and index < self.obj_all_shape[
            -2], f"invalid index, valid range: [0, {self.obj_all_shape[-2]})"
        if index >= row_value and index < row_value + self.real_display_rows:
            await self.dgrid.scroll_to_index(index - row_value)
        else:
            real_row_value = index // self.real_display_rows
            await self.send_and_wait(
                self.row_split_slider.update_event(value=real_row_value))
            await self._on_split_slider_change(None)
            await self.dgrid.scroll_to_index(index % self.real_display_rows)

    async def _on_scroll_to_index(self):
        index = self.scroll_to_index_input.int()
        await self.scroll_to_index(index)

    async def _on_scroll_to_first_nan(self):
        row_value, col_value, flatted_index = self._get_current_row_column_flatted_index(
        )

        first_nan: Optional[int] = None
        for k, v in self.obj_flatted_min_maxs.items():
            if v.nanIndices is not None:
                nan_indices_this_flatted = v.nanIndices[v.nanIndices[:, 0] ==
                                                        flatted_index]
                if nan_indices_this_flatted.shape[0] > 0:
                    first_nan = v.nanIndices[0, 1].item()
                    break
        if first_nan is not None:
            await self.scroll_to_index(first_nan)

    def _get_array_dict_from_offset(self, flat_index: int, start_row: int,
                                    end_row: int, start_column: int,
                                    end_column: int):
        # determine all arrays covered by start column and end column
        start_column = max(start_column, 0)
        end_column = min(end_column, self.num_columns_cumsum[-1])
        start_column_index = bisect.bisect_left(self.num_columns_cumsum,
                                                start_column)
        end_column_index = bisect.bisect_left(self.num_columns_cumsum,
                                              end_column)
        start_column_offset = 0 if start_column_index == 0 else self.num_columns_cumsum[
            start_column_index - 1]
        end_column_offset = 0 if end_column_index == 0 else self.num_columns_cumsum[
            end_column_index - 1]
        if start_column_index == end_column_index:
            # only one array is covered
            array_key = list(self.obj_flatted_3d.keys())[start_column_index]
            array = self.obj_flatted_3d[array_key]
            array_sliced = array[flat_index, start_row:end_row, start_column -
                                 start_column_offset:start_column -
                                 start_column_offset + end_column -
                                 start_column]
            return {
                array_key:
                mui.MatrixDataGridItem(array_sliced,
                                       start_column - start_column_offset)
            }
        else:
            # multiple arrays are covered
            array_dict: Dict[str, mui.MatrixDataGridItem] = {}
            for i in range(start_column_index, int(end_column_index) + 1):
                array_key = list(self.obj_flatted_3d.keys())[i]
                array = self.obj_flatted_3d[array_key]
                if i == start_column_index:
                    array_sliced = array[flat_index, start_row:end_row,
                                         start_column - start_column_offset:]
                    col_offset_array = start_column - start_column_offset

                elif i == end_column_index:
                    array_sliced = array[flat_index,
                                         start_row:end_row, :end_column -
                                         end_column_offset]
                    col_offset_array = 0
                else:
                    array_sliced = array[flat_index, start_row:end_row, :]
                    col_offset_array = 0
                array_dict[array_key] = mui.MatrixDataGridItem(
                    array_sliced, col_offset_array)
            return array_dict

    def _get_current_row_column_flatted_index(self):
        col_value = self.column_split_slider.value
        assert not isinstance(col_value, mui.Undefined)
        col_value = int(col_value)
        row_value = self.row_split_slider.value
        assert not isinstance(row_value, mui.Undefined)
        row_value = int(row_value)
        if not self.slices:
            flat_index = 0
        else:
            slice_values = []
            for i, slice_input in enumerate(self.slices):
                dim = self.slider_base_shape[i]
                slice_value = slice_input.value
                assert not isinstance(slice_value, mui.Undefined)
                slice_values.append(min(max(int(slice_value), 0), dim - 1))
            # calculate flatted index
            flat_index = 0
            cur_stride = 1
            for i, v in enumerate(slice_values[::-1]):
                flat_index += v * cur_stride
                cur_stride *= self.slider_base_shape[-1 - i]

        return row_value, col_value, flat_index

    def _get_subarray_from_slice_and_col_row_split_slider(
        self
    ) -> Tuple[Dict[str, mui.MatrixDataGridItem], Tuple[int, int, int]]:
        row_value, col_value, flat_index = self._get_current_row_column_flatted_index(
        )
        subarray = self._get_array_dict_from_offset(
            flat_index, row_value * self.real_display_rows,
            (row_value + 1) * self.real_display_rows,
            col_value * self.real_display_columns,
            (col_value + 1) * self.real_display_columns)
        return subarray, (row_value * self.real_display_rows,
                          col_value * self.real_display_columns, flat_index)

    async def _on_split_slider_change(self, value: Any):
        subarr, offset = self._get_subarray_from_slice_and_col_row_split_slider(
        )
        real_rows = next(iter(subarr.values())).array.shape[0]
        real_columns = sum([o.array.shape[1] for o in subarr.values()])
        await self.matrix_typo.send_and_wait(
            self.matrix_typo.update_event(
                value=
                f" {offset[0]}:{offset[0]+real_rows}, {offset[1]}:{offset[1]+real_columns}]"
            ))
        new_footer_data = self._get_footer_data(offset[2])
        await self.send_and_wait(
            self.dgrid.update_event(dataList=subarr,
                                    rowOffset=offset[0],
                                    customFooterDatas=new_footer_data))


class NumpyArrayGridTable(mui.FlexBox):

    def __init__(self,
                 init_array_items: Optional[Dict[str,
                                                 Union[Dict[str, np.ndarray],
                                                       np.ndarray, int, float, bool]]] = None,
                 max_columns: int = 25,
                 max_size_row_split: int = 1000000):
        super().__init__()
        self.max_columns = max_columns
        self.max_size_row_split = max_size_row_split
        btn = mui.Button("Viewer").prop(size="small", loading=False)
        self.grid_container = mui.HBox([])
        dialog = mui.Dialog([
            self.grid_container.prop(flex=1, height="70vh", width="100%")
        ]).prop(title="Array Viewer",
                maxWidth="xl",
                fullWidth=True,
                includeFormControl=False)

        self.dialog = dialog
        btn.event_click.on_standard(
            self._on_btn_select).configure(stop_propagation=True)
        value_cell = mui.MatchCase([
            mui.MatchCase.ExprCase("x != \"scalar\"", btn),
            mui.MatchCase.Case(mui.undefined, mui.Typography("").set_override_props(value="value")),
        ]).set_override_props(condition="shape")
        cbox = mui.Checkbox().prop(size="small", disabled=True)
        column_defs = [
            mui.DataGrid.ColumnDef("name", accessorKey="name"),
            mui.DataGrid.ColumnDef("dtype", accessorKey="dtype"),
            mui.DataGrid.ColumnDef("shape", accessorKey="shape"),

            mui.DataGrid.ColumnDef("contiguous",
                                   accessorKey="contiguous",
                                   cell=cbox),
            mui.DataGrid.ColumnDef("value", cell=value_cell),
        ]
        self.array_items: Dict[str, Union[Dict[str, np.ndarray],
                                          np.ndarray, int, float, bool]] = {}
        if init_array_items is not None:
            self.array_items = init_array_items

        dgrid = mui.DataGrid(column_defs,
                             self._extract_table_data_from_array_items()).prop(
                                 idKey="id",
                                 rowHover=True,
                                 virtualized=True,
                                 enableFilter=True,
                                 size="small",
                                 fullWidth=True)
        dgrid.bind_prop(cbox, "contiguous")

        self.init_add_layout([dgrid.prop(flex=1), dialog])
        self.dgrid = dgrid
        self.prop(width="100%", height="100%", overflow="hidden")

    async def update_array_items(self,
                                 array_items: Dict[str, Union[Dict[str,
                                                                   np.ndarray],
                                                              np.ndarray, int, float, bool]]):
        self.array_items.update(array_items)
        item_datas = self._extract_table_data_from_array_items()
        await self.send_and_wait(self.dgrid.update_event(dataList=item_datas))

    async def set_new_array_items(self,
                                  new_array_items: Dict[str,
                                                        Union[Dict[str,
                                                                   np.ndarray],
                                                              np.ndarray, int, float, bool]]):
        self.array_items = new_array_items
        item_datas = self._extract_table_data_from_array_items()
        await self.send_and_wait(self.dgrid.update_event(dataList=item_datas))

    async def clear_array_items(self):
        self.array_items = {}
        await self.send_and_wait(self.dgrid.update_event(dataList=[]))

    def _extract_table_data_from_array_items(self):
        table_data: List[Dict[str, Any]] = []
        for k, v in self.array_items.items():
            if isinstance(v, dict):
                contiguous = all(
                    [v2.flags['C_CONTIGUOUS'] for v2 in v.values()])
                table_data.append({
                    "id": str(k),
                    "name": k,
                    "shape": "multiple",
                    "contiguous": contiguous,
                    "dtype": "multiple",
                })
            elif isinstance(v, np.ndarray):
                table_data.append({
                    "id": str(k),
                    "name": k,
                    "shape": str(v.shape),
                    "contiguous": bool(v.flags['C_CONTIGUOUS']),
                    "dtype": str(v.dtype),
                })
            elif isinstance(v, (int, float, bool)):
                table_data.append({
                    "id": str(k),
                    "name": k,
                    "shape": "scalar",
                    "contiguous": True,
                    "dtype": get_qualname_of_type(type(v)),
                    "value": str(v)
                })
        return table_data

    async def _on_btn_select(self, event: mui.Event):
        keys = event.keys
        assert not isinstance(keys, mui.Undefined)
        key = keys[0]
        item = self.array_items[key]
        assert not isinstance(item, (int, float, bool))
        await self.grid_container.set_new_layout([
            NumpyArrayGrid(item, self.max_columns,
                           self.max_size_row_split).prop(width="100%",
                                                         height="100%",
                                                         overflow="hidden")
        ])
        await self.dialog.set_open(True)
