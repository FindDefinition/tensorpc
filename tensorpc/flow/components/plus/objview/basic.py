from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
from tensorpc.flow.components import mui, three
from tensorpc.flow.components.plus.arraycommon import can_cast_to_np_array, try_cast_to_np_array
from tensorpc.flow.components.plus.objinspect.tree import BasicObjectTree
import numpy as np 
from tensorpc.flow import appctx, marker

def _smoothstep(x, x_min: float=0, x_max: float=1):
    return np.clip((x - x_min) / (x_max - x_min), 0, 1)

class Tree(BasicObjectTree):
    def __init__(self, root: Any):
        super().__init__(root, use_init_as_root=True)
        self.prop(minHeight=0,
                minWidth=0,
                width="100%",
                height="100%",
                overflow="hidden")

class Image(mui.Image):
    def __init__(self, arr_or_path, *, lower: float = 0.0, upper: float = 1.0):
        super().__init__()

        if isinstance(arr_or_path, str):
            suffix = Path(arr_or_path).suffix
            if suffix in [".jpg", ".jpeg", ".png"]:
                with open(arr_or_path, "rb") as f:
                    self.prop(image=f.read())
            else:
                import cv2 
                img = cv2.imread(arr_or_path)
                self.prop(image=self.encode_image_bytes(img))
        else:
            assert can_cast_to_np_array(arr_or_path)
            img = try_cast_to_np_array(arr_or_path)
            assert img is not None and img.ndim in [2, 3]
            assert img.dtype == np.uint8 or arr_or_path.dtype == np.float32
            if img.dtype == np.float32:
                if lower != 0.0 or upper != 1.0:
                    img = _smoothstep(img, lower, upper)
                img = (img * 255).astype(np.uint8)
            self.prop(image=self.encode_image_bytes(img))
        # self.prop(height="100%", width="100%", overflow="hidden")
        self.prop(maxWidth="400px", enableZoom=True)
        # self.update_raw_props({
        #     "object-fit": "contain",
        # })

class ImageBatch(mui.FlexBox):
    def __init__(self, arr, *, lower: float = 0.0, upper: float = 1.0, is_channel_first: bool = False):
        assert can_cast_to_np_array(arr)
        imgs = try_cast_to_np_array(arr)
        assert imgs is not None and imgs.ndim == 4
        # assume NHWC
        assert imgs.shape[0] > 0
        if imgs.dtype == np.float32:
            if lower != 0.0 or upper != 1.0:
                imgs = _smoothstep(imgs, lower, upper)
            imgs = (imgs * 255).astype(np.uint8)
        self._imgs = imgs
        self._slider = mui.Slider(0, imgs.shape[0] - 1, 1, callback=self._on_slider)
        self._img = mui.Image()
        # self._img.prop(overflow="hidden", flex=1)
        self._img.prop(maxWidth="400px", enableZoom=True)

        # self._img.update_raw_props({
        #     "object-fit": "contain",
        # })
        if is_channel_first:
            self._img.prop(image=self._img.encode_image_bytes(imgs[0].transpose(1, 2, 0)))
        else:
            self._img.prop(image=self._img.encode_image_bytes(imgs[0]))
        self._is_channel_first = is_channel_first

        super().__init__([
            self._img,
            self._slider,
        ])
        self.prop(maxWidth="400px", flexFlow="column nowrap", alignItems="stretch")

    async def _on_slider(self, val):
        if self._is_channel_first:
            await self._img.show(self._imgs[val].transpose(1, 2, 0))
        else:
            await self._img.show(self._imgs[val])

class ImageBatchChannelFirst(ImageBatch):
    def __init__(self, arr, *, lower: float = 0.0, upper: float = 1.0):
        super().__init__(arr, lower=lower, upper=upper, is_channel_first=True)


class Video(mui.VideoPlayer):
    def __init__(self, bytes_or_path: Union[bytes, str], suffix: Optional[str] = None):
        self._bytes_or_path = bytes_or_path
        self._modify_time_ns = time.time_ns()
        if isinstance(bytes_or_path, bytes):
            assert suffix is not None
        self._suffix = suffix
        self._key = "__tensorpc_objview_video.mp4"
        if suffix is not None:
            self._key = f"__tensorpc_objview_video{suffix}"
        super().__init__(f"tensorpc://{self._key}")
        self.prop(maxWidth="400px")

    @marker.mark_did_mount
    async def _on_mount(self):
        appctx.get_app().add_file_resource(self._key, self._serve_video)

    @marker.mark_will_unmount
    async def _on_unmount(self):
        appctx.get_app().remove_file_resource(self._key) 

    def _serve_video(self, req: mui.FileResourceRequest) -> mui.FileResource:
        if isinstance(self._bytes_or_path, str):
            return mui.FileResource(name=self._key, path=self._bytes_or_path)
        else:
            return mui.FileResource(name=self._key, content=self._bytes_or_path, modify_timestamp_ns=self._modify_time_ns)

class VideoMp4(Video):
    def __init__(self, bytes_or_path: Union[bytes, str]):
        super().__init__(bytes_or_path, suffix=".mp4")

def _parse_df_to_table(df, column_width: int = 75):
    import pandas as pd

    columns = df.columns
    dtypes = list(df.dtypes)
    column_defs: List[mui.DataGridColumnDef] = [
        mui.DataGrid.ColumnDef("id", accessorKey="id", width=column_width),
    ]
    column_def_dict: Dict[str, mui.DataGridColumnDef] = {}
    for column, dt in zip(columns, dtypes):
        if not isinstance(column, str):
            column = str(column)
        if dt != np.object_:
            col_type = mui.DataGridColumnSpecialType.Number
        else:
            col_type = mui.undefined 
        cdef = mui.DataGridColumnDef(id=column, header=column, specialType=col_type, width=column_width) 
        if dt != np.object_:
            cdef.specialProps = mui.DataGridColumnSpecialProps(
                mui.DataGridNumberCell(precision=6))
        column_defs.append(cdef)
        column_def_dict[column] = cdef
    # get rows
    rows = []
    for idx, row in df.iterrows():
        # get row dict, e.g. {id: xxx, ...columns}
        row_dict = row.to_dict()
        new_row_dict = {}
        for k, v in row_dict.items():
            if not isinstance(k, str):
                k = str(k)
            if v is None:
                new_row_dict[k] = ""
            else:
                if column_def_dict[k].specialType != mui.DataGridColumnSpecialType.Number:
                    new_row_dict[k] = str(v)
                else:
                    new_row_dict[k] = v
        new_row_dict["id"] = str(idx)
        rows.append(new_row_dict)
    
    dgrid = mui.DataGrid(column_defs, rows)
    dgrid.prop(rowHover=True,
                stickyHeader=False,
                virtualized=True,
                size="small",
                # enableColumnFilter=True,
                fullWidth=True,
                tableLayout="fixed")
    dgrid.prop(tableSxProps={
        '& .MuiTableCell-sizeSmall': {
            "padding": '2px 2px',
        },
    })
    return dgrid


class DataFrame(mui.FlexBox):
    def __init__(self, data: Any, column_width: int = 75):
        import pandas as pd
        df = pd.DataFrame(data)
        self.dgrid = _parse_df_to_table(df, column_width)
        super().__init__([self.dgrid])
        self.prop(flex=1)

class DataFrameTransposed(mui.FlexBox):
    def __init__(self, data: Any, column_width: int = 75):
        import pandas as pd
        df = pd.DataFrame(data).transpose()
        self.dgrid = _parse_df_to_table(df, column_width)
        super().__init__([self.dgrid])
        self.prop(flex=1)

class Unique(mui.FlexBox):
    pass

class HistogramPlot(mui.FlexBox):
    def __init__(self, arr):
        pass 
        pass 

class LinePlot(mui.FlexBox):
    pass

class ScatterPlot(mui.FlexBox):
    pass

