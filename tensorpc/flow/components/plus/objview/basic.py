from pathlib import Path
import time
from typing import Any, Optional, Union
from tensorpc.flow.components import mui, three
from tensorpc.flow.components.plus.arraycommon import can_cast_to_np_array, try_cast_to_np_array
from tensorpc.flow.components.plus.objinspect.tree import BasicObjectTree
import numpy as np 
import cv2 
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
        # self.update_sx_props({
        #     "object-fit": "contain",
        # })

class ImageBatch(mui.FlexBox):
    def __init__(self, arr, *, lower: float = 0.0, upper: float = 1.0):
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

        # self._img.update_sx_props({
        #     "object-fit": "contain",
        # })
        self._img.prop(image=self._img.encode_image_bytes(imgs[0]))

        super().__init__([
            self._img,
            self._slider,
        ])
        self.prop(maxWidth="400px", flexFlow="column nowrap", alignItems="stretch")

    async def _on_slider(self, val):
        await self._img.show(self._imgs[val])

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

