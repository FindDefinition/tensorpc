from typing import Generic, cast
from functools import partial

import tensorpc.core.dataclass_dispatch as dataclasses
import enum
from tensorpc.core.datamodel.events import DraftChangeEvent, DraftChangeEventHandler, DraftEventType

from typing import (TYPE_CHECKING, Any,
                    Awaitable, Callable, Coroutine, Dict, Iterable, List,
                    Optional, Set, Tuple, Type, TypeVar, Union)

from typing_extensions import Literal, TypeAlias, TypedDict, Self
from pydantic import field_validator, model_validator

from tensorpc.core.datamodel.draft import DraftBase, insert_assign_draft_op
from tensorpc.dock import appctx
from tensorpc.dock.core.appcore import Event, get_batch_app_event
from tensorpc.dock.core.common import (handle_standard_event)
from tensorpc.dock.core.uitypes import RTCTrackInfo
from .core import MUIComponentBase, MUIFlexBoxProps
from ...core.component import (
    Component, ContainerBaseProps, DraftOpUserData, 
    FrontendEventType, NumberType, UIType,
    Undefined, ValueType, undefined)
from ...core.datamodel import DataModel
from aiortc import (
    MediaStreamTrack,
    VideoStreamTrack,
    AudioStreamTrack
)

class VideoControlType(enum.IntEnum):
    SetMediaSource = 0
    AppendBuffer = 1
    CloseStream = 2

@dataclasses.dataclass
class _BaseVideoProps:
    autoPlay: Union[Undefined, bool] = undefined
    controls: Union[Undefined, bool] = undefined
    controlsList: Union[Undefined, str] = undefined
    loop: Union[Undefined, bool] = undefined
    muted: Union[Undefined, bool] = undefined
    poster: Union[Undefined, str] = undefined
    playsInline: Union[Undefined, bool] = undefined
    src: Union[Undefined, str] = undefined

@dataclasses.dataclass
class VideoBasicStreamProps(MUIFlexBoxProps, _BaseVideoProps):
    mimeCodec: str = ""

@dataclasses.dataclass
class VideoRTCStreamProps(MUIFlexBoxProps, _BaseVideoProps):
    pass


class VideoBasicStream(MUIComponentBase[VideoBasicStreamProps]):

    def __init__(self,
                 mime_codec: str) -> None:
        super().__init__(UIType.VideoBasicStream, VideoBasicStreamProps, 
            allowed_events=[FrontendEventType.ComponentReady.value])
        self.event_video_stream_ready = self._create_event_slot_noarg(FrontendEventType.ComponentReady)


    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync=is_sync)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def set_media_source(self, mime_codec: str):
        return await self.send_and_wait(
            self.create_comp_event({
                "type": int(VideoControlType.SetMediaSource),
                "mimeCodec": mime_codec,
            }))

    async def append_buffer(self, idx: int, buffer: bytes):
        return await self.send_and_wait(
            self.create_comp_event({
                "type": int(VideoControlType.AppendBuffer),
                "idx": idx,
                "buffer": buffer,
            }))

    async def close(self):
        return await self.send_and_wait(
            self.create_comp_event({
                "type": int(VideoControlType.CloseStream),
            }))

class VideoRTCControlType(enum.IntEnum):
    StartStream = 0
    CloseStream = 1

_T = TypeVar("_T", bound=VideoStreamTrack)

class VideoRTCStream(MUIComponentBase[VideoRTCStreamProps], Generic[_T]):

    def __init__(self, video_track: _T) -> None:
        super().__init__(UIType.VideoRTCStream, VideoRTCStreamProps, 
            allowed_events=[])
        self._video_track = video_track

        self.event_after_mount.on(self._on_component_mount)

    def _on_component_mount(self):
        appctx.get_app()._register_rtc_track(self, [
            RTCTrackInfo(track=self._video_track, kind="video")
        ])

    def _on_component_unmount(self):
        appctx.get_app()._unregister_rtc_track(self)

    @property
    def video_track(self) -> _T:
        return self._video_track

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync=is_sync)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def start(self):
        return await self.send_and_wait(
            self.create_comp_event({
                "type": int(VideoRTCControlType.StartStream),
            }))


    async def stop(self):
        return await self.send_and_wait(
            self.create_comp_event({
                "type": int(VideoRTCControlType.CloseStream),
            }))
