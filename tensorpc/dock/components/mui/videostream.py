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
    LOGGER, Component, ContainerBaseProps, DraftOpUserData, 
    FrontendEventType, NumberType, UIType,
    Undefined, ValueType, undefined)
from ...core.datamodel import DataModel
from aiortc import (
    MediaStreamTrack,
    VideoStreamTrack,
    AudioStreamTrack
)
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCRtpSender, RTCSessionDescription

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
    disableContextMenu: Union[Undefined, bool] = undefined


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

def force_codec(pc: RTCPeerConnection, sender: RTCRtpSender, forced_codec: str) -> None:
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences(
        [codec for codec in codecs if codec.mimeType == forced_codec]
    )

class VideoRTCStream(MUIComponentBase[VideoRTCStreamProps], Generic[_T]):

    def __init__(self, video_track: _T, force_codec: Optional[str] = None) -> None:
        super().__init__(UIType.VideoRTCStream, VideoRTCStreamProps, 
            allowed_events=[FrontendEventType.RTCSdpRequest])
        self._video_track = video_track
        self._force_codec = force_codec
        self.event_rtc_sdp_request = self._create_event_slot(FrontendEventType.RTCSdpRequest)
        self.event_rtc_sdp_request.on(self._on_rtc_sdp_request)
        self._pcs: set[RTCPeerConnection] = set()
        self._rtc_info = RTCTrackInfo(track=self._video_track, kind="video", force_codec=self._force_codec)

    async def _on_rtc_sdp_request(self, params: Any):
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        pc = RTCPeerConnection()
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            LOGGER.info("Connection state is %s", pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                self._pcs.discard(pc)
        for item in [self._rtc_info]:
            sender = pc.addTrack(item.track)
            if item.force_codec is not None:
                force_codec(pc, sender, item.force_codec)
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

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

    async def start(self, rtc_config: Optional[Any] = None):
        msg = {
            "type": int(VideoRTCControlType.StartStream),
        }
        if rtc_config is not None:
            msg["config"] = rtc_config
        return await self.send_and_wait(
            self.create_comp_event(msg))


    async def stop(self):
        return await self.send_and_wait(
            self.create_comp_event({
                "type": int(VideoRTCControlType.CloseStream),
            }))
