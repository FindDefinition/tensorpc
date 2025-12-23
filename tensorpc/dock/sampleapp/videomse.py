import av
import io
import fractions

import time
import asyncio
import asyncio
from typing import Optional
from tensorpc.dock import plus, mui
from tensorpc.dock import mark_create_layout, mark_did_mount

from tensorpc.dock.marker import mark_will_unmount
from aiortc import VideoStreamTrack
import cv2 
from av import VideoFrame
from av.container.output import OutputContainer
from av.video.stream import VideoStream
AUDIO_PTIME = 0.020  # 20ms audio packetization
VIDEO_CLOCK_RATE = 90000
VIDEO_PTIME = 1 / 30  # 30fps
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)

import numpy as np 

def draw_heart(image, center, size, color, thickness=2):
    """
    Draws a heart shape on an image.

    Args:
        image (numpy.ndarray): The image to draw on.
        center (tuple): (x, y) coordinates of the heart's center.
        size (int): Controls the overall size of the heart.
        color (tuple): BGR color tuple (e.g., (0, 0, 255) for red).
        thickness (int): Thickness of the heart's outline.
    """
    t = np.arange(0, 2 * np.pi, 0.1)
    x_coords = size * (16 * np.sin(t)**3)
    y_coords = size * (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))

    # Translate and invert y-axis (OpenCV y-axis is inverted compared to standard math)
    x_coords = x_coords + center[0]
    y_coords = center[1] - y_coords

    # Convert to integer coordinates
    x_coords = x_coords.astype(int)
    y_coords = y_coords.astype(int)

    # Draw lines connecting the points
    for i in range(len(x_coords) - 1):
        cv2.line(image, (x_coords[i], y_coords[i]), (x_coords[i+1], y_coords[i+1]), color, thickness)

class SimpleTimer:
    def __init__(self) -> None:
        self._timestamp = 0
        self._start = time.time()

    async def next_timestamp(self) -> tuple[int, fractions.Fraction]:
        self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
        wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
        await asyncio.sleep(wait)
        return self._timestamp, VIDEO_TIME_BASE

class SimpleVideoStreamTrack(VideoStreamTrack):
    """
    A video track that returns an animated flag.
    """

    def __init__(self, width: int, height: int, fps: int = 30):
        super().__init__()  # don't forget this!
        self.width = width
        self.height = height
        self.fps = fps
        self._mouse_move_center: tuple[float, float] = (width // 2, height // 2)
        self._keyboard_move_center = (width // 2, height // 2)
        self._cur_mouse_event: Optional[mui.PointerEvent] = None
        # self._cur_mouse_events: dict[str, Optional[mui.PointerEvent]] = {
        self._cur_mouse_btn_state = {
            0: False,  # left
            1: False,  # middle
            2: False,  # right
        }

        # }
        self._cur_keyboard_events: dict[str, Optional[mui.KeyboardHoldEvent]] = {
            "KeyW": None,
            "KeyA": None,
            "KeyS": None,
            "KeyD": None,
            "ShiftLeft": None,
        }


    def _heavy_compute(self, device):
        import torch 
        # to avoid video generation block ui, we need to run model in thread.
        torch.cuda.set_device(device)
        pass

    def update_mouse_movement(self, ev: mui.PointerEvent):
        assert ev.movementX is not None and ev.movementY is not None
        new_x = min(max(self._mouse_move_center[0] + ev.movementX, 0), self.width)
        new_y = min(max(self._mouse_move_center[1] + ev.movementY, 0), self.height)
        self._mouse_move_center = (new_x, new_y)

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = np.zeros((self.height, self.width, 3), np.uint8)
        # if self._mouse_move_center is not None:
        left_pressed = self._cur_mouse_btn_state[0]
        right_pressed = self._cur_mouse_btn_state[2]
        middle_pressed = self._cur_mouse_btn_state[1]
        circle_color = None
        if left_pressed:
            circle_color = (255, 0, 0)  # Blue for left button
        elif right_pressed:
            circle_color = (0, 0, 255)  # Red for right button
        elif middle_pressed:
            circle_color = (0, 255, 0)  # Green for middle button
        if circle_color is not None:
            # draw a circle at the mouse position
            x = int(self._mouse_move_center[0])
            y = int(self._mouse_move_center[1])
            cv2.circle(frame, (x, y), 20, circle_color, -1)
        keyboard_wasd_move_speed = 10
        code_to_delta = {
            "KeyW": (0, -keyboard_wasd_move_speed),
            "KeyA": (-keyboard_wasd_move_speed, 0),
            "KeyS": (0, keyboard_wasd_move_speed),
            "KeyD": (keyboard_wasd_move_speed, 0),
        }
        for k, ev in self._cur_keyboard_events.items():
            if ev is not None:
                delta = code_to_delta[k]
                self._keyboard_move_center = (
                    self._keyboard_move_center[0] + delta[0],
                    self._keyboard_move_center[1] + delta[1],
                )
                self._cur_keyboard_events[k] = None
        # draw keyboard heart
        draw_heart(
            frame, 
            center=self._keyboard_move_center, 
            size=3, 
            color=(0, 0, 255), 
            thickness=2
        )
        text_to_write = [
            "Press Z to lock pointer, Esc to unlock",
            "WASD to move the heart",
        ]
        cv2.putText(frame, text_to_write[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.putText(frame, text_to_write[1], (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        # uncomment to simulate heavy compute in pytorch
        # frame = await asyncio.get_running_loop().run_in_executor(None, self._heavy_compute, ...)
        frame_av = VideoFrame.from_ndarray(frame, format="bgr24")
        frame_av.pts = pts
        frame_av.time_base = time_base
        return frame_av

class FragmentMP4Encoder:
    def __init__(self):
        self._cur_container: Optional[OutputContainer] = None
        self._cur_stream: Optional[VideoStream] = None
        self._output_buffer: Optional[io.BytesIO] = None

    def _setup_container_and_stream(self, width: int, height: int):
        output_buffer = io.BytesIO()
        # Key options: 'format', 'flush_packets', and 'movflags'
        output_container = av.open(
            output_buffer, 
            mode='w', 
            format='mp4', 
            options={
                'flush_packets': '1', 
                'movflags': 'frag_keyframe+empty_moov+default_base_moof'}
        )
        fps = 30 # example frame rate
        # Set keyint_min and g options to ensure regular keyframes
        stream = output_container.add_stream(
            'libx264', 
            rate=fps, 
            options={
                'keyint_min': str(fps), 
                'g': str(fps), 
                'sc_threshold': '0', 
                'tune': 'zerolatency'
            }
        )
        stream.width = width  # example width
        stream.height = height # example height
        stream.pix_fmt = 'yuv420p'
        # Example: generating a simple synthetic frame (replace with your actual frame source)
        self._cur_container = output_container
        self._cur_stream = stream
        self._output_buffer = output_buffer
        self.start_ts = time.time()

    def encode(self, frame: av.VideoFrame) -> Optional[bytes]:
        if self._cur_container is None:
            self._setup_container_and_stream(frame.width, frame.height)
        assert self._cur_stream is not None 
        assert self._cur_container is not None
        assert self._output_buffer is not None
        should_stop = False
        for packet in self._cur_stream.encode(frame):
            # if packet.is_keyframe:
            #     should_stop = True
            if time.time() - self.start_ts >= 1.0:
                should_stop = True
            self._cur_container.mux(packet)
        if should_stop:
            for packet in self._cur_stream.encode(None):
                self._cur_container.mux(packet)

            data = self._output_buffer.getvalue()
            self._cur_container.close()
            self._cur_container = None
            self._cur_stream = None
            self._output_buffer = None
            return data
        return None

    def close(self) -> Optional[bytes]:
        if self._cur_container is None:
            return None 
        assert self._cur_stream is not None 
        assert self._cur_container is not None
        assert self._output_buffer is not None
        for packet in self._cur_stream.encode(None):
            self._cur_container.mux(packet)
        data = self._output_buffer.getvalue()
        self._cur_container.close()
        self._cur_container = None
        self._cur_stream = None
        self._output_buffer = None
        return data

USE_IMAGE = False 
class VideoRTCStreamApp:
    @mark_create_layout
    def my_layout(self):
        self.track = SimpleVideoStreamTrack(width=640, height=480)

        self._codec = 'video/mp4; codecs="avc1.64001e"'
        self.video = mui.VideoBasicStream(self._codec)
        self._image_video_event = asyncio.Event()
        self._image_task = None
        self.image = mui.Image().update_raw_props({
            "objectFit": "contain",
        })
        self.image.prop(flex=1, minHeight=0, minWidth=0)
        self.video.prop(flex=1, minHeight=0, minWidth=0, controls=True)
        root_box = mui.HBox([
            self.image if USE_IMAGE else self.video,
        ]).prop(width="100%", height="100%", overflow="hidden", minHeight=0, minWidth=0)
        root_box.event_pointer_move.on(self._on_pointer_move)
        root_box.event_pointer_down.on(self._on_pointer_down)
        root_box.event_pointer_up.on(self._on_pointer_up)
        root_box.event_keyboard_hold.on(self._on_keyboard_hold).configure(
            key_codes=["KeyW", "KeyA", "KeyS", "KeyD", "ShiftLeft"],
            key_hold_interval_delay=33.33,
        )
        root_box.event_pointer_lock_released.on(self._on_pointer_lock_release)
        self._enable_events = False
        # enable controls
        root_box.event_keyup.on(self._on_key_up).configure(
            key_codes=["KeyZ", "KeyX", "Escape"],
        )
        self.button = mui.Button("Click Me!", self._on_click)
        self._event_box = root_box
        return mui.VBox([
            root_box,
            self.button,
        ]).prop(width="100%", height="100%", overflow="hidden", minHeight=0, minWidth=0)

    @mark_did_mount
    async def _on_mount(self):
        # if USE_IMAGE:
        self._image_video_event = asyncio.Event()
        self._image_task = asyncio.create_task(self._image_video_loop(self._image_video_event))
        # else:
        await self.video.set_media_source(self._codec)

    @mark_will_unmount
    async def _on_unmount(self):
        # if USE_IMAGE:
        self._image_video_event.set()
        if self._image_task is not None:
            await self._image_task
        # else:
        # await self.video.close()

    async def _image_video_loop(self, ev: asyncio.Event):
        # output_buffer = io.BytesIO()
        # # Key options: 'format', 'flush_packets', and 'movflags'
        # output_container = av.open(
        #     output_buffer, 
        #     mode='w', 
        #     format='mp4', 
        #     options={'flush_packets': '1', 'movflags': 'frag_keyframe+empty_moov+default_base_moof'}
        # )
        # fps = 30 # example frame rate
        # # Set keyint_min and g options to ensure regular keyframes
        # stream = output_container.add_stream(
        #     'libx264', 
        #     rate=fps, 
        #     options={
        #         'keyint_min': str(fps), 
        #         'g': str(fps), 
        #         'sc_threshold': '0', 
        #         'tune': 'zerolatency'
        #     }
        # )
        # stream.width = 640  # example width
        # stream.height = 480 # example height
        # stream.pix_fmt = 'yuv420p'
        # # Example: generating a simple synthetic frame (replace with your actual frame source)

        # while True:
        #     if ev.is_set():
        #         break 
        #     frame = await self.track.recv()
        #     # img = frame.to_ndarray(format="rgb24")
        #     # await self.image.show(img)

        #     for packet in stream.encode(frame):
        #         output_container.mux(packet)
        #         await self.video.append_buffer(0, bytes(packet))
        # for packet in stream.encode(None):
        #     output_container.mux(packet)
        #     await self.video.append_buffer(0, bytes(packet))
        # # Close the container
        # output_container.close()
        encoder = FragmentMP4Encoder()
        while True:
            if ev.is_set():
                break 
            frame = await self.track.recv()
            data = encoder.encode(frame)
            if data is not None:
                await self.video.append_buffer(0, data)
        buffer = encoder.close()
        if buffer is not None:
            await self.video.append_buffer(0, buffer)
        await self.video.close()
        self._image_task = None

    async def _on_pointer_move(self, data: mui.PointerEvent):
        if not self._enable_events:
            return
        # self.track._cur_mouse_event = data
        self.track.update_mouse_movement(data)
        # self.track._mouse_move_center = (data.offsetX, data.offsetY)

    async def _on_pointer_down(self, data: mui.PointerEvent):
        if not self._enable_events:
            return

        # self.track._cur_mouse_event = data
        self.track._cur_mouse_btn_state[data.button] = True

    async def _on_pointer_up(self, data: mui.PointerEvent):
        if not self._enable_events:
            return

        self.track._cur_mouse_btn_state[data.button] = False

    async def _on_keyboard_hold(self, data: mui.KeyboardHoldEvent):
        if not self._enable_events:
            return
        print(data)
        self.track._cur_keyboard_events[data.code] = data

    async def _on_key_up(self, data: mui.KeyboardEvent):
        if data.code == "KeyZ":
            await self._event_box.request_pointer_lock()
            self._enable_events = True
        elif data.code == "KeyX" or data.code == "Escape":
            await self._event_box.exit_pointer_lock()
            self._enable_events = False

    async def _on_pointer_lock_release(self):
        self._enable_events = False

    def _on_click(self):
        print("Button clicked!")
def main():
    # Use BytesIO for in-memory buffer or a filename for disk file
    output_buffer = io.BytesIO()
    # Key options: 'format', 'flush_packets', and 'movflags'
    output_container = av.open(
        "/mnt/personal/yanyan/tensorpc/debug.mp4", 
        mode='w', 
        format='mp4', 
        options={'flush_packets': '1', 'movflags': 'frag_keyframe+empty_moov'}
    )
    fps = 30 # example frame rate
    # Set keyint_min and g options to ensure regular keyframes
    stream = output_container.add_stream(
        'libx264', 
        rate=fps, 
        options={'keyint_min': str(fps), 'g': str(fps), 'sc_threshold': '0', 'tune': 'zerolatency'}
    )
    stream.width = 640  # example width
    stream.height = 480 # example height
    stream.pix_fmt = 'yuv420p'
    # Example: generating a simple synthetic frame (replace with your actual frame source)
    import numpy as np
    duration = 5 # seconds
    total_frames = duration * fps

    for frame_i in range(total_frames):
        # Create a dummy frame (replace with actual frame data)
        img = np.zeros((480, 640, 3), dtype=np.uint8) 
        # ... modify img ...

        frame = av.VideoFrame.from_ndarray(img, format='rgb24')

        # Set PTS (Presentation Time Stamp) manually
        # The time base of the stream should be used for PTS calculation
        frame.pts = frame_i 

        for packet in stream.encode(frame):
            output_container.mux(packet)
            # After muxing, data is written to the buffer/file in fragments
            # You can access new bytes from the buffer here if streaming
    # Flush remaining packets
    for packet in stream.encode(None):
        output_container.mux(packet)

    # Close the container
    output_container.close()


if __name__ == "__main__":
    main()