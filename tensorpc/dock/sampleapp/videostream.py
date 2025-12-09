import asyncio
from typing import Optional
from tensorpc.dock import plus, mui
from tensorpc.dock import mark_create_layout, mark_did_mount

from tensorpc.dock.marker import mark_will_unmount
from aiortc import VideoStreamTrack
import cv2 
from av import VideoFrame

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

class SimpleVideoStreamTrack(VideoStreamTrack):
    """
    A video track that returns an animated flag.
    """

    def __init__(self, width: int, height: int, fps: int = 30):
        super().__init__()  # don't forget this!
        self.width = width
        self.height = height
        self.fps = fps
        self._mouse_move_center: Optional[tuple[int, int]] = None
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
        }


    def _heavy_compute(self, device):
        import torch 
        # to avoid video generation block ui, we need to run model in thread.
        torch.cuda.set_device(device)
        pass

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = np.zeros((self.height, self.width, 3), np.uint8)
        
        if self._mouse_move_center is not None:
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
        # print(self._keyboard_move_center)
        # draw keyboard heart
        draw_heart(
            frame, 
            center=self._keyboard_move_center, 
            size=3, 
            color=(0, 0, 255), 
            thickness=2
        )
        # uncomment to simulate heavy compute in pytorch
        # frame = await asyncio.get_running_loop().run_in_executor(None, self._heavy_compute, ...)
        frame_av = VideoFrame.from_ndarray(frame, format="bgr24")
        frame_av.pts = pts
        frame_av.time_base = time_base
        return frame_av

class VideoRTCStreamApp:
    @mark_create_layout
    def my_layout(self):
        self.track = SimpleVideoStreamTrack(width=640, height=480)
        self.video = mui.VideoRTCStream(self.track)
        root_box = mui.HBox([
            self.video.prop(flex=1),
        ]).prop(width=640, height=480, border="1px solid red")
        root_box.event_pointer_move.on(self._on_pointer_move)
        root_box.event_pointer_down.on(self._on_pointer_down)
        root_box.event_pointer_up.on(self._on_pointer_up)
        root_box.event_keyboard_hold.on(self._on_keyboard_hold).configure(
            key_codes=["KeyW", "KeyA", "KeyS", "KeyD"],
            key_hold_interval_delay=33.33,
        )
        self._enable_events = False
        # enable controls
        root_box.event_keyup.on(self._on_key_up).configure(
            key_codes=["KeyZ", "KeyX"],
        )

        return mui.HBox([
            root_box
        ]).prop(width="100%", height="100%", overflow="auto")

    @mark_did_mount
    async def _on_mount(self):
        await self.video.start()
        # await self.video.set_media_source("video/mp2t; codecs=\"avc1.4d002a\"") 
    @mark_will_unmount
    async def _on_unmount(self):
        await self.video.stop()

    async def _on_pointer_move(self, data: mui.PointerEvent):
        if not self._enable_events:
            return
        # self.track._cur_mouse_event = data
        self.track._mouse_move_center = (data.offsetX, data.offsetY)

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

        self.track._cur_keyboard_events[data.code] = data

    async def _on_key_up(self, data: mui.KeyboardEvent):
        if data.code == "KeyZ":
            self._enable_events = True
        elif data.code == "KeyX":
            self._enable_events = False

