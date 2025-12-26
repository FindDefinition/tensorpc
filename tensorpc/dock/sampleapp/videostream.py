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

class LoopVideoStreamTrack(VideoStreamTrack):
    """
    A video track that returns an animated flag.
    """

    def __init__(self, path: str):
        from torchcodec.decoders import VideoDecoder
        super().__init__()  # don't forget this!
        device = "cpu"  # or e.g. "cuda" !
        decoder = VideoDecoder(path, device=device)
        self.video_frames = decoder[:]
        self.cnt = 0

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
        # await asyncio.sleep(0.1)

        pts, time_base = await self.next_timestamp()
        frame_ten = self.video_frames[self.cnt % len(self.video_frames)]
        frame = frame_ten.permute(1, 2, 0).numpy()
        self.cnt += 1
        frame_av = VideoFrame.from_ndarray(frame, format="rgb24")
        frame_av.pts = pts
        frame_av.time_base = time_base
        return frame_av

USE_IMAGE = False

class VideoRTCStreamApp:
    @mark_create_layout
    def my_layout(self):
        self.track = SimpleVideoStreamTrack(width=640, height=480)
        self.video = mui.VideoRTCStream(self.track)
        self.video.prop(disableContextMenu=True)
        self._image_video_event = asyncio.Event()
        self._image_task = None
        self.image = mui.Image().update_raw_props({
            "objectFit": "contain",
        })
        self.image.prop(flex=1, minHeight=0, minWidth=0)
        self.video.prop(flex=1, minHeight=0, minWidth=0)
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
        if USE_IMAGE:
            self._image_video_event = asyncio.Event()
            self._image_task = asyncio.create_task(self._image_video_loop(self._image_video_event))
        else:
            await self.video.start()

    @mark_will_unmount
    async def _on_unmount(self):
        if USE_IMAGE:
            self._image_video_event.set()
            if self._image_task is not None:
                await self._image_task
        else:
            await self.video.stop()

    async def _image_video_loop(self, ev: asyncio.Event):
        while True:
            if ev.is_set():
                break 
            frame = await self.track.recv()
            img = frame.to_ndarray(format="rgb24")
            await self.image.show(img)
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