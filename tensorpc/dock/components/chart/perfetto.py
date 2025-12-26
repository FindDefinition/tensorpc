from tensorpc.core import core_io
from typing import (TYPE_CHECKING, Any, Callable, Coroutine, Dict, Iterable,
                    List, Optional, Tuple, Type, TypeVar, Union)
from ..mui import IFrame, ValueType

class Perfetto(IFrame):
    def __init__(self, data: Optional[bytes] = None, title: Optional[str] = None, keep_api_open: bool = True):
        init_data = None 
        self._keep_api_open = keep_api_open
        if data is not None and title is not None:
            init_data = {
                "perfetto": {
                    "buffer": core_io.JSArrayBuffer(data),
                    "title": title,
                    "keepApiOpen": self._keep_api_open,
                }
            }
        super().__init__("https://ui.perfetto.dev", init_data)
        self.prop(pingPongMessage=("PING", "PONG"))

    # async def set_trace_data(self, data: bytes, title: str):
    #     assert isinstance(data, bytes)
    #     await self.post_message({
    #         "perfetto": {
    #             "buffer": core_io.JSArrayBuffer(data),
    #             "title": title,
    #         }
    #     })

    async def set_trace_data(self, data: bytes, title: str):
        assert isinstance(data, bytes)
        await self.send_and_wait(self.update_event(data=({
            "perfetto": {
                "buffer": core_io.JSArrayBuffer(data),
                "title": title,
                "keepApiOpen": self._keep_api_open,
            }
        })))

    async def scroll_to_range(self, start: ValueType, end: ValueType, view_percentage: Optional[float] = None):
        """start and end are seconds, not micro seconds or nano seconds.
        """
        msg = {
            "timeStart": start,
            "timeEnd": end,
            "keepApiOpen": self._keep_api_open,
        }
        if view_percentage is not None:
            assert view_percentage >= 0 and view_percentage <= 1
            msg["viewPercentage"] = view_percentage
        # control msg, so don't store data
        await self.post_message(data=({
            "perfetto": msg
        }), store_data=False)

