import enum
import inspect
import types
from functools import partial
from typing import (Any, Callable, Dict, Hashable, Iterable, List, Optional,
                    Set, Tuple, Type, Union)

from tensorpc.flow.flowapp.components import mui, three
from tensorpc.flow.flowapp.core import FrontendEventType



class AnyFlexLayout(mui.FlexLayout):
    def __init__(self) -> None:
        super().__init__([])
        self.register_event_handler(FrontendEventType.Drop.value, self._on_drop)

    async def _on_drop(self, comp_name):
        if comp_name is not None:
            await self.update_childs({comp_name[1]: comp_name[0]})
