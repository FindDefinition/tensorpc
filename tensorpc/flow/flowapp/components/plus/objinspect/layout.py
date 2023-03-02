import enum
import inspect
import types
from functools import partial
from typing import (Any, Callable, Dict, Hashable, Iterable, List, Optional,
                    Set, Tuple, Type, Union)

from tensorpc.flow.flowapp.components import mui, three, plus
from tensorpc.flow.flowapp.core import FrontendEventType
from tensorpc.flow.flowapp.components.plus.objinspect.tree import TreeDragTarget


class AnyFlexLayout(mui.FlexLayout):
    def __init__(self) -> None:
        super().__init__([])
        self.register_event_handler(FrontendEventType.Drop.value, self._on_drop)
        self.register_event_handler(FrontendEventType.ComplexLayoutCloseTab.value, self._on_tab_close)

    async def _on_drop(self, target: Optional[TreeDragTarget]):
        if target is not None:
            # print("DROP", comp_name)
            obj = target.obj
            if isinstance(obj, mui.FlexBox):
                wrapped_obj = obj
            else:
                wrapped_obj = mui.flex_wrapper(obj)
            uid = target.tab_id if target.tab_id else target.tree_id
            await self.update_childs({uid: wrapped_obj})

    async def _on_tab_close(self, data):
        # print("TAB CLOSE", data)
        name = data["complexLayoutTabNodeId"]
        await self.remove_childs_by_keys([name])
