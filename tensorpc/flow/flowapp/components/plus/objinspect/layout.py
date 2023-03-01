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
        self.register_event_handler(FrontendEventType.ComplexLayoutCloseTab.value, self._on_tab_close)

    async def _on_drop(self, comp_name):
        if comp_name is not None:
            print("DROP", comp_name)
            await self.update_childs({comp_name[1]: comp_name[0]})

    async def _on_tab_close(self, data):
        print("TAB CLOSE", data)

        name = data["complexLayoutTabNodeId"]
        await self.remove_childs_by_keys([name])
