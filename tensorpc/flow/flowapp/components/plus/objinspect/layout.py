import enum
import inspect
import types
from functools import partial
from typing import (Any, Callable, Dict, Hashable, Iterable, List, Optional,
                    Set, Tuple, Type, Union)

from tensorpc.flow.flowapp.components import mui, three, plus
from tensorpc.flow.flowapp.core import FlowSpecialMethods, FrontendEventType, _get_obj_def_path
from tensorpc.flow.flowapp.components.plus.objinspect.tree import TreeDragTarget
from tensorpc.flow.flowapp.reload import reload_object_methods


class AnyFlexLayout(mui.FlexLayout):
    def __init__(self) -> None:
        super().__init__([])
        self.register_event_handler(FrontendEventType.Drop.value, self._on_drop)
        self.register_event_handler(FrontendEventType.ComplexLayoutCloseTab.value, self._on_tab_close)
        self.register_event_handler(FrontendEventType.ComplexLayoutTabReload.value, self._on_tab_reload)

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


    async def _on_tab_reload(self, name):
        # print("TAB CLOSE", data)
        layout = self._child_comps[name]
        if isinstance(layout, mui.FlexBox):
            if layout._wrapped_obj is not None:
                # for anylayout, we support layout reload
                # and onmount/onunmount reload.
                metas = reload_object_methods(layout._wrapped_obj)
                if metas is not None:
                    special_methods = FlowSpecialMethods(metas)
                    special_methods.bind(layout._wrapped_obj)
                    if special_methods.create_layout is not None:
                        layout_flex = special_methods.create_layout.get_binded_fn()()
                        assert isinstance(layout_flex,
                                            mui.FlexBox), f"create_layout must return a flexbox when use anylayout"
                        layout_flex._flow_comp_def_path = _get_obj_def_path(layout._wrapped_obj)
                        layout_flex._wrapped_obj = layout._wrapped_obj
                        await self.update_childs({name: layout_flex})

                        # await layout.set_new_layout(special_methods.create_layout.get_binded_fn()())
            else:
                metas = reload_object_methods(layout)
                if metas is not None:
                    special_methods = FlowSpecialMethods(metas)
                    special_methods.bind(layout)
                    if special_methods.create_layout is not None:
                        await layout.set_new_layout(special_methods.create_layout.get_binded_fn()())