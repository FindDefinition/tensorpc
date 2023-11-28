
from typing import Any, Callable, Coroutine
from tensorpc.core.serviceunit import AppFuncType, ServFunctionMeta
from tensorpc.flow.flowapp.components import mui, three
from tensorpc.flow.flowapp.core import FlowSpecialMethods, FrontendEventType, _get_obj_def_path
from tensorpc.flow.flowapp.appcore import Event, get_app, get_editable_app
from functools import partial


async def preview_layout_reload(layout_setter: Callable[[mui.FlexBox], Coroutine[Any, Any, None]], layout: mui.FlexBox,
                                    create_layout: ServFunctionMeta):
    if create_layout.user_app_meta is not None and create_layout.user_app_meta.type == AppFuncType.CreatePreviewLayout:
        if layout._wrapped_obj is not None:
            layout_flex = create_layout.get_binded_fn()()
            assert isinstance(
                layout_flex, mui.FlexBox
            ), f"create_layout must return a flexbox when use anylayout"
            layout_flex._flow_comp_def_path = _get_obj_def_path(
                layout._wrapped_obj)
            layout_flex._wrapped_obj = layout._wrapped_obj
            layout_flex.set_flow_event_context_creator(
                layout._flow_event_context_creator)
            # self.__install_preview_event_listeners(layout_flex)
            await layout_setter(layout_flex)
            # await preview_container.set_new_layout([layout_flex])
        else:
            layout_flex = create_layout.get_binded_fn()()
            layout_flex.set_flow_event_context_creator(
                layout._flow_event_context_creator)
            # self.__install_preview_event_listeners(layout_flex)
            await layout.set_new_layout(layout_flex)
        return layout_flex
    return None 