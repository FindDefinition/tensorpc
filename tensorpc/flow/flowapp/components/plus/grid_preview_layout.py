from functools import partial
from typing import Any, Callable, Coroutine, Dict, Hashable, Iterable, List, Literal, Optional, Set, Tuple, Type, Union
from tensorpc.core.serviceunit import AppFuncType, ServFunctionMeta
from tensorpc.flow.flowapp.appcore import get_editable_app
from tensorpc.flow.flowapp.components import mui, three
from tensorpc.flow.flowapp import appctx
from tensorpc.flow.flowapp.components.mui import LayoutType
from tensorpc.flow.flowapp.components.plus.reload_utils import preview_layout_reload
from tensorpc.flow.flowapp.core import AppComponentCore, UIType
from tensorpc.flow.flowapp.reload import FlowSpecialMethods
from tensorpc.flow.marker import mark_create_layout
from tensorpc.flow.flowapp.objtree import UserObjTreeProtocol

class GridPreviewContainer(mui.FlexBox):
    def __init__(self, preview_layout: mui.FlexBox, name: str):
        super().__init__({
            "header": mui.HBox([
                mui.HBox([mui.Icon(mui.IconType.DragIndicator).prop(iconSize="small")]).prop(className="grid-layout-drag-handle", alignItems="center", cursor="move"),
                mui.Typography(name).prop(fontSize="14px", fontFamily="monospace"),
            ]),
            "layout": preview_layout
        })
        self.prop(flexDirection="column", width="100%", height="100%", overflowY="auto")

class GridPreviewLayout(mui.FlexBox):
    def __init__(self, init_children: Dict[str, Any], tree_root: Optional[UserObjTreeProtocol] = None) -> None:
        self._init_children = init_children
        self._init_tree_root = tree_root
        super().__init__()
        self.init_add_layout([*self._layout_func()])

    @mark_create_layout
    def _layout_func(self):
        # res = mui.FlexBox()
        reload_mgr = appctx.get_reload_manager()
        if self._init_tree_root is not None:
            init_root = self._init_tree_root
            self.set_flow_event_context_creator(
                lambda: init_root.enter_context(init_root))
        preview_layouts: mui.LayoutType = {}
        preview_layouts_v2: List[mui.GridItem] = []
        max_cols = 4 
        width_rate = 3
        cols = width_rate * max_cols
        cnt = 0
        for name, obj in self._init_children.items():
            metas = reload_mgr.query_type_method_meta(
                type(obj), True)
            special_methods = FlowSpecialMethods(metas)
            if special_methods.create_preview_layout is not None:
                if self._init_tree_root is None:
                    preview_layout = mui.flex_preview_wrapper(
                        obj, metas, reload_mgr)
                else:
                    with self._init_tree_root.enter_context(self._init_tree_root):
                        preview_layout = mui.flex_preview_wrapper(
                            obj, metas, reload_mgr)
                container = GridPreviewContainer(preview_layout, name)
                preview_layouts[name] = container
                grid_x = cnt % max_cols
                grid_y = cnt // max_cols
                preview_layouts_v2.append(mui.GridItem(container, name, mui.GridItemProps(i=name, x=grid_x * width_rate, y=grid_y, w=width_rate, h=1)))

                # container.update_sx_props({
                #     "data-grid": mui.GridItemProps(i=name, x=grid_x, y=grid_y, w=1, h=1),
                # })
                get_editable_app().observe_layout(
                    preview_layout, partial(self._on_preview_layout_reload, container=container))
                cnt += 1
        # res.init_add_layout([
        #     mui.GridLayout(preview_layouts_v2).prop(flex=1, cols=12, draggableHandle=".grid-layout-drag-handle", rowHeight=300)
        # ])
        # print(preview_layouts_v2, self._init_children)
        self.prop(flexDirection="row", flex=1, width="100%", height="100%")
        return [mui.GridLayout(preview_layouts_v2).prop(flex=1, cols=cols, 
                                                        draggableHandle=".grid-layout-drag-handle", rowHeight=300)]

    async def _on_preview_layout_reload(self, layout: mui.FlexBox,
                                        create_layout: ServFunctionMeta, container: GridPreviewContainer):
        # print("DO PREVIEW LAYOUT RELOAD", create_layout.user_app_meta)
        layout_flex = await preview_layout_reload(lambda x: container.update_childs({"layout": x}), layout, create_layout)
        if layout_flex is not None:
            get_editable_app().observe_layout(layout_flex, partial(self._on_preview_layout_reload, container=container))
            return layout_flex
