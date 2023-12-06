from functools import partial
import math
from typing import Any, Callable, Coroutine, Dict, Hashable, Iterable, List, Literal, Optional, Set, Tuple, Type, Union
from tensorpc.core.moduleid import get_qualname_of_type
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
import dataclasses

from tensorpc.flow.flowapp.components.plus.core import (ALL_OBJECT_LAYOUT_HANDLERS, ObjectGridLayoutItem, ObjectLayoutHandler,
                   DataClassesType)
from tensorpc.flow.flowapp.components.plus.handlers.common import DefaultHandler

from typing import List, Tuple

def layout_rectangles(rectangles: List[Tuple[int, int]], bounding_width: int) -> List[Tuple[int, int, Tuple[int, int]]]:
    # Sort rectangles by height
    rectangles.sort(key=lambda r: r[1], reverse=True)

    # Initialize variables
    x, y, row_height, layout = 0, 0, 0, []

    # Layout rectangles
    for rectangle in rectangles:
        width, height = rectangle

        # If rectangle doesn't fit in current row, start a new row
        if x + width > bounding_width:
            x = 0
            y += row_height
            row_height = 0

        # Place rectangle
        layout.append((x, y, rectangle))
        x += width
        row_height = max(row_height, height)

    return layout

def layout_rectangles_with_priority(rectangles: List[Tuple[int, int, int]], bounding_width: int) -> List[Tuple[int, int, Tuple[int, int]]]:
    # Sort rectangles by height
    rectangles.sort(key=lambda r: (r[1], r[2]), reverse=True)

    # Initialize variables
    x, y, row_height, layout = 0, 0, 0, []

    # Layout rectangles
    for rectangle in rectangles:
        width, height, _ = rectangle

        # If rectangle doesn't fit in current row, start a new row
        if x + width > bounding_width:
            x = 0
            y += row_height
            row_height = 0

        # Place rectangle
        layout.append((x, y, rectangle))
        x += width
        row_height = max(row_height, height)

    return layout

class GridPreviewContainer(mui.FlexBox):
    def __init__(self, preview_layout: mui.FlexBox, name: str):
        super().__init__({
            "header": mui.HBox([
                mui.HBox([mui.Icon(mui.IconType.DragIndicator).prop(iconSize="small")]).prop(className="grid-layout-drag-handle", alignItems="center", cursor="move"),
                mui.Typography(name).prop(fontSize="14px", fontFamily="monospace", noWrap=True),
            ]),
            "layout": preview_layout
        })
        self.prop(flexDirection="column", width="100%", height="100%", overflowY="auto")
        self.prop(border="1px solid black")

class GridPreviewLayout(mui.FlexBox):
    def __init__(self, init_children: Dict[str, Any], tree_root: Optional[UserObjTreeProtocol] = None) -> None:
        self._init_children = init_children
        self._init_tree_root = tree_root
        self._type_to_handler_object: Dict[Type[Any],
                                           ObjectLayoutHandler] = {}
        self._default_handler = DefaultHandler()

        super().__init__()
        self.init_add_layout([*self._layout_func()])

    def _parse_obj_to_grid_item(self, obj: Any):
        obj_type = type(obj)

        reload_mgr = appctx.get_reload_manager()
        is_dcls = dataclasses.is_dataclass(obj)
        preview_layout: Optional[mui.FlexBox] = None
        metas = reload_mgr.query_type_method_meta(
            obj_type, True)
        special_methods = FlowSpecialMethods(metas)
        handler: Optional[ObjectLayoutHandler] = None
        if obj_type in self._type_to_handler_object:
            handler = self._type_to_handler_object[obj_type]
        elif is_dcls and DataClassesType in self._type_to_handler_object:
            handler = self._type_to_handler_object[DataClassesType]
        else:
            if special_methods.create_preview_layout is not None:
                if self._init_tree_root is None:
                    preview_layout = mui.flex_preview_wrapper(
                        obj, metas, reload_mgr)
                else:
                    with self._init_tree_root.enter_context(self._init_tree_root):
                        preview_layout = mui.flex_preview_wrapper(
                            obj, metas, reload_mgr)
            else:
                obj_qualname = get_qualname_of_type(type(obj))
                handler_type: Optional[Type[ObjectLayoutHandler]] = None
                if obj is not None:
                    # check standard type first, if not found, check datasetclass type.
                    if obj_type in ALL_OBJECT_LAYOUT_HANDLERS:
                        handler_type = ALL_OBJECT_LAYOUT_HANDLERS[obj_type]
                    elif obj_qualname in ALL_OBJECT_LAYOUT_HANDLERS:
                        handler_type = ALL_OBJECT_LAYOUT_HANDLERS[
                            obj_qualname]
                    elif is_dcls and DataClassesType in ALL_OBJECT_LAYOUT_HANDLERS:
                        handler_type = ALL_OBJECT_LAYOUT_HANDLERS[
                            DataClassesType]
                if handler_type is not None:
                    handler = handler_type()
        if preview_layout is not None:
            preview_grid_item = preview_layout.find_user_meta_by_type(ObjectGridLayoutItem)
            if preview_grid_item is None:
                preview_grid_item = ObjectGridLayoutItem(1.0, 1.0)
            return preview_layout, preview_grid_item, True
        elif handler is not None:
            layout = handler.create_layout(obj)
            item = handler.get_grid_layout_item()
            assert isinstance(layout, mui.FlexBox), "you must return a mui Flexbox in create_layout"
            assert isinstance(item, ObjectGridLayoutItem), "you must return a ObjectGridLayoutItem in get_grid_layout_item"
            return layout, item, False
        else:
            return None, None, False

    @mark_create_layout
    def _layout_func(self):
        # res = mui.FlexBox()
        reload_mgr = appctx.get_reload_manager()
        if self._init_tree_root is not None:
            init_root = self._init_tree_root
            self.set_flow_event_context_creator(
                lambda: init_root.enter_context(init_root))
        preview_layouts_before_packing: List[Tuple[str, mui.FlexBox, Tuple[int, int, int], bool]] = []
        grid_items: List[mui.GridItem] = []

        max_cols = 4 
        width_rate = 4

        height_rate = 4
        cols = width_rate * max_cols
        for name, obj in self._init_children.items():
            layout, grid_item, is_preview_layout = self._parse_obj_to_grid_item(obj)
            if layout is None or grid_item is None:
                continue 
            layout_w = int(round(grid_item.width * width_rate))
            layout_h = int(round(grid_item.height * height_rate))
            layout_w = min(layout_w, cols)
            preview_layouts_before_packing.append((name, layout, (layout_w, layout_h, grid_item.priority), is_preview_layout))

        rectangles = [x[2] for x in preview_layouts_before_packing]
        rect_layout = layout_rectangles_with_priority(rectangles, cols)

        for (name, layout, (layout_w, layout_h, _), is_preview_layout), new_layout in zip(preview_layouts_before_packing, rect_layout):
            obj = self._init_children[name]
            obj_type = type(obj)
            obj_type_name = obj_type.__name__
            container = GridPreviewContainer(layout, obj_type_name)
            if is_preview_layout:
                get_editable_app().observe_layout(
                    layout, partial(self._on_preview_layout_reload, container=container))

            grid_items.append(mui.GridItem(container, name, mui.GridItemProps(i=name, x=new_layout[0], y=new_layout[1], w=layout_w, h=layout_h)))
        # res.init_add_layout([
        #     mui.GridLayout(preview_layouts_v2).prop(flex=1, cols=12, draggableHandle=".grid-layout-drag-handle", rowHeight=300)
        # ])
        # print(preview_layouts_v2, self._init_children)
        self.prop(flexDirection="row", flex=1, width="100%", height="100%")
        return [mui.GridLayout(grid_items).prop(flex=1, cols=cols, 
                                                draggableHandle=".grid-layout-drag-handle", rowHeight=50)]

    async def _on_preview_layout_reload(self, layout: mui.FlexBox,
                                        create_layout: ServFunctionMeta, container: GridPreviewContainer):
        # print("DO PREVIEW LAYOUT RELOAD", create_layout.user_app_meta)
        layout_flex = await preview_layout_reload(lambda x: container.update_childs({"layout": x}), layout, create_layout)
        if layout_flex is not None:
            get_editable_app().observe_layout(layout_flex, partial(self._on_preview_layout_reload, container=container))
            return layout_flex
