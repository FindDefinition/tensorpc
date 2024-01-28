from contextlib import nullcontext
from functools import partial
import math
from typing import Any, Callable, Coroutine, Dict, Hashable, Iterable, List, Literal, Optional, Set, Tuple, Type, Union
from tensorpc.core.moduleid import get_qualname_of_type
from tensorpc.core.serviceunit import AppFuncType, ServFunctionMeta
from tensorpc.flow.flowapp.appcore import get_editable_app
from tensorpc.flow.flowapp.components import mui
from tensorpc.flow.flowapp.components.mui import LayoutType
from tensorpc.flow.flowapp.components.plus.reload_utils import preview_layout_reload
from tensorpc.flow.flowapp.reload import FlowSpecialMethods
from tensorpc.flow.marker import mark_create_layout
from tensorpc.flow.flowapp.objtree import UserObjTreeProtocol
import dataclasses

from tensorpc.flow.flowapp.components.plus.core import (
    ALL_OBJECT_LAYOUT_HANDLERS, ObjectGridItemConfig, ObjectLayoutHandler,
    DataClassesType)
# from tensorpc.flow.flowapp.components.plus.handlers.common import DefaultHandler

from typing import List, Tuple



def layout_rectangles_with_priority(
        rectangles: List[Tuple[int, int, int]],
        bounding_width: int) -> List[Tuple[int, int, Tuple[int, int]]]:
    # author: copilot
    # Sort rectangles by height
    rectangles_with_index = [(i, x) for i, x in enumerate(rectangles)]
    rectangles_with_index.sort(key=lambda r: (r[1][1], r[1][2]), reverse=True)

    # Initialize variables
    x, y, row_height = 0, 0, 0
    layout: List[Tuple[int, int, Tuple[int, int]]] = [(0, 0, (0, 0)) for _ in range(len(rectangles))]

    # Layout rectangles
    for i, rectangle in rectangles_with_index:
        width, height, _ = rectangle

        # If rectangle doesn't fit in current row, start a new row
        if x + width > bounding_width:
            x = 0
            y += row_height
            row_height = 0

        # Place rectangle
        layout[i] = (x, y, (width, height))
        # layout.append((x, y, rectangle))
        x += width
        row_height = max(row_height, height)

    return layout


class GridPreviewContainer(mui.FlexBox):

    def __init__(self, preview_layout: mui.FlexBox, name: str, close_callback: Optional[Callable[[], Coroutine[None, None, None]]] = None):
        super().__init__({
            "header":
            mui.HBox([
                mui.HBox([
                    mui.Icon(mui.IconType.DragIndicator).prop(iconSize="small")
                ]).prop(className="grid-layout-drag-handle",
                        alignItems="center",
                        cursor="move"),
                mui.Typography(name).prop(fontSize="14px",
                                          fontFamily="monospace",
                                          noWrap=True),
            ]),
            "layout":
            preview_layout
        })
        self.prop(flexDirection="column",
                  width="100%",
                  height="100%",
                  overflowY="auto")
        self.prop(border="1px solid black")


@dataclasses.dataclass
class _GridLayoutItem:
    obj: Any
    layout: mui.FlexBox
    item_cfg: ObjectGridItemConfig
    is_preview_layout: bool

@dataclasses.dataclass
class _GridLayoutedItem:
    layout_item: _GridLayoutItem
    grid_item: mui.GridItem

class GridPreviewLayout(mui.FlexBox):

    def __init__(self,
                 init_children: Dict[str, Any],
                 tree_root: Optional[UserObjTreeProtocol] = None,
                 max_cols: int = 4,
                 width_rate: int = 4,
                 height_rate: int = 4,
                 use_typename_as_title: bool = False) -> None:
        self._init_children = init_children
        self._tree_root = tree_root
        self._type_to_handler_object: Dict[Type[Any], ObjectLayoutHandler] = {}
        # self._default_handler = DefaultHandler()
        self.max_cols = max_cols
        self.width_rate = width_rate
        self.height_rate = height_rate
        self.grid_items: List[_GridLayoutedItem] = []
        self.use_typename_as_title = use_typename_as_title
        super().__init__()
        self.init_add_layout([*self._layout_func()])

    def get_object_by_name(self, name: str) -> Any:
        for x in self.grid_items:
            if x.grid_item.name == name:
                return x.layout_item.obj
        raise KeyError(name)

    def _check_type_support_preview(self, type: Type) -> bool:
        if type in self._type_to_handler_object:
            return True
        reload_mgr = self.flow_app_comp_core.reload_mgr
        metas = reload_mgr.query_type_method_meta(type, True, True)
        special_methods = FlowSpecialMethods(metas)
        if special_methods.create_preview_layout is not None:
            return True 
        return ALL_OBJECT_LAYOUT_HANDLERS.check_type_exists(type)

    def _parse_obj_to_grid_item(self, obj: Any):
        from tensorpc.flow.flowapp import appctx
        if isinstance(obj, mui.FlexBox):
            obj_grid_item = obj.find_user_meta_by_type(
                ObjectGridItemConfig)
            if obj_grid_item is None:
                obj_grid_item = ObjectGridItemConfig(1.0, 1.0)
            return _GridLayoutItem(obj, obj, obj_grid_item, False)
        obj_type = type(obj)

        reload_mgr = appctx.get_reload_manager()
        is_dcls = dataclasses.is_dataclass(obj)
        preview_layout: Optional[mui.FlexBox] = None
        metas = reload_mgr.query_type_method_meta(obj_type, True, True)
        special_methods = FlowSpecialMethods(metas)
        handler: Optional[ObjectLayoutHandler] = None
        if obj_type in self._type_to_handler_object:
            handler = self._type_to_handler_object[obj_type]
        elif is_dcls and DataClassesType in self._type_to_handler_object:
            handler = self._type_to_handler_object[DataClassesType]
        else:
            if special_methods.create_preview_layout is not None:
                if self._tree_root is None:
                    preview_layout = mui.flex_preview_wrapper(
                        obj, metas, reload_mgr)
                else:
                    with self._tree_root.enter_context(
                            self._tree_root):
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
                        handler_type = ALL_OBJECT_LAYOUT_HANDLERS[obj_qualname]
                    elif is_dcls and DataClassesType in ALL_OBJECT_LAYOUT_HANDLERS:
                        handler_type = ALL_OBJECT_LAYOUT_HANDLERS[
                            DataClassesType]
                if handler_type is not None:
                    handler = handler_type()
        if preview_layout is not None:
            preview_grid_item = preview_layout.find_user_meta_by_type(
                ObjectGridItemConfig)
            if preview_grid_item is None:
                preview_grid_item = ObjectGridItemConfig(1.0, 1.0)
            return _GridLayoutItem(obj, preview_layout, preview_grid_item, True)
        elif handler is not None:
            layout = handler.create_layout(obj)
            item = handler.get_grid_layout_item(obj)
            assert isinstance(
                layout,
                mui.FlexBox), "you must return a mui Flexbox in create_layout"
            assert isinstance(
                item, ObjectGridItemConfig
            ), "you must return a ObjectGridItemConfig in get_grid_layout_item"
            return _GridLayoutItem(obj, layout, item, False)
        else:
            return None

    def _layout_items_inplace(self, items: Dict[str, _GridLayoutItem]):
        cols = self.width_rate * self.max_cols

        preview_layouts_before_packing: List[Tuple[str, mui.FlexBox,
                                                   Tuple[int, int,
                                                         int], bool]] = []
        for name, grid_item in items.items():
            layout_w = int(round(grid_item.item_cfg.width * self.width_rate))
            layout_h = int(round(grid_item.item_cfg.height * self.height_rate))
            layout_w = min(layout_w, cols)
            preview_layouts_before_packing.append(
                (name, grid_item.layout, (layout_w, layout_h,
                                          grid_item.item_cfg.priority),
                 grid_item.is_preview_layout))

        rectangles = [x[2] for x in preview_layouts_before_packing]
        rect_layout = layout_rectangles_with_priority(rectangles, cols)
        for (name, layout, (layout_w, layout_h, _),
             is_preview_layout), new_layout in zip(
                 preview_layouts_before_packing, rect_layout):
            item = items[name]
            item.item_cfg.x = new_layout[0]
            item.item_cfg.y = new_layout[1]
            item.item_cfg.w = layout_w
            item.item_cfg.h = layout_h

    def _grid_layout_items_to_ui_items(self, name_to_grid_items: Dict[str, _GridLayoutItem], use_typename_as_title: bool):
        grid_items: List[_GridLayoutedItem] = []
        for name, grid_layout_item in name_to_grid_items.items():
            obj = grid_layout_item.obj
            obj_type = type(obj)
            obj_type_name = obj_type.__name__
            container = GridPreviewContainer(grid_layout_item.layout, obj_type_name if use_typename_as_title else name)
            if grid_layout_item.is_preview_layout:
                get_editable_app().observe_layout(
                    grid_layout_item.layout,
                    partial(self._on_preview_layout_reload,
                            container=container))
            item = mui.GridItem(
                    container, name,
                    mui.GridItemProps(i=name,
                                      x=grid_layout_item.item_cfg.x,
                                      y=grid_layout_item.item_cfg.y,
                                      w=grid_layout_item.item_cfg.w,
                                      h=grid_layout_item.item_cfg.h))
            grid_items.append(_GridLayoutedItem(grid_layout_item, item))
        return grid_items

    @mark_create_layout
    def _layout_func(self):
        # res = mui.FlexBox()
        if self._tree_root is not None:
            init_root = self._tree_root
            self.set_flow_event_context_creator(
                lambda: init_root.enter_context(init_root))
        cols = self.width_rate * self.max_cols
        name_to_grid_item: Dict[str, _GridLayoutItem] = {}
        for name, obj in self._init_children.items():
            grid_item = self._parse_obj_to_grid_item(obj)
            if grid_item is None:
                continue
            name_to_grid_item[name] = grid_item
        self._layout_items_inplace(name_to_grid_item)
        grid_items: List[_GridLayoutedItem] = self._grid_layout_items_to_ui_items(name_to_grid_item, self.use_typename_as_title)
        # res.init_add_layout([
        #     mui.GridLayout(preview_layouts_v2).prop(flex=1, cols=12, draggableHandle=".grid-layout-drag-handle", rowHeight=300)
        # ])
        # print(preview_layouts_v2, self._init_children)
        self.prop(flexDirection="row", flex=1, width="100%", height="100%")
        self.grid_items = grid_items
        return [
            mui.GridLayout([x.grid_item for x in grid_items]).prop(
                flex=1,
                cols=int(cols),
                draggableHandle=".grid-layout-drag-handle",
                rowHeight=50
                )
        ]

    async def _on_preview_layout_reload(self, layout: mui.FlexBox,
                                        create_layout: ServFunctionMeta,
                                        container: GridPreviewContainer):
        # print("DO PREVIEW LAYOUT RELOAD", create_layout.user_app_meta)
        ctx = nullcontext()
        if self._tree_root is not None:
            ctx = self._tree_root.enter_context(
                    self._tree_root)
        with ctx:
            layout_flex = await preview_layout_reload(
                lambda x: container.update_childs({"layout": x}), layout,
                create_layout)
        if layout_flex is not None:
            get_editable_app().observe_layout(
                layout_flex,
                partial(self._on_preview_layout_reload, container=container))
            return layout_flex

    def set_tree_root(self, tree_root: UserObjTreeProtocol):
        self._tree_root = tree_root
        self.set_flow_event_context_creator(
            lambda: tree_root.enter_context(tree_root))

    async def set_new_items(self, item: Dict[str, Any]):
        name_to_grid_item: Dict[str, _GridLayoutItem] = {}
        for name, obj in item.items():
            grid_item = self._parse_obj_to_grid_item(obj)
            if grid_item is None:
                continue
            name_to_grid_item[name] = grid_item
        self._layout_items_inplace(name_to_grid_item)
        grid_items: List[_GridLayoutedItem] = self._grid_layout_items_to_ui_items(name_to_grid_item, self.use_typename_as_title)
        self.grid_items = grid_items
        cols = self.width_rate * self.max_cols
        await self.set_new_layout([mui.GridLayout([x.grid_item for x in grid_items]).prop(
                flex=1,
                cols=cols,
                draggableHandle=".grid-layout-drag-handle",
                rowHeight=50)])

    async def update_items(self, item: Dict[str, Any]):
        name_to_grid_item: Dict[str, _GridLayoutItem] = {}
        prev_layouted_items = self.grid_items.copy()
        name_to_prev_layouted_items = {x.grid_item.name: x for x in prev_layouted_items}
        for name, obj in item.items():
            grid_item = self._parse_obj_to_grid_item(obj)
            if grid_item is None:
                continue
            name_to_grid_item[name] = grid_item
            if name in name_to_prev_layouted_items:
                name_to_prev_layouted_items.pop(name)
        name_to_grid_item.update({x.grid_item.name: x.layout_item for x in name_to_prev_layouted_items.values()})
        self._layout_items_inplace(name_to_grid_item)
        grid_items: List[_GridLayoutedItem] = self._grid_layout_items_to_ui_items(name_to_grid_item, self.use_typename_as_title)
        cols = self.width_rate * self.max_cols
        await self.set_new_layout([mui.GridLayout([x.grid_item for x in grid_items]).prop(
                flex=1,
                cols=cols,
                draggableHandle=".grid-layout-drag-handle",
                rowHeight=50)])

    async def delete_items(self, item: List[str]):
        name_to_grid_item: Dict[str, _GridLayoutItem] = {}
        prev_layouted_items = self.grid_items.copy()
        name_to_prev_layouted_items = {x.grid_item.name: x for x in prev_layouted_items}
        for name in name_to_prev_layouted_items.keys():
            if name in item:
                name_to_prev_layouted_items.pop(name)
        name_to_grid_item.update({x.grid_item.name: x.layout_item for x in name_to_prev_layouted_items.values()})
        self._layout_items_inplace(name_to_grid_item)
        grid_items: List[_GridLayoutedItem] = self._grid_layout_items_to_ui_items(name_to_grid_item, self.use_typename_as_title)
        cols = self.width_rate * self.max_cols
        await self.set_new_layout([mui.GridLayout([x.grid_item for x in grid_items]).prop(
                flex=1,
                cols=cols,
                draggableHandle=".grid-layout-drag-handle",
                rowHeight=50)])
        
    async def clear_items(self):
        cols = self.width_rate * self.max_cols
        await self.set_new_layout([mui.GridLayout([]).prop(
                flex=1,
                cols=cols,
                draggableHandle=".grid-layout-drag-handle",
                rowHeight=50)])
