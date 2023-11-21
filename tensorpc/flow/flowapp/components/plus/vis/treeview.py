import enum
from tensorpc.flow.jsonlike import IconButtonData
from ..objinspect.tree import BasicObjectTree
from ..objinspect.core import CustomTreeItemHandler
from ..objinspect.analysis import get_tree_context_noexcept

from typing import (Any, Callable, Dict, Hashable, Iterable, List, Optional,
                    Set, Tuple, Type)
from tensorpc.flow.flowapp.components import mui, three
from .core import CanvasItemCfg, CanvasItemProxy


def lock_component(comp: mui.Component):
    if comp._flow_user_data is not None:
        assert isinstance(comp._flow_user_data, CanvasItemCfg)
        comp._flow_user_data.lock = True 
    else:
        comp._flow_user_data = CanvasItemCfg(lock=True)
    return comp

def set_component_visible(comp: mui.Component, visible: bool):
    if comp._flow_user_data is not None:
        assert isinstance(comp._flow_user_data, CanvasItemCfg)
        comp._flow_user_data.visible = visible 
    else:
        comp._flow_user_data = CanvasItemCfg(visible=visible)
    return comp


class CanvasButtonType(enum.Enum):
    Visibility = "visibility"
    Delete = "delete"

class CanvasTreeItemHandler(CustomTreeItemHandler):

    def _get_icon_button(self, obj: mui.Component) -> List[IconButtonData]:
        res = [
            IconButtonData(CanvasButtonType.Visibility.value, mui.IconType.Visibility, "toggle visibility"),
            IconButtonData(CanvasButtonType.Delete.value, mui.IconType.Delete, "toggle visibility"),
        ]
        if isinstance(obj._flow_user_data, CanvasItemCfg):
            if obj._flow_user_data.lock:
                res.pop()
            if not obj._flow_user_data.visible:
                res[0].icon = mui.IconType.VisibilityOff
            if not isinstance(obj, (three.Object3dBase, three.Object3dContainerBase)):
                res.pop(0)
        return res 

    async def get_childs(self, obj: Any) -> Optional[Dict[str, Any]]:
        """if return None, we will use default method to extract childs
        of object.
        """
        # print(obj, isinstance(obj, mui.Component) and three.is_three_component(obj), "WTF")
        if isinstance(obj, mui.Component) and three.is_three_component(obj):
            if isinstance(obj, mui.ContainerBase):
                return obj._child_comps
            return {}
        return {}

    def patch_node(self, obj: Any, node: mui.JsonLikeNode) -> Optional[mui.JsonLikeNode]:
        """modify/patch node created from `parse_obj_to_tree_node`
        """
        if isinstance(obj, mui.Component) and three.is_three_component(obj):
            # buttons: visibility, delete
            if obj._flow_user_data is None:
                item_cfg = CanvasItemCfg()
                obj._flow_user_data = item_cfg
            if isinstance(obj._flow_user_data, CanvasItemCfg):
                if obj._flow_user_data.type_str_override is not None:
                    node.typeStr = obj._flow_user_data.type_str_override
                if obj._flow_user_data.alias is not None:
                    node.alias = obj._flow_user_data.alias
                obj._flow_user_data.node = node
                
            node.fixedIconBtns = self._get_icon_button(obj)
        return None 

    async def handle_button(self, obj_trace: List[Any], node_trace: List[mui.JsonLikeNode], button_id: str) -> Optional[bool]:
        obj = obj_trace[-1]
        node = node_trace[-1]
        if isinstance(obj, mui.Component):
            item_cfg = obj._flow_user_data
            if item_cfg is None:
                item_cfg = CanvasItemCfg()
                obj._flow_user_data = item_cfg
            if button_id == CanvasButtonType.Visibility.value:
                item_cfg.visible = not item_cfg.visible
                node.fixedIconBtns = self._get_icon_button(obj)
                await get_tree_context_noexcept().tree.update_subtree(node)
                if isinstance(obj, (three.Object3dBase, three.Object3dContainerBase)):
                    await obj.update_object3d(visible=item_cfg.visible)
            elif button_id == CanvasButtonType.Delete.value:
                if len(obj_trace) > 1 and not item_cfg.lock:
                    obj_container = obj_trace[-2]
                    if isinstance(obj_container, mui.ContainerBase):
                        await obj_container.remove_childs_by_keys([node.name])
                        await get_tree_context_noexcept().get_tree_instance(BasicObjectTree).update_tree()
                        return True
        return None
    
    async def handle_context_menu(self, obj_trace: List[Any], node_trace: List[mui.JsonLikeNode], userdata: Dict[str, Any]) -> Optional[bool]:
        return None
