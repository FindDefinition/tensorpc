import enum
from tensorpc.flow.jsonlike import IconButtonData
from ..objinspect.tree import BasicObjectTree
from ..core import CustomTreeItemHandler
from ..objinspect.analysis import get_tree_context_noexcept

from typing import (Any, Callable, Dict, Hashable, Iterable, List, Optional,
                    Set, Tuple, Type)
from tensorpc.flow.flowapp.components import mui, three
from .core import CanvasItemCfg, CanvasItemProxy


def lock_component(comp: mui.Component):
    user_meta = comp.find_user_meta_by_type(CanvasItemCfg)
    if user_meta is not None:
        user_meta.lock = True 
    else:
        user_meta = CanvasItemCfg(lock=True)
        comp._flow_user_datas.append(user_meta)
    return comp

def set_component_visible(comp: mui.Component, visible: bool):
    user_meta = comp.find_user_meta_by_type(CanvasItemCfg)
    if user_meta is not None:
        user_meta.visible = visible 
    else:
        user_meta = CanvasItemCfg(visible=visible)
        comp._flow_user_datas.append(user_meta)
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
        user_meta = obj.find_user_meta_by_type(CanvasItemCfg)
        if user_meta is not None:
            if user_meta.lock:
                res.pop()
            if not user_meta.visible:
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
            user_meta = obj.find_user_meta_by_type(CanvasItemCfg)
            if user_meta is None:
                user_meta = CanvasItemCfg()
                obj._flow_user_datas.append(user_meta)
            if isinstance(user_meta, CanvasItemCfg):
                if user_meta.type_str_override is not None:
                    node.typeStr = user_meta.type_str_override
                if user_meta.alias is not None:
                    node.alias = user_meta.alias
                user_meta.node = node
            node.fixedIconBtns = self._get_icon_button(obj)
        return None 

    async def handle_button(self, obj_trace: List[Any], node_trace: List[mui.JsonLikeNode], button_id: str) -> Optional[bool]:
        obj = obj_trace[-1]
        node = node_trace[-1]
        if isinstance(obj, mui.Component):
            item_cfg = obj.find_user_meta_by_type(CanvasItemCfg)
            if item_cfg is None:
                item_cfg = CanvasItemCfg()
                obj._flow_user_datas.append(item_cfg)
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
