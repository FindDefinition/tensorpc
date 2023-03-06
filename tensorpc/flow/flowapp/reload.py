from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type
from tensorpc.core.moduleid import get_obj_type_meta, TypeMeta
from tensorpc.core.serviceunit import ObjectReloadManager, ReloadableDynamicClass, ServFunctionMeta
from tensorpc.flow.constants import TENSORPC_ANYLAYOUT_FUNC_NAME, TENSORPC_LEGACY_LAYOUT_FUNC_NAME
from tensorpc.core.serviceunit import AppFuncType

class FlowSpecialMethods:

    def __init__(self, metas: List[ServFunctionMeta]) -> None:
        self.create_layout: Optional[ServFunctionMeta] = None
        self.auto_run: Optional[ServFunctionMeta] = None
        self.did_mount: Optional[ServFunctionMeta] = None
        self.will_unmount: Optional[ServFunctionMeta] = None
        self.create_object: Optional[ServFunctionMeta] = None

        self.metas = metas
        for m in self.metas:
            # assert m.is_binded, "metas must be binded before this class"
            if m.name == TENSORPC_ANYLAYOUT_FUNC_NAME:
                self.create_layout = m
            elif m.name == TENSORPC_LEGACY_LAYOUT_FUNC_NAME:
                self.create_layout = m
            elif m.user_app_meta is not None:
                if m.user_app_meta.type == AppFuncType.CreateLayout:
                    self.create_layout = m
                elif m.user_app_meta.type == AppFuncType.ComponentDidMount:
                    self.did_mount = m
                elif m.user_app_meta.type == AppFuncType.ComponentWillUnmount:
                    self.will_unmount = m
                elif m.user_app_meta.type == AppFuncType.CreateObject:
                    self.create_object = m
                elif m.user_app_meta.type == AppFuncType.AutoRun:
                    self.auto_run = m

    def bind(self, obj):
        if self.create_layout is not None:
            self.create_layout.bind(obj)
        if self.auto_run is not None:
            self.auto_run.bind(obj)
        if self.did_mount is not None:
            self.did_mount.bind(obj)
        if self.will_unmount is not None:
            self.will_unmount.bind(obj)
        if self.create_object is not None:
            self.create_object.bind(obj)


def reload_object_methods(obj: Any,
                          previous_metas: Optional[
                              List[ServFunctionMeta]] = None,
                        reload_mgr: Optional[ObjectReloadManager] = None):
    obj_type = type(obj)
    tmeta = get_obj_type_meta(obj_type)
    if tmeta is None:
        return None
    if reload_mgr is not None:
        res = reload_mgr.reload_type(type(obj))
        module_dict = res[0].module_dict
    else:
        module_dict = tmeta.get_reloaded_module_dict()
    if module_dict is None:
        return None
    new_obj_type = tmeta.get_local_type_from_module_dict(module_dict)
    new_metas = ReloadableDynamicClass.get_metas_of_regular_methods(
        new_obj_type)
    code_changed_metas: List[ServFunctionMeta] = []
    # print(new_metas)
    if previous_metas is not None:
        name_to_meta = {m.name: m for m in previous_metas}
    else:
        name_to_meta = {}
    for new_meta in new_metas:
        new_method = new_meta.bind(obj)
        if new_meta.name in name_to_meta:
            meta = name_to_meta[new_meta.name]
            setattr(obj, new_meta.name, new_method)
            if new_meta.code != meta.code:
                code_changed_metas.append(new_meta)
        else:
            setattr(obj, new_meta.name, new_method)
            code_changed_metas.append(new_meta)
    return code_changed_metas


@dataclass
class AppObjectMeta:
    is_anylayout: bool = False

class AppReloadManager(ObjectReloadManager):
    """to resolve some side effects, users should
    always use reload manager defined in app.
    """

    def __init__(self) -> None:
        super().__init__()
        self.obj_layout_meta_cache: Dict[Any, AppObjectMeta] = {}

    def query_obj_is_anylayout(self, obj):
        obj_type = type(obj)
        if obj_type in self.obj_layout_meta_cache:
            return self.obj_layout_meta_cache[obj_type].is_anylayout
        new_metas = self.query_type_method_meta(obj_type)
        flow_special = FlowSpecialMethods(new_metas)
        self.obj_layout_meta_cache[obj_type] = AppObjectMeta(
            flow_special.create_layout is not None)
        return self.obj_layout_meta_cache[obj_type].is_anylayout
