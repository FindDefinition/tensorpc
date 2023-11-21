from dataclasses import dataclass
from typing import (Any, Callable, Dict, Generic, Hashable, List, Optional,
                    Type, TypeVar)

from tensorpc.core.moduleid import (TypeMeta, get_obj_type_meta,
                                    get_qualname_of_type)
from tensorpc.core.serviceunit import (AppFuncType, AppFunctionMeta, ObjectReloadManager,
                                       ObservedFunctionRegistry,
                                       ObservedFunctionRegistryProtocol,
                                       ReloadableDynamicClass,
                                       ServFunctionMeta)
from tensorpc.flow.constants import (TENSORPC_ANYLAYOUT_EFFECT_FUNC_NAME, TENSORPC_ANYLAYOUT_FUNC_NAME,
                                     TENSORPC_ANYLAYOUT_PREVIEW_FUNC_NAME,
                                     TENSORPC_LEGACY_LAYOUT_FUNC_NAME)

T = TypeVar("T")

class FlowSpecialMethods:

    def __init__(self, metas: List[ServFunctionMeta]) -> None:
        self.create_layout: Optional[ServFunctionMeta] = None
        self.auto_runs: List[ServFunctionMeta] = []
        self.effects: List[ServFunctionMeta] = []

        self.did_mount: Optional[ServFunctionMeta] = None
        self.will_unmount: Optional[ServFunctionMeta] = None
        self.create_object: Optional[ServFunctionMeta] = None
        self.create_preview_layout: Optional[ServFunctionMeta] = None

        self.metas = metas
        for m in self.metas:
            # assert m.is_binded, "metas must be binded before this class"
            if m.name == TENSORPC_ANYLAYOUT_FUNC_NAME:
                self.create_layout = m
                # handle legacy name-based layout function
                if m.user_app_meta is None:
                    m.user_app_meta = AppFunctionMeta(AppFuncType.CreateLayout)

            elif m.name == TENSORPC_LEGACY_LAYOUT_FUNC_NAME:
                self.create_layout = m
                # handle legacy name-based layout function
                if m.user_app_meta is None:
                    m.user_app_meta = AppFunctionMeta(AppFuncType.CreateLayout)

            elif m.name == TENSORPC_ANYLAYOUT_PREVIEW_FUNC_NAME:
                self.create_preview_layout = m
                # handle legacy name-based layout function
                if m.user_app_meta is None:
                    m.user_app_meta = AppFunctionMeta(AppFuncType.CreatePreviewLayout)
            elif m.name == TENSORPC_ANYLAYOUT_EFFECT_FUNC_NAME:
                self.create_preview_layout = m
                # handle legacy name-based layout function
                if m.user_app_meta is None:
                    m.user_app_meta = AppFunctionMeta(AppFuncType.Effect)
                    self.effects.append(m)

            elif m.user_app_meta is not None:
                if m.user_app_meta.type == AppFuncType.CreateLayout:
                    self.create_layout = m
                elif m.user_app_meta.type == AppFuncType.ComponentDidMount:
                    self.did_mount = m
                elif m.user_app_meta.type == AppFuncType.ComponentWillUnmount:
                    self.will_unmount = m
                elif m.user_app_meta.type == AppFuncType.CreateObject:
                    self.create_object = m
                elif m.user_app_meta.type == AppFuncType.CreatePreviewLayout:
                    self.create_preview_layout = m
                elif m.user_app_meta.type == AppFuncType.AutoRun:
                    self.auto_runs.append(m)
                elif m.user_app_meta.type == AppFuncType.Effect:
                    self.effects.append(m)

    def contains_special_method(self):
        res =  self.create_layout is not None
        res |= self.did_mount is not None
        res |= self.will_unmount is not None
        res |= self.create_object is not None
        res |= self.create_preview_layout is not None
        res |= bool(self.auto_runs)
        res |= bool(self.effects)
        return res 
    
    def collect_all_special_meta(self):
        res: List[ServFunctionMeta] = []
        if self.create_layout is not None:
            res.append(self.create_layout)
        for r in self.auto_runs:
            res.append(r)
        for r in self.effects:
            res.append(r)

        if self.did_mount is not None:
            res.append(self.did_mount)
        if self.will_unmount is not None:
            res.append(self.will_unmount)
        if self.create_object is not None:
            res.append(self.create_object)
        if self.create_preview_layout is not None:
            res.append(self.create_preview_layout)
        return res 

    def contains_autorun(self):
        return bool(self.auto_runs)

    def bind(self, obj):
        if self.create_layout is not None:
            self.create_layout.bind(obj)
        for r in self.auto_runs:
            r.bind(obj)
        if self.did_mount is not None:
            self.did_mount.bind(obj)
        if self.will_unmount is not None:
            self.will_unmount.bind(obj)
        if self.create_object is not None:
            self.create_object.bind(obj)
        if self.create_preview_layout is not None:
            self.create_preview_layout.bind(obj)
        for r in self.effects:
            r.bind(obj)


def reload_object_methods(
        obj: Any,
        previous_metas: Optional[List[ServFunctionMeta]] = None,
        reload_mgr: Optional[ObjectReloadManager] = None):
    """this function only reload leaf type, methods in base type won't be reloaded.
    """
    obj_type = type(obj)
    tmeta = get_obj_type_meta(obj_type)
    if tmeta is None:
        return None, False
    qualname_to_code: Dict[str, str] = {}
    is_reload = False
    if reload_mgr is not None:
        res = reload_mgr.reload_type(type(obj))
        is_reload = res.is_reload
        module_dict = res.module_entry.module_dict
        if res.file_entry.qualname_to_code is not None:
            qualname_to_code = res.file_entry.qualname_to_code
    else:
        module_dict = tmeta.get_reloaded_module_dict()
        is_reload = True
    if module_dict is None:
        return None, False
    new_obj_type = tmeta.get_local_type_from_module_dict(module_dict)
    if reload_mgr is not None:
        new_metas = reload_mgr.query_type_method_meta(new_obj_type)
    else:
        new_metas = ReloadableDynamicClass.get_metas_of_regular_methods(
            new_obj_type, qualname_to_code=qualname_to_code)
    # code_changed_metas: List[ServFunctionMeta] = []
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
            # if new_meta.code != meta.code:
            #     code_changed_metas.append(new_meta)
        else:
            setattr(obj, new_meta.name, new_method)
            # code_changed_metas.append(new_meta)
    return new_metas, is_reload

def bind_and_reset_object_methods(
        obj: Any,
        new_metas: List[ServFunctionMeta]):
    for new_meta in new_metas:
        new_method = new_meta.bind(obj)
        setattr(obj, new_meta.name, new_method)
    return


@dataclass
class AppObjectMeta:
    is_anylayout: bool = False


class AppReloadManager(ObjectReloadManager):
    """to resolve some side effects, users should
    always use reload manager defined in app.
    """

    def __init__(self, observed_registry: Optional[ObservedFunctionRegistryProtocol] = None) -> None:
        super().__init__(observed_registry)
        self.obj_layout_meta_cache: Dict[Any, AppObjectMeta] = {}

    def query_obj_is_anylayout(self, obj):
        obj_type = type(obj)
        if obj_type in self.obj_layout_meta_cache:
            return self.obj_layout_meta_cache[obj_type].is_anylayout
        new_metas = self.query_type_method_meta(obj_type, include_base=True)
        flow_special = FlowSpecialMethods(new_metas)
        self.obj_layout_meta_cache[obj_type] = AppObjectMeta(
            flow_special.create_layout is not None)
        return self.obj_layout_meta_cache[obj_type].is_anylayout
