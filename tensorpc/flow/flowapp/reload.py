from typing import Any, List, Optional

from tensorpc.core.moduleid import get_obj_type_meta
from tensorpc.core.serviceunit import ReloadableDynamicClass, ServFunctionMeta

def reload_object_methods(obj: Any, previous_metas: Optional[List[ServFunctionMeta]] = None):
    obj_type = type(obj)
    tmeta = get_obj_type_meta(obj_type)
    if tmeta is None:
        return None 
    module_dict = tmeta.get_reloaded_module_dict()
    if module_dict is None:
        return None 
    new_obj_type = tmeta.get_local_type_from_module_dict(module_dict)
    new_metas = ReloadableDynamicClass.get_metas_of_regular_methods(new_obj_type)
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
