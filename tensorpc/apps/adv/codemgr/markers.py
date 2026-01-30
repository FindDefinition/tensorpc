from typing import Any, Callable, Literal, Optional, TypeVar

from tensorpc.utils.registry import HashableRegistry

T = TypeVar("T")
from tensorpc.apps.adv.codemgr.core import BackendHandle, BaseNodeCodeMeta, BaseParseResult

MARKERS_REGISTRY = HashableRegistry[Callable]()

@MARKERS_REGISTRY.register_no_key
def mark_global_script(name: str, node_id: str, position: tuple[float, float], 
        ref_node_id: Optional[str] = None, ref_import_path: Optional[list[str]] = None) -> Callable[[T], T]:
    # we actually don't use this metadata, we read ast directly.
    def wrapper(fn_wrapped: T) -> T:
        return fn_wrapped   
    return wrapper

@MARKERS_REGISTRY.register_no_key
def mark_global_script_end() -> Callable[[T], T]:
    # we actually don't use this metadata, we read ast directly.
    def wrapper(fn_wrapped: T) -> T:
        return fn_wrapped   
    return wrapper

@MARKERS_REGISTRY.register_no_key
def mark_symbol_dep() -> Callable[[T], T]:
    # we actually don't use this metadata, we read ast directly.
    def wrapper(fn_wrapped: T) -> T:
        return fn_wrapped   
    return wrapper

@MARKERS_REGISTRY.register_no_key
def mark_symbol_dep_end() -> Callable[[T], T]:
    # we actually don't use this metadata, we read ast directly.
    def wrapper(fn_wrapped: T) -> T:
        return fn_wrapped   
    return wrapper

@MARKERS_REGISTRY.register_no_key
def mark_ref_node_dep() -> Callable[[T], T]:
    # we actually don't use this metadata, we read ast directly.
    def wrapper(fn_wrapped: T) -> T:
        return fn_wrapped   
    return wrapper

@MARKERS_REGISTRY.register_no_key
def mark_ref_node_dep_end() -> Callable[[T], T]:
    # we actually don't use this metadata, we read ast directly.
    def wrapper(fn_wrapped: T) -> T:
        return fn_wrapped   
    return wrapper

@MARKERS_REGISTRY.register_no_key
def mark_ref_node(func: Any, ntype: int, node_id: str, position: tuple[float, float], 
        ref_node_id: Optional[str] = None,
        inlineflow_name: Optional[str] = None) -> Callable[[T], T]:
    # we actually don't use this metadata, we read ast directly.
    return func

@MARKERS_REGISTRY.register_no_key
def mark_subflow_def(name: str, node_id: str, position: tuple[float, float], inlineflow_name: Optional[str] = None, is_folder: bool = False) -> Any:
    # we actually don't use this metadata, we read ast directly.
    return 

@MARKERS_REGISTRY.register_no_key
def mark_class_node(name: str, node_id: str, position: tuple[float, float], inlineflow_name: Optional[str] = None, is_folder: bool = False) -> Any:
    return None 

@MARKERS_REGISTRY.register_no_key
def mark_inlineflow() -> Callable[[T], T]:
    # we actually don't use this metadata, we read ast directly.
    def wrapper(fn_wrapped: T) -> T:
        return fn_wrapped   
    return wrapper

@MARKERS_REGISTRY.register_no_key
def mark_out_indicator(node_id: str, position: tuple[float, float], conn_node_id: str, conn_handle_id: str, alias: Optional[str] = None) -> Callable[[T], T]:
    # we actually don't use this metadata, we read ast directly.
    def wrapper(fn_wrapped: T) -> T:
        return fn_wrapped   
    return wrapper

__TENSORPC_ADV_SYMBOL_DCLS_META__ = "__tensorpc_adv_symbol_dcls_meta__"

@MARKERS_REGISTRY.register_no_key
def mark_symbol_group(node_id: str, position: tuple[float, float], ref_node_id: Optional[str] = None) -> Callable[[T], T]:
    # we actually don't use this metadata, we read ast directly.
    def wrapper(fn_wrapped: T) -> T:
        setattr(fn_wrapped, __TENSORPC_ADV_SYMBOL_DCLS_META__, BaseNodeCodeMeta(node_id, position, ref_node_id))
        return fn_wrapped   
    return wrapper

@MARKERS_REGISTRY.register_no_key
def mark_fragment_def(node_id: str, position: tuple[float, float],  # flags: int, 
        alias_map: Optional[str] = None,
        inlineflow_name: Optional[str] = None) -> Callable[[T], T]:
    # we actually don't use this metadata, we read ast directly.
    def wrapper(fn_wrapped: T) -> T:
        return fn_wrapped   
    return wrapper

@MARKERS_REGISTRY.register_no_key
def mark_class_def() -> Callable[[T], T]:
    # we actually don't use this metadata, we read ast directly.
    def wrapper(fn_wrapped: T) -> T:
        return fn_wrapped   
    return wrapper

@MARKERS_REGISTRY.register_no_key
def mark_user_edge(id: str, source: str, source_handle: str, target: str, target_handle: str) :
    return None 

@MARKERS_REGISTRY.register_no_key
def mark_markdown_node(id: str, position: tuple[float, float], width: float, height: float, content: str):
    return None 

