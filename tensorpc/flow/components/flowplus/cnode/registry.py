import contextlib
import inspect
from typing import Callable, Optional, TypeVar, Union
import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.flow.components import flowui, mui
from tensorpc.core.moduleid import get_module_id_of_type
import contextvars
from tensorpc.flow.jsonlike import (as_dict_no_undefined,
                                    as_dict_no_undefined_no_deepcopy,
                                    merge_props_not_undefined)

@dataclasses.dataclass
class ComputeNodeConfig:
    func: Callable
    key: str
    name: str
    module_id: str
    icon_cfg: Optional[mui.IconProps] = None
    box_props: Optional[mui.FlexBoxProps] = None
    resizer_props: Optional[flowui.NodeResizerProps] = None
    layout_overflow: Optional[mui._OverflowType] = None
    is_dynamic_cls: bool = False
    # static layout
    layout: Optional[mui.FlexBox] = None

    def get_resizer(self):
        if self.resizer_props is not None:
            resizer = flowui.NodeResizer()
            merge_props_not_undefined(
                resizer.props, self.resizer_props)
            return resizer
        return None

class CustomNodeEditorContext:

    def __init__(self, cfg: Optional[ComputeNodeConfig] = None) -> None:
        self.cfg = cfg


COMPUTE_FLOW_NODE_EDITOR_CONTEXT_VAR: contextvars.ContextVar[
    Optional[CustomNodeEditorContext]] = contextvars.ContextVar(
        "computeflow_node_editor_context", default=None)


def get_node_editor_context() -> Optional[CustomNodeEditorContext]:
    return COMPUTE_FLOW_NODE_EDITOR_CONTEXT_VAR.get()


@contextlib.contextmanager
def enter_node_editor_context_object():
    ctx = CustomNodeEditorContext()
    token = COMPUTE_FLOW_NODE_EDITOR_CONTEXT_VAR.set(ctx)
    try:
        yield ctx
    finally:
        COMPUTE_FLOW_NODE_EDITOR_CONTEXT_VAR.reset(token)


T = TypeVar("T")


class ComputeNodeRegistry:

    def __init__(self, allow_duplicate: bool = True):
        self.global_dict: dict[str, ComputeNodeConfig] = {}
        self.allow_duplicate = allow_duplicate

    def register(
            self,
            func=None,
            *,
            key: Optional[str] = None,
            name: Optional[str] = None,
            icon_cfg: Optional[mui.IconProps] = None,
            resize_minmax_size: Optional[tuple[tuple[int, int],
                                               Optional[tuple[int,
                                                              int]]]] = None,
            box_props: Optional[mui.FlexBoxProps] = None,
            resizer_props: Optional[flowui.NodeResizerProps] = None,
            layout_overflow: Optional[mui._OverflowType] = None,
            layout: Optional[mui.FlexBox] = None) -> Union[T, Callable[[T], T]]:

        def wrapper(func: T) -> T:
            assert inspect.isclass(func) or inspect.isfunction(
                func
            ), "register_compute_node should be used on class or function"
            key_ = key
            module_id = get_module_id_of_type(func)
            if key_ is None:
                key_ = module_id
            name_ = name
            if name_ is None:
                name_ = func.__name__
            resizer_props_ = resizer_props
            if resizer_props_ is None:
                if resize_minmax_size is not None:
                    min_size, max_size = resize_minmax_size
                    resizer_props_ = flowui.NodeResizerProps(
                        minWidth=min_size[0], minHeight=min_size[1])
                    if max_size is not None:
                        resizer_props_.maxWidth = max_size[0]
                        resizer_props_.maxHeight = max_size[1]
            node_cfg = ComputeNodeConfig(func=func, 
                                         key=key_,
                                         name=name_,
                                         icon_cfg=icon_cfg,
                                         module_id=module_id,
                                         box_props=box_props,
                                         resizer_props=resizer_props_,
                                         layout_overflow=layout_overflow)
            editor_ctx = get_node_editor_context()
            if editor_ctx is not None:
                # when this function is used in custom editor, no need to register it.
                node_cfg.is_dynamic_cls = True
                editor_ctx.cfg = node_cfg
            else:
                if not self.allow_duplicate and key_ in self.global_dict:
                    raise KeyError("key {} already exists".format(key_))
                self.global_dict[key_] = node_cfg
            return func

        if func is None:
            return wrapper
        else:
            return wrapper(func)

    def __contains__(self, key: str):
        return key in self.global_dict

    def __getitem__(self, key: str):
        return self.global_dict[key]

    def items(self):
        yield from self.global_dict.items()


NODE_REGISTRY = ComputeNodeRegistry()


def register_compute_node(
        func=None,
        *,
        key: Optional[str] = None,
        name: Optional[str] = None,
        icon_cfg: Optional[mui.IconProps] = None,
        resize_minmax_size: Optional[tuple[tuple[int, int],
                                           Optional[tuple[int, int]]]] = None,
        box_props: Optional[mui.FlexBoxProps] = None,
        resizer_props: Optional[flowui.NodeResizerProps] = None,
        layout_overflow: Optional[mui._OverflowType] = None,
        layout: Optional[mui.FlexBox] = None):
    return NODE_REGISTRY.register(func,
                                  key=key,
                                  name=name,
                                  icon_cfg=icon_cfg,
                                  resize_minmax_size=resize_minmax_size,
                                  box_props=box_props,
                                  resizer_props=resizer_props,
                                  layout_overflow=layout_overflow,
                                  layout=layout)
