import abc
import contextlib
import enum
import inspect
from typing import (TYPE_CHECKING, Any, AsyncGenerator, AsyncIterator,
                    Awaitable, Callable, Coroutine, Deque, Dict, List, Literal, Mapping,
                    Optional, Tuple, Type, TypedDict, TypeVar, Union, cast,
                    get_origin)
from tensorpc.apps.cflow.coremodel import ResourceDesc
import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.dock.components import flowui, mui
from tensorpc.core.moduleid import get_module_id_of_type
import contextvars
from tensorpc.apps.cflow.nodes.cnode.handle import AnnoHandle, parse_function_to_handles
from tensorpc.dock.jsonlike import (as_dict_no_undefined,
                                    as_dict_no_undefined_no_deepcopy,
                                    merge_props_not_undefined)
import dataclasses as dataclasses_plain
from pydantic_core import core_schema
from pydantic import GetCoreSchemaHandler
if TYPE_CHECKING:
    from tensorpc.apps.cflow.executors.base import NodeExecutorBase

class ComputeNodeFlags(enum.IntFlag):
    EXEC_ALWAYS_LOCAL = enum.auto()

class ComputeNodeBase(abc.ABC):
    @abc.abstractmethod
    def compute(
        self, *args, **kwargs
    ) -> Union[Coroutine[None, None, Optional[Mapping[str, Any]]],
               AsyncGenerator[Mapping[str, Any], None]]:
        raise NotImplementedError

    def get_compute_func(self) -> Callable:
        return self.compute

    def get_node_preview_layout(self, drafts: Any) -> Optional[mui.FlexBox]:
        return None

    def get_node_detail_layout(self, drafts: Any) -> Optional[mui.FlexBox]:
        return None

    def get_remote_preview_container(self) -> Optional[mui.FlexBox]:
        """Compute flow will call this and set remote box as child of this container.
        """
        return None

    def get_remote_detail_container(self) -> Optional[mui.FlexBox]:
        """Compute flow will call this and set remote box as child of this container.
        """
        return None

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any,
                                     _handler: GetCoreSchemaHandler):
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.any_schema(),
        )

    @classmethod
    def validate(cls, v):
        if not isinstance(v, ComputeNodeBase):
            raise ValueError('ComputeNodeBase required')
        return v

class ComputeNodeFuncWrapper(ComputeNodeBase):
    def __init__(self, desc: "ComputeNodeDesc") -> None:
        self._desp = desc
        assert inspect.isfunction(desc.func), "func should be a function"
        self._compute_func = desc.func 

    def compute(
        self, *args, **kwargs
    ) -> Union[Coroutine[None, None, Optional[Mapping[str, Any]]],
               AsyncGenerator[Mapping[str, Any], None]]:
        raise NotImplementedError

    def get_compute_func(self):
        return self._compute_func

    def get_node_preview_layout(self, drafts: Any) -> Optional[mui.FlexBox]:
        if self._desp.layout_creator is not None:
            return self._desp.layout_creator(drafts)
        return None

    def get_node_detail_layout(self, drafts: Any) -> Optional[mui.FlexBox]:
        if self._desp.detail_layout_creator is not None:
            return self._desp.detail_layout_creator(drafts)
        return None

@dataclasses.dataclass
class ComputeNodeDesc:
    func: Union[Callable, type[ComputeNodeBase]]
    key: str
    name: str
    module_id: str
    icon_cfg: Optional[mui.IconProps] = None
    box_props: Optional[mui.FlexBoxProps] = None
    resizer_props: Optional[flowui.NodeResizerProps] = None
    layout_overflow: Optional[mui.OverflowType] = None
    is_dynamic_cls: bool = False
    resource_desp: Optional[ResourceDesc] = None
    # static layout
    # remote layout is only allowed in class-based node.
    layout_creator: Optional[Callable[[Any], mui.FlexBox]] = None
    detail_layout_creator: Optional[Callable[[Any], mui.FlexBox]] = None
    # state def
    state_dcls: Optional[type] = None
    # options
    flags: ComputeNodeFlags = ComputeNodeFlags(0)
    # private executor
    # WARNING: node with private executor always run in local.
    temp_executor_creator: Optional[Callable[[Any], Any]] = None
    layout_in_remote: bool = False

    def get_resizer(self):
        if self.resizer_props is not None:
            resizer = flowui.NodeResizer()
            merge_props_not_undefined(
                resizer.props, self.resizer_props)
            return resizer
        return None

    def get_node_init_width_height(self) -> Optional[tuple[int, int]]:
        if self.resizer_props is not None:
            if not isinstance(self.resizer_props.minWidth, int) or not isinstance(self.resizer_props.minHeight, int):
                return None
            return (self.resizer_props.minWidth, self.resizer_props.minHeight)
        return None

class CustomNodeEditorContext:

    def __init__(self, cfg: Optional[ComputeNodeDesc] = None) -> None:
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
        self.global_dict: dict[str, ComputeNodeDesc] = {}
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
            layout_overflow: Optional[mui.OverflowType] = None,
            layout_creator: Optional[Callable[[Any], mui.FlexBox]] = None,
            detail_layout_creator: Optional[Callable[[Any], mui.FlexBox]] = None,
            resource_desp: Optional[ResourceDesc] = None,
            state_dcls: Optional[type] = None,
            flags: ComputeNodeFlags = ComputeNodeFlags(0),
            temp_executor_creator: Optional[Callable[[Any], Any]] = None) -> Union[T, Callable[[T], T]]:

        def wrapper(func: T) -> T:
            assert inspect.isclass(func) or inspect.isfunction(
                func
            ), "register_compute_node should be used on class or function"
            if inspect.isclass(func):
                assert issubclass(func, ComputeNodeBase), "class should be subclass of ComputeNodeBase"
                assert layout_creator is None, "you should override `get_node_layout` method instead of use layout_creator when using class"
            key_ = key
            editor_ctx = get_node_editor_context()
            if editor_ctx is not None:
                module_id = ""
                key_ = ""
            else:
                module_id = get_module_id_of_type(func)
                if key_ is None:
                    key_ = module_id
            name_ = name
            if name_ is None:
                name_ = func.__name__
            if state_dcls is not None:
                assert inspect.isclass(state_dcls), "state_dcls should be a class"
                assert dataclasses.is_pydantic_dataclass(state_dcls), "state_dcls must be a pydantic dataclass (support from-dict)"
                try:
                    state_dcls()
                except:
                    raise ValueError("state_dcls must be default constructable")
            resizer_props_ = resizer_props
            if resizer_props_ is None:
                if resize_minmax_size is not None:
                    min_size, max_size = resize_minmax_size
                    resizer_props_ = flowui.NodeResizerProps(
                        minWidth=min_size[0], minHeight=min_size[1])
                    if max_size is not None:
                        resizer_props_.maxWidth = max_size[0]
                        resizer_props_.maxHeight = max_size[1]
            node_cfg = ComputeNodeDesc(func=func, 
                                         key=key_,
                                         name=name_,
                                         icon_cfg=icon_cfg,
                                         module_id=module_id,
                                         box_props=box_props,
                                         resizer_props=resizer_props_,
                                         layout_overflow=layout_overflow,
                                         layout_creator=layout_creator,
                                         detail_layout_creator=detail_layout_creator,
                                         resource_desp=resource_desp,
                                         flags=flags,
                                         state_dcls=state_dcls,
                                         temp_executor_creator=temp_executor_creator)
            # parse function annotation to validate it.
            # TODO add class support
            if inspect.isclass(func):
                node_obj = func()
                assert isinstance(node_obj, ComputeNodeBase)
                parse_function_to_handles(node_obj.compute, is_dynamic_cls=editor_ctx is not None)
            else:
                parse_function_to_handles(func, is_dynamic_cls=editor_ctx is not None)

            if editor_ctx is not None:
                # when this function is used in custom editor, no need to register it.
                node_cfg.is_dynamic_cls = True
                editor_ctx.cfg = node_cfg
            else:
                if not self.allow_duplicate and key_ in self.global_dict:
                    raise KeyError("key {} already exists".format(key_))
                self.global_dict[key_] = node_cfg
            return cast(T, func)

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
        layout_overflow: Optional[mui.OverflowType] = None,
        layout_creator: Optional[Callable[[Any], mui.FlexBox]] = None,
        detail_layout_creator: Optional[Callable[[Any], mui.FlexBox]] = None,
        resource_desp: Optional[ResourceDesc] = None,
        state_dcls: Optional[type] = None,
        flags: ComputeNodeFlags = ComputeNodeFlags(0),
        temp_executor_creator: Optional[Callable[[Any], Any]] = None):
    editor_ctx = get_node_editor_context()
    if editor_ctx is None:
        assert key is not None, "you must provide a GLOBAL unique key for the node to make sure code of node can be moved."
    return NODE_REGISTRY.register(func,
                                  key=key,
                                  name=name,
                                  icon_cfg=icon_cfg,
                                  resize_minmax_size=resize_minmax_size,
                                  box_props=box_props,
                                  resizer_props=resizer_props,
                                  layout_overflow=layout_overflow,
                                  layout_creator=layout_creator,
                                  detail_layout_creator=detail_layout_creator,
                                  resource_desp=resource_desp,
                                  flags=flags,
                                  temp_executor_creator=temp_executor_creator,
                                  state_dcls=state_dcls)

def parse_code_to_compute_cfg(code: str):
    with enter_node_editor_context_object() as ctx:
        exec(code, {})
        cfg = ctx.cfg
        assert cfg is not None, "no compute node registered, you must define a standard compute node and register it"
        return cfg

@dataclasses_plain.dataclass
class ComputeNodeRuntime:
    cfg: ComputeNodeDesc
    cnode: ComputeNodeBase
    # required by scheduler
    inp_handles: list[AnnoHandle]
    out_handles: list[AnnoHandle]
    executor: Optional[Any] = None # TODO find a way to replace Any with real type
    impl_code: str = ""
    # remote executor that stores current node instance.
    remote_executor: Optional[Any] = None

    def has_private_exec(self):
        return self.cfg.temp_executor_creator is not None

    def create_temp_executor(self) -> "NodeExecutorBase":
        from tensorpc.apps.cflow.executors.base import NodeExecutorBase
        assert self.cfg.temp_executor_creator is not None
        temp_executor = self.cfg.temp_executor_creator(self)
        assert isinstance(temp_executor, NodeExecutorBase)
        return temp_executor

def get_compute_node_runtime(cfg: ComputeNodeDesc, code: str = "") -> ComputeNodeRuntime:
    if inspect.isclass(cfg.func):
        cnode = cfg.func()
        inp_handles, out_handles = parse_function_to_handles(cnode.compute, cfg.is_dynamic_cls)
        return ComputeNodeRuntime(cfg, cnode, inp_handles, out_handles)
    inp_handles, out_handles = parse_function_to_handles(cfg.func, cfg.is_dynamic_cls)
    return ComputeNodeRuntime(cfg, ComputeNodeFuncWrapper(cfg), inp_handles, out_handles, impl_code=code)


def get_registry_func_modules_for_remote():
    # get all modules of registered nodes for remote executor to import
    module_ids: set[str] = set()
    for key, cfg in NODE_REGISTRY.items():
        obj = cfg.func
        module_id = get_module_id_of_type(obj)
        module_ids.add(module_id)
    return list(module_ids)