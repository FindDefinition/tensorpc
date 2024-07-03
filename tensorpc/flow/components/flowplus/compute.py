import abc
import asyncio
import contextlib
import contextvars
import enum
import inspect
from os import read
from pathlib import Path
import sys
import threading
import traceback
from collections import deque
from typing import (TYPE_CHECKING, Any, AsyncGenerator, AsyncIterator,
                    Awaitable, Coroutine, Deque, Dict, List, Literal, Mapping,
                    Optional, Tuple, Type, TypedDict, TypeVar, Union,
                    get_origin)

from colorama import init
from typing_extensions import get_type_hints, is_typeddict

from tensorpc.constants import TENSORPC_FILE_NAME_PREFIX
import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core.annocore import (AnnotatedArg, AnnotatedReturn,
                                    extract_annotated_type_and_meta, get_args,
                                    is_async_gen, is_optional,
                                    lenient_issubclass,
                                    parse_annotated_function)
from tensorpc.core.asynctools import cancel_task
from tensorpc.core.moduleid import (get_module_id_of_type,
                                    get_object_type_from_module_id,
                                    get_qualname_of_type)
from tensorpc.flow import marker
from tensorpc.flow.appctx.core import (data_storage_has_item,
                                       read_data_storage,
                                       read_data_storage_by_glob_prefix,
                                       remove_data_storage, save_data_storage)
from tensorpc.flow.components import flowui, mui
from tensorpc.flow.core.appcore import CORO_ANY, find_all_components
from tensorpc.flow.core.core import AppEvent, UserMessage
from tensorpc.flow.jsonlike import (as_dict_no_undefined,
                                    as_dict_no_undefined_no_deepcopy,
                                    merge_props_not_undefined)
if TYPE_CHECKING:
    from tensorpc.flow.components.flowplus.customnode import CustomNode

TENSORPC_FLOWUI_NODEDATA_KEY = "__tensorpc_flowui_nodedata_key"
NoneType = type(None)


def get_cflow_template_key(template_key: str):
    return f"__cflow_templates/{template_key}"


@dataclasses.dataclass
class HandleMeta:
    is_array: bool = False
    is_dict: bool = False


class ComputeFlowClasses:
    NodeWrapper = "ComputeFlowNodeWrapper"
    NodeWrappedSelected = "ComputeFlowNodeWrapperSelected"
    Header = "ComputeFlowHeader"
    IOHandleContainer = "ComputeFlowIOHandleContainer"
    InputHandle = "ComputeFlowInputHandle"
    OutputHandle = "ComputeFlowOutputHandle"
    NodeItem = "ComputeFlowNodeItem"
    CodeTypography = "ComputeFlowCodeTypography"


class ReservedNodeTypes:
    Custom = "tensorpc.cflow.CustomNode"
    AsyncGenCustom = "tensorpc.cflow.AsyncGenCustomNode"
    JsonInput = "tensorpc.cflow.JsonInputNode"
    ObjectTreeViewer = "tensorpc.cflow.ObjectTreeViewerNode"
    Expr = "tensorpc.cflow.ExprNode"


def _default_compute_flow_css():
    return {
        f".{ComputeFlowClasses.Header}": {
            "borderTopLeftRadius": "7px",
            "borderTopRightRadius": "7px",
            "justifyContent": "center",
            "backgroundColor": "#eee",
            "alignItems": "center",
        },
        f".{ComputeFlowClasses.NodeWrapper}": {
            "flexDirection": "column",
            "borderRadius": "7px",
            "alignItems": "stretch",
            "minWidth": "150px",
            "background": "white",
        },
        f".{ComputeFlowClasses.InputHandle}": {
            "position": "absolute",
            "top": "50%",
        },
        f".{ComputeFlowClasses.IOHandleContainer}": {
            "flexDirection": "row",
            "alignItems": "center",
            "position": "relative",
            "minHeight": "24px",
        },
        f".{ComputeFlowClasses.CodeTypography}": {
            "fontFamily":
            "IBMPlexMono,SFMono-Regular,Consolas,Liberation Mono,Menlo,Courier,monospace",
        },
        f".{ComputeFlowClasses.OutputHandle}": {
            "position": "absolute",
            "top": "50%",
        },
        f".{ComputeFlowClasses.NodeItem}": {
            "borderBottom": "1px solid grey"
        },
        f".{ComputeFlowClasses.NodeItem}:last-child": {
            "borderBottom": "none",
        },
        ".react-flow__node.selected": {
            f".{ComputeFlowClasses.NodeWrappedSelected}": {
                "borderStyle": "dashed",
            }
        },
        ".react-flow__handle": {
            "borderRadius": "100%",
            "height": "12px",
            "width": "12px",
            "border": "1px solid red",
            "background": "#eee"
        },
        ".react-flow__handle.connecting": {
            "background": "#ff6060"
        },
        ".react-flow__handle.valid": {
            "background": "#55dd99"
        },
        ".react-flow__handle-left": {
            "left": "-6px",
        },
        ".react-flow__handle-right": {
            "right": "-6px",
        },
        ".react-flow__resize-control.handle": {
            "width": "8px",
            "height": "8px",
        }
    }


class HandleTypePrefix:
    Input = "inp"
    Output = "out"
    Control = "ctrl"
    Option = "opt"


def is_typeddict_or_typeddict_async_gen(type):
    is_tdict = is_typeddict(type)
    if is_tdict:
        return True
    if is_async_gen(type):
        return is_typeddict(get_args(type)[0])
    return False


@dataclasses.dataclass
class NodeSideLayoutOptions:
    vertical: Union[bool, mui.Undefined] = mui.undefined


@dataclasses.dataclass
class FlowOptions:
    enable_side_layout_view: bool = True
    disable_all_resizer: bool = False


@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class AnnoHandle:
    type: Literal["source", "target"]
    prefix: str
    name: str
    is_optional: bool
    anno: AnnotatedArg
    meta: Optional[HandleMeta] = None


class NodeStatus(enum.IntEnum):
    Ready = 0
    Running = 1
    Error = 2
    Done = 3


class NodeContextMenuItemNames:
    Run = "Run"
    RunThisNode = "Run this node"
    StopGraphRun = "Stop graph run"

    CopyNode = "Copy node"
    DeleteNode = "Delete node"
    RenameNode = "Rename node"
    DebugUpdateNodeInternals = "Debug Update Internals"


class PaneContextMenuItemNames:
    SideLayout = "SideLayoutVisible"
    DisableAllResizer = "DisableAllResizer"
    ManageTemplates = "ManageTemplates"


@dataclasses.dataclass
class NodeConfig:
    width: Optional[int] = None
    height: Optional[int] = None


@dataclasses.dataclass
class WrapperConfig:
    boxProps: Optional[mui.FlexBoxProps] = None
    resizerProps: Optional[flowui.NodeResizerProps] = None


T_cnode = TypeVar("T_cnode", bound="ComputeNode")


class ComputeNode:
    def __init__(self,
                 id: str,
                 name: str,
                 node_type: Optional[str] = None,
                 init_cfg: Optional[NodeConfig] = None,
                 init_pos: Optional[flowui.XYPosition] = None,
                 icon_cfg: Optional[mui.IconProps] = None) -> None:
        self._name = name
        self.id = id
        if node_type is None:
            node_type = type(self).__name__
        self._node_type = node_type
        self._init_cfg = init_cfg
        self._icon_cfg = icon_cfg
        self._init_pos = init_pos
        self.init_node()
        # check after init node because some init node may change the compute function
        # e.g. CustomNode
        annos = self.get_compute_annotation()
        ranno = annos[1]
        if ranno is None or (not is_typeddict_or_typeddict_async_gen(
                ranno.type) and ranno.type is not NoneType):
            raise ValueError(
                "Compute function must be annotated with TypedDict or AsyncGenerator[TypedDict] return type."
            )
        if self.is_async_gen and ranno.type is not NoneType:
            assert is_async_gen(
                ranno.type
            ), "you must anno AsyncGenerator if your func is async gen"

    @property
    def is_async_gen(self):
        return inspect.isasyncgenfunction(self.compute)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def icon_cfg(self) -> Optional[mui.IconProps]:
        return self._icon_cfg

    @property
    def init_cfg(self) -> Optional[NodeConfig]:
        return self._init_cfg

    @property
    def init_wrapper_config(self) -> Optional[WrapperConfig]:
        return None

    def init_node(self):
        """init node, you can override this method to do some init work."""
        pass

    async def init_node_async(self, is_node_mounted: bool):
        """init node when compute flow mount, you can override 
        this method to do some async init work.
        """
        pass

    @abc.abstractmethod
    def compute(
        self, *args, **kwargs
    ) -> Union[Coroutine[None, None, Optional[Mapping[str, Any]]],
               AsyncGenerator[Mapping[str, Any], None]]:
        raise NotImplementedError

    def get_side_layout(self) -> Optional[mui.FlexBox]:
        return None

    def get_node_layout(self) -> Optional[mui.FlexBox]:
        return None

    def state_dict(self) -> Dict[str, Any]:
        init_cfg_dict = None
        if self._init_cfg is not None:
            init_cfg_dict = dataclasses.asdict(self._init_cfg)
        return {
            TENSORPC_FLOWUI_NODEDATA_KEY: {
                "id": self.id,
                "name": self.name,
                "module_id": get_module_id_of_type(type(self)),
                "node_type": self._node_type,
                "init_cfg": init_cfg_dict,
                "init_pos": self._init_pos,
            }
        }

    def get_compute_annotation(self):
        return parse_annotated_function(self.compute)

    def get_compute_function(self):
        """get the compute function of the node.
        usually be used if you use a wrapped function as compute function.
        """
        return self.compute

    @staticmethod
    def from_state_dict_default(data: Dict[str, Any],
                                cls: Type[T_cnode]) -> T_cnode:
        internal = data[TENSORPC_FLOWUI_NODEDATA_KEY]
        init_cfg = None
        if "init_cfg" in internal:
            if internal["init_cfg"] is not None:
                init_cfg = NodeConfig(**internal["init_cfg"])
        return cls(internal["id"], internal["name"], internal["node_type"],
                   init_cfg, internal["init_pos"])

    @classmethod
    async def from_state_dict(cls, data: Dict[str, Any]):
        return ComputeNode.from_state_dict_default(data, cls)

    # async def handle_connection_change(self, node: "ComputeNode", handle_id: str, is_delete: bool):
    #     pass

    def func_anno_to_handle(self):
        annos = self.get_compute_annotation()
        arg_annos = annos[0]
        inp_iohandles: List[AnnoHandle] = []
        out_iohandles: List[AnnoHandle] = []
        for arg_anno in arg_annos:
            param = arg_anno.param
            assert param is not None
            is_optional_val = param.default is not param.empty
            handle_meta = None
            if arg_anno.annometa is not None and isinstance(
                    arg_anno.annometa, HandleMeta):
                handle_meta = arg_anno.annometa
            iohandle = AnnoHandle("source", HandleTypePrefix.Input,
                                  arg_anno.name, is_optional_val, arg_anno,
                                  handle_meta)
            inp_iohandles.append(iohandle)
        ranno_obj = annos[1]
        assert ranno_obj is not None
        ranno = ranno_obj.type
        if ranno is NoneType:
            return inp_iohandles, out_iohandles
        assert is_typeddict_or_typeddict_async_gen(ranno)
        if is_async_gen(ranno):
            tdict_annos = get_type_hints(get_args(ranno)[0],
                                         include_extras=True)
        else:
            tdict_annos = get_type_hints(ranno, include_extras=True)
        for k, v in tdict_annos.items():
            v, anno_meta = extract_annotated_type_and_meta(v)
            handle_meta = None
            if anno_meta is not None and isinstance(anno_meta, HandleMeta):
                handle_meta = anno_meta
            ohandle = AnnoHandle(
                "target", HandleTypePrefix.Output, k, is_optional(v),
                AnnotatedArg("", None, ranno_obj.type, ranno_obj.annometa),
                handle_meta)
            out_iohandles.append(ohandle)
        return inp_iohandles, out_iohandles


@dataclasses.dataclass
class ComputeNodeRegistryItem:
    type: Type[ComputeNode]
    icon_cfg: Optional[mui.IconProps] = None
    name: str = ""
    node_type: Optional[str] = None


T = TypeVar("T", bound=Type[ComputeNode])


class ComputeNodeRegistry:
    def __init__(self, allow_duplicate: bool = True):
        self.global_dict: Dict[str, ComputeNodeRegistryItem] = {}
        self.allow_duplicate = allow_duplicate

    def register(self,
                 func=None,
                 *,
                 key: Optional[str] = None,
                 name: Optional[str] = None,
                 icon_cfg: Optional[mui.IconProps] = None,
                 node_type: Optional[str] = None):
        def wrapper(func: T) -> T:
            key_ = key
            if key_ is None:
                key_ = get_module_id_of_type(func)
            name_ = name
            if name_ is None:
                name_ = func.__name__
            if not self.allow_duplicate and key_ in self.global_dict:
                raise KeyError("key {} already exists".format(key_))
            self.global_dict[key_] = ComputeNodeRegistryItem(
                type=func, name=name_, icon_cfg=icon_cfg, node_type=node_type)
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


def register_compute_node(func=None,
                          *,
                          key: Optional[str] = None,
                          name: Optional[str] = None,
                          icon_cfg: Optional[mui.IconProps] = None,
                          node_type: Optional[str] = None):
    if node_type is None:
        node_type = key
    return NODE_REGISTRY.register(func,
                                  key=key,
                                  name=name,
                                  icon_cfg=icon_cfg,
                                  node_type=node_type)


class IOHandle(mui.FlexBox):
    def __init__(self, prefix: str, name: str, is_input: bool,
                 annohandle: AnnoHandle):
        self._is_input = is_input
        self.name = name
        self.id = f"{prefix}-{name}"
        htype = "target" if is_input else "source"
        hpos = "left" if is_input else "right"
        handle_classes = ComputeFlowClasses.InputHandle if is_input else ComputeFlowClasses.OutputHandle
        if annohandle.is_optional and is_input:
            handle_style = {"border": "1px solid #4caf50"}
            param = annohandle.anno.param
            assert param is not None
            default = param.default
            if isinstance(default, (int, float, bool)):
                name = f"{name} = {default}"
            elif default is None:
                name = f"{name} = None"
        else:
            handle_style = mui.undefined
        layout: mui.LayoutType = [
            flowui.Handle(htype, hpos, self.id).prop(className=handle_classes,
                                                     style=handle_style),
            mui.Typography(name).prop(
                variant="caption",
                flex=1,
                marginLeft="8px",
                marginRight="8px",
                textAlign="start" if is_input else "end",
                className=ComputeFlowClasses.CodeTypography)
        ]
        if not is_input:
            layout = layout[::-1]
        super().__init__(layout)
        self.annohandle = annohandle
        self.prop(
            className=
            f"{ComputeFlowClasses.IOHandleContainer} {ComputeFlowClasses.NodeItem}"
        )

    @property
    def is_optional(self):
        return self.annohandle.is_optional


class ComputeNodeWrapper(mui.FlexBox):
    def __init__(self, cnode: ComputeNode):
        self.header = mui.Typography(cnode.name).prop(variant="body1")
        self.icon_container = mui.Fragment([])
        icon_cfg = cnode.icon_cfg
        if icon_cfg is None:
            if cnode._node_type is not None and cnode._node_type in NODE_REGISTRY:
                icon_cfg = NODE_REGISTRY[cnode._node_type].icon_cfg
        if icon_cfg is not None:
            self.icon_container = mui.Fragment([
                mui.Icon(mui.IconType.Add).prop(iconSize="small",
                                                icon=icon_cfg.icon,
                                                muiColor=icon_cfg.muiColor)
            ])
        self.header_container = mui.HBox([
            self.icon_container, self.header
        ]).prop(className=
                f"{ComputeFlowClasses.Header} {ComputeFlowClasses.NodeItem}")

        inp_handles, out_handles = self._func_anno_to_ioargs(cnode)
        self.inp_handles = inp_handles
        self.out_handles = out_handles
        self.input_args = mui.Fragment([*inp_handles])
        self.output_args = mui.Fragment([*out_handles])

        self.middle_node_layout: Optional[mui.FlexBox] = None
        node_layout = cnode.get_node_layout()
        if node_layout is not None:
            self.middle_node_layout = node_layout
        self.cnode = cnode
        self._run_status = mui.Typography("run").prop(variant="caption")
        self.status_box = mui.HBox([
            self._run_status,
        ]).prop(className=f"{ComputeFlowClasses.NodeItem}")
        self.middle_node_container = mui.Fragment(([
            mui.VBox([self.middle_node_layout]).prop(
                className=ComputeFlowClasses.NodeItem,
                flex=1,
                overflow="hidden")
        ] if self.middle_node_layout is not None else []))
        resizer = self._get_resizer_from_cnode(cnode)
        self.resizers: mui.LayoutType = []
        if resizer is not None:
            self.resizers = [resizer]
        self._resizer_container = mui.Fragment([*self.resizers])
        super().__init__([
            self.header_container, self.input_args, self.middle_node_container,
            self.output_args, self.status_box, self._resizer_container
        ])
        self.prop(
            className=
            f"{ComputeFlowClasses.NodeWrapper} {ComputeFlowClasses.NodeWrappedSelected}"
        )
        self.prop(borderWidth="1px", borderStyle="solid", borderColor="black")
        if cnode.init_wrapper_config is not None and cnode.init_wrapper_config.boxProps is not None:
            merge_props_not_undefined(self.props,
                                      cnode.init_wrapper_config.boxProps)

        self.status = NodeStatus.Ready

    def _get_resizer_from_cnode(
            self, cnode: ComputeNode) -> Optional[flowui.NodeResizer]:
        if cnode.init_wrapper_config is not None:
            if cnode.init_wrapper_config.resizerProps is not None:
                resizer = flowui.NodeResizer()

                merge_props_not_undefined(
                    resizer.props, cnode.init_wrapper_config.resizerProps)
                return resizer
        return None

    async def update_header(self, new_header: str):
        await self.header.write(new_header)
        self.cnode.name = new_header

    async def update_icon_cfg(self, icon_cfg: mui.IconProps):
        icon = mui.Icon(mui.IconType.Add).prop(iconSize="small",
                                               icon=icon_cfg.icon,
                                               muiColor=icon_cfg.muiColor)
        await self.icon_container.set_new_layout([icon])

    async def set_cnode(self,
                        cnode: ComputeNode,
                        raise_on_fail: bool = False,
                        do_cnode_init_async: bool = True):
        prev_cnode = self.cnode
        try:
            inp_handles, out_handles = self._func_anno_to_ioargs(cnode)
            self.inp_handles = inp_handles
            self.out_handles = out_handles
            node_layout = cnode.get_node_layout()
            if node_layout is not None:
                self.middle_node_layout = node_layout
            self.cnode = cnode
            await self.header.write(cnode.name)
            await self.input_args.set_new_layout([*inp_handles])
            await self.output_args.set_new_layout([*out_handles])
            resizer = self._get_resizer_from_cnode(cnode)
            if resizer is not None:
                await self._resizer_container.set_new_layout([resizer])
            icon_cfg = cnode.icon_cfg
            if icon_cfg is None:
                if cnode._node_type is not None and cnode._node_type in NODE_REGISTRY:
                    icon_cfg = NODE_REGISTRY[cnode._node_type].icon_cfg
            if icon_cfg is not None:
                icon = mui.Icon(mui.IconType.Add).prop(
                    iconSize="small",
                    icon=icon_cfg.icon,
                    muiColor=icon_cfg.muiColor)
                await self.icon_container.set_new_layout([icon])
            if node_layout is not None:
                await self.middle_node_container.set_new_layout([
                    mui.VBox([node_layout
                              ]).prop(className=ComputeFlowClasses.NodeItem,
                                      flex=1,
                                      overflow="hidden")
                ])
            if do_cnode_init_async:
                await cnode.init_node_async(True)
            return raise_on_fail
        except Exception as exc:
            if raise_on_fail:
                raise exc
            else:
                traceback.print_exc()
                await self.send_exception(exc)
                return await self.set_cnode(prev_cnode,
                                            raise_on_fail=True,
                                            do_cnode_init_async=False)

    def _func_anno_to_ioargs(self, cnode: ComputeNode):
        inp_ahandles, out_ahandles = cnode.func_anno_to_handle()
        inp_iohandles: List[IOHandle] = []
        out_iohandles: List[IOHandle] = []
        for ahandle in inp_ahandles:
            iohandle = IOHandle(HandleTypePrefix.Input, ahandle.name, True,
                                ahandle)
            inp_iohandles.append(iohandle)
        for ahandle in out_ahandles:
            iohandle = IOHandle(HandleTypePrefix.Output, ahandle.name, False,
                                ahandle)
            out_iohandles.append(iohandle)
        return inp_iohandles, out_iohandles

    def update_status_event(self, status: NodeStatus) -> AppEvent:
        self.status = status
        ev = AppEvent("", {})
        if status == NodeStatus.Ready:
            ev += self.update_event(borderColor="black",
                                    boxShadow=mui.undefined)
            ev += self._run_status.update_event(value="Ready")
        elif status == NodeStatus.Running:
            ev += self.update_event(borderColor="green",
                                    boxShadow="0px 0px 10px 0px green")
            ev += self._run_status.update_event(value="Running")
        elif status == NodeStatus.Error:
            ev += self.update_event(borderColor="red", boxShadow=mui.undefined)
            ev += self._run_status.update_event(value="Error")
        elif status == NodeStatus.Done:
            ev += self.update_event(borderColor="black",
                                    boxShadow=mui.undefined)
            ev += self._run_status.update_event(value="Ready")
        return ev

    def update_status_locally(self, status: NodeStatus) -> None:
        self.status = status
        if status == NodeStatus.Ready:
            self.prop(borderColor="black", boxShadow=mui.undefined)
            self._run_status.prop(value="Ready")
        elif status == NodeStatus.Running:
            self.prop(borderColor="green", boxShadow="0px 0px 10px 0px green")
            self._run_status.prop(value="Running")
        elif status == NodeStatus.Error:
            self.prop(borderColor="red", boxShadow=mui.undefined)
            self._run_status.prop(value="Error")
        elif status == NodeStatus.Done:
            self.prop(borderColor="black", boxShadow=mui.undefined)
            self._run_status.prop(value="Ready")

    async def update_status(self, status: NodeStatus):
        return await self.send_and_wait(self.update_status_event(status))

    def state_dict(self) -> Dict[str, Any]:
        res = self.cnode.state_dict()
        return {"cnode": res, "status": int(self.status)}

    @classmethod
    async def from_state_dict(cls, data: Dict[str, Any],
                              cnode_cls: Type[ComputeNode]):
        cnode = await (cnode_cls.from_state_dict(data["cnode"]))
        await cnode.init_node_async(False)
        res = cls(cnode)
        res.update_status_locally(data["status"])
        return res


class TemplateRemoveManager(mui.FlexBox):
    def __init__(self, items: List[Tuple[str, mui.ValueType]], init_code: str, init_path: str):
        self._template_select = mui.Select("template", items,
                                           self._handle_template_select)
        self._del_btn = mui.IconButton(
            mui.IconType.Delete, self._remove_template).prop(
                confirmTitle="Remove Template",
                confirmMessage="Are you sure to remove this template?")
        self._code_editor = mui.MonacoEditor(init_code, "python",
                                             init_path).prop(flex=1, readOnly=True)
        super().__init__([
            mui.HBox([self._template_select.prop(flex=1),
                      self._del_btn]).prop(padding="8px"),
            self._code_editor,
        ])
        self.prop(height="70vh", flexDirection="column")
        self.event_before_mount.on(self._handle_mount)

    @staticmethod 
    async def create_from_app_storage():
        items, code, path = await TemplateRemoveManager._get_template_items_and_code()
        return TemplateRemoveManager(items, code, path)

    @staticmethod 
    async def _get_template_items_and_code():
        all_templates = await read_data_storage_by_glob_prefix(
            "__cflow_templates/*")
        items: List[Tuple[str, mui.ValueType]] = []
        for template_key_path in all_templates:
            template_key = template_key_path.split("/")[-1]
            items.append((template_key, template_key_path))
        code = ""
        path = ""
        if items:
            code = await read_data_storage(str(items[0][1]))
            path = f"<{TENSORPC_FILE_NAME_PREFIX}-cflow-{items[0][1]}>"
        return items, code, path

    async def _handle_mount(self):
        items, code, path = await self._get_template_items_and_code()
        await self._template_select.update_items(items, 0)
        await self._code_editor.send_and_wait(
            self._code_editor.update_event(
                value=code, path=path))

    async def _remove_template(self):
        from tensorpc.flow.components.flowplus.customnode import CustomNode
        if self._template_select.props.items:
            template_key_path = str(self._template_select.props.value)
            template_key = template_key_path.split("/")[-1]
            await remove_data_storage(template_key_path)
            all_cflows = find_all_components(ComputeFlow)
            for cflow in all_cflows:
                with enter_flow_ui_context_object(cflow.graph_ctx):
                    for node in cflow.graph.nodes:
                        wrapper = node.get_component_checked(
                            ComputeNodeWrapper)
                        if isinstance(wrapper.cnode, CustomNode):
                            if wrapper.cnode._template_key == template_key:
                                await wrapper.cnode.detach_from_template()
                await cflow.update_templates()
            # fetch all templates again
            await self._handle_mount()

    async def _handle_template_select(self, value: mui.ValueType):
        template_key_path = str(value)
        template_code = await read_data_storage(template_key_path)
        await self._code_editor.send_and_wait(
            self._code_editor.update_event(value=template_code, path=f"<{TENSORPC_FILE_NAME_PREFIX}-cflow-{template_key_path}>"))


@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class _ComputeTaskResult:
    result: Any
    exception: Optional[BaseException] = None
    exc_msg: Optional[UserMessage] = None


async def _awaitable_to_coro(aw: Awaitable):
    try:
        return _ComputeTaskResult(result=await aw)
    except Exception as exc:
        tb = sys.exc_info()[2]
        traceback.print_exc()
        exc_msg = UserMessage.from_exception("", exc, tb)
        return _ComputeTaskResult(result=None, exception=exc, exc_msg=exc_msg)


class ComputeFlow(mui.FlexBox):
    def __init__(self,
                 storage_key: str,
                 cnodes: Optional[List[ComputeNode]] = None,
                 edges: Optional[List[flowui.Edge]] = None,
                 type_to_cnode_cls: Optional[Dict[str,
                                                  Type[ComputeNode]]] = None,
                 dont_read_from_storage: bool = False):
        """
        Args:
            storage_key (str): the key to store the flow data in app storage
            cnodes (Optional[List[ComputeNode]], optional): Init list of compute nodes. Defaults to None.
            edges (Optional[List[flowui.Edge]], optional): Init list of edges. Defaults to None.
        """
        if cnodes is None:
            cnodes = []
        if edges is None:
            edges = []
        if type_to_cnode_cls is None:
            type_to_cnode_cls = {}
        self.type_to_cnode_cls = {
            ReservedNodeTypes.Custom:
            NODE_REGISTRY[ReservedNodeTypes.Custom].type
        }
        self.type_to_cnode_cls.update(type_to_cnode_cls)
        nodes: List[flowui.Node] = []
        for cnode in cnodes:
            node = self._cnode_to_node(cnode)
            nodes.append(node)
        self.storage_key = storage_key
        self.graph = flowui.Flow(
            nodes, edges,
            [flowui.MiniMap(),
             flowui.Controls(),
             flowui.Background()]).prop(zoomActivationKeyCode="z",
                                        disableKeyboardA11y=True,
                                        zoomOnScroll=False)
        self.side_container = mui.VBox([]).prop(height="100%",
                                                width="100%",
                                                overflow="hidden")
        self.side_container_bottom = mui.VBox([]).prop(height="100%",
                                                       width="100%",
                                                       overflow="hidden")

        self._node_setting_name = mui.TextField("Node Name")
        self._node_setting = mui.VBox([
            self._node_setting_name,
        ])
        self._node_setting_dialog = mui.Dialog([self._node_setting],
                                               self._handle_dialog_close)
        self._node_setting_dialog.prop(title="Node Setting")
        self._template_manage_dialog = mui.Dialog([]).prop(
            maxWidth="xl", fullWidth=True, title="Manage Templates")
        self._template_manage_dialog.event_modal_close.on(
            lambda x: self._template_manage_dialog.set_new_layout([]))
        self.graph_container = mui.HBox([
            self.graph, self._node_setting_dialog, self._template_manage_dialog
        ]).prop(width="100%", height="100%", overflow="hidden")
        self.graph_container.update_sx_props(_default_compute_flow_css())
        self._graph_with_bottom_container = mui.Allotment(
            mui.Allotment.ChildDef([
                mui.Allotment.Pane(self.graph_container),
                mui.Allotment.Pane(self.side_container_bottom, visible=False),
            ])).prop(defaultSizes=[200, 150], vertical=True)
        self.global_container = mui.Allotment(
            mui.Allotment.ChildDef([
                mui.Allotment.Pane(self._graph_with_bottom_container),
                mui.Allotment.Pane(self.side_container, visible=False),
            ])).prop(defaultSizes=[200, 100])
        self.graph.event_selection_change.on(self._on_selection)
        self._dont_read_from_storage = dont_read_from_storage
        super().__init__([self.global_container])
        target_conn_valid_map = {
            HandleTypePrefix.Input: {
                # each input (target) can only connect one output (source)
                HandleTypePrefix.Output:
                1
                # TODO deal with array/dict handle
            }
        }
        self.prop(width="100%", height="100%", overflow="hidden")
        self.graph.prop(targetValidConnectMap=target_conn_valid_map)
        menu_items = [
            mui.MenuItem(NodeContextMenuItemNames.Run,
                         NodeContextMenuItemNames.Run,
                         icon=mui.IconType.PlayArrow),
            mui.MenuItem(NodeContextMenuItemNames.RunThisNode,
                         NodeContextMenuItemNames.RunThisNode,
                         icon=mui.IconType.PlayCircleOutline),
            mui.MenuItem(NodeContextMenuItemNames.StopGraphRun,
                         NodeContextMenuItemNames.StopGraphRun,
                         icon=mui.IconType.Stop),
            mui.MenuItem("divider0", divider=True),
            mui.MenuItem(NodeContextMenuItemNames.CopyNode,
                         NodeContextMenuItemNames.CopyNode,
                         inset=True),
            mui.MenuItem(NodeContextMenuItemNames.DeleteNode,
                         NodeContextMenuItemNames.DeleteNode,
                         icon=mui.IconType.Delete),
            mui.MenuItem(NodeContextMenuItemNames.RenameNode,
                         NodeContextMenuItemNames.RenameNode,
                         inset=True),
            mui.MenuItem(NodeContextMenuItemNames.DebugUpdateNodeInternals,
                         NodeContextMenuItemNames.DebugUpdateNodeInternals,
                         inset=True),
        ]
        self.view_pane_menu_items = [
            mui.MenuItem(PaneContextMenuItemNames.ManageTemplates,
                         "Manage Templates",
                         inset=True),
            mui.MenuItem("divider_system", divider=True),
            mui.MenuItem(PaneContextMenuItemNames.SideLayout,
                         "Side Layout",
                         icon=mui.IconType.Done,
                         inset=False),
            mui.MenuItem(PaneContextMenuItemNames.DisableAllResizer,
                         "Disable All Resizer",
                         icon=None,
                         inset=True),
            mui.MenuItem("divider_view", divider=True),
        ]
        pane_menu_items: List[mui.MenuItem] = []
        for k, v in NODE_REGISTRY.items():
            icon_cfg = v.icon_cfg
            icon = mui.undefined
            if icon_cfg is not None:
                icon = icon_cfg.icon
            pane_menu_items.append(mui.MenuItem(k, v.name, icon=icon))

        self.registry_pane_items = pane_menu_items
        self.graph.prop(nodeContextMenuItems=menu_items,
                        paneContextMenuItems=self.view_pane_menu_items +
                        pane_menu_items)
        self.graph.event_node_context_menu.on(self._on_node_contextment)
        self.graph.event_pane_context_menu.on(self._on_pane_contextment)

        self.graph_ctx = ComputeFlowContext(self)
        self.flow_options = FlowOptions()

        self.set_flow_event_context_creator(
            lambda: enter_flow_ui_context_object(self.graph_ctx))

        self._shutdown_ev = asyncio.Event()

        self._schedule_task: Optional[asyncio.Task] = None

    async def _handle_dialog_close(self, value: mui.DialogCloseEvent):
        if value.ok:
            assert not isinstance(value.userData, mui.Undefined)
            node_id = value.userData["node_id"]
            new_name = self._node_setting_name.str()
            if new_name:
                await self.update_cnode_header(node_id, new_name)

    def _cnode_to_node(self, cnode: ComputeNode) -> flowui.Node:
        cnode_wrapper = ComputeNodeWrapper(cnode)
        node = flowui.Node(cnode.id,
                           flowui.NodeData(cnode_wrapper),
                           type="app",
                           deletable=False)
        node.dragHandle = f".{ComputeFlowClasses.Header}"
        if cnode._init_pos is not None:
            node.position = cnode._init_pos
        if cnode.init_cfg is not None:
            style = {}
            if cnode.init_cfg.width is not None:
                style["width"] = cnode.init_cfg.width
            if cnode.init_cfg.height is not None:
                style["height"] = cnode.init_cfg.height
            node.style = style
        return node

    async def update_cnode_header(self, node_id: str, new_header: str):
        node = self.graph.get_node_by_id(node_id)
        wrapper = node.get_component_checked(ComputeNodeWrapper)
        await wrapper.update_header(new_header)

    async def update_cnode_icon_cfg(self, node_id: str,
                                    icon_cfg: mui.IconProps):
        node = self.graph.get_node_by_id(node_id)
        wrapper = node.get_component_checked(ComputeNodeWrapper)
        await wrapper.update_icon_cfg(icon_cfg)

    async def update_templates(self):
        glob_prefix = get_cflow_template_key("*")
        res = await read_data_storage_by_glob_prefix(glob_prefix)
        menu_items: List[mui.MenuItem] = []
        for k in res.keys():
            k_path = Path(k)
            k_name = k_path.name
            menu_items.append(mui.MenuItem(k, k_name))
        if menu_items:
            final_pane_menu_items = self.view_pane_menu_items + self.registry_pane_items + [
                mui.MenuItem("divider_template", divider=True)
            ]
            final_pane_menu_items.extend(menu_items)
            await self.graph.send_and_wait(
                self.graph.update_event(
                    paneContextMenuItems=final_pane_menu_items))

    async def update_cnode(self,
                           node_id: str,
                           cnode: ComputeNode,
                           unchange_when_length_equal: bool = False):
        if unchange_when_length_equal:
            raise NotImplementedError(
                "unchange_when_length_equal not implemented, wait for reactflow 12."
            )
        prev_node = self.graph.get_node_by_id(node_id)
        wrapper = prev_node.get_component_checked(ComputeNodeWrapper)
        prev_inp_handles = wrapper.inp_handles
        prev_inp_handle_ids = [h.id for h in prev_inp_handles]
        prev_out_handles = wrapper.out_handles
        prev_out_handle_ids = [h.id for h in prev_out_handles]
        set_failed = await wrapper.set_cnode(cnode)
        if set_failed:
            # cnode remain unchanged, so just return
            return
        node_cfg = cnode.init_cfg
        if node_cfg is not None:
            style = {}
            if node_cfg.width is not None:
                style["width"] = node_cfg.width
            if node_cfg.height is not None:
                style["height"] = node_cfg.height
            await self.graph.update_node_style(node_id, style)
        cur_inp_handles = wrapper.inp_handles
        cur_inp_handle_ids = [h.id for h in cur_inp_handles]
        cur_out_handles = wrapper.out_handles
        cur_out_handle_ids = [h.id for h in cur_out_handles]
        # check inp handles
        edge_id_to_be_removed: List[str] = []
        if not unchange_when_length_equal or (
                unchange_when_length_equal
                and len(prev_inp_handles) != len(cur_inp_handles)):
            # if not equal, we assign new edge by handle id. if id not exist, we remove the edge
            for handle_id in prev_inp_handle_ids:
                if handle_id not in cur_inp_handle_ids:
                    prev_edges = self.graph.get_edges_by_node_and_handle_id(
                        node_id, handle_id)
                    edge_id_to_be_removed.extend([e.id for e in prev_edges])
        # check out handles
        if not unchange_when_length_equal or (
                unchange_when_length_equal
                and len(prev_out_handles) != len(cur_out_handles)):
            for handle_id in prev_out_handle_ids:
                if handle_id not in cur_out_handle_ids:
                    print("WTF")
                    prev_edges = self.graph.get_edges_by_node_and_handle_id(
                        node_id, handle_id)
                    edge_id_to_be_removed.extend([e.id for e in prev_edges])
        await self.graph.update_node_internals([node_id])
        await self.graph.delete_edges_by_ids(edge_id_to_be_removed)

    async def _on_node_contextment(self, data):
        item_id = data["itemId"]
        node_id = data["nodeId"]
        if item_id == NodeContextMenuItemNames.Run:
            await self._schedule_roots_by_node_id(node_id)
        elif item_id == NodeContextMenuItemNames.DeleteNode:
            await self.graph.delete_nodes_by_ids([node_id])
            await save_data_storage(self.storage_key, self.state_dict())
        elif item_id == NodeContextMenuItemNames.RenameNode:
            node = self.graph.get_node_by_id(node_id)
            wrapper = node.get_component_checked(ComputeNodeWrapper)
            await self._node_setting_name.send_and_wait(
                self._node_setting_name.update_event(value=wrapper.cnode.name))
            await self._node_setting_dialog.set_open(True,
                                                     {"node_id": node_id})
            await save_data_storage(self.storage_key, self.state_dict())
        elif item_id == NodeContextMenuItemNames.DebugUpdateNodeInternals:
            await self.graph.update_node_internals([node_id])

    async def _on_pane_contextment(self, data):
        item_id = data["itemId"]
        mouse_x = data["clientOffset"]["x"]
        mouse_y = data["clientOffset"]["y"]
        if item_id == PaneContextMenuItemNames.SideLayout:
            if self.flow_options.enable_side_layout_view:
                # disable side layout view
                await self._update_side_layout_visible(False, False)
                await self.graph.update_pane_context_menu_items([
                    mui.MenuItem(PaneContextMenuItemNames.SideLayout,
                                 icon=None,
                                 inset=True)
                ])
                self.flow_options.enable_side_layout_view = False
            else:
                await self.graph.update_pane_context_menu_items([
                    mui.MenuItem(PaneContextMenuItemNames.SideLayout,
                                 icon=mui.IconType.Done,
                                 inset=False)
                ])
                self.flow_options.enable_side_layout_view = True
        elif item_id == PaneContextMenuItemNames.DisableAllResizer:
            if self.flow_options.disable_all_resizer:
                await self.graph.send_and_wait(
                    self.graph.update_event(invisiblizeAllResizer=False))
                await self.graph.update_pane_context_menu_items([
                    mui.MenuItem(PaneContextMenuItemNames.DisableAllResizer,
                                 icon=None,
                                 inset=True)
                ])
                self.flow_options.disable_all_resizer = False
            else:
                await self.graph.send_and_wait(
                    self.graph.update_event(invisiblizeAllResizer=True))
                await self.graph.update_pane_context_menu_items([
                    mui.MenuItem(PaneContextMenuItemNames.DisableAllResizer,
                                 icon=mui.IconType.Done,
                                 inset=False)
                ])
                self.flow_options.disable_all_resizer = True
        elif item_id == PaneContextMenuItemNames.ManageTemplates:
            manager = await TemplateRemoveManager.create_from_app_storage()
            await self._template_manage_dialog.set_new_layout(
                [manager])
            await self._template_manage_dialog.set_open(True)
        else:
            if item_id in NODE_REGISTRY:
                item = NODE_REGISTRY[item_id]
                new_id = self.graph.create_unique_node_id(item.name)
                cnode = item.type(new_id,
                                  item.name,
                                  node_type=item.node_type,
                                  init_pos=flowui.XYPosition(mouse_x, mouse_y),
                                  icon_cfg=item.icon_cfg)
                await cnode.init_node_async(False)
                node = self._cnode_to_node(cnode)
            else:
                assert item_id.startswith(get_cflow_template_key(""))
                item_path = Path(item_id)
                template_key = item_path.name
                new_id = self.graph.create_unique_node_id(template_key)

                # should be template node
                custom_node_item = NODE_REGISTRY[ReservedNodeTypes.Custom]
                custom_node_cls = custom_node_item.type
                cnode = custom_node_cls(new_id,
                                        template_key,
                                        node_type=ReservedNodeTypes.Custom,
                                        init_pos=flowui.XYPosition(
                                            mouse_x, mouse_y),
                                        icon_cfg=custom_node_item.icon_cfg)
                cnode._template_key = template_key  # type: ignore
                await cnode.init_node_async(False)
                node = self._cnode_to_node(cnode)
            await self.graph.add_node(node, screen_to_flow=True)

    @marker.mark_did_mount
    async def _on_flow_mount(self):
        with enter_flow_ui_context_object(self.graph_ctx):
            if not self._dont_read_from_storage and (await
                                                     (data_storage_has_item(
                                                         self.storage_key))):
                data = await read_data_storage(self.storage_key)
                graph_props = data["graph"]
                node_states = data["node_id_to_state"]
                node_id_to_remove: List[str] = []
                for node in graph_props["nodes"]:
                    node["deletable"] = False
                    node_id = node["id"]
                    if node_id in node_states:
                        state = node_states[node_id]
                        assert "data" in node
                        internals = state["cnode"][
                            TENSORPC_FLOWUI_NODEDATA_KEY]
                        node_type = internals["node_type"]
                        node_module_id = internals["module_id"]
                        if node_type in self.type_to_cnode_cls:
                            cnode_cls = self.type_to_cnode_cls[node_type]
                        else:
                            cnode_cls = get_object_type_from_module_id(
                                node_module_id)
                            if cnode_cls is None:
                                node_id_to_remove.append(node_id)
                                continue
                        try:
                            wrapper = await ComputeNodeWrapper.from_state_dict(
                                state, cnode_cls)
                        except Exception as e:
                            node_id_to_remove.append(node_id)
                            await self.send_exception(e)
                            traceback.print_exc()
                            continue
                        node_data = node["data"]
                        node_data["component"] = wrapper
                    else:
                        if node["type"] == "app":
                            node_id_to_remove.append(node_id)
                graph_child_def = flowui.Flow.ChildDef(**graph_props)
                # remove edges that connect to removed nodes
                remain_edges: List[flowui.Edge] = []
                for edge in graph_child_def.edges:
                    if edge.source in node_id_to_remove or edge.target in node_id_to_remove:
                        continue
                    remain_edges.append(edge)
                graph_child_def.edges = remain_edges
                # remove nodes that failed to init
                print("node_id_to_remove", node_id_to_remove)
                graph_child_def.nodes = [
                    n for n in graph_child_def.nodes
                    if n.id not in node_id_to_remove
                ]
                self.graph.childs_complex.nodes = graph_child_def.nodes
                self.graph.childs_complex.edges = graph_child_def.edges
                await self.graph.set_new_layout(self.graph.childs_complex)
            else:
                for node in self.graph.nodes:
                    wrapper = node.get_component_checked(ComputeNodeWrapper)
                    await wrapper.cnode.init_node_async(True)
        await self.update_templates()
        self._shutdown_ev = asyncio.Event()

    @marker.mark_will_unmount
    async def _on_flow_unmount(self):
        self._shutdown_ev.set()
        await save_data_storage(self.storage_key, self.state_dict())
        if self._schedule_task is not None:
            await self._schedule_task

    async def _handle_edge_delete(self, edges: List[Any]):
        await save_data_storage(self.storage_key, self.state_dict())

    async def _handle_new_edge(self, data: Dict[str, Any]):
        await save_data_storage(self.storage_key, self.state_dict())

    async def _update_side_layout_visible(self, bottom: bool, right: bool):
        ev = self.global_container.update_pane_props_event(
            1, {"visible": right})
        ev += self._graph_with_bottom_container.update_pane_props_event(
            1, {"visible": bottom})
        await self.send_and_wait(ev)

    async def _on_selection(self, selection: flowui.EventSelection):
        if selection.nodes:
            if not self.flow_options.enable_side_layout_view:
                return
            node = self.graph.get_node_by_id(selection.nodes[0])
            wrapper = node.get_component_checked(ComputeNodeWrapper)
            side_layout = wrapper.cnode.get_side_layout()
            if side_layout is not None:
                option = side_layout.find_user_meta_by_type(
                    NodeSideLayoutOptions)
                vertical = False
                if option is not None and option.vertical:
                    vertical = True
                if vertical:
                    await self.side_container_bottom.set_new_layout(
                        [side_layout])
                    await self._update_side_layout_visible(True, False)

                else:
                    await self.side_container.set_new_layout([side_layout])
                    await self._update_side_layout_visible(False, True)

        else:
            await self.side_container.set_new_layout([])
            await self.side_container_bottom.set_new_layout([])
            await self._update_side_layout_visible(False, False)

    def state_dict(self):
        data = as_dict_no_undefined_no_deepcopy(self.graph.childs_complex)
        node_id_to_state: Dict[str, dict] = {}
        for node in data["nodes"]:
            if "data" in node:
                node_data = node["data"]
                if "component" in node_data:
                    comp = node_data["component"]
                    assert isinstance(comp, ComputeNodeWrapper)
                    node_id_to_state[node["id"]] = comp.state_dict()
                    node_data.pop("component")
        return {
            "graph": {
                "nodes": data["nodes"],
                "edges": data["edges"],
            },
            "node_id_to_state": node_id_to_state,
        }

    async def _schedule_roots_by_node_id(self, node_id: str):
        if not self.graph.has_node_id(node_id):
            return
        node = self.graph.get_node_by_id(node_id)
        all_nodes = self.graph.get_all_nodes_in_connected_graph(node)
        roots: List[flowui.Node] = []
        for node in all_nodes:
            if self.graph.get_source_node_and_handles(node.id):
                continue
            roots.append(node)
        if self._schedule_task is not None:
            self.graph_ctx._wait_node_inputs.update({n.id: {} for n in roots})
        else:
            with enter_flow_ui_context_object(self.graph_ctx):
                self._schedule_task = asyncio.create_task(
                    self._schedule(roots, {}, self._shutdown_ev))

    async def schedule_next(self, node_id: str, node_output: Dict[str, Any]):
        node_inputs = self._get_next_node_inputs({node_id: node_output})
        if self._schedule_task is not None:
            self.graph_ctx._wait_node_inputs.update(node_inputs)
        else:
            with enter_flow_ui_context_object(self.graph_ctx):
                self._schedule_task = asyncio.create_task(
                    self._schedule([
                        self.graph.get_node_by_id(i)
                        for i in node_inputs.keys()
                    ], node_inputs, self._shutdown_ev))

    async def schedule_node(self, node_id: str, node_inputs: Dict[str, Any]):
        node = self.graph.get_node_by_id(node_id)
        if self._schedule_task is not None:
            self.graph_ctx._wait_node_inputs.update({node.id: node_inputs})
        else:
            with enter_flow_ui_context_object(self.graph_ctx):
                self._schedule_task = asyncio.create_task(
                    self._schedule([node], {node_id: node_inputs},
                                   self._shutdown_ev))

    def _filter_node_cant_schedule(self, nodes: List[flowui.Node],
                                   node_inputs: Dict[str, Dict[str, Any]],
                                   anode_iters: Dict[str, AsyncIterator]):
        new_nodes: List[flowui.Node] = []
        for n in nodes:
            wrapper = n.get_component_checked(ComputeNodeWrapper)
            if n.id in anode_iters:
                new_nodes.append(n)
                continue
            if n.id in node_inputs:
                node_inp = node_inputs[n.id]
            else:
                node_inp = {}
            not_found = False
            for handle in wrapper.inp_handles:
                if not handle.is_optional and handle.name not in node_inp:
                    not_found = True
                    break
            if not_found:
                continue
            new_nodes.append(n)
        return new_nodes

    def _get_next_node_inputs(self, node_id_to_outputs: Dict[str, Dict[str,
                                                                       Any]]):
        new_node_inputs: Dict[str, Dict[str, Any]] = {}
        for node_id, output in node_id_to_outputs.items():
            # TODO handle array/dict handle
            node_target_and_handles = self.graph.get_target_node_and_handles(
                node_id)
            for target_node, source_handle, target_handle in node_target_and_handles:
                assert source_handle is not None and target_handle is not None
                source_handle_name = source_handle.split("-")[1]
                target_handle_name = target_handle.split("-")[1]
                if source_handle_name in output:
                    if target_node.id not in new_node_inputs:
                        new_node_inputs[target_node.id] = {}
                    new_node_inputs[target_node.id][
                        target_handle_name] = output[source_handle_name]
        return new_node_inputs

    async def _schedule(self, nodes: List[flowui.Node],
                        node_inputs: Dict[str, Dict[str, Any]],
                        shutdown_ev: asyncio.Event):
        await save_data_storage(self.storage_key, self.state_dict())
        nodes_to_schedule: List[flowui.Node] = nodes
        ctx = get_compute_flow_context()
        if ctx is not None:
            wait_nodes, wait_inputs = ctx.fetch_wait_nodes_and_inputs()
            nodes_to_schedule = nodes_to_schedule + wait_nodes
            node_inputs = {**wait_inputs, **node_inputs}

        cur_node_inputs = node_inputs.copy()
        cur_anode_iters: Dict[str, AsyncIterator] = {}
        shutdown_task = asyncio.create_task(shutdown_ev.wait())
        try:
            ctx = get_compute_flow_context()
            while (nodes_to_schedule
                   or (ctx is not None and ctx._wait_node_inputs)):
                new_nodes_to_schedule: List[flowui.Node] = []
                new_node_inputs: Dict[str, Dict[str, Any]] = {}
                # 1. validate node inputs, all input handle (not optional) must be
                #    provided in node inputs
                if ctx is not None:
                    wait_nodes, wait_inputs = ctx.fetch_wait_nodes_and_inputs()
                    nodes_to_schedule = nodes_to_schedule + wait_nodes
                    cur_node_inputs = {**cur_node_inputs, **wait_inputs}
                valid_nodes = self._filter_node_cant_schedule(
                    nodes_to_schedule, cur_node_inputs, cur_anode_iters)
                if not valid_nodes:
                    break
                # 2. remove duplicate nodes
                valid_nodes_id_set = set(n.id for n in valid_nodes)
                new_valid_nodes: List[flowui.Node] = []
                for n in valid_nodes:
                    if n.id in valid_nodes_id_set:
                        new_valid_nodes.append(n)
                        valid_nodes_id_set.remove(n.id)
                # 3. do schedule, collect tasks
                tasks: List[asyncio.Task] = []
                task_to_noded: Dict[asyncio.Task, flowui.Node] = {}
                node_outputs: Dict[str, Dict[str, Any]] = {}
                for n in valid_nodes:
                    if n.id in cur_anode_iters:
                        node_aiter = cur_anode_iters[n.id]
                        task = asyncio.create_task(_awaitable_to_coro(
                            anext(node_aiter)),
                                                   name=f"node-{n.id}")
                        tasks.append(task)
                        task_to_noded[task] = n
                        continue
                    node_inp = cur_node_inputs.get(n.id, {})
                    wrapper = n.get_component_checked(ComputeNodeWrapper)
                    await wrapper.update_status(NodeStatus.Running)
                    compute_func = wrapper.cnode.get_compute_function()
                    if wrapper.cnode.is_async_gen and inspect.isasyncgenfunction(
                            compute_func):
                        node_aiter = compute_func(**node_inp)
                        cur_anode_iters[n.id] = node_aiter
                        task = asyncio.create_task(_awaitable_to_coro(
                            anext(node_aiter)),
                                                   name=f"node-{n.id}")
                        tasks.append(task)
                        task_to_noded[task] = n
                    else:
                        try:
                            data = compute_func(**node_inp)
                        except Exception as exc:
                            await wrapper.update_status(NodeStatus.Error)
                            traceback.print_exc()
                            await self.send_exception(exc)
                            continue
                        if inspect.iscoroutine(data):
                            task = asyncio.create_task(
                                _awaitable_to_coro(data), name=f"node-{n.id}")
                            tasks.append(task)
                            task_to_noded[task] = n
                        else:
                            assert isinstance(data, dict)
                            await wrapper.update_status(NodeStatus.Ready)
                            node_outputs[n.id] = data
                is_shutdown = False
                print("Waiting")
                cur_node_tasks = tasks
                while cur_node_tasks:
                    cur_wait_tasks = cur_node_tasks + [shutdown_task]
                    (done, pending) = await asyncio.wait(
                        cur_wait_tasks, return_when=asyncio.FIRST_COMPLETED)
                    if shutdown_task in done:
                        await self.send_error("Flow shutdown", "")
                        is_shutdown = True
                        for task in pending:
                            await cancel_task(task)
                        break
                    # only shotdown task left, so we finish all node task.
                    if len(pending) == 1 and shutdown_task in pending:
                        break
                    new_tasks: List[asyncio.Task] = []
                    for ptask in pending:
                        if ptask is shutdown_task:
                            continue
                        new_tasks.append(ptask)
                    cur_node_tasks = new_tasks
                if is_shutdown:
                    break
                results = []
                for task in tasks:
                    if task.exception() is not None:
                        results.append(task.exception())
                    else:
                        results.append(task.result())
                # results = await asyncio.gather(*tasks, return_exceptions=True)
                for (tk, task_res) in zip(tasks, results):
                    node = task_to_noded[tk]
                    node_id = node.id
                    wrapper = node.get_component_checked(ComputeNodeWrapper)
                    assert isinstance(task_res, _ComputeTaskResult)
                    res = task_res.result
                    exc = task_res.exception
                    if isinstance(exc, StopAsyncIteration):
                        cur_anode_iters.pop(node_id)
                        await wrapper.update_status(NodeStatus.Ready)
                        continue
                    if isinstance(exc, BaseException):
                        await wrapper.update_status(NodeStatus.Error)
                        exc_msg = task_res.exc_msg
                        assert exc_msg is not None
                        exc_msg.uid = self._flow_uid_encoded
                        await self.put_app_event(
                            self.create_user_msg_event(exc_msg))
                        continue
                    if res is None:
                        res = {}  # for node without output, we support None.
                    if not isinstance(res, dict):
                        await wrapper.update_status(NodeStatus.Error)
                        await self.send_error(
                            f"Node {node_id} compute return type must be dict",
                            f"{type(res)}")
                        continue
                    handles = wrapper.out_handles
                    node_out_valid = True
                    for handle in handles:
                        if not handle.is_optional:
                            if handle.name not in res:
                                await self.send_error(
                                    f"Node {node_id} compute return dict missing {handle.name}",
                                    "")
                                node_out_valid = False
                                break
                    if node_out_valid:
                        node_outputs[node_id] = res
                        if node_id not in cur_anode_iters:
                            await wrapper.update_status(NodeStatus.Ready)
                    else:
                        await wrapper.update_status(NodeStatus.Error)
                # print("Node Outputs", node_outputs)
                new_node_inputs.update(
                    self._get_next_node_inputs(node_outputs))
                for node_id in new_node_inputs.keys():
                    node = self.graph.get_node_by_id(node_id)
                    new_nodes_to_schedule.append(node)
                for n in nodes_to_schedule:
                    if n.id in cur_anode_iters:
                        new_nodes_to_schedule.append(n)
                cur_node_inputs = new_node_inputs
                # print(cur_node_inputs, [n.id for n in new_nodes_to_schedule])
                nodes_to_schedule = new_nodes_to_schedule
        except Exception as exc:
            await self.send_exception(exc)
            traceback.print_exc()
            raise exc
        print("Done")
        self._schedule_task = None


class ComputeFlowContext:
    def __init__(self, cflow: ComputeFlow) -> None:
        self.cflow = cflow
        self.schedule_next_lock = threading.Lock()

        self._wait_node_inputs: Dict[str, Dict[str, Any]] = {}

    def fetch_wait_nodes_and_inputs(self):
        wait_nodes: List[flowui.Node] = [
            self.cflow.graph.get_node_by_id(nid)
            for nid in self._wait_node_inputs.keys()
        ]
        res = self._wait_node_inputs.copy()
        self._wait_node_inputs.clear()
        return wait_nodes, res


COMPUTE_FLOW_CONTEXT_VAR: contextvars.ContextVar[
    Optional[ComputeFlowContext]] = contextvars.ContextVar(
        "computeflow_context", default=None)


def get_compute_flow_context() -> Optional[ComputeFlowContext]:
    return COMPUTE_FLOW_CONTEXT_VAR.get()


@contextlib.contextmanager
def enter_flow_ui_context(cflow: ComputeFlow):
    ctx = ComputeFlowContext(cflow)
    token = COMPUTE_FLOW_CONTEXT_VAR.set(ctx)
    try:
        yield ctx
    finally:
        COMPUTE_FLOW_CONTEXT_VAR.reset(token)


@contextlib.contextmanager
def enter_flow_ui_context_object(ctx: ComputeFlowContext):
    token = COMPUTE_FLOW_CONTEXT_VAR.set(ctx)
    try:
        yield ctx
    finally:
        COMPUTE_FLOW_CONTEXT_VAR.reset(token)


async def schedule_next(node_id: str, node_output: Dict[str, Any]):
    """schedule next nodes by current node id and current node output
    """
    ctx = get_compute_flow_context()
    assert ctx is not None, "you must enter flow ui context before call this function"
    await ctx.cflow.schedule_next(node_id, node_output)


async def schedule_node(node_id: str, node_inputs: Dict[str, Any]):
    """schedule current node by node id and current node inputs
    """
    ctx = get_compute_flow_context()
    assert ctx is not None, "you must enter flow ui context before call this function"
    await ctx.cflow.schedule_node(node_id, node_inputs)


class CNodeTest1(ComputeNode):
    class OutputDict(TypedDict):
        c: int

    async def compute(self, a: int, b: int) -> OutputDict:
        return {"c": a + b}


class CNodeSourceTest1(ComputeNode):
    class OutputDict(TypedDict):
        a: int

    def init_node(self):
        self.ui_input = mui.Input("value", init="0")

    async def compute(self) -> OutputDict:
        return {"a": self.ui_input.int()}

    def get_node_layout(self) -> Optional[mui.FlexBox]:
        return mui.HBox([self.ui_input])


class CNodeViewer(ComputeNode):
    def init_node(self):
        self.md = mui.Markdown()

    async def compute(self, inp: Any) -> None:
        return await self.md.write(str(inp))

    def get_node_layout(self) -> Optional[mui.FlexBox]:
        return mui.HBox([self.md])


class AsyncSource(ComputeNode):
    class OutputDict(TypedDict):
        a: int

    async def compute(self) -> AsyncGenerator[OutputDict, None]:
        for i in range(10):
            await asyncio.sleep(0.5)
            yield {"a": i}


def _get_test_cflow():
    snode = CNodeSourceTest1("source",
                             "Source",
                             init_pos=flowui.XYPosition(-150, 100))
    asnode = AsyncSource("asource",
                         "asource",
                         init_pos=flowui.XYPosition(-150, 300))
    cnode1 = CNodeTest1("1", "Test1", init_pos=flowui.XYPosition(100, 100))
    cnode2 = CNodeViewer("2", "Viewer", init_pos=flowui.XYPosition(400, 100))
    cflow = ComputeFlow(
        "test_develop",
        [
            # snode,
            # asnode,
            # cnode1,
            # cnode2,
        ],
        dont_read_from_storage=False)
    return cflow
