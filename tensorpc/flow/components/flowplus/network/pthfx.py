import ast
from collections import namedtuple
import contextlib
import dataclasses
from operator import getitem
import traceback
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from tensorpc.core.moduleid import get_qualname_of_type
from tensorpc.core.tree_id import UniqueTreeIdForTree
from tensorpc.flow.components import flowui, mui
from typing_extensions import override
from torch.fx import GraphModule, Interpreter, Tracer
from torch.export import ExportedProgram
from torch.export.graph_signature import (
    ExportGraphSignature,
    InputKind,
    OutputKind,
    OutputSpec,
    TensorArgument,
)
from tensorpc.flow.components.flowplus.network.defs import PytorchNodeMeta, EdgeTensorMeta
import contextvars
from tensorpc.utils.rich_logging import get_logger

import torch
import torch.fx
from torch.return_types import all_return_types
LOGGER = get_logger("tensorpc.flowui.pytorch")


class GraphContext:

    def __init__(self):
        self.node_id_to_data = {}


NODE_CONTEXT: contextvars.ContextVar[Optional[
    torch.fx.Node]] = contextvars.ContextVar("PTH_FX_NODE_CTX", default=None)
GRAPH_CONTEXT: contextvars.ContextVar[
    Optional[GraphContext]] = contextvars.ContextVar("PTH_FX_GRAPH_CTX",
                                                     default=None)


@contextlib.contextmanager
def enter_node_context(node: torch.fx.Node):
    tok = NODE_CONTEXT.set(node)
    try:
        yield
    finally:
        NODE_CONTEXT.reset(tok)


def get_node_context_noexcept():
    obj = NODE_CONTEXT.get()
    assert obj is not None, "can only be called in op method"
    return obj


@contextlib.contextmanager
def enter_graph_context(ctx: GraphContext):
    tok = GRAPH_CONTEXT.set(ctx)
    try:
        yield
    finally:
        GRAPH_CONTEXT.reset(tok)


def get_graph_context_noexcept():
    obj = GRAPH_CONTEXT.get()
    assert obj is not None, "can only be called in op method"
    return obj


_BUILTIN_PREFIX = get_qualname_of_type(type(getattr)).split(".")[0]


def _inspect_th_ret_type(rt):
    _ignore_fields = ["count", "index"]
    all_fields = dir(rt)
    res_fields = []
    for f in all_fields:
        if not f.startswith("__") and not f.startswith("n_"):
            if f not in _ignore_fields:
                res_fields.append(f)
    return res_fields


_ATEN_NAME_MAP = {
    "aten::_native_batch_norm_legit_functional": "batch_norm",
    "aten::scaled_dot_product_attention": "SDPA",
}


@dataclasses.dataclass
class FunctionalFlowTree:
    root: Dict
    module_id_to_tree_ids: Dict[str, List[List[int]]]
    all_node_ids_with_stack: List[str]
    tree_id_to_node: Dict[str, Dict]

    def get_node_by_list_idx(self, idxes: List[int]):
        cur = self.root
        for i in idxes:
            cur = cur["childs"][i]
        return cur



@dataclasses.dataclass
class PytorchFlowOutputPartial(flowui.SymbolicGraphOutput[PytorchNodeMeta,
                                                   EdgeTensorMeta]):
    id_to_nodes: Dict[str, flowui.Node] = dataclasses.field(default_factory=dict)
    id_to_edges: Dict[str, flowui.Edge] = dataclasses.field(default_factory=dict)
    node_id_to_inp_handle_to_edges: Dict[
        str, Dict[Optional[str],
                  List[flowui.Edge]]] = dataclasses.field(default_factory=dict)
    node_id_to_out_handle_to_edges: Dict[
        str, Dict[Optional[str],
                  List[flowui.Edge]]] = dataclasses.field(default_factory=dict)

def _default_qname_to_cared_params_and_name(name: str, qname: str, submodule_id: str, module: torch.nn.Module) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        subm = module.get_submodule(submodule_id)
        layer_params: Optional[Dict[str, Any]] = None
        if qname.startswith("torch.nn.modules"):
            # use __constants__ to get constant params
            if hasattr(subm, "__constants__"):
                try:
                    constants = subm.__constants__
                    layer_params = {}
                    for k in constants:
                        layer_params[k] = getattr(subm, k)
                except:
                    traceback.print_exc()
                
        if qname.startswith("torch.nn.modules.conv.Conv"):
            conv_layer_types = (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Conv1d, torch.nn.ConvTranspose1d,
                                torch.nn.Conv3d, torch.nn.ConvTranspose3d)
            assert isinstance(subm, conv_layer_types)
            if layer_params is None:
                layer_params = {
                    "in_channels": subm.in_channels,
                    "out_channels": subm.out_channels,
                    "kernel_size": subm.kernel_size,
                    "stride": subm.stride,
                    "padding": subm.padding,
                    "dilation": subm.dilation,
                    "groups": subm.groups,
                }
            return layer_params, f"{name}({subm.in_channels}x{subm.out_channels})"
        elif qname == "torch.nn.modules.linear.Linear":
            assert isinstance(subm, torch.nn.Linear)
            if layer_params is None:
                layer_params = {
                    "in_features": subm.in_features,
                    "out_features": subm.out_features,
                }
            return layer_params, f"{name}({subm.in_features}x{subm.out_features})"
        elif qname == "torch.nn.modules.container.Sequential":
            assert isinstance(subm, torch.nn.Sequential)
            child_type_names = []
            for m in subm:
                child_type_names.append(type(m).__name__)
            layer_params = {
                "childs": child_type_names,
            }
            return layer_params, f"{name}({len(subm)})"
        elif qname == "torch.nn.modules.sparse.Embedding":
            assert isinstance(subm, torch.nn.modules.sparse.Embedding)
            if layer_params is None:
                layer_params = {
                    "num_embeddings": subm.num_embeddings,
                    "embedding_dim": subm.embedding_dim,
                    "padding_idx": subm.padding_idx,
                    "max_norm": subm.max_norm,
                    "norm_type": subm.norm_type,
                    "scale_grad_by_freq": subm.scale_grad_by_freq,
                    "sparse": subm.sparse
                }
            return layer_params, f"{name}({subm.num_embeddings}, {subm.embedding_dim})"
        else:
            return layer_params, None

    except AttributeError:
        traceback.print_exc()
        return None, None 
    return None, None 

@dataclasses.dataclass
class PytorchFlowOutput(flowui.SymbolicGraphOutput[PytorchNodeMeta,
                                                   EdgeTensorMeta]):
    ftree: Optional[FunctionalFlowTree] = None
    use_multiple_handle_node: bool = False

    def create_graph_with_expanded_ids(
        self,
        expanded_ids: List[str],
        expanded_id_is_module: bool = False,
        module: Optional[torch.nn.Module] = None,
        submodule_id: Optional[str] = None,
        submodule_id_is_module: bool = True,
    ) -> PytorchFlowOutputPartial:
        """Create a new graph with expanded nodes.

        Args:
            expanded_ids (List[str]): list of expanded ids
            expanded_id_is_module (bool, optional): if True, expanded_ids are module ids. Defaults to False.
            module (Optional[torch.nn.Module], optional): module. When `submodule_id` exists, it is submodule, not
                original module. Defaults to None.
            submodule_id (Optional[str], optional): submodule id. Defaults to None.
            submodule_id_is_module (bool, optional): if True, submodule_id is module id. Defaults to True.
        """
        assert self.ftree is not None
        if not expanded_id_is_module:
            id_need_expand: Set[str] = set(expanded_ids)
        else:
            id_need_expand: Set[str] = set()
            for module_id in expanded_ids:
                if module_id not in self.ftree.module_id_to_tree_ids:
                    continue
                tree_ids = self.ftree.module_id_to_tree_ids[module_id]
                for tree_id in tree_ids:
                    for j in range(len(tree_id)):
                        id_need_expand.add(".".join(map(str, tree_id[:j + 1])))
        submod_node_range: Tuple[int, int] = (-1, -1)
        submod_node_module: Optional[str] = None
        if submodule_id is not None:
            if submodule_id_is_module:
                for expanded_id in expanded_ids:
                    assert expanded_id.startswith(submodule_id)
                assert submodule_id in self.ftree.module_id_to_tree_ids
                tree_id = self.ftree.module_id_to_tree_ids[submodule_id][0]
                ftree_node = self.ftree.tree_id_to_node[".".join(map(str, tree_id))]
            else:
                # submodule_id is ftree id
                ftree_node = self.ftree.tree_id_to_node[submodule_id]
            submod_node_range = (ftree_node["start"], ftree_node["end"])
            submod_node_module = ftree_node["module"]

        # iterate ftree, if id not in id_need_expand, it will be merged to single node.
        stack = [self.ftree.root]
        merge_list: List[Tuple[flowui.Node, List[str]]] = []
        merged_node_data: List[PytorchNodeMeta] = []
        submod_merge_nodes: List[flowui.Node] = []
        while stack:
            cur = stack.pop()
            if cur["id"] in id_need_expand:
                stack.extend(cur["childs"])
            else:
                qname = cur['qname']
                cls_name = qname.split(".")[-1]
                merged_node = flowui.Node(id=f"M-{cur['module']}-{cls_name}",
                                          data=flowui.NodeData(label=cls_name))
                merged_node.style = {"backgroundColor": "aliceblue"}
                nodes_to_merge = self.ftree.all_node_ids_with_stack[
                    cur["start"]:cur["end"]]
                merge_list.append((merged_node, nodes_to_merge))
                merged_meta = PytorchNodeMeta(op=qname,
                                    is_merged=True,
                                    module_id=UniqueTreeIdForTree.from_parts(cur["module"].split(".")),
                                    ftree_id=cur["id"],
                                    module_qname=qname)
                additional_args = None
                if module is not None:
                    merged_module_id = cur["module"]
                    new_name: Optional[str] = None
                    # print(merged_module_id, submod_node_module)
                    if submod_node_module is not None and submod_node_module != "":
                        # if submodule is used, we only update params and name for nodes in submodule
                        # other nodes will filtered during create subflow.
                        if merged_module_id.startswith(submod_node_module):
                            submod_node_module_parts = submod_node_module.split(".")
                            merged_module_id_parts = merged_module_id.split(".")
                            merged_module_id = ".".join(merged_module_id_parts[len(submod_node_module_parts):])
                            additional_args, new_name = _default_qname_to_cared_params_and_name(cls_name, qname, merged_module_id, module)
                    else:
                        additional_args, new_name = _default_qname_to_cared_params_and_name(cls_name, qname, merged_module_id, module)
                    if new_name is not None:
                        if not isinstance(merged_node.data, mui.Undefined):
                            merged_node.data.label = new_name
                # check merged node contains single node, if so, we use additional params from op node
                if len(nodes_to_merge) == 1:
                    node_id = nodes_to_merge[0]
                    if node_id in self.node_id_to_data:
                        if additional_args is None:
                            additional_args = self.node_id_to_data[node_id].additional_args
                        merged_meta.stack_trace = self.node_id_to_data[node_id].stack_trace

                # for merged node, we use first stack trace from child nodes 
                for node_id_test in nodes_to_merge:
                    if node_id_test in self.node_id_to_data:
                        stt = self.node_id_to_data[node_id_test].stack_trace
                        if stt is not None:
                            merged_meta.stack_trace = self.node_id_to_data[node_id_test].stack_trace
                            break   
                merged_meta.additional_args = additional_args
                merged_node_data.append(
                    merged_meta)
                if cur["start"] >= submod_node_range[0] and cur["end"] <= submod_node_range[1]:
                    submod_merge_nodes.append(merged_node)
        
        # merge nodes
        nodes = self.nodes
        edges = self.edges
        internals = flowui.FlowInternals()
        internals.set_from_nodes_edges(nodes, edges)
        internals, _, prev_node_id_to_data, prev_edge_id_to_data = internals.merge_nodes_with_data(
            merge_list, merged_node_data, self.node_id_to_data, self.edge_id_to_data)

        if self.use_multiple_handle_node:
            for merged_node, _ in merge_list:
                inp_handles = list(internals.node_id_to_inp_handle_to_edges[merged_node.id].keys())
                out_handles = list(internals.node_id_to_out_handle_to_edges[merged_node.id].keys())
                assert not isinstance(merged_node.data, mui.Undefined)
                merged_node.data.sourceHandleIds = out_handles
                merged_node.data.targetHandleIds = inp_handles

        if submodule_id is not None:
            submod_node_ids = self.ftree.all_node_ids_with_stack[submod_node_range[0]:submod_node_range[1]]
            all_merged_node_ids_set = set()
            for merged_node, merged_node_ids in merge_list:
                all_merged_node_ids_set.update(merged_node_ids)

            # we ensure all expand modules are submodule of provided submodule_id
            # so all merged node are child nodes
            new_submod_ids = [n.id for n in submod_merge_nodes]

            for submod_node_id in submod_node_ids:
                if submod_node_id not in all_merged_node_ids_set:
                    new_submod_ids.append(submod_node_id)
            internals, inp_node_edges, out_node_edges = internals.create_sub_flow(new_submod_ids)
            for n, edges in inp_node_edges:
                n.style = mui.undefined
                if n.id in prev_node_id_to_data:
                    new_data = PytorchNodeMeta("placeholder", is_io_node=True)
                    if edges[0].id in prev_edge_id_to_data:
                        new_data.output_desps = [prev_edge_id_to_data[edges[0].id].raw]
                    prev_node_id_to_data[n.id] = new_data
            for n, edges in out_node_edges:
                n.style = mui.undefined
                # since we just copy node as output node, we need to create a io meta. 
                new_data = PytorchNodeMeta("placeholder", is_io_node=True)
                if edges[0].id in prev_edge_id_to_data:
                    new_data.output_desps = [prev_edge_id_to_data[edges[0].id].raw]
                prev_node_id_to_data[n.id] = new_data

        if not self.use_multiple_handle_node:
            internals = internals.create_internals_with_none_handle()
        return PytorchFlowOutputPartial(nodes=internals.nodes,
                                          edges=internals.edges,
                                          node_id_to_data=prev_node_id_to_data,
                                          edge_id_to_data=prev_edge_id_to_data,
                                          id_to_edges=internals.id_to_edge,
                                          id_to_nodes=internals.id_to_node,
                                          node_id_to_inp_handle_to_edges=internals.node_id_to_inp_handle_to_edges,
                                          node_id_to_out_handle_to_edges=internals.node_id_to_out_handle_to_edges)

    def create_graph_with_expanded_modules(
            self,
            expanded_modules: List[str],
            module: Optional[torch.nn.Module] = None,
            submodule_id: Optional[str] = None,
            submodule_id_is_module: bool = True):
        return self.create_graph_with_expanded_ids(expanded_modules,
                                                   expanded_id_is_module=True,
                                                   module=module,
                                                   submodule_id=submodule_id,
                                                   submodule_id_is_module=submodule_id_is_module)


class PytorchExportBuilder(flowui.SymbolicFlowBuilder[PytorchNodeMeta,
                                                      EdgeTensorMeta]):

    def __init__(self, use_multiple_handle_node: bool = False):
        super().__init__(use_multiple_handle_node=use_multiple_handle_node)
        self._ftree_cache: Optional[FunctionalFlowTree] = None

    def _build_tree_from_module_stack(self,
                                      node_id_to_meta: Dict[str,
                                                            PytorchNodeMeta]):
        """Tree Structure:
        {
            "id": int index in parent or 0 for root (convert to str),
            "childs": List[Dict],
            "start": int # start index in overall nodes (with stack),
            "end": int # end index (exclusive) in overall nodes (with stack),
            "module": str,
            "qname": str,
        }
        """
        cnt = 0
        root_node = {
            "id": "",
            "childs": [],
            "start": 0,
            "emd": -1,
            "module": "",
            "qname": "",
        }
        stack: List[dict] = [root_node]
        stack_child_cnts: List[int] = [-1]

        module_id_to_tree_ids: Dict[str, List[List[int]]] = {"": []}
        all_node_ids_with_stack: List[str] = []
        tree_id_to_node: Dict[str, Dict] = {}
        for node_id, meta in node_id_to_meta.items():
            if meta.module_stack is not None:
                all_node_ids_with_stack.append(node_id)
                # compare node stack with current stack
                # pop until the same
                for i, (v, qname) in enumerate(meta.module_stack):
                    if v not in module_id_to_tree_ids:
                        module_id_to_tree_ids[v] = []
                    if i < len(stack) - 1:
                        if stack[i + 1]["module"] != v:
                            cur_length = len(stack)
                            for j in range(i + 1, cur_length):
                                item = stack.pop()
                                stack_child_cnts.pop()
                                item["end"] = cnt
                            assert len(
                                stack
                            ) >= 1, f"stack must have at least one item"
                    # after pop, we need to push
                    if i >= len(stack) - 1:
                        stack_child_cnts[-1] += 1
                        new_item = {
                            "id": ".".join(map(str, stack_child_cnts)),
                            "childs": [],
                            "start": cnt,
                            "end": -1,
                            "module": v,
                            "qname": qname,
                        }
                        tree_id_to_node[new_item["id"]] = new_item
                        module_id_to_tree_ids[v].append(
                            stack_child_cnts.copy())
                        stack[-1]["childs"].append(new_item)
                        stack_child_cnts.append(-1)
                        stack.append(new_item)
                # pop if stack is longer
                if len(meta.module_stack) < len(stack) - 1:
                    cur_length = len(stack)
                    for j in range(len(meta.module_stack) + 1, cur_length):
                        item = stack.pop()
                        stack_child_cnts.pop()
                        item["end"] = cnt
                cnt += 1
        # pop all
        for i in range(1, len(stack)):
            item = stack.pop()
            item["end"] = cnt
        return FunctionalFlowTree(root_node["childs"][0],
                                  module_id_to_tree_ids,
                                  all_node_ids_with_stack, tree_id_to_node)

    def build_pytorch_detached_flow(
            self,
            module: torch.nn.Module,
            out_immedinates: Sequence[flowui.SymbolicImmediate]):
        out_datas: List[PytorchNodeMeta] = []
        for imme in out_immedinates:
            meta = PytorchNodeMeta("placeholder", is_io_node=True)
            if imme.userdata is not None:
                # should be EdgeTensorMeta
                assert isinstance(imme.userdata, EdgeTensorMeta)
                meta.output_desps = [imme.userdata.raw]
            out_datas.append(meta)
        graph_res = self.build_detached_flow(out_immedinates, False, out_datas)
        if self._ftree_cache is None:
            self._ftree_cache = self._build_tree_from_module_stack(
                self._id_to_node_data)
        return PytorchFlowOutput(graph_res.nodes, graph_res.edges,
                                 graph_res.node_type_map,
                                 graph_res.node_id_to_data,
                                 graph_res.edge_id_to_data, self._ftree_cache,
                                 self._use_multiple_handle_node)


class FlowUIInterpreter(Interpreter):

    def __init__(self,
                 gm: Union["GraphModule", ExportedProgram],
                 builder: PytorchExportBuilder,
                 original_mod: Optional[torch.nn.Module] = None,
                 verbose: bool = False):

        if isinstance(gm, ExportedProgram):
            assert original_mod is not None
            self._original_mod = original_mod
            self._gm = gm.graph_module
            self._is_export = True
            self._export_param_dict = {}
            self._export_buffer_mu_keep_flags = []

            for p in gm.graph_signature.input_specs:
                if p.kind == InputKind.PARAMETER:
                    target = p.target
                    if target is not None:
                        self._export_param_dict[
                            p.arg.name] = original_mod.get_parameter(target)
                elif p.kind == InputKind.BUFFER:
                    target = p.target
                    if target is not None:
                        self._export_param_dict[
                            p.arg.name] = original_mod.get_buffer(target)
                assert p.kind != InputKind.CONSTANT_TENSOR, "model shouldn't contain any constant tensor, convert them to buffer."

            for p in gm.graph_signature.output_specs:
                if p.kind == OutputKind.BUFFER_MUTATION:
                    self._export_buffer_mu_keep_flags.append(False)
                else:
                    self._export_buffer_mu_keep_flags.append(True)
        else:
            self._original_mod = original_mod
            self._gm = gm
            self._is_export = False
            self._export_param_dict = {}
            self._export_buffer_mu_keep_flags = []
        super().__init__(self._gm)

        self._verbose = verbose

        self._builder = builder

        self._op_name_to_th_ret_types = {}
        for rt in all_return_types:
            op_name = get_qualname_of_type(rt).split(".")[-1]
            self._op_name_to_th_ret_types[op_name] = rt
        self._torch_builtin_prefix = ".".join(
            get_qualname_of_type(type(torch.topk)).split(".")[:-1])

    @override
    def run_node(self, n: torch.fx.Node) -> Any:
        with enter_node_context(n):
            return super().run_node(n)

    def call_module(self, target: Any, args: Tuple, kwargs: dict) -> Any:
        mod = self.fetch_attr(target)
        if isinstance(mod, torch.nn.Module):
            name = type(mod).__name__
        else:
            name = str(mod)
        msg = f"call_module {target} {name} {args} {kwargs}"
        if self._verbose:
            print(msg)
        inp_handles, additional_args = self._get_inp_handles_and_addi_args(
            name, args, kwargs)
        if not inp_handles:
            return super().call_function(mod, args, kwargs)

        op, output_desps, _ = self.create_op_node(name, list(inp_handles.keys()),
                                               [f"{name}-out"], target, args,
                                               kwargs, module_stack_fx=[(target, get_qualname_of_type(type(mod)))])
        op.style = {"backgroundColor": "aliceblue"}

        c_list = self._builder.call_op_node(op, inp_handles)
        return c_list[0]

    def call_function(self, target: Any, args: Tuple, kwargs: dict) -> Any:
        op_ret_type_fields = None
        node = get_node_context_noexcept()
        if target is getattr:
            return super().call_function(target, args, kwargs)
        if isinstance(target, str):
            name = str(target)
        else:
            qname = get_qualname_of_type(target)
            name = qname.split(".")[-1]
            if qname.startswith(self._torch_builtin_prefix):
                if name in self._op_name_to_th_ret_types:
                    op_ret_type = self._op_name_to_th_ret_types[name]
                    op_ret_type_fields = _inspect_th_ret_type(op_ret_type)
        op_has_param = False
        op_has_buffer = False
        schema = None
        schema_returns = None
        if self._is_export:
            qname = get_qualname_of_type(type(target))
            if qname.startswith("torch"):
                for arg in args:
                    if isinstance(arg, torch.nn.Parameter):
                        op_has_param = True
                        break
                for arg in args:
                    if not isinstance(arg, torch.nn.Parameter) and isinstance(arg, torch.Tensor):
                        op_has_buffer = True
                        break
                # analysis aten ops schema to get number of outputs
                schema = target._schema
                name = schema.name
                num_output = len(schema.returns)
                schema_returns = schema.returns
                if num_output > 1:
                    op_ret_type_fields = [f"O-{i}" for i in range(num_output)]
            elif target is getitem:
                # come from split
                if not isinstance(args[0], flowui.SymbolicImmediate):
                    return super().call_function(target, args, kwargs)
        msg = f"call_function {get_qualname_of_type(type(target))} {target} {name} "
        if self._verbose:
            print(msg)
        raw_op_name = name
        if name in _ATEN_NAME_MAP:
            name = _ATEN_NAME_MAP[name]
        if name.startswith("aten::"):
            # remove aten:: prefix
            name = name[6:]
        inp_handles, additional_args = self._get_inp_handles_and_addi_args(
            name, args, kwargs, schema)
        if not inp_handles:
            return super().call_function(target, args, kwargs)
        if op_ret_type_fields is None:
            if self._is_export:
                # for split, export can return list of tensors instead of 
                # single symbolic result.
                # TODO if split dim is dynamic
                if "val" in node.meta:
                    val = node.meta["val"]
                    if not isinstance(val, (tuple, list)):
                        # list of faketensor or symint
                        val = [val]
                    out_fields = [f"O-{i}" for i in range(len(val))]
                else:
                    out_fields = [f"out"]
            else:
                out_fields = [f"out"]
        else:
            out_fields = op_ret_type_fields
        if name == "slice":
            # remove slice if is identity
            if len(args) == 4 and len(kwargs) == 0:
                start = args[2]
                end = args[3]
                # 9223372036854775807 is max int64
                if end == 9223372036854775807:
                    end = None
                if (start == 0 or start is None) and end is None:
                    return args[0]
        op, output_desps, meta = self.create_op_node(name, list(inp_handles.keys()),
                                               out_fields, target, args,
                                               kwargs, raw_op_name, additional_args)
        if schema is not None:
            meta.op_sig = str(schema)
        op.style = {}
        if op_has_buffer:
            op.style["backgroundColor"] = "azure"
        elif op_has_param:
            op.style["backgroundColor"] = "beige"
        else:
            op.style["backgroundColor"] = "silver"
        c_list = self._builder.call_op_node(op, inp_handles)
        if output_desps is not None:
            assert len(output_desps) == len(c_list), "TODO"
            for i, od in enumerate(output_desps):
                c_list[i].userdata = EdgeTensorMeta(raw=od)
        if op_ret_type_fields is not None:
            if self._is_export:
                return c_list
            nt = namedtuple(name, op_ret_type_fields)
            return nt(*c_list)
        if schema_returns is not None and isinstance(schema_returns[0].type, torch.ListType):
            return c_list
        return c_list[0]

    def _resolve_wrong_module_id(self, module_scope: str):
        parts = module_scope.split(".")
        i = 0
        failed = False
        real_parts: List[str] = []
        while i < len(parts):
            part = parts[i]
            if part.startswith("slice("):
                try:
                    slice_obj = eval(part)
                except:
                    LOGGER.error("eval slice in module id failed. %s %s", module_scope, part)
                    failed = True 
                    break
                parent = ".".join(parts[:i])
                assert self._original_mod is not None
                submod = self._original_mod.get_submodule(parent) if parent else self._original_mod
                next_two_part = parts[i + 1:i + 3]
                if len(next_two_part) == 2 and next_two_part[0] == "_modules":
                    try:
                        assert isinstance(submod, torch.nn.ModuleList)
                        real_idx = int(next_two_part[1])
                        real_mod = submod[slice_obj][real_idx] # type: ignore
                        found = False
                        for j, mod in enumerate(submod):
                            if mod is real_mod:
                                real_parts.append(str(j))
                                found = True
                                break
                        if not found:
                            LOGGER.error("resolve slice in module id failed. %s %s", module_scope, part)
                            failed = True
                            break
                        i += 3
                    except:
                        LOGGER.error("resolve slice in module id failed. %s", module_scope)
                        failed = True 
                        break
                else:
                    LOGGER.error("resolve slice in module id failed. %s", module_scope)
                    failed = True
                    break
            else:
                real_parts.append(part)
                i += 1
        if not failed:
            LOGGER.info("resolve slice in module id. %s -> %s", module_scope, ".".join(real_parts))
            module_scope = ".".join(real_parts)
        return module_scope, failed

    def create_op_node(self,
                       name: str,
                       inputs: List[Optional[str]],
                       outputs: List[Optional[str]],
                       target: Any,
                       args: Tuple,
                       kwargs: dict,
                       raw_op_name: Optional[str] = None,
                       addi_args: Optional[Dict[str, Any]] = None,
                       module_stack_fx: Optional[List[Tuple[str, str]]] = None):
        if name == "linear":
            w = args[1]
            name = f"Linear {w.shape[1]}x{w.shape[0]}"
        elif name.startswith("conv"):
            w = args[1]
            ndim = w.ndim - 2
            name = f"Conv{ndim}d {w.shape[1]}x{w.shape[0]}"
        elif name == "view":
            shape_str = ",".join(map(str, args[1]))
            name = f"view|{shape_str}"
        elif name == "transpose":
            shape_str = ",".join(map(str, args[1:]))
            name = f"transpose|{shape_str}"
        elif name == "permute":
            shape_str = ",".join(map(str, args[1]))
            name = f"permute|{shape_str}"
        elif name in ["add", "sub", "mul", "div"]:
            if not isinstance(args[1],
                              (flowui.SymbolicImmediate, torch.Tensor)):
                name = f"{name}|{args[1]}"
        elif name == "slice":
            if len(args) >= 4:
                dim = args[1]
                start = args[2]
                end = args[3]
                # 9223372036854775807 is max int64
                if end == 9223372036854775807:
                    end = ""
                name = f"slice({dim})[{start}:{end}]"
        elif name == "unsqueeze":
            if len(args) == 2:
                dim = args[1]
                name = f"unsqueeze({dim})"
        elif name == "split":
            if len(args) == 3:
                size = args[1]
                dim = args[2]
                name = f"split({dim}, {size})"
        elif name == "getitem":
            if len(args) == 2:
                idx = args[1]
                name = f"getitem[{idx}]"
        elif name == "_to_copy":
            to_str_parts = []
            if "dtype" in kwargs:
                to_str_parts.append(f"{kwargs['dtype']}")
            if "device" in kwargs:
                to_str_parts.append(f"{kwargs['device']}")
            if "layout" in kwargs:
                to_str_parts.append(f"{kwargs['layout']}")
            if to_str_parts:
                name = f"to({', '.join(to_str_parts)})"

        node = get_node_context_noexcept()
        # attach node meta datas
        # nn_module_stack available for both fx and export.
        module_scope_uid: Optional[UniqueTreeIdForTree] = None
        module_stack: Optional[List[Tuple[str, str]]] = None
        module_qname: Optional[str] = None
        if module_stack_fx is not None:
            module_stack = module_stack_fx
            module_scope = module_stack[-1][0]
            module_scope_uid = UniqueTreeIdForTree.from_parts(
                module_scope.split("."))
            module_qname = module_stack[-1][1]
        if "nn_module_stack" in node.meta and len(
                node.meta["nn_module_stack"]) > 0:
            nn_module_stack = node.meta["nn_module_stack"]
            # 'nn_module_stack': {
            #     'L__self__': ('', 'torchvision.models.resnet.ResNet'),
            #     'L__self___layer3': ('layer3', 'torch.nn.modules.container.Sequential'),
            #     'L__self___layer3_1': ('layer3.1', 'torchvision.models.resnet.BasicBlock'),
            #     'getattr_L__self___layer3___1___bn1': ('layer3.1.bn1', 'torch.nn.modules.batchnorm.BatchNorm2d')
            # },

            # TODO known issue: if top-level node contains any decorator or is nn.Sequential, 
            # nn_module_stack will contains wrong module id.
            module_stack = [v for v in nn_module_stack.values()]
            for i in range(len(module_stack)):
                module_scope_item = module_stack[i][0]
                if "slice(" in module_scope_item:
                    module_scope_item, failed = self._resolve_wrong_module_id(module_scope_item)
                    if failed:
                        break
                    module_stack[i] = (module_scope_item, module_stack[i][1])
            module_scope = module_stack[-1][0]
            module_scope_uid = UniqueTreeIdForTree.from_parts(
                module_scope.split("."))
            module_qname = list(nn_module_stack.values())[-1][1]
        output_desps: Optional[Sequence[Any]] = None
        if "val" in node.meta:
            val = node.meta["val"]
            if not isinstance(val, (tuple, list)):
                # list of faketensor or symint
                val = [val]
            assert len(val) == len(outputs), f"TODO {val}"
            output_desps = val
        if raw_op_name is None:
            raw_op_name = name
        meta = PytorchNodeMeta(raw_op_name, module_scope_uid, module_stack,
                               module_qname, output_desps, additional_args=addi_args)
        if "stack_trace" in node.meta:
            meta.stack_trace = node.meta["stack_trace"]
        sym_node = self._builder.create_op_node(name,
                                                inputs,
                                                outputs,
                                                node_data=meta)
        return sym_node, output_desps, meta

    def _get_inp_handles_and_addi_args(self, name: str, args, kwargs, schema=None):
        kwargs_merged = {}
        if schema is not None:
            for i, arg in enumerate(args):
                if i < len(schema.arguments):
                    kwargs_merged[f"I-{schema.arguments[i].name}"] = arg
                else:
                    kwargs_merged[f"I-{i}"] = arg
        else:
            for i, arg in enumerate(args):
                kwargs_merged[f"I-{i}"] = arg
        kwargs_merged.update(kwargs)
        inp_handles = {}
        additional_args = {}
        for k, v in kwargs_merged.items():
            if isinstance(v, flowui.SymbolicImmediate):
                inp_handles[k] = v
            elif isinstance(v, list) and len(v) > 0:
                # for cat
                if isinstance(v[0], flowui.SymbolicImmediate):
                    inp_handles[k] = v
                else:
                    additional_args[k] = v
            else:
                additional_args[k] = v
        return inp_handles, additional_args

    def call_method(self, target: Any, args: Tuple, kwargs: dict) -> Any:
        if isinstance(target, torch.nn.Module):
            name = type(target).__name__
        else:
            name = str(target)
        msg = f"call_method {target} {name}"
        if self._verbose:
            print(msg)
        inp_handles, additional_args = self._get_inp_handles_and_addi_args(
            name, args, kwargs)
        if not inp_handles:
            return super().call_function(target, args, kwargs)
        op, output_desps, _ = self.create_op_node(name, list(inp_handles.keys()),
                                               [f"{name}-out"], target, args,
                                               kwargs)

        op.style = {"backgroundColor": "green"}

        c_list = self._builder.call_op_node(op, inp_handles)
        return c_list[0]

    def run_on_graph_placeholders(self):
        placeholders = self.graph.find_nodes(op="placeholder")
        assert isinstance(placeholders,
                          list), f"placeholders {placeholders} must be list"
        inputs = []
        for arg in placeholders:
            if arg.name in self._export_param_dict:
                inp = self._export_param_dict[arg.name]
            else:
                if self._is_export:
                    inp_meta = PytorchNodeMeta("placeholder", is_io_node=True)
                    inp, inp_node = self._builder.create_input(arg.name, node_data=inp_meta, default_input_handle=arg.name)
                    if "val" in arg.meta:
                        assert not isinstance(arg.meta["val"],
                                              (tuple, list)), f"TODO {arg.meta['val']}"
                        inp.userdata = EdgeTensorMeta(raw=arg.meta["val"])
                        inp_meta.output_desps = [arg.meta["val"]]
                else:
                    inp, inp_node = self._builder.create_input(arg.name, default_input_handle=arg.name)
                inp.source_handle = arg.name
            inputs.append(inp)
        graph_ctx = GraphContext()
        with enter_graph_context(graph_ctx):
            res = self.run(*inputs)
        if self._is_export:
            # remove all BUFFER_MUTATION in outputs
            assert isinstance(res, tuple)
            res_list = list(res)
            new_res_list = []
            for i, r in enumerate(res_list):
                # may be tensor here (BUFFER)
                if isinstance(r, flowui.SymbolicImmediate):
                    keep = self._export_buffer_mu_keep_flags[i]
                    if keep:
                        new_res_list.append(r)
            return tuple(new_res_list)
        else:
            return res
