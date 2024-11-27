from collections import namedtuple
import contextlib
from curses import meta
import dataclasses
from operator import getitem
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from cv2 import merge
from regex import F
import rich
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
import contextvars

import torch
import torch.fx
from torch.return_types import all_return_types


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
class PytorchNodeMeta:
    op: str
    module_id: Optional[UniqueTreeIdForTree] = None
    # (module_id, module_qname)
    module_stack: Optional[List[Tuple[str, str]]] = None
    module_qname: Optional[str] = None
    output_desps: Optional[Sequence[Any]] = None
    is_merged: bool = False
    ftree_id: Optional[str] = None
    is_io_node: bool = False

    params: Optional[Dict[str, torch.Tensor]] = None
    buffers: Optional[Dict[str, torch.Tensor]] = None


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
class EdgeTensorMeta:
    raw: Any  # FakeTensor or SymInt

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

@dataclasses.dataclass
class PytorchFlowOutput(flowui.SymbolicGraphOutput[PytorchNodeMeta,
                                                   EdgeTensorMeta]):
    ftree: Optional[FunctionalFlowTree] = None

    def create_graph_with_expanded_ids(
        self,
        expanded_ids: List[str],
        expanded_id_is_module: bool = False,
        module: Optional[torch.nn.Module] = None,
        submodule_id: Optional[str] = None,
        submodule_id_is_module: bool = True,
    ) -> PytorchFlowOutputPartial:
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
                merged_node = flowui.Node(id=f"M-{cur['module']}-{qname}",
                                          data=flowui.NodeData(label=cls_name))
                merged_node.style = {"backgroundColor": "lightgray"}
                nodes_to_merge = self.ftree.all_node_ids_with_stack[
                    cur["start"]:cur["end"]]
                merge_list.append((merged_node, nodes_to_merge))
                merged_node_data.append(
                    PytorchNodeMeta(op=qname,
                                    is_merged=True,
                                    module_id=UniqueTreeIdForTree.from_parts(cur["module"].split(".")),
                                    ftree_id=cur["id"],
                                    module_qname=qname))
                if cur["start"] >= submod_node_range[0] and cur["end"] <= submod_node_range[1]:
                    submod_merge_nodes.append(merged_node)
        
        # merge nodes
        nodes = self.nodes
        edges = self.edges
        internals = flowui.FlowInternals()
        internals.set_from_nodes_edges(nodes, edges)
        # internals, new_edge_id_to_prev_edges = internals.merge_nodes(
        #     merge_list)
        internals, _, prev_node_id_to_data, prev_edge_id_to_data = internals.merge_nodes_with_data(
            merge_list, merged_node_data, self.node_id_to_data, self.edge_id_to_data)
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
                if n.id in prev_node_id_to_data:
                    new_data = PytorchNodeMeta("placeholder", is_io_node=True)
                    if edges[0].id in prev_edge_id_to_data:
                        new_data.output_desps = [prev_edge_id_to_data[edges[0].id].raw]
                    prev_node_id_to_data[n.id] = new_data
        
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

    def __init__(self):
        super().__init__()
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
            out_immedinates: Sequence[flowui.SymbolicImmediate],
            disable_handle: bool = False):
        out_datas: List[PytorchNodeMeta] = []
        for imme in out_immedinates:
            meta = PytorchNodeMeta("placeholder", is_io_node=True)
            if imme.userdata is not None:
                # should be EdgeTensorMeta
                assert isinstance(imme.userdata, EdgeTensorMeta)
                meta.output_desps = [imme.userdata.raw]
            out_datas.append(meta)
        graph_res = self.build_detached_flow(out_immedinates, disable_handle, out_datas)
        if self._ftree_cache is None:
            self._ftree_cache = self._build_tree_from_module_stack(
                self._id_to_node_data)
        return PytorchFlowOutput(graph_res.nodes, graph_res.edges,
                                 graph_res.node_type_map,
                                 graph_res.node_id_to_data,
                                 graph_res.edge_id_to_data, self._ftree_cache)


class FlowUIInterpreter(Interpreter):

    def __init__(self,
                 gm: Union["GraphModule", ExportedProgram],
                 builder: PytorchExportBuilder,
                 original_mod: Optional[torch.nn.Module] = None,
                 verbose: bool = False):

        if isinstance(gm, ExportedProgram):
            assert original_mod is not None
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

            for p in gm.graph_signature.output_specs:
                if p.kind == OutputKind.BUFFER_MUTATION:
                    self._export_buffer_mu_keep_flags.append(False)
                else:
                    self._export_buffer_mu_keep_flags.append(True)
        else:
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
            args, kwargs)
        if not inp_handles:
            return super().call_function(mod, args, kwargs)

        op, output_desps = self.create_op_node(name, list(inp_handles.keys()),
                                               [f"{name}-out"], target, args,
                                               kwargs)
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
        if self._is_export:
            qname = get_qualname_of_type(type(target))
            if qname.startswith("torch"):
                for arg in args:
                    if isinstance(arg, torch.nn.Parameter):
                        op_has_param = True
                        break
                # analysis aten ops schema to get number of outputs
                schema = target._schema
                name = schema.name
                num_output = len(schema.returns)
                if num_output > 1:
                    op_ret_type_fields = [f"out-{i}" for i in range(num_output)]
            elif target is getitem:
                # come from split
                if not isinstance(args[0], flowui.SymbolicImmediate):
                    return super().call_function(target, args, kwargs)
        msg = f"call_function {type(target)} {target} {name} "
        if self._verbose:
            print(msg)

        inp_handles, additional_args = self._get_inp_handles_and_addi_args(
            args, kwargs)
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
                    out_fields = [f"out-{i}" for i in range(len(val))]
                else:
                    out_fields = [f"out"]
            else:
                out_fields = [f"out"]
        else:
            out_fields = op_ret_type_fields
        raw_op_name = name
        if name in _ATEN_NAME_MAP:
            name = _ATEN_NAME_MAP[name]
        if name.startswith("aten::"):
            # remove aten:: prefix
            name = name[6:]
        if name == "_to_copy":
            return args[0]
        op, output_desps = self.create_op_node(name, list(inp_handles.keys()),
                                               out_fields, target, args,
                                               kwargs, raw_op_name)
        op.style = {
            "backgroundColor": "aliceblue" if op_has_param else "silver"
        }

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
        return c_list[0]

    def create_op_node(self,
                       name: str,
                       inputs: List[Optional[str]],
                       outputs: List[Optional[str]],
                       target: Any,
                       args: Tuple,
                       kwargs: dict,
                       raw_op_name: Optional[str] = None):
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
        node = get_node_context_noexcept()
        # attach node meta datas
        # nn_module_stack available for both fx and export.
        module_scope_uid: Optional[UniqueTreeIdForTree] = None
        module_stack: Optional[List[Tuple[str, str]]] = None
        module_qname: Optional[str] = None
        if "nn_module_stack" in node.meta and len(
                node.meta["nn_module_stack"]) > 0:
            nn_module_stack = node.meta["nn_module_stack"]
            # 'nn_module_stack': {
            #     'L__self__': ('', 'torchvision.models.resnet.ResNet'),
            #     'L__self___layer3': ('layer3', 'torch.nn.modules.container.Sequential'),
            #     'L__self___layer3_1': ('layer3.1', 'torchvision.models.resnet.BasicBlock'),
            #     'getattr_L__self___layer3___1___bn1': ('layer3.1.bn1', 'torch.nn.modules.batchnorm.BatchNorm2d')
            # },
            module_scope = list(nn_module_stack.values())[-1][0]
            module_stack = [v for v in nn_module_stack.values()]
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
        assert raw_op_name is not None
        meta = PytorchNodeMeta(raw_op_name, module_scope_uid, module_stack,
                               module_qname, output_desps)
        sym_node = self._builder.create_op_node(name,
                                                inputs,
                                                outputs,
                                                node_data=meta)
        return sym_node, output_desps

    def _get_inp_handles_and_addi_args(self, args, kwargs):
        kwargs_merged = {}
        for i, arg in enumerate(args):
            kwargs_merged[f"inp-{i}"] = arg
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
            args, kwargs)
        if not inp_handles:
            return super().call_function(target, args, kwargs)
        op, output_desps = self.create_op_node(name, list(inp_handles.keys()),
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
                    inp, inp_node = self._builder.create_input(arg.name, node_data=inp_meta)
                    if "val" in arg.meta:
                        assert not isinstance(arg.meta["val"],
                                              (tuple, list)), f"TODO {arg.meta['val']}"
                        inp.userdata = EdgeTensorMeta(raw=arg.meta["val"])
                        inp_meta.output_desps = [arg.meta["val"]]
                else:
                    inp, inp_node = self._builder.create_input(arg.name)

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


def _main():
    from torchvision.models import resnet18
    from torch.nn import functional as F

    class TestNoForwardMod(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Conv2d(10, 10, 1, 1)

        def forward(self, x):
            return F.relu(self.linear(x))

        def func(self, x):
            return F.gelu(x)

    class TestMod(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 64, 7)
            self.bn = torch.nn.BatchNorm2d(64)
            self.relu = torch.nn.ReLU()
            self.pool = torch.nn.MaxPool2d(2)
            self.fc = torch.nn.Conv2d(64, 10, 1, 1)
            self.mod = TestNoForwardMod()

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = F.gelu(x)
            x = self.pool(x)
            # x = torch.flatten(x, 1)
            x = self.fc(x)
            return self.mod(x)

    class TestModX(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.mod = TestMod()

        def forward(self, x):
            return self.mod(x)

    r18 = resnet18()
    # r18 = TestModX()
    # gm = torch.fx.symbolic_trace(m)
    # gm = torch.export.export(m, (torch.rand(8, 4),))
    with torch.device("meta"):
        gm = torch.export.export(r18.to("meta"),
                                 (torch.rand(1, 3, 224, 224), ))
    import rich
    # for node in gm.graph.nodes:
    #     rich.print(node.name, node.op, node.meta)
    # rich.print(gm.graph_module)
    # return
    # rich.print(gm.module())
    # print(gm.graph_module)
    # print(gm.graph_module)
    builder = PytorchExportBuilder()
    interpreter = FlowUIInterpreter(gm,
                                    builder,
                                    original_mod=r18,
                                    verbose=False)
    # inp, inp_node = builder.create_input("inp")
    # outputs = interpreter.run(inp)
    outputs = interpreter.run_on_graph_placeholders()
    ftree = builder._build_tree_from_module_stack(builder._id_to_node_data)
    pth_flow = builder.build_pytorch_detached_flow(r18, outputs, disable_handle=False)
    pth_flow.create_graph_with_expanded_modules(["layer3.1"], r18)
    # rich.print(ftree.root)
    # rich.print(ftree.all_node_ids_with_stack)
    return

    res = [outputs]
    rich.print(outputs)
    graph_res = builder.build_detached_flow(outputs)


def _main_fx():
    from torchvision.models import resnet18

    r18 = resnet18()
    gm = torch.fx.symbolic_trace(r18)
    # gm = torch.export.export(m, (torch.rand(8, 4),))
    import rich
    for node in gm.graph.nodes:
        rich.print(node.meta)
    # print(gm.graph_module)
    # print(gm.graph_module)
    builder = PytorchExportBuilder()
    interpreter = FlowUIInterpreter(gm,
                                    builder,
                                    original_mod=r18,
                                    verbose=True)
    # inp, inp_node = builder.create_input("inp")
    # outputs = interpreter.run(inp)
    outputs = interpreter.run_on_graph_placeholders()
    res = [outputs]
    rich.print(outputs)
    # graph_res = builder.build_detached_flow(outputs)


if __name__ == "__main__":
    _main()
