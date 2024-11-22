from collections import namedtuple
import contextlib
import dataclasses
from operator import getitem
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union
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

NODE_CONTEXT: contextvars.ContextVar[Optional[torch.fx.Node]] = contextvars.ContextVar("PTH_FX_NODE_CTX", default=None)
GRAPH_CONTEXT: contextvars.ContextVar[Optional[GraphContext]] = contextvars.ContextVar("PTH_FX_GRAPH_CTX", default=None)

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
    module_id: Optional[UniqueTreeIdForTree] = None

class PytorchExportBuilder(flowui.SymbolicFlowBuilder[PytorchNodeMeta]):
    pass 

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
                        self._export_param_dict[p.arg.name] = original_mod.get_parameter(target)
                elif p.kind == InputKind.BUFFER:
                    target = p.target
                    if target is not None:
                        self._export_param_dict[p.arg.name] = original_mod.get_buffer(target)

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
        inp_handles, additional_args = self._get_inp_handles_and_addi_args(args, kwargs)
        if not inp_handles:
            return super().call_function(mod, args, kwargs)
            
        op = self.create_op_node(name, list(inp_handles.keys()), [f"{name}-out"], target, args, kwargs)
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
                    op_ret_type_fields = [f"out{i}" for i in range(num_output)]
            elif target is getitem:
                # come from split
                if not isinstance(args[0], flowui.SymbolicImmediate):
                    return super().call_function(target, args, kwargs)
        msg = f"call_function {type(target)} {target} {name} "
        if self._verbose:
            print(msg)

        inp_handles, additional_args = self._get_inp_handles_and_addi_args(args, kwargs)
        if not inp_handles:
            return super().call_function(target, args, kwargs)
        if op_ret_type_fields is None:
            out_fields = [f"{name}-out"]
        else:
            out_fields = op_ret_type_fields
        if name in _ATEN_NAME_MAP:
            name = _ATEN_NAME_MAP[name]
        if name.startswith("aten::"):
            # remove aten:: prefix
            name = name[6:]
        if name == "_to_copy":
            return args[0]
        op = self.create_op_node(name, list(inp_handles.keys()), out_fields, target, args, kwargs)
        op.style = {"backgroundColor": "aliceblue" if op_has_param else "silver"}

        c_list = self._builder.call_op_node(op, inp_handles)
        if op_ret_type_fields is not None:
            if self._is_export:
                return c_list
            nt = namedtuple(name, op_ret_type_fields)
            return nt(*c_list)
        return c_list[0]

    def create_op_node(self, name: str, inputs: List[Optional[str]], outputs: List[Optional[str]], target: Any, args: Tuple, kwargs: dict):
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
            if not isinstance(args[1], (flowui.SymbolicImmediate, torch.Tensor)):
                name = f"{name}|{args[1]}"
        node = get_node_context_noexcept()
        # attach node meta datas
        # nn_module_stack available for both fx and export.
        module_scope_uid: Optional[UniqueTreeIdForTree] = None 
        if "nn_module_stack" in node.meta and len(node.meta["nn_module_stack"]) > 0:
            nn_module_stack = node.meta["nn_module_stack"]
            # 'nn_module_stack': {
            #     'L__self__': ('', 'torchvision.models.resnet.ResNet'),
            #     'L__self___layer3': ('layer3', 'torch.nn.modules.container.Sequential'),
            #     'L__self___layer3_1': ('layer3.1', 'torchvision.models.resnet.BasicBlock'),
            #     'getattr_L__self___layer3___1___bn1': ('layer3.1.bn1', 'torch.nn.modules.batchnorm.BatchNorm2d')
            # },
            module_scope = list(nn_module_stack.values())[-1][0]
            module_scope_uid = UniqueTreeIdForTree.from_parts(module_scope.split("."))
        sym_node = self._builder.create_op_node(name, inputs, outputs, node_data=PytorchNodeMeta(module_id=module_scope_uid)) 
        return sym_node

    def _get_inp_handles_and_addi_args(self, args, kwargs):
        kwargs_merged = {}
        for i, arg in enumerate(args):
            kwargs_merged[str(i)] = arg
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
        inp_handles, additional_args = self._get_inp_handles_and_addi_args(args, kwargs)
        if not inp_handles:
            return super().call_function(target, args, kwargs)
        op = self.create_op_node(name, list(inp_handles.keys()), [f"{name}-out"], target, args, kwargs)

        op.style = {"backgroundColor": "green"}

        c_list = self._builder.call_op_node(op, inp_handles)
        return c_list[0]

    def run_on_graph_placeholders(self):
        placeholders = self.graph.find_nodes(op="placeholder")
        assert isinstance(placeholders, list), f"placeholders {placeholders} must be list"
        inputs = []
        for arg in placeholders:
            if arg.name in self._export_param_dict:
                inp = self._export_param_dict[arg.name]
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

    r18 = resnet18()
    # gm = torch.fx.symbolic_trace(m)
    # gm = torch.export.export(m, (torch.rand(8, 4),))
    with torch.device("meta"):
        gm = torch.export.export(r18.to("meta"), (torch.rand(1, 3, 224, 224),))
    import rich
    for node in gm.graph.nodes:
        rich.print(node.name, node.op, node.meta)

    rich.print(gm.module())
    # print(gm.graph_module)
    # print(gm.graph_module)
    builder = PytorchExportBuilder()
    interpreter = FlowUIInterpreter(gm, builder, original_mod=r18, verbose=True)
    # inp, inp_node = builder.create_input("inp")
    # outputs = interpreter.run(inp)
    outputs = interpreter.run_on_graph_placeholders()
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
    interpreter = FlowUIInterpreter(gm, builder, original_mod=r18, verbose=True)
    # inp, inp_node = builder.create_input("inp")
    # outputs = interpreter.run(inp)
    outputs = interpreter.run_on_graph_placeholders()
    res = [outputs]
    rich.print(outputs)
    # graph_res = builder.build_detached_flow(outputs)

if __name__ == "__main__":
    _main()
