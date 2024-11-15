from collections import namedtuple
from typing import TYPE_CHECKING, Any, List, Tuple
from tensorpc.core.moduleid import get_qualname_of_type
from tensorpc.flow.components import flowui, mui

from torch.fx import GraphModule, Interpreter, Tracer
import torch
import torch.fx
from torch.return_types import all_return_types

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


class FlowUIInterpreter(Interpreter):

    def __init__(self,
                 gm: "GraphModule",
                 builder: flowui.SymbolicFlowBuilder,
                 verbose: bool = False):
        super().__init__(gm)
        
        self._verbose = verbose

        self._builder = builder

        self._op_name_to_th_ret_types = {}
        for rt in all_return_types:
            op_name = get_qualname_of_type(rt).split(".")[-1]
            self._op_name_to_th_ret_types[op_name] = rt
        self._torch_builtin_prefix = ".".join(
            get_qualname_of_type(type(torch.topk)).split(".")[:-1])

    def call_module(self, target: Any, args: Tuple, kwargs: dict) -> Any:
        mod = self.fetch_attr(target)
        if isinstance(mod, torch.nn.Module):
            name = type(mod).__name__
        else:
            name = str(mod)
        msg = f"call_module {target} {name} {args} {kwargs}"
        if self._verbose:
            print(msg)
        kwargs_merged = {}
        for i, arg in enumerate(args):
            kwargs_merged[str(i)] = arg
        kwargs_merged.update(kwargs)
        inp_handles = {}
        additional_args = {}
        for k, v in kwargs_merged.items():
            if isinstance(v, flowui.SymbolicImmediate):
                inp_handles[k] = v
            else:
                additional_args[k] = v
        if not inp_handles:
            return super().call_function(mod, args, kwargs)
        op = self._builder.create_op_node(name, list(inp_handles.keys()),
                                          [f"{name}-out"])
        op.style = {"backgroundColor": "aliceblue"}

        c_list = self._builder.call_op_node(op, inp_handles)
        return c_list[0]

    def call_function(self, target: Any, args: Tuple, kwargs: dict) -> Any:
        op_ret_type_fields = None
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

        # if target == "getattr"
        torch.return_types.topk
        kwargs_merged = {}
        for i, arg in enumerate(args):
            kwargs_merged[str(i)] = arg
        kwargs_merged.update(kwargs)
        inp_handles = {}
        additional_args = {}
        msg = f"call_function {type(target)} {name} {args} {kwargs}"
        if self._verbose:
            print(msg)

        for k, v in kwargs_merged.items():
            if isinstance(v, flowui.SymbolicImmediate):
                inp_handles[k] = v
            else:
                additional_args[k] = v
        if not inp_handles:
            return super().call_function(target, args, kwargs)
        if op_ret_type_fields is None:
            out_fields = [f"{name}-out"]
        else:
            out_fields = op_ret_type_fields
        op = self._builder.create_op_node(name, list(inp_handles.keys()),
                                          out_fields)
        op.style = {"backgroundColor": "silver"}

        c_list = self._builder.call_op_node(op, inp_handles)
        if op_ret_type_fields is not None:
            nt = namedtuple(name, op_ret_type_fields)
            return nt(*c_list)
        return c_list[0]

    def call_method(self, target: Any, args: Tuple, kwargs: dict) -> Any:
        if isinstance(target, torch.nn.Module):
            name = type(target).__name__
        else:
            name = str(target)
        msg = f"call_method {name} {args} {kwargs}"
        if self._verbose:
            print(msg)

        kwargs_merged = {}
        for i, arg in enumerate(args):
            kwargs_merged[str(i)] = arg
        kwargs_merged.update(kwargs)
        inp_handles = {}
        additional_args = {}
        for k, v in kwargs_merged.items():
            if isinstance(v, flowui.SymbolicImmediate):
                inp_handles[k] = v
            else:
                additional_args[k] = v
        if not inp_handles:
            return super().call_function(target, args, kwargs)
        op = self._builder.create_op_node(name, list(inp_handles.keys()),
                                          [f"{name}-out"])
        op.style = {"backgroundColor": "green"}

        c_list = self._builder.call_op_node(op, inp_handles)
        return c_list[0]


    def run_on_graph_placeholders(self):
        placeholders = self.graph.find_nodes(op="placeholder")
        assert isinstance(placeholders, list), f"placeholders {placeholders} must be list"
        inputs = []
        for arg in placeholders:
            inp, inp_node = self._builder.create_input(arg.name)
            inputs.append(inp)
        return self.run(*inputs)


def create_torch_fx_graph(
        gm: "GraphModule",
        inputs) -> Tuple[List[flowui.Node], List[flowui.Edge]]:
    builder = flowui.SymbolicFlowBuilder()

    interpreter = FlowUIInterpreter(gm, builder)
    outputs = interpreter.run(*inputs)
    print(outputs, type(outputs))


def _main():

    class MyModule(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.rand(3, 4))
            self.linear = torch.nn.Linear(4, 5)

        def forward(self, x):
            return torch.topk(
                torch.sum(self.linear(x + self.linear.weight).relu(), dim=-1),
                3).values

    m = MyModule()
    gm = torch.fx.symbolic_trace(m)
    builder = flowui.SymbolicFlowBuilder()
    interpreter = FlowUIInterpreter(gm, builder, True)
    inp, inp_node = builder.create_input("inp")
    # outputs = interpreter.run(inp)
    outputs = interpreter.run_on_graph_placeholders()

    print(outputs)


if __name__ == "__main__":
    _main()
