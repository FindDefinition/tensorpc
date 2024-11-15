from tensorpc.flow import flowui
from tensorpc.flow.jsonlike import as_dict_no_undefined
import rich 
def _main():
    symbolic_flow_bd = flowui.SymbolicFlowBuilder()
    a, a_node = symbolic_flow_bd.create_input("a")
    b, b_node = symbolic_flow_bd.create_input("b")

    add_op = symbolic_flow_bd.create_op_node("Add", ["a", "b"], ["c"])
    c_list = symbolic_flow_bd.call_op_node(add_op, {"a": a, "b": b})
    nodes, edges = symbolic_flow_bd.build_detached_flow(c_list)

    
    rich.print(as_dict_no_undefined(nodes), as_dict_no_undefined(edges))

def _main2():
    symbolic_flow_bd = flowui.SymbolicFlowBuilder()
    a, a_node = symbolic_flow_bd.create_input("a")
    b, b_node = symbolic_flow_bd.create_input("b")

    add_op = symbolic_flow_bd.create_op_node("Add", [None], [None])
    c_list = symbolic_flow_bd.call_op_node(add_op, {None: [a, b]})
    nodes, edges = symbolic_flow_bd.build_detached_flow(c_list)

    
    rich.print(as_dict_no_undefined(nodes), as_dict_no_undefined(edges))

if __name__ == "__main__":
    _main()