import enum
from pathlib import Path
import rich
from tensorpc.constants import PACKAGE_ROOT
from tensorpc.dock import flowui

from typing import Callable, Coroutine, Literal, Optional, Any
from tensorpc.apps.adv.model import ADVEdgeModel, ADVHandlePrefix, ADVNewNodeConfig, ADVNodeHandle, ADVNodeType, ADVRoot, ADVProject, ADVNodeModel, ADVFlowModel, InlineCode

def _get_simple_flow(name: str, op: Literal["+", "-", "*", "/"], sym_import_path: list[str]):
    fragment = f"""
ADV.mark_outputs("c")
return a {op} b
    """
    if op == "+":
        op_name = "add"
    elif op == "-":
        op_name = "sub"
    elif op == "*":
        op_name = "mul"
    elif op == "/":
        op_name = "div"
    else:
        raise ValueError(f"Unsupported op: {op}")
    return ADVFlowModel(nodes={
        "sym_def": ADVNodeModel(
            id="sym_def", 
            nType=ADVNodeType.SYMBOLS,
            position=flowui.XYPosition(0, 0), 
            ref_node_id="sym_def",
            ref_import_path=sym_import_path,
        ),
        "func": ADVNodeModel(
            id="func", 
            nType=ADVNodeType.FRAGMENT,
            position=flowui.XYPosition(200, 0), 
            name=f"{name}_func",
            inlinesf_name=name,
            impl=InlineCode(fragment),
        ),
        "o0": ADVNodeModel(
            id="o0", 
            nType=ADVNodeType.OUT_INDICATOR,
            position=flowui.XYPosition(400, 0), 
            name="",
        ),
        }, edges={
            "e0": ADVEdgeModel(
                id="e0", 
                source="func",
                sourceHandle=f"{ADVHandlePrefix.Output}-c",
                target="o0",
                targetHandle=f"{ADVHandlePrefix.OutIndicator}-outputs",
                isAutoEdge=False,
            )
        })


def get_simple_nested_model():
    global_script_0 = f"""
import numpy as np 
    """
    symbolgroup0 = f"""
@dataclasses.dataclass
class SymbolGroup0:
    a: int 
    b: float
    c: float
    d: int
    """
    symbolgroupNested = f"""
@dataclasses.dataclass
class SymbolGroupNested:
    a: int 
    b: float
    c: float
    d: int
    """

    fragment_add = f"""
ADV.mark_outputs("c")
return a + b
    """
    fragment1 = f"""
ADV.mark_outputs("d->D")
return c + a
    """

    nested_model_symbol_lib = ADVNodeModel(
        id="sym_lib", 
        nType=ADVNodeType.FRAGMENT,
        position=flowui.XYPosition(0, 200), 
        name="sym_lib",
        flow=ADVFlowModel(nodes={
            "sym_def": ADVNodeModel(
                id="sym_def", 
                nType=ADVNodeType.SYMBOLS,
                position=flowui.XYPosition(0, 0), 
                name="SymbolGroup0",
                impl=InlineCode(symbolgroup0),
            ),
        }, edges={})
    )
    nested_model_op_lib = ADVNodeModel(
        id="op_lib", 
        nType=ADVNodeType.FRAGMENT,
        position=flowui.XYPosition(600, 600), 
        name="op_lib",
        flow=ADVFlowModel(nodes={
            "sym_lib": nested_model_symbol_lib,
            "add": ADVNodeModel(
                id="add", 
                nType=ADVNodeType.FRAGMENT,
                position=flowui.XYPosition(0, 0), 
                name="add",
                flow=_get_simple_flow("add", "+", ["op_lib", "sym_lib"]),
            ),
            "sub": ADVNodeModel(
                id="sub", 
                nType=ADVNodeType.FRAGMENT,
                position=flowui.XYPosition(200, 0), 
                name="sub",
                flow=_get_simple_flow("sub", "-", ["op_lib", "sym_lib"]),
            ),
            "div": ADVNodeModel(
                id="div", 
                nType=ADVNodeType.FRAGMENT,
                position=flowui.XYPosition(400, 0), 
                name="div",
                flow=_get_simple_flow("div", "/", ["op_lib", "sym_lib"]),
            ),
            "mul": ADVNodeModel(
                id="mul", 
                nType=ADVNodeType.FRAGMENT,
                position=flowui.XYPosition(600, 0), 
                name="mul",
                flow=_get_simple_flow("mul", "*", ["op_lib", "sym_lib"]),
            ),
        }, edges={})
    )

    nested_model = ADVNodeModel(
        id="nf1", 
        nType=ADVNodeType.FRAGMENT,
        position=flowui.XYPosition(600, 100), 
        name="nested0",
        flow=ADVFlowModel(nodes={
            "s1": ADVNodeModel(
                id="s1", 
                nType=ADVNodeType.SYMBOLS,
                position=flowui.XYPosition(0, 0), 
                name="SymbolGroupNested",
                impl=InlineCode(symbolgroupNested),
            ),
            "mul": ADVNodeModel(
                id="mul", 
                nType=ADVNodeType.FRAGMENT,
                position=flowui.XYPosition(200, 0), 
                name="mul_func",
                inlinesf_name="nested0",
                ref_node_id="func",
                ref_import_path=["op_lib", "mul"],
            ),
            "oic0": ADVNodeModel(
                id="oic0", 
                nType=ADVNodeType.OUT_INDICATOR,
                position=flowui.XYPosition(400, 100), 
                name="",
            ),

        }, edges={
            "ea": ADVEdgeModel(
                id="ea", 
                source="s1",
                sourceHandle=f"{ADVHandlePrefix.Output}-a",
                target="mul",
                targetHandle=f"{ADVHandlePrefix.Input}-a",
                isAutoEdge=False,
            ),
            "eb": ADVEdgeModel(
                id="eb", 
                source="s1",
                sourceHandle=f"{ADVHandlePrefix.Output}-b",
                target="mul",
                targetHandle=f"{ADVHandlePrefix.Input}-b",
                isAutoEdge=False,
            ),

            "eo": ADVEdgeModel(
                id="eo", 
                source="mul",
                sourceHandle=f"{ADVHandlePrefix.Output}-c",
                target="oic0",
                targetHandle=f"{ADVHandlePrefix.OutIndicator}-outputs",
                isAutoEdge=False,
            )

        })
    )
    # return ADVProject(
    #     flow=nested_model.flow,
    #     import_prefix="tensorpc.adv.test_project",
    #     path=str(PACKAGE_ROOT / "adv" / "test_project"),

    # )
    res_proj = ADVProject(
        flow=ADVFlowModel(nodes={
            "op_lib": nested_model_op_lib,
            "g1": ADVNodeModel(
                id="g1", 
                nType=ADVNodeType.GLOBAL_SCRIPT,
                position=flowui.XYPosition(0, 200), 
                name="GlobalScript0",
                impl=InlineCode(global_script_0),
            ),

            "n1": ADVNodeModel(
                id="n1", 
                nType=ADVNodeType.SYMBOLS,
                position=flowui.XYPosition(0, 0), 
                name="SymbolGroup0",
                impl=InlineCode(symbolgroup0),
            ),
            "add": ADVNodeModel(
                id="add", 
                nType=ADVNodeType.FRAGMENT,
                position=flowui.XYPosition(200, 0), 
                name="add_func",
                inlinesf_name="inline0",
                impl=InlineCode(fragment_add),
            ),
            "f1": ADVNodeModel(
                id="f1", 
                nType=ADVNodeType.FRAGMENT,
                position=flowui.XYPosition(400, 100), 
                name="add_func2",
                inlinesf_name="inline0",
                impl=InlineCode(fragment1),
            ),
            "add-ref": ADVNodeModel(
                id="add-ref", 
                nType=ADVNodeType.FRAGMENT,
                position=flowui.XYPosition(200, 200), 
                name="add_func",
                inlinesf_name="inline0",
                ref_node_id="add",
                ref_import_path=[],
                alias_map="c->C", 
            ),
            "oic0": ADVNodeModel(
                id="oic0", 
                nType=ADVNodeType.OUT_INDICATOR,
                position=flowui.XYPosition(800, 200), 
                name="OutputAlias",
            ),
            "nf1": nested_model,
            "nf1-mul-ref": ADVNodeModel(
                id="nf1-mul-ref", 
                nType=ADVNodeType.FRAGMENT,
                position=flowui.XYPosition(800, 400), 
                name="fn_nested",
                # inlinesf_name="inline0",
                ref_node_id="mul",
                ref_import_path=["op_lib"],
                # alias_map="c->C", 
            ),
            # "sub-ref": ADVNodeModel(
            #     id="nf2-sub-ref", 
            #     nType=ADVNodeType.FRAGMENT,
            #     position=flowui.XYPosition(800, 600), 
            #     name="fn_nested",
            #     # inlinesf_name="inline0",
            #     ref_node_id="sub",
            #     ref_import_path=["nested0", "nested1"],
            #     # alias_map="c->C", 
            # ),

        }, edges={
            # "e0": ADVEdgeModel(
            #     id="e0", 
            #     source="f1",
            #     sourceHandle=f"{ADVHandlePrefix.Output}-d",
            #     target="oic0",
            #     targetHandle=f"{ADVHandlePrefix.OutIndicator}-outputs",
            #     isAutoEdge=False,
            # )
            "e-add-ref-a": ADVEdgeModel(
                id="e-add-ref-a", 
                source="n1",
                sourceHandle=f"{ADVHandlePrefix.Output}-a",
                target="add-ref",
                targetHandle=f"{ADVHandlePrefix.Input}-a",
                isAutoEdge=False,
            )

        }),

        import_prefix="tensorpc.apps.adv.managed",
        path=str(PACKAGE_ROOT / "apps" / "adv" / "managed"),
    )
    ngid_to_path, ngid_to_fpath = res_proj.assign_path_to_all_node()
    res_proj.node_gid_to_path = ngid_to_path
    res_proj.node_gid_to_frontend_path = ngid_to_fpath
    res_proj.update_ref_path(ngid_to_fpath)


    return res_proj
