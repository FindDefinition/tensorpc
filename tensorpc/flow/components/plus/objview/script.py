from typing import Any
from tensorpc.flow.components import mui

def get_init_obj_convert_code():
    torch_import_stmt = ""
    # if "torch" in sys.modules:
    #     torch_import_stmt = "import torch"
    code = f"""
from tensorpc.flow import mui, three
from tensorpc.flow.components.plus.objview import (HistogramPlot, Image,
                                                   ImageBatch, LinePlot,
                                                   ScatterPlot, Tree, Unique,
                                                   Video, DataFrame, 
                                                   DataFrameTransposed,
                                                   ImageBatchChannelFirst)
import numpy as np 
{torch_import_stmt}
def convert(x):
    return x
    """
    return code


def get_frame_obj_layout_from_code(fname: str, value: str, obj: Any):
    code_comp = compile(value, fname, "exec")
    mod_globals = {}
    exec(code_comp, mod_globals)
    assert "convert" in mod_globals
    convert_func = mod_globals["convert"]
    obj_converted = convert_func(obj)
    if obj is obj_converted:
        return obj_converted, None 
    if not isinstance(obj_converted, (str, mui.Component, list)):
        return obj_converted, None 
    # we support str (markdown), official view layout (FlexBox) or list of FlexBox/str.
    # we limit the length of list to 10 to make sure user code won't crash the app.
    layouts = create_frame_obj_view_layout(obj_converted)
    if layouts is None:
        return obj_converted, None
    return obj_converted, layouts


def create_frame_obj_view_layout(obj_converted: Any):
    if not isinstance(obj_converted, list):
        obj_converted_lst = [obj_converted]
    else:
        obj_converted_lst = obj_converted
    layouts = []
    for item in obj_converted_lst[:10]:
        if isinstance(item, str):
            layouts.append(mui.Markdown(item))
        elif isinstance(item, mui.Component):
            layouts.append(item)
    if not layouts:
        return None 
    return layouts