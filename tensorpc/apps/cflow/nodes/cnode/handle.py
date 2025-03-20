from typing import Callable, Optional, TypeVar
import tensorpc.core.dataclass_dispatch as dataclasses
from typing_extensions import Literal, get_type_hints, is_typeddict, Annotated, TypeAlias
from tensorpc.core.annolib import (AnnotatedArg, AnnotatedReturn,
                                    extract_annotated_type_and_meta, get_args,
                                    is_async_gen, is_not_required, 
                                    parse_annotated_function)
from tensorpc import compat
from tensorpc.dock.components.flowplus.style import ComputeFlowClasses
from tensorpc.dock.components import flowui, mui

NoneType = type(None)

class HandleTypePrefix:
    Input = "inp"
    Output = "out"
    SpecialDict = "specialdict"
    DriverInput = "driverinp"
    DriverOutput = "driverout"

@dataclasses.dataclass
class HandleMeta:
    is_handle_dict: bool = False

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class AnnoHandle:
    type: Literal["source", "target", "handledictsource"]
    prefix: str
    name: str
    is_optional: bool
    anno: AnnotatedArg
    meta: Optional[HandleMeta] = None

T_handle = TypeVar("T_handle")

SpecialHandleDict: TypeAlias = Annotated[dict[str, T_handle], HandleMeta(True)]

def is_typeddict_or_typeddict_async_gen(type):
    is_tdict = is_typeddict(type)
    if is_tdict:
        return True
    if is_async_gen(type):
        return is_typeddict(get_args(type)[0])
    return False

def parse_function_to_handles(func: Callable, is_dynamic_cls: bool):
    annos = parse_annotated_function(func, is_dynamic_cls)
    arg_annos = annos[0]
    inp_iohandles: list[AnnoHandle] = []
    out_iohandles: list[AnnoHandle] = []
    for arg_anno in arg_annos:
        param = arg_anno.param
        assert param is not None
        is_optional_val = param.default is not param.empty
        handle_meta = None
        if arg_anno.annometa is not None and arg_anno.annometa:
            if isinstance(
                arg_anno.annometa[0], HandleMeta):
                handle_meta = arg_anno.annometa[0]
        if handle_meta is not None and handle_meta.is_handle_dict:
            handle_type = "handledictsource"
            prefix = HandleTypePrefix.SpecialDict
        else:
            handle_type = "source"
            prefix = HandleTypePrefix.Input
        iohandle = AnnoHandle(handle_type, prefix,
                                arg_anno.name, is_optional_val, arg_anno,
                                handle_meta)
        inp_iohandles.append(iohandle)
    ranno_obj = annos[1]
    assert ranno_obj is not None, "your compute node function must contains a return annotation (None or typeddict or dataclass)"
    ranno = ranno_obj.type
    if ranno is NoneType:
        return inp_iohandles, out_iohandles
    assert is_typeddict_or_typeddict_async_gen(ranno), "return anno of your compute node function must be (typeddict or dataclass)"
    global_ns = None
    if not compat.Python3_10AndLater:
        global_ns = {}
    if is_async_gen(ranno):
        tdict_annos = get_type_hints(get_args(ranno)[0],
                                        include_extras=True,
                                        globalns=global_ns)
    else:
        tdict_annos = get_type_hints(ranno,
                                        include_extras=True,
                                        globalns=global_ns)
    for k, v in tdict_annos.items():
        v, anno_meta = extract_annotated_type_and_meta(v)
        handle_meta = None
        if anno_meta is not None and isinstance(anno_meta, HandleMeta):
            handle_meta = anno_meta
        ohandle = AnnoHandle(
            "target", HandleTypePrefix.Output, k, is_not_required(v),
            AnnotatedArg("", None, ranno_obj.type, ranno_obj.annometa),
            handle_meta)
        out_iohandles.append(ohandle)
    return inp_iohandles, out_iohandles

class IOHandle(mui.FlexBox):
    def __init__(self, prefix: str, name: str, is_input: bool,
                 annohandle: AnnoHandle):
        self._is_input = is_input
        self.name = name
        self.id = f"{prefix}-{name}"
        htype = "target" if is_input else "source"
        hpos = "left" if is_input else "right"
        if annohandle.type == "handledictsource":
            name = f"{{}}{name}"
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
            flowui.Handle(htype, hpos, self.id).prop(className=f"{ComputeFlowClasses.IOHandleBase} {handle_classes}",
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


def parse_func_to_handle_components(func: Callable, is_dynamic_cls: bool):
    inp_ahandles, out_ahandles = parse_function_to_handles(func, is_dynamic_cls)
    inp_iohandles: list[IOHandle] = []
    out_iohandles: list[IOHandle] = []
    for ahandle in inp_ahandles:
        iohandle = IOHandle(ahandle.prefix, ahandle.name, True,
                            ahandle)
        inp_iohandles.append(iohandle)
    for ahandle in out_ahandles:
        iohandle = IOHandle(ahandle.prefix, ahandle.name, False,
                            ahandle)
        out_iohandles.append(iohandle)
    return inp_iohandles, out_iohandles
