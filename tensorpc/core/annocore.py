
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, TypedDict, Union, Generic
from typing_extensions import Literal, Annotated, get_origin, get_args, get_type_hints
from dataclasses import dataclass
import inspect

def lenient_issubclass(cls: Any,
                       class_or_tuple: Any) -> bool:  # pragma: no cover
    return isinstance(cls, type) and issubclass(cls, class_or_tuple)

def is_annotated(ann_type: Any) -> bool:
    # https://github.com/pydantic/pydantic/blob/35144d05c22e2e38fe093c533ff3a05ce9a30116/pydantic/_internal/_typing_extra.py#L99C1-L104C1
    origin = get_origin(ann_type)
    return origin is not None and lenient_issubclass(origin, Annotated)

def is_optional(ann_type: Any) -> bool:
    origin = get_origin(ann_type)
    return origin is not None and lenient_issubclass(origin, Optional)

def is_async_gen(ann_type: Any) -> bool:
    # https://github.com/pydantic/pydantic/blob/35144d05c22e2e38fe093c533ff3a05ce9a30116/pydantic/_internal/_typing_extra.py#L99C1-L104C1
    origin = get_origin(ann_type)
    return origin is not None and lenient_issubclass(origin, AsyncGenerator)

@dataclass
class AnnotatedArg:
    name: str 
    param: Optional[inspect.Parameter] 
    type: Any 
    annometa: Optional[Any] = None 

@dataclass
class AnnotatedReturn:
    type: Any 
    annometa: Optional[Any] = None 

def extract_annotated_type_and_meta(ann_type: Any) -> Tuple[Any, Optional[Any]]:
    if is_annotated(ann_type):
        annometa = ann_type.__metadata__
        ann_type = get_args(ann_type)[0]
        return ann_type, annometa
    return ann_type, None

def parse_annotated_function(func: Callable) -> Tuple[List[AnnotatedArg], Optional[AnnotatedReturn]]:
    annos = get_type_hints(func, include_extras=True)
    
    specs = inspect.signature(func)
    name_to_parameter = {p.name: p for p in specs.parameters.values()}
    anno_args: List[AnnotatedArg] = []
    return_anno: Optional[AnnotatedReturn] = None
    for name, anno in annos.items():
        if name == "return":
            anno, annotated_metas = extract_annotated_type_and_meta(anno)
            return_anno = AnnotatedReturn(anno, annotated_metas)
        else:
            param = name_to_parameter[name]
            anno, annotated_metas = extract_annotated_type_and_meta(anno)

            arg_anno = AnnotatedArg(name, param, anno, annotated_metas)
            anno_args.append(arg_anno)
    for name, param in name_to_parameter.items():
        if name not in annos and param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            anno_args.append(AnnotatedArg(name, param, Any))
    return anno_args, return_anno

class WTF(TypedDict):
    pass 

class WTF2(WTF):
    c: int

class A:
    def add(self, a: int, b: int) -> WTF2:
        return WTF2(c=a + b)

    @staticmethod
    def add_stc(a: int, b: int) -> int:
        return a + b

if __name__ == "__main__":
    a = A()
    print(issubclass(WTF2,dict))
    print(dir(WTF2))
    print(parse_annotated_function(a.add))
    print(parse_annotated_function(a.add_stc))