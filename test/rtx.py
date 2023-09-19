
from typing import (TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, Coroutine, Dict,
                    Iterable, List, Optional, Set, Tuple, Type, TypeVar, Union, get_type_hints)

from typing_extensions import ParamSpec
from tensorpc.flow.flowapp.components.typemetas import annotated_function_to_dataclass

def func(a: int = 5, b: float = 2):
    pass 

print(get_type_hints(func))

print(annotated_function_to_dataclass(func))