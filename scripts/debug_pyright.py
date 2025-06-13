import inspect
import dataclasses
from tensorpc.core.annolib import DataclassType
from typing import Any, Callable, ClassVar, Optional, Type, Union, TypeVar, cast

@dataclasses.dataclass
class FooDcls:
    dcls: Union[Type[DataclassType], Callable]


def foo(func):

    assert inspect.isclass(func) or inspect.isfunction(
        func
    )
    if inspect.isclass(func):
        assert dataclasses.is_dataclass(
            func
        )
    FooDcls(
        dcls=func
    )