from typing import Callable, Optional
from typing_extensions import ParamSpec, TypeVar

T = TypeVar('T')
P = ParamSpec('P')


def take_annotation_from(
    this: Callable[P, T]
) -> Callable[[Callable], Callable[P, T]]:

    def decorator(real_function: Callable) -> Callable[P, T]:

        def new_function(*args: P.args, **kwargs: P.kwargs) -> T:
            return real_function(*args, **kwargs)

        return new_function

    return decorator
