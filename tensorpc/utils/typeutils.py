from typing import Callable, Optional, cast
from typing_extensions import ParamSpec, TypeVar

T = TypeVar('T')
P = ParamSpec('P')



def copy_sig(source_func: Callable[P, T]) -> Callable[[Callable[..., T]], Callable[P, T]]:
    """Decorator to copy the function signature from source_func
    Especially useful when overwriting methods with many arguments
    CAREFUL: Only works towards the outside. Inside the function the types won't show up.
    
    @copy_sig(source_func)
    def other_func(*args: Any, **kwargs: Any):
        ...
    """
    def return_func(func: Callable[..., T]) -> Callable[P, T]:
        return cast(Callable[P, T], func)
    return return_func

def take_annotation_from(
    this: Callable[P, T]
) -> Callable[[Callable], Callable[P, T]]:

    def decorator(real_function: Callable) -> Callable[P, T]:

        def new_function(*args: P.args, **kwargs: P.kwargs) -> T:
            return real_function(*args, **kwargs)

        return new_function

    return decorator
