from typing import Any, Callable, Coroutine, Dict, Iterable, List, Generic, Mapping, Optional, Set, Tuple, TypeVar, Union
from typing_extensions import Literal, overload

T = TypeVar("T")

T_k = TypeVar("T_k")
T_v = TypeVar("T_v")

class DictProxy(Generic[T_k, T_v]):
    def __init__(self):
        self._dict: Dict[T_k, T_v] = {}

    def set_internal(self, other: Dict[T_k, T_v]):
        self._dict = other

    def __getitem__(self, key: T_k) -> T_v:
        return self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __contains__(self, key):
        return key in self._dict

    def __setitem__(self, key: T_k, value: T_v):
        self._dict[key] = value

    def __delitem__(self, key: T_k):
        del self._dict[key]

    def __repr__(self):
        return repr(self._dict)

    def __str__(self):
        return str(self._dict)
    
    def items(self):
        return self._dict.items()

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    @overload
    def get(self, key: T_k, /) -> T_v | None: ...

    @overload
    def get(self, key: T_k, /, default: T_v | T) -> T_v | T: ...

    def get(self, key: T_k, /, default: Union[T, T_v] = None) -> Union[T, T_v]:
        return self._dict.get(key, default)

    def pop(self, key: T_k, /, default: Optional[T_v] = None) -> Optional[T_v]:
        return self._dict.pop(key, default)

    def clear(self):
        self._dict.clear()

    def update(self, other: Mapping[T_k, T_v]): 
        self._dict.update(other)

    def copy(self):
        return self._dict.copy()

    def __eq__(self, other):
        return self._dict == other

    def __ne__(self, other):
        return self._dict != other

    def __hash__(self):
        return hash(self._dict)

    def __bool__(self):
        return bool(self._dict)

