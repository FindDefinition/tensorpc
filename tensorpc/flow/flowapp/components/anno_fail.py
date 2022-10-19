# Copyright 2022 Yan Yan
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Generic, TypeVar, Callable , Type, Any, Optional
from typing_extensions import ParamSpec, Concatenate
import dataclasses
P = ParamSpec('P')
T = TypeVar('T')

T2 = TypeVar('T2')
T3 = TypeVar('T3')

@dataclasses.dataclass
class Props:
    a: int = 3
    b: int = 5
    c: int = 4


def init_anno_fwd(this: Callable[P, Any], val: Optional[T3] = None) -> Callable[[Callable], Callable[P, T3]]:
    def decorator(real_function: Callable) -> Callable[P, T3]:
        def new_function(*args: P.args, **kwargs: P.kwargs) -> T3:
            return real_function(*args, **kwargs)
        return new_function
    return decorator

def init_anno_fwd_v2(this: Callable[Concatenate[Any, P], Any], val: Optional[T3] = None) -> Callable[[Callable], Callable[P, T3]]:
    def decorator(real_function: Callable) -> Callable[P, T3]:
        def new_function(*args: P.args, **kwargs: P.kwargs) -> T3:
            return real_function(*args, **kwargs)
        return new_function
    return decorator
    

class Component(Generic[T]):
    def __init__(self,
                 prop_cls: Callable[P, T],
                 prop_cls2: Type[T],
                 *args: P.args,
                 **kwargs: P.kwargs) -> None:
        self.__props = prop_cls(*args, **kwargs)
        self.__prop_cls = prop_cls
        self.prop_cls2 = prop_cls2


    @property
    def prop_pylance_works(self) -> Type[T]:
        def wrapper(*args: P.args, **kwargs: P.kwargs):
            for k, v in kwargs.items():
                setattr(self.__props, k, v)
            return self.__props
        return wrapper 


    @property
    def prop_pylance_dont_work(self) -> Callable[P, T]:
        def wrapper(*args: P.args, **kwargs: P.kwargs):
            for k, v in kwargs.items():
                setattr(self.__props, k, v)
            return self.__props
        return wrapper 

    def some_func(self, val: Callable[P, T], *args: P.args, **kwargs: P.kwargs):
        return  

    @property
    def prop_pylance_dont_work_too(self):
        @init_anno_fwd(partial(self.some_func, val=self.__prop_cls), self.__props)
        def wrapper(**kwargs):
            for k, v in kwargs.items():
                setattr(self.__props, k, v)
            return self.__props
        return wrapper 

    @property
    def prop_pylance_works_too_but_trivial(self):
        @init_anno_fwd(Props, self.__props)
        def wrapper(**kwargs):
            for k, v in kwargs.items():
                setattr(self.__props, k, v)
            return self.__props
        return wrapper 

    def runX(self, prop: Callable[P, Any]):
        def wrapper(*args: P.args, **kwargs: P.kwargs):
            for k, v in kwargs.items():
                setattr(self.__props, k, v)
            return self.__props
        return wrapper 

    def runX2(self):
        return self.runX(self.prop_cls2)



a = Component(Props, Props)

a.prop_pylance_works
a.prop_pylance_dont_work
a.prop_pylance_dont_work_too
a.prop_pylance_works_too_but_trivial
a.runX(Props)()
a.runX2()
