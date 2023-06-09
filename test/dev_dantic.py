from dataclasses import dataclass as _dataclass_need_patch
from functools import partial
dataclass = partial(_dataclass_need_patch, eq=False)


@dataclass(eq=False)
class A:
    a: int

@dataclass(eq=False)
class B(A):
    b: int

A(1)