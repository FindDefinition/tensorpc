# from dataclasses import dataclass as _dataclass_need_patch
# from functools import partial
# dataclass = partial(_dataclass_need_patch, eq=False)

import tensorpc.core.dataclass_dispatch as dataclasses

from typing import (TYPE_CHECKING, Any, AsyncGenerator, AsyncIterable,
                    Awaitable, Callable, Coroutine, Dict, Iterable, List,
                    Optional, Set, Tuple, Type, TypeVar, Union)
from tensorpc.flow import mui 
from pydantic import BaseModel
# @dataclass(eq=False)
class A:
    pass 

# @dataclass(eq=False)
# class B(A):
#     b: int

# A(1)

class GridLayoutProps(BaseModel):
    width: Union[mui.Undefined, int] = mui.undefined
    autoSize: Union[bool, mui.Undefined] = mui.undefined
    cols: Union[int, mui.Undefined] = mui.undefined
    draggableHandle: Union[mui.Undefined, str] = mui.undefined
    rowHeight: Union[mui.Undefined, int] = mui.undefined

wtf2 = GridLayoutProps.model_validate(dict(cols=16,
                draggableHandle=".grid-layout-drag-handle",
                rowHeight=50))
# wtf = mui.GridLayout([]).prop(flex=1,
#                 cols=16,
#                 draggableHandle=".grid-layout-drag-handle",
#                 rowHeight=50)
# tab_theme = mui.Theme(
#     components={
#         "MuiTab": {
#             "styleOverrides": {
#                 "root": {
#                     "padding": "0",
#                     "minWidth": "28px",
#                     "minHeight": "28px",
#                 }
#             }
#         }
#     })
