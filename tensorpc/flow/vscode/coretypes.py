
from tensorpc.flow.jsonlike import JsonLikeNode, as_dict_no_undefined, Undefined, undefined
from typing import (TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable,
                    Coroutine, Dict, Generic, Iterable, List, Optional, Set,
                    Tuple, Type, TypeVar, Union)
import enum 
from tensorpc.core import dataclass_dispatch

from pathlib import Path 

class VscodeTensorpcMessageType(enum.IntEnum):
    UpdateActiveTab = 0
    UpdateCursorPosition = 1
    ShowFunctionArguments = 2
    ShowFunctionLocals = 3

class VscodeTensorpcQueryType(enum.IntEnum):
    TraceTrees = 0
    DeleteTraceTree = 1

@dataclass_dispatch.dataclass
class Position:
    line: int
    character: int


@dataclass_dispatch.dataclass
class Selection:
    start: Position
    end: Position
    anchor: Position
    active: Position


@dataclass_dispatch.dataclass
class VscodeTensorpcMessage:
    type: VscodeTensorpcMessageType
    currentUri: str
    workspaceUri: str
    selections: Optional[List[Selection]] = None

    def get_workspace_path(self):
        if self.workspaceUri == "":
            return None 
        # workspaceUri: file:///home/xxx/xxx/xxx
        if self.workspaceUri.startswith("file://"):
            return Path(self.workspaceUri[7:])
        return None

@dataclass_dispatch.dataclass
class VscodeTensorpcQuery:
    type: VscodeTensorpcQueryType
    workspaceUri: str
    data: Any

    def get_workspace_path(self):
        if self.workspaceUri == "":
            return None 
        # workspaceUri: file:///home/xxx/xxx/xxx
        if self.workspaceUri.startswith("file://"):
            return Path(self.workspaceUri[7:])
        return None

@dataclass_dispatch.dataclass
class VscodeTraceItem:
    qualname: str
    childs: List["VscodeTraceItem"]
    path: str
    lineno: int
    duration: float = -1
    timestamp: Union[int, Undefined] = undefined
    rootKey: Union[str, Undefined] = undefined
    callerPath: Union[str, Undefined] = undefined
    callerLineno: Union[int, Undefined] = undefined

@dataclass_dispatch.dataclass
class VscodeTraceQuery:
    timestamp: Union[int, Undefined] = undefined
    rootKey: Union[str, Undefined] = undefined

@dataclass_dispatch.dataclass
class VscodeTraceQueries:
    queries: List[VscodeTraceQuery]


@dataclass_dispatch.dataclass
class VscodeTraceQueryResult:
    updates: List[VscodeTraceItem]
    deleted: List[str]


