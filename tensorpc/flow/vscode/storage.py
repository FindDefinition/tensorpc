from tensorpc.core import dataclass_dispatch
from tensorpc.flow.constants import TENSORPC_APP_STORAGE_VSCODE_TRACE_PATH
from tensorpc.flow.vscode.coretypes import VscodeTraceItem, VscodeTraceQueries, VscodeTraceQueryResult, VscodeBreakpoint
import time 
from typing import (TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable,
                    Coroutine, Dict, Generic, Iterable, List, Optional, Set,
                    Tuple, Type, TypeVar, Union)
from tensorpc.flow.jsonlike import JsonLikeNode, as_dict_no_undefined, Undefined, undefined
from tensorpc.flow.core.appcore import get_app, get_app_context

@dataclass_dispatch.dataclass
class AppVscodeState:
    """Sync vscode storage, no touch on app storage.
    """
    is_init: bool = False
    breakpoint_dict: Dict[str, tuple[List[VscodeBreakpoint], int]] = dataclass_dispatch.field(default_factory=dict)

    def set_workspace_breakpoints(self, workspace_uri: str, breakpoints: List[VscodeBreakpoint]):
        self.is_init = True
        self.breakpoint_dict[workspace_uri] = (breakpoints, time.time_ns())

    def get_all_breakpoints(self):
        return self.breakpoint_dict

@dataclass_dispatch.dataclass
class AppDataStorageForVscodeBase:
    """Vscode can only interact with app, not component.
    so we store all vscode data in app.
    """
    trace_trees: Dict[str, VscodeTraceItem] 

    def add_or_update_trace_tree(self, key: str, items: List[VscodeTraceItem]):
        self.trace_trees[key] = VscodeTraceItem("", items, "", -1, timestamp=time.time_ns(), rootKey=key)

    def remove_trace_tree(self, key: str):
        self.trace_trees.pop(key, None)

    def handle_vscode_trace_query(self, queries: VscodeTraceQueries) -> VscodeTraceQueryResult:
        updates: List[VscodeTraceItem] = []
        deleted: List[str] = []
        all_query_ids = set()            
        for query in queries.queries:
            assert not isinstance(query.timestamp, Undefined)
            assert not isinstance(query.rootKey, Undefined)
            all_query_ids.add(query.rootKey)
            if query.rootKey in self.trace_trees:
                item = self.trace_trees[query.rootKey]
                if item.timestamp != query.timestamp:
                    updates.append(item)
            else:
                deleted.append(query.rootKey)
        for k, v in self.trace_trees.items():
            if k not in all_query_ids:
                updates.append(v)
        return VscodeTraceQueryResult(updates, deleted)   

    def to_dict(self):
        res = as_dict_no_undefined(self)
        # don't save breakpoints
        res.pop("breakpoints")
        return res
        
@dataclass_dispatch.dataclass
class AppDataStorageForVscode(AppDataStorageForVscodeBase):
    async def add_or_update_trace_tree_with_update(self, key: str, items: List[VscodeTraceItem]):
        self.add_or_update_trace_tree(key, items)
        await self._update_vscode_storage()

    async def remove_trace_tree_with_update(self, key: str):
        self.remove_trace_tree(key)
        await self._update_vscode_storage()

    async def _update_vscode_storage(self):
        app = get_app()
        await app.app_storage.save_data_storage(TENSORPC_APP_STORAGE_VSCODE_TRACE_PATH, self.to_dict())

