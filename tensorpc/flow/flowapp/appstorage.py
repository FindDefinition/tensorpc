from typing import Dict
from tensorpc.core.tree_id import UniqueTreeIdForTree, UniqueTreeId
from tensorpc.flow.client import MasterMeta
from tensorpc.flow.core.appcore import get_app, get_app_context
from tensorpc.flow.coretypes import StorageDataItem, StorageDataLoadedItem
from typing import (TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable,
                    Coroutine, Dict, Generic, Iterable, List, Optional, Set,
                    Tuple, Type, TypeVar, Union)
from tensorpc.flow.jsonlike import JsonLikeNode, Undefined, parse_obj_to_jsonlike
from pathlib import Path
import pickle
import time
from tensorpc.flow.serv_names import serv_names
from tensorpc import simple_chunk_call_async


class AppStorage:

    def __init__(self, master_meta: MasterMeta, is_remote_comp: bool = False):

        self.__flowapp_master_meta = master_meta
        self.__flowapp_storage_cache: Dict[str, StorageDataItem] = {}
        if is_remote_comp:
            self.__flowapp_graph_id = ""
            self.__flowapp_node_id = ""
        else:
            self.__flowapp_graph_id = master_meta.graph_id
            self.__flowapp_node_id = master_meta.node_id

        self._is_remote_comp = is_remote_comp

        self._remote_grpc_url: Optional[str] = None

    def set_remote_grpc_url(self, url: Optional[str]):
        self._remote_grpc_url = url

    def set_graph_node_id(self, graph_id: str, node_id: str):
        self.__flowapp_graph_id = graph_id
        self.__flowapp_node_id = node_id

    def is_available(self):
        if not self._is_remote_comp:
            return True 
        else:
            return self._remote_grpc_url is not None

    async def _remote_call(self, serv_name: str, *args, **kwargs):
        if self._is_remote_comp:
            assert self._remote_grpc_url is not None, "app storage in remote comp can only be used when mounted"
            url = self._remote_grpc_url
            return await simple_chunk_call_async(
                url, serv_names.APP_RELAY_APP_STORAGE_FROM_REMOTE, serv_name,
                args, kwargs)
        else:
            url = self.__flowapp_master_meta.grpc_url
            return await simple_chunk_call_async(url, serv_name, *args,
                                                 **kwargs)

    async def save_data_storage(self,
                                key: str,
                                data: Any,
                                node_id: Optional[Union[str, Undefined]] = None,
                                graph_id: Optional[str] = None,
                                in_memory_limit: int = 1000,
                                raise_if_exist: bool = False):
        Path(key)  # check key is valid path
        data_enc = pickle.dumps(data)
        if not self._is_remote_comp:
            assert self.__flowapp_master_meta.is_inside_devflow, "you must call this in devflow apps."
        if graph_id is None:
            graph_id = self.__flowapp_graph_id
        if node_id is None:
            node_id = self.__flowapp_node_id
        elif isinstance(node_id, Undefined):
            # graph storage
            node_id = None
        meta = parse_obj_to_jsonlike(data, key,
                                     UniqueTreeIdForTree.from_parts([key]))
        in_memory_limit_bytes = in_memory_limit * 1024 * 1024
        meta.userdata = {
            "timestamp": time.time_ns(),
        }
        item = StorageDataItem(data_enc, meta)
        if len(data_enc) <= in_memory_limit_bytes:
            self.__flowapp_storage_cache[key] = item
        if len(data_enc) > in_memory_limit_bytes:
            raise ValueError("you can't store object more than 1GB size",
                             len(data_enc))
        await self._remote_call(serv_names.FLOW_DATA_SAVE,
                                graph_id,
                                node_id,
                                key,
                                data_enc,
                                meta,
                                item.timestamp,
                                raise_if_exist=raise_if_exist)

    async def data_storage_has_item(self,
                                    key: str,
                                    node_id: Optional[Union[str, Undefined]] = None,
                                    graph_id: Optional[str] = None):
        Path(key)  # check key is valid path
        meta = self.__flowapp_master_meta
        if not self._is_remote_comp:
            assert self.__flowapp_master_meta.is_inside_devflow, "you must call this in devflow apps."
        if graph_id is None:
            graph_id = self.__flowapp_graph_id
        if node_id is None:
            node_id = self.__flowapp_node_id
        elif isinstance(node_id, Undefined):
            # graph storage
            node_id = None
        if key in self.__flowapp_storage_cache:
            return True
        else:
            return await self._remote_call(serv_names.FLOW_DATA_HAS_ITEM,
                                           graph_id, node_id, key)

    async def read_data_storage(self,
                                key: str,
                                node_id: Optional[Union[str, Undefined]] = None,
                                graph_id: Optional[str] = None,
                                in_memory_limit: int = 100,
                                raise_if_not_found: bool = True):
        Path(key)  # check key is valid path
        if not self._is_remote_comp:
            assert self.__flowapp_master_meta.is_inside_devflow, "you must call this in devflow apps."
        if graph_id is None:
            graph_id = self.__flowapp_graph_id
        if node_id is None:
            node_id = self.__flowapp_node_id
        elif isinstance(node_id, Undefined):
            # graph storage
            node_id = None
        if key in self.__flowapp_storage_cache:
            item_may_invalid = self.__flowapp_storage_cache[key]
            res: Optional[StorageDataItem] = await self._remote_call(
                serv_names.FLOW_DATA_READ,
                graph_id,
                node_id,
                key,
                item_may_invalid.timestamp,
                raise_if_not_found=raise_if_not_found)
            if raise_if_not_found:
                assert res is not None
            if res is None:
                return None
            if res.empty():
                return pickle.loads(item_may_invalid.data)
            else:
                return pickle.loads(res.data)
        else:
            res: Optional[StorageDataItem] = await self._remote_call(
                serv_names.FLOW_DATA_READ,
                graph_id,
                node_id,
                key,
                raise_if_not_found=raise_if_not_found)
            if raise_if_not_found:
                assert res is not None
            if res is None:
                return None
            in_memory_limit_bytes = in_memory_limit * 1024 * 1024
            data = pickle.loads(res.data)
            if len(res.data) <= in_memory_limit_bytes:
                self.__flowapp_storage_cache[key] = res
            return data

    async def glob_read_data_storage(self,
                                               key: str,
                                               node_id: Optional[Union[str, Undefined]] = None,
                                               graph_id: Optional[str] = None):
        Path(key)  # check key is valid path
        if not self._is_remote_comp:
            assert self.__flowapp_master_meta.is_inside_devflow, "you must call this in devflow apps."
        if graph_id is None:
            graph_id = self.__flowapp_graph_id
        if node_id is None:
            node_id = self.__flowapp_node_id
        elif isinstance(node_id, Undefined):
            # graph storage
            node_id = None
        res: Dict[str, StorageDataItem] = await self._remote_call(
            serv_names.FLOW_DATA_READ_GLOB_PREFIX, graph_id, node_id, key)
        return {k: StorageDataLoadedItem(pickle.loads(d.data), d.meta) for k, d in res.items()}

    async def remove_data_storage_item(self,
                                       key: Optional[str],
                                       node_id: Optional[Union[str, Undefined]] = None,
                                       graph_id: Optional[str] = None):
        if key is not None:
            Path(key)
        if not self._is_remote_comp:
            assert self.__flowapp_master_meta.is_inside_devflow, "you must call this in devflow apps."
        if graph_id is None:
            graph_id = self.__flowapp_graph_id
        if node_id is None:
            node_id = self.__flowapp_node_id
        elif isinstance(node_id, Undefined):
            # graph storage
            node_id = None
        await self._remote_call(serv_names.FLOW_DATA_DELETE_ITEM, graph_id,
                                node_id, key)
        if key is None:
            self.__flowapp_storage_cache.clear()
        else:
            if key in self.__flowapp_storage_cache:
                self.__flowapp_storage_cache.pop(key)

    async def rename_data_storage_item(self,
                                       key: str,
                                       newname: str,
                                       node_id: Optional[Union[str, Undefined]] = None,
                                       graph_id: Optional[str] = None):
        Path(key)
        if not self._is_remote_comp:
            assert self.__flowapp_master_meta.is_inside_devflow, "you must call this in devflow apps."
        if graph_id is None:
            graph_id = self.__flowapp_graph_id
        if node_id is None:
            node_id = self.__flowapp_node_id
        elif isinstance(node_id, Undefined):
            # graph storage
            node_id = None
        await self._remote_call(serv_names.FLOW_DATA_RENAME_ITEM, graph_id,
                                node_id, key, newname)
        if key in self.__flowapp_storage_cache:
            if newname not in self.__flowapp_storage_cache:
                item = self.__flowapp_storage_cache.pop(key)
                self.__flowapp_storage_cache[newname] = item

    async def list_data_storage(self,
                                node_id: Optional[Union[str, Undefined]] = None,
                                graph_id: Optional[str] = None,
                                glob_prefix: Optional[str] = None):
        if not self._is_remote_comp:
            assert self.__flowapp_master_meta.is_inside_devflow, "you must call this in devflow apps."
        if graph_id is None:
            graph_id = self.__flowapp_graph_id
        if node_id is None:
            node_id = self.__flowapp_node_id
        elif isinstance(node_id, Undefined):
            # graph storage
            node_id = None
        res: List[dict] = await self._remote_call(
            serv_names.FLOW_DATA_LIST_ITEM_METAS, graph_id, node_id, glob_prefix)
        return [JsonLikeNode(**x) for x in res]

    async def list_all_data_storage_nodes(self,
                                          graph_id: Optional[str] = None):
        if not self._is_remote_comp:
            assert self.__flowapp_master_meta.is_inside_devflow, "you must call this in devflow apps."
        if graph_id is None:
            graph_id = self.__flowapp_graph_id
        res: List[str] = await self._remote_call(
            serv_names.FLOW_DATA_QUERY_DATA_NODE_IDS, graph_id)
        return res
