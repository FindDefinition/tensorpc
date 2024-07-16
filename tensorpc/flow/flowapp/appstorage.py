from typing import Dict
from tensorpc.core.tree_id import UniqueTreeIdForTree, UniqueTreeId
from tensorpc.flow.client import MasterMeta
from tensorpc.flow.core.appcore import get_app, get_app_context
from tensorpc.flow.coretypes import StorageDataItem
from typing import (TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable,
                    Coroutine, Dict, Generic, Iterable, List, Optional, Set,
                    Tuple, Type, TypeVar, Union)
from tensorpc.flow.jsonlike import JsonLikeNode, parse_obj_to_jsonlike
from pathlib import Path 
import pickle 
import time 
from tensorpc.flow.serv_names import serv_names
from tensorpc import simple_chunk_call_async

class AppStorage:
    def __init__(self, master_meta: MasterMeta):

        self.__flowapp_master_meta = master_meta
        self.__flowapp_storage_cache: Dict[str, StorageDataItem] = {}

    async def save_data_storage(self,
                                key: str,
                                data: Any,
                                node_id: Optional[str] = None,
                                graph_id: Optional[str] = None,
                                in_memory_limit: int = 1000,
                                raise_if_exist: bool = False):
        Path(key) # check key is valid path
        data_enc = pickle.dumps(data)
        assert self.__flowapp_master_meta.is_inside_devflow, "you must call this in devflow apps."
        if graph_id is None:
            graph_id = self.__flowapp_master_meta.graph_id
        if node_id is None:
            node_id = self.__flowapp_master_meta.node_id
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
        await simple_chunk_call_async(self.__flowapp_master_meta.grpc_url,
                                      serv_names.FLOW_DATA_SAVE, graph_id,
                                      node_id, key, data_enc, meta,
                                      item.timestamp,
                                      raise_if_exist=raise_if_exist)

    async def data_storage_has_item(self,
                                key: str,
                                node_id: Optional[str] = None,
                                graph_id: Optional[str] = None):
        Path(key) # check key is valid path
        meta = self.__flowapp_master_meta
        assert self.__flowapp_master_meta.is_inside_devflow, "you must call this in devflow apps."
        if graph_id is None:
            graph_id = self.__flowapp_master_meta.graph_id
        if node_id is None:
            node_id = self.__flowapp_master_meta.node_id
        if key in self.__flowapp_storage_cache:
            return True 
        else:
            return await simple_chunk_call_async(
                meta.grpc_url, serv_names.FLOW_DATA_HAS_ITEM, graph_id, node_id,
                key) 
    
    async def read_data_storage(self,
                                key: str,
                                node_id: Optional[str] = None,
                                graph_id: Optional[str] = None,
                                in_memory_limit: int = 100,
                                raise_if_not_found: bool = True):
        Path(key) # check key is valid path
        meta = self.__flowapp_master_meta
        assert self.__flowapp_master_meta.is_inside_devflow, "you must call this in devflow apps."
        if graph_id is None:
            graph_id = self.__flowapp_master_meta.graph_id
        if node_id is None:
            node_id = self.__flowapp_master_meta.node_id
        if key in self.__flowapp_storage_cache:
            item_may_invalid = self.__flowapp_storage_cache[key]
            res: Optional[StorageDataItem] = await simple_chunk_call_async(meta.grpc_url,
                                                serv_names.FLOW_DATA_READ,
                                                graph_id, node_id, key,
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
            res: Optional[StorageDataItem] = await simple_chunk_call_async(
                meta.grpc_url, serv_names.FLOW_DATA_READ, graph_id, node_id,
                key, raise_if_not_found=raise_if_not_found)
            if raise_if_not_found:
                assert res is not None 
            if res is None:
                return None 
            in_memory_limit_bytes = in_memory_limit * 1024 * 1024
            data = pickle.loads(res.data)
            if len(res.data) <= in_memory_limit_bytes:
                self.__flowapp_storage_cache[key] = res
            return data

    async def read_data_storage_by_glob_prefix(self,
                                key: str,
                                node_id: Optional[str] = None,
                                graph_id: Optional[str] = None):
        Path(key) # check key is valid path
        meta = self.__flowapp_master_meta
        assert self.__flowapp_master_meta.is_inside_devflow, "you must call this in devflow apps."
        if graph_id is None:
            graph_id = self.__flowapp_master_meta.graph_id
        if node_id is None:
            node_id = self.__flowapp_master_meta.node_id
        res: Dict[str, StorageDataItem] = await simple_chunk_call_async(
            meta.grpc_url, serv_names.FLOW_DATA_READ_GLOB_PREFIX, graph_id, node_id,
            key)
        return {k: pickle.loads(d.data) for k, d in res.items()}

    async def remove_data_storage_item(self,
                                       key: Optional[str],
                                       node_id: Optional[str] = None,
                                       graph_id: Optional[str] = None):
        if key is not None:
            Path(key)
        meta = self.__flowapp_master_meta
        assert self.__flowapp_master_meta.is_inside_devflow, "you must call this in devflow apps."
        if graph_id is None:
            graph_id = self.__flowapp_master_meta.graph_id
        if node_id is None:
            node_id = self.__flowapp_master_meta.node_id
        await simple_chunk_call_async(meta.grpc_url,
                                      serv_names.FLOW_DATA_DELETE_ITEM,
                                      graph_id, node_id, key)
        if key is None:
            self.__flowapp_storage_cache.clear()
        else:
            if key in self.__flowapp_storage_cache:
                self.__flowapp_storage_cache.pop(key)

    async def rename_data_storage_item(self,
                                       key: str,
                                       newname: str,
                                       node_id: Optional[str] = None,
                                       graph_id: Optional[str] = None):
        Path(key)
        meta = self.__flowapp_master_meta
        assert self.__flowapp_master_meta.is_inside_devflow, "you must call this in devflow apps."
        if graph_id is None:
            graph_id = self.__flowapp_master_meta.graph_id
        if node_id is None:
            node_id = self.__flowapp_master_meta.node_id
        await simple_chunk_call_async(meta.grpc_url,
                                      serv_names.FLOW_DATA_RENAME_ITEM,
                                      graph_id, node_id, key, newname)
        if key in self.__flowapp_storage_cache:
            if newname not in self.__flowapp_storage_cache:
                item = self.__flowapp_storage_cache.pop(key)
                self.__flowapp_storage_cache[newname] = item

    async def list_data_storage(self,
                                node_id: Optional[str] = None,
                                graph_id: Optional[str] = None):
        meta = self.__flowapp_master_meta
        assert self.__flowapp_master_meta.is_inside_devflow, "you must call this in devflow apps."
        if graph_id is None:
            graph_id = self.__flowapp_master_meta.graph_id
        if node_id is None:
            node_id = self.__flowapp_master_meta.node_id
        res: List[dict] = await simple_chunk_call_async(
            meta.grpc_url, serv_names.FLOW_DATA_LIST_ITEM_METAS, graph_id,
            node_id)
        for x in res:
            # TODO remove this (old style uid compat)
            if "|" not in x["id"]:
                x["id"] = UniqueTreeId.from_parts([x["id"]]).uid_encoded
        return [JsonLikeNode(**x) for x in res]

    async def list_all_data_storage_nodes(self,
                                          graph_id: Optional[str] = None):
        meta = self.__flowapp_master_meta
        assert self.__flowapp_master_meta.is_inside_devflow, "you must call this in devflow apps."
        if graph_id is None:
            graph_id = self.__flowapp_master_meta.graph_id
        res: List[str] = await simple_chunk_call_async(
            meta.grpc_url, serv_names.FLOW_DATA_QUERY_DATA_NODE_IDS, graph_id)
        return res
