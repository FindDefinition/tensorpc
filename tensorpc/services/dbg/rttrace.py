

from typing import Any, Optional
from tensorpc.dbg.rttracer import ChromeTraceStorage, RTTracerContext
from tensorpc.dbg.tracer import json_dump_to_bytes
import gzip
class RTTraceStorageService:
    def __init__(self):
        self._storage_dict: dict[str, ChromeTraceStorage] = {}

    def store_trace(self, key: str, storage: ChromeTraceStorage):
        self._storage_dict[key] = storage

    def get_trace_result(self, key: str) -> Optional[bytes]:
        storage = self._storage_dict.get(key)
        if storage is None:
            return None 
        # compress trace result
        res = json_dump_to_bytes(storage.get_trace_result())
        gzip_res = gzip.compress(res)

        return gzip_res