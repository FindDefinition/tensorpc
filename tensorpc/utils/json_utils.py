from typing import Any

try:
    import orjson as json  # type: ignore

    def json_dump_to_bytes(obj: Any) -> bytes:
        # json dump/load is very slow when trace data is large
        # so we use orjson if available
        return json.dumps(obj)
    def json_load_from_bytes(data: bytes) -> Any:
        return json.loads(data)
except ImportError:
    import json  # type: ignore

    def json_dump_to_bytes(obj: Any) -> bytes:
        return json.dumps(obj).encode()
    def json_load_from_bytes(data: bytes) -> Any:
        return json.loads(data)
