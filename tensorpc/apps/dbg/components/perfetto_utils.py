


import io
from typing import Any, List, Optional
import zipfile
import json 


def zip_trace_result(all_data: List[bytes], all_extra_events: List[Optional[Any]]):
    _use_perfetto_undoc_zip_of_gzip = True
    zip_ss = io.BytesIO()
    zip_mode = zipfile.ZIP_DEFLATED if not _use_perfetto_undoc_zip_of_gzip else zipfile.ZIP_STORED
    compresslevel = 9 if not _use_perfetto_undoc_zip_of_gzip else None
    with zipfile.ZipFile(zip_ss, mode="w", compression=zip_mode, compresslevel=compresslevel) as zf:
        for i, data in enumerate(all_data):
            zf.writestr(f"{i}.json", data)
        for i, data in enumerate(all_extra_events):
            if data:
                zf.writestr(f"{i}_extra.json", json.dumps({
                    "traceEvents": data
                }))
    res = zip_ss.getvalue()
    return res 