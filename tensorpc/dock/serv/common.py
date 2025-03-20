


import inspect
import io
from pathlib import Path
from typing import Callable, Optional, Union, Coroutine

from tensorpc.core.defs import FileDesp, FileResourceRequest, FileResource


async def handle_file_resource(req: FileResourceRequest, handler: Callable[[FileResourceRequest], Union[FileResource, Coroutine[None, None, FileResource]]], chunk_size: int, count: Optional[int]):
    # base = req.key
    offset = req.offset
    res = handler(req)
    if inspect.iscoroutine(res):
        res = await res
    assert isinstance(res, (str, bytes, FileResource))
    if isinstance(res, (str, bytes)):
        if isinstance(res, str):
            res = res.encode()
        bio = io.BytesIO(res)
        if offset is not None:
            bio.seek(offset)
        chunk = bio.read(chunk_size)
        while chunk:
            yield chunk
            if count is not None:
                count = count - chunk_size
                if count <= 0:
                    break
            chunk = bio.read(chunk_size)
    else:
        fname = res.name
        if res.chunk_size is not None:
            assert res.chunk_size > 1024
            chunk_size = res.chunk_size
        if res.path is not None:
            # yield FileDesp(Path(res.path).name, res.content_type)
            with open(res.path, "rb") as f:
                if offset is not None:
                    f.seek(offset)
                chunk = f.read(chunk_size)
                while chunk:
                    yield chunk
                    if count is not None:
                        count = count - chunk_size
                        if count <= 0:
                            break
                    chunk = f.read(chunk_size)
        elif res.content is not None:
            content = res.content
            if isinstance(content, str):
                content = content.encode()
            bio = io.BytesIO(content)
            if offset is not None:
                bio.seek(offset)
            chunk = bio.read(chunk_size)
            # yield FileDesp(fname, res.content_type)
            while chunk:
                yield chunk
                if count is not None:
                    count = count - chunk_size
                    if count <= 0:
                        break
                chunk = bio.read(chunk_size)
        else:
            raise NotImplementedError
