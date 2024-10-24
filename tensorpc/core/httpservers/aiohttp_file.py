from os import stat_result
import os
from typing import IO, TYPE_CHECKING, Any, AsyncGenerator, Optional, Tuple, Union
from aiohttp import web
from pathlib import Path 
from aiohttp.abc import AbstractStreamWriter
import abc

from tensorpc.core.defs import FileDesp, FileResource
if TYPE_CHECKING:
    from aiohttp.web_request import BaseRequest

class FileProxy(abc.ABC):
    @abc.abstractmethod
    def get_file_metadata(self) -> FileResource:
        ...

    @abc.abstractmethod
    async def get_file(self, offset: int, count: int) -> AsyncGenerator[Tuple[bytes, bool]]:
        ...

class FileProxyResponse(web.FileResponse):
    def __init__(
        self,
        proxy: FileProxy,
        chunk_size: int = 256 * 1024,
        status: int = 200,
        reason: Optional[str] = None,
        headers: Optional[Any] = None,
    ) -> None:
        # we need to provide a dummy but valid path to super's constructor
        super().__init__(
            path=str(Path(__file__).resolve()),
            chunk_size=chunk_size,
            status=status,
            reason=reason,
            headers=headers,
        ) 
        self._file_proxy = proxy

    async def _sendfile(
        self, request: "BaseRequest", fobj: IO[Any], offset: int, count: int
    ) -> AbstractStreamWriter:
        # To keep memory usage low,fobj is transferred in chunks
        # controlled by the constructor's chunk_size argument.
        writer = await super().prepare(request)
        assert writer is not None

        async for chunk, is_exc in self._file_proxy.get_file(offset, count):
            if is_exc:
                raise ValueError(chunk)
            await writer.write(chunk)
        await writer.drain()
        await super().write_eof()
        return writer

    def _get_file_path_stat_and_gzip(
        self, check_for_gzipped_file: bool
    ) -> Tuple[Path, stat_result, bool]:
        """Return the file path, stat result, and gzip status.

        This method should be called from a thread executor
        since it calls os.stat which may block.
        """
        meta = self._file_proxy.get_file_metadata()
        path = meta.name
        stat = meta.stat
        if stat is None:
            assert meta.length is not None 
            # create a fake stat result
            stat = os.stat_result((0, 0, 0, 0, 0, 0, meta.length))
        return Path(path), stat, False
