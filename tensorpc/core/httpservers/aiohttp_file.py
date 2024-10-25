import asyncio
from os import stat_result
import os
from typing import cast, IO, TYPE_CHECKING, Any, AsyncGenerator, Optional, Tuple, Union, Final
from aiohttp import web
from pathlib import Path
from aiohttp.abc import AbstractStreamWriter
import abc
import stat
from contextlib import suppress
from types import MappingProxyType

from tensorpc.core.defs import FileDesp, FileResource
from aiohttp.helpers import ETAG_ANY, ETag, must_be_empty_body
from aiohttp.typedefs import LooseHeaders, PathLike
from aiohttp import hdrs
from aiohttp.web_response import StreamResponse
from aiohttp.web_exceptions import (
    HTTPForbidden,
    HTTPNotFound,
    HTTPPartialContent,
    HTTPRequestRangeNotSatisfiable,
)
from stat import S_ISREG
from mimetypes import MimeTypes
import sys
CONTENT_TYPES: Final[MimeTypes] = MimeTypes()

if sys.version_info < (3, 9):
    CONTENT_TYPES.encodings_map[".br"] = "br"

# File extension to IANA encodings map that will be checked in the order defined.
ENCODING_EXTENSIONS = MappingProxyType(
    {ext: CONTENT_TYPES.encodings_map[ext] for ext in (".br", ".gz")}
)

FALLBACK_CONTENT_TYPE = "application/octet-stream"

if TYPE_CHECKING:
    from aiohttp.web_request import BaseRequest


class FileProxy(abc.ABC):
    @abc.abstractmethod
    def get_file_metadata(self) -> FileResource:
        ...

    @abc.abstractmethod
    async def get_file(
            self, offset: int, count: int
    ) -> AsyncGenerator[Tuple[Union[str, bytes], bool], None]:
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

    async def _sendfile_custom(self, request: "BaseRequest", offset: int,
                               count: int) -> AbstractStreamWriter:
        # To keep memory usage low,fobj is transferred in chunks
        # controlled by the constructor's chunk_size argument.
        writer = await StreamResponse.prepare(self, request)
        assert writer is not None
        async for chunk, is_exc in self._file_proxy.get_file(offset, count):
            if is_exc:
                raise ValueError(chunk)
            assert isinstance(chunk, bytes)
            await writer.write(chunk)
        await writer.drain()
        await super().write_eof()
        return writer

    def _get_file_path_stat_and_gzip(
            self,
            check_for_gzipped_file: bool) -> Tuple[Path, stat_result, bool]:
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
            # print("???LENGTH", meta.length)
            init_tuple = (0, 0, 0, 0, 0, 0, meta.length, 0, 0, 0)
            stat = os.stat_result(init_tuple)
            additional_tuple = tuple([0] * (stat.n_fields - len(init_tuple)))
            stat = os.stat_result(init_tuple + additional_tuple)
        return Path(path), stat, False

    def _get_file_path_stat_encoding_proxy(
            self, accept_encoding: str
    ) -> Tuple[Path, os.stat_result, Optional[str], bool]:
        """Return the file path, stat result, and encoding.

        If an uncompressed file is returned, the encoding is set to
        :py:data:`None`.

        This method should be called from a thread executor
        since it calls os.stat which may block.
        """
        meta = self._file_proxy.get_file_metadata()
        path = meta.name
        stat = meta.stat
        is_fake_stat = False
        if stat is None:
            assert meta.length is not None
            # create a fake stat result
            init_tuple = (0, 0, 0, 0, 0, 0, meta.length, 0, 0, 0)
            stat = os.stat_result(init_tuple)
            additional_tuple = tuple([0] * (stat.n_fields - len(init_tuple)))
            stat = os.stat_result(init_tuple + additional_tuple)
            is_fake_stat = True
        return Path(path), stat, None, is_fake_stat

    async def prepare(
            self, request: "BaseRequest") -> Optional[AbstractStreamWriter]:
        loop = asyncio.get_running_loop()
        # Encoding comparisons should be case-insensitive
        # https://www.rfc-editor.org/rfc/rfc9110#section-8.4.1
        accept_encoding = request.headers.get(hdrs.ACCEPT_ENCODING, "").lower()
        try:
            file_path, st, file_encoding, is_fake_stat = await loop.run_in_executor(
                None, self._get_file_path_stat_encoding_proxy, accept_encoding)
        except OSError:
            # Most likely to be FileNotFoundError or OSError for circular
            # symlinks in python >= 3.13, so respond with 404.
            self.set_status(HTTPNotFound.status_code)
            return await super().prepare(request)
        meta = self._file_proxy.get_file_metadata()
        st_mtime_ns = st.st_mtime_ns
        st_mtime = st.st_mtime
        if st_mtime_ns is None:
            st_mtime_ns = 0
        if is_fake_stat and meta.modify_timestamp_ns is not None:
            st_mtime_ns = meta.modify_timestamp_ns
            st_mtime = st_mtime_ns / 1e9
        etag_value = f"{st_mtime_ns:x}-{st.st_size:x}"
        last_modified = st_mtime

        # https://www.rfc-editor.org/rfc/rfc9110#section-13.1.1-2
        ifmatch = request.if_match
        if ifmatch is not None and not self._etag_match(
                etag_value, ifmatch, weak=False):
            return await self._precondition_failed(request)

        unmodsince = request.if_unmodified_since
        if (unmodsince is not None and ifmatch is None
                and st_mtime > unmodsince.timestamp()):
            return await self._precondition_failed(request)

        # https://www.rfc-editor.org/rfc/rfc9110#section-13.1.2-2
        ifnonematch = request.if_none_match
        if ifnonematch is not None and self._etag_match(
                etag_value, ifnonematch, weak=True):
            return await self._not_modified(request, etag_value, last_modified)

        modsince = request.if_modified_since
        if (modsince is not None and ifnonematch is None
                and st_mtime <= modsince.timestamp()):
            return await self._not_modified(request, etag_value, last_modified)

        status = self._status
        file_size = st.st_size
        count = file_size
        # print("???", file_size)
        start = None

        ifrange = request.if_range
        if ifrange is None or st_mtime <= ifrange.timestamp():
            # If-Range header check:
            # condition = cached date >= last modification date
            # return 206 if True else 200.
            # if False:
            #   Range header would not be processed, return 200
            # if True but Range header missing
            #   return 200
            try:
                rng = request.http_range
                start = rng.start
                end = rng.stop
            except ValueError:
                # https://tools.ietf.org/html/rfc7233:
                # A server generating a 416 (Range Not Satisfiable) response to
                # a byte-range request SHOULD send a Content-Range header field
                # with an unsatisfied-range value.
                # The complete-length in a 416 response indicates the current
                # length of the selected representation.
                #
                # Will do the same below. Many servers ignore this and do not
                # send a Content-Range header with HTTP 416
                self.headers[hdrs.CONTENT_RANGE] = f"bytes */{file_size}"
                self.set_status(HTTPRequestRangeNotSatisfiable.status_code)
                return await super().prepare(request)

            # If a range request has been made, convert start, end slice
            # notation into file pointer offset and count
            if start is not None or end is not None:
                if start < 0 and end is None:  # return tail of file
                    start += file_size
                    if start < 0:
                        # if Range:bytes=-1000 in request header but file size
                        # is only 200, there would be trouble without this
                        start = 0
                    count = file_size - start
                else:
                    # rfc7233:If the last-byte-pos value is
                    # absent, or if the value is greater than or equal to
                    # the current length of the representation data,
                    # the byte range is interpreted as the remainder
                    # of the representation (i.e., the server replaces the
                    # value of last-byte-pos with a value that is one less than
                    # the current length of the selected representation).
                    count = (
                        min(end if end is not None else file_size, file_size) -
                        start)

                if start >= file_size:
                    # HTTP 416 should be returned in this case.
                    #
                    # According to https://tools.ietf.org/html/rfc7233:
                    # If a valid byte-range-set includes at least one
                    # byte-range-spec with a first-byte-pos that is less than
                    # the current length of the representation, or at least one
                    # suffix-byte-range-spec with a non-zero suffix-length,
                    # then the byte-range-set is satisfiable. Otherwise, the
                    # byte-range-set is unsatisfiable.
                    self.headers[hdrs.CONTENT_RANGE] = f"bytes */{file_size}"
                    self.set_status(HTTPRequestRangeNotSatisfiable.status_code)
                    return await super().prepare(request)

                status = HTTPPartialContent.status_code
                # Even though you are sending the whole file, you should still
                # return a HTTP 206 for a Range request.
                self.set_status(status)

        # If the Content-Type header is not already set, guess it based on the
        # extension of the request path. The encoding returned by guess_type
        #  can be ignored since the map was cleared above.
        if hdrs.CONTENT_TYPE not in self.headers:
            self.content_type = (CONTENT_TYPES.guess_type(self._path)[0]
                                 or FALLBACK_CONTENT_TYPE)

        if file_encoding:
            self.headers[hdrs.CONTENT_ENCODING] = file_encoding
            self.headers[hdrs.VARY] = hdrs.ACCEPT_ENCODING
            # Disable compression if we are already sending
            # a compressed file since we don't want to double
            # compress.
            self._compression = False

        self.etag = etag_value  # type: ignore[assignment]
        self.last_modified = st_mtime  # type: ignore[assignment]
        self.content_length = count

        self.headers[hdrs.ACCEPT_RANGES] = "bytes"

        real_start = cast(int, start)

        if status == HTTPPartialContent.status_code:
            self.headers[hdrs.CONTENT_RANGE] = "bytes {}-{}/{}".format(
                real_start, real_start + count - 1, file_size)

        # If we are sending 0 bytes calling sendfile() will throw a ValueError
        if count == 0 or must_be_empty_body(request.method, self.status):
            return await StreamResponse.prepare(self, request)

        if start:  # be aware that start could be None or int=0 here.
            offset = start
        else:
            offset = 0

        res = await self._sendfile_custom(request, offset, count)
        return res
