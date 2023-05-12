# Copyright 2022 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from tensorpc import marker, prim

import numpy as np 
class Simple:

    def echo(self, x):
        return x
    
class SpeedTestServer:

    def recv_data(self, x):
        return 
    
    def send_data(self, size_mb: int):
        return np.zeros([size_mb * 1024 * 1024], dtype=np.uint8)

class FileOps:

    def print_in_server(self, content):
        print(content)

    def get_file(self, path, start_chunk=0, chunk_size=65536):
        """service that get a large file from server.
        you need to use remote_generator instead of remote_call.
        If error occurs in client, you can use chunk index to
        recover transfer.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError("{} not found.".format(path))
        with path.open("rb") as f:
            f.seek(start_chunk * chunk_size)
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                yield data

    def get_file_size(self, path) -> int:
        path = Path(path)
        if not path.exists():
            return -1
        return path.stat().st_size

    def path_exists(self, path) -> bool:
        path = Path(path)
        return path.exists()

    def glob(self, folder, pattern):
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError("{} not found.".format(folder))
        res = list(folder.glob(pattern))
        if prim.is_json_call():
            return list(map(str, res))
        # for python client, we can send Path objects which have more information.
        return res

    def rglob(self, folder, pattern):
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError("{} not found.".format(folder))
        res = list(folder.glob(pattern))
        if prim.is_json_call():
            return list(map(str, res))
        return res

    @marker.mark_client_stream
    async def upload_file(self,
                          gen_iter,
                          path,
                          exist_ok: bool = False,
                          parents: bool = False):
        """service that upload a large file to server.
        you need to use client_stream instead of remote_call.
        for transfer recovery, we need to save states to server
        which isn't covered in this example.
        """
        path = Path(path)
        if path.exists() and not exist_ok:
            raise FileExistsError("{} exists.".format(path))
        if not path.parent.exists():
            if parents:
                path.parent.mkdir(mode=0o755, parents=parents)
            else:
                raise ValueError("{} parent not exist.".format(path))
        try:
            with path.open("wb") as f:
                async for chunk in gen_iter:
                    f.write(chunk)
        except Exception as e:
            path.unlink()
            raise e
