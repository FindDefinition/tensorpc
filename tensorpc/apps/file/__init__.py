import tensorpc
from pathlib import Path
from typing import Union
from tensorpc.constants import TENSORPC_SPLIT
import tqdm

FILE_OPS_SERV = f"tensorpc.services.collection{TENSORPC_SPLIT}FileOps"


def _file_gen(path: Path, start_chunk=0, chunk_size=65536):
    with path.open("rb") as f:
        f.seek(start_chunk * chunk_size)
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            yield data


def upload_file(url,
                path: Union[str, Path],
                server_path: Union[str, Path],
                exist_ok: bool = False,
                parents: bool = False):
    path = Path(path)
    assert path.exists()
    with tensorpc.RemoteManager(url) as robj:
        robj.client_stream(f"{FILE_OPS_SERV}.upload_file",
                           _file_gen(path),
                           server_path,
                           exist_ok=exist_ok,
                           parents=parents)


def get_file(url, path, save_folder="."):
    save_folder = Path(save_folder)
    with tensorpc.RemoteManager(url) as robj:
        with open(str(save_folder / Path(path).name), "wb") as f:
            finished = False
            size = robj.remote_call(f"{FILE_OPS_SERV}.get_file_size", path)
            if size < 0:
                raise ValueError("file {} not exist".format(path))
            for i in range(5):
                chunk_idx = 0
                try:
                    with tqdm.tqdm(total=size) as pbar:
                        for data in robj.remote_generator(
                                f"{FILE_OPS_SERV}.get_file", path, chunk_idx):
                            f.write(data)
                            pbar.update(len(data))
                            chunk_idx += 1
                    finished = True
                    break
                except:
                    print("retry {} with chunk idx {}".format(i, chunk_idx))
        if not finished:
            raise RuntimeError("get file failed")
