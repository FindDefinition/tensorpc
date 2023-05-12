
from pathlib import Path
import fire
from tensorpc.apps.file import get_file

def main(addr: str, server_path: str, store_path: str = ""):
    if store_path == "":
        store_path = str(Path.cwd())

    get_file(addr, server_path, store_path)


if __name__ == "__main__":
    fire.Fire(main)
