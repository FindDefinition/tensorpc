from pathlib import Path
from tensorpc.constants import TENSORPC_FILE_NAME_PREFIX
from tensorpc.core.moduleid import InMemoryFS, get_obj_type_meta

import inspect

def main():
    fs = InMemoryFS()
    with open(Path(__file__).parent / "wtf.py", "r") as f:
        test_data = f.read()
    inmemory_path = f"<{TENSORPC_FILE_NAME_PREFIX}-wtf>"
    fs.add_file(inmemory_path, test_data)
    mod = fs.load_in_memory_module(inmemory_path)

    print(mod.__file__)
    A = mod.User
    # print(inspect.getfile(A))
    print(A.__module__)

    type_meta = get_obj_type_meta(A)
    if type_meta is not None:
        type_meta.get_reloaded_module(fs)
        print(type_meta)


if __name__ == "__main__":
    main()