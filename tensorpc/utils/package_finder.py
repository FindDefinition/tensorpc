"""Find package from a file.

We will iterate every parent until "__init__.py" isn't found, then use the package name to find the package.
"""
from pathlib import Path 
import importlib.util
from typing import Optional

def find_submodule_from_file(file_path: str) -> Optional[str]:
    """Find submodule from a file."""
    file_path_p = Path(file_path)
    if not file_path_p.is_file():
        raise ValueError(f"File {file_path} does not exist.")
    if file_path_p.stem == "__init__":
        parts: list[str] = []
        notfound_cnt = 0
    else:
        parts: list[str] = [file_path_p.stem]
        notfound_cnt = 1
    # Iterate every parent until "__init__.py" isn't found
    while file_path_p != file_path_p.parent:
        parent = file_path_p.parent
        if not (parent / "__init__.py").is_file():
            break
        parts.append(parent.stem)
        file_path_p = file_path_p.parent
    if len(parts) == notfound_cnt:
        return None 
    return ".".join(reversed(parts))

if __name__ == "__main__":
    print(find_submodule_from_file(str(Path(__file__).resolve())))
