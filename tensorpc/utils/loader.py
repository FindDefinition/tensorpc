import dataclasses
import importlib
import importlib.util
import sys
from pathlib import Path
from types import FrameType
from typing import Optional
import re 
from importlib.metadata import PackageNotFoundError, distribution, version
from tensorpc.core.inspecttools import get_co_qualname_from_frame

VALID_PYTHON_MODULE_NAME_PATTERN = re.compile(r"[a-zA-Z_][0-9a-zA-Z_]*")

@dataclasses.dataclass
class FrameModuleMeta:
    module: str
    qualname: str
    is_path: bool

def locate_package(package_name, cwd_check=False) -> Optional[Path]:
    """locale package by find_spec, perfered method.
    """
    assert package_name != "__init__"
    if not VALID_PYTHON_MODULE_NAME_PATTERN.match(package_name):
        return None
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return None
    if spec.origin is None:
        return None

    origin = Path(spec.origin)
    cwd = Path.cwd()
    if origin.name == "__init__.py":
        # has submodule
        if origin.parent.parent == cwd and cwd_check:
            msg = (
                "find a package in cwd, you must use locate_package_subproc"
                " or run program in a directory that dont contain a package.")
            raise ValueError(msg)
        return origin.parent
    return origin


def locate_top_package(file_path, check_dist=False, cwd_check=False):
    """you need to provide a setup.py in your project and
    install it with pip install -e . or regular approach.
    if a package is located via cwd, a warn is produced and cwd package
    is returned.
    """
    file_path = Path(file_path).resolve()
    res = None
    cwd = Path.cwd()
    for path in file_path.parents:
        package_root = locate_package(path.stem)
        if package_root is not None:
            if package_root != path:
                continue
            if check_dist:
                try:
                    _ = distribution(path.stem)
                    res = path
                    break
                except PackageNotFoundError:
                    pass
            else:
                res = path
                if path.parent == cwd and cwd_check:
                    msg = (
                        "find a package in cwd, you must use locate_top_package_subproc"
                        " or run program in a directory that dont inside a package."
                    )
                    raise ValueError(msg)
                break
    return res

def _get_mod_name_from_path(path):
    path = Path(path)
    return path.name.split(".")[0]

def try_capture_import_parts(source_path, package_name=None):
    """
    if your function path is project/mod0/func_mod.py,
    then the import_parts is [project, mod0, func_mod]
    """
    source_path = Path(source_path).resolve()
    if package_name is None:
        # find outer module name
        path = locate_top_package(source_path)
        if path is None:
            return None
        path_candidates = [path]
        found_modules = []
        for path in path_candidates:
            module_name = _get_mod_name_from_path(path)
            module_loader = importlib.util.find_spec(module_name)
            if module_loader is not None:
                found_modules.append(module_name)
        assert len(
            path_candidates
        ) == 1, "can't determine unique project in {}. you should provide name by yourself.".format(
            found_modules)
        package_name = found_modules[0]
    relative_path = source_path.relative_to(path_candidates[0])
    relative_path = relative_path.parent / _get_mod_name_from_path(
        relative_path)
    return [package_name, *relative_path.parts]


def get_frame_module_meta(frame: FrameType):
    qname = get_co_qualname_from_frame(frame)
    frame_path = frame.f_code.co_filename
    try:
        parts = try_capture_import_parts(frame_path)
        if parts is not None:
            return FrameModuleMeta(".".join(parts), qname, False)
        else:
            return FrameModuleMeta(frame_path, qname, True)
    except:
        return FrameModuleMeta(frame_path, qname, True)

