from collections.abc import Mapping, Sequence
import contextlib
import dataclasses
import enum
from functools import partial
import inspect
from pathlib import Path
import time
from types import FrameType
from typing import Any, Callable, ContextManager, Optional, Union
from typing_extensions import Literal

from tensorpc.core.moduleid import get_qualname_of_type
from .parser import DoubleGlob, IdentityItem, IndexItem, ModuleStackQuery, ModuleVariableQueryExpr, ModuleWeightQuery, PartialGlob, PlainItem, QueryItem, SingleQuery, ModuleVariableQuery, TypeItem, parse_pmql


class SpecialModuleType(enum.IntEnum):
    NORMAL = 0
    MODULE_LIST = 1
    MODULE_DICT = 2
    SEQUENTIAL = 3

RESERVED_ATTRS = {
    "__fqn__",
    "__type__",
    "__pth_module__",
    "__pth_child_cnt__",
    "__type_qualname__",
}


@dataclasses.dataclass
class PthQueryResult:
    fqn: str 
    data: Any 

def _convert_pth_module_to_pytree(module: Any, fqn: str = ""):
    from torch.nn import Module, ModuleList, ModuleDict, Sequential
    assert isinstance(module, Module), "Expected a torch.nn.Module"
    res = {
        "__fqn__": fqn,
        "__type__": SpecialModuleType.NORMAL,
        "__pth_module__": module,
        "__pth_child_cnt__": 0,
        "__type_qualname__": get_qualname_of_type(type(module))
    }
    if isinstance(module, ModuleList):
        res["__type__"] = SpecialModuleType.MODULE_LIST
        for i, m in enumerate(module):
            res[str(i)] = _convert_pth_module_to_pytree(m, fqn=f"{fqn}.{i}" if fqn else str(i))
            res["__pth_child_cnt__"] += 1
    elif isinstance(module, ModuleDict):
        res["__type__"] = SpecialModuleType.MODULE_DICT
        for k, m in module.items():
            res[k] = _convert_pth_module_to_pytree(m, fqn=f"{fqn}.{k}" if fqn else k)
            res["__pth_child_cnt__"] += 1

    elif isinstance(module, Sequential):
        res["__type__"] = SpecialModuleType.SEQUENTIAL
        for i, m in enumerate(module):
            res[str(i)] = _convert_pth_module_to_pytree(m, fqn=f"{fqn}.{i}" if fqn else str(i))
            res["__pth_child_cnt__"] += 1

    else:
        res["__type__"] = SpecialModuleType.NORMAL
        for name, mod in module.named_children():
            res[name] = _convert_pth_module_to_pytree(mod, fqn=f"{fqn}.{name}" if fqn else name)
            res["__pth_child_cnt__"] += 1
    return res 

def _pytree_path_to_fs_path(path: tuple[Any, ...]):
    from torch.utils import _pytree as pytree
    path_str = []
    for p in path:
        if isinstance(p, pytree.MappingKey):
            path_str.append(f"{p.key}")
        elif isinstance(p, pytree.SequenceKey):
            path_str.append(f"{p.idx}")
        else:
            raise ValueError(f"Unknown path type {type(p)}")
    return "/".join(path_str)

def _do_query_on_module_dict_no_double_star(cur_items: list[dict[str, Any]], query_items: list[QueryItem]):
    for item in query_items:
        new_items: list[dict[str, Any]] = []
        for cur_item in cur_items:
            if isinstance(item, PlainItem):
                if item.id not in cur_item:
                    continue
                next_item = cur_item[item.id]
                new_items.append(next_item)
            elif isinstance(item, PartialGlob):
                glob_re = item.get_glob_regex()
                for k, v in cur_item.items():
                    if k not in RESERVED_ATTRS and glob_re.match(k):
                        new_items.append(v)
            elif isinstance(item, IndexItem):
                if item.is_all:
                    for next_item in cur_item.values():
                        if isinstance(next_item, dict):
                            new_items.append(next_item)
                else:
                    idx = item.index
                    if isinstance(idx, int):
                        idx = str(idx)
                    if idx in cur_item:
                        next_item = cur_item[idx]
                        new_items.append(next_item)
            elif isinstance(item, TypeItem):
                for next_item in cur_item.values():
                    if isinstance(next_item, dict):
                        qname = next_item["__type_qualname__"]
                        if item.id in qname:
                            new_items.append(next_item)
            if isinstance(item, IdentityItem):
                new_items.append(cur_item)
        cur_items = new_items
    return cur_items

def _do_query_on_var_dict_no_double_star(cur_items: list[tuple[str, Any]], query_items: list[QueryItem]):
    for item in query_items:
        new_items: list[tuple[str, Any]] = []
        for fqn, cur_item in cur_items:
            if isinstance(item, PlainItem):
                if not isinstance(cur_item, Mapping):
                    continue 
                if item.id not in cur_item:
                    continue
                next_item = cur_item[item.id]
                new_items.append((fqn + f".{item.id}" if fqn else item.id, next_item))
            elif isinstance(item, PartialGlob):
                glob_re = item.get_glob_regex()
                if not isinstance(cur_item, Mapping):
                    continue 
                for k, v in cur_item.items():
                    if glob_re.match(k):
                        new_items.append((fqn + f".{k}" if fqn else k, v))
            elif isinstance(item, IndexItem):
                if item.is_all:
                    if isinstance(cur_item, Mapping):
                        for k, v in cur_item.items():
                            new_items.append((fqn + f".{k}" if fqn else k, v))
                    elif isinstance(cur_item, Sequence) and not isinstance(cur_item, str):
                        for i, v in enumerate(cur_item):
                            k = str(i)
                            new_items.append((fqn + f".{k}" if fqn else k, v))
                else:
                    idx = item.index
                    if isinstance(idx, int) and isinstance(cur_item, Sequence) and not isinstance(cur_item, str):
                        if idx < 0:
                            idx = idx + len(cur_item)
                        if idx < len(cur_item):
                            k = str(idx)
                            new_items.append((fqn + f".{k}" if fqn else k, cur_item[idx]))
                    if isinstance(idx, str) and isinstance(cur_item, Mapping):
                        if idx in cur_item:
                            next_item = cur_item[idx]
                            new_items.append((fqn + f".{idx}" if fqn else idx, next_item))
            elif isinstance(item, TypeItem):
                if isinstance(cur_item, Mapping):
                    it = cur_item.items()
                    for k, next_item in cur_item.items():
                        qname = get_qualname_of_type(type(next_item))
                        if item.id in qname:
                            new_items.append((fqn + f".{k}" if fqn else k, next_item))

                elif isinstance(cur_item, Sequence) and not isinstance(cur_item, str):
                    for i, v in enumerate(cur_item):
                        qname = get_qualname_of_type(type(v))
                        if item.id in qname:
                            k = str(i)
                            new_items.append((fqn + f".{k}" if fqn else k, v))
                else:
                    continue 
        cur_items = new_items
    return cur_items

def _validate_mod_query_on_single_path(cur_items: list[tuple[str, dict[str, Any]]], query_items: list[QueryItem]):
    # assert len(cur_items) == len(query_items)
    for (item_key, item), query in zip(cur_items, query_items):
        if isinstance(query, PlainItem):
            if query.id != item_key:
                return False
        elif isinstance(query, PartialGlob):
            glob_re = query.get_glob_regex()
            if not glob_re.match(item_key):
                return False 
        elif isinstance(query, IndexItem):
            if not query.is_all:
                idx = query.index
                if isinstance(idx, int):
                    idx = str(idx)
                if idx != item_key:
                    return False
        elif isinstance(query, TypeItem):
            qname = item["__type_qualname__"]
            if query.id not in qname :
                return False
        else:
            raise NotImplementedError
    return True 

def _validate_varquery_on_single_path(cur_items: list[tuple[Union[str, int], dict[str, Any]]], query_items: list[QueryItem]):
    # assert len(cur_items) == len(query_items)
    for (item_key, item), query in zip(cur_items, query_items):
        if isinstance(query, PlainItem):
            if query.id != item_key:
                return False
        elif isinstance(query, PartialGlob):
            glob_re = query.get_glob_regex()
            if not glob_re.match(item_key):
                return False 
        elif isinstance(query, IndexItem):
            if not query.is_all:
                idx = query.index
                if idx != item_key:
                    return False
        elif isinstance(query, TypeItem):
            qname = get_qualname_of_type(type(item))
            if query.id not in qname :
                return False
        else:
            raise NotImplementedError
    return True 

def _path_to_str(path: tuple[Any, ...]):
    from torch.utils import _pytree as pytree
    path_str = []
    for p in path:
        if isinstance(p, pytree.MappingKey):
            path_str.append(f"{p.key}")
        elif isinstance(p, pytree.SequenceKey):
            path_str.append(f"{p.idx}")
        else:
            raise ValueError(f"Unknown path type {type(p)}")
    return ".".join(path_str)

def _do_query_on_module_dict(data_dict: dict[str, Any], query: SingleQuery):
    from torch.utils import _pytree as pytree
    from torch.utils._pytree import SequenceKey, MappingKey
    double_star_idx = len(query.items)
    length_after_double_star = 0
    for i, item in enumerate(query.items):
        if isinstance(item, DoubleGlob):
            double_star_idx = i
            length_after_double_star = len(query.items) - i - 1
            break

    cur_items: list[dict[str, Any]] = [data_dict]
    cur_items = _do_query_on_module_dict_no_double_star(cur_items, query.items[:double_star_idx])
    res_items: list[dict[str, Any]] = cur_items
    if double_star_idx < len(query.items) and cur_items:
        res_items = []
        # use pytree
        for cur_item in cur_items:
            path_with_item, _ = pytree.tree_flatten_with_path(cur_item, is_leaf=lambda x: isinstance(x, dict) and x["__pth_child_cnt__"] == 0)
            for path, item in path_with_item:
                if isinstance(item, dict) and "__pth_child_cnt__" in item:
                    cur_item_for_query: Any = cur_item
                    # print(_path_to_str(path), len(path), length_after_double_star, len(path) - length_after_double_star, cur_item_for_query["__type_qualname__"])
                    if len(path) >= length_after_double_star:
                        trace: list[tuple[str, dict]] = []
                        for j in range(len(path)):
                            path_item = path[j]
                            if isinstance(path_item, MappingKey):
                                key = path_item.key
                                cur_item_for_query = cur_item_for_query[path_item.key]
                            elif isinstance(path_item, SequenceKey):
                                key = str(path_item.idx)
                                cur_item_for_query = cur_item_for_query[path_item.idx]
                            else:
                                raise NotImplementedError
                            trace.append((key, cur_item_for_query))
                            #     print(cur_item_for_query["__type_qualname__"])
                        # print(cur_item_for_query["__type_qualname__"], cur_item_for_query["__fqn__"],)
                        # print([t[0] for t in trace], [t[1]["__fqn__"] for t in trace[-length_after_double_star:]])
                        is_valid = _validate_mod_query_on_single_path(trace[-length_after_double_star:], query.items[double_star_idx + 1:])
                        if is_valid:
                            res_items.append(item)
    return res_items

def _do_query_on_module_dict_v2(data_dict: dict[str, Any], query: SingleQuery):
    from torch.utils import _pytree as pytree
    from torch.utils._pytree import SequenceKey, MappingKey
    double_star_idx = len(query.items)
    length_after_double_star = 0
    length_before_double_star = double_star_idx
    for i, item in enumerate(query.items):
        if isinstance(item, DoubleGlob):
            double_star_idx = i
            length_after_double_star = len(query.items) - i - 1
            length_before_double_star = i
            break

    path_with_item, _ = pytree.tree_flatten_with_path(data_dict, is_leaf=lambda x: isinstance(x, dict) and x["__pth_child_cnt__"] == 0)
    res: list[Any] = []
    for path, item in path_with_item:
        if len(path) >= length_before_double_star + length_after_double_star and isinstance(item, dict) and "__pth_child_cnt__" in item:
            trace: list[tuple[str, dict]] = []
            cur_item_for_query: Any = data_dict
            for j in range(len(path)):
                path_item = path[j]
                if isinstance(path_item, MappingKey):
                    key = path_item.key
                    cur_item_for_query = cur_item_for_query[path_item.key]
                elif isinstance(path_item, SequenceKey):
                    key = str(path_item.idx)
                    cur_item_for_query = cur_item_for_query[path_item.idx]
                else:
                    raise NotImplementedError
                trace.append((key, cur_item_for_query))
            is_valid_before = _validate_mod_query_on_single_path(trace[:length_before_double_star], query.items[:double_star_idx])
            is_valid_after = _validate_mod_query_on_single_path(trace[-length_after_double_star:], query.items[double_star_idx + 1:])
            if is_valid_before and is_valid_after:
                res.append(item)
    return res 


def _do_var_query(data_dict: Any, query: SingleQuery):
    from torch.utils import _pytree as pytree
    from torch.utils._pytree import SequenceKey, MappingKey
    double_star_idx = len(query.items)
    length_after_double_star = 0
    length_before_double_star = double_star_idx
    for i, item in enumerate(query.items):
        if isinstance(item, DoubleGlob):
            double_star_idx = i
            length_after_double_star = len(query.items) - i - 1
            length_before_double_star = i
            break
    if double_star_idx == len(query.items):
        res_items_with_fqn = _do_query_on_var_dict_no_double_star([("", data_dict)], query.items)
        return [PthQueryResult(fqn, d) for fqn, d in res_items_with_fqn]
    path_with_item, _ = pytree.tree_flatten_with_path(data_dict)
    res: list[PthQueryResult] = []
    for path, item in path_with_item:
        if len(path) >= length_before_double_star + length_after_double_star:
            trace: list[tuple[Union[str, int], dict]] = []
            cur_item_for_query: Any = data_dict
            for j in range(len(path)):
                path_item = path[j]
                if isinstance(path_item, MappingKey):
                    key = path_item.key
                    cur_item_for_query = cur_item_for_query[path_item.key]
                elif isinstance(path_item, SequenceKey):
                    key = path_item.idx
                    cur_item_for_query = cur_item_for_query[path_item.idx]
                else:
                    raise NotImplementedError
                trace.append((key, cur_item_for_query))
            is_valid_before = _validate_varquery_on_single_path(trace[:length_before_double_star], query.items[:double_star_idx])
            is_valid_after = _validate_varquery_on_single_path(trace[-length_after_double_star:], query.items[double_star_idx + 1:])
            if is_valid_before and is_valid_after:
                fqn = _path_to_str(path)
                res.append(PthQueryResult(fqn, item))
    return res 

def simple_module_query(module: Any, query: Union[SingleQuery, ModuleWeightQuery], use_v2: bool = False):
    import torch
    res: list[PthQueryResult] = []

    assert isinstance(module, torch.nn.Module), "Expected a torch.nn.Module"
    if isinstance(query, ModuleWeightQuery):
        sq = query.mod_query
    else:
        sq = query
    if len(sq.items) == 1 and isinstance(sq.items[0], TypeItem):
        # handle special form for module only
        type_item_id = sq.items[0].id
        for fqn, m in module.named_modules():
            m_qualname = get_qualname_of_type(type(m))
            if type_item_id in m_qualname:
                res.append(PthQueryResult(fqn, m))
        return res
    module_dict = _convert_pth_module_to_pytree(module)

    # t = time.time()
    if use_v2:
        found_items = _do_query_on_module_dict_v2(module_dict, sq)
    else:
        found_items = _do_query_on_module_dict(module_dict, sq)
    # print(time.time() - t)
    if isinstance(query, ModuleWeightQuery):
        # get weight if exists
        for item in found_items:
            mod = item["__pth_module__"]
            assert isinstance(mod, torch.nn.Module), "Expected a torch.nn.Module"

            parts = query.var_name.split(".")
            try:
                param = mod.get_parameter(parts[0])
            except AttributeError:
                try:
                    param = mod.get_buffer(parts[0])
                except AttributeError:
                    continue 
            should_continue = False
            for part in parts[1:]:
                if hasattr(param, part):
                    param = getattr(param, part)
                else:
                    should_continue = True 
            if should_continue:
                continue 
            res.append(PthQueryResult(item["__fqn__"], param))

    else:
        res = [PthQueryResult(m["__fqn__"], m["__pth_module__"]) for m in found_items]
    return res

def _fwd_hook_for_mvq(mod, args, kwargs, output, queries: list[ModuleVariableQueryExpr], callback: Callable[[list[Any]], None], fqn: str):
    query_res_all: list[Any] = []
    for query in queries:
        query_key = query.type
        if query_key == "args":
            fqn_item = f"{fqn}@args"
            query_data = args 
        elif query_key == "kwargs":
            fqn_item = f"{fqn}@kwargs"
            query_data = kwargs
        elif query_key == "ret":
            fqn_item = f"{fqn}@ret"
            query_data = output
        else:
            raise ValueError(f"Unknown query key: {query_key}")
        query_res = _do_var_query(query_data, query.query)
        query_res = [dataclasses.replace(r, fqn=f"{fqn_item}::{r.fqn}" if r.fqn else fqn_item) for r in query_res]
        query_res_all.extend(query_res)
        del query_res
        del query_data
    callback(query_res_all)
    del query_res_all

def _bwd_hook_for_mvq(mod, args, output, queries: list[ModuleVariableQueryExpr], callback: Callable[[list[Any]], None], fqn: str):
    query_res_all: list[Any] = []
    for query in queries:
        query_key = query.type

        assert query_key != "kwargs", "backward hook does not support kwargs"
        if query_key == "args":
            fqn_item = f"{fqn}@grad_inputs"
            query_data = args 
        elif query_key == "ret":
            fqn_item = f"{fqn}@grad_outputs"
            query_data = output
        else:
            raise ValueError(f"Unknown query key: {query_key}")
        query_res = _do_var_query(query_data, query.query)
        query_res = [dataclasses.replace(r, fqn=f"{fqn_item}::{r.fqn}" if r.fqn else fqn_item) for r in query_res]
        query_res_all.extend(query_res)

        del query_res
        del query_data
    callback(query_res_all)
    del query_res_all

_PYTORCH_ROOT: Optional[Path] = None
_PYTORCH_MODULE_ROOT: Optional[Path] = None

def _cached_get_pth_root():
    global _PYTORCH_ROOT
    global _PYTORCH_MODULE_ROOT
    import torch 
    from torch.nn.modules import module 
    if _PYTORCH_ROOT is None:
        _PYTORCH_ROOT = Path(torch.__file__).resolve().parent
    if _PYTORCH_MODULE_ROOT is None:
        _PYTORCH_MODULE_ROOT = Path(module.__file__).resolve()
    return _PYTORCH_ROOT, _PYTORCH_MODULE_ROOT

def _get_first_non_module_frame(frame: Optional[FrameType]):
    if frame is None or frame.f_back is None:
        return None 
    frame = frame.f_back
    module_pth_root = _cached_get_pth_root()[1]
    while frame is not None:
        co_fname = frame.f_code.co_filename
        co_fname_path = Path(co_fname).resolve()
        # print(co_fname_path, module_pth_root)
        if module_pth_root != co_fname_path:
            return frame 
        frame = frame.f_back
    return None  

def _fwd_hook_for_msq(mod, args, output, queries: list[SingleQuery], callback: Callable[[list[Any]], None], fqn: str):
    cur_frame = inspect.currentframe()
    frame = _get_first_non_module_frame(cur_frame)
    if frame is None:
        return 
    query_data = frame.f_locals
    query_res_all: list[Any] = []
    for query in queries:
        query_res = _do_var_query(query_data, query)
        query_res = [dataclasses.replace(r, fqn=f"{fqn}@caller::{r.fqn}") for r in query_res]
        query_res_all.extend(query_res)
        del query_res
    callback(query_res_all)
    del query_data
    del frame
    del cur_frame

class QueryHook:
    def __init__(self, hooks: list[Any]):
        self.hooks = hooks

    def remove(self):
        for h in self.hooks:
            h.remove()

def install_module_hook_query(module: Any, queries: list[tuple[Union[ModuleVariableQuery, ModuleStackQuery], Callable[[list[Any]], None]]]):
    handles: list[Any] = []
    for item in queries:
        query = item[0]
        assert isinstance(query, (ModuleVariableQuery, ModuleStackQuery)), "currently only ModuleVariableQuery and ModuleStackQuery is supported in fwd"
        mods = simple_module_query(module, query.mod_query)
        if not mods:
            raise ValueError(f"No module found for query: {query}")
        for mod_res in mods:
            mod_fqn = mod_res.fqn
            with_kwargs = False
            if isinstance(query, ModuleVariableQuery):
                hook = partial(_fwd_hook_for_mvq, 
                            queries=query.var_queries,
                            callback=item[1],
                            fqn=mod_fqn)
                with_kwargs = True
            else:
                hook = partial(_fwd_hook_for_msq, 
                            queries=query.var_queries,
                            callback=item[1],
                            fqn=mod_fqn)
            handle = mod_res.data.register_forward_hook(hook, with_kwargs=with_kwargs)
            handles.append(handle)
    return QueryHook(handles)

def install_module_bwd_hook_query(module: Any, queries: list[tuple[ModuleVariableQuery, Callable[[list[Any]], None]]]):
    import torch
    handles: list[Any] = []
    for item in queries:
        query = item[0]
        assert isinstance(query, (ModuleVariableQuery)), "currently only ModuleVariableQuery is supported in bwd"
        mods = simple_module_query(module, query.mod_query)
        for mod_res in mods:
            mod_fqn = mod_res.fqn
            for qitem in query.var_queries:
                assert qitem.type != "kwargs", "backward hook does not support kwargs"
            hook = partial(_bwd_hook_for_mvq, 
                        queries=query.var_queries,
                        callback=item[1],
                        fqn=mod_fqn)
            assert isinstance(mod_res.data, torch.nn.Module)
            handle = mod_res.data.register_full_backward_hook(hook)
            handles.append(handle)
    return QueryHook(handles)

class RuntimeQueryContext:
    def __init__(self):
        self.result: dict[str, list[PthQueryResult]] = {}

    def handle_result(self, res: list[PthQueryResult], key: str, to_cpu: bool = True, to_cpu_ctx_creator: Optional[Callable[[], ContextManager]]=None):
        if key not in self.result:
            self.result[key] = []
        import torch
        from torch.utils import _pytree as pytree
        res_data = [r.data for r in res]
        if to_cpu:
            ctx = contextlib.nullcontext()
            if to_cpu_ctx_creator is not None:
                ctx = to_cpu_ctx_creator()
            with ctx:
                res_data = pytree.tree_map(lambda x: x.detach().cpu() if isinstance(x, torch.Tensor) else x, res_data)
        res = [dataclasses.replace(r, data=d) for r, d in zip(res, res_data)]
        self.result[key].extend(res)

@contextlib.contextmanager
def module_query_context(module: Any, /, to_cpu: bool = True, disabled: bool = False, 
        to_cpu_ctx_creator: Optional[Callable[[], ContextManager]] = None, **queries: str):
    ctx = RuntimeQueryContext()
    if disabled:
        yield ctx 
        return
    queries_item_list: list[tuple[Union[ModuleVariableQuery, ModuleStackQuery], Callable[[list[Any]], None]]] = []
    for query_key, query_str in queries.items():
        query = parse_pmql(query_str) 
        assert isinstance(query, (ModuleVariableQuery, ModuleStackQuery)), "only support mvq and msq"
        queries_item_list.append((query, partial(ctx.handle_result, key=query_key, to_cpu=to_cpu, to_cpu_ctx_creator=to_cpu_ctx_creator)))
    handle = install_module_hook_query(module, queries_item_list)
    try:
        yield ctx 
    finally:
        handle.remove()

@contextlib.contextmanager
def module_bwd_query_context(module: Any, /, to_cpu: bool = True, disabled: bool = False, 
        to_cpu_ctx_creator: Optional[Callable[[], ContextManager]] = None, **queries: str):
    ctx = RuntimeQueryContext()
    if disabled:
        yield ctx 
        return
    queries_item_list: list[tuple[ModuleVariableQuery, Callable[[list[Any]], None]]] = []
    for query_key, query_str in queries.items():
        query = parse_pmql(query_str) 
        assert isinstance(query,  ModuleVariableQuery), "only support mvq"
        queries_item_list.append((query, partial(ctx.handle_result, key=query_key, to_cpu=to_cpu, to_cpu_ctx_creator=to_cpu_ctx_creator)))
    handle = install_module_bwd_hook_query(module, queries_item_list)
    try:
        yield ctx 
    finally:
        handle.remove()

def _analysis_result(res: list[PthQueryResult], handler: Callable[[list[PthQueryResult]], None]):
    import torch
    with torch.no_grad():
        handler(res)

@contextlib.contextmanager
def module_analysis_context(module: Any, handler: Callable[[list[PthQueryResult]], None], /, **queries: str):
    queries_item_list: list[tuple[Union[ModuleVariableQuery, ModuleStackQuery], Callable[[list[Any]], None]]] = []
    for query_key, query_str in queries.items():
        query = parse_pmql(query_str) 
        assert isinstance(query, (ModuleVariableQuery, ModuleStackQuery)), "only support mvq and msq"
        queries_item_list.append((query, partial(_analysis_result, handler=handler)))
    handle = install_module_hook_query(module, queries_item_list)
    try:
        yield
    finally:
        handle.remove()