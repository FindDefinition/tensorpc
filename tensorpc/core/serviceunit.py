import ast
import dataclasses
import importlib
import importlib.util
import inspect
import os
import runpy
import sys
import time
import tokenize
import types
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import (Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple,
                    Type)

from tensorpc import compat
from tensorpc.constants import (TENSORPC_FLOW_FUNC_META_KEY,
                                TENSORPC_FUNC_META_KEY, TENSORPC_SPLIT)
from tensorpc.core import inspecttools
from tensorpc.core.funcid import (get_toplevel_class_node,
                                  get_toplevel_func_node)
from tensorpc.core.moduleid import TypeMeta, get_obj_type_meta


class ParamType(Enum):
    PosOnly = "PosOnly"
    PosOrKw = "PosOrKw"
    VarPos = "VarPos"
    KwOnly = "KwOnly"
    VarKw = "VarKw"


_SIG_TYPE_TO_META_TYPE = {
    inspect.Parameter.KEYWORD_ONLY: ParamType.KwOnly,
    inspect.Parameter.POSITIONAL_ONLY: ParamType.PosOnly,
    inspect.Parameter.POSITIONAL_OR_KEYWORD: ParamType.PosOrKw,
    inspect.Parameter.VAR_KEYWORD: ParamType.VarKw,
    inspect.Parameter.VAR_POSITIONAL: ParamType.VarPos,
}


@dataclasses.dataclass
class ParamMeta(object):
    name: str
    type: ParamType

    def __init__(self, name: str, param: inspect.Parameter):
        # TODO annotation and default
        self.name = name
        self.type = _SIG_TYPE_TO_META_TYPE[param.kind]

    def to_json(self):
        return {
            "name": self.name,
            "type": self.type.value,
        }


class ServiceType(Enum):
    Normal = "Normal"
    Exit = "Exit"
    AsyncInit = "AsyncInit"
    BiStream = "BidirectinoalStream"  # only support grpc for now
    ClientStream = "ClientStream"  # only support grpc for now
    AsyncWebSocket = "AsyncWebSocket"  # only support ws
    WebSocketEventProvider = "EventProvider"  # only support ws
    WebSocketOnConnect = "WebSocketOnConnect"  # only support ws
    WebSocketOnDisConnect = "WebSocketOnDisConnect"  # only support ws


class AppFuncType(Enum):
    CreateLayout = "CreateLayout"
    AutoRun = "AutoRun"
    CreateObject = "CreateObject"
    RunInExecutor = "RunInExecutor"
    ComponentDidMount = "ComponentDidMount"
    ComponentWillUnmount = "ComponentWillUnmount"


class AppFunctionMeta:

    def __init__(self, type: AppFuncType, name: str = "") -> None:
        self.type = type
        self.name = name

    def to_dict(self):
        return {"type": self.type.value, "name": self.name}

    def __repr__(self):
        return f"{self.type}|{self.name}"


@dataclasses.dataclass
class ServFunctionMeta:
    fn: Callable
    name: str
    type: ServiceType
    args: List[ParamMeta]
    is_gen: bool
    is_async: bool
    is_static: bool
    is_binded: bool
    code: str = ""
    qualname: str = ""
    user_app_meta: Optional[AppFunctionMeta] = None
    binded_fn: Optional[Callable] = None

    def __init__(self,
                 fn: Callable,
                 name: str,
                 type: ServiceType,
                 sig: inspect.Signature,
                 is_gen: bool,
                 is_async: bool,
                 is_static: bool,
                 is_binded: bool,
                 qualname: str = "",
                 user_app_meta: Optional[AppFunctionMeta] = None) -> None:
        self.name = name
        self.type = type
        self.args = [ParamMeta(n, p) for n, p in sig.parameters.items()]
        if not is_static:
            self.args = self.args[1:]
        self.is_gen = is_gen
        self.is_async = is_async
        self.is_static = is_static
        self.is_binded = is_binded
        self.fn = fn
        self.code = ""
        self.qualname = qualname
        self.user_app_meta = user_app_meta

    def to_json(self):
        if self.user_app_meta is not None:
            user_app_meta = self.user_app_meta.to_dict()
        else:
            user_app_meta = None
        return {
            "name": self.name,
            "type": self.type.value,
            "args": [a.to_json() for a in self.args],
            "is_gen": self.is_gen,
            "is_async": self.is_async,
            "is_static": self.is_static,
            "is_binded": self.is_binded,
            "code": self.code,
            "user_app_meta": user_app_meta,
        }

    def bind(self, obj):
        if not self.is_binded:
            if not self.is_static:
                new_method = types.MethodType(self.fn, obj)
            else:
                new_method = self.fn
            self.binded_fn = new_method
            return self.binded_fn
        assert self.binded_fn is not None
        return self.binded_fn

    def get_binded_fn(self):
        assert self.binded_fn is not None
        return self.binded_fn


@dataclass
class FileCacheEntry:
    size: int
    mtime: Optional[float]
    fullname: str
    lines: List[str]
    qualname_to_code: Optional[Dict[str, str]] = None

    @property
    def invalid(self):
        return self.mtime is None

    @property
    def source(self):
        return "\n".join(self.lines)


@dataclass
class TypeCacheEntry:
    type: Type[Any]
    module: ModuleType
    module_dict: Dict[str, Any]
    method_metas: Optional[List[ServFunctionMeta]] = None


@dataclass
class ModuleCacheEntry:
    module: ModuleType
    module_dict: Dict[str, Any]


def _splitlines_no_ff(source):
    """Split a string into lines ignoring form feed and other chars.

    This mimics how the Python parser splits source code.
    """
    idx = 0
    lines = []
    next_line = ''
    while idx < len(source):
        c = source[idx]
        next_line += c
        idx += 1
        # Keep \r\n together
        if c == '\r' and idx < len(source) and source[idx] == '\n':
            next_line += '\n'
            idx += 1
        if c in '\r\n':
            lines.append(next_line)
            next_line = ''

    if next_line:
        lines.append(next_line)
    return lines


def _pad_whitespace(source):
    r"""Replace all chars except '\f\t' in a line with spaces."""
    result = ''
    for c in source:
        if c in '\f\t':
            result += c
        else:
            result += ' '
    return result


class SourceSegmentGetter:

    def __init__(self, source: str) -> None:
        self.lines = _splitlines_no_ff(source)

    def get_source_segment(self, node, *, padded=False):
        """Get source code segment of the *source* that generated *node*.

        If some location information (`lineno`, `end_lineno`, `col_offset`,
        or `end_col_offset`) is missing, return None.

        If *padded* is `True`, the first line of a multi-line statement will
        be padded with spaces to match its original position.
        """
        try:
            lineno = node.lineno - 1
            end_lineno = node.end_lineno - 1
            col_offset = node.col_offset
            end_col_offset = node.end_col_offset
        except AttributeError:
            return None

        lines = self.lines
        if end_lineno == lineno:
            return lines[lineno].encode()[col_offset:end_col_offset].decode()

        if padded:
            padding = _pad_whitespace(
                lines[lineno].encode()[:col_offset].decode())
        else:
            padding = ''

        first = padding + lines[lineno].encode()[col_offset:].decode()
        last = lines[end_lineno].encode()[:end_col_offset].decode()
        lines = lines[lineno + 1:end_lineno]

        lines.insert(0, first)
        lines.append(last)
        return ''.join(lines)


def get_qualname_to_code(lines: List[str]):
    source = "".join(lines)
    tree = ast.parse(source)
    nodes = get_toplevel_func_node(tree)
    nodes += get_toplevel_class_node(tree)
    qualname_to_code: Dict[str, str] = {}
    ssg = SourceSegmentGetter(source)
    for n, nss in nodes:
        ns = ".".join([nx.name for nx in nss])
        qualname = ns + "." + n.name
        # TODO this function won't handle decorator
        # here we can't use ast.get_source_segment
        # because it's very slow. we need to cache
        # lines.
        code = ssg.get_source_segment(n, padded=False)
        assert code is not None
        qualname_to_code[qualname] = code
    return qualname_to_code


class ObjectReloadManager:
    """to resolve some side effects, users should
    always use reload manager defined in app.
    """

    def __init__(self) -> None:
        self.file_cache: Dict[str, FileCacheEntry] = {}
        self.type_cache: Dict[str, TypeCacheEntry] = {}
        self.type_meta_cache: Dict[str, TypeMeta] = {}
        self.type_method_meta_cache: Dict[Type, List[ServFunctionMeta]] = {}
        self.module_cache: Dict[str, ModuleCacheEntry] = {}

    def check_file_cache(self, path: str):
        if path not in self.file_cache:
            return
        entry = self.file_cache[path]
        if entry.invalid:
            return
        try:
            stat = os.stat(entry.fullname)
        except OSError:
            self.file_cache.pop(path)
            return
        if entry.size != stat.st_size or entry.mtime != stat.st_mtime:
            self.file_cache.pop(path)

    def cached_get_obj_meta(self, type):
        if type in self.type_meta_cache:
            meta = self.type_meta_cache[type]
        else:
            meta = get_obj_type_meta(type)
            if meta is not None:
                self.type_meta_cache[type] = meta
        return meta

    def _update_file_cache(self, path: str):
        fullname = path
        stat = None
        try:
            stat = os.stat(path)
        except OSError:
            basename = path
            # Try looking through the module search path, which is only useful
            # when handling a relative filename.
            if os.path.isabs(path):
                raise ValueError("can't reload this type", type)
            found = False
            for dirname in sys.path:
                try:
                    fullname = os.path.join(dirname, basename)
                except (TypeError, AttributeError):
                    # Not sufficiently string-like to do anything useful with.
                    continue
                try:
                    stat = os.stat(fullname)
                    found = True
                    break
                except OSError:
                    pass
            if not found:
                raise ValueError("can't reload this type", type)
        assert stat is not None
        try:
            with tokenize.open(fullname) as fp:
                lines = fp.readlines()
        except OSError:
            lines = []
        size, mtime = stat.st_size, stat.st_mtime
        entry = FileCacheEntry(size, mtime, fullname, lines)
        self.file_cache[path] = entry
        if lines and compat.Python3_8AndLater:
            try:
                qualname_to_code = get_qualname_to_code(lines)
                entry.qualname_to_code = qualname_to_code
            except:
                pass

    def reload_type(self, type):
        meta = self.cached_get_obj_meta(type)
        if meta is None:
            raise ValueError("can't reload this type", type)
        # determine should perform real reload
        path = inspect.getfile(type)

        self.check_file_cache(path)
        if path in self.file_cache:
            # no need to reload.
            return self.module_cache[path], meta
        # invalid type method cache
        new_type_method_meta_cache = {}
        for t, vv in self.type_method_meta_cache.items():
            try:
                patht = inspect.getfile(t)
            except:
                continue
            if patht != path:
                new_type_method_meta_cache[t] = vv
        self.type_method_meta_cache = new_type_method_meta_cache
        # do reload
        res = meta.get_reloaded_module()
        if res is None:
            raise ValueError("can't reload this type", type)
        self.module_cache[path] = ModuleCacheEntry(res[1], res[0])
        # new_type = meta.get_local_type_from_module_dict(res[0])
        # type_cache_entry = TypeCacheEntry(new_type, res[1], res[0])
        # self.type_cache[type] = type_cache_entry
        self._update_file_cache(path)
        return self.module_cache[path], meta

    def query_type_method_meta(self, type):
        path = inspect.getfile(type)
        if type in self.type_method_meta_cache:
            return self.type_method_meta_cache[type]
        qualname_to_code = None
        # use qualname_to_code from ast to resolve some problem
        # if we just use inspect, inspect will use newest code
        # qualname_to_code always use code stored in manager.
        if path in self.file_cache:
            qualname_to_code = self.file_cache[path].qualname_to_code
        else:
            try:
                with tokenize.open(path) as f:
                    lines = f.readlines()
                qualname_to_code = get_qualname_to_code(lines)
            except:
                pass
        new_metas = ReloadableDynamicClass.get_metas_of_regular_methods(
            type, include_base=False, qualname_to_code=qualname_to_code)
        self.type_method_meta_cache[type] = new_metas
        return new_metas


class DynamicClass:

    def __init__(
        self,
        module_name: str,
    ) -> None:
        self.module_name = module_name
        module_cls = module_name.split(TENSORPC_SPLIT)
        self.module_path = module_cls[0]
        self.alias: Optional[str] = None
        self.is_standard_module = False
        self.standard_module: Optional[types.ModuleType] = None
        self.file_path = ""
        if len(module_cls) == 3:
            self.alias = module_cls[-1]
            self.cls_name = module_cls[-2]
        else:
            self.cls_name = module_cls[-1]
        try:
            if self.module_path.startswith("!"):
                self.module_path = self.module_path[1:]
                self.file_path = self.module_path
                assert Path(self.module_path).exists(
                ), f"your {self.module_path} not exists"
                # treat module_path as a file path
                # import sys
                mod_name = Path(self.module_path).stem + "_" + uuid.uuid4().hex
                spec = importlib.util.spec_from_file_location(
                    mod_name, self.module_path)
                assert spec is not None, f"your {self.module_path} not exists"
                self.standard_module = importlib.util.module_from_spec(spec)
                assert spec.loader is not None, "shouldn't happen"
                spec.loader.exec_module(self.standard_module)
                sys.modules[mod_name] = self.standard_module
                self.module_dict = self.standard_module.__dict__
                self.is_standard_module = False
                # self.module_dict = runpy.run_path(self.module_path)
            else:
                self.standard_module = importlib.import_module(
                    self.module_path)
                file_path = inspect.getfile(self.standard_module)
                assert file_path is not None, f"don't support compiled library, {file_path} must be .py"
                self.file_path = file_path
                self.module_dict = self.standard_module.__dict__
                self.is_standard_module = True
        except ImportError:
            print(f"Can't Import {module_name}. Check your project or PWD")
            raise
        obj_type = self.module_dict[self.cls_name]
        self.obj_type = obj_type
        self.module_key = f"{self.module_path}{TENSORPC_SPLIT}{self.cls_name}"


class ReloadableDynamicClass(DynamicClass):

    def __init__(self,
                 module_name: str,
                 reload_mgr: Optional[ObjectReloadManager] = None) -> None:
        super().__init__(module_name)
        if reload_mgr is not None:
            self.serv_metas = reload_mgr.query_type_method_meta(self.obj_type)
        else:
            self.serv_metas = self.get_metas_of_regular_methods(self.obj_type)

    @staticmethod
    def get_metas_of_regular_methods(
            type_obj,
            include_base: bool = False,
            qualname_to_code: Optional[Dict[str, str]] = None):
        serv_metas: List[ServFunctionMeta] = []
        members = inspecttools.get_members_by_type(type_obj, not include_base)
        for k, v in members:
            if inspecttools.isclassmethod(v) or inspecttools.isproperty(v):
                # ignore property and classmethod
                continue
            if k.startswith("__") and k.endswith("__"):
                # ignore all magic methods
                continue
            is_gen = inspect.isgeneratorfunction(v)
            is_async_gen = inspect.isasyncgenfunction(v)
            is_async = inspect.iscoroutinefunction(v) or is_async_gen
            is_static = inspecttools.isstaticmethod(type_obj, k)
            if is_async:
                is_gen = is_async_gen
            v_static = inspect.getattr_static(type_obj, k)
            app_meta: Optional[AppFunctionMeta] = None
            if hasattr(v_static, TENSORPC_FLOW_FUNC_META_KEY):
                app_meta = getattr(v_static, TENSORPC_FLOW_FUNC_META_KEY)

            v_sig = inspect.signature(v)
            serv_meta = ServFunctionMeta(v,
                                         k,
                                         ServiceType.Normal,
                                         v_sig,
                                         is_gen,
                                         is_async,
                                         is_static,
                                         False,
                                         qualname=v.__qualname__,
                                         user_app_meta=app_meta)
            code = None
            if qualname_to_code is not None:
                # TODO check mro
                if v.__qualname__ in qualname_to_code:
                    code = [qualname_to_code[v.__qualname__]]
            if code is None:
                code, _ = inspect.getsourcelines(v)
            serv_meta.code = "".join(code)
            serv_metas.append(serv_meta)
        return serv_metas

    def get_object_creator_if_exists(self):
        for m in self.serv_metas:
            if m.user_app_meta is not None:
                if m.user_app_meta.type == AppFuncType.CreateObject:
                    return m.fn
        return None

    def reload_obj_methods(self,
                           obj,
                           callback_dict: Optional[Dict[str, Any]] = None,
                           reload_mgr: Optional[ObjectReloadManager] = None):
        # reload regular methods/static methods
        new_cb: Dict[str, Any] = {}
        if callback_dict is None:
            callback_dict = {}
        callback_inv_dict = {v: k for k, v in callback_dict.items()}
        if reload_mgr is not None:
            res = reload_mgr.reload_type(type(obj))
            self.module_dict = res[0].module_dict
        else:
            if self.is_standard_module and self.standard_module is not None:
                importlib.reload(self.standard_module)
                self.module_dict = self.standard_module.__dict__
            else:
                self.module_dict = runpy.run_path(self.module_path)
        new_obj_type = self.module_dict[self.cls_name]
        if reload_mgr is not None:
            new_metas = reload_mgr.query_type_method_meta(new_obj_type)
        else:
            new_metas = self.get_metas_of_regular_methods(new_obj_type)
        # new_name_to_meta = {m.name: m for m in new_metas}
        name_to_meta = {m.name: m for m in self.serv_metas}
        code_changed_cb: List[str] = []
        code_changed_metas: List[ServFunctionMeta] = []
        for new_meta in new_metas:
            new_method = new_meta.bind(obj)
            # if not new_meta.is_static:
            #     new_method =  types.MethodType(new_meta.fn, obj)
            # else:
            #     new_method = new_meta.fn
            if new_meta.name in name_to_meta:
                meta = name_to_meta[new_meta.name]
                method = getattr(obj, meta.name)
                setattr(obj, new_meta.name, new_method)
                if method in callback_inv_dict:
                    new_cb[callback_inv_dict[method]] = new_method
                if new_meta.code != meta.code:
                    code_changed_cb.append(new_meta.qualname)
                    code_changed_metas.append(new_meta)
            else:
                setattr(obj, new_meta.name, new_method)
        self.serv_metas = new_metas
        self.obj_type = new_obj_type

        return new_cb, code_changed_metas


def get_cls_obj_from_module_name(module_name: str):
    """x.y.z::Class::Alias
    x.y.z::Class
    x.y.z::Class.f1
    Alias.f1
    !/your/path/to/module.py::Class::Alias
    """
    rc = DynamicClass(module_name)
    return rc.obj_type, rc.alias, rc.module_key


class FunctionUserMeta:

    def __init__(self,
                 type: ServiceType,
                 event_name: str = "",
                 is_dynamic: bool = False) -> None:
        self.type = type
        self._event_name = event_name
        self.is_dynamic = is_dynamic

    @property
    def event_name(self):
        assert self._event_name != ""
        return self._event_name


class EventProvider:

    def __init__(self,
                 service_key: str,
                 event_name: str,
                 fn: Callable,
                 is_static: bool,
                 is_dynamic: bool = False) -> None:
        self.service_key = service_key
        self.event_name = event_name
        self.fn = fn
        self.is_static = is_static
        self.is_binded = False

        self.is_dynamic = is_dynamic

    def copy(self):
        return EventProvider(self.service_key, self.event_name, self.fn,
                             self.is_static, self.is_dynamic)


class ServiceUnit(DynamicClass):
    """x.y.z::Class::Alias
    x.y.z::Class
    x.y.z::Class.f1
    Alias.f1
    !/path/to/module.py::Class::Alias
    !C:\\path\\to\\module.py::Class::Alias
    """

    def __init__(self, module_name: str, config: Dict[str, Any]) -> None:
        super().__init__(module_name)
        assert config is not None, "please use {} in yaml if config is empty"
        # self.obj_type, self.alias, self.module_key = get_cls_obj_from_module_name(
        #     module_name)
        self.services: Dict[str, ServFunctionMeta] = {}
        self.exit_fn: Optional[Any] = None
        self._is_exit_fn_async: bool = False
        self.ws_onconn_fn: Optional[Callable[[Any], None]] = None
        self.async_init: Optional[Callable[[], Coroutine[None, None,
                                                         None]]] = None
        self.ws_ondisconn_fn: Optional[Callable[[Any], None]] = None
        self.name_to_events: Dict[str, EventProvider] = {}
        self.serv_metas = self._init_all_metas(self.obj_type)
        assert len(
            self.services
        ) > 0, f"your service {module_name} must have at least one valid method"
        # self.obj = self.obj_type(**config)
        self.obj: Optional[Any] = None
        self.config = config

    def reload_metas(self, reload_mgr: Optional[ObjectReloadManager] = None):
        # reload regular methods/static methods
        new_cb: Dict[str, Any] = {}
        if reload_mgr is not None:
            res = reload_mgr.reload_type(type(self.obj))
            self.module_dict = res[0].module_dict
        else:
            if self.is_standard_module and self.standard_module is not None:
                importlib.reload(self.standard_module)
                self.module_dict = self.standard_module.__dict__
            else:
                self.module_dict = runpy.run_path(self.module_path)
        new_obj_type = self.module_dict[self.cls_name]
        new_metas = self._init_all_metas(new_obj_type)
        self.serv_metas = new_metas
        self.obj_type = new_obj_type
        self.init_service(rebind=True)
        return new_cb

    def _init_all_metas(self, obj_type) -> List[ServFunctionMeta]:
        members = inspecttools.get_members_by_type(obj_type, False)
        self.services.clear()
        self.name_to_events.clear()
        new_metas: List[ServFunctionMeta] = []
        for k, v in members:
            if inspecttools.isclassmethod(v) or inspecttools.isproperty(v):
                # ignore property and classmethod
                continue
            if k.startswith("_"):
                # ignore all protected, private and magic methods
                continue
            serv_key = f"{self.module_key}.{k}"

            serv_type = ServiceType.Normal
            is_gen = inspect.isgeneratorfunction(v)
            is_async_gen = inspect.isasyncgenfunction(v)
            is_async = inspect.iscoroutinefunction(v) or is_async_gen
            is_static = inspecttools.isstaticmethod(obj_type, k)
            if is_async:
                is_gen = is_async_gen
            # TODO Why?
            v_static = inspect.getattr_static(obj_type, k)
            v_sig = inspect.signature(v)
            ev_provider: Optional[EventProvider] = None
            if hasattr(v_static, TENSORPC_FUNC_META_KEY):
                # for special methods
                meta: FunctionUserMeta = getattr(v_static,
                                                 TENSORPC_FUNC_META_KEY)
                # meta: FunctionUserMeta = inspect.getattr_static(v, TENSORPC_FUNC_META_KEY)
                serv_type = meta.type
                if serv_type == ServiceType.WebSocketEventProvider:
                    num_parameters = len(
                        v_sig.parameters) - (0 if is_static else 1)
                    msg = f"event can't have any parameter, but {serv_key} have {num_parameters} param"
                    assert num_parameters == 0, msg
                    assert is_async and not is_async_gen, "event provider must be async function"
                    # self.events.append(EventProvider(serv_key, meta.event_name, v, is_static))
                    ev_provider = EventProvider(serv_key, meta.event_name, v,
                                                is_static, meta.is_dynamic)
            app_meta: Optional[AppFunctionMeta] = None
            if hasattr(v_static, TENSORPC_FLOW_FUNC_META_KEY):
                app_meta = getattr(v_static, TENSORPC_FLOW_FUNC_META_KEY)

            if serv_type == ServiceType.Exit:
                assert self.exit_fn is None, "you can only register one exit"
                self.exit_fn = v
                self._is_exit_fn_async = is_async
            if serv_type == ServiceType.AsyncInit:
                assert self.async_init is None, "you can only register one exit"
                self.async_init = v
            if serv_type == ServiceType.WebSocketOnConnect:
                assert self.ws_onconn_fn is None, "you can only register one ws_onconn_fn"
                self.ws_onconn_fn = v
            if serv_type == ServiceType.WebSocketOnDisConnect:
                assert self.ws_ondisconn_fn is None, "you can only register one ws_onconn_fn"
                self.ws_ondisconn_fn = v

            serv_meta = ServFunctionMeta(v,
                                         k,
                                         serv_type,
                                         v_sig,
                                         is_gen,
                                         is_async,
                                         is_static,
                                         False,
                                         user_app_meta=app_meta)
            # if module_name == "distflow.services.for_test:Service2:Test3" and k == "client_stream":
            #     print(dir(v))
            # print(module_name, serv_meta, is_async, is_async_gen)
            new_metas.append(serv_meta)
            assert serv_key not in self.services
            self.services[serv_key] = serv_meta
            if ev_provider is not None:
                self.name_to_events[serv_key] = ev_provider
            if self.alias is not None:
                alias_key = f"{self.alias}.{k}"
                assert alias_key not in self.services
                self.services[alias_key] = serv_meta
                if ev_provider is not None:
                    ev_provider.service_key = alias_key
                    self.name_to_events[alias_key] = ev_provider
        return new_metas

    def init_service(self,
                     external_obj: Optional[Any] = None,
                     rebind: bool = False):
        # lazy init
        if self.obj is None or rebind:
            if not rebind:
                if external_obj is not None:
                    self.obj = external_obj
                else:
                    self.obj = self.obj_type(**self.config)
            else:
                assert self.obj is not None
            if self.exit_fn is not None:
                self.exit_fn = types.MethodType(self.exit_fn, self.obj)
            if self.async_init is not None:
                self.async_init = types.MethodType(self.async_init, self.obj)
            if self.ws_onconn_fn is not None:
                self.ws_onconn_fn = types.MethodType(self.ws_onconn_fn,
                                                     self.obj)
            if self.ws_ondisconn_fn is not None:
                self.ws_ondisconn_fn = types.MethodType(
                    self.ws_ondisconn_fn, self.obj)
            for k, meta in self.services.items():
                # bind fn if not static
                if not meta.is_static and not meta.is_binded:
                    meta.fn = types.MethodType(meta.fn, self.obj)
                    meta.is_binded = True
            for ev in self.name_to_events.values():
                if not ev.is_static and not ev.is_binded:
                    new_fn = types.MethodType(ev.fn, self.obj)
                    ev.fn = new_fn
                    ev.is_binded = True

    def get_all_event_providers(self):
        return self.name_to_events

    def get_service_unit_ids(self):
        if self.alias is not None:
            return [self.module_key, self.alias]
        else:
            return [self.module_key]

    def get_service_ids(self):
        return list(self.services.keys())

    def get_service_metas_json(self):
        return {k: v.to_json() for k, v in self.services.items()}

    def get_service_and_meta(self, serv_key: str):
        self.init_service()
        meta = self.services[serv_key]
        return meta.fn, meta

    def run_service(self, serv_key: str, *args, **kwargs):
        self.init_service()
        assert self.obj is not None
        fn = self.services[serv_key].fn
        return fn(*args, **kwargs)

    def run_service_from_fn(self, fn: Callable, *args, **kwargs):
        self.init_service()
        assert self.obj is not None
        return fn(*args, **kwargs)

    def websocket_onconnect(self, client):
        if self.ws_onconn_fn is not None:
            self.ws_onconn_fn(client)

    def websocket_ondisconnect(self, client):
        if self.ws_ondisconn_fn is not None:
            self.ws_ondisconn_fn(client)

    async def run_async_init(self):
        if self.async_init is not None:
            await self.async_init()

    async def run_exit(self):
        if self.exit_fn is not None:
            if self._is_exit_fn_async:
                await self.exit_fn()
            else:
                self.exit_fn()

    def run_exit_sync(self):
        if self.exit_fn is not None:
            if self._is_exit_fn_async:
                return
            else:
                self.exit_fn()


class ServiceUnits:

    def __init__(self, sus: List[ServiceUnit]) -> None:
        self.sus = sus
        self.key_to_su: Dict[str, ServiceUnit] = {}
        self._service_id_to_key: Dict[int, str] = {}
        unique_module_ids: Set[str] = set()
        cnt: int = 0
        for su in sus:
            for suid in su.get_service_unit_ids():
                assert suid not in self.key_to_su
                unique_module_ids.add(suid)
            for sid in su.get_service_ids():
                self.key_to_su[sid] = su
                self._service_id_to_key[cnt] = sid
                cnt += 1

    def init_service(self):
        for s in self.sus:
            s.init_service()

    def get_service_and_meta(self, serv_key: str):
        return self.key_to_su[serv_key].get_service_and_meta(serv_key)

    def get_service_id_to_name(self):
        return self._service_id_to_key

    def get_service(self, serv_key: str):
        return self.key_to_su[serv_key].get_service_and_meta(serv_key)[0]

    def run_service(self, serv_key: str, *args, **kwargs):
        return self.get_service(serv_key)(*args, **kwargs)

    async def run_exit(self):
        for s in self.sus:
            await s.run_exit()

    def run_exit_sync(self):
        for s in self.sus:
            s.run_exit_sync()

    def get_all_service_metas_json(self):
        return {su.module_key: su.get_service_metas_json() for su in self.sus}

    def websocket_onconnect(self, client):
        for s in self.sus:
            s.websocket_onconnect(client)

    def websocket_ondisconnect(self, client):
        for s in self.sus:
            s.websocket_ondisconnect(client)

    def get_all_event_providers(self):
        res: Dict[str, EventProvider] = {}
        for su in self.sus:
            res.update(su.get_all_event_providers())
        return res

    async def run_async_init(self):
        self.init_service()
        for s in self.sus:
            await s.run_async_init()
