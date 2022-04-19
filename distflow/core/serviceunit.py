import inspect
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum
import dataclasses
from pathlib import Path
import importlib
from distflow.constants import DISTFLOW_FUNC_META_KEY
import types
from distflow.core import inspecttools


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
    BiStream = "BidirectinoalStream"  # only support grpc for now
    ClientStream = "ClientStream"  # only support grpc for now
    AsyncWebSocket = "AsyncWebSocket"  # only support ws
    WebSocketOnConnect = "WebSocketOnConnect"  # only support ws
    WebSocketOnDisConnect = "WebSocketOnDisConnect"  # only support ws


@dataclasses.dataclass
class ServFunctionMeta:
    name: str
    type: ServiceType
    args: List[ParamMeta]
    is_gen: bool
    is_async: bool
    is_static: bool

    def __init__(self, name: str, type: ServiceType, sig: inspect.Signature,
                 is_gen: bool, is_async: bool, is_static: bool) -> None:
        self.name = name
        self.type = type
        self.args = [ParamMeta(n, p) for n, p in sig.parameters.items()]
        if not is_static:
            self.args = self.args[1:]
        self.is_gen = is_gen
        self.is_async = is_async
        self.is_static = is_static

    def to_json(self):
        return {
            "name": self.name,
            "type": self.type.value,
            "args": [a.to_json() for a in self.args],
            "is_gen": self.is_gen,
            "is_async": self.is_async,
            "is_static": self.is_static,
        }

def _get_cls_obj_from_module_name(module_name: str):
    module_cls = module_name.split(":")
    module_path = module_cls[0]
    cls_name = module_cls[1]
    alias: Optional[str] = None
    if len(module_cls) == 3:
        alias = module_cls[2]
    try:
        mod = importlib.import_module(module_path)
    except ImportError:
        print(f"Can't Import {module_name}. Check your project or PWD")
        raise
    cls_obj = mod.__dict__[cls_name]
    return cls_obj, alias, f"{module_path}:{cls_name}"


class FunctionUserMeta:
    def __init__(self, type: ServiceType) -> None:
        self.type = type


class ServiceUnit:
    """x.y.z:Class:Alias
    x.y.z:Class
    x.y.z:Class.f1
    Alias.f1
    """
    def __init__(self, module_name: str, config: Dict[str, Any]) -> None:
        assert config is not None, "please use {} in yaml if config is empty"
        self.obj_type, self.alias, self.module_key = _get_cls_obj_from_module_name(
            module_name)
        members = inspecttools.get_members_by_type(self.obj_type, False)
        self.services: Dict[str, Tuple[Callable, ServFunctionMeta]] = {}
        self.exit_fn: Optional[Callable[[], None]] = None
        self.ws_onconn_fn: Optional[Callable[[Any], None]] = None
        self.ws_ondisconn_fn: Optional[Callable[[], None]] = None

        for k, v in members:
            if inspecttools.isclassmethod(v) or inspecttools.isproperty(v):
                # ignore property and classmethod
                continue
            if k.startswith("__"):
                # ignore all private and magic methods
                continue
            serv_type = ServiceType.Normal
            is_gen = inspect.isgeneratorfunction(v)
            is_async_gen = inspect.isasyncgenfunction(v)
            is_async = inspect.iscoroutinefunction(v) or is_async_gen
            isstatic = inspecttools.isstaticmethod(self.obj_type, k)
            if is_async:
                is_gen = is_async_gen
            # TODO Why?
            v_static = inspect.getattr_static(self.obj_type, k)
            if hasattr(v_static, DISTFLOW_FUNC_META_KEY):
                # for special methods
                meta: FunctionUserMeta = getattr(v_static, DISTFLOW_FUNC_META_KEY)
                # meta: FunctionUserMeta = inspect.getattr_static(v, DISTFLOW_FUNC_META_KEY)
                serv_type = meta.type
            if serv_type == ServiceType.Exit:
                assert self.exit_fn is None, "you can only register one exit"
                self.exit_fn = v
            if serv_type == ServiceType.WebSocketOnConnect:
                assert self.ws_onconn_fn is None, "you can only register one ws_onconn_fn"
                self.ws_onconn_fn = v
            if serv_type == ServiceType.WebSocketOnDisConnect:
                assert self.ws_ondisconn_fn is None, "you can only register one ws_onconn_fn"
                self.ws_ondisconn_fn = v
            serv_meta = ServFunctionMeta(k, serv_type, inspect.signature(v),
                                         is_gen, is_async, isstatic)
            # if module_name == "distflow.services.for_test:Service2:Test3" and k == "client_stream":
            #     print(dir(v))
            # print(module_name, serv_meta, is_async, is_async_gen)

            serv_key = f"{self.module_key}.{k}"
            assert serv_key not in self.services
            self.services[serv_key] = (v, serv_meta)
            if self.alias is not None:
                alias_key = f"{self.alias}.{k}"
                assert alias_key not in self.services
                self.services[alias_key] = (v, serv_meta)
        assert len(
            self.services
        ) > 0, f"your service {module_name} must have at least one valid method"
        # self.obj = self.obj_type(**config)
        self.obj: Optional[Any] = None
        self.config = config

    def init_service(self):
        # lazy init
        if self.obj is None:
            self.obj = self.obj_type(**self.config)
            if self.exit_fn is not None:
                self.exit_fn = types.MethodType(self.exit_fn, self.obj)
            if self.ws_onconn_fn is not None:
                self.ws_onconn_fn = types.MethodType(self.ws_onconn_fn, self.obj)
            if self.ws_ondisconn_fn is not None:
                self.ws_ondisconn_fn = types.MethodType(self.ws_ondisconn_fn, self.obj)

            for k, (fn, meta) in self.services.items():
                # bind fn if not static
                if not meta.is_static:
                    new_fn = types.MethodType(fn, self.obj)
                    self.services[k] = (new_fn, meta)

    def get_service_unit_ids(self):
        if self.alias is not None:
            return [self.module_key, self.alias]
        else:
            return [self.module_key]

    def get_service_ids(self):
        return list(self.services.keys())

    def get_service_metas_json(self):
        return {k: v[1].to_json() for k, v in self.services.items()}

    def get_service_and_meta(self, serv_key: str):
        self.init_service()
        return self.services[serv_key]

    def run_service(self, serv_key: str, *args, **kwargs):
        self.init_service()
        assert self.obj is not None
        fn, _ = self.services[serv_key]
        return fn(*args, **kwargs)

    def run_service_from_fn(self, fn: Callable, *args, **kwargs):
        self.init_service()
        assert self.obj is not None
        return fn(*args, **kwargs)

    def websocket_onconnect(self, client):
        if self.ws_onconn_fn is not None:
            self.ws_onconn_fn(client)

    def websocket_ondisconnect(self):
        if self.ws_ondisconn_fn is not None:
            self.ws_ondisconn_fn()

    def run_exit(self):
        if self.exit_fn is not None:
            self.exit_fn()


class ServiceUnits:
    def __init__(self, sus: List[ServiceUnit]) -> None:
        self.sus = sus
        self.key_to_su: Dict[str, ServiceUnit] = {}
        unique_module_ids: Set[str] = set()
        for su in sus:
            for suid in su.get_service_unit_ids():
                assert suid not in self.key_to_su
                unique_module_ids.add(suid)
            for sid in su.get_service_ids():
                self.key_to_su[sid] = su

    def get_service_and_meta(self, serv_key: str):
        return self.key_to_su[serv_key].get_service_and_meta(serv_key)

    def get_service(self, serv_key: str):
        return self.key_to_su[serv_key].get_service_and_meta(serv_key)[0]

    def run_service(self, serv_key: str, *args, **kwargs):
        return self.get_service(serv_key)(*args, **kwargs)

    def run_exit(self):
        for s in self.sus:
            s.run_exit()

    def get_all_service_metas_json(self):
        return {su.module_key: su.get_service_metas_json() for su in self.sus}

    def websocket_onconnect(self, client):
        for s in self.sus:
            s.websocket_onconnect(client)

    def websocket_ondisconnect(self):
        for s in self.sus:
            s.websocket_ondisconnect()



if __name__ == "__main__":
    su = ServiceUnit("distflow.services.for_test:ServiceForTest:Test", {})
    print(su.run_service("Test.add", 1, 2))