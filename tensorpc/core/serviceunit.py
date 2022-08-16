import dataclasses
import importlib
import inspect
import runpy
import types
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from tensorpc.constants import TENSORPC_FUNC_META_KEY, TENSORPC_SPLIT
from tensorpc.core import inspecttools
from tensorpc.core.defs import DynamicEvent


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
    WebSocketEventProvider = "EventProvider"  # only support ws
    WebSocketOnConnect = "WebSocketOnConnect"  # only support ws
    WebSocketOnDisConnect = "WebSocketOnDisConnect"  # only support ws


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
    code: str =""

    def __init__(self, fn: Callable, name: str, type: ServiceType,
                 sig: inspect.Signature, is_gen: bool, is_async: bool,
                 is_static: bool, is_binded: bool) -> None:
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

    def to_json(self):
        return {
            "name": self.name,
            "type": self.type.value,
            "args": [a.to_json() for a in self.args],
            "is_gen": self.is_gen,
            "is_async": self.is_async,
            "is_static": self.is_static,
            "is_binded": self.is_binded,
            "code": self.code,
        }

class DynamicClass:
    def __init__(self, module_name: str, ) -> None:
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
                # treat module_path as a file path
                self.module_path = self.module_path[1:]
                self.file_path = self.module_path
                assert Path(self.module_path).exists(), f"your {self.module_path} not exists"
                self.module_dict = runpy.run_path(self.module_path)
                self.is_standard_module = False
            else:
                self.standard_module = importlib.import_module(self.module_path)
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
        self.module_key =  f"{self.module_path}{TENSORPC_SPLIT}{self.cls_name}"


class ReloadableDynamicClass(DynamicClass):
    def __init__(self, module_name: str, ) -> None:
        super().__init__(module_name)
        self.serv_metas = self.get_metas_of_regular_methods(self.obj_type)

    @staticmethod
    def get_metas_of_regular_methods(type_obj):
        serv_metas: List[ServFunctionMeta] = []
        members = inspecttools.get_members_by_type(type_obj, True)
        for k, v in members:
            if inspecttools.isclassmethod(v) or inspecttools.isproperty(v):
                # ignore property and classmethod
                continue
            if k.startswith("__"):
                # ignore all magic methods
                continue
            is_gen = inspect.isgeneratorfunction(v)
            is_async_gen = inspect.isasyncgenfunction(v)
            is_async = inspect.iscoroutinefunction(v) or is_async_gen
            is_static = inspecttools.isstaticmethod(type_obj, k)
            if is_async:
                is_gen = is_async_gen
            v_sig = inspect.signature(v)
            serv_meta = ServFunctionMeta(v, k, ServiceType.Normal, v_sig, is_gen,
                                         is_async, is_static, False)
            code, _ = inspect.getsourcelines(v)
            serv_meta.code = "".join(code)
            serv_metas.append(serv_meta)
        return serv_metas


    def reload_obj_methods(self, obj, callback_dict: Optional[Dict[str, Any]] = None):
        # reload regular methods/static methods
        new_cb: Dict[str, Any] = {}
        if callback_dict is None:
            callback_dict = {}
        callback_inv_dict = {v: k for k, v in callback_dict.items()}
        if self.is_standard_module and self.standard_module is not None:
            importlib.reload(self.standard_module) 
            self.module_dict = self.standard_module.__dict__
        else:
            self.module_dict = runpy.run_path(self.module_path)
        new_obj_type = self.module_dict[self.cls_name]
        new_metas = self.get_metas_of_regular_methods(new_obj_type)
        # new_name_to_meta = {m.name: m for m in new_metas}
        name_to_meta = {m.name: m for m in self.serv_metas}

        for new_meta in new_metas:
            if not new_meta.is_static:
                new_method =  types.MethodType(new_meta.fn, obj)
            else:
                new_method = new_meta.fn
            if new_meta.name in name_to_meta:
                meta = name_to_meta[new_meta.name]
                method = getattr(obj, meta.name)
                setattr(obj, new_meta.name, new_method)
                if method in callback_inv_dict:
                    new_cb[callback_inv_dict[method]] = new_method
            else:
                setattr(obj, new_meta.name, new_method)
        self.serv_metas = new_metas
        self.obj_type = new_obj_type
        return new_cb


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
        self.ws_ondisconn_fn: Optional[Callable[[Any], None]] = None
        self.name_to_events: Dict[str, EventProvider] = {}
        self.serv_metas = self._init_all_metas(self.obj_type)
        assert len(
            self.services
        ) > 0, f"your service {module_name} must have at least one valid method"
        # self.obj = self.obj_type(**config)
        self.obj: Optional[Any] = None
        self.config = config

    def reload_metas(self):
        # reload regular methods/static methods
        new_cb: Dict[str, Any] = {}
        if self.is_standard_module and self.standard_module is not None:
            importlib.reload(self.standard_module) 
            self.module_dict = self.standard_module.__dict__
        else:
            self.module_dict = runpy.run_path(self.module_path)
        new_obj_type = self.module_dict[self.cls_name]
        new_metas = self._init_all_metas(new_obj_type)
        self.serv_metas = new_metas
        self.obj_type = new_obj_type
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
            if serv_type == ServiceType.Exit:
                assert self.exit_fn is None, "you can only register one exit"
                self.exit_fn = v
                self._is_exit_fn_async = is_async
            if serv_type == ServiceType.WebSocketOnConnect:
                assert self.ws_onconn_fn is None, "you can only register one ws_onconn_fn"
                self.ws_onconn_fn = v
            if serv_type == ServiceType.WebSocketOnDisConnect:
                assert self.ws_ondisconn_fn is None, "you can only register one ws_onconn_fn"
                self.ws_ondisconn_fn = v

            serv_meta = ServFunctionMeta(v, k, serv_type, v_sig, is_gen,
                                         is_async, is_static, False)
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
            

    def init_service(self, external_obj: Optional[Any] = None):
        # lazy init
        if self.obj is None:
            if external_obj is not None:
                self.obj = external_obj
            else:
                self.obj = self.obj_type(**self.config)
            if self.exit_fn is not None:
                self.exit_fn = types.MethodType(self.exit_fn, self.obj)
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


if __name__ == "__main__":
    su = ServiceUnit("tensorpc.services.for_test::Service1::Test", {})
    print(su.run_service("Test.add", 1, 2))

    cl = ReloadableDynamicClass("tensorpc.services.for_test::Service1::Test")
    obj = cl.obj_type()
    print(obj.add(1, 2))
    print(input("hold"))
    cl.reload_obj_methods(obj)
    print(obj.add(1, 2))
