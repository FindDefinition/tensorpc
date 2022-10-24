import asyncio
import threading

from tensorpc.core import serviceunit
from tensorpc.core.server_core import (get_server_context,
                                       is_in_server_context,
                                       get_global_context,
                                       is_in_global_context)
from tensorpc.core.defs import DynamicEvent, DynamicEvents


def get_server_exposed_props():
    return get_server_context().exposed_props


def get_exec_lock():
    return get_server_exposed_props().exec_lock


def get_service_units() -> serviceunit.ServiceUnits:
    return get_server_exposed_props().service_units


def get_shutdown_event() -> threading.Event:
    return get_server_exposed_props().shutdown_event


def get_async_shutdown_event() -> asyncio.Event:
    return get_server_exposed_props().async_shutdown_event


def is_json_call():
    """tell service whether rpc is a json call, used for support client 
    written in other language
    """
    return get_server_context().json_call


def get_service(key):
    get_service_func = get_server_exposed_props().service_units.get_service
    if get_service_func is None:
        raise ValueError("get service not available during startup")
    return get_service_func(key)


def get_current_service_key():
    return get_server_context().service_key


def get_local_url():
    return get_global_context().local_url


def get_server_meta():
    return get_global_context().server_meta


def get_server_grpc_port():
    return get_server_meta().port


def get_server_http_port():
    return get_server_meta().http_port


def has_http_client_session():
    return get_global_context().http_client_session is not None


def get_http_client_session():
    sess = get_global_context().http_client_session
    if sess is not None:
        return sess
    raise ValueError("only async server support global session")
