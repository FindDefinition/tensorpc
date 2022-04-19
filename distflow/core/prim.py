import asyncio
import threading

from distflow.core import serviceunit
from distflow.core.server_core import (get_server_context,
                                                 is_in_server_context)


def get_server_exposed_props():
    return get_server_context().exposed_props


def get_exec_lock():
    return get_server_exposed_props().exec_lock


def get_service_units() -> serviceunit.ServiceUnits:
    return get_server_exposed_props().service_units


def get_shutdown_event() -> threading.Event:
    return get_server_exposed_props().shutdown_event

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
    return get_server_exposed_props().local_url

def get_loop():
    loop = get_server_exposed_props().loop
    if loop is None:
        raise ValueError("you haven't provide any loop to service core")
    return loop

def run_coro(coro):
    """run coro in loop when you provide a loop to service core.
    should only be used when you schedule a ws task from grpc client.
    TODO only allow call this function from grpc client
    """
    loop = get_server_exposed_props().loop
    if loop is None:
        raise ValueError("you haven't provide any loop to service core")
    return asyncio.run_coroutine_threadsafe(coro, loop)
