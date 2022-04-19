from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from distflow.constants import DISTFLOW_FUNC_META_KEY
from distflow.core.serviceunit import FunctionUserMeta, ServiceType

def meta_decorator(func=None, meta: Optional[FunctionUserMeta] = None):
    if meta is None:
        raise ValueError("this shouldn't happen")

    def wrapper(func):
        if hasattr(func, DISTFLOW_FUNC_META_KEY):
            raise ValueError(
                "you can only use one meta decorator in a function.")
        setattr(func, DISTFLOW_FUNC_META_KEY, meta)
        return func
    if func is not None:
        return wrapper(func)
    else:
        return wrapper

def mark_exit(func=None):
    meta = FunctionUserMeta(ServiceType.Exit)
    return meta_decorator(func, meta)

def mark_client_stream(func=None):
    meta = FunctionUserMeta(ServiceType.ClientStream)
    return meta_decorator(func, meta)


def mark_bidirectional_stream(func=None):
    meta = FunctionUserMeta(ServiceType.BiStream)
    return meta_decorator(func, meta)


def mark_websocket_peer(func=None):
    meta = FunctionUserMeta(ServiceType.AsyncWebSocket)
    return meta_decorator(func, meta)


def mark_websocket_onconnect(func=None):
    meta = FunctionUserMeta(ServiceType.WebSocketOnConnect)
    return meta_decorator(func, meta)


def mark_websocket_ondisconnect(func=None):
    meta = FunctionUserMeta(ServiceType.WebSocketOnDisConnect)
    return meta_decorator(func, meta)
