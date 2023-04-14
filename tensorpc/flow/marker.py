# Copyright 2022 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import enum
from typing import Callable, Optional
from tensorpc.constants import TENSORPC_FLOW_FUNC_META_KEY
from tensorpc.core.serviceunit import AppFuncType, AppFunctionMeta
import inspect


def meta_decorator(func=None,
                   meta: Optional[AppFunctionMeta] = None,
                   name: Optional[str] = None,
                   func_checker: Optional[Callable[[Callable], None]] = None):
    if meta is None:
        raise ValueError("this shouldn't happen")

    def wrapper(func):
        if meta is None:
            raise ValueError("this shouldn't happen")
        if func_checker:
            func_checker(func)
        if hasattr(func, TENSORPC_FLOW_FUNC_META_KEY):
            raise ValueError(
                "you can only use one meta decorator in a function.")
        if isinstance(func, staticmethod):
            name_ = func.__func__.__name__
        else:
            name_ = func.__name__
        if name is not None:
            name_ = name
        meta.name = name_

        setattr(func, TENSORPC_FLOW_FUNC_META_KEY, meta)
        return func

    if func is not None:
        return wrapper(func)
    else:
        return wrapper


def mark_autorun(func=None):
    meta = AppFunctionMeta(AppFuncType.AutoRun)
    return meta_decorator(func, meta)


def mark_create_layout(func=None):
    meta = AppFunctionMeta(AppFuncType.CreateLayout)
    return meta_decorator(func, meta)

def mark_create_preview_layout(func=None):
    meta = AppFunctionMeta(AppFuncType.CreatePreviewLayout)
    return meta_decorator(func, meta)


def mark_create_object(func=None):
    meta = AppFunctionMeta(AppFuncType.CreateObject)

    def func_checker(func):
        assert isinstance(
            func, staticmethod), "create_object decorator only works on static"
        assert len(inspect.signature(func.__func__).parameters
                   ) == 0, "create_object decorator can't have any params"

    return meta_decorator(func, meta, func_checker=func_checker)


def mark_run_in_executor(func=None):
    meta = AppFunctionMeta(AppFuncType.RunInExecutor)
    return meta_decorator(func, meta)


def mark_did_mount(func=None):
    meta = AppFunctionMeta(AppFuncType.ComponentDidMount)
    return meta_decorator(func, meta)


def mark_will_unmount(func=None):
    meta = AppFunctionMeta(AppFuncType.ComponentWillUnmount)
    return meta_decorator(func, meta)
