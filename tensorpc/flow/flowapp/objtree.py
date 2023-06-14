""" Object Tree

Object tree is a tree structure of objects, which can be used to store
objects in a tree structure, and can be used to find objects by type.

It's used to provide loose coupling between objects, and can be easily
intergrated into GUI TreeView.

Object Inspector in tensorpc supports UserObjTree natively. if you want to
add new object type to Object Inspector, you need to register it.


"""

import asyncio
import contextlib
import contextvars
import inspect
from typing import (Any, Callable, Coroutine, Dict, Generator, Iterator, List, Optional, Protocol,
                    Type, TypeVar, Union)

from typing_extensions import ContextManager

T = TypeVar("T")

def get_qualname_of_type(klass: Type) -> str:
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + '.' + klass.__qualname__


class ObjTreeContextProtocol(Protocol):
    node: "UserObjTreeProtocol"


class UserObjTreeProtocol(Protocol):

    def get_childs(self) -> Dict[str, Union[Any, "UserObjTreeProtocol"]]:
        ...

    def enter_context(
            self, node: "UserObjTreeProtocol"
    ) -> ContextManager["ObjTreeContextProtocol"]:
        ...

    
    def update_tree(self) -> None:
        ...

    async def update_tree_async(self) -> None:
        ...

    def attach_update_tree_callback(self, func: Callable[[], Union[Coroutine[None, None, None], None]]):
        ...

class ObjTreeContext:

    def __init__(self, node: "UserObjTreeProtocol") -> None:
        self.node = node


T_treeitem = TypeVar("T_treeitem", bound=UserObjTreeProtocol)

OBJ_TREE_CONTEXT_VAR: contextvars.ContextVar[
    Optional[ObjTreeContextProtocol]] = contextvars.ContextVar(
        "objtree_context", default=None)


def get_objtree_context() -> Optional[ObjTreeContextProtocol]:
    return OBJ_TREE_CONTEXT_VAR.get()


class UserObjTree:

    def __init__(self) -> None:
        self._childs: Dict[str, Union[Any, "UserObjTreeProtocol"]] = {}
        self._objtree_update_tree_callback: Optional[Callable[[], Union[Coroutine[None, None, None], None]]] = None

    def get_childs(self) -> Dict[str, Union[Any, "UserObjTreeProtocol"]]:
        return self._childs

    @contextlib.contextmanager
    def enter_context(
        self, node: "UserObjTreeProtocol"
    ) -> Generator["ObjTreeContextProtocol", None, None]:
        ctx = ObjTreeContext(node)
        token = OBJ_TREE_CONTEXT_VAR.set(ctx)
        try:
            yield ctx
        finally:
            OBJ_TREE_CONTEXT_VAR.reset(token)

    def find(self, obj_type: Type[T]) -> T:
        """find a child object of current context node by type of obj.
        if not exist, raise an error.
        """
        return find(obj_type)

    def find_may_exist(self, obj_type: Type[T]) -> Optional[T]:
        """find a child object of current context node by type of obj.
        if not exist, return None.
        """
        return find_may_exist(obj_type)
    
    def update_tree(self):
        # TODO if we run in executor, we need to get loop in main thread.
        asyncio.run_coroutine_threadsafe(self.update_tree_async(), asyncio.get_running_loop())

    async def update_tree_async(self):
        if self._objtree_update_tree_callback is not None:
            res = self._objtree_update_tree_callback()
            if inspect.iscoroutine(res):
                await res

    def attach_update_tree_callback(self, func: Callable[[], Union[Coroutine[None, None, None], None]]):
        self._objtree_update_tree_callback = func


def find_tree_child_item_may_exist(root: UserObjTreeProtocol, obj_type: Type[T],
                                   node_type: Type[T_treeitem], 
                                   validator: Optional[Callable[[T], bool]] = None) -> Optional[T]:
    obj_type_qname = get_qualname_of_type(obj_type)

    childs_dict = root.get_childs()
    res_foreach: List[UserObjTreeProtocol] = []

    for k, v in childs_dict.items():
        v_type_qname = get_qualname_of_type(type(v))
        if v_type_qname == obj_type_qname:
            if validator is None or (validator is not None and validator(v)): # type: ignore
                return v # type: ignore
        if isinstance(v, node_type):
            res_foreach.append(v)
            # res = find_tree_child_item_may_exist(v, obj_type, node_type)
            # if res is not None:
            #     return res
    for v in res_foreach:
        res = find_tree_child_item_may_exist(v, obj_type, node_type)
        if res is not None:
            return res
    return None


def _get_tree_child_items_recursive(root: UserObjTreeProtocol, obj_type: Type[T],
                         node_type: Type[T_treeitem],
                         validator: Optional[Callable[[T], bool]] = None) -> List[T]:
    childs_dict = root.get_childs()
    res: List[T] = []
    # we use qualname to compare type, because type may be different
    # when we reload module which is quite often in GUI.
    obj_type_qname = get_qualname_of_type(obj_type)
    for k, v in childs_dict.items():
        v_type_qname = get_qualname_of_type(type(v))
        if v_type_qname == obj_type_qname:
            if validator is None or (validator is not None and validator(v)): # type: ignore
                res.append(v) # type: ignore
        elif isinstance(v, node_type):
            res.extend(_get_tree_child_items_recursive(v, obj_type, node_type))
    return res

def _check_node(node: UserObjTreeProtocol, obj_type: Type[T], validator: Optional[Callable[[T], bool]] = None):
    obj_type_qname = get_qualname_of_type(obj_type)
    # check root
    v_type_qname = get_qualname_of_type(type(node))
    if v_type_qname == obj_type_qname:
        if validator is None or (validator is not None and validator(node)): # type: ignore
            return True
    return False

def get_tree_child_items(root: UserObjTreeProtocol, obj_type: Type[T],
                         node_type: Type[T_treeitem],
                         validator: Optional[Callable[[T], bool]] = None) -> List[T]:
    res: List[T] = []
    if _check_node(root, obj_type, validator):
        res.append(root) # type: ignore
    res += _get_tree_child_items_recursive(root, obj_type, node_type, validator)
    return res

def find_tree_child_item(root: UserObjTreeProtocol, obj_type: Type[T],
                         node_type: Type[T_treeitem],
                         validator: Optional[Callable[[T], bool]] = None) -> T:
    res = find_tree_child_item_may_exist(root, obj_type, node_type, validator)
    assert res is not None, f"can't find type {obj_type} in root."
    return res


def find(obj_type: Type[T], validator: Optional[Callable[[T], bool]] = None) -> T:
    """find a child object of current context node by type of obj.
    if not exist, raise an error.
    """
    ctx = get_objtree_context()
    assert ctx is not None
    if _check_node(ctx.node, obj_type, validator):
        return ctx.node # type: ignore

    return find_tree_child_item(ctx.node, obj_type, UserObjTree, validator)


def find_may_exist(obj_type: Type[T], validator: Optional[Callable[[T], bool]] = None) -> Optional[T]:
    """find a child object of current context node by type of obj.
    if not exist, return None.
    """
    ctx = get_objtree_context()
    if ctx is None:
        return None 
    if _check_node(ctx.node, obj_type, validator):
        return ctx.node # type: ignore
    # assert ctx is not None
    return find_tree_child_item_may_exist(ctx.node, obj_type, UserObjTree, validator)
