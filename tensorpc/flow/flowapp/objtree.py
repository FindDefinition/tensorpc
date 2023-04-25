""" Object Tree

Object tree is a tree structure of objects, which can be used to store
objects in a tree structure, and can be used to find objects by type.

It's used to provide loose coupling between objects, and can be easily
intergrated into GUI TreeView.

Object Inspector in tensorpc supports UserObjTree natively. if you want to
add new object type to Object Inspector, you need to register it.


"""

from asyncio import iscoroutine
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
        self.childs: Dict[str, Union[Any, "UserObjTreeProtocol"]] = {}
        self._objtree_update_tree_callback: Optional[Callable[[], Union[Coroutine[None, None, None], None]]] = None

    def get_childs(self) -> Dict[str, Union[Any, "UserObjTreeProtocol"]]:
        return self.childs

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
                                   node_type: Type[T_treeitem]) -> Optional[T]:
    childs_dict = root.get_childs()
    obj_type_qname = get_qualname_of_type(obj_type)
    for k, v in childs_dict.items():
        v_type_qname = get_qualname_of_type(type(v))
        if v_type_qname == obj_type_qname:
            return v # type: ignore
        if isinstance(v, node_type):
            res = find_tree_child_item_may_exist(v, obj_type, node_type)
            if res is not None:
                return res
    return None


def get_tree_child_items(root: UserObjTreeProtocol, obj_type: Type[T],
                         node_type: Type[T_treeitem]) -> List[T]:
    childs_dict = root.get_childs()
    res: List[T] = []
    # we use qualname to compare type, because type may be different
    # when we reload module which is quite often in GUI.
    obj_type_qname = get_qualname_of_type(obj_type)
    for k, v in childs_dict.items():
        v_type_qname = get_qualname_of_type(type(v))
        if v_type_qname == obj_type_qname:
            res.append(v) # type: ignore
        elif isinstance(v, node_type):
            res.extend(get_tree_child_items(v, obj_type, node_type))
    return res


def find_tree_child_item(root: UserObjTreeProtocol, obj_type: Type[T],
                         node_type: Type[T_treeitem]) -> T:
    res = find_tree_child_item_may_exist(root, obj_type, node_type)
    assert res is not None, f"can't find type {obj_type} in root."
    return res


def find(obj_type: Type[T]) -> T:
    """find a child object of current context node by type of obj.
    if not exist, raise an error.
    """
    ctx = get_objtree_context()
    assert ctx is not None
    return find_tree_child_item(ctx.node, obj_type, UserObjTree)


def find_may_exist(obj_type: Type[T]) -> Optional[T]:
    """find a child object of current context node by type of obj.
    if not exist, return None.
    """
    ctx = get_objtree_context()
    if ctx is None:
        return None 
    # assert ctx is not None
    return find_tree_child_item_may_exist(ctx.node, obj_type, UserObjTree)
