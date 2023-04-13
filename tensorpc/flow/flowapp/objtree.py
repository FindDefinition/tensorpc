import contextlib
import contextvars
from typing import (Any, Dict, Generator, Iterator, List, Optional, Protocol,
                    Type, TypeVar, Union)

from typing_extensions import ContextManager

T = TypeVar("T")


class ObjTreeContextProtocol(Protocol):
    node: "ObjTreeProtocol"


class ObjTreeProtocol(Protocol):

    def get_childs(self) -> Dict[str, Union[Any, "ObjTreeProtocol"]]:
        ...

    def enter_context(
            self, node: "ObjTreeProtocol"
    ) -> ContextManager["ObjTreeContextProtocol"]:
        ...


class ObjTreeContext:

    def __init__(self, node: "ObjTreeProtocol") -> None:
        self.node = node


T_treeitem = TypeVar("T_treeitem", bound=ObjTreeProtocol)

OBJ_TREE_CONTEXT_VAR: contextvars.ContextVar[
    Optional[ObjTreeContextProtocol]] = contextvars.ContextVar(
        "objtree_context", default=None)


def get_objtree_context() -> Optional[ObjTreeContextProtocol]:
    return OBJ_TREE_CONTEXT_VAR.get()


class ObjTree:

    def __init__(self) -> None:
        self.childs: Dict[str, Union[Any, "ObjTreeProtocol"]] = {}

    def get_childs(self) -> Dict[str, Union[Any, "ObjTreeProtocol"]]:
        return self.childs

    @contextlib.contextmanager
    def enter_context(
        self, node: "ObjTreeProtocol"
    ) -> Generator["ObjTreeContextProtocol", None, None]:
        ctx = ObjTreeContext(node)
        token = OBJ_TREE_CONTEXT_VAR.set(ctx)
        try:
            yield ctx
        finally:
            OBJ_TREE_CONTEXT_VAR.reset(token)


def find_tree_child_item_may_exist(root: ObjTreeProtocol, obj_type: Type[T],
                                   node_type: Type[T_treeitem]) -> Optional[T]:
    childs_dict = root.get_childs()
    for k, v in childs_dict.items():
        if isinstance(v, obj_type):
            return v
        if isinstance(v, node_type):
            res = find_tree_child_item_may_exist(v, obj_type, node_type)
            if res is not None:
                return res
    return None


def get_tree_child_items(root: ObjTreeProtocol, obj_type: Type[T],
                         node_type: Type[T_treeitem]) -> List[T]:
    childs_dict = root.get_childs()
    res: List[T] = []
    for k, v in childs_dict.items():
        if isinstance(v, obj_type):
            res.append(v)
        elif isinstance(v, node_type):
            res.extend(get_tree_child_items(v, obj_type, node_type))
    return res


def find_tree_child_item(root: ObjTreeProtocol, obj_type: Type[T],
                         node_type: Type[T_treeitem]) -> T:
    res = find_tree_child_item_may_exist(root, obj_type, node_type)
    assert res is not None, f"can't find type {obj_type} in root."
    return res


def find(obj_type: Type[T]) -> T:
    ctx = get_objtree_context()
    assert ctx is not None
    return find_tree_child_item(ctx.node, obj_type, ObjTree)


def find_may_exist(obj_type: Type[T]) -> Optional[T]:
    ctx = get_objtree_context()
    assert ctx is not None
    return find_tree_child_item_may_exist(ctx.node, obj_type, ObjTree)
