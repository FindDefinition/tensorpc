import ast
import enum
from typing import Optional 

class UserNodeType(enum.Enum):
    Compute = "Compute"

TENSORPC_USER_NODE_META_KEY = "__tensorpc_user_node_meta"

class UserNodeMeta:
    def __init__(self, type: UserNodeType) -> None:
        self.type = type

def usernode_meta_decorator(func=None,
                   meta: Optional[UserNodeMeta] = None):
    if meta is None:
        raise ValueError("this shouldn't happen")

    def wrapper(func):
        if meta is None:
            raise ValueError("this shouldn't happen")
        if hasattr(func, TENSORPC_USER_NODE_META_KEY):
            raise ValueError(
                "you can only use one meta decorator in a function.")
        setattr(func, TENSORPC_USER_NODE_META_KEY, meta)
        return func

    if func is not None:
        return wrapper(func)
    else:
        return wrapper
