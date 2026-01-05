import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.dock.components.mui.editor import MonacoRange
from tensorpc.apps.adv.model import ADVNodeHandle
from typing import Any, Optional, Self


@dataclasses.dataclass(kw_only=True)
class BaseParseResult:
    succeed: bool
    error_msg: str = ""
    inline_error_msgs: list[tuple[MonacoRange, str]] = dataclasses.field(default_factory=list)

@dataclasses.dataclass(kw_only=True)
class BackendHandle:
    handle: ADVNodeHandle
    # we want to keep the order of handles
    index: int 
     # list of (node_id, handle_id) except edges that connect to output indicators
    target_node_handle_id: set[tuple[str, str]] = dataclasses.field(default_factory=set)

    is_subflow_output: bool = False

    @property 
    def symbol_name(self) -> str:
        return self.handle.symbol_name

    @property 
    def id(self) -> str:
        return self.handle.id

    def copy(self, node_id: Optional[str] = None, offset: Optional[int] = None, is_sym_handle: bool = False, prefix: Optional[str] = None) -> Self:
        if node_id is None:
            node_id = self.handle.source_node_id
        if offset is None:
            offset = 0
        new_id = self.handle.id
        if prefix is not None:
            new_id_no_prefix = "-".join(new_id.split("-", 1)[1:])
            new_id = f"{prefix}-{new_id_no_prefix}"
        return dataclasses.replace(
            self,
            handle=dataclasses.replace(
                self.handle,
                id=new_id,
                source_node_id=node_id,
                is_sym_handle=is_sym_handle,
            ),
            index=self.index + offset,
            target_node_handle_id=[],
        )

    def rename_symbol(self, new_name: str) -> Self:
        new_handle = dataclasses.replace(
            self.handle,
            symbol_name=new_name,
        )
        return dataclasses.replace(
            self,
            handle=new_handle,
        )

@dataclasses.dataclass 
class BaseNodeCodeMeta:
    id: str 
    position: tuple[float, float]
