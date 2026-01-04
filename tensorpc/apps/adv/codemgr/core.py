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
    target_id_pairs: list[tuple[str, str]]  # list of (node_id, handle_id)

    @property 
    def symbol_name(self) -> str:
        return self.handle.symbol_name

    def copy(self, node_id: Optional[str] = None, offset: Optional[int] = None) -> Self:
        if node_id is None:
            node_id = self.handle.source_node_id
        if offset is None:
            offset = 0
        return dataclasses.replace(
            self,
            handle=dataclasses.replace(
                self.handle,
                source_node_id=node_id
            ),
            index=self.index + offset,
            target_id_pairs=[],
        )
