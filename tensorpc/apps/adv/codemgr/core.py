import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.dock.components.mui.editor import MonacoRange

@dataclasses.dataclass(kw_only=True)
class BaseParseResult:
    succeed: bool
    error_msg: str = ""
    inline_error_msgs: list[tuple[MonacoRange, str]] = dataclasses.field(default_factory=list)
