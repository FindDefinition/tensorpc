import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.dock.components.mui.editor import MonacoRange
from tensorpc.apps.adv.model import ADVNodeHandle, ADVNodeModel
from typing import Any, Optional, Self, Union
import abc 


@dataclasses.dataclass
class ImplCodeSpec:
    lines: list[str]
    lineno_offset: int 
    column: int 
    num_lines: int
    end_column: int

    @property 
    def end_lineno_offset(self):
        assert self.lineno_offset > 0 and self.num_lines > 0 
        return self.lineno_offset + self.num_lines - 1


@dataclasses.dataclass(kw_only=True)
class BaseParseResult:
    # root flow don't have parent node, so this can be None.
    node: Optional[ADVNodeModel]
    succeed: bool = True
    error_msg: str = ""
    inline_error_msgs: list[tuple[MonacoRange, str]] = dataclasses.field(default_factory=list)
    lineno: int = -1
    loc: Optional[ImplCodeSpec] = None

    @staticmethod
    def get_node_meta_kwargs(node: ADVNodeModel) -> list[str]:
        # most of nodes in a flow needs to serialize their position and id.
        # e.g. `@ADV.mark_symbol_def(node_id=..., position=[100, 200], ref_node_id=...)`
        # this function is used to generate kwarg strings.
        assert node is not None 
        position_tuple = (node.position.x, node.position.y)
        id_str = node.id 

        res = [
            f'node_id="{id_str}"',
            f'position={position_tuple}',
        ]
        if node.ref_node_id is not None:
            res.append(f'ref_node_id="{node.ref_node_id}"')
        return res

    def to_code_lines(self, id_to_parse_res: dict[str, "BaseParseResult"]) -> ImplCodeSpec:
        raise NotImplementedError

    def get_global_loc(self) -> ImplCodeSpec:
        assert self.lineno > 0 and self.loc is not None
        return dataclasses.replace(
            self.loc,
            lineno_offset=self.lineno + self.loc.lineno_offset,
        )


@dataclasses.dataclass(kw_only=True)
class BackendHandle:
    handle: ADVNodeHandle
    # we want to keep the order of handles
    index: int 
     # list of (node_id, handle_id) except edges that connect to output indicators
    target_node_handle_id: set[tuple[str, str]] = dataclasses.field(default_factory=set)
    is_inlineflow_out: bool = False
    # this store all qualified names that this handle type depend on.
    # e.g. list[torch.Tensor] will have ["torch.Tensor"]
    type_dep_qnames: list[str] = dataclasses.field(default_factory=list)

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
    ref_node_id: Optional[str] = None

@dataclasses.dataclass 
class RefNodeMeta:
    id: str 
    ref_node_id: str
    position: tuple[Union[int, float], Union[int, float]]
    alias_map: str = ""
    is_local_ref: bool = False


class BaseParser:
    pass 

