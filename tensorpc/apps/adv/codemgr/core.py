from tensorpc.apps.adv.constants import TENSORPC_ADV_FOLDER_FLOW_NAME
import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.dock.components.mui.editor import MonacoRange
from tensorpc.apps.adv.model import ADVHandleFlags, ADVNodeFlags, ADVNodeHandle, ADVNodeModel, ADVNodeRefInfo, ADVNodeType
from typing import Any, Optional, Self, TypeVar, Union
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
        if not (self.lineno_offset > 0 and self.num_lines > 0):
            return -1
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

    def get_node_checked(self) -> ADVNodeModel:
        assert self.node is not None, "node is None in parse result."
        return self.node

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
        if node.ref is not None:
            res.append(f'ref_node_id="{node.ref.node_id}"')
        return res

    def to_code_lines(self) -> ImplCodeSpec:
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
    def name(self) -> str:
        return self.handle.name

    @property 
    def symbol_name(self) -> str:
        return self.handle.symbol_name

    @property 
    def id(self) -> str:
        return self.handle.id

    def copy(self, node_id: Optional[str] = None, offset: Optional[int] = None, is_sym_handle: bool = False, prefix: Optional[str] = None) -> Self:
        source_info = self.handle.source_info
        if node_id is not None:
            assert source_info is not None
            source_info = dataclasses.replace(
                source_info,
                node_id=node_id,
            )
        if offset is None:
            offset = 0
        new_id = self.handle.id
        if prefix is not None:
            new_id_no_prefix = "-".join(new_id.split("-", 1)[1:])
            new_id = f"{prefix}-{new_id_no_prefix}"
        new_handle = dataclasses.replace(
            self.handle,
            id=new_id,
            source_info=source_info,
        )
        if is_sym_handle:
            new_handle.flags |= int(ADVHandleFlags.IS_SYM_HANDLE)
        else:
            new_handle.flags &= ~int(ADVHandleFlags.IS_SYM_HANDLE)
        return dataclasses.replace(
            self,
            handle=new_handle,
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

    @property 
    def is_method_self(self) -> bool:
        return self.handle.flags & ADVHandleFlags.IS_METHOD_SELF != 0

@dataclasses.dataclass 
class BaseNodeCodeMeta:
    id: str 
    position: tuple[float, float]
    ref_node_id: Optional[str] = None

@dataclasses.dataclass 
class RefNodeMeta:
    id: str 
    ref_node_id: str
    flags: int
    position: tuple[Union[int, float], Union[int, float]]
    alias_map: str = ""
    is_local_ref: bool = False


class BaseParser:
    pass 

_T = TypeVar("_T", bound=BaseParseResult)

@dataclasses.dataclass
class BackendNode:
    node: ADVNodeModel
    node_def: ADVNodeModel
    node_def_parent: Optional[ADVNodeModel]

    parse_res: Optional[BaseParseResult] = None
    is_local_ref: bool = False
    is_subflow_def: bool = False
    # whether node def parent is folder
    is_parent_folder: bool = False
    is_ext_node: bool = False
    is_node_def_folder: bool = False

    # @property 
    # def is_ext_node(self):
    #     return self.ext_parse_res is not None

    # @property 
    # def is_node_def_folder(self):
    #     return isinstance(self.parse_res, FlowParseResult) and self.parse_res.has_subflow
    @property 
    def id(self):
        return self.node.id

    @property 
    def is_class_node(self):
        return self.node.nType == ADVNodeType.CLASS

    @property 
    def is_method_def(self):
        return self.node.is_method() and self.node.ref is None

    def get_def_qualname(self) -> str:
        node_def = self.node_def
        node_def_parent = self.node_def_parent
        if node_def_parent is None:
            return node_def.name 
        else:
            if (node_def.flags & ADVNodeFlags.IS_METHOD) or (node_def.flags & ADVNodeFlags.IS_CLASSMETHOD):
                return f"{node_def_parent.name}.{node_def.name}"
            else:
                return node_def.name

    def get_parse_res_checked(self, cls_type: type[_T]) -> _T:
        assert isinstance(self.parse_res, cls_type), f"parse_res must be {cls_type} in node {self.node.id}, got {type(self.parse_res)}"
        return self.parse_res

    def get_parse_res_raw_checked(self) -> BaseParseResult:
        assert isinstance(self.parse_res, BaseParseResult), f"parse_res must not be None"
        return self.parse_res

    def get_qualname_from_import(self) -> str:
        def_qname = self.get_def_qualname()
        node_def = self.node_def
        if (node_def.flags & ADVNodeFlags.IS_METHOD) or (node_def.flags & ADVNodeFlags.IS_CLASSMETHOD):
            # methods and ctors don't support alias
            return def_qname
        node = self.node
        res = node.name
        if node.ref is not None:
            res = self.node_def.name
            if node.name != self.node_def.name and node.name != "":
                # alias
                res = node.name
        return res

    def get_name_may_alias(self):
        node = self.node
        res = node.name
        node_def = self.node_def

        if (node_def.flags & ADVNodeFlags.IS_METHOD) or (node_def.flags & ADVNodeFlags.IS_CLASSMETHOD):
            # methods and ctors don't support alias
            return res
        if node.ref is not None:
            res = self.node_def.name
            if node.name != self.node_def.name and node.name != "":
                # alias
                res = node.name
        return res

    @staticmethod
    def get_class_import_stmt(node: ADVNodeModel, dot_prefix: str) -> str:
        assert node.nType == ADVNodeType.CLASS
        ref_import_path = node.path 
        ref_import_path = ref_import_path + [node.name]
        import_stmt = f"from {dot_prefix}{'.'.join(ref_import_path)} import {node.name}"
        return import_stmt

    def get_import_stmt(self, dot_prefix: str) -> Optional[str]:
        node_desc = self
        node = node_desc.node
        ref = node.ref
        if ref is not None:
            ref_import_path = ref.import_path
            assert ref_import_path is not None 
            if node_desc.is_node_def_folder:
                ref_import_path = ref_import_path + [TENSORPC_ADV_FOLDER_FLOW_NAME]
            elif node_desc.node_def.flow is not None:
                # inline flow node or class
                flow_node_name = node_desc.node_def.name
                ref_import_path = ref_import_path + [flow_node_name]
            if node_desc.node_def.is_defined_in_class():
                assert node_desc.node_def_parent is not None 
                cls_name = node_desc.node_def_parent.name
                import_stmt = f"from {dot_prefix}{'.'.join(ref_import_path)} import {cls_name}"
            else:
                import_stmt = f"from {dot_prefix}{'.'.join(ref_import_path)} import {node_desc.node_def.name}"

                if node.name != "":
                    import_stmt += f" as {node.name}"
            return import_stmt
        else:
            assert node_desc.is_subflow_def
            # is subflow def node
            assert node.flow is not None, "only ref node or subflow node can be external node"
            if node.inlinesf_name is not None:
                if node_desc.is_node_def_folder:
                    # we only need to import it when it is used in inline flow.
                    return f"from .{node.name}.{TENSORPC_ADV_FOLDER_FLOW_NAME} import {node.name}"
                else:
                    return f"from .{node.name} import {node.name}"
        return None 

    def get_func_call_expr(self, input_handles: list[BackendHandle],
            out_node_handle_to_node: dict[tuple[str, str], tuple[ADVNodeModel, BackendHandle]]) -> str:
        arg_name_parts: list[tuple[str, str]] = []
        func_name = self.get_name_may_alias()
        node_desc = self
        # TODO handle default value
        self_handle: Optional[BackendHandle] = None
        if node_desc.node_def.flags & ADVNodeFlags.IS_METHOD:
            if self.is_method_def:
                # method def node don't have self handle.
                if node_desc.node_def.is_init_fn():
                    assert node_desc.node_def_parent is not None # class node
                    func_call_str = node_desc.node_def_parent.name
                else:
                    func_call_str = f"self.{func_name}"
            else:
                self_handle = input_handles[0]
                input_handles = input_handles[1:]
                if node_desc.node_def.is_init_fn():
                    assert node_desc.node_def_parent is not None # class node
                    func_call_str = node_desc.node_def_parent.name
                else:    
                    if self_handle.handle.source_info is None:
                        func_call_str = f"(ADV.MISSING).{func_name}"
                    else:
                        source_nid = self_handle.handle.source_info.node_id
                        source_hid = self_handle.handle.source_info.handle_id
                        source_handle = out_node_handle_to_node[(source_nid, source_hid)][1]
                        func_call_str = f"{source_handle.handle.name}.{func_name}"
        else:
            func_call_str = node_desc.get_qualname_from_import()

        use_kwarg = False
        # always use kwarg for init fn and class node (init or dataclass)
        if node_desc.node_def.is_init_fn():
            use_kwarg = True 
        if node_desc.node_def.nType == ADVNodeType.CLASS:
            use_kwarg = True
        for h in input_handles:
            arg_name = h.handle.name
            if h.handle.source_info is None:
                arg_name_parts.append((arg_name, "ADV.MISSING"))
            else:
                source_nid = h.handle.source_info.node_id
                source_hid = h.handle.source_info.handle_id
                source_handle = out_node_handle_to_node[(source_nid, source_hid)][1]
                arg_name_parts.append((arg_name, source_handle.handle.name))
        if use_kwarg:
            arg_names = ", ".join(f"{x[0]}={x[1]}" for x in arg_name_parts)
        else:
            arg_names = ", ".join(x[1] for x in arg_name_parts)
        return f"{func_call_str}({arg_names})"

    def get_ref_node_meta_anno_str(self, type_str: Optional[str] = None):
        node = self.node
        if type_str is None:
            type_str = "Any"
        node_desc = self
        anno = ""
        if node.ref is not None:
            # use Annotated to attach ref node meta
            arg_parts = [f"\"{node.id}\"", f"\"{node.ref.node_id}\"", f"{int(node.flags)}", f"({node.position.x}, {node.position.y})"]
            if node.alias_map != "":
                arg_parts.append(f"\"{node.alias_map}\"")
            else:
                arg_parts.append(f"\"\"")
            if node_desc.is_local_ref:
                arg_parts.append(f"True")
            arg_str = ", ".join(arg_parts)
            anno = f": Annotated[{type_str}, ADV.RefNodeMeta({arg_str})]" 
        return anno

