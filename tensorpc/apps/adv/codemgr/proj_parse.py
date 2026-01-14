import ast
from pathlib import Path
from typing import Any, Callable, Literal, Optional, TypeGuard, TypeVar, Union

from tensorpc.apps.adv.codemgr.core import RefNodeMeta
from tensorpc.apps.adv.constants import TENSORPC_ADV_FOLDER_FLOW_NAME
from tensorpc.apps.adv.logger import ADV_LOGGER
import dataclasses as dataclasses_plain

from tensorpc.apps.adv.codemgr import markers as adv_markers
from tensorpc.apps.adv.model import ADVEdgeModel, ADVFlowModel, ADVNodeModel, ADVNodeType, InlineCode
from tensorpc.core.funcid import ast_constant_expr_to_value
from tensorpc.dock.components.flowui import XYPosition
from tensorpc.utils.uniquename import UniqueNamePool


@dataclasses_plain.dataclass
class SingleADVMarker:
    marker_name: str
    marker_node: ast.Call
    kwargs: dict[str, Any]
    node_between: list[ast.stmt] = dataclasses_plain.field(default_factory=list)

@dataclasses_plain.dataclass
class BlockedADVMarker:
    block_node: Union[ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef]
    marker_name: str
    kwargs: dict[str, Any]
    marker_node: ast.Call
    lineno: int


@dataclasses_plain.dataclass
class GlobalScriptCodeDesc:
    marker: SingleADVMarker
    code_lines: list[str]

@dataclasses_plain.dataclass
class RefImportDesc:
    module: str 
    level: int
    name: str

@dataclasses_plain.dataclass
class RefNodeDesc:
    marker: SingleADVMarker

@dataclasses_plain.dataclass
class OutIndicatorDesc:
    marker: SingleADVMarker

@dataclasses_plain.dataclass
class SymbolGroupDesc:
    name: str
    marker: BlockedADVMarker
    code_lines: list[str]

@dataclasses_plain.dataclass
class FragmentDefDesc:
    name: str
    marker: BlockedADVMarker
    code_lines: list[str]

@dataclasses_plain.dataclass
class SubflowDefDesc:
    marker: SingleADVMarker

@dataclasses_plain.dataclass
class InlineFlowDefDesc:
    # we need to parse inline flow func body
    # to determine user edges and ref nodes.
    name: str
    marker: BlockedADVMarker
    body: list[ast.stmt]

@dataclasses_plain.dataclass
class ADVFlowCodeDesc:
    path: list[str]
    # code_lines: list[str]
    global_script_descs: list[GlobalScriptCodeDesc]
    ref_import_descs: list[RefImportDesc]
    ref_node_descs: list[RefNodeDesc]
    subflow_descs: list[SubflowDefDesc]
    out_indicator_descs: list[OutIndicatorDesc]
    symbol_group_descs: list[SymbolGroupDesc]
    fragment_def_descs: list[FragmentDefDesc]
    inline_flow_def_descs: list[InlineFlowDefDesc]
    user_edge_nodes: list[SingleADVMarker] = dataclasses_plain.field(default_factory=list)


class ADVProjectParser:
    def __init__(self, path_code_accessor: Callable[[list[str]], str]):
        self._path_code_accessor = path_code_accessor

    def _parse_flow_path_to_desc(self, path: list[str]):
        code = self._path_code_accessor(path)
        flow_desc = self._parse_flow_code_to_desc(path, code)
        return flow_desc

    def _parse_flow_code_to_desc(self, path: list[str], code: str):
        lines = code.splitlines()
        tree = ast.parse(code)
        markers = self._parse_markers(tree)
        # 1. global scripts
        marker_idx = 0
        global_script_descs: list[GlobalScriptCodeDesc] = []
        ref_import_descs: list[RefImportDesc] = []
        ref_node_descs: list[RefNodeDesc] = []
        out_indicator_descs: list[OutIndicatorDesc] = []
        symbol_group_descs: list[SymbolGroupDesc] = []
        fragment_def_descs: list[FragmentDefDesc] = []
        inline_flow_def_descs: list[InlineFlowDefDesc] = []
        user_edge_nodes: list[SingleADVMarker] = []
        subflow_descs: list[SubflowDefDesc] = []
        # import rich 
        # rich.print(markers)
        while marker_idx < len(markers):
            marker = markers[marker_idx]
            if isinstance(marker, SingleADVMarker):
                if marker.marker_name == adv_markers.mark_global_script.__name__:
                    assert marker_idx < len(markers) - 1, "Global script marker must be paired."
                    next_marker = markers[marker_idx + 1]
                    assert isinstance(next_marker, SingleADVMarker), "Global script marker must be paired."
                    assert next_marker.marker_name == adv_markers.mark_global_script_end.__name__
                    global_script_code_lines = lines[marker.marker_node.lineno: next_marker.marker_node.lineno - 1]
                    global_script_descs.append(GlobalScriptCodeDesc(
                        marker=marker,
                        code_lines=global_script_code_lines,
                    ))
                    marker_idx += 2
                elif marker.marker_name == adv_markers.mark_ref_node_dep.__name__:
                    import_nodes = marker.node_between
                    for import_stmt in import_nodes:
                        assert isinstance(import_stmt, ast.ImportFrom)
                        module = import_stmt.module
                        level = import_stmt.level
                        for alias in import_stmt.names:
                            # we don't use asname here.
                            name = alias.name
                            assert alias.asname is None, "Ref node import cannot use 'as' alias."
                            ref_desc = RefImportDesc(
                                module=module if module is not None else "",
                                level=level,
                                name=name,
                            )
                            ref_import_descs.append(ref_desc)
                    end_found = False 
                    marker_idx += 1
                    while marker_idx < len(markers):
                        next_marker = markers[marker_idx]
                        if isinstance(next_marker, SingleADVMarker):
                            if next_marker.marker_name == adv_markers.mark_ref_node_dep_end.__name__:
                                end_found = True 
                                marker_idx += 1
                                break
                            else:
                                if next_marker.marker_name == adv_markers.mark_subflow_def.__name__:
                                    subflow_descs.append(SubflowDefDesc(next_marker))
                                else:
                                    assert next_marker.marker_name == adv_markers.mark_ref_node.__name__
                                    ref_node_descs.append(RefNodeDesc(next_marker))
                                marker_idx += 1
                        else:
                            raise ValueError("Ref node region only allow mark_ref_node_dep_end or mark_ref_node.")
                    assert end_found, "Ref node dep marker must be ended with ref_node_dep_end marker."
                elif marker.marker_name == adv_markers.mark_out_indicator.__name__:
                    out_indicator_descs.append(OutIndicatorDesc(marker))
                    marker_idx += 1
                elif marker.marker_name == adv_markers.mark_symbol_dep.__name__:
                    # we don't care mark_symbol_dep here.
                    marker_idx += 1
                    continue 
                elif marker.marker_name == adv_markers.mark_symbol_dep_end.__name__:
                    # we don't care mark_symbol_dep here.
                    marker_idx += 1
                    continue 
                elif marker.marker_name == adv_markers.mark_user_edge.__name__:
                    user_edge_nodes.append(marker)
                    marker_idx += 1
                else:
                    raise ValueError(f"Unknown single ADV marker {marker.marker_name}.")
            else:
                if marker.marker_name == adv_markers.mark_symbol_group.__name__:
                    lineno = marker.marker_node.lineno
                    end_lineno = marker.block_node.end_lineno
                    assert end_lineno is not None 
                    code_lines = lines[lineno: end_lineno + 1]
                    symbol_group_descs.append(SymbolGroupDesc(
                        name=marker.block_node.name,
                        marker=marker,
                        code_lines=code_lines,
                    ))

                elif marker.marker_name == adv_markers.mark_fragment_def.__name__ or marker.marker_name == adv_markers.mark_inlineflow.__name__:
                    assert isinstance(marker.block_node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    return_anno = marker.block_node.returns
                    assert return_anno is not None 
                    start_lineno = return_anno.lineno
                    end_lineno = marker.block_node.end_lineno
                    assert end_lineno is not None 
                    code_lines = lines[start_lineno: end_lineno + 1]
                    if marker.marker_name == adv_markers.mark_fragment_def.__name__:
                        fragment_def_desc = FragmentDefDesc(
                            name=marker.block_node.name,
                            marker=marker,
                            code_lines=code_lines,
                        )
                        fragment_def_descs.append(fragment_def_desc)
                    else:
                        inline_flow_def_desc = InlineFlowDefDesc(
                            name=marker.block_node.name,
                            marker=marker,
                            body=marker.block_node.body,
                        )
                        inline_flow_def_descs.append(inline_flow_def_desc)
                else:
                    raise ValueError(f"Unknown blocked ADV marker {marker.marker_name}.")
                marker_idx += 1
        desc = ADVFlowCodeDesc(
            path=path,
            # code_lines=lines,
            user_edge_nodes=user_edge_nodes,
            global_script_descs=global_script_descs,
            ref_import_descs=ref_import_descs,
            ref_node_descs=ref_node_descs,
            out_indicator_descs=out_indicator_descs,
            symbol_group_descs=symbol_group_descs,
            fragment_def_descs=fragment_def_descs,
            inline_flow_def_descs=inline_flow_def_descs,
            subflow_descs=subflow_descs,
        )
        return desc

    def _is_adv_single_mark_call(self, node: ast.AST) -> TypeGuard[ast.Call]:
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "ADV":
                    if node.func.attr in adv_markers.MARKERS_REGISTRY:
                        return True 
        return False

    def _parse_markers(self, tree: ast.Module) -> list[Union[SingleADVMarker, BlockedADVMarker]]:
        res: list[Union[SingleADVMarker, BlockedADVMarker]] = []
        ast_stmt_between_single_markers: list[ast.stmt] = []
        prev_single_marker: Optional[SingleADVMarker] = None
        for node in tree.body:
            if isinstance(node, ast.Expr) and self._is_adv_single_mark_call(node.value):
                node = node.value
                assert isinstance(node.func, ast.Attribute)
                marker_name = node.func.attr
                marker_kwargs = self._parse_marker_kwargs(node)
                single_marker = SingleADVMarker(marker_node=node, marker_name=marker_name, kwargs=marker_kwargs)
                if prev_single_marker is not None:
                    prev_single_marker.node_between = ast_stmt_between_single_markers
                ast_stmt_between_single_markers = []
                prev_single_marker = single_marker
                res.append(single_marker)
            elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                decorators = node.decorator_list 
                if len(decorators) > 0:
                    may_adv_decorator = decorators[0]
                    if self._is_adv_single_mark_call(may_adv_decorator):
                        assert isinstance(may_adv_decorator.func, ast.Attribute)
                        marker_name = may_adv_decorator.func.attr
                        lineno = node.lineno
                        if len(decorators) > 1:
                            lineno = decorators[1].lineno
                        marker_kwargs = self._parse_marker_kwargs(may_adv_decorator)

                        res.append(BlockedADVMarker(
                            block_node=node, 
                            marker_node=may_adv_decorator, 
                            marker_name=marker_name,
                            kwargs=marker_kwargs,
                            lineno=lineno,
                        ))
                ast_stmt_between_single_markers.append(node)
            else:
                ast_stmt_between_single_markers.append(node)
                
        return res 

    def _parse_marker_kwargs(self, marker_node: ast.Call) -> dict[str, Any]:
        kwargs = marker_node.keywords
        res: dict[str, Any] = {}
        for kw in kwargs:
            assert kw.arg is not None 
            key = kw.arg
            value = ast_constant_expr_to_value(kw.value)
            res[key] = value
        return res  

    def _parse_desc_to_flow_model(self, path: list[str], visited: set[str]):
        code = self._path_code_accessor(path)
        flow_desc = self._parse_flow_code_to_desc(path, code)
        # import rich 
        # rich.print(flow_desc)

        node_id_to_node: dict[str, ADVNodeModel] = {}
        # we won't serialize edge id to code, so edge id is generated in each parse.
        edge_id_to_edge: dict[str, ADVEdgeModel] = {}
        for desc in flow_desc.global_script_descs:
            position = desc.marker.kwargs["position"]
            node_id = desc.marker.kwargs["node_id"]
            adv_node = ADVNodeModel(
                id=node_id, 
                position=XYPosition(x=position[0], y=position[1]),
                nType=ADVNodeType.GLOBAL_SCRIPT.value,
                name=desc.marker.kwargs["name"],
                impl=InlineCode("\n".join(desc.code_lines))
            )
            node_id_to_node[node_id] = adv_node
        for desc in flow_desc.out_indicator_descs:
            node_id = desc.marker.kwargs["node_id"]
            position = desc.marker.kwargs["position"]
            adv_node = ADVNodeModel(
                id=node_id, 
                position=XYPosition(x=position[0], y=position[1]),
                nType=ADVNodeType.OUT_INDICATOR.value,
                name="",
            )
            node_id_to_node[node_id] = adv_node
        for desc in flow_desc.symbol_group_descs:
            node_id = desc.marker.kwargs["node_id"]
            position = desc.marker.kwargs["position"]
            adv_node = ADVNodeModel(
                id=node_id, 
                position=XYPosition(x=position[0], y=position[1]),
                nType=ADVNodeType.SYMBOLS.value,
                name=desc.name,
                impl=InlineCode("\n".join(desc.code_lines))
            )
            node_id_to_node[node_id] = adv_node
        edge_id_pool = UniqueNamePool()
        for marker in flow_desc.user_edge_nodes:
            source = marker.kwargs["source"]
            source_handle = marker.kwargs["source_handle"]
            target = marker.kwargs["target"]
            target_handle = marker.kwargs["target_handle"]
            edge_id = edge_id_pool(f"E-{source}({source_handle})->{target}({target_handle})")
            edge_model = ADVEdgeModel(
                id=edge_id,
                source=source,
                sourceHandle=source_handle,
                target=target,
                targetHandle=target_handle,
                isAutoEdge=False,
            )
            edge_id_to_edge[edge_id] = edge_model
        name_to_frag: dict[str, ADVNodeModel] = {}
        for desc in flow_desc.fragment_def_descs:
            node_id = desc.marker.kwargs["node_id"]
            position = desc.marker.kwargs["position"]
            
            alias_map = desc.marker.kwargs.get("alias_map", "")
            inlineflow_name = desc.marker.kwargs.get("inlineflow_name", None)
            adv_node = ADVNodeModel(
                id=node_id, 
                position=XYPosition(x=position[0], y=position[1]),
                nType=ADVNodeType.FRAGMENT.value,
                name=desc.name,
                impl=InlineCode("\n".join(desc.code_lines)),
                alias_map=alias_map,
                inlinesf_name=inlineflow_name,
            )
            name_to_frag[desc.name] = adv_node
            node_id_to_node[node_id] = adv_node
        # handle ref nodes after fragment parse 
        # to handle local ref.
        for desc in flow_desc.subflow_descs:
            name = desc.marker.kwargs["name"]
            node_id = desc.marker.kwargs["node_id"]
            position = desc.marker.kwargs["position"]
            inlineflow_name = desc.marker.kwargs.get("inlineflow_name", None)
            ref_import_path = path + [name]
            subflow = self._parse_desc_to_flow_model(ref_import_path, visited)
            adv_node = ADVNodeModel(
                flow=subflow,
                id=node_id, 
                name=name,
                position=XYPosition(x=position[0], y=position[1]),
                nType=ADVNodeType.FRAGMENT.value,
                inlinesf_name=inlineflow_name,
            )
            node_id_to_node[node_id] = adv_node

        ref_node_desc_dict: dict[str, ADVNodeModel] = {}
        ref_node_name_to_imports: dict[str, RefImportDesc] = {}
        for import_desc in flow_desc.ref_import_descs:
            ref_node_name_to_imports[import_desc.name] = import_desc
        for desc in flow_desc.ref_node_descs:
            node_id = desc.marker.kwargs["node_id"]
            position = desc.marker.kwargs["position"]
            ref_node_id = desc.marker.kwargs.get("ref_node_id", None)
            inlineflow_name = desc.marker.kwargs.get("inlineflow_name", None)
            alias_map = desc.marker.kwargs.get("alias_map", "")

            ref_node_name_node = desc.marker.marker_node.args[0]
            assert isinstance(ref_node_name_node, ast.Name)
            ref_node_name = ref_node_name_node.id
            import_desc = ref_node_name_to_imports[ref_node_name]
            # build import path of ref node
            # for non-subflow ref imports, it already contains full import path.
            # so no need to deal with level.
            ref_import_path = import_desc.module.split(".")
            is_folder = False
            if ref_import_path[-1] == TENSORPC_ADV_FOLDER_FLOW_NAME:
                is_folder = True  
                ref_import_path = ref_import_path[:-1]
            # TODO parse flow node from import path?
            adv_node = ADVNodeModel(
                id=node_id, 
                position=XYPosition(x=position[0], y=position[1]),
                nType=ADVNodeType.FRAGMENT.value,
                ref_node_id=ref_node_id,
                inlinesf_name=inlineflow_name,
                alias_map=alias_map,
                ref_import_path=ref_import_path,
            )

            ref_node_desc_dict[node_id] = adv_node
        for inline_flow_desc in flow_desc.inline_flow_def_descs:
            # we only need to extract ref node infos here.
            for stmt in inline_flow_desc.body:
                # currently each stmt is always assign/annassign stmt with call except return.
                assert isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.Return))
                if isinstance(stmt, ast.AnnAssign):
                    # ref node meta is stored in Annotated[..., ADV.RefNodeMeta(...)]
                    anno_expr = stmt.annotation
                    assert isinstance(anno_expr, ast.Subscript)
                    slice_value = anno_expr.slice
                    assert isinstance(slice_value, ast.Tuple)
                    adv_ref_meta_expr = slice_value.elts[1]
                    assert isinstance(adv_ref_meta_expr, ast.Call)
                    arg_values: list[Any] = [ast_constant_expr_to_value(arg) for arg in adv_ref_meta_expr.args]
                    real_meta = RefNodeMeta(*arg_values)
                    if real_meta.id not in ref_node_desc_dict:
                        adv_node = ADVNodeModel(
                            id=real_meta.id, 
                            position=XYPosition(x=real_meta.position[0], y=real_meta.position[1]),
                            nType=ADVNodeType.FRAGMENT.value,
                            ref_node_id=real_meta.ref_node_id,
                            inlinesf_name=inline_flow_desc.name,
                            alias_map=real_meta.alias_map,
                            ref_import_path=path,
                        )
                        ref_node_desc_dict[real_meta.id] = adv_node
        
        node_id_to_node.update(ref_node_desc_dict)
        # import rich 
        # rich.print(node_id_to_node)
        # rich.print(edge_id_to_edge)
        flow_model = ADVFlowModel(
            nodes=node_id_to_node,
            edges=edge_id_to_edge,
        )
        return flow_model

# def parse_adv_project(import_path_accessor: Callable[[list[str]], str]) -> ADVProject:
#     pass 