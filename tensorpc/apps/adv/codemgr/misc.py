from typing import Any, Callable, Literal, Optional, Self, TypeGuard, TypeVar, Union

from tensorpc.apps.adv.codemgr.core import BackendHandle, BaseParseResult, BaseParser, ImplCodeSpec, BackendNode
import tensorpc.core.dataclass_dispatch as dataclasses

from tensorpc.apps.adv.codemgr import markers as adv_markers

@dataclasses.dataclass(kw_only=True)
class OutIndicatorParseResult(BaseParseResult):
    handle: BackendHandle

    def to_code_lines(self):
        assert self.node is not None 
        kwargs_parts = self.get_node_meta_kwargs(self.node)
        source_info = self.handle.handle.source_info
        if source_info is not None:
            source_node_id_str = f"\"{source_info.node_id}\"" 
            handle_id_str = f"\"{source_info.handle_id}\"" 
        else:
            source_node_id_str = "None"
            handle_id_str = "None"
        kwargs_parts.extend([
            f'conn_node_id={source_node_id_str}',
            f'conn_handle_id={handle_id_str}',
        ])
        if self.node.name != "":
            kwargs_parts.append(f'alias="{self.node.name}"')
        kwarg_str = ", ".join(kwargs_parts)
        decorator = f"ADV.{adv_markers.mark_out_indicator.__name__}({kwarg_str})"
        return ImplCodeSpec([
            f"{decorator}",
        ], -1, -1, 1, -1)

    def is_io_handle_changed(self, other_res: Self):
        """Compare io handles between two flow parse result. 
        if changed, all flows that depend on this fragment node (may flow) need to be re-parsed.
        TODO default change?
        """
        h1 = self.handle 
        h2 = other_res.handle
        if h1.symbol_name != h2.symbol_name or h1.name != h2.name or h1.handle.type != h2.handle.type or h1.handle.default != h2.handle.default:
            return True
        return False

@dataclasses.dataclass(kw_only=True)
class MarkdownParseResult(BaseParseResult):
    pass

@dataclasses.dataclass(kw_only=True)
class GlobalScriptParseResult(BaseParseResult):
    code: str

    def to_code_lines(self):
        assert self.node is not None 
        kwargs_parts = self.get_node_meta_kwargs(self.node)
        if self.node.ref is not None:
            kwargs_parts.append(f'ref_import_path={repr(self.node.ref.import_path)}')
        kwarg_str = ", ".join(kwargs_parts)
        mark_stmt = f"ADV.{adv_markers.mark_global_script.__name__}(name=\"{self.node.name}\", {kwarg_str})"
        mark_end_stmt = f"ADV.{adv_markers.mark_global_script_end.__name__}()"
        lines = self.code.splitlines()
        assert len(lines) > 0
        end_column = len(lines[-1]) + 1
        return ImplCodeSpec([
            mark_stmt,
            *lines,
            mark_end_stmt,
        ], 1, 1, len(lines), end_column)
