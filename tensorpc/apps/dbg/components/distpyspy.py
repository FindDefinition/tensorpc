from pathlib import Path
from tensorpc.core.datamodel.events import DraftChangeEvent
from tensorpc.core.tree_id import UniqueTreeIdForTree
from tensorpc.dock.components import chart, mui

from typing import Any, Hashable, Optional, Set, Union
from tensorpc.core import dataclass_dispatch as dataclasses
import tensorpc.core.datamodel as D

@dataclasses.dataclass
class PyspyFrame:
    name: str 
    filename: str 
    module: Optional[str]
    short_filename: str
    line: int 
    locals: Optional[Any] = None
    # for mui AutoComplete
    label: str = ""

@dataclasses.dataclass
class PyspyTrace:
    pid: int 
    thread_id: int 
    thread_name: str
    frames: list[PyspyFrame]

@dataclasses.dataclass
class PyspyTraceWithLabel(PyspyTrace):
    label: str = ""


@dataclasses.dataclass
class RootState:
    # groups: list[mui.JsonLikeNode]
    # tree select fornat
    selected_group: list[PyspyTraceWithLabel]
    # autocomplete select format
    selected_process: Optional[dict[str, Any]] = None
    selected_stack: Optional[dict[str, Any]] = None


class PyspyViewer(mui.FlexBox):
    def __init__(self):
        
        self._tree = mui.TanstackJsonLikeTree()
        self._select = mui.Autocomplete("Processes", []).prop(textFieldProps=mui.TextFieldProps(muiMargin="dense"), size="small")
        self._stack_select = mui.Autocomplete("Stack", []).prop(textFieldProps=mui.TextFieldProps(muiMargin="dense"), size="small")

        self._editor = mui.MonacoEditor("", "python", "default")
        self._editor.prop(readOnly=True)
        self._tree.event_select.on(self._on_tree_select)
        self._root_data: dict[tuple[str, int], list[PyspyTraceWithLabel]] = {}
        
        self.dm = mui.DataModel(RootState([]), [
            self._tree.prop(flex=1, ignoreRoot=True, padding="10px"),
            mui.Divider(orientation="vertical"),
            mui.VBox([
                mui.HBox([
                    self._select.prop(flex=1),
                    self._stack_select.prop(flex=2),
                ]),
                self._editor.prop(flex=1),
            ]).prop(flex=3)
        ])
        draft = self.dm.get_draft()
        self.dm.install_draft_change_handler(draft.selected_stack, 
            self._on_selected_proc_change, installed_comp=self._select)

        self._select.bind_draft_change(draft.selected_process)
        self._select.bind_fields(options=draft.selected_group)
        self._stack_select.bind_draft_change(draft.selected_stack)
        self._stack_select.bind_fields(options=D.not_null(draft.selected_process["frames"], []))

        super().__init__([
            self.dm
        ]) 
        self.prop(flexFlow="row nowrap")

    async def _on_tree_select(self, selected: dict[str, bool]):
        if not selected:
            return 
        first_uid_encoded = list(selected.keys())[0]
        first_uid = UniqueTreeIdForTree(first_uid_encoded)
        fname = first_uid.parts[0]
        line = int(first_uid.parts[1])
        key = (fname, line)
        assert key in self._root_data
        groups = self._root_data[key]
        draft = self.dm.get_draft()
        draft.selected_group = groups
        if groups:
            draft.selected_process = dataclasses.asdict(groups[0])
            if groups[0].frames:
                draft.selected_stack = dataclasses.asdict(groups[0].frames[0])

    async def _on_selected_proc_change(self, ev: DraftChangeEvent):
        new_val = ev.new_value  
        if new_val is not None:
            proc_obj = PyspyFrame(**new_val)
            fname = proc_obj.filename
            line = proc_obj.line
            fname_p = Path(fname)
            if fname_p.exists():
                # open file
                with open(fname, "r") as f:
                    code = f.read()
                await self.send_and_wait(self._editor.update_event(
                    value=code, path=fname
                ))
                await self._editor.set_line_number(line)
        else:
            await self.send_and_wait(self._editor.update_event(
                value="", path="default"
            ))

    async def set_pyspy_raw_data(self, data: dict[str, Any], group_by_full_trace: bool = False):
        all_traces: list[PyspyTraceWithLabel] = []
        for uid, info in data.items():
            trace = PyspyTraceWithLabel(**info) 
            trace.label = uid
            for frame in trace.frames:
                frame.label = f"{frame.name} ({frame.filename}:{frame.line})"
            all_traces.append(trace)
        if group_by_full_trace:
            raise NotImplementedError
        else:
            # group by fname-lineno of last frame
            grouped_traces: dict[tuple[str, int], list[PyspyTraceWithLabel]] = {}
            for trace in all_traces:
                if len(trace.frames) == 0:
                    continue
                last_frame = trace.frames[0]
                key = (last_frame.filename, last_frame.line)
                if key not in grouped_traces:
                    grouped_traces[key] = []
                grouped_traces[key].append(trace)
            group_nodes: list[mui.JsonLikeNode] = []
            for (fname, line), v in grouped_traces.items():
                fname_p = Path(fname)
                group_nodes.append(mui.JsonLikeNode(
                    UniqueTreeIdForTree.from_parts([fname, str(line)]),
                    f"{fname_p.name}:{line}",
                    mui.JsonLikeType.Object.value,
                ))
            self._root_data = grouped_traces
            root_node = mui.JsonLikeNode.create_dummy()
            root_node.children = group_nodes
            await self._tree.send_and_wait(self._tree.update_event(tree=root_node))
            if group_nodes:
                await self._on_tree_select({group_nodes[0].id.uid_encoded: True})