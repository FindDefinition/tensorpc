from functools import partial

from sympy import expand
from tensorpc.core.tree_id import UniqueTreeIdForTree
from tensorpc.flow import mui, flowui, three, plus, appctx, mark_did_mount, mark_create_layout
from tensorpc.flow.components.flowplus.network.pthfx import (
    FlowUIInterpreter, PytorchExportBuilder, PytorchFlowOutput,
    PytorchFlowOutputPartial, PytorchFlowOutputPartial)
import torch
from torch.nn import ModuleDict, ModuleList
import torch.fx
import torch.export
import dataclasses
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Union
from tensorpc.flow.components.plus.pthcommon import PytorchModuleTreeItem
from tensorpc.flow.jsonlike import IconButtonData


@dataclasses.dataclass
class ExpandState:
    expanded: List[str]


class PytorchModuleTreeItemEx(PytorchModuleTreeItem):

    def get_json_like_node(self, id: UniqueTreeIdForTree) -> mui.JsonLikeNode:
        res = super().get_json_like_node(id)
        if isinstance(self._mod, (ModuleDict, ModuleList)):
            res.iconBtns = mui.undefined
        return res


class PytorchModuleViewer(mui.FlexBox):

    def __init__(self,
                 external_submodule_id: Optional[str] = None,
                 external_module: Optional[torch.nn.Module] = None,
                 external_pth_flow: Optional[PytorchFlowOutput] = None,
                 external_ftree_id: Optional[str] = None):
        graph = flowui.Flow(
            [], [], [flowui.MiniMap(),
                     flowui.Controls(),
                     flowui.Background()])
        self.graph = graph
        self.is_external_mode = external_submodule_id is not None and external_pth_flow is not None and external_module is not None
        self._simple_tree = plus.BasicObjectTree(use_fast_tree=True,
                                                 clear_data_when_unmount=True)
        self._info_container = mui.VBox([]).prop(margin="5px")
        self._simple_tree.tree.prop(expansionIconTrigger=True)
        self._simple_tree.event_async_select_single.on(
            self._on_module_tree_select)
        self._side_container = mui.VBox([
            self._simple_tree.prop(flex=1),
            mui.Divider(),
            self._info_container.prop(flex=1, overflow="auto"),
        ]).prop(height="100%")
        node_menu_items = [
            mui.MenuItem("expand", "Expand Node"),
            mui.MenuItem("subflow", "Show Sub Flow"),
        ]
        view_pane_menu_items = [
            mui.MenuItem("layout", "Dagre Layout"),
            mui.MenuItem("layout-tight", "Dagre Layout Tight"),
            mui.MenuItem("layout-longest", "Dagre Layout Longest"),
        ]
        self.graph.event_pane_context_menu.on(self._on_pane_contextmenu)
        self.graph.event_selection_change.on(self._on_selection_change)
        # we need to use component ready instead of after_mount
        # because we send component event, the handler will ready
        # when event_component_ready received.
        self.graph.event_component_ready.on(self._on_graph_ready)

        self.graph.prop(onlyRenderVisibleElements=True,
                        paneContextMenuItems=view_pane_menu_items,
                        nodeContextMenuItems=node_menu_items)
        self.graph_container = mui.HBox([
            self.graph.prop(defaultLayoutSize=(150, 40))
        ]).prop(width="100%", height="100%", overflow="hidden")
        self.global_container = mui.Allotment(
            mui.Allotment.ChildDef([
                mui.Allotment.Pane(self.graph_container),
                mui.Allotment.Pane(self._side_container),
            ])).prop(defaultSizes=[200, 100])
        self._subflow_dialog = mui.Dialog([])
        self._subflow_dialog.prop(height="70vh",
                                  width="70vw",
                                  dialogMaxWidth="xl")
        self._subflow_dialog.event_modal_close.on(
            self._handle_subflow_dialog_close)
        if not self.is_external_mode:
            super().__init__([
                self.global_container,
                self._subflow_dialog,
            ])
        else:
            super().__init__([
                self.global_container,
            ])
        self.prop(width="100%", height="100%", overflow="hidden")
        self._external_ftree_id = external_ftree_id
        self._external_submodule_id = external_submodule_id
        self._cur_pth_flow: Optional[PytorchFlowOutput] = None
        self._cur_module: Optional[torch.nn.Module] = None
        if external_pth_flow is not None and external_submodule_id is not None and external_module is not None:
            self._cur_pth_flow = external_pth_flow
            self._cur_module = external_module

        self._cur_graph_metadata: Optional[PytorchFlowOutputPartial] = None
        self._current_state: Optional[ExpandState] = None

        self._dagre_options = flowui.DagreLayoutOptions(ranksep=25, )

    async def _init_set_exported_flow(self, pth_flow: PytorchFlowOutput,
                                      module: torch.nn.Module):
        if self.is_external_mode and self._external_submodule_id is not None:
            self._current_state = ExpandState([self._external_submodule_id])
        else:
            self._current_state = ExpandState([])
        if self._external_ftree_id is not None:
            ext_mod_id = self._external_ftree_id
        else:
            ext_mod_id = self._external_submodule_id
        merged_graph_res = pth_flow.create_graph_with_expanded_modules(
            self._current_state.expanded,
            submodule_id=ext_mod_id,
            submodule_id_is_module=self._external_ftree_id is None)
        self._cur_graph_metadata = merged_graph_res
        self.graph.event_node_context_menu.clear()
        self.graph.event_node_context_menu.on(
            partial(self._on_node_contextmenu,
                    pth_flow=pth_flow,
                    state=self._current_state))
        await self.graph.set_flow_and_do_dagre_layout(merged_graph_res.nodes,
                                                      merged_graph_res.edges,
                                                      self._dagre_options)
        await self._info_container.set_new_layout([])
        with torch.device("meta"):
            mod_meta = module.to("meta")
            self._cur_module = mod_meta
            module_id_prefix = ""
            expand_level = 0
            btns = [IconButtonData("subflow", mui.IconType.Preview)]
            if self._external_submodule_id is not None:
                module_id_prefix = self._external_submodule_id
                expand_level = 1
                btns = mui.undefined
            await self._simple_tree.set_root_object_dict(
                {
                    "":
                    PytorchModuleTreeItemEx(
                        mod_meta,
                        module_id_prefix,
                        on_lazy_expand=self._on_tree_item_lazy_expand,
                        on_button_click=self._on_tree_item_button_click,
                        btns=btns),
                },
                expand_level=expand_level)

    async def _on_graph_ready(self):
        # do init when flow is external (e.g. created from main flow)
        if self.is_external_mode:
            assert self._cur_pth_flow is not None
            assert self._cur_module is not None
            await self._init_set_exported_flow(self._cur_pth_flow,
                                               self._cur_module)

    def _patch_module_uid(self, module_uid: str):
        if self._external_submodule_id is None:
            return module_uid
        # module_uid has format ["root", ""] + module_parts
        # _external_submodule_id don't contains prefix ["root", ""]
        # so patched format is ["root", ""] + _external_submodule_id + module_uid[2:]
        module_uid_parts = module_uid.split(".")
        res_parts = [
            "root", ""
        ] + self._external_submodule_id.split(".") + module_uid_parts[2:]
        return ".".join(res_parts)

    async def _module_tree_select(self, module_uid_to_sel: str):
        if self._cur_graph_metadata is not None:
            selections: List[str] = []
            for k, v in self._cur_graph_metadata.node_id_to_data.items():
                module_id = v.module_id
                if module_id is not None:
                    module_id_str = ".".join(module_id.parts)
                    if module_id_str.startswith(module_uid_to_sel):
                        selections.append(k)
                        continue
            await self.graph.select_nodes(selections)
            await self.graph.locate_nodes(selections,
                                          keep_zoom=True,
                                          duration=200)

    async def _node_tree_select(self, node_ids: List[str]):
        if self._cur_graph_metadata is not None:
            await self.graph.locate_nodes(node_ids,
                                          keep_zoom=True,
                                          duration=200)
            # await self.graph.select_nodes(node_ids)

    async def _on_module_tree_select(self, ev):
        uid = ev.uid
        parts = uid.parts
        uid_str = ".".join(parts)
        uid_str_patched = self._patch_module_uid(uid_str)
        parts = uid_str_patched.split(".")
        module_uid_to_sel = ".".join(
            parts[2:])  # first part is "root", second part is ""
        return await self._module_tree_select(module_uid_to_sel)

    async def _handle_subflow_dialog_close(self, ev: mui.DialogCloseEvent):
        await self._subflow_dialog.set_new_layout([])

    async def _on_tree_item_button_click(self,
                                         module_id: str,
                                         btn_key: str,
                                         is_ftree_id: bool = False):
        if btn_key == "subflow":
            if self._cur_pth_flow is not None and self._cur_module is not None:
                assert self._cur_pth_flow.ftree is not None
                ftree_id = None
                if is_ftree_id:
                    ftree_id = module_id
                    if module_id not in self._cur_pth_flow.ftree.tree_id_to_node:
                        return
                    module_id = self._cur_pth_flow.ftree.tree_id_to_node[
                        module_id]["module"]
                    submodule = self._cur_module.get_submodule(module_id)
                else:
                    if module_id not in self._cur_pth_flow.ftree.module_id_to_tree_ids:
                        # modules that con't contains forward ops
                        return
                    submodule = self._cur_module.get_submodule(module_id)
                viewer_subflow = PytorchModuleViewer(
                    external_submodule_id=module_id,
                    external_module=submodule,
                    external_pth_flow=self._cur_pth_flow,
                    external_ftree_id=ftree_id)
                await self._subflow_dialog.set_new_layout([
                    viewer_subflow.prop(width="100%", height="100%"),
                ])
                await self._subflow_dialog.set_open(True)

    async def _on_tree_item_lazy_expand(self, module_id: str):
        # we already set prefix to external_module_id if exists
        # so we don't need to patch here.
        dagre = flowui.DagreLayoutOptions(ranksep=20, )
        if self._current_state is not None and self._cur_pth_flow is not None:
            new_ids = [module_id]
            for expanded_module_id in self._current_state.expanded:
                if not expanded_module_id.startswith(module_id):
                    new_ids.append(expanded_module_id)
            self._current_state.expanded = new_ids
            if self._external_ftree_id is not None:
                ext_mod_id = self._external_ftree_id
            else:
                ext_mod_id = self._external_submodule_id
            merged_graph_res = self._cur_pth_flow.create_graph_with_expanded_modules(
                self._current_state.expanded,
                submodule_id=ext_mod_id,
                submodule_id_is_module=self._external_ftree_id is None)
            await self.graph.set_flow_and_do_dagre_layout(
                merged_graph_res.nodes, merged_graph_res.edges, dagre)
            self._cur_graph_metadata = merged_graph_res
            await self._module_tree_select(module_id)
            await self._info_container.set_new_layout([])

    async def _on_pane_contextmenu(self, data):
        item_id = data["itemId"]
        dagre = dataclasses.replace(self._dagre_options)
        # network-simplex, tight-tree or longest-path
        if item_id == "layout":
            dagre.ranker = "network-simplex"
            await self.graph.do_dagre_layout(dagre)
        if item_id == "layout-tight":
            dagre.ranker = "tight-tree"
            await self.graph.do_dagre_layout(dagre)
        if item_id == "layout-longest":
            dagre.ranker = "longest-path"
            await self.graph.do_dagre_layout(dagre)

    async def export_module_to_flow(self,
                                    module: torch.nn.Module,
                                    args: Tuple[Any, ...],
                                    kwargs: Optional[Dict[str, Any]] = None):
        if self.is_external_mode:
            raise ValueError("Cannot export module to flow in external mode")
        with torch.device("meta"):
            mod_meta = module.to("meta")
            gm = torch.export.export(mod_meta, args, kwargs)
        builder = PytorchExportBuilder()
        interpreter = FlowUIInterpreter(gm, builder, module, verbose=False)
        outputs = interpreter.run_on_graph_placeholders()
        assert isinstance(outputs, (list, tuple))
        pth_flow = builder.build_pytorch_detached_flow(module, outputs)
        self._cur_pth_flow = pth_flow
        await self._init_set_exported_flow(pth_flow, module)

    async def _on_node_contextmenu(self, data, pth_flow: PytorchFlowOutput,
                                   state: ExpandState):
        item_id = data["itemId"]
        node_id = data["nodeId"]
        dagre = self._dagre_options
        if self._external_ftree_id is not None:
            ext_mod_id = self._external_ftree_id
        else:
            ext_mod_id = self._external_submodule_id

        if item_id == "expand":
            # node = self.graph.get_node_by_id(node_id)
            use_module_expand: bool = True
            if self.is_external_mode:
                assert use_module_expand
            if self._cur_graph_metadata is not None:
                node_meta = self._cur_graph_metadata.node_id_to_data.get(
                    node_id)
                if node_meta is not None:
                    if not use_module_expand:
                        ftree_id = node_meta.ftree_id
                        if ftree_id is not None:
                            state.expanded.append(ftree_id)
                            merged_graph_res = pth_flow.create_graph_with_expanded_ids(
                                state.expanded)
                            await self.graph.set_flow_and_do_dagre_layout(
                                merged_graph_res.nodes, merged_graph_res.edges,
                                dagre)
                            self._cur_graph_metadata = merged_graph_res
                            if module_id_str != "":
                                parts.insert(0, "")
                            uid_obj = UniqueTreeIdForTree.from_parts(
                                ["root", *parts]).uid_encoded
                            await self._simple_tree._on_expand(uid_obj)
                            await self._info_container.set_new_layout([])
                    else:
                        module_id = node_meta.module_id
                        if module_id is not None:
                            parts = module_id.parts.copy()
                            module_id_str = ".".join(parts)
                            state.expanded.append(module_id_str)
                            merged_graph_res = pth_flow.create_graph_with_expanded_modules(
                                state.expanded,
                                submodule_id=ext_mod_id,
                                submodule_id_is_module=self._external_ftree_id
                                is None)
                            await self.graph.set_flow_and_do_dagre_layout(
                                merged_graph_res.nodes, merged_graph_res.edges,
                                dagre)
                            self._cur_graph_metadata = merged_graph_res
                            if self._external_submodule_id is not None:
                                num_ex_part = len(
                                    self._external_submodule_id.split("."))
                                parts = parts[num_ex_part:]
                            if module_id_str != "":
                                parts.insert(0, "")
                            uid_obj = UniqueTreeIdForTree.from_parts(
                                ["root", *parts]).uid_encoded
                            await self._simple_tree._on_expand(
                                uid_obj, lazy_expand_event=False)
                            await self._info_container.set_new_layout([])
        elif item_id == "subflow":
            if self._cur_graph_metadata is not None and not self.is_external_mode:
                node_meta = self._cur_graph_metadata.node_id_to_data.get(
                    node_id)

                if node_meta is not None and node_meta.ftree_id is not None:
                    await self._on_tree_item_button_click(
                        node_meta.ftree_id, "subflow", is_ftree_id=True)

    def _get_shape_type_from_raw(self, raw: Any) -> Tuple[str, List[int]]:
        if type(raw).__name__ == "FakeTensor":
            shape = list(raw.shape)
            type_str = "Tensor"
        else:
            shape = []
            type_str = type(raw).__name__
        return type_str, shape

    async def _on_selection_change(self, ev: flowui.EventSelection):
        if ev.nodes and len(ev.nodes) == 1:
            node_id = ev.nodes[0]
            if self._cur_graph_metadata is not None and self._cur_module is not None:
                if node_id in self._cur_graph_metadata.node_id_to_data:
                    data = self._cur_graph_metadata.node_id_to_data[node_id]
                    layouts: List[mui.MUIComponentBase] = []
                    module_id = data.module_id
                    qname = data.module_qname
                    if data.is_io_node:
                        if data.output_desps is not None:
                            out = data.output_desps[0]
                            type_str, shape = self._get_shape_type_from_raw(
                                out)
                            layouts.append(mui.Markdown(f"`{type_str}`"))
                            layouts.append(mui.Markdown(f"`{shape}`"))
                    else:
                        if module_id is not None and qname is not None:
                            module_id_str = ".".join(module_id.parts)
                            if self._external_submodule_id is not None:
                                assert module_id_str.startswith(
                                    self._external_submodule_id), f"{module_id_str} {self._external_submodule_id}"
                                if self._external_submodule_id != "":
                                    module_id_str = module_id_str[len(
                                        self._external_submodule_id) + 1:]
                            try:
                                module = self._cur_module.get_submodule(module_id_str)
                                layouts.append(
                                    mui.Markdown(f"`Module`: `{qname}`"))
                                layouts.append(
                                    mui.Markdown(f"`id`: `{module_id_str}`"))
                                is_seq = "Sequential" in qname
                                is_module_list = "ModuleList" in qname
                                is_module_dict = "ModuleDict" in qname
                                is_container = is_seq or is_module_list or is_module_dict
                                if qname.startswith("torch.") and not is_container:
                                    # official torch module
                                    layouts.append(mui.Divider())
                                    for name, param in module.named_parameters():
                                        shape_str = ",".join(map(str, param.shape))
                                        layouts.append(
                                            mui.Markdown(f"`{name}(P)`: `[{shape_str}]`"))
                                    for name, param in module.named_buffers():
                                        shape_str = ",".join(map(str, param.shape))
                                        layouts.append(
                                            mui.Markdown(f"`{name}(B)`: `[{shape_str}]`"))

                            except AttributeError:
                                pass
                            layouts.append(mui.Divider())

                            inp_handle_to_edges = self._cur_graph_metadata.node_id_to_inp_handle_to_edges[node_id]
                            out_handle_to_edges = self._cur_graph_metadata.node_id_to_out_handle_to_edges[node_id]
                            md_lines = []
                            for i, handle_to_edges in enumerate([inp_handle_to_edges, out_handle_to_edges]):
                                md_lines.append("#### Inputs" if i == 0 else "#### Outputs")
                                for handle, edges in handle_to_edges.items():
                                    # print(handle, len(edges))
                                    for edge in edges:
                                        # print(edge.id, edge.source, edge.target)
                                        edge_data = self._cur_graph_metadata.edge_id_to_data.get(edge.id)
                                        if edge_data is not None:
                                            raw = edge_data.raw 
                                            type_str = ""
                                            type_str, shape = self._get_shape_type_from_raw(
                                                raw)
                                            if type(raw).__name__ == "FakeTensor":
                                                md_lines.append(f"`{shape}`")
                                            else:
                                                md_lines.append(f"`{type_str}`")
                            layouts.append(
                                mui.Markdown("\n\n".join(md_lines)))
                    if layouts:
                        await self._info_container.set_new_layout([*layouts])
                    # print(data)
        elif ev.edges and len(ev.edges) == 1:
            edge_id = ev.edges[0]
            if self._cur_graph_metadata is not None:
                if edge_id in self._cur_graph_metadata.edge_id_to_data:
                    data = self._cur_graph_metadata.edge_id_to_data[edge_id]
                    edge = self._cur_graph_metadata.id_to_edges[edge_id]
                    type_str, shape = self._get_shape_type_from_raw(
                        data.raw)
                    await self._info_container.set_new_layout([
                        mui.Markdown(f"`{type_str}`"),
                        mui.Markdown(f"`{shape}`"),
                        mui.Divider(),
                        mui.Button(
                            "Input",
                            lambda: self._node_tree_select([edge.source])),
                        mui.Button(
                            "Output",
                            lambda: self._node_tree_select([edge.target])),
                    ])
