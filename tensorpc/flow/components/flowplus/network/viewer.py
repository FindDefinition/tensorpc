import copy
import enum
from functools import partial
import inspect
import json
import os
from pathlib import Path
import time

from tensorpc.core.moduleid import get_qualname_of_type
from tensorpc.core.tree_id import UniqueTreeIdForTree
from tensorpc.flow import mui, flowui, three, plus, appctx, mark_did_mount, mark_create_layout
from tensorpc.flow.components.flowplus.network.pthfx import (
    FlowUIInterpreter, PytorchExportBuilder, PytorchFlowOutput,
    PytorchFlowOutputPartial, PytorchFlowOutputPartial, PytorchNodeMeta)
import torch
from torch.nn import ModuleDict, ModuleList
import torch.fx
import torch.export
import dataclasses
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Union
from tensorpc.flow.components.plus.objinspect.tree import BasicObjectTree
from tensorpc.flow.components.plus.pthcommon import PytorchModuleTreeItem
from tensorpc.flow.components.plus.styles import get_tight_icon_tab_theme, get_tight_icon_tab_theme_horizontal
from tensorpc.flow.jsonlike import IconButtonData, as_dict_no_undefined
from tensorpc.utils.rich_logging import get_logger
from tensorpc import compat
import torch.utils
from tensorpc.flow.components.plus.config import ConfigDialogEvent, ConfigPanelDialog, ConfigPanelDialogPersist

from tensorpc.utils.tb_parser import parse_python_traceback
LOGGER = get_logger("tensorpc.flowui.pytorch")

def _get_flow_css(use_multiple_handle_node: bool):
    multiple_handle_css = {
        ".react-flow__node__handles": {
            # add ellipsis to node text
            "position": "absolute",
            "display": "flex",
            # "flexDirection": "row",
            "justifyContent": "space-around",
            "width": "100%",
            "left": "0",
        },
        ".react-flow__node__handles_target": {
            "top": "0",
            "transform": "50%"
        },
        ".react-flow__node__handles_source": {
            "bottom": "0",
            "transform": "-50%"

        },
        ".react-flow__handle": {
            "position": "relative",
            "left": "0",
        },
    }
    res = {
        ".react-flow__node__content": {
            # add ellipsis to node text
            "overflow": "hidden",
            "textOverflow": "ellipsis",
            "whiteSpace": "nowrap",
            "width": "100%",
        },
    }
    if use_multiple_handle_node:
        res.update(multiple_handle_css)
    return res

@dataclasses.dataclass
class ExpandState:
    expanded: List[str]

ELK_FILE_RESOURCE_KEY = "flow_elk.json"

class PytorchModuleTreeItemEx(PytorchModuleTreeItem):

    def get_json_like_node(self, id: UniqueTreeIdForTree) -> mui.JsonLikeNode:
        res = super().get_json_like_node(id)
        if isinstance(self._mod, (ModuleDict, ModuleList)):
            res.iconBtns = mui.undefined
        return res

class TabType(enum.Enum):
    INFO = "Info"
    ARGS = "Args"
    STACKTRACE = "StackTrace"
    DEBUG = "Debug"
    GLOBAL = "global"
    MODULE_TREE = "ModuleTree"

class PytorchModuleViewer(mui.FlexBox):

    def __init__(self,
                 external_submodule_id: Optional[str] = None,
                 external_module: Optional[torch.nn.Module] = None,
                 external_pth_flow: Optional[PytorchFlowOutput] = None,
                 external_ftree_id: Optional[str] = None,
                 max_nested_depth: int = 4,
                 *,
                 _nested_depth: int = 0):
        graph = flowui.Flow(
            [], [], [flowui.MiniMap(),
                     flowui.Controls(),
                     flowui.Background()])
        self.graph = graph
        self.is_external_mode = external_submodule_id is not None and external_pth_flow is not None and external_module is not None
        self._module_tree = plus.BasicObjectTree(use_fast_tree=True,
                                                 clear_data_when_unmount=True,
                                                 auto_folder_limit=-1)
        self._info_container = mui.VBox([]).prop(padding="5px", width="100%",
                                                        height="100%",
                                                        overflow="auto")
        
        self._args_tree = BasicObjectTree(use_fast_tree=True, clear_data_when_unmount=True).prop(flex=1)
        self._args_container = mui.VBox([self._args_tree]).prop(padding="5px", width="100%",
                                                        height="100%",
                                                        overflow="hidden")
        self._debug_json_tree = mui.JsonViewer()
        self._dbg_container = mui.VBox([self._debug_json_tree]).prop(padding="5px", width="100%",
                                                        height="100%",
                                                        overflow="auto")
        
        self._stack_trace_container = mui.VBox([]).prop(padding="5px", width="100%",
                                                        height="100%",
                                                        overflow="auto",
                                                        alignItems="flex-start")

        self._global_container = mui.VBox([
            mui.IconButton(
                mui.IconType.Download).prop(size="small", href=f"tensorpc://{ELK_FILE_RESOURCE_KEY}", target="_blank"),
        ])
        self._global_container.prop(padding="5px", width="100%",
                                    height="100%",
                                    overflow="auto")
        self._module_prop_tree = BasicObjectTree(use_fast_tree=True, clear_data_when_unmount=True).prop(flex=1)
        self._module_prop_container = mui.VBox([self._module_prop_tree]).prop(padding="5px", width="100%",
                                                        height="100%",
                                                        overflow="auto")
        tab_defs = [
            mui.TabDef("",
                       TabType.INFO.value,
                       self._info_container,
                       icon=mui.IconType.Info,
                       tooltip="Info"),
            mui.TabDef("",
                       TabType.ARGS.value,
                       self._args_container,
                       icon=mui.IconType.DataObject,
                       tooltip="Args"),
            mui.TabDef("",
                       TabType.STACKTRACE.value,
                       self._stack_trace_container,
                       icon=mui.IconType.Timeline,
                       tooltip="stacktrace"),
            mui.TabDef("",
                       TabType.DEBUG.value,
                       self._dbg_container,
                       icon=mui.IconType.BugReport,
                       tooltip="Flow Debug Info"),
            mui.TabDef("",
                       TabType.MODULE_TREE.value,
                       self._module_prop_container,
                       icon=mui.IconType.AccountTree,
                       tooltip="Module Tree"),
            mui.TabDef("",
                       TabType.GLOBAL.value,
                       self._global_container,
                       icon=mui.IconType.Settings,
                       tooltip="Global Info Settings"),
        ]

        self._tabs = mui.Tabs(tab_defs, init_value=TabType.INFO.value).prop(panelProps=mui.FlexBoxProps(
                                  width="100%", padding=0, overflow="hidden", flex=1),
                                                  borderBottom=1,
                                                  borderColor='divider')

            # mui.ThemeProvider([mui.HBox([self._tabs]).prop(flex=1)],
            #                   get_tight_icon_tab_theme()),

        self._current_tabs_value = "Info"
        self._tabs.event_change.on(self._on_tabs_change)
        self._module_tree.tree.prop(expansionIconTrigger=True)
        self._module_tree.event_async_select_single.on(
            self._on_module_tree_select)
        self._side_container = mui.VBox([
            self._module_tree.prop(flex=1),
                mui.Divider(),
                mui.VBox([
                    mui.ThemeProvider([self._tabs], get_tight_icon_tab_theme_horizontal())
                ]).prop(flex=1, overflow="hidden"),
        ]).prop(height="100%", overflow="hidden")
        self._enable_nested_dialog = _nested_depth < max_nested_depth
        node_menu_items = [
            mui.MenuItem("expand", "Expand Node"),
        ]
        if self._enable_nested_dialog:
            node_menu_items.append(mui.MenuItem("subflow", "Show Sub Flow"))
        view_pane_menu_items = [
            mui.MenuItem("dagre", "Dagre Layout"),
            mui.MenuItem("elk", "Elk Layout"),
            mui.MenuItem("d1", divider=True),
            mui.MenuItem("dagre-cfg", "Dagre Layout Advanced"),
            mui.MenuItem("elk-cfg", "Elk Layout Advanced"),

        ]
        self.graph.event_pane_context_menu.on(self._on_pane_contextmenu)
        self.graph.event_selection_change.on(self._on_selection_change)
        self.graph.event_after_mount.on(self._on_graph_ready)

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
        self._subflow_dialog.prop(height=f"{85 - 4 * _nested_depth}vh",
                                  width=f"{80 - 4 * _nested_depth}vw",
                                  dialogMaxWidth=False,
                                  includeFormControl=False,
                                  fullWidth=False)
        self._subflow_dialog.event_modal_close.on(
            self._handle_subflow_dialog_close)

        self._dagre_options_default = flowui.DagreLayoutOptions(ranksep=25, )
        # auto config items of a dataclass object only available when its value exists (not undefined), so we
        # init some default values here.
        self._elk_options_default = flowui.ElkLayoutOptions(
            spacing=flowui.ElkSpacing(nodeNodeBetweenLayers=25),
            considerModelOrder=flowui.ElkConsiderModelOrder(),
            nodePlacement=flowui.ElkNodePlacement(),
            defaultNodeProps=flowui.ElkDefaultNodeProps(
                portAlignment=flowui.ElkPortAlignment(default="CENTER"),
            ))
        reset_btn = mui.Button("Reset").prop(fullWidth=True, size="small")
        self._dagre_cfg_dialog = ConfigPanelDialogPersist(
            copy.deepcopy(self._dagre_options_default), lambda ev: self.graph.do_dagre_layout(ev.cfg), [
                reset_btn
            ])
        reset_btn.event_click.on(lambda: self._dagre_cfg_dialog.set_config_object(copy.deepcopy(self._dagre_options_default)))
        self._dagre_cfg_dialog.prop(okLabel="Launch Layout", title="Dagre Layout Config", dividers=True)
        reset_btn = mui.Button("Reset").prop(fullWidth=True, size="small")

        self._elk_cfg_dialog = ConfigPanelDialogPersist(
            copy.deepcopy(self._elk_options_default), lambda ev: self.graph.do_elk_layout(ev.cfg), [
                reset_btn
            ])
        self._elk_cfg_dialog.prop(okLabel="Launch Layout", title="Elk Layout Config", dividers=True)
        reset_btn.event_click.on(lambda: self._elk_cfg_dialog.set_config_object(copy.deepcopy(self._elk_options_default)))
        if not self.is_external_mode or self._enable_nested_dialog:
            super().__init__([
                self.global_container,
                self._subflow_dialog,
                self._dagre_cfg_dialog,
                self._elk_cfg_dialog,
            ])
        else:
            super().__init__([
                self.global_container,
                self._dagre_cfg_dialog,
                self._elk_cfg_dialog,
            ])
        self.prop(width="100%", height="100%", overflow="hidden")
        self._use_multiple_handle_node = True
        self.graph_container.update_raw_props(_get_flow_css(use_multiple_handle_node=False))

        self._external_ftree_id = external_ftree_id
        self._external_submodule_id = external_submodule_id
        self._toplevel_pth_flow: Optional[PytorchFlowOutput] = None
        self._toplevel_module: Optional[torch.nn.Module] = None
        if external_pth_flow is not None and external_submodule_id is not None and external_module is not None:
            self._toplevel_pth_flow = external_pth_flow
            self._toplevel_module = external_module

        self._cur_graph_metadata: Optional[PytorchFlowOutputPartial] = None
        self._current_state: Optional[ExpandState] = None
        self._layout_use_elk = True

        self._torch_util_path = Path(torch.utils.__file__).parent.resolve()
        self.event_after_mount.on(self._on_init)
        self.event_after_unmount.on(self._on_unmount)

        self._nested_depth = _nested_depth
        self._max_nested_depth = max_nested_depth

    async def _on_init(self):
        if not self.is_external_mode:
            appctx.get_app().add_file_resource(ELK_FILE_RESOURCE_KEY, self._elk_format_download)

    async def _on_unmount(self):
        if not self.is_external_mode:
            appctx.get_app().remove_file_resource(ELK_FILE_RESOURCE_KEY)

    async def _on_tabs_change(self, value: str):
        self._current_tabs_value = value

    def get_current_submodule(self):
        if self._external_submodule_id is not None and self._toplevel_module is not None:
            return self._toplevel_module.get_submodule(self._external_submodule_id)
        return self._toplevel_module

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

        module_may_be_sub = self.get_current_submodule()
        merged_graph_res = pth_flow.create_graph_with_expanded_modules(
            self._current_state.expanded,
            module=module_may_be_sub,
            submodule_id=ext_mod_id,
            submodule_id_is_module=self._external_ftree_id is None)
        self._cur_graph_metadata = merged_graph_res
        self.graph.event_node_context_menu.clear()
        self.graph.event_node_context_menu.on(
            partial(self._on_node_contextmenu,
                    pth_flow=pth_flow,
                    state=self._current_state))
        await self.clear_tabs()
        await self._set_graph_node_edges_and_layout(merged_graph_res.nodes,
                                                      merged_graph_res.edges,
                                                      fit_view=True)
        await self._info_container.set_new_layout([])
        with torch.device("meta"):
            mod_meta = module.to("meta")
            self._toplevel_module = mod_meta
            module_id_prefix = ""
            expand_level = 0
            btns = [IconButtonData("subflow", mui.IconType.Preview)]
            if self._external_submodule_id is not None:
                # we need to get submodule from external_submodule_id
                mod_meta = mod_meta.get_submodule(self._external_submodule_id)
                module_id_prefix = self._external_submodule_id
                expand_level = 1
                btns = mui.undefined
            await self._module_tree.set_root_object_dict(
                {
                    "":
                    PytorchModuleTreeItemEx(
                        mod_meta,
                        module_id_prefix,
                        on_lazy_expand=self._on_tree_item_lazy_expand,
                        on_button_click=self._on_tree_item_button_click,
                        btns=btns),
                },
                expand_level=expand_level,
                expand_all=True)

    async def _on_graph_ready(self):
        # do init when flow is external (e.g. created from main flow)
        if self.is_external_mode:
            assert self._toplevel_pth_flow is not None
            assert self._toplevel_module is not None
            await self._init_set_exported_flow(self._toplevel_pth_flow,
                                               self._toplevel_module)

    def _patch_module_uid(self, module_uid: str):
        if self._external_submodule_id is None or self._external_submodule_id == "":
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
                # TODO we don't clear node_id_to_data when create subflow. should we clear?
                if k in self._cur_graph_metadata.id_to_nodes:
                    if module_id is not None:
                        module_id_str = ".".join(module_id.parts)
                        if module_id_str.startswith(module_uid_to_sel):
                            selections.append(k)
                            continue
            await self.graph.select_nodes(selections)
            await self.graph.locate_nodes(selections,
                                          keep_zoom=True,
                                          duration=200)

    async def _set_graph_node_edges_and_layout(self, nodes: List[flowui.Node], edges: List[flowui.Edge], fit_view: bool = False):
        if self._layout_use_elk:
            await self.graph.set_flow_and_do_elk_layout(nodes, edges, self._elk_cfg_dialog.config, fit_view=fit_view)
        else:
            await self.graph.set_flow_and_do_dagre_layout(nodes, edges, self._dagre_cfg_dialog.config, fit_view=fit_view)

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
        # print([ev.uid, self._external_submodule_id, self._external_ftree_id, module_uid_to_sel])

        return await self._module_tree_select(module_uid_to_sel)

    async def _handle_subflow_dialog_close(self, ev: mui.DialogCloseEvent):
        await self._subflow_dialog.set_new_layout([])

    async def clear_tabs(self):
        await self._info_container.set_new_layout([])
        await self._args_tree.set_root_object_dict({})
        await self._stack_trace_container.set_new_layout([])
        await self._module_prop_tree.set_root_object_dict({})

    async def _on_tree_item_button_click(self,
                                         module_id: str,
                                         btn_key: str,
                                         is_ftree_id: bool = False):
        if btn_key == "subflow":
            if self._toplevel_pth_flow is not None and self._toplevel_module is not None:
                assert self._toplevel_pth_flow.ftree is not None
                ftree_id = None
                if is_ftree_id:
                    ftree_id = module_id
                    if module_id not in self._toplevel_pth_flow.ftree.tree_id_to_node:
                        return
                    module_id = self._toplevel_pth_flow.ftree.tree_id_to_node[
                        module_id]["module"]
                else:
                    if module_id not in self._toplevel_pth_flow.ftree.module_id_to_tree_ids:
                        # modules that con't contains forward ops
                        return
                viewer_subflow = PytorchModuleViewer(
                    external_submodule_id=module_id,
                    external_module=self._toplevel_module,
                    external_pth_flow=self._toplevel_pth_flow,
                    external_ftree_id=ftree_id, 
                    max_nested_depth=self._max_nested_depth,
                    _nested_depth=self._nested_depth + 1)
                await self._subflow_dialog.set_new_layout([
                    viewer_subflow.prop(width="100%", height="100%"),
                ])
                await self._subflow_dialog.set_open(True)

    async def _on_tree_item_lazy_expand(self, module_id: str):
        # we already set prefix to external_module_id if exists
        # so we don't need to patch here.
        if self._current_state is not None and self._toplevel_pth_flow is not None:
            new_ids = [module_id]
            for expanded_module_id in self._current_state.expanded:
                if not expanded_module_id.startswith(module_id):
                    new_ids.append(expanded_module_id)
            self._current_state.expanded = new_ids
            if self._external_ftree_id is not None:
                ext_mod_id = self._external_ftree_id
            else:
                ext_mod_id = self._external_submodule_id
            merged_graph_res = self._toplevel_pth_flow.create_graph_with_expanded_modules(
                self._current_state.expanded,
                module=self.get_current_submodule(),
                submodule_id=ext_mod_id,
                submodule_id_is_module=self._external_ftree_id is None)
            await self._set_graph_node_edges_and_layout(
                merged_graph_res.nodes, merged_graph_res.edges)
            self._cur_graph_metadata = merged_graph_res
            await self._module_tree_select(module_id)
            await self._info_container.set_new_layout([])

    async def _on_pane_contextmenu(self, data):
        item_id = data["itemId"]
        if item_id == "dagre":
            await self.graph.do_dagre_layout(self._dagre_cfg_dialog.config)
        if item_id == "elk":
            await self.graph.do_elk_layout(self._elk_cfg_dialog.config)
        if item_id == "dagre-cfg":
            await self._dagre_cfg_dialog.open_config_dialog()
        if item_id == "elk-cfg":
            await self._elk_cfg_dialog.open_config_dialog()

    async def export_module_to_flow(self,
                                    module: torch.nn.Module,
                                    args: Tuple[Any, ...],
                                    kwargs: Optional[Dict[str, Any]] = None,
                                    strict: bool = True,
                                    for_training: bool = False,
                                    verbose: bool = False,
                                    external_program: Optional[torch.export.ExportedProgram] = None):
        if self.is_external_mode:
            raise ValueError("Cannot export module to flow in external mode")
        t = time.time()
        mod_qname = get_qualname_of_type(type(module))
        if isinstance(module, (torch.fx.GraphModule,)):
            gm = module
        elif external_program is not None:
            gm = external_program
        else: 
            with torch.device("meta"):
                mod_meta = module.to("meta")
                LOGGER.warning(f"Start export {mod_qname}")
                if for_training:
                    gm = torch.export.export_for_training(mod_meta, args, kwargs, strict=strict)
                else:
                    gm = torch.export.export(mod_meta, args, kwargs, strict=strict)
        LOGGER.warning(f"Export {mod_qname} time: {time.time() - t}")
        builder = PytorchExportBuilder(use_multiple_handle_node=self._use_multiple_handle_node)
        interpreter = FlowUIInterpreter(gm, builder, module, verbose=verbose)
        outputs = interpreter.run_on_graph_placeholders()
        assert isinstance(outputs, (list, tuple))
        pth_flow = builder.build_pytorch_detached_flow(module, outputs)
        self._toplevel_pth_flow = pth_flow
        await self._init_set_exported_flow(pth_flow, module)
        return gm
        
    async def _on_node_contextmenu(self, data, pth_flow: PytorchFlowOutput,
                                   state: ExpandState):
        item_id = data["itemId"]
        node_id = data["nodeId"]
        dagre = self._dagre_cfg_dialog.config
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
                if node_meta is not None and node_meta.is_merged:
                    if not use_module_expand:
                        ftree_id = node_meta.ftree_id
                        if ftree_id is not None:
                            state.expanded.append(ftree_id)
                            merged_graph_res = pth_flow.create_graph_with_expanded_ids(
                                state.expanded)
                            await self._set_graph_node_edges_and_layout(
                                merged_graph_res.nodes, merged_graph_res.edges)
                            self._cur_graph_metadata = merged_graph_res
                            if module_id_str != "":
                                parts.insert(0, "")
                            uid_obj = UniqueTreeIdForTree.from_parts(
                                ["root", *parts]).uid_encoded
                            await self._module_tree.expand_uid(uid_obj)
                            await self._info_container.set_new_layout([])
                    else:
                        module_id = node_meta.module_id
                        if module_id is not None:
                            parts = module_id.parts.copy()
                            module_id_str = ".".join(parts)
                            state.expanded.append(module_id_str)
                            merged_graph_res = pth_flow.create_graph_with_expanded_modules(
                                state.expanded,
                                module=self.get_current_submodule(),
                                submodule_id=ext_mod_id,
                                submodule_id_is_module=self._external_ftree_id
                                is None)
                            await self._set_graph_node_edges_and_layout(
                                merged_graph_res.nodes, merged_graph_res.edges)
                            self._cur_graph_metadata = merged_graph_res
                            if self._external_submodule_id is not None:
                                num_ex_part = len(
                                    self._external_submodule_id.split("."))
                                parts = parts[num_ex_part:]
                            if module_id_str != "":
                                parts.insert(0, "")
                            uid_obj = UniqueTreeIdForTree.from_parts(
                                ["root", *parts]).uid_encoded
                                
                            await self._module_tree.expand_uid(
                                uid_obj, lazy_expand_event=False)
                            await self._info_container.set_new_layout([])
                            await self._module_tree_select(module_id_str)

        elif item_id == "subflow":
            if self._cur_graph_metadata is not None and (not self.is_external_mode or self._enable_nested_dialog):
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

    def _stacktrace_path_validator(self, path: str):
        path_resolved = Path(path).resolve()
        common_path_util_path = os.path.commonpath(
            [path_resolved, self._torch_util_path])
        if Path(common_path_util_path) == self._torch_util_path:
            return False
        return True

    async def _set_stacktrace_tab(self, stack_trace_str: str, is_merged_node: bool):
        # parse python stack trace to format: List[Tuple[(path, lineno), stmts]]
        stack_trace = parse_python_traceback(stack_trace_str, self._stacktrace_path_validator)
        stacktrace_layouts: List[Union[mui.MUIComponentBase, mui.MUIContainerBase]] = []
        if is_merged_node:
            stacktrace_layouts.append(mui.Markdown("> Stack trace of first child node."))
        for (path, lineno), lines in stack_trace:
            # keep last three parts of path
            path_last_three_parts = Path(path).parts[-3:]
            path_short = os.path.join(*path_last_three_parts)
            link = mui.Link(f"{path_short}:{lineno}").prop(isButton=True, href=mui.undefined, 
                textOverflow="ellipsis", overflow="hidden", whiteSpace="nowrap", variant="caption")
            link.event_click.on(partial(appctx.copy_text_to_clipboard, f"{path}:{lineno}"))
            stacktrace_layouts.append(mui.TooltipFlexBox(f"Copy (click): {path}:{lineno}", [
                link,
            ]).prop(enterDelay=500))
            md_lines: List[str] = []
            if lines:
                md_lines.append("```")
                for line in lines:
                    md_lines.append(line)
                md_lines.append("```")
                stacktrace_layouts.append(mui.Markdown("\n".join(md_lines)))
        await self._stack_trace_container.set_new_layout([*stacktrace_layouts])

    def _dtype_shortcut(self, dtype: torch.dtype):
        if dtype == torch.bool:
            return "bool"
        if dtype.is_floating_point:
            if dtype.is_complex:
                return f"cf{dtype.itemsize * 8}"
            if dtype == torch.bfloat16:
                return "bf16"
            else:
                return f"f{dtype.itemsize * 8}"
        else:
            if dtype.is_signed:
                return f"i{dtype.itemsize * 8}"
            else:
                return f"u{dtype.itemsize * 8}"

    def _get_node_io_layouts(self, node_id: str):
        layouts: List[Union[mui.MUIComponentBase, mui.MUIContainerBase]] = []
        if self._cur_graph_metadata is not None:
            inp_handle_to_edges = self._cur_graph_metadata.node_id_to_inp_handle_to_edges[node_id]
            out_handle_to_edges = self._cur_graph_metadata.node_id_to_out_handle_to_edges[node_id]
            for i, handle_to_edges in enumerate([inp_handle_to_edges, out_handle_to_edges]):
                btns: List[mui.Button] = []
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
                                dtype_str = self._dtype_shortcut(raw.dtype)

                                btn_name = (f"{shape}|{dtype_str}")
                            else:
                                btn_name = (f"{type_str}")
                        else:
                            btn_name = "Unknown"
                        target_node_id = edge.source if i == 0 else edge.target
                        if handle is not None:
                            btn_name = f"{handle}: {btn_name}"
                        btn = mui.Button(
                            btn_name,
                            partial(self._node_tree_select, [target_node_id]))
                        btns.append(btn.prop(loading=False))
                if btns:
                    layouts.append(mui.Typography(
                        "Inputs" if i == 0 else "Outputs").prop(variant="body1"))
                    layouts.append(mui.ButtonGroup(btns).prop(fullWidth=True, size="small", variant="outlined", orientation="vertical"))
        return layouts

    def _get_node_desp_layouts(self, data: PytorchNodeMeta, module_id_str: str, module: torch.nn.Module):
        qname = data.module_qname
        assert qname is not None 
        layouts: List[Union[mui.MUIComponentBase, mui.MUIContainerBase]] = []
        copy_data = None
        try:
            if compat.Python3_13AndLater:
                lineno = type(module).__firstlineno__ # type: ignore
            else:
                _, lineno = inspect.getsourcelines(type(module))
            path = inspect.getabsfile(type(module))
            Path(path).exists()
            copy_data = f"{path}:({lineno})"
        except:
            pass
        layouts.append(
            mui.Markdown(f":deepskyblue[`{qname}`]"))
        if data.is_merged:
            id_or_op_md = mui.Markdown(f"`id`: `{module_id_str}`")
        else:
            if data.op_sig is not None:
                id_or_op_md = mui.TooltipFlexBox(data.op_sig, [
                    mui.Markdown(f":forestgreen[`{data.op}`]")
                ])
            else:
                id_or_op_md = mui.Markdown(f":forestgreen[`{data.op}`]")
        if copy_data is not None:
            btn = mui.IconButton(mui.IconType.ContentCopy, partial(appctx.copy_text_to_clipboard, copy_data))
            btn.prop(size="small", iconSize="small")
            layouts.append(
                mui.HBox([
                    btn,
                    id_or_op_md,
                ]))
        else:
            layouts.append(
                id_or_op_md)

        is_seq = "Sequential" in qname
        is_module_list = "ModuleList" in qname
        is_module_dict = "ModuleDict" in qname
        is_container = is_seq or is_module_list or is_module_dict
        recurse = qname.startswith("torch.") and not is_container
        # official torch module
        param_md_lines: List[str] = []
        for name, param in module.named_parameters(recurse=recurse):
            shape_str = ",".join(map(str, param.shape))
            param_md_lines.append(
                f"* `{name}(P)`: `[{shape_str}]`")
        for name, param in module.named_buffers(recurse=recurse):
            shape_str = ",".join(map(str, param.shape))
            param_md_lines.append(
                f"* `{name}(B)`: `[{shape_str}]`")
        if param_md_lines:
            layouts.append(mui.Divider())
            layouts.append(
                mui.Markdown("\n".join(param_md_lines)))
            
        return layouts

    async def _on_selection_change(self, ev: flowui.EventSelection):
        if ev.nodes and len(ev.nodes) == 1:
            node_id = ev.nodes[0]
            if self._cur_graph_metadata is not None:
                node = self._cur_graph_metadata.id_to_nodes[node_id]
                node_json = as_dict_no_undefined(node)
                inp_edges: List[flowui.Edge] = []
                for edges in self._cur_graph_metadata.node_id_to_inp_handle_to_edges[node_id].values():
                    inp_edges.extend(edges)
                out_edges: List[flowui.Edge] = []
                for edges in self._cur_graph_metadata.node_id_to_out_handle_to_edges[node_id].values():
                    out_edges.extend(edges)
                await self._debug_json_tree.write({
                    "node": node_json,
                    "inp_edges": [as_dict_no_undefined(e) for e in inp_edges],
                    "out_edges": [as_dict_no_undefined(e) for e in out_edges],
                })
            cur_submodule = self.get_current_submodule()

            if self._cur_graph_metadata is not None and cur_submodule is not None:
                if node_id in self._cur_graph_metadata.node_id_to_data:
                    tab_is_empty = {x.value: False for x in TabType}
                    node = self._cur_graph_metadata.id_to_nodes[node_id]
                    data = self._cur_graph_metadata.node_id_to_data[node_id]
                    layouts: List[Union[mui.MUIComponentBase, mui.MUIContainerBase]] = []
                    module_id = data.module_id
                    qname = data.module_qname
                    if data.additional_args is not None and len(data.additional_args) > 0:
                        await self._args_tree.set_root_object_dict(data.additional_args, expand_all=True)
                        await self._args_tree.expand_all()
                    else:
                        await self._args_tree.set_root_object_dict({})
                        tab_is_empty[TabType.ARGS.value] = True
                    if data.stack_trace is not None:
                        await self._set_stacktrace_tab(data.stack_trace, data.is_merged)
                    else:
                        await self._stack_trace_container.set_new_layout([])
                        # if stacktrace not available, reset tabs to info if stacktrace is selected
                        tab_is_empty[TabType.STACKTRACE.value] = True
                    should_clear_module_prop_tree = True
                    if data.is_io_node:
                        if data.output_desps is not None:
                            out = data.output_desps[0]
                            type_str, shape = self._get_shape_type_from_raw(
                                out)
                            
                            if not isinstance(node.data, mui.Undefined) and not isinstance(node.data.label, mui.Undefined):
                                if type_str == "Tensor":
                                    dtype_shortcut = self._dtype_shortcut(out.dtype)
                                    layouts.append(mui.Markdown(f"`{node.data.label}`: `{type_str}|{dtype_shortcut}`"))
                                else:
                                    layouts.append(mui.Markdown(f"`{node.data.label}`: `{type_str}`"))
                            else:
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
                                module = cur_submodule.get_submodule(module_id_str)
                                if data.is_merged:
                                    should_clear_module_prop_tree = False
                                    await self._module_prop_tree.set_root_object_dict({module_id.parts[-1]: module}, expand_all=True)
                                node_desp_layouts = self._get_node_desp_layouts(
                                    data, module_id_str, module)
                                layouts.extend(node_desp_layouts)
                            except AttributeError:
                                pass
                    io_layouts = self._get_node_io_layouts(node_id)
                    if io_layouts:
                        layouts.append(mui.Divider())
                        layouts.extend(io_layouts)
                    if should_clear_module_prop_tree:
                        await self._module_prop_tree.set_root_object_dict({})
                    if layouts:
                        await self._info_container.set_new_layout([*layouts])
                    if self._tabs.props.value != self._current_tabs_value:
                        # switch to previous user clicked tab if not empty
                        if not tab_is_empty[self._current_tabs_value]:
                            await self._tabs.set_value(self._current_tabs_value)
                        elif tab_is_empty[self._tabs.props.value]:
                            await self._tabs.set_value(TabType.INFO.value)
                    else:
                        if tab_is_empty[self._tabs.props.value]:
                            await self._tabs.set_value(TabType.INFO.value)
                    return 
            
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
                        mui.Markdown(f"`{edge_id}`"),
                        mui.Markdown(f"`{shape}`"),
                        mui.Divider(),
                        mui.Button(
                            "Input",
                            partial(self._node_tree_select, ([edge.source]))),
                        mui.Button(
                            "Output",
                            partial(self._node_tree_select, ([edge.target]))),
                    ])
                    return
        await self._info_container.set_new_layout([])
        await self._args_tree.set_root_object_dict({})
        await self._stack_trace_container.set_new_layout([])

    def _elk_format_download(self, req: mui.FileResourceRequest):
        if self._cur_graph_metadata is not None:
            nodes = self._cur_graph_metadata.nodes 
            edges = self._cur_graph_metadata.edges
            elk = {
                "id": "root",
                "layoutOptions": {
                    "elk.algorithm": "layered",
                    "elk.direction": "DOWN",
                },
                "children": [],
                "edges": [],
            }
            fixed_order = True
            for n in nodes:
                inp_handle_to_edges = self._cur_graph_metadata.node_id_to_inp_handle_to_edges[n.id]
                out_handle_to_edges = self._cur_graph_metadata.node_id_to_out_handle_to_edges[n.id]
                ports = []
                handle_idx = 0
                for handle, _ in inp_handle_to_edges.items():
                    ports.append({
                        "id": f"{n.id}_PORT_in_{handle}",
                        # "properties": {
                        #     "side": "NORTH",
                        #     "index": handle_idx,
                        # }
                        "labels": [
                            {
                                "text": f"{handle}",
                            }
                        ]
                    })
                    if fixed_order:
                        ports[-1]["properties"] = {
                            "side": "NORTH",
                            "index": handle_idx,
                        }
                    handle_idx += 1
                for handle, _ in out_handle_to_edges.items():
                    ports.append({
                        "id": f"{n.id}_PORT_out_{handle}",
                        # "properties": {
                        #     "side": "SOUTH",
                        #     "index": handle_idx,
                        # }
                        "labels": [
                            {
                                "text": f"{handle}",
                            }
                        ]
                    })
                    if fixed_order:
                        ports[-1]["properties"] = {
                            "side": "SOUTH",
                            "index": handle_idx,
                        }
                    handle_idx += 1

                elk["children"].append({
                    "id": n.id,
                    "width": 150,
                    "height": 40,
                    "ports": ports,
                    # "properties": {
                    #     "org.eclipse.elk.portConstraints": "FIXED_ORDER",
                    # }
                })
                if fixed_order:
                    elk["children"][-1]["properties"] = {
                        "org.eclipse.elk.portConstraints": "FIXED_ORDER",
                    }
                node_data = n.get_node_data()
                if node_data is not None:
                    elk["children"][-1]["labels"] = [
                        {
                            "text": node_data.label,
                        }
                    ]
            for e in edges:
                elk["edges"].append({
                    "id": e.id,
                    "source": e.source,
                    "target": e.target,
                    "sourcePort": f"{e.source}_PORT_out_{e.sourceHandle}",
                    "targetPort": f"{e.target}_PORT_in_{e.targetHandle}",
                })
            elk_binary = json.dumps(elk, indent=2).encode()
            return mui.FileResource(name=ELK_FILE_RESOURCE_KEY, content=elk_binary)

        return mui.FileResource.empty()


    async def _on_dagre_layout_with_cfg(self, cfg_ev: ConfigDialogEvent[flowui.DagreLayoutOptions]):
        await self.graph.do_dagre_layout(cfg_ev.cfg) 