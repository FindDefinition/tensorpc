from typing import Annotated, Any, Callable, Literal, Mapping, Optional, Self, Union, cast
from tensorpc.core.datamodel import typemetas
import tensorpc.core.dataclass_dispatch as dataclasses
import enum
from tensorpc.apps.adv.model import (ADVPaneContextMenu, ADVNodeFlags, ADVNodeModel, ADVNodeType)
from tensorpc.dock import mui


@dataclasses.dataclass
class ADVNewNodeConfig:
    name: str



class _CMCfgField(enum.IntFlag):
    NAME = enum.auto()
    INLINE_FLOW_NAME = enum.auto()
    ALIAS_MAP = enum.auto()
    FUNC_BASE_TYPE = enum.auto()
    IS_DATACLASS = enum.auto()


@dataclasses.dataclass
class ADVNodeCMConfig:
    name: str
    inline_flow_name: Annotated[str, typemetas.DynamicEnum(alias="Inline Flow Name")]
    alias_map: Annotated[str, typemetas.CommonObject(alias="Output Alias", tooltip="e.g. n1->new1,n2->new2")]
    func_base_type: Annotated[Literal["Global", "Method", "Class Method", "Static Method"], typemetas.CommonObject(alias="Function Base Type")] = "Global"
    is_dataclass: Annotated[bool, typemetas.CommonObject(alias="Is Dataclass")] = False

    @staticmethod
    def _get_pane_act_exc_fields(pane_act: ADVPaneContextMenu, flow_node_type: ADVNodeType):
        base_field_flag = _CMCfgField.NAME
        pane_act_field_map = {
            ADVPaneContextMenu.AddFragment: _CMCfgField.INLINE_FLOW_NAME,
            ADVPaneContextMenu.AddNestedFragment: _CMCfgField.INLINE_FLOW_NAME,
            ADVPaneContextMenu.AddGlobalScript: _CMCfgField(0),
            ADVPaneContextMenu.AddClass: _CMCfgField.INLINE_FLOW_NAME | _CMCfgField.IS_DATACLASS | _CMCfgField.ALIAS_MAP,
            ADVPaneContextMenu.AddSymbolGroup: _CMCfgField(0),
            ADVPaneContextMenu.AddOutput: _CMCfgField(0),
            ADVPaneContextMenu.AddMarkdown: _CMCfgField(0),
            ADVPaneContextMenu.AddInlineFlowDesc: _CMCfgField(0),
        } 
        if flow_node_type == ADVNodeType.CLASS:
            # regular fragments don't have special type.
            pane_act_field_map[ADVPaneContextMenu.AddFragment] |= _CMCfgField.FUNC_BASE_TYPE
        valid_fields = pane_act_field_map[pane_act] | base_field_flag

        exc_fields: list[str] = []
        for field in _CMCfgField:
            if not (valid_fields & field):
                if field == _CMCfgField.NAME:
                    exc_fields.append("name")
                elif field == _CMCfgField.INLINE_FLOW_NAME:
                    exc_fields.append("inline_flow_name")
                elif field == _CMCfgField.ALIAS_MAP:
                    exc_fields.append("alias_map")
                elif field == _CMCfgField.FUNC_BASE_TYPE:
                    exc_fields.append("func_base_type")
                elif field == _CMCfgField.IS_DATACLASS:
                    exc_fields.append("is_dataclass")
        return exc_fields

    @staticmethod
    def from_pane_action(pane_act: ADVPaneContextMenu, flow_node_type: ADVNodeType) -> tuple["ADVNodeCMConfig", ADVNodeType, list[str]]:
        excluded_fields: list[str] = ADVNodeCMConfig._get_pane_act_exc_fields(pane_act, flow_node_type)
        pane_act_enum = ADVPaneContextMenu(pane_act)
        if pane_act_enum == ADVPaneContextMenu.AddFragment or pane_act_enum == ADVPaneContextMenu.AddNestedFragment:
            ntype = ADVNodeType.FRAGMENT 
            if pane_act_enum == ADVPaneContextMenu.AddNestedFragment:
                name_default = "new_subflow"
            else:
                name_default = "new_fragment"
        elif pane_act_enum == ADVPaneContextMenu.AddGlobalScript:
            ntype = ADVNodeType.GLOBAL_SCRIPT
            name_default = "new_global_script"
        elif pane_act_enum == ADVPaneContextMenu.AddSymbolGroup:
            ntype = ADVNodeType.SYMBOLS
            name_default = "NewSymbolGroup"
        elif pane_act_enum == ADVPaneContextMenu.AddOutput:
            name_default = ""
            ntype = ADVNodeType.OUT_INDICATOR
        elif pane_act_enum == ADVPaneContextMenu.AddMarkdown:
            name_default = ""
            ntype = ADVNodeType.MARKDOWN
        elif pane_act_enum == ADVPaneContextMenu.AddInlineFlowDesc:
            name_default = "new_inline_flow"
            ntype = ADVNodeType.FRAGMENT
        elif pane_act_enum == ADVPaneContextMenu.AddClass:
            ntype = ADVNodeType.CLASS
            name_default = "NewClass"

        else:
            raise NotImplementedError

        res_cfg = ADVNodeCMConfig(
            name=name_default,
            inline_flow_name="",
            alias_map="",
        )

        return res_cfg, ntype, excluded_fields

    @staticmethod
    def from_node(node: ADVNodeModel, parent_node_type: ADVNodeType) -> tuple["ADVNodeCMConfig", list[str]]:
        res = ADVNodeCMConfig(
            name=node.name,
            inline_flow_name=node.inlinesf_name or "",
            alias_map=node.alias_map,
            func_base_type="Global"
        ) 
        if node.flags & int(ADVNodeFlags.IS_CLASSMETHOD):
            res.func_base_type = "Class Method"
        elif node.flags & int(ADVNodeFlags.IS_STATICMETHOD):
            res.func_base_type = "Static Method"
        elif node.flags & int(ADVNodeFlags.IS_METHOD):
            res.func_base_type = "Method"
        excluded_fields: list[str] = []
        if node.nType != ADVNodeType.CLASS:
            excluded_fields.append("is_dataclass")
        if node.nType != ADVNodeType.FRAGMENT:
            if node.nType == ADVNodeType.CLASS:
                excluded_fields += ["func_base_type"]
            else:
                excluded_fields += ["func_base_type", "alias_map", "inline_flow_name"]
        else:
            if node.ref is None:
                # node def don't have alias map
                excluded_fields += ["alias_map"]
            else:
                # node ref can't change method type
                excluded_fields += ["func_base_type"]
        if parent_node_type != ADVNodeType.CLASS:
            excluded_fields.append("func_base_type")
        if node.ref is not None:
            excluded_fields.extend(["is_dataclass", "func_base_type"])

        excluded_fields = list(set(excluded_fields))
        return res, excluded_fields


