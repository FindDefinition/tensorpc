import asyncio
import json
from tensorpc.apps.cflow.coremodel import ResourceDesp
from tensorpc.apps.cflow.executors.base import ExecutorRemoteDesp, ExecutorType
from tensorpc.apps.cflow.flow import ComputeFlow
from tensorpc.apps.cflow.executors.simple import SSHCreationNodeExecutor
from tensorpc.dock import mark_create_layout

from tensorpc.apps.cflow.nodes import register_compute_node, get_node_state_draft, ComputeNodeBase, SpecialHandleDict
from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.dock.components import mui
from typing import TypedDict, Any

@dataclasses.dataclass
class _DelayState:
    value: str = "1"

def _json_input_layout(drafts):
    editor = mui.SimpleCodeEditor("0", "json")
    if drafts is not None:
        # FIXME: fix this
        editor.bind_draft_change(drafts.value)
    return mui.VBox([editor.prop(editorPadding=5)
                         ]).prop(width="200px",
                                 maxHeight="300px",
                                 overflow="auto")
class _JsonOutputDict(TypedDict):
    x: Any

@register_compute_node(key="Delay",
                       name="Delay",
                       icon_cfg=mui.IconProps(icon=mui.IconType.DataObject),
                       layout_creator=_json_input_layout,
                       state_dcls=_DelayState)
async def delay_node(x) -> _JsonOutputDict:
    state, draft = get_node_state_draft(_DelayState)
    data = json.loads(state.value)
    await asyncio.sleep(data)
    return {'x': x}


class ComputeFlowApp:
    @mark_create_layout
    def my_layout(self):
        # appctx.get_app().set_enable_language_server(True)
        # pyright_setting = appctx.get_app().get_language_server_settings()
        # pyright_setting.python.analysis.pythonPath = sys.executable
        # pyright_setting.python.analysis.extraPaths = [
        #     str(PACKAGE_ROOT.parent),
        # ]
        executors = [
            SSHCreationNodeExecutor("remote", ResourceDesp(), "localhost:22", "root", "1", [
                "conda activate base\n"
            ])
        ]
        self.cflow = ComputeFlow(executors=executors)
        return self.cflow

