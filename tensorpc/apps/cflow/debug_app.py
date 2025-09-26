import asyncio
import json
from tensorpc.apps.cflow.coremodel import ResourceDesc
from tensorpc.apps.cflow.executors.base import ExecutorRemoteDesc, ExecutorType
from tensorpc.apps.cflow.flow import ComputeFlow
from tensorpc.apps.cflow.executors.simple import SSHCreationNodeExecutor
from tensorpc.apps.cflow.nodes.defaultnodes import _SSHRunnerState
from tensorpc.autossh.core import SSHConnDesc
from tensorpc.constants import PACKAGE_ROOT, TENSORPC_DEV_SECRET_PATH
from tensorpc.dock import mark_create_layout
import yaml
from tensorpc.apps.cflow.nodes import register_compute_node, get_node_state_draft, ComputeNodeBase, SpecialHandleDict
from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.dock.components import mui, three
from typing import TypedDict, Any

from tensorpc.dock.components.models.flow import BaseEdgeModel
from tensorpc.dock.marker import mark_did_mount

@dataclasses.dataclass
class _DelayState:
    value: str = "10"

def _json_input_layout(drafts):
    editor = mui.SimpleCodeEditor("0", "json")
    if drafts is not None:
        # FIXME: fix this
        editor.bind_draft_change_uncontrolled(drafts.value)
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
    print("delay data", data)
    num_step = int(data * 10)
    for j in range(num_step):
        await asyncio.sleep(0.1)
        print("delay step", j)
    return {'x': x}

def _3d_layout(drafts):
    cam = three.PerspectiveCamera(fov=75, near=0.1, far=1000)
    cam.prop(position=(0, 0, 5))

    view = three.View([
        cam,
        three.CameraControl().prop(makeDefault=True),
        three.Mesh([
            three.BoxGeometry(),
            three.MeshBasicMaterial().prop(color="orange",
                                            transparent=True),
        ]),
    ]).prop(flex=1, overflow="hidden")

    return mui.VBox([view
                         ]).prop(width="300px",
                                 height="300px",
                                 overflow="hidden")


@register_compute_node(key="3D Test",
                       name="3D Test",
                       icon_cfg=mui.IconProps(icon=mui.IconType.DataObject),
                       layout_creator=_3d_layout)
async def d3_test() -> None:
    return None

class ComputeFlowApp:
    @mark_create_layout
    def my_layout(self):
        with open(TENSORPC_DEV_SECRET_PATH, "r") as f:
            secret = yaml.safe_load(f)["cflow_debug"]
        init_cmd = ""
        if "init_cmd" in secret:
            init_cmd = f"{secret['init_cmd']}\n"
        port = secret.get("port", 22)
        self.ssh_desc = SSHConnDesc(
            f"localhost:{port}", secret["username"], secret["password"], init_cmd=init_cmd)
        executors = [
            SSHCreationNodeExecutor("remote", ResourceDesc(), self.ssh_desc, [
                init_cmd
            ])
        ]
        self.cflow = ComputeFlow(executors=executors)
        return self.cflow

    @mark_did_mount
    async def _mount(self):
        # create debug flow for fast evaluation
        draft = self.cflow.dm.get_draft_type_only()
        async with self.cflow.dm.draft_update():
            jinput = self.cflow._debug_add_sys_node("tensorpc.cflow.Json", (100, 200))
            delay = self.cflow._debug_add_sys_node("Delay", (400, 200))
            output = self.cflow._debug_add_sys_node("tensorpc.cflow.ObjectTreeViewer", (700, 200))
            ssh = self.cflow._debug_add_sys_node("tensorpc.cflow.SSHRunner", (1000, 200))

            delay.vExecId = "remote"
            self.cflow._debug_add_edge(BaseEdgeModel("1", jinput.id, delay.id, "out-json", "inp-x"))
            self.cflow._debug_add_edge(BaseEdgeModel("2", delay.id, output.id, "out-x", "specialdict-obj"))
            draft.selected_node = delay.id
            draft.node_states[ssh.id] = _SSHRunnerState(self.ssh_desc.url_with_port, self.ssh_desc.username, self.ssh_desc.password)
