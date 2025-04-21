from tensorpc.apps.distssh.constants import TENSORPC_DISTSSH_UI_KEY
from tensorpc.constants import TENSORPC_APPS_DISTSSH_DEFAULT_PORT
from tensorpc.dock import mui, three, plus, mark_create_layout, appctx
class FTSSHDevApp:
    @mark_create_layout
    def my_layout(self):
        remote_box = mui.RemoteBoxGrpc("10.21.80.73", 59999, TENSORPC_DISTSSH_UI_KEY)
        return mui.VBox([
            remote_box.prop(flex=1)
        ]).prop(width="100%", height="100%", overflow="hidden")

