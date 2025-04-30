from tensorpc.dock import (marker, mui,
                           chart, plus, three, appctx)
from tensorpc.apps.dbg.components.dbgpanel import MasterDebugPanel

class DebugPanel:
    @marker.mark_create_layout
    def my_layout(self):
        return mui.VBox([
            MasterDebugPanel().prop(flex=1),
        ]).prop(width="100%", overflow="hidden")
