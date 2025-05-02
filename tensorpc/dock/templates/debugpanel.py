from tensorpc.apps.dbg.components.dbgpanel import MasterDebugPanel
from tensorpc.dock import mui, three, plus, mark_create_layout, appctx

class App:
    @mark_create_layout
    def my_layout(self):
        return MasterDebugPanel()
