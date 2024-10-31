from tensorpc.flow import (marker, mui,
                           chart, plus, three, appctx)


class DebugPanel:
    @marker.mark_create_layout
    def my_layout(self):
        return mui.VBox([
            plus.MasterDebugPanel().prop(flex=1),
        ]).prop(width="100%", overflow="hidden")
