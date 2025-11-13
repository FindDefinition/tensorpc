from tensorpc.dock import (marker, mui,
                           chart, plus, three, appctx)
from tensorpc.apps.dbg.components.dbgpanel import MasterDebugPanel

class DebugPanel:
    @marker.mark_create_layout
    def my_layout(self):
        self.panel = MasterDebugPanel()
        return mui.VBox([
            self.panel.prop(flex=1),
        ]).prop(width="100%", overflow="hidden")


    async def external_set_perfetto_data(self, data: bytes, all_timestamps: list[int], key: str):

        await self.panel.external_set_perfetto_data(data, all_timestamps, key)
