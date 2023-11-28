
from tensorpc.flow import (App, EditableApp, EditableLayoutApp, leaflet,
                           mark_autorun, mark_create_layout, marker, mui,
                           plotly, plus, three, UserObjTree, appctx, V)


class TestPreview0:
    def __init__(self) -> None:
        super().__init__()

    @marker.mark_create_preview_layout
    def layout_func(self):
        return mui.VBox([
            mui.Button("WTF"),
            mui.Markdown("## 6")
        ])

