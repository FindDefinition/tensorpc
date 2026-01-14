
from tensorpc.dock import (App, EditableApp, EditableLayoutApp, leaflet,
                           mark_autorun, mark_create_layout, marker, mui,
                           chart, plus, three, UserObjTree, appctx, V)


_CONSTRAINED_CODE = """utils = {};
def addKeysToUtils():
    '''Enter the content for the function here'''

addKeysToUtils()
"""

class App:
    @mark_create_layout
    def my_layout(self):
        self.editor = mui.MonacoEditor(_CONSTRAINED_CODE, "python",
                                       "default_path").prop(minWidth=0, minHeight=0, flex=1)
        self.editor.prop(enableConstrainedEditing=True)
        self.editor.prop(constrainedRanges=[
            mui.MonacoConstrainedRange(
                range=(3, 1, 3, 50),
                label="Function Definition",
                allowMultiline=True
            ),
            mui.MonacoConstrainedRange(
                range=(1, 1, 1, 6),
                label="utilName",
            ),

        ])
        editor_acts: list[mui.MonacoEditorAction] = [
            mui.MonacoEditorAction(id="ToggleEditableAreas", 
                label="Toggle Editable Areas", contextMenuOrder=1.5,
                contextMenuGroupId="tensorpc-editor-action", 
            ),
        ]
        self.editor.prop(actions=editor_acts)
        self.editor.event_editor_save.on(lambda ev: print(ev))
        self.editor.event_editor_action.on(self._handle_editor_acts)
        res = mui.HBox([
            self.editor
        ]).prop(width="100%", height="100%", overflow="hidden")
        return res 

    async def _handle_editor_acts(self, act: mui.MonacoActionEvent):
        if act.action == "ToggleEditableAreas":
            await self.editor.toggle_editable_areas()
