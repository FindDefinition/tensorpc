from pathlib import Path
from tensorpc.dock import mui, three, plus, mark_create_layout, appctx


class App:
    @mark_create_layout
    def my_layout(self):
        code = Path(__file__).read_text()
        self.editor = mui.MonacoEditor(code, "python", "dev").prop(flex=1, options=mui.MonacoEditorOptions(glyphMargin=True))
        return mui.VBox([
            mui.Button("Set Bkpt", self._on_btn),
            self.editor,
        ]).prop(height="100%", width="100%", overflow="hidden")

    async def _on_btn(self):
        await self.editor.set_breakpoints([
            mui.MonacoBreakpoint(True, 12)
        ])