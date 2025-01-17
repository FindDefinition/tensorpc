from tensorpc.flow import mui, marker
import dataclasses


@dataclasses.dataclass
class Model:
    name: str 
    count: int 

class App:
    @marker.mark_create_layout
    def layout(self):
        model = Model("test", 1)
        model_draft = mui.DataModel.get_draft_external(model)
        self.dm = mui.DataModel(Model("test", 1), [
            mui.Markdown().bind_fields(value=model_draft.name),
            mui.Markdown().bind_fields(value=f"to_string({model_draft.count})"),
            mui.Button("IncCount", self._handle_button)
        ])
        return mui.VBox([
            self.dm
        ])

    async def _handle_button(self):
        async with self.dm.draft_update() as draft:
            draft.count += 1