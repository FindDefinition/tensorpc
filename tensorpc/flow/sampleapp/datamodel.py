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
        # this draft is only used for binding fields, don't use it to update values
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
        draft = self.dm.get_draft()
        draft.count += 1
