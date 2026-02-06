from tensorpc.dock import mui, three, plus, appctx, mark_create_layout, flowui, models

from typing import Callable, Coroutine, Literal, Optional, Any
import rich 
import sys 

class RefNodeSelectDialog(mui.Dialog):
    def __init__(self, on_click: Callable[[str, tuple[mui.NumberType, mui.NumberType], Optional[str]], Coroutine[Any, Any, None]]):
        self.title = mui.Typography().prop(variant="body1")
        self.qname = mui.Typography().prop(variant="caption", enableTooltipWhenOverflow=True)


        self.info_icon = mui.Icon()
        self.title.bind_fields(value="title")
        self.qname.bind_fields(value="qname")
        self.info_icon.bind_fields(tooltip="info")

        self.card = mui.Paper([
            self.title,
            self.qname,
            self.info_icon
        ]).prop(width="150px", height="150px", margin="10px", flexFlow="column",)
        # card get bigger when hover
        self.card.update_raw_props({
            ":hover": {
                "transform": "scale(1.05)",
                "transition": "transform 0.2s",
            }
        })
        self.card.event_click.on_standard(self._handle_click)
        self.inline_flow_select = mui.Autocomplete("inline flow", []).prop(textFieldProps=mui.TextFieldProps(size="small", muiMargin="dense"))
        self.content = mui.DataFlexBox(self.card).prop(flex=1, overflowY="auto", flexFlow="row wrap",
            filterKey="qname", flexDirection="row")
        self.search = mui.TextField("nodes").prop(size="small", muiMargin="dense")
        self.search.prop(valueChangeTarget=(self.content, "filter"))
        self._on_click = on_click
        super().__init__([
            mui.HBox([
                self.search.prop(flex=1),
                self.inline_flow_select.prop(flex=1),
            ]),
            self.content,
        ])
        self.prop(overflow="hidden", title="Create Ref Node", 
            display="flex", dialogMaxWidth=False, fullWidth=False,
            width="75vw", height="75vh", includeFormControl=False, flexDirection="column")

        self.position: tuple[mui.NumberType, mui.NumberType] = (0, 0)

    async def _handle_click(self, event: mui.Event):
        gid = event.get_keys_checked()[0]
        inlineflow_name = None 
        if self.inline_flow_select.value is not None:
            inlineflow_name = self.inline_flow_select.value["label"]
        try:
            await self._on_click(gid, self.position, inlineflow_name)
        finally:
            await self.set_open(False)
            await self.send_and_wait(self.inline_flow_select.update_event(value=None))
