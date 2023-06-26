import asyncio
import dataclasses
import enum
import inspect
import os
import time
from typing import Any, Callable, Coroutine, Dict, Iterable, List, Optional, Set, Tuple, Union
from typing_extensions import Literal
from tensorpc.core.serviceunit import AppFuncType, ServFunctionMeta
from tensorpc.flow.flowapp import appctx

from tensorpc.flow.flowapp.components import mui, three

from tensorpc.constants import TENSORPC_FILE_NAME_PREFIX
from tensorpc.flow.marker import mark_did_mount, mark_will_unmount
from tensorpc.flow.flowapp.core import (_get_obj_def_path)

@dataclasses.dataclass
class MarkdownBlock:
    content: str
    type: Literal["markdown", "code"] = "markdown"

def _parse_markdown_very_trivial(content: str):
    """this function only check ```Python ``` block, then split
    markdown into several markdown blocks and code blocks.
    """
    res_blocks: List[MarkdownBlock] = []
    remain_code_index = 0
    code_block_prefix = "```Python"
    code_block_suffix = "```"
    while True:
        code_block_start = content.find(code_block_prefix, remain_code_index)
        if code_block_start == -1:
            res_blocks.append(MarkdownBlock(content[remain_code_index:], "markdown"))
            break
        code_block_end = content.find(code_block_suffix, code_block_start + len(code_block_prefix))
        if code_block_end == -1:
            res_blocks.append(MarkdownBlock(content[remain_code_index:], "markdown"))
            break
        res_blocks.append(MarkdownBlock(content[remain_code_index:code_block_start], "markdown"))
        res_blocks.append(MarkdownBlock(content[code_block_start + len(code_block_prefix):code_block_end], "code"))
        remain_code_index = code_block_end + len(code_block_suffix)
    return res_blocks

class AppInMemory(mui.FlexBox):
    """app with editor (app must be anylayout)
    """
    def __init__(self, path: str, code: str, is_horizontal: bool = True):
        wrapped_path = f"<{TENSORPC_FILE_NAME_PREFIX}-{path}>"
        self.editor = mui.MonacoEditor(code, "python", wrapped_path).prop(minWidth=0, minHeight=0)
        self.path = wrapped_path 
        self.code = code 
        self.app_cls_name = "App"
        self.show_box = mui.FlexBox()
        super().__init__([
            self.editor.prop(flex=1),
            mui.Divider("horizontal" if is_horizontal else "vertical"),
            self.show_box.prop(flex=1),
        ])
        self._layout_for_reload: Optional[mui.FlexBox] = None
        self.prop(flexFlow="row" if is_horizontal else "column")
        self.editor.event_editor_save.on(self._on_editor_save)

    @mark_did_mount
    async def _on_mount(self):
        reload_mgr = self.flow_app_comp_core.reload_mgr
        reload_mgr.in_memory_fs.add_file(self.path, self.code)
        mod = reload_mgr.in_memory_fs.load_in_memory_module(self.path)
        app_cls = mod.__dict__[self.app_cls_name]
        layout = mui.flex_wrapper(app_cls())
        self._layout_for_reload = layout
        await self.show_box.update_childs({"layout": layout})
        appctx.get_editable_app()._flowapp_observe(layout, self._handle_reload_layout)

    @mark_will_unmount
    async def _on_unmount(self):
        if self._layout_for_reload is not None:
            appctx.get_editable_app()._flowapp_remove_observer(self._layout_for_reload)

    async def _handle_reload_layout(self, layout: mui.FlexBox,
                                    create_layout: ServFunctionMeta):
        # if create_layout.user_app_meta is not None and create_layout.user_app_meta.type == AppFuncType.CreateLayout:
        layout_flex = create_layout.get_binded_fn()()
        assert isinstance(
            layout_flex, mui.FlexBox
        ), f"create_layout must return a flexbox when use anylayout"
        layout_flex.set_wrapped_obj(layout.get_wrapped_obj())
        await self.show_box.update_childs({"layout": layout_flex})

    async def _on_editor_save(self, value: str):
        reload_mgr = self.flow_app_comp_core.reload_mgr
        reload_mgr.in_memory_fs.modify_file(self.path, value)
        await appctx.get_editable_app()._reload_object_with_new_code(self.path, value)

class MarkdownTutorial(mui.VirtualizedBox):
    """ this component parse markdowns in a very simple way, don't use it in production, it's only for tutorials.
    """
    def __init__(self, md_content: str, path_uid: str):
        res_blocks = _parse_markdown_very_trivial(md_content)
        layout: mui.LayoutType = []
        for i, block in enumerate(res_blocks):
            if block.type == "markdown":
                if block.content.strip() == "":
                    continue
                layout.append(mui.Markdown(block.content))
            elif block.type == "code":

                layout.append(AppInMemory(f"{path_uid}-{i}", block.content.lstrip()).prop(minHeight="400px", padding="10px"))
        super().__init__(layout)
        self.prop(flexFlow="column", padding="10px")

