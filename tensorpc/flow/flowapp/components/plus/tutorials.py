import asyncio
import dataclasses
import enum
import inspect
from operator import is_
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
import yaml

@dataclasses.dataclass
class MarkdownMetadata:
    type: Optional[Literal["Canvas", "Notebook"]] = None

@dataclasses.dataclass
class MarkdownBlock:
    content: str
    type: Literal["markdown", "code"] = "markdown"

def _parse_markdown_very_trivial(content: str):
    """this function only check ```Python ``` block, then split
    markdown into several markdown blocks and code blocks.
    """
    # find comment firstly
    comment_prefix = "<!--"
    comment_suffix = "-->"
    comment_start = content.find(comment_prefix)
    comment_end = content.find(comment_suffix, comment_start + len(comment_prefix))
    metadata = MarkdownMetadata()
    if comment_start != -1 and comment_end != -1:
        yaml_str = content[comment_start + len(comment_prefix):comment_end]
        yaml_data = yaml.safe_load(yaml_str)
        metadata = MarkdownMetadata(**yaml_data)
        # remove comment
        content = content[:comment_start] + content[comment_end + len(comment_suffix):]
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
    return res_blocks, metadata


class AppInMemory(mui.FlexBox):
    """app with editor (app must be anylayout)
    """
    # @dataclasses.dataclass
    class Config:
        is_horizontal: bool = True
        height: Union[mui.ValueType, mui.Undefined] = mui.undefined

    def __init__(self, path: str, code: str, is_horizontal: bool = True,
            external_onsave: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None):
        wrapped_path = f"<{TENSORPC_FILE_NAME_PREFIX}-{path}>"
        self.editor = mui.MonacoEditor(code, "python", wrapped_path).prop(minWidth=0, minHeight=0)
        self.path = wrapped_path 
        self.code = code 
        self.app_cls_name = "App"
        self.show_box = mui.FlexBox().prop(overflowY="auto")
        self.divider = mui.Divider("horizontal" if is_horizontal else "vertical")
        super().__init__([
            self.editor.prop(flex=1),
            self.divider,
            self.show_box.prop(flex=1),
        ])
        self._layout_for_reload: Optional[mui.FlexBox] = None
        self.prop(flexFlow="row" if is_horizontal else "column")
        self.editor.event_editor_save.on(self._on_editor_save)

        self._external_onsave = external_onsave

    @mark_did_mount
    async def _on_mount(self):
        reload_mgr = self.flow_app_comp_core.reload_mgr
        reload_mgr.in_memory_fs.add_file(self.path, self.code)
        mod = reload_mgr.in_memory_fs.load_in_memory_module(self.path)
        app_cls = mod.__dict__[self.app_cls_name]
        if hasattr(app_cls, "Config"):
            cfg_cls = getattr(app_cls, "Config")
            assert issubclass(cfg_cls, AppInMemory.Config)
            if cfg_cls.is_horizontal:
                await self.send_and_wait(self.update_event(flexFlow="row") + self.divider.update_event(orientation="horizontal"))
            else:
                await self.send_and_wait(self.update_event(flexFlow="column") + self.divider.update_event(orientation="vertical"))
            if cfg_cls.height is not mui.undefined:
                await self.send_and_wait(self.update_event(height=cfg_cls.height))
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
        wobj = layout.get_wrapped_obj()
        await self.show_box.update_childs({"layout": layout_flex})

    async def _on_editor_save(self, value: str):
        reload_mgr = self.flow_app_comp_core.reload_mgr
        reload_mgr.in_memory_fs.modify_file(self.path, value)
        if self._external_onsave is not None:
            await self._external_onsave(value)
        await appctx.get_editable_app()._reload_object_with_new_code(self.path, value)

class CodeBlock(mui.FlexBox):
    """app with editor (app must be anylayout)
    """
    # @dataclasses.dataclass
    class Config:
        is_horizontal: bool = True
        height: Union[mui.ValueType, mui.Undefined] = mui.undefined

    def __init__(self, code: str):
        self.editor = mui.MonacoEditor(code, "python", "").prop(minWidth=0, minHeight=0)
        self.code = code 
        super().__init__([
            mui.Button("run", self._on_run),
            self.editor.prop(flex=1),
        ])
        self._layout_for_reload: Optional[mui.FlexBox] = None
        self.prop(flexFlow="column")
        self.editor.event_editor_save.on(self._on_editor_save)

    async def _on_editor_save(self, value: str):
        self.code = value 

    async def _on_run(self):
        code_comp = compile(self.code, f"", "exec")
        exec(code_comp)

class MarkdownTutorial(mui.FlexBox):
    """ this component parse markdowns in a very simple way, don't use it in production, it's only for tutorials.
    """
    def __init__(self, md_content: str, path_uid: str):
        res_blocks, metadata = _parse_markdown_very_trivial(md_content)
        layout: mui.LayoutType = []
        if metadata.type == "Canvas":
            from tensorpc.flow import plus 
            complex_canvas = plus.ComplexCanvas(init_enable_grid=False).prop(width="100%", flex=1)
            blocks: mui.LayoutType = []
            for i, block in enumerate(res_blocks):
                if block.type == "markdown":
                    if block.content.strip() == "":
                        continue
                    blocks.append(mui.Markdown(block.content))
                elif block.type == "code":
                    blocks.append(CodeBlock(block.content.lstrip()).prop(height="200px", padding="10px"))
            book = mui.VirtualizedBox(blocks).prop(overflow="auto", flex=1)
            layout = [complex_canvas, book]
        else:
            blocks: mui.LayoutType = []
            for i, block in enumerate(res_blocks):
                if block.type == "markdown":
                    if block.content.strip() == "":
                        continue
                    blocks.append(mui.Markdown(block.content))
                elif block.type == "code":
                    blocks.append(AppInMemory(f"{path_uid}-{i}", block.content.lstrip()).prop(minHeight="400px", padding="10px"))
            book = mui.VirtualizedBox(blocks)
            layout = [book]
        super().__init__(layout)
        self.prop(flexFlow="column nowrap", padding="10px", overflow="hidden", minHeight=0, minWidth=0, height="100%", width="100%")

