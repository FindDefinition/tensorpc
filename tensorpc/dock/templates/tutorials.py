from typing import Any, Dict, List, Union
from tensorpc.dock.components import plus
from tensorpc import PACKAGE_ROOT
from tensorpc.dock import mui, three, plus, mark_create_layout, mark_did_mount, appctx

class App:
    @mark_create_layout
    def my_layout(self):
        tutorials_path = PACKAGE_ROOT / "examples" / "tutorials"
        tutorials: Dict[str, Any] = {}
        paths = list(tutorials_path.rglob("*.md"))
        paths.sort(key=lambda p: list(map(int,
                                          p.stem.split("-")[0].split("."))))
        for p in paths:
            md_relative_path = p.relative_to(tutorials_path)
            parts = md_relative_path.parts
            tutorials_cur = tutorials
            for part in parts[:-1]:
                if part not in tutorials:
                    tutorials_cur[part] = {}
                tutorials_cur = tutorials_cur[part]
            md_content = p.read_text()
            tutorials_cur[md_relative_path.stem] = plus.MarkdownTutorial(
                md_content, str(md_relative_path)).prop(width="100%",
                                                        height="100%",
                                                        overflow="auto")
        self.tutorials = tutorials
        self.panel = plus.InspectPanel({}, use_fast_tree=True)
        return self.panel.prop(width="100%", height="100%", overflow="hidden")

    @mark_did_mount
    async def _on_init(self):
        await self.panel.inspector.add_object_to_tree(self.tutorials,
                                              key="tutorials",
                                              expand_level=2)
