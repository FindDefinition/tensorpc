from . import handlers as _handlers
from .core import (ALL_OBJECT_LAYOUT_HANDLERS, ALL_OBJECT_PREVIEW_HANDLERS,
                   ObjectLayoutHandler, ObjectPreviewHandler,
                   register_user_obj_tree_type)
from .inspector import ObjectInspector
from .inspectpanel import InspectPanel
from .layout import AnyFlexLayout
from .tree import ObjectTree, TreeDragTarget, BasicObjectTree
from .controllers import CallbackSlider, ThreadLocker, MarkdownViewer


def register_obj_preview_handler(cls):
    return ALL_OBJECT_PREVIEW_HANDLERS.register(cls)


def register_obj_layout_handler(cls):
    return ALL_OBJECT_LAYOUT_HANDLERS.register(cls)
