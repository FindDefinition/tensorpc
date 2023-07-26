import dataclasses
import enum
import inspect
import urllib.request
from typing import Any, Callable, Coroutine, Dict, Hashable, Iterable, List, Literal, Optional, Set, Tuple, Type, Union

import numpy as np

from tensorpc.flow import marker
from tensorpc.flow.flowapp.appcore import find_component_by_uid_with_type_check
from tensorpc.flow.flowapp.components import mui, three
from tensorpc.flow.flowapp.components.plus.config import ConfigPanel
from tensorpc.flow.flowapp.core import FrontendEventType
from tensorpc.flow.flowapp.coretypes import TreeDragTarget
from tensorpc.flow.flowapp import colors
from tensorpc.flow.jsonlike import TreeItem
from tensorpc.utils.registry import HashableSeqRegistryKeyOnly
from tensorpc.flow.flowapp.components.core import get_tensor_container

# UNKNOWN_VIS_REGISTRY: HashableSeqRegistryKeyOnly[Callable[[Any, str, "SimpleCanvas"], Coroutine[None, None, bool]]] = HashableSeqRegistryKeyOnly()


class ComplexCanvas(mui.FlexBox):
    """
    a blender-like canvas
    Design:
        * put controls to left as toggle buttons
        * put canvas object tree view to right
        * support helpers such as light, camera, etc.
        * support switch to camera view
    """
    pass 
