import asyncio
from pathlib import Path
from typing import Optional

import aiohttp
from tensorpc.flow import mui, three, plus, mark_create_layout, appctx
import sys
from tensorpc import PACKAGE_ROOT
import numpy as np

from tensorpc.flow.marker import mark_did_mount
from tensorpc import prim
class DevApp:

    @mark_create_layout
    def my_layout(self):

        return plus.ComplexCanvas()

