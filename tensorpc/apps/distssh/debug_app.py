import asyncio
from pathlib import Path
from typing import Optional

import aiohttp
from tensorpc.apps.distssh.constants import TENSORPC_DISTSSH_UI_KEY
from tensorpc.constants import TENSORPC_APPS_DISTSSH_DEFAULT_PORT, TENSORPC_DEV_SECRET_PATH
from tensorpc.core.tracers.codefragtracer import CodeFragTracerResult
from tensorpc.dock import mui, three, plus, mark_create_layout, appctx
import sys
from tensorpc import PACKAGE_ROOT
import numpy as np
import yaml 

class FTSSHDevApp:

    @mark_create_layout
    def my_layout(self):
        # with open(TENSORPC_DEV_SECRET_PATH, "r") as f:
        #     secret = yaml.safe_load(f)["distssh_debug"]
        remote_box = mui.RemoteBoxGrpc("localhost", TENSORPC_APPS_DISTSSH_DEFAULT_PORT, TENSORPC_DISTSSH_UI_KEY)
        return mui.VBox([
            remote_box.prop(flex=1)
        ]).prop(width="100%", height="100%", overflow="hidden")

