import dataclasses
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
from tensorpc.core.defs import File

from tensorpc.flow.flowapp.components import mui, three
from tensorpc.flow.flowapp.core import FrontendEventType
from tensorpc.flow.flowapp import appctx, colors
import io
import json
import numpy as np
from numpy.lib.npyio import NpzFile 
import pickle 

class SimpleFileReader(mui.FlexBox):
    """support json/pickle (.json/.pkl/.pickle) and numpy (.npy/.npz) files
    """
    def __init__(self):
        self.text = mui.Typography("Drop file here")
        self.text.prop(color="secondary")
        self.text.prop(align="center")
        self.text.prop(variant="body2")

        super().__init__([self.text])
        self.all_allowed_exts = [".json", ".pkl", ".pickle", ".npy", ".npz"]
        self.prop(droppable=True, allow_file=True, flex_direction="column",
                    border="4px solid white",
                    sx_over_drop={"border": "4px solid green"}, width="100%", height="100%", overflow="hidden",
                    justify_content="center", align_items="center")

        self.register_event_handler(FrontendEventType.Drop.value, self.on_drop_file)

    async def on_drop_file(self, file: File):
        suffix = file.name[file.name.rfind("."):]
        assert suffix in self.all_allowed_exts, f"unsupported file type: {suffix}"
        if suffix == ".json":
            data = json.loads(file.content)
        elif suffix in [".pkl", ".pickle"]:
            data = pickle.loads(file.content)
        elif suffix in [".npy", ".npz"]:
            byteio = io.BytesIO(file.content)
            data = np.load(byteio, allow_pickle=True)
            if isinstance(data, NpzFile):
                data = dict(data)
        else:
            raise NotImplementedError
        await self.text.write(f"Loaded {file.name}")
        await appctx.obj_inspector_set_object(data, "droppedFile")
