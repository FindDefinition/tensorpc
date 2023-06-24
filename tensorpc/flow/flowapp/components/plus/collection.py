import dataclasses
from functools import partial
from pathlib import Path
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
import asyncio
import sys
from runpy import run_path
import shlex


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
        self.prop(droppable=True,
                  allowFile=True,
                  flexDirection="column",
                  border="4px solid white",
                  sxOverDrop={"border": "4px solid green"},
                  width="100%",
                  height="100%",
                  overflow="hidden",
                  justifyContent="center",
                  alignItems="center")

        self.register_event_handler(FrontendEventType.Drop.value,
                                    self.on_drop_file)

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
        await appctx.inspector.set_object(data, "droppedFile")



class ScriptExecutor(mui.FlexBox):
    def __init__(self):
        self.path = mui.TextField(label="Path").prop(muiMargin="dense")
        self.args = mui.TextField(label="Args").prop(muiMargin="dense")

        super().__init__(
            [self.path, self.args,
             mui.HBox([
                 mui.Button("Run", self._run),
                 mui.Button("Cancel", self._cancel),

             ])])
        self.prop(flexDirection="column")
        self._external_argv_task: Optional[asyncio.Future] = None

    async def _run(self):
        if self._external_argv_task is not None:
            raise RuntimeError("already running")
        self._external_argv_task = asyncio.create_task(
            appctx.run_in_executor_with_exception_inspect(
                partial(self._run_app_script,
                        path=self.path.str(),
                        argv=shlex.split(" ".join([self.path.str(), self.args.str()]))), ))

    async def _cancel(self):
        if self._external_argv_task is None:
            raise RuntimeError("not running")
        self._external_argv_task.cancel()

    def _run_app_script(self, path: str, argv: List[str]):
        assert Path(path).exists()
        argv_bkp = sys.argv
        sys.argv = argv
        try:
            run_path(path, run_name="__main__")
        finally:
            sys.argv = argv_bkp
            self._external_argv_task = None
