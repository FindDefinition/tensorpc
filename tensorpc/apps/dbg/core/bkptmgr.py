from pathlib import Path
from types import FrameType
from tensorpc.core.astex.sourcecache import SourceChangeDiffCache
import dataclasses 
from tensorpc.dock.vscode.coretypes import VscodeBreakpoint
from tensorpc.apps.dbg.constants import FrameLocMeta

class BreakpointManager:
    def __init__(self):
        self._scd_cache = SourceChangeDiffCache()
        self._vscode_breakpoints_dict: dict[str, dict[tuple[Path, int],
                                                      VscodeBreakpoint]] = {}
        self._vscode_breakpoints_ts_dict: dict[Path, int] = {}

    def get_frame_loc_meta(self, frame: FrameType):
        may_changed_frame_lineno = self._scd_cache.query_mapped_linenos(
            frame.f_code.co_filename, frame.f_lineno)
        if may_changed_frame_lineno < 1:
            may_changed_frame_lineno = frame.f_lineno
        return FrameLocMeta(
            path=frame.f_code.co_filename,
            lineno=frame.f_lineno,
            mapped_lineno=may_changed_frame_lineno,
        )

    def check_vscode_bkpt_is_enabled(self, frame_loc_meta: FrameLocMeta):
        key = (Path(frame_loc_meta.path).resolve(), frame_loc_meta.mapped_lineno)
        for bkpts in self._vscode_breakpoints_dict.values():
            if key in bkpts:
                return bkpts[key].enabled
        return False 

    def set_vscode_breakpoints(self, bkpt_dict: dict[str, tuple[list[VscodeBreakpoint], int]]):
        for wuri, (bkpts, ts) in bkpt_dict.items():
            new_bkpts: list[VscodeBreakpoint] = []
            for x in bkpts:
                if x.enabled and x.lineText is not None and (
                        ".breakpoint" in x.lineText
                        or ".vscode_breakpoint" in x.lineText):
                    new_bkpts.append(x)
            if wuri not in self._vscode_breakpoints_dict:
                self._vscode_breakpoints_dict[wuri] = {}
            self._vscode_breakpoints_dict[wuri] = {
                (Path(x.path).resolve(), x.line + 1): x
                for x in new_bkpts
            }
            # save bkpt timestamp
            for x in new_bkpts:
                self._vscode_breakpoints_ts_dict[Path(x.path).resolve()] = ts

    def check_vscode_bkpt_is_enabled_after_set_vscode_bkpt(self, frame_loc_meta: FrameLocMeta):
        mtime = None 
        may_changed_frame_lineno = self._scd_cache.query_mapped_linenos(
            frame_loc_meta.path, frame_loc_meta.lineno)
        cache_entry = self._scd_cache.cache[frame_loc_meta.path]
        mtime = cache_entry.mtime
        if mtime is None:
            return False
        frame_loc_meta.mapped_lineno = may_changed_frame_lineno
        mtime_ns = int(mtime * 1e9)
        if Path(frame_loc_meta.path).resolve() in self._vscode_breakpoints_ts_dict:
            vscode_bkpt_ts = self._vscode_breakpoints_ts_dict[Path(frame_loc_meta.path).resolve()]
            if mtime_ns > vscode_bkpt_ts:
                # vscode bkpt state is outdated, skip current check.
                return False
        is_cur_bkpt_is_vscode = self.check_vscode_bkpt_is_enabled(
            frame_loc_meta)
        return is_cur_bkpt_is_vscode