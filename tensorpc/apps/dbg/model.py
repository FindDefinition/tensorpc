from queue import Queue
from types import FrameType
from typing import Annotated, Any, Awaitable, Callable, Optional
from tensorpc.core import dataclass_dispatch as dataclasses 
from tensorpc.core import inspecttools
from tensorpc.apps.dbg.constants import BreakpointType, FrameLocMeta, TracerType, TraceResult, TracerConfig, TraceMetrics, RecordMode
from tensorpc.core.datamodel.draft import DraftFieldMeta
from tensorpc.utils.uniquename import UniqueNamePool


@dataclasses.dataclass
class FrameInfo:
    name: str
    qualname: str
    path: str 
    lineno: int


@dataclasses.dataclass
class TracerRuntimeState:
    cfg: TracerConfig
    metric: TraceMetrics
    frame_loc: FrameLocMeta
    force_stop: bool = False

    def increment_trace_state(self, new_frame_loc: FrameLocMeta) -> tuple[int, bool]:
        is_record_stop = False
        cfg = self.cfg
        metric = self.metric
        is_same_bkpt = False
        is_inf_record = cfg.mode == RecordMode.INFINITE
        if cfg.mode == RecordMode.SAME_BREAKPOINT:
            frame_uid = (
                new_frame_loc.path, new_frame_loc.lineno)
            cur_frame_uid = (
                self.frame_loc.path, self.frame_loc.lineno)
            is_same_bkpt = frame_uid == cur_frame_uid
        breakpoint_cnt = metric.breakpoint_count
        if not is_inf_record:
            breakpoint_cnt -= 1
        if (breakpoint_cnt == 0 and not is_inf_record
            ) or is_same_bkpt or self.force_stop:
            is_record_stop = True
        return breakpoint_cnt, is_record_stop


@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class TracerState:
    runtime: Optional[TracerRuntimeState]
    results: Annotated[dict[str, tuple[int, TraceResult]], DraftFieldMeta(is_external=True)] = dataclasses.field(default_factory=dict)

    @staticmethod
    def create_new_runtime(cfg: TracerConfig, frame_loc: FrameLocMeta) -> TracerRuntimeState:
        runtime = TracerRuntimeState(cfg, TraceMetrics(cfg.breakpoint_count), frame_loc)
        return runtime

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class VscodeSelectedObject:
    info: FrameInfo
    expr: str
    obj: Annotated[Any, DraftFieldMeta(is_external=True)] = None


@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class Breakpoint:
    type: BreakpointType
    info: FrameInfo
    frame_loc: FrameLocMeta
    frame_select_items: list[dict[str, Any]]
    selected_frame_item: Optional[dict[str, Any]] = None
    frame: Annotated[Optional[FrameType], DraftFieldMeta(is_external=True)] = None
    release_fn: Annotated[Optional[Callable[[], Awaitable[None]]], DraftFieldMeta(is_external=True)] = None
    launch_trace_fn: Annotated[Optional[Callable[[TracerConfig], Awaitable[None]]], DraftFieldMeta(is_external=True)] = None
    is_external: bool = False
    queue: Annotated[Optional[Queue], DraftFieldMeta(is_external=True)] = None

    @staticmethod 
    def generate_frame_select_items(frame: FrameType):
        # here we don't save all frames, only save the offset.
        cur_frame: Optional[FrameType] = frame
        frame_select_opts: list[dict[str, Any]] = []
        offset = 0
        uid_pool = UniqueNamePool()
        while cur_frame is not None:
            info = Breakpoint.get_frame_info_from_frame(cur_frame)
            frame_select_opts.append({"label": uid_pool(info.qualname), "offset": offset, **dataclasses.asdict(info)})
            offset += 1
            cur_frame = cur_frame.f_back
        return frame_select_opts

    def get_launch_trace_fn(self) -> Callable[[TracerConfig], Awaitable[None]]:
        if self.launch_trace_fn is None:
            raise ValueError("launch_trace_fn is None")
        return self.launch_trace_fn

    def get_release_fn(self) -> Callable[[], Awaitable[None]]:
        if self.release_fn is None:
            raise ValueError("release_fn is None")
        return self.release_fn

    @staticmethod
    def get_frame_info_from_frame(frame: FrameType) -> FrameInfo:
        qname = inspecttools.get_co_qualname_from_frame(frame)
        return FrameInfo(frame.f_code.co_name, qname,
                        frame.f_code.co_filename,
                        frame.f_lineno)


@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class PyDbgModel:
    tracer_state: TracerState
    # frontend props 
    # when vscode select some expr, backend may eval that expr and set this field.
    vscode_selected_obj: Optional[VscodeSelectedObject] = None
    bkpt: Optional[Breakpoint] = None

