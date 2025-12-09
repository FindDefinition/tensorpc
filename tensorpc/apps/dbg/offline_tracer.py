import contextlib
from pathlib import Path
import time
from typing import Optional, cast

import yaml
from tensorpc.apps.dbg.components.dbgpanel import merge_perfetto_trace_results
from tensorpc.apps.dbg.tracer import DebugTracerWrapper, VizTracerAndPytorchTracer
from tensorpc.apps.dbg.bkpt import _get_viztracer, _try_get_distributed_meta
from tensorpc.apps.dbg.constants import (TENSORPC_DBG_FRAME_INSPECTOR_KEY,
                                    TENSORPC_DBG_TRACE_VIEW_KEY,
                                    TENSORPC_ENV_DBG_ENABLE,
                                    BreakpointType, TraceLaunchType,
                                    TracerConfig, TraceResult, TracerType,
                                    RecordFilterConfig, DebugDistributedInfo,
                                    LOGGER)
import uuid 
import dataclasses
import gzip 
from tensorpc.dock.client import list_all_app_in_machine, list_all_running_apps_in_relay
from tensorpc.apps.dbg.components.dbgpanel import INIT_YAML_CONFIG
@dataclasses.dataclass
class WrapperTraceResult:
    meta: DebugDistributedInfo
    result: TraceResult
    timestamp: int

    def torch_dist_gather_results(self):
        import torch.distributed as dist
        assert dist.is_initialized(), "torch.distributed is not initialized"
        assert self.meta.backend == "pytorch", "only torch distributed is supported"
        obj_lists = [None for _ in range(self.meta.world_size)]
        dist.all_gather_object(obj_lists, (self.timestamp, self.result))
        return cast(list[tuple[int, TraceResult]], obj_lists)

    def torch_dist_gather_results_to_ui(self):
        obj_lists = self.torch_dist_gather_results()
        ui_data = merge_perfetto_trace_results(cast(list[Optional[tuple[int, TraceResult]]], obj_lists))
        return ui_data

    def convert_results_to_ui(self):
        ui_data = merge_perfetto_trace_results(cast(list[Optional[tuple[int, TraceResult]]], [(self.timestamp, self.result)]))
        return ui_data

    def _submit_to_ui(self, ui_data: tuple[bytes, list[int]], via_relay: bool = False):
        if via_relay:
            app_metas = list_all_running_apps_in_relay()
        else:
            app_metas = list_all_app_in_machine()
        for meta in app_metas:
            # only support tensorpc.apps.dbg.panel.DebugPanel
            if "DebugPanel" in meta.module_name:
                client = meta.create_client()
                with client:
                    client.app_chunked_remote_call("external_set_perfetto_data", ui_data[0], ui_data[1], "offline_tracer")

    def submit_to_ui(self, via_relay: bool = False):
        # TODO we currently assume ui is in rank 0.
        if self.meta.backend == "pytorch" and self.meta.world_size > 1:
            ui_data = self.torch_dist_gather_results_to_ui()
        else:
            ui_data = self.convert_results_to_ui()
        if self.meta.rank == 0:
            return self._submit_to_ui(ui_data, via_relay)

    def dump_to_file(self, path: str):
        suffix = Path(path).suffix.lower()
        assert suffix == ".gz", "profile data is gzipped."
        if self.meta.backend == "pytorch" and self.meta.world_size > 1:
            ui_data = self.torch_dist_gather_results_to_ui()
        else:
            ui_data = self.convert_results_to_ui()
        if self.meta.rank == 0:
            with open(path, "wb") as f:
                f.write(ui_data[0])


@contextlib.contextmanager
def offline_debug_tracer(cfg: TracerConfig):
    meta = _try_get_distributed_meta()
    proc_name = f"{meta.get_backend_short()}-{meta.rank}"
    if meta.backend is not None:
        tracer_name = f"{meta.backend}|{meta.rank}/{meta.world_size}"
    else:
        tracer_name = f"Process"

    wrapper = DebugTracerWrapper()
    cfg.trace_timestamp = time.time_ns()
    tracer, tracer_type = _get_viztracer(cfg, name=tracer_name)

    wrapper.set_tracer(cfg, tracer, tracer_type,
                            tracer_name, meta)
    wrapper.start()
    final_res = WrapperTraceResult(meta, TraceResult([]), cfg.trace_timestamp)
    try:
        yield final_res
    finally:
        wrapper.stop()
        raw_res = wrapper.save(proc_name)
        assert raw_res is not None
        trace_res = TraceResult(raw_res)
        
        uid = uuid.uuid4().hex
        trace_res_compressed = [
            dataclasses.replace(x, data=gzip.compress(x.data), is_tar=False, fname=f"{uid}.gz")
            for x in trace_res.single_results
        ]
        trace_res = dataclasses.replace(
                                trace_res,
                                single_results=trace_res_compressed)
        trace_res = trace_res.get_raw_event_removed()

        final_res.result = trace_res


@contextlib.contextmanager
def offline_pth_only_tracer(profile_memory: bool = False):
    
    filter_cfg = yaml.safe_load(INIT_YAML_CONFIG)
    filter_obj = RecordFilterConfig(**filter_cfg)
    cfg = TracerConfig(enable=True,
                       record_filter=filter_obj,
                       tracer=TracerType.PYTORCH,
                       profile_memory=profile_memory)
    if cfg.tracer == TracerType.VIZTRACER:
        cfg.trace_name = f"{cfg.trace_name}|viz"
    elif cfg.tracer == TracerType.PYTORCH:
        cfg.trace_name = f"{cfg.trace_name}|pth"
    elif cfg.tracer == TracerType.VIZTRACER_PYTORCH:
        cfg.trace_name = f"{cfg.trace_name}|v+p"

    with offline_debug_tracer(cfg) as tracer_wrapper:
        yield tracer_wrapper


@contextlib.contextmanager
def offline_viztracer_only_tracer(max_stack_depth: int = 10):
    
    filter_cfg = yaml.safe_load(INIT_YAML_CONFIG)
    filter_obj = RecordFilterConfig(**filter_cfg)
    cfg = TracerConfig(enable=True,
                       record_filter=filter_obj,
                       tracer=TracerType.VIZTRACER,
                       max_stack_depth=max_stack_depth)
    if cfg.tracer == TracerType.VIZTRACER:
        cfg.trace_name = f"{cfg.trace_name}|viz"
    elif cfg.tracer == TracerType.PYTORCH:
        cfg.trace_name = f"{cfg.trace_name}|pth"
    elif cfg.tracer == TracerType.VIZTRACER_PYTORCH:
        cfg.trace_name = f"{cfg.trace_name}|v+p"

    with offline_debug_tracer(cfg) as tracer_wrapper:
        yield tracer_wrapper


@contextlib.contextmanager
def offline_viztracer_pytorch_tracer(profile_memory: bool = False, max_stack_depth: int = 10):
    
    filter_cfg = yaml.safe_load(INIT_YAML_CONFIG)
    filter_obj = RecordFilterConfig(**filter_cfg)
    cfg = TracerConfig(enable=True,
                       record_filter=filter_obj,
                       tracer=TracerType.VIZTRACER_PYTORCH,
                       profile_memory=profile_memory,
                       max_stack_depth=max_stack_depth)
    if cfg.tracer == TracerType.VIZTRACER:
        cfg.trace_name = f"{cfg.trace_name}|viz"
    elif cfg.tracer == TracerType.PYTORCH:
        cfg.trace_name = f"{cfg.trace_name}|pth"
    elif cfg.tracer == TracerType.VIZTRACER_PYTORCH:
        cfg.trace_name = f"{cfg.trace_name}|v+p"

    with offline_debug_tracer(cfg) as tracer_wrapper:
        yield tracer_wrapper

