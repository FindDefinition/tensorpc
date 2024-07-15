from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from tensorpc.flow.coretypes import VscodeTraceItem
from ..core.appcore import get_app, get_app_context
from tensorpc.core.tracers.core import TraceEventType, FrameEventCall
from tensorpc.core.tracers.calltracer import CallTracerContext

def parse_frame_result_to_trace_item(frame_results: List[FrameEventCall]):
    fr_stack: List[Tuple[FrameEventCall, VscodeTraceItem]] = []
    res: List[VscodeTraceItem] = []
    # print([(x.qualname, x.type, x.depth) for x in frame_results])
    for fr in frame_results:
        if fr.type == TraceEventType.Call:
            item = VscodeTraceItem(fr.qualname, [], fr.filename, fr.lineno, fr.timestamp / 1e9)
            fr_stack.append((fr, item))

        elif fr.type == TraceEventType.Return:
            poped = fr_stack.pop()
            poped[1].duration = fr.timestamp / 1e9 - poped[1].duration
            if len(fr_stack) == 0:
                res.append(poped[1])
            else:
                fr_stack[-1][1].childs.append(poped[1])
    return res


def run_trace(func: Callable,
            args: Tuple,
            kwargs: Dict[str, Any],
            traced_folders: Optional[Set[Union[str, Path]]] = None,
            max_depth: int = 10000):
    with CallTracerContext(trace_call_only=False, max_depth=max_depth, traced_folders=traced_folders) as ctx:
        func(*args, **kwargs)
    return parse_frame_result_to_trace_item(ctx.result_call_stack)


async def run_trace_and_save_to_app(key: str, func: Callable,
            args: Tuple,
            kwargs: Dict[str, Any],
            traced_folders: Optional[Set[Union[str, Path]]] = None,
            max_depth: int = 10000):
    res = run_trace(func, args, kwargs, traced_folders, max_depth)
    app = get_app()
    storage = await app.get_vscode_storage_lazy()
    await storage.add_or_update_trace_tree_with_update(key, res)