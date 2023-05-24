            
import contextvars
from typing import Optional
from tensorpc import simple_remote_call, RemoteManager
import contextlib
from tensorpc.autossh.serv_names import serv_names 

from tensorpc.autossh.scheduler.core import TaskOutput, TaskStatus
from .constants import TmuxSchedulerEnvVariables

TASK_CONTEXT_VAR: contextvars.ContextVar[
    Optional[RemoteManager]] = contextvars.ContextVar("tmux_sched_task_context",
                                                   default=None)


def get_task_context() -> Optional[RemoteManager]:
    return TASK_CONTEXT_VAR.get()


@contextlib.contextmanager
def enter_task_conetxt(robj: RemoteManager):
    token = TASK_CONTEXT_VAR.set(robj)
    try:
        yield robj
    finally:
        TASK_CONTEXT_VAR.reset(token)


class TaskClient:
    """used inside task to communicate with scheduler.
    if not in task, all operations are no-op.

    if user launch a task without spawn new process (use func id or not distributed)
    this client will reuse the scheduler client to improve speed.
    """
    def __init__(self) -> None:
        env = TmuxSchedulerEnvVariables()
        self.port = env.port 
        self.uid = env.uid 

    @contextlib.contextmanager
    def _scheduler_robj(self):
        ctx = get_task_context()
        if ctx is not None:
            yield ctx 
        else:
            with RemoteManager(f"localhost:{self.port}") as robj:
                yield robj

    def update_task(self, progress: float, output: Optional[TaskOutput] = None):
        if self.port is not None:
            with self._scheduler_robj() as robj:
                robj.remote_call(serv_names.SCHED_TASK_UPDATE_TASK, self.uid, progress, output)

    def check_need_cancel(self):
        if self.port is None:
            return False 
        with self._scheduler_robj() as robj:
            status = robj.remote_call(serv_names.SCHED_TASK_CHECK_STATUS, self.uid)
        if status == TaskStatus.NeedToCancel:
            return True 
        return False 

    def check_need_cancel_torch_dist(self):
        if self.port is None:
            return False 
        import torch.distributed as dist
        if not dist.is_initialized():
            return self.check_need_cancel()
        with self._scheduler_robj() as robj:
            status = robj.remote_call(serv_names.SCHED_TASK_CHECK_STATUS, self.uid)
        world_size = dist.get_world_size()
        res_list = [status] * world_size
        dist.all_gather_object(res_list, status)
        if any([x == TaskStatus.NeedToCancel for x in res_list]):
            return True 
        return False  