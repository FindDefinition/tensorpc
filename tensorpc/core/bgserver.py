import asyncio
import dataclasses
from functools import partial
import os
import queue
import traceback
from typing import Any, Optional, Union, TypeVar, Type
import uuid

from tensorpc.constants import TENSORPC_BG_PROCESS_NAME_PREFIX, TENSORPC_MAIN_PID
from tensorpc.core.asyncserver import serve_service_core as serve_service_core_async

from tensorpc.core.client import RemoteManager
from tensorpc.core.defs import ServiceDef, Service
from tensorpc.core.server import serve_service_core
import threading
import atexit
from tensorpc.core import BUILTIN_SERVICES
from tensorpc.core.server_core import ProtobufServiceCore, ServerMeta, ServiceCore
from tensorpc.compat import InMacOS, InLinux
from tensorpc.dbg.constants import TENSORPC_DBG_SPLIT
from tensorpc.utils.rich_logging import get_logger 

LOGGER = get_logger("tensorpc.core")
T = TypeVar("T")

@dataclasses.dataclass
class _BackgroundServerState:
    thread: threading.Thread
    service_core: ServiceCore
    port: int
    server_id: str
    server_uuid: str
    userdata: Any


class BackgroundServer:
    """A background server that runs in a separate thread.
    use single-thread async server.

    ### Fork Behavior
    When you fork a process after background server started, the background server will be stopped before fork.

    """
    def __init__(self):
        self._state: Optional[_BackgroundServerState] = None
        # if you use forked process, this won't be called in python < 3.13
        atexit.register(self.stop)
        self._is_fork_handler_registered = False

        self._prev_proc_title: Optional[str] = None
        self._cur_proc_title: Optional[str] = None

    @property 
    def port(self):
        assert self._state is not None, "you must start the server first"
        return self._state.port

    @property
    def service_core(self):
        assert self._state is not None, "you must start the server first"
        return self._state.service_core

    @property
    def is_started(self):
        return self._state is not None and self._state.thread.is_alive()

    @property
    def cur_proc_title(self):
        return self._cur_proc_title

    def _try_set_proc_title(self, uid: str, id: str, status: int = 0):
        assert self._state is not None 
        parts = [
            TENSORPC_BG_PROCESS_NAME_PREFIX, id, str(self._state.port), str(status), uid,
        ]
        title = TENSORPC_DBG_SPLIT.join(parts)
        try:
            import setproctitle  # type: ignore
            if self._prev_proc_title is None:
                self._prev_proc_title = setproctitle.getproctitle()
            self._cur_proc_title = title
            setproctitle.setproctitle(title)
        except ImportError:
            pass

    def start_async(self,
                    service_def: Optional[ServiceDef] = None,
                    port: int = -1,
                    id: Optional[str] = None,
                    wait_for_start: bool = True,
                    userdata: Optional[Any] = None):
        if id is not None:
            if TENSORPC_MAIN_PID != os.getpid():
                # forked process
                if InMacOS:
                    raise NotImplementedError("forked process with setproctitle is not supported in MacOS")
        try:
            assert not self.is_started
            if service_def is None:
                service_def = ServiceDef([])
                service_def.services.extend(BUILTIN_SERVICES)
            port_res_queue = queue.Queue()
            service_def.services.append(
                Service("tensorpc.services.collection::ProcessObserver",
                        {"q": port_res_queue}))
            url = '[::]:{}'.format(port)
            smeta = ServerMeta(port=port, http_port=-1)
            service_core = ProtobufServiceCore(url, service_def, False, smeta)
            ev = threading.Event()
            thread = threading.Thread(target=serve_service_core_async,
                                            kwargs={
                                                "service_core": service_core,
                                                "create_loop": True,
                                                "start_thread_ev": ev
                                            })
            thread.daemon = True
            thread.start()
            if InMacOS or InLinux:
                if not self._is_fork_handler_registered:
                    os.register_at_fork(before=partial(self.stop, is_fork=True))
                    self._is_fork_handler_registered = True
            uid = uuid.uuid4().hex # [:8]
            port = port_res_queue.get(timeout=20)
            state = _BackgroundServerState(
                thread=thread,
                service_core=service_core,
                port=port,
                server_id="bgserver",
                server_uuid=uid,
                userdata=userdata
            )
            self._state = state
            if id is not None:
                state.server_id = id
                self._try_set_proc_title(uid, id)
            if wait_for_start:
                ev.wait()
        except:
            traceback.print_exc()
            raise
        return port

    def get_userdata_typed(self, userdata_type: Type[T]) -> T:
        assert self._state is not None
        res = self._state.userdata
        assert isinstance(res, userdata_type)
        return res

    def set_running_proc_status(self, status: Union[int, str]):
        assert self._state is not None
        assert self._state.port > 0 and self._state.server_id is not None and self._state.server_uuid is not None
        uid = self._state.server_uuid
        title = f"{TENSORPC_BG_PROCESS_NAME_PREFIX}-{self._state.server_id}-{self._state.port}-{status}-{uid}"
        try:
            import setproctitle  # type: ignore
            if self._prev_proc_title is None:
                self._prev_proc_title = setproctitle.getproctitle()
            self._cur_proc_title = title
            setproctitle.setproctitle(title)
        except ImportError:
            pass

    def stop(self, is_fork: bool = False):
        if self.is_started:
            assert self._state is not None
            loop = self._state.service_core._loop
            if InLinux:
                if self._prev_proc_title is not None:
                    try:
                        import setproctitle  # type: ignore
                        setproctitle.setproctitle(self._prev_proc_title)
                    except ImportError:
                        pass
            if loop is not None:
                loop.call_soon_threadsafe(self._state.service_core.async_shutdown_event.set)
            _thread = self._state.thread
            self._state = None
            self._cur_proc_title = None 
            _thread.join()
            if is_fork:
                LOGGER.warning("shutdown background server because of fork")
            else:
                LOGGER.warning("shutdown background server")

    def execute_service(
        self,
        service_key: str,
        *args,
        **kwargs,
    ):
        assert self._state is not None, "you must start the server first"
        loop = self._state.service_core._loop
        assert loop is not None, "loop is not set"
        future = asyncio.run_coroutine_threadsafe(
            self._state.service_core.execute_async_service_locally(
                service_key, args, kwargs), loop)
        return future.result()


BACKGROUND_SERVER = BackgroundServer()
