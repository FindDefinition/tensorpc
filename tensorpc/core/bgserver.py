import asyncio
from functools import partial
import os
import queue
import traceback
from typing import Optional, Union
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

class BackgroundServer:
    """A background server that runs in a separate thread.
    use single-thread async server.

    ### Fork Behavior
    When you fork a process after background server started, the background server will be stopped before fork.

    """
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self.port = -1
        # if you use forked process, this won't be called in python < 3.13
        atexit.register(self.stop)

        self._service_core: Optional[ServiceCore] = None

        self.server_id: Optional[str] = None
        self.server_uuid: Optional[str] = None

        self._is_fork_handler_registered = False

        self._prev_proc_title: Optional[str] = None

    @property
    def service_core(self):
        assert self._service_core is not None, "you must start the server first"
        return self._service_core

    @property
    def is_started(self):
        return self._thread is not None and self._thread.is_alive()

    def _try_set_proc_title(self, uid: str, id: str, status: int = 0):
        assert self.port > 0
        parts = [
            TENSORPC_BG_PROCESS_NAME_PREFIX, id, str(self.port), str(status), uid,
        ]
        title = TENSORPC_DBG_SPLIT.join(parts)
        try:
            import setproctitle  # type: ignore
            if self._prev_proc_title is None:
                self._prev_proc_title = setproctitle.getproctitle()
            setproctitle.setproctitle(title)
        except ImportError:
            pass

    def start_async(self,
                    service_def: Optional[ServiceDef] = None,
                    port: int = -1,
                    id: Optional[str] = None,
                    wait_for_start: bool = True):
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
            # if port < 0:
            service_def.services.append(
                Service("tensorpc.services.collection::ProcessObserver",
                        {"q": port_res_queue}))
            url = '[::]:{}'.format(port)
            smeta = ServerMeta(port=port, http_port=-1)
            service_core = ProtobufServiceCore(url, service_def, False, smeta)
            self._service_core = service_core
            ev = threading.Event()
            thread = threading.Thread(target=serve_service_core_async,
                                            kwargs={
                                                "service_core": service_core,
                                                "create_loop": True,
                                                "start_thread_ev": ev
                                            })
            self._thread = thread
            self._thread.daemon = True
            self._thread.start()
            if InMacOS or InLinux:
                if not self._is_fork_handler_registered:
                    os.register_at_fork(before=partial(self.stop, is_fork=True))
                    self._is_fork_handler_registered = True
            uid = uuid.uuid4().hex # [:8]
            self.server_uuid = uid
            # if port < 0:
            port = port_res_queue.get(timeout=20)
            self.port = port
            if id is not None:
                self.server_id = id
                self._try_set_proc_title(uid, id)
            if wait_for_start:
                ev.wait()
        except:
            traceback.print_exc()
            raise
        return port

    def set_running_proc_status(self, status: Union[int, str]):
        assert self.port > 0 and self.server_id is not None and self.server_uuid is not None
        uid = self.server_uuid
        title = f"{TENSORPC_BG_PROCESS_NAME_PREFIX}-{self.server_id}-{self.port}-{status}-{uid}"
        try:
            import setproctitle  # type: ignore
            if self._prev_proc_title is None:
                self._prev_proc_title = setproctitle.getproctitle()
            setproctitle.setproctitle(title)
        except ImportError:
            pass

    def stop(self, is_fork: bool = False):
        if self.is_started:
            assert self._thread is not None
            assert self._service_core is not None 
            loop = self._service_core._loop
            if InLinux:
                if self._prev_proc_title is not None:
                    try:
                        import setproctitle  # type: ignore
                        setproctitle.setproctitle(self._prev_proc_title)
                    except ImportError:
                        pass
            if loop is not None:
                loop.call_soon_threadsafe(self._service_core.async_shutdown_event.set)
            # robj = RemoteManager(f"localhost:{self.port}")
            # robj.shutdown()
            _thread = self._thread
            self._thread = None
            self._service_core = None
            self.server_id = None
            self.port = -1
            self.server_uuid = None
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
        assert self._service_core is not None, "you must start the server first"
        loop = self._service_core._loop
        assert loop is not None, "loop is not set"
        future = asyncio.run_coroutine_threadsafe(
            self._service_core.execute_async_service_locally(
                service_key, args, kwargs), loop)
        return future.result()


BACKGROUND_SERVER = BackgroundServer()
