import asyncio
import queue
from typing import Optional, Union
import uuid
from tensorpc.constants import TENSORPC_BG_PROCESS_NAME_PREFIX
from tensorpc.core.asyncserver import serve_service_core as serve_service_core_async

from tensorpc.core.client import RemoteManager
from tensorpc.core.defs import ServiceDef, Service
from tensorpc.core.server import serve_service_core
import threading
import atexit
from tensorpc.core import BUILTIN_SERVICES
from tensorpc.core.server_core import ProtobufServiceCore, ServerMeta, ServiceCore
import sys 

class BackgroundServer:
    """A background server that runs in a separate thread.
    use single-thread async server."""
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self.port = -1
        # if you use forked process, this won't be called in python < 3.13
        atexit.register(self.stop)

        self._service_core: Optional[ServiceCore] = None

        self.server_id: Optional[str] = None
        self.server_uuid: Optional[str] = None

    @property
    def service_core(self):
        assert self._service_core is not None, "you must start the server first"
        return self._service_core

    @property
    def is_started(self):
        return self._thread is not None and self._thread.is_alive()

    def _try_set_proc_title(self, uid: str, id: str, status: int = 0):
        assert self.port > 0
        title = f"{TENSORPC_BG_PROCESS_NAME_PREFIX}-{id}-{self.port}-{status}-{uid}"
        try:
            import setproctitle  # type: ignore
            setproctitle.setproctitle(title)
        except ImportError:
            pass

    def start_async(self,
                    service_def: Optional[ServiceDef] = None,
                    port: int = -1,
                    id: Optional[str] = None,
                    wait_for_start: bool = True):
        assert not self.is_started
        if service_def is None:
            service_def = ServiceDef([])
            service_def.services.extend(BUILTIN_SERVICES)
        port_res_queue = queue.Queue()
        if port < 0:
            service_def.services.append(
                Service("tensorpc.services.collection::ProcessObserver",
                        {"q": port_res_queue}))
        url = '[::]:{}'.format(port)
        smeta = ServerMeta(port=port, http_port=-1)
        service_core = ProtobufServiceCore(url, service_def, False, smeta)
        self._service_core = service_core
        ev = threading.Event()
        self._thread = threading.Thread(target=serve_service_core_async,
                                        kwargs={
                                            "service_core": service_core,
                                            "create_loop": True,
                                            "start_thread_ev": ev
                                        })
        self._thread.daemon = True
        self._thread.start()
        uid = uuid.uuid4().hex # [:8]
        self.server_uuid = uid
        if port < 0:
            port = port_res_queue.get(timeout=20)
        self.port = port
        if id is not None:
            self.server_id = id
            self._try_set_proc_title(uid, id)
        if wait_for_start:
            ev.wait()
        return port

    def set_running_proc_status(self, status: Union[int, str]):
        assert self.port > 0 and self.server_id is not None and self.server_uuid is not None
        uid = self.server_uuid
        title = f"{TENSORPC_BG_PROCESS_NAME_PREFIX}-{self.server_id}-{self.port}-{status}-{uid}"
        try:
            import setproctitle  # type: ignore
            setproctitle.setproctitle(title)
        except ImportError:
            pass

    def stop(self):
        if self.is_started:
            assert self._thread is not None
            assert self._service_core is not None 
            loop = self._service_core._loop
            if loop is not None:
                loop.call_soon_threadsafe(self._service_core.async_shutdown_event.set)
            # robj = RemoteManager(f"localhost:{self.port}")
            # robj.shutdown()
            self._thread.join()
            self._thread = None
            self._service_core = None
            self.server_id = None
            self.port = -1
            self.server_uuid = None

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
