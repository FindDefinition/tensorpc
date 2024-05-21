import asyncio
import queue
from typing import Optional
from tensorpc.core.asyncserver import serve as serve_async

from tensorpc.core.client import RemoteManager
from tensorpc.core.defs import ServiceDef, Service
from tensorpc.core.server import serve
import threading
import atexit
from tensorpc.core import BUILTIN_SERVICES


class BackgroundServer:

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self.port = -1
        # atexit.register(self.stop)

    @property
    def is_started(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self,
              service_def: Optional[ServiceDef] = None,
              port: int = -1,
              max_workers: int = 2):
        assert not self.is_started
        if service_def is None:
            service_def = ServiceDef([])
            service_def.services.extend(BUILTIN_SERVICES)
        port_res_queue = queue.Queue()
        if port < 0:
            service_def.services.append(
                Service("tensorpc.services.collection::ProcessObserver",
                        {"q": port_res_queue}))
        self._thread = threading.Thread(target=serve,
                                        kwargs={
                                            "service_def": service_def,
                                            "port": port,
                                            "max_threads": max_workers
                                        })
        self._thread.daemon = True
        self._thread.start()
        if port < 0:
            port = port_res_queue.get(timeout=20)
        self.port = port
        return port

    def start_async(self,
                    service_def: Optional[ServiceDef] = None,
                    port: int = -1):
        assert not self.is_started
        if service_def is None:
            service_def = ServiceDef([])
            service_def.services.extend(BUILTIN_SERVICES)
        port_res_queue = queue.Queue()
        if port < 0:
            service_def.services.append(
                Service("tensorpc.services.collection::ProcessObserver",
                        {"q": port_res_queue}))

        self._thread = threading.Thread(target=serve_async,
                                        kwargs={
                                            "service_def": service_def,
                                            "port": port,
                                            "create_loop": True
                                        })
        self._thread.daemon = True
        self._thread.start()
        if port < 0:
            port = port_res_queue.get(timeout=20)
        self.port = port
        return port

    def stop(self):
        if self.is_started:
            assert self._thread is not None
            robj = RemoteManager(f"localhost:{self.port}")
            robj.shutdown()
            self._thread.join()
            self._thread = None


BACKGROUND_SERVER = BackgroundServer()
