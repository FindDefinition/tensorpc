from .constants import PACKAGE_ROOT
from tensorpc.core.client import (RemoteObject, RemoteException, RemoteManager,
                                            simple_chunk_call, simple_client,
                                            simple_remote_call)
from tensorpc.core import prim, marker

from tensorpc.core.asyncclient import (AsyncRemoteManager,
                                        AsyncRemoteObject,
                                        simple_chunk_call_async,
                                        simple_remote_call_async)

