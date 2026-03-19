from collections.abc import Generator
from typing import Any, AsyncGenerator, Optional, Union

from .base import AsyncRPCCommBase

from tensorpc.core.asyncclient import AsyncRemoteManager
from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.core.distributed.logger import LOGGER
import grpc 
import time 

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class GRPCNodeState:
    robj: AsyncRemoteManager
    last_seen_ts: int
    down_count_remain: int
    max_fail_before_mark_down: int

    @property 
    def is_down(self):
        if self.max_fail_before_mark_down < 0:
            return False
        return self.down_count_remain <= 0

@dataclasses.dataclass
class AsyncGRPCCommConfig:
    common_timeout: Optional[int] = None
    # when a node is marked as down, we will raise TimeoutError immediately without trying to send RPC to it.
    max_fail_before_mark_down: int = -1
    num_retry: int = 0

class GrpcCommDownError(TimeoutError):
    pass

class AsyncGRPCComm(AsyncRPCCommBase):
    def __init__(self, cfg: AsyncGRPCCommConfig):
        self._cfg = cfg
        self._remote_states: dict[str, GRPCNodeState] = {}

    async def _get_or_create_remote(self, target_addr: str) -> GRPCNodeState:
        rm_state = self._remote_states.get(target_addr)
        if rm_state is not None:
            return rm_state
        
        rm = AsyncRemoteManager(target_addr)
        await rm.wait_for_channel_ready()
        rm_state = GRPCNodeState(
            robj=rm,
            last_seen_ts=int(time.time()),
            down_count_remain=self._cfg.max_fail_before_mark_down,
            max_fail_before_mark_down=self._cfg.max_fail_before_mark_down,
        )
        self._remote_states[target_addr] = rm_state
        return rm_state

    async def remote_call(self,
                          target_addr: str,
                          key: str,
                          *args: Any,
                          rpc_timeout: Optional[int] = None,
                          rpc_wait_for_ready: bool = False,
                          **kwargs: Any) -> Any:
        """Make an asynchronous remote call to the target."""
        rm_state = await self._get_or_create_remote(target_addr)
        timeout = rpc_timeout or self._cfg.common_timeout
        if rm_state.is_down:
            LOGGER.warning(f"Node {target_addr} is marked as down, skip sending RPC and return TimeoutError immediately.")
            raise GrpcCommDownError(f"Node {target_addr} is marked as down.")
        try:
            ret = await rm_state.robj.remote_call(key, *args, rpc_timeout=timeout, rpc_wait_for_ready=rpc_wait_for_ready, **kwargs)
            # reset down count after successful call
            rm_state.down_count_remain = self._cfg.max_fail_before_mark_down
            return ret
        except grpc.aio.AioRpcError as e:
            if self._cfg.max_fail_before_mark_down > 0:

                if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    rm_state.down_count_remain -= 1
                    LOGGER.warning(f"RPC to {target_addr} timed out. Remaining retries before marking down: {rm_state.down_count_remain}")
                    raise GrpcCommDownError(f"RPC to {target_addr} timed out.") from e
                else:
                    # for other types of gRPC errors, we can choose to not mark the node as down, since it might be a transient error.
                    LOGGER.error(f"RPC to {target_addr} failed with error: {e}")
                    # rm_state.down_count_remain -= 1
                    raise
            else:
                LOGGER.error(f"RPC to {target_addr} failed with error: {e}")
                raise
        
    async def remote_generator(self,
                          target_addr: str,
                          key: str,
                          *args: Any,
                          rpc_timeout: Optional[int] = None,
                          **kwargs: Any) -> AsyncGenerator[Any, None]:
        """Make an asynchronous remote generator call to the target."""
        rm_state = await self._get_or_create_remote(target_addr)
        timeout = rpc_timeout or self._cfg.common_timeout
        try:
            async for item in rm_state.robj.remote_generator(key, *args, rpc_timeout=timeout, **kwargs):
                yield item
            rm_state.down_count_remain = self._cfg.max_fail_before_mark_down
        except grpc.aio.AioRpcError as e:
            if self._cfg.max_fail_before_mark_down > 0:
                if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    rm_state.down_count_remain -= 1
                    LOGGER.warning(f"RPC generator to {target_addr} timed out. Remaining retries before marking down: {rm_state.down_count_remain}")
                    raise GrpcCommDownError(f"RPC generator to {target_addr} timed out.") from e
                else:
                    LOGGER.error(f"RPC generator to {target_addr} failed with error: {e}")
                    rm_state.down_count_remain -= 1
                    raise
            else:
                LOGGER.error(f"RPC generator to {target_addr} failed with error: {e}")
                raise

    async def client_stream(self,
                          target_addr: str,
                          key: str,
                          stream_iter: Union[AsyncGenerator[Any, None], Generator[Any, None]],
                          *args: Any,
                          rpc_timeout: Optional[int] = None,
                          **kwargs: Any) -> Any:
        """Make an asynchronous client stream call to the target."""
        rm_state = await self._get_or_create_remote(target_addr)
        timeout = rpc_timeout or self._cfg.common_timeout
        if rm_state.is_down:
            LOGGER.warning(f"Node {target_addr} is marked as down, skip sending RPC and return TimeoutError immediately.")
            raise GrpcCommDownError(f"Node {target_addr} is marked as down.")
        try:
            ret = await rm_state.robj.client_stream(key, stream_iter, *args, rpc_timeout=timeout, **kwargs)
            rm_state.down_count_remain = self._cfg.max_fail_before_mark_down
            return ret
        except grpc.aio.AioRpcError as e:
            if self._cfg.max_fail_before_mark_down > 0:

                if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    rm_state.down_count_remain -= 1
                    LOGGER.warning(f"Client stream RPC to {target_addr} timed out. Remaining retries before marking down: {rm_state.down_count_remain}")
                    raise GrpcCommDownError(f"Client stream RPC to {target_addr} timed out.") from e
                else:
                    LOGGER.error(f"Client stream RPC to {target_addr} failed with error: {e}")
                    rm_state.down_count_remain -= 1
                    raise
            else:
                LOGGER.error(f"Client stream RPC to {target_addr} failed with error: {e}")
                raise

    async def bi_stream(self,
                          target_addr: str,
                          key: str,
                          stream_iter: Union[AsyncGenerator[Any, None], Generator[Any, None]],
                          *args: Any,
                          rpc_timeout: Optional[int] = None,
                          **kwargs: Any) -> AsyncGenerator[Any, None]:
        """Make an asynchronous bi stream call to the target."""
        rm_state = await self._get_or_create_remote(target_addr)
        timeout = rpc_timeout or self._cfg.common_timeout
        if rm_state.is_down:
            LOGGER.warning(f"Node {target_addr} is marked as down, skip sending RPC and return TimeoutError immediately.")
            raise GrpcCommDownError(f"Node {target_addr} is marked as down.")
        try:
            async for item in rm_state.robj.bi_stream(key, stream_iter, *args, rpc_timeout=timeout, **kwargs):
                yield item
            rm_state.down_count_remain = self._cfg.max_fail_before_mark_down
        except grpc.aio.AioRpcError as e:
            if self._cfg.max_fail_before_mark_down > 0:
                if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    rm_state.down_count_remain -= 1
                    LOGGER.warning(f"Bi-stream RPC to {target_addr} timed out. Remaining retries before marking down: {rm_state.down_count_remain}")
                    raise GrpcCommDownError(f"Bi-stream RPC to {target_addr} timed out.") from e
                else:
                    LOGGER.error(f"Bi-stream RPC to {target_addr} failed with error: {e}")
                    rm_state.down_count_remain -= 1
                    raise
            else:
                LOGGER.error(f"Bi-stream RPC to {target_addr} failed with error: {e}")
                raise

    async def close(self):
        """Close the communication channel and release resources."""
        for rm_state in self._remote_states.values():
            await rm_state.robj.close()