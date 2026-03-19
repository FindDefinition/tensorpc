from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Generator, Optional, Union

class AsyncRPCCommBase(ABC):
    @abstractmethod
    async def remote_call(self,
                          target_addr: str,
                          key: str,
                          *args: Any,
                          rpc_timeout: Optional[int] = None,
                          rpc_wait_for_ready: bool = False,
                          **kwargs: Any) -> Any:
        """Make an asynchronous remote call to the target."""

    @abstractmethod
    def remote_generator(self,
                          target_addr: str,
                          key: str,
                          *args: Any,
                          rpc_timeout: Optional[int] = None,
                          **kwargs: Any) -> AsyncGenerator[Any, None]:
        """Make an asynchronous remote generator call to the target."""
        raise NotImplementedError

    @abstractmethod
    async def client_stream(self,
                          target_addr: str,
                          key: str,
                          stream_iter: Union[AsyncGenerator[Any, None], Generator[Any, None, None]],
                          *args: Any,
                          rpc_timeout: Optional[int] = None,
                          **kwargs: Any) -> Any:
        """Make an asynchronous client stream call to the target."""

    @abstractmethod
    def bi_stream(self,
                          target_addr: str,
                          key: str,
                          stream_iter: Union[AsyncGenerator[Any, None], Generator[Any, None, None]],
                          *args: Any,
                          rpc_timeout: Optional[int] = None,
                          **kwargs: Any) -> AsyncGenerator[Any, None]:
        """Make an asynchronous bi stream call to the target."""

    async def close(self):
        """Close the communication channel and release resources."""
        pass
