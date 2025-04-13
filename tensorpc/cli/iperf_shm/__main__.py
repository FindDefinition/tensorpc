import asyncio
import traceback

import fire
from tensorpc import simple_chunk_call_async
import numpy as np
import time

from tensorpc.apps.collections.shm_kvstore import ShmKVStoreAsyncClient
from tensorpc.core.asyncclient import AsyncRemoteManager


async def main_async(addr: str, size: int):
    try:
        np.random.seed(5)
        data = np.random.uniform(size=[size * 1024 * 1024 // 4]).astype(np.float32)
        async with AsyncRemoteManager(addr) as robj:
            shm_client = ShmKVStoreAsyncClient(robj)
            start = time.time()
            await shm_client.store_array_tree("test", data)
            end_time = time.time()
            print(f"store_array_tree usetime: {end_time - start}, speed: {size / (end_time - start)} MB/s")
            start = time.time()
            data = await shm_client.get_array_tree("test")
            end_time = time.time()
            print(f"get_array_tree usetime: {end_time - start}, speed: {size / (end_time - start)} MB/s")
    except:
        traceback.print_exc()
        raise


def main(addr: str, size: int):
    return asyncio.run(main_async(addr, size))


if __name__ == "__main__":
    fire.Fire(main)
