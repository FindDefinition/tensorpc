import asyncio
import traceback

import fire
from tensorpc import simple_chunk_call_async
import numpy as np
import time

from tensorpc.apps.collections.shm_kvstore import ShmKVStoreAsyncClient, ShmTrOnlyKVStoreTensorClient
from tensorpc.core.asyncclient import AsyncRemoteManager
from tensorpc.core.client import RemoteManager


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

async def main_async_v2(addr: str, size: int):
    import torch 
    try:
        np.random.seed(5)
        data = [
            np.random.uniform(size=[size * 1024 * 1024 // 4]).astype(np.float32),
            # np.random.uniform(size=[size * 1024 * 1024 // 16]).astype(np.float32),
            # np.random.uniform(size=[size * 1024 * 1024 // 16]).astype(np.float32),
            # np.random.uniform(size=[size * 1024 * 1024 // 16]).astype(np.float32),
        ]
        print(data[0].nbytes)
        with RemoteManager(addr) as robj:
            shm_client = ShmTrOnlyKVStoreTensorClient(robj)
            for j in range(3):
                start = time.time()
                shm_client.store_tensor_tree("test", [torch.from_numpy(d) for d in data])
                end_time = time.time()
                print(f"store_array_tree usetime: {end_time - start}, speed: {size / (end_time - start)} MB/s")
                start = time.time()
                dara_received = shm_client.get_tensor_tree("test")
                end_time = time.time()
                print(f"get_array_tree usetime: {end_time - start}, speed: {size / (end_time - start)} MB/s")
            for i in range(len(data)):
                assert np.allclose(dara_received[i].numpy(), data[i])
    except:
        traceback.print_exc()
        raise

def main(addr: str, size: int):
    return asyncio.run(main_async_v2(addr, size))


if __name__ == "__main__":
    fire.Fire(main)
