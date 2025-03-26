import asyncio
import traceback

import fire
from tensorpc import simple_chunk_call_async
import numpy as np
import time


async def main_async(addr: str, size: int, is_send: bool):
    try:
        np.random.seed(5)
        data = np.random.uniform(size=[size * 1024 * 1024 // 4]).astype(np.float32)
        if is_send:
            start = time.time()
            await simple_chunk_call_async(
                addr,
                "tensorpc.services.collection::SpeedTestServer.recv_data",
                data)
        else:
            # cache data to avoid measure the time of data generation
            await simple_chunk_call_async(
                addr,
                "tensorpc.services.collection::SpeedTestServer.send_data",
                size)
            start = time.time()
            await simple_chunk_call_async(
                addr,
                "tensorpc.services.collection::SpeedTestServer.send_data",
                size)
        end_time = time.time()
        speed = (size) / (end_time - start)
        print("usetime: {}, speed: {:.2f} MB/s".format(end_time - start,
                                                       speed))
    except:
        traceback.print_exc()
        raise


def main(addr: str, size: int, is_send: bool):
    return asyncio.run(main_async(addr, size, is_send))


if __name__ == "__main__":
    fire.Fire(main)
