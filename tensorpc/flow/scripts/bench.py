# Copyright 2022 Yan Yan
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorpc 

import aiohttp
import asyncio 
from cumm import tensorview as tv 
async def main():
    async with tensorpc.AsyncRemoteManager("localhost:51051") as robj:
        with tv.measure_and_print("GRPC Time"):
            for i in range(100):
                await robj.remote_call("tensorpc.services.collection:Simple.echo", 185)
    async with aiohttp.ClientSession() as sess:
        with tv.measure_and_print("HTTP Time"):
            for i in range(100):
                await tensorpc.http_remote_call(sess, "http://localhost:51052/api/rpc", 
                    "tensorpc.services.collection:Simple.echo", 185)

    pass 
if __name__ == "__main__":
    asyncio.run(main())