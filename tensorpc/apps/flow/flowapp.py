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

"""Flow APP: simple GUI application in devflow
"""

import asyncio 

async def async_range(start, stop=None, step=1):
    """same as range but schedule other tasks to run in every iteration
    """
    if stop:
        range_ = range(start, stop, step)
    else:
        range_ = range(start)
    for i in range_:
        yield i
        # allow other task run, important for 
        # long-time loop based sync task
        await asyncio.sleep(0)
