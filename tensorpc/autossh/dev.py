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

import re 

pat = re.compile(r"(?:\033\]633;[ABPCFGD](;.*)?\007)")

match = pat.search("\x1b]633;C\x07\x1b]633;D\x07\x1b]633;P;Cwd=/home/yy\x07\x1b]633;A\x07(base) yy@yy-Lenovo-Legion-R9000K2021H:~$ \x1b]633;B\x07\r")
print(match)