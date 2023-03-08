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

import subprocess
import fire


def start_flow_remote_worker(name: str, port: int, http_port: int):
    try:
        output = subprocess.check_output(["tmux", "-V"])
    except FileNotFoundError:
        print("Can't find tmux. please install it first.")
        raise
    try:
        sess_output = subprocess.check_output(["tmux", "ls"]).decode("utf-8")
        sess_lines = sess_output.strip().split("\n")
        sess_names = [s.split(":")[0] for s in sess_lines]
    except:
        sess_names = []
    if name in sess_names:
        return
    output = subprocess.check_output(["tmux", "new-session", "-d", "-s", name])
    cmd = f"python -m tensorpc.serve --port={port} --http_port={http_port} && exit"
    output = subprocess.check_output(
        ["tmux", "send-keys", "-t", name, f"{cmd}", "ENTER"])
    return


if __name__ == "__main__":
    fire.Fire(start_flow_remote_worker)
