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

from typing import Dict
from tensorpc.flow import App, EditableApp, EditableLayoutApp, mui
from tensorpc.flow.flowapp.core import Component

from tensorpc.autossh import SSHClient
import asyncssh
from pathlib import Path


class DownloadFromSSH(EditableLayoutApp):

    def app_create_layout(self) -> Dict[str, Component]:
        self.ssh_url = mui.TextField("Url")
        self.ssh_username = mui.TextField("Username")
        self.ssh_password = mui.TextField("Password").prop(type="password")

        self.ssh_files = mui.TextField("Files").prop(multiline=True)
        self.target_loc = mui.TextField("Target")
        self.run = mui.Button("Run", self.download_files)

        return {
            "container":
            mui.VBox(
                {
                    "url": self.ssh_url,
                    "ssh_files": self.ssh_files,
                    "ssh_username": self.ssh_username,
                    "ssh_password": self.ssh_password,
                    "target_loc": self.target_loc,
                    "run": self.run,
                }
                ).prop(width=480,
                height=480)
        }

    async def download_files(self):
        client = SSHClient(self.ssh_url.value, self.ssh_username.value,
                           self.ssh_password.value, None)
        assert Path(self.target_loc.value).exists()
        async with client.simple_connect() as conn:
            files = self.ssh_files.value.split("\n")
            files = list(
                filter(lambda x: len(x), map(lambda x: x.strip(), files)))
            for f in files:
                await asyncssh.scp((conn, f),
                                   self.target_loc.value,
                                   preserve=True,
                                   recurse=True)
                print("FINISHED!", f)
