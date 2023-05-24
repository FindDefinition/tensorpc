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

from tensorpc.utils import get_service_key_by_type


class _ServiceNames:

    @property
    def SCHED_TASK_INIT(self):
        from tensorpc.autossh.services.scheduler import Scheduler
        return get_service_key_by_type(Scheduler, Scheduler.init_task.__name__)

    @property
    def SCHED_TASK_SET_EXCEPTION(self):
        from tensorpc.autossh.services.scheduler import Scheduler
        return get_service_key_by_type(Scheduler, Scheduler.set_task_exception.__name__)
    
    @property
    def SCHED_TASK_SET_FINISHED(self):
        from tensorpc.autossh.services.scheduler import Scheduler
        return get_service_key_by_type(Scheduler, Scheduler.set_task_finished.__name__)
    
    @property
    def SCHED_TASK_QUERY_UPDATES(self):
        from tensorpc.autossh.services.scheduler import Scheduler
        return get_service_key_by_type(Scheduler, Scheduler.query_task_updates.__name__)
    
    @property
    def SCHED_TASK_GET_ALL_TASK(self):
        from tensorpc.autossh.services.scheduler import Scheduler
        return get_service_key_by_type(Scheduler, Scheduler.get_all_task_state.__name__)
    
    @property
    def SCHED_TASK_SUBMIT_TASK(self):
        from tensorpc.autossh.services.scheduler import Scheduler
        return get_service_key_by_type(Scheduler, Scheduler.submit_task.__name__)

    @property
    def SCHED_TASK_UPDATE_TASK(self):
        from tensorpc.autossh.services.scheduler import Scheduler
        return get_service_key_by_type(Scheduler, Scheduler.update_task.__name__)
    @property
    def SCHED_TASK_CANCEL_TASK(self):
        from tensorpc.autossh.services.scheduler import Scheduler
        return get_service_key_by_type(Scheduler, Scheduler.cancel_task.__name__)
    
    @property
    def SCHED_TASK_KILL_TASK(self):
        from tensorpc.autossh.services.scheduler import Scheduler
        return get_service_key_by_type(Scheduler, Scheduler.kill_task.__name__)
    
    @property
    def SCHED_TASK_CHECK_STATUS(self):
        from tensorpc.autossh.services.scheduler import Scheduler
        return get_service_key_by_type(Scheduler, Scheduler.check_task_status.__name__)
    
    @property
    def SCHED_TASK_SET_STATUS(self):
        from tensorpc.autossh.services.scheduler import Scheduler
        return get_service_key_by_type(Scheduler, Scheduler.set_task_status.__name__)

    @property
    def SCHED_TASK_DELETE(self):
        from tensorpc.autossh.services.scheduler import Scheduler
        return get_service_key_by_type(Scheduler, Scheduler.delete_task.__name__)

    @property
    def SCHED_TASK_RESOURCE_USAGE(self):
        from tensorpc.autossh.services.scheduler import Scheduler
        return get_service_key_by_type(Scheduler, Scheduler.get_resource_usage.__name__)
    
    @property
    def SCHED_TASK_QUERY_TMUX_PANES(self):
        from tensorpc.autossh.services.scheduler import Scheduler
        return get_service_key_by_type(Scheduler, Scheduler.query_task_tmux_lines.__name__)

serv_names = _ServiceNames()
