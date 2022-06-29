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
    def FLOW_UPDATE_NODE_STATUS(self):
        from tensorpc.apps.flow.serv.core import Flow
        return get_service_key_by_type(Flow, Flow.update_node_status.__name__)

    @property
    def FLOW_PUT_WORKER_EVENT(self):
        from tensorpc.apps.flow.serv.core import Flow
        return get_service_key_by_type(Flow,
                                       Flow.put_event_from_worker.__name__)

    @property
    def FLOWWORKER_PUT_WORKER_EVENT(self):
        from tensorpc.apps.flow.serv.worker import FlowWorker
        return get_service_key_by_type(FlowWorker,
                                       FlowWorker.put_relay_event.__name__)
    @property
    def FLOWWORKER_PUT_WORKER_EVENT_JSON(self):
        from tensorpc.apps.flow.serv.worker import FlowWorker
        return get_service_key_by_type(FlowWorker,
                                       FlowWorker.put_relay_event_json.__name__)

    @property
    def FLOWWORKER_CREATE_SESSION(self):
        from tensorpc.apps.flow.serv.worker import FlowWorker
        return get_service_key_by_type(FlowWorker,
                                       FlowWorker.create_ssh_session.__name__)

    @property
    def FLOWWORKER_CREATE_CONNECTION(self):
        from tensorpc.apps.flow.serv.worker import FlowWorker
        return get_service_key_by_type(FlowWorker,
                                       FlowWorker.create_connection.__name__)

    @property
    def FLOWWORKER_STOP(self):
        from tensorpc.apps.flow.serv.worker import FlowWorker
        return get_service_key_by_type(FlowWorker, FlowWorker.stop.__name__)

    @property
    def FLOWWORKER_QUERY_STATUS(self):
        from tensorpc.apps.flow.serv.worker import FlowWorker
        return get_service_key_by_type(FlowWorker, FlowWorker.query_nodes_last_event.__name__)

serv_names = _ServiceNames()