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
    def FLOW_SSH_INPUT(self):
        from tensorpc.apps.flow.serv.core import Flow
        return get_service_key_by_type(Flow, Flow.command_node_input.__name__)

    @property
    def FLOW_PUT_WORKER_EVENT(self):
        from tensorpc.apps.flow.serv.core import Flow
        return get_service_key_by_type(Flow,
                                       Flow.put_event_from_worker.__name__)

    @property
    def FLOW_ADD_MESSAGE(self):
        from tensorpc.apps.flow.serv.core import Flow
        return get_service_key_by_type(Flow,
                                       Flow.add_message.__name__)

    @property
    def FLOW_PUT_APP_EVENT(self):
        from tensorpc.apps.flow.serv.core import Flow
        return get_service_key_by_type(Flow,
                                       Flow.put_app_event.__name__)

    @property
    def FLOWWORKER_PUT_WORKER_EVENT(self):
        from tensorpc.apps.flow.serv.worker import FlowWorker
        return get_service_key_by_type(FlowWorker,
                                       FlowWorker.put_relay_event.__name__)

    @property
    def FLOWWORKER_PUT_WORKER_EVENT_JSON(self):
        from tensorpc.apps.flow.serv.worker import FlowWorker
        return get_service_key_by_type(
            FlowWorker, FlowWorker.put_relay_event_json.__name__)

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
    def FLOWWORKER_CLOSE_CONNECTION(self):
        from tensorpc.apps.flow.serv.worker import FlowWorker
        return get_service_key_by_type(
            FlowWorker, FlowWorker.close_grpc_connection.__name__)

    @property
    def FLOWWORKER_STOP(self):
        from tensorpc.apps.flow.serv.worker import FlowWorker
        return get_service_key_by_type(FlowWorker, FlowWorker.stop.__name__)

    @property
    def FLOWWORKER_QUERY_STATUS(self):
        from tensorpc.apps.flow.serv.worker import FlowWorker
        return get_service_key_by_type(
            FlowWorker, FlowWorker.query_nodes_status.__name__)

    @property
    def FLOWWORKER_SELECT_NODE(self):
        from tensorpc.apps.flow.serv.worker import FlowWorker
        return get_service_key_by_type(FlowWorker,
                                       FlowWorker.select_node.__name__)

    @property
    def FLOWWORKER_SET_TERMINAL_STATE(self):
        from tensorpc.apps.flow.serv.worker import FlowWorker
        return get_service_key_by_type(FlowWorker,
                                       FlowWorker.save_terminal_state.__name__)

    @property
    def FLOWWORKER_COMMAND_NODE_INPUT(self):
        from tensorpc.apps.flow.serv.worker import FlowWorker
        return get_service_key_by_type(FlowWorker,
                                       FlowWorker.command_node_input.__name__)

    @property
    def FLOWWORKER_SSH_CHANGE_SIZE(self):
        from tensorpc.apps.flow.serv.worker import FlowWorker
        return get_service_key_by_type(FlowWorker,
                                       FlowWorker.ssh_change_size.__name__)
    @property
    def FLOWWORKER_STOP_SESSION(self):
        from tensorpc.apps.flow.serv.worker import FlowWorker
        return get_service_key_by_type(FlowWorker,
                                       FlowWorker.stop_session.__name__)
    @property
    def FLOWWORKER_EXIT(self):
        from tensorpc.apps.flow.serv.worker import FlowWorker
        return get_service_key_by_type(FlowWorker,
                                       FlowWorker.exit.__name__)

    @property
    def FLOWWORKER_SYNC_GRAPH(self):
        from tensorpc.apps.flow.serv.worker import FlowWorker
        return get_service_key_by_type(FlowWorker,
                                       FlowWorker.sync_graph.__name__)

    @property
    def FLOWWORKER_DELETE_MESSAGE(self):
        from tensorpc.apps.flow.serv.worker import FlowWorker
        return get_service_key_by_type(FlowWorker,
                                       FlowWorker.delete_message.__name__)

    @property
    def FLOWWORKER_QUERY_MESSAGE(self):
        from tensorpc.apps.flow.serv.worker import FlowWorker
        return get_service_key_by_type(FlowWorker,
                                       FlowWorker.query_message.__name__)

    @property
    def FLOWWORKER_ADD_MESSAGE(self):
        from tensorpc.apps.flow.serv.worker import FlowWorker
        return get_service_key_by_type(FlowWorker,
                                       FlowWorker.add_message.__name__)

    @property
    def FLOWWORKER_QUERY_MESSAGE_DETAIL(self):
        from tensorpc.apps.flow.serv.worker import FlowWorker
        return get_service_key_by_type(FlowWorker,
                                       FlowWorker.query_single_message_detail.__name__)
    
    @property
    def FLOWWORKER_PUT_APP_EVENT(self):
        from tensorpc.apps.flow.serv.worker import FlowWorker
        return get_service_key_by_type(FlowWorker,
                                       FlowWorker.put_app_event.__name__)
    @property
    def FLOWWORKER_RUN_APP_UI_EVENT(self):
        from tensorpc.apps.flow.serv.worker import FlowWorker
        return get_service_key_by_type(FlowWorker,
                                       FlowWorker.run_ui_event.__name__)
    @property
    def FLOWWORKER_APP_GET_LAYOUT(self):
        from tensorpc.apps.flow.serv.worker import FlowWorker
        return get_service_key_by_type(FlowWorker,
                                       FlowWorker.get_layout.__name__)

    @property
    def FLOWWORKER_RUN_APP_EDITOR_EVENT(self):
        from tensorpc.apps.flow.serv.worker import FlowWorker
        return get_service_key_by_type(FlowWorker,
                                       FlowWorker.run_app_editor_event.__name__)

    @property
    def APP_RUN_UI_EVENT(self):
        from tensorpc.apps.flow.serv.flowapp import FlowApp
        return get_service_key_by_type(FlowApp,
                                       FlowApp.run_ui_event.__name__)

    @property
    def APP_RUN_APP_EDITOR_EVENT(self):
        from tensorpc.apps.flow.serv.flowapp import FlowApp
        return get_service_key_by_type(FlowApp,
                                       FlowApp.run_app_editor_event.__name__)

    @property
    def APP_GET_LAYOUT(self):
        from tensorpc.apps.flow.serv.flowapp import FlowApp
        return get_service_key_by_type(FlowApp,
                                       FlowApp.get_layout.__name__)

    @property
    def APP_RUN_SCHEDULE_EVENT(self):
        from tensorpc.apps.flow.serv.flowapp import FlowApp
        return get_service_key_by_type(FlowApp,
                                       FlowApp.run_schedule_event.__name__)

serv_names = _ServiceNames()