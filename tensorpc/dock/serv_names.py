# Copyright 2024 Yan Yan
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
        from tensorpc.dock.serv.core import Flow
        return get_service_key_by_type(Flow, Flow.update_node_status.__name__)

    @property
    def FLOW_SSH_INPUT(self):
        from tensorpc.dock.serv.core import Flow
        return get_service_key_by_type(Flow, Flow.command_node_input.__name__)

    @property
    def FLOW_PUT_WORKER_EVENT(self):
        from tensorpc.dock.serv.core import Flow
        return get_service_key_by_type(Flow,
                                       Flow.put_event_from_worker.__name__)

    @property
    def FLOW_ADD_MESSAGE(self):
        from tensorpc.dock.serv.core import Flow
        return get_service_key_by_type(Flow, Flow.add_message.__name__)

    @property
    def FLOW_QUERY_APP_NODE_URLS(self):
        from tensorpc.dock.serv.core import Flow
        return get_service_key_by_type(Flow, Flow.query_app_node_urls.__name__)
    
    @property
    def FLOW_QUERY_ALL_RUNNING_APPS(self):
        from tensorpc.dock.serv.core import Flow
        return get_service_key_by_type(Flow, Flow.query_all_running_app_nodes.__name__)

    @property
    def FLOW_PUT_APP_EVENT(self):
        from tensorpc.dock.serv.core import Flow
        return get_service_key_by_type(Flow, Flow.put_app_event.__name__)

    @property
    def FLOW_PUT_APP_EVENT_STREAM(self):
        from tensorpc.dock.serv.core import Flow
        return get_service_key_by_type(Flow, Flow.put_app_event_stream.__name__)

    @property
    def FLOW_RUN_APP_SERVICE(self):
        from tensorpc.dock.serv.core import Flow
        return get_service_key_by_type(Flow, Flow.run_app_service.__name__)

    @property
    def FLOW_RUN_APP_ASYNC_GEN_SERVICE(self):
        from tensorpc.dock.serv.core import Flow
        return get_service_key_by_type(Flow, Flow.run_app_async_gen_service.__name__)

    @property
    def APP_RUN_SINGLE_EVENT(self):
        from tensorpc.dock.serv.flowapp import FlowApp
        return get_service_key_by_type(FlowApp,
                                       FlowApp.run_single_event.__name__)

    @property
    def APP_SIMPLE_RPC(self):
        from tensorpc.dock.serv.flowapp import FlowApp
        return get_service_key_by_type(FlowApp, FlowApp.handle_simple_rpc.__name__)
    
    @property
    def APP_RUN_SERVICE(self):
        from tensorpc.dock.serv.flowapp import FlowApp
        return get_service_key_by_type(FlowApp, FlowApp.run_app_service.__name__)
    
    @property
    def APP_RUN_ASYNC_GEN_SERVICE(self):
        from tensorpc.dock.serv.flowapp import FlowApp
        return get_service_key_by_type(FlowApp, FlowApp.run_app_async_gen_service.__name__)

    @property
    def APP_RELAY_APP_EVENT_FROM_REMOTE(self):
        from tensorpc.dock.serv.flowapp import FlowApp
        return get_service_key_by_type(FlowApp,
                                       FlowApp.relay_app_event_from_remote_component.__name__)

    @property
    def APP_RELAY_APP_STORAGE_FROM_REMOTE(self):
        from tensorpc.dock.serv.flowapp import FlowApp
        return get_service_key_by_type(FlowApp,
                                       FlowApp.relay_app_storage_from_remote_comp.__name__)

    @property
    def APP_REMOTE_COMP_SHUTDOWN(self):
        from tensorpc.dock.serv.flowapp import FlowApp
        return get_service_key_by_type(FlowApp,
                                       FlowApp.remote_comp_shutdown.__name__)

    @property
    def APP_RUN_REMOTE_COMP_EVENT(self):
        from tensorpc.dock.serv.flowapp import FlowApp
        return get_service_key_by_type(FlowApp, FlowApp.handle_msg_from_remote_comp.__name__)

    @property
    def APP_GET_LAYOUT(self):
        from tensorpc.dock.serv.flowapp import FlowApp
        return get_service_key_by_type(FlowApp, FlowApp.get_layout.__name__)

    @property
    def APP_GET_FILE(self):
        from tensorpc.dock.serv.flowapp import FlowApp
        return get_service_key_by_type(FlowApp, FlowApp.get_file.__name__)

    @property
    def APP_GET_FILE_METADATA(self):
        from tensorpc.dock.serv.flowapp import FlowApp
        return get_service_key_by_type(FlowApp, FlowApp.get_file_metadata.__name__)

    @property
    def APP_GET_VSCODE_BREAKPOINTS(self):
        from tensorpc.dock.serv.flowapp import FlowApp
        return get_service_key_by_type(FlowApp,
                                       FlowApp.get_vscode_breakpoints.__name__)

    @property
    def FLOW_DATA_SAVE(self):
        from tensorpc.dock.serv.core import Flow
        return get_service_key_by_type(Flow,
                                       Flow.save_data_to_storage.__name__)

    @property
    def FLOW_DATA_UPDATE(self):
        from tensorpc.dock.serv.core import Flow
        return get_service_key_by_type(Flow,
                                       Flow.update_data_in_storage.__name__)

    @property
    def FLOW_DATA_READ(self):
        from tensorpc.dock.serv.core import Flow
        return get_service_key_by_type(Flow,
                                       Flow.read_data_from_storage.__name__)
    @property
    def FLOW_DATA_READ_GLOB_PREFIX(self):
        from tensorpc.dock.serv.core import Flow
        return get_service_key_by_type(Flow,
                                       Flow.read_data_from_storage_by_glob_prefix.__name__)

    @property
    def FLOW_DATA_HAS_ITEM(self):
        from tensorpc.dock.serv.core import Flow
        return get_service_key_by_type(Flow,
                                       Flow.has_data_item.__name__)


    @property
    def FLOW_DATA_LIST_ITEM_METAS(self):
        from tensorpc.dock.serv.core import Flow
        return get_service_key_by_type(Flow, Flow.query_data_attrs.__name__)

    @property
    def FLOW_DATA_QUERY_DATA_NODE_IDS(self):
        from tensorpc.dock.serv.core import Flow
        return get_service_key_by_type(Flow,
                                       Flow.query_all_data_node_ids.__name__)

    @property
    def FLOW_DATA_DELETE_ITEM(self):
        from tensorpc.dock.serv.core import Flow
        return get_service_key_by_type(Flow,
                                       Flow.delete_datastorage_data.__name__)

    @property
    def FLOW_DATA_DELETE_FOLDER(self):
        from tensorpc.dock.serv.core import Flow
        return get_service_key_by_type(Flow,
                                       Flow.remove_folder_from_storage.__name__)

    @property
    def FLOW_DATA_RENAME_ITEM(self):
        from tensorpc.dock.serv.core import Flow
        return get_service_key_by_type(Flow,
                                       Flow.rename_datastorage_data.__name__)

    @property
    def FLOW_GET_SSH_NODE_DATA(self):
        from tensorpc.dock.serv.core import Flow
        return get_service_key_by_type(Flow, Flow.get_ssh_node_data.__name__)

    @property
    def REMOTE_COMP_RUN_SINGLE_EVENT(self):
        from tensorpc.dock.serv.remote_comp import RemoteComponentService
        return get_service_key_by_type(RemoteComponentService, RemoteComponentService.run_single_event.__name__)

    @property
    def REMOTE_COMP_SIMPLE_RPC(self):
        from tensorpc.dock.serv.remote_comp import RemoteComponentService
        return get_service_key_by_type(RemoteComponentService, RemoteComponentService.handle_simple_rpc.__name__)

    @property
    def REMOTE_COMP_GET_LAYOUT(self):
        from tensorpc.dock.serv.remote_comp import RemoteComponentService
        return get_service_key_by_type(RemoteComponentService, RemoteComponentService.get_layout_dict.__name__)
    
    @property
    def REMOTE_COMP_SET_LAYOUT_OBJECT(self):
        from tensorpc.dock.serv.remote_comp import RemoteComponentService
        return get_service_key_by_type(RemoteComponentService, RemoteComponentService.set_layout_object.__name__)

    @property
    def REMOTE_COMP_HAS_LAYOUT_OBJECT(self):
        from tensorpc.dock.serv.remote_comp import RemoteComponentService
        return get_service_key_by_type(RemoteComponentService, RemoteComponentService.has_layout_object.__name__)

    @property
    def REMOTE_COMP_REMOVE_LAYOUT_OBJECT(self):
        from tensorpc.dock.serv.remote_comp import RemoteComponentService
        return get_service_key_by_type(RemoteComponentService, RemoteComponentService.remove_layout_object.__name__)

    @property
    def REMOTE_COMP_GET_LAYOUT_ROOT_BY_KEY(self):
        from tensorpc.dock.serv.remote_comp import RemoteComponentService
        return get_service_key_by_type(RemoteComponentService, RemoteComponentService.get_layout_root_and_app_by_key.__name__)

    @property
    def REMOTE_COMP_GET_FILE(self):
        from tensorpc.dock.serv.remote_comp import RemoteComponentService
        return get_service_key_by_type(RemoteComponentService, RemoteComponentService.get_file.__name__)

    @property
    def REMOTE_COMP_GET_FILE_METADATA(self):
        from tensorpc.dock.serv.remote_comp import RemoteComponentService
        return get_service_key_by_type(RemoteComponentService, RemoteComponentService.get_file_metadata.__name__)

    @property
    def REMOTE_COMP_RUN_REMOTE_COMP_EVENT(self):
        from tensorpc.dock.serv.remote_comp import RemoteComponentService
        return get_service_key_by_type(RemoteComponentService, RemoteComponentService.handle_msg_from_remote_comp.__name__)

    @property
    def REMOTE_COMP_MOUNT_APP_GENERATOR(self):
        from tensorpc.dock.serv.remote_comp import RemoteComponentService
        return get_service_key_by_type(RemoteComponentService, RemoteComponentService.mount_app_generator.__name__)


serv_names = _ServiceNames()
