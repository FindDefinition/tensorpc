from tensorpc.utils import get_service_key_by_type


class _ServiceNames:

    @property
    def DBG_ENTER_BREAKPOINT(self):
        from tensorpc.services.dbg.tools import BackgroundDebugTools
        return get_service_key_by_type(BackgroundDebugTools, BackgroundDebugTools.enter_breakpoint.__name__)
    
    @property
    def DBG_LEAVE_BREAKPOINT(self):
        from tensorpc.services.dbg.tools import BackgroundDebugTools
        return get_service_key_by_type(BackgroundDebugTools, BackgroundDebugTools.leave_breakpoint.__name__)

    @property
    def DBG_SET_DISTRIBUTED_META(self):
        from tensorpc.services.dbg.tools import BackgroundDebugTools
        return get_service_key_by_type(BackgroundDebugTools, BackgroundDebugTools.set_distributed_meta.__name__)

    @property
    def DBG_CURRENT_FRAME_META(self):
        from tensorpc.services.dbg.tools import BackgroundDebugTools
        return get_service_key_by_type(BackgroundDebugTools, BackgroundDebugTools.get_cur_debug_info.__name__)

    @property
    def DBG_SET_BKPTS_AND_GET_CURRENT_INFO(self):
        from tensorpc.services.dbg.tools import BackgroundDebugTools
        return get_service_key_by_type(BackgroundDebugTools, BackgroundDebugTools.set_vscode_breakpoints_and_get_cur_info.__name__)

    @property
    def DBG_INIT_BKPT_DEBUG_PANEL(self):
        from tensorpc.services.dbg.tools import BackgroundDebugTools
        return get_service_key_by_type(BackgroundDebugTools, BackgroundDebugTools.init_bkpt_debug_panel.__name__)

    @property
    def DBG_HANDLE_CODE_SELECTION_MSG(self):
        from tensorpc.services.dbg.tools import BackgroundDebugTools
        return get_service_key_by_type(BackgroundDebugTools, BackgroundDebugTools.handle_code_selection_msg.__name__)

    @property
    def DBG_SET_SKIP_BREAKPOINT(self):
        from tensorpc.services.dbg.tools import BackgroundDebugTools
        return get_service_key_by_type(BackgroundDebugTools, BackgroundDebugTools.set_skip_breakpoint.__name__)

    @property
    def DBG_BKGD_GET_CURRENT_FRAME(self):
        from tensorpc.services.dbg.tools import BackgroundDebugTools
        return get_service_key_by_type(BackgroundDebugTools, BackgroundDebugTools.bkgd_get_cur_frame.__name__)

    @property
    def DBG_SET_VSCODE_BKPTS(self):
        from tensorpc.services.dbg.tools import BackgroundDebugTools
        return get_service_key_by_type(BackgroundDebugTools, BackgroundDebugTools.set_vscode_breakpoints.__name__)

    @property
    def DBG_TRY_FETCH_VSCODE_BREAKPOINTS(self):
        from tensorpc.services.dbg.tools import BackgroundDebugTools
        return get_service_key_by_type(BackgroundDebugTools, BackgroundDebugTools.try_fetch_vscode_breakpoints.__name__)

    @property
    def DBG_SET_TRACE_DATA(self):
        from tensorpc.services.dbg.tools import BackgroundDebugTools
        return get_service_key_by_type(BackgroundDebugTools, BackgroundDebugTools.set_trace_data.__name__)

    @property
    def DBG_SET_TRACER(self):
        from tensorpc.services.dbg.tools import BackgroundDebugTools
        return get_service_key_by_type(BackgroundDebugTools, BackgroundDebugTools.set_tracer.__name__)

    @property
    def DBG_GET_TRACE_DATA(self):
        from tensorpc.services.dbg.tools import BackgroundDebugTools
        return get_service_key_by_type(BackgroundDebugTools, BackgroundDebugTools.get_trace_data.__name__)

    @property
    def DBG_GET_TRACE_DATA_TIMESTAMP(self):
        from tensorpc.services.dbg.tools import BackgroundDebugTools
        return get_service_key_by_type(BackgroundDebugTools, BackgroundDebugTools.get_trace_data_timestamp.__name__)

    @property
    def DBG_GET_TRACE_DATA_KEYS(self):
        from tensorpc.services.dbg.tools import BackgroundDebugTools
        return get_service_key_by_type(BackgroundDebugTools, BackgroundDebugTools.get_trace_data_keys.__name__)

    @property
    def DBG_FORCE_TRACE_STOP(self):
        from tensorpc.services.dbg.tools import BackgroundDebugTools
        return get_service_key_by_type(BackgroundDebugTools, BackgroundDebugTools.force_trace_stop.__name__)

    @property
    def DBG_TRACEVIEW_SET_VARIABLE_INSPECT(self):
        from tensorpc.services.dbg.tools import BackgroundDebugTools
        return get_service_key_by_type(BackgroundDebugTools, BackgroundDebugTools.set_traceview_variable_inspect.__name__)

    @property
    def RT_TRACE_SET_STORAGE(self):
        from tensorpc.services.dbg.rttrace import RTTraceStorageService
        return get_service_key_by_type(RTTraceStorageService, RTTraceStorageService.store_trace.__name__)

    @property
    def RT_TRACE_GET_TRACE_RESULT(self):
        from tensorpc.services.dbg.rttrace import RTTraceStorageService
        return get_service_key_by_type(RTTraceStorageService, RTTraceStorageService.get_trace_result.__name__)

serv_names = _ServiceNames()
