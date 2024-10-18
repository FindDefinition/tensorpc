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

serv_names = _ServiceNames()
