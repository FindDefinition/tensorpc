from typing import AsyncIterator, Iterator, TypeVar, Optional
from tensorpc.flow.hold.holdctx import hold_ctx, hold_ctx_async
import contextlib
from tensorpc.flow.flowapp.app import App, AppSpecialEventType, get_app_context


@contextlib.contextmanager
def app_hold_ctx(key: str,
                 reload_file=True,
                 *,
                 _frame_cnt=3) -> Iterator[Optional[App]]:
    app_ctx = get_app_context()
    assert app_ctx is not None, "you must use this function on any callback of app"
    with hold_ctx(key, reload_file, _frame_cnt=_frame_cnt) as ctx:
        try:
            app_ctx.app.hold_context = ctx
            app_ctx.app._flowapp_special_eemitter.emit(
                AppSpecialEventType.EnterHoldContext.value, ctx)
            yield app_ctx.app
        finally:
            app_ctx.app.hold_context = None
            app_ctx.app._flowapp_special_eemitter.emit(
                AppSpecialEventType.ExitHoldContext.value, None)


@contextlib.asynccontextmanager
async def app_hold_ctx_async(key: str,
                             reload_file=True,
                             *,
                             _frame_cnt=3) -> AsyncIterator[Optional[App]]:
    app_ctx = get_app_context()
    assert app_ctx is not None, "you must use this function on any callback of app"
    async with hold_ctx_async(key, reload_file, _frame_cnt=_frame_cnt) as ctx:
        try:
            app_ctx.app.hold_context = ctx
            app_ctx.app._flowapp_special_eemitter.emit(
                AppSpecialEventType.EnterHoldContext.value, ctx)
            yield app_ctx.app
        finally:
            app_ctx.app.hold_context = None
            app_ctx.app._flowapp_special_eemitter.emit(
                AppSpecialEventType.ExitHoldContext.value, None)
