import asyncio
from contextlib import suppress


async def cancel_task(task):
    # more info: https://stackoverflow.com/a/43810272/1113207
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


async def cancel_task_ignore_all_error(task):
    # more info: https://stackoverflow.com/a/43810272/1113207
    task.cancel()
    with suppress(Exception):
        await task
