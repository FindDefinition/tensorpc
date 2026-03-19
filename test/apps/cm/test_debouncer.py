import asyncio
from unittest.mock import patch

import pytest

from tensorpc.apps.cm.worker import AsyncDebouncer


@pytest.mark.asyncio
async def test_async_debouncer_keeps_only_latest_pending_call():
	debouncer = AsyncDebouncer(wait_time=0.03)
	calls: list[str] = []

	async def callback(name: str):
		calls.append(name)

	await debouncer.call(callback, "first")
	await asyncio.sleep(0.01)
	await debouncer.call(callback, "second")
	await asyncio.sleep(0.06)

	assert calls == ["second"]


@pytest.mark.asyncio
async def test_async_debouncer_waits_for_running_callback_before_queuing_next():
	debouncer = AsyncDebouncer(wait_time=0)
	started = asyncio.Event()
	release_first = asyncio.Event()
	calls: list[str] = []

	async def callback(name: str):
		calls.append(name)
		if name == "first":
			started.set()
			await release_first.wait()

	await debouncer.call(callback, "first")
	await started.wait()

	second_call_task = asyncio.create_task(debouncer.call(callback, "second"))
	await asyncio.sleep(0.01)
	assert second_call_task.done() is False
	assert calls == ["first"]

	release_first.set()
	await second_call_task
	await asyncio.sleep(0.01)

	assert calls == ["first", "second"]


@pytest.mark.asyncio
async def test_async_debouncer_swallows_callback_exceptions_and_recovers():
	debouncer = AsyncDebouncer(wait_time=0)
	calls: list[str] = []

	async def callback(name: str):
		calls.append(name)
		if name == "boom":
			raise RuntimeError("expected test failure")

	with patch("tensorpc.apps.cm.worker.CM_LOGGER.exception") as exception_logger:
		await debouncer.call(callback, "boom")
		await asyncio.sleep(0.01)
		await debouncer.call(callback, "ok")
		await asyncio.sleep(0.01)

	assert calls == ["boom", "ok"]
	exception_logger.assert_called_once_with("Exception occurred in debounced function:")
