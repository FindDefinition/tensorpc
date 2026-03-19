from __future__ import annotations

from collections import OrderedDict
from typing import Generic, TypeVar


K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
	"""A simple fixed-capacity LRU cache.

	- Most recently used items stay in cache.
	- Least recently used item is evicted when capacity is exceeded.
	"""

	def __init__(self, capacity: int):
		if capacity <= 0:
			raise ValueError("capacity must be > 0")
		self.capacity = capacity
		self._items: OrderedDict[K, V] = OrderedDict()

	def __len__(self) -> int:
		return len(self._items)

	def __contains__(self, key: K) -> bool:
		return key in self._items

	def get(self, key: K, default: V | None = None) -> V | None:
		if key not in self._items:
			return default
		self._items.move_to_end(key)
		return self._items[key]

	def put(self, key: K, value: V) -> None:
		if key in self._items:
			self._items.move_to_end(key)
		self._items[key] = value
		if len(self._items) > self.capacity:
			self._items.popitem(last=False)

	def pop(self, key: K, default: V | None = None) -> V | None:
		return self._items.pop(key, default)

	def clear(self) -> None:
		self._items.clear()

	def items(self) -> list[tuple[K, V]]:
		return list(self._items.items())
