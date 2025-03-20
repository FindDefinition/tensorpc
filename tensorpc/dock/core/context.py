from typing import Any, Callable, ContextManager, Optional, Set, Tuple


ALL_APP_CONTEXT_GETTERS: Set[Tuple[Callable[[], Optional[Any]], Callable[[Any], ContextManager]]] = set()

