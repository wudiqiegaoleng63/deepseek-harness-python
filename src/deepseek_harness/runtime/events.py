"""Small async event bus used by the plugin runtime and live Agent views."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

Payload: TypeAlias = Any
MaybeAwaitable: TypeAlias = Awaitable[Any] | Any
EventHandler: TypeAlias = Callable[[Payload], MaybeAwaitable]
WaterfallHandler: TypeAlias = Callable[
    [Payload, Callable[[Payload], Awaitable[Any]]], MaybeAwaitable
]


async def _resolve(value: MaybeAwaitable) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


@dataclass(frozen=True, slots=True)
class _Subscription:
    event: str
    priority: int
    order: int
    handler: EventHandler | WaterfallHandler


class EventBus:
    """Ordered serial events plus explicit waterfall dispatch.

    Handlers are process-local and are removed through the disposer returned by
    :meth:`subscribe`.  This mirrors DSH's reversible plugin effects without
    imposing a static type-generation step on Python consumers.
    """

    def __init__(self) -> None:
        self._subscriptions: dict[str, list[_Subscription]] = {}
        self._next_order = 0

    def subscribe(
        self,
        event: str,
        handler: EventHandler | WaterfallHandler,
        *,
        priority: int = 0,
    ) -> Callable[[], None]:
        subscription = _Subscription(event, priority, self._next_order, handler)
        self._next_order += 1
        entries = self._subscriptions.setdefault(event, [])
        entries.append(subscription)
        entries.sort(key=lambda item: (-item.priority, item.order))
        active = True

        def dispose() -> None:
            nonlocal active
            if not active:
                return
            active = False
            current = self._subscriptions.get(event)
            if current is None:
                return
            try:
                current.remove(subscription)
            except ValueError:
                pass
            if not current:
                self._subscriptions.pop(event, None)

        return dispose

    async def emit(self, event: str, payload: Payload = None) -> None:
        """Invoke all current listeners in deterministic priority order."""

        for subscription in tuple(self._subscriptions.get(event, ())):
            await _resolve(subscription.handler(payload))  # type: ignore[arg-type]

    async def waterfall(
        self,
        event: str,
        payload: Payload,
        terminal: Callable[[Payload], MaybeAwaitable],
    ) -> Any:
        """Run listeners as a cooperative ``next(value)`` chain.

        A listener that does not call ``next`` owns the result, which allows a
        policy plugin to reject or replace a request before the terminal action.
        """

        listeners = tuple(self._subscriptions.get(event, ()))

        async def dispatch(index: int, value: Payload) -> Any:
            if index >= len(listeners):
                return await _resolve(terminal(value))
            handler = listeners[index].handler
            delegated = False

            async def next_handler(next_value: Payload = value) -> Any:
                nonlocal delegated
                delegated = True
                return await dispatch(index + 1, next_value)

            result = await _resolve(handler(value, next_handler))  # type: ignore[misc]
            if delegated:
                return result
            return result

        return await dispatch(0, payload)
