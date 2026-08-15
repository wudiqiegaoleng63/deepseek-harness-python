"""Hierarchical service context with reversible effects."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, TypeAlias

from .events import EventBus

Disposer: TypeAlias = Callable[[], Any | Awaitable[Any]]
MaybeValue: TypeAlias = Any | Awaitable[Any]


async def _resolve(value: MaybeValue) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class Context:
    """A service scope owned by one plugin fiber.

    Child contexts see parent services but own their registrations and
    disposers.  A plugin can therefore compose agent-local capabilities and
    unwind them without mutating the host context.
    """

    def __init__(self, parent: Context | None = None, *, events: EventBus | None = None) -> None:
        self.parent = parent
        self.events = events if events is not None else (parent.events if parent else EventBus())
        self._services: dict[str, Any] = {}
        self._disposers: list[Disposer] = []
        self._disposed = False

    def child(self) -> Context:
        self._ensure_open()
        return Context(self)

    def provide(self, key: str, value: Any, *, replace: bool = False) -> Disposer:
        self._ensure_open()
        if key in self._services and not replace:
            raise RuntimeError(f"service already provided in this scope: {key}")
        previous = self._services.get(key)
        had_previous = key in self._services
        self._services[key] = value

        active = True

        def dispose() -> None:
            nonlocal active
            if not active:
                return
            active = False
            if self._services.get(key) is not value:
                return
            if had_previous:
                self._services[key] = previous
            else:
                self._services.pop(key, None)

        self.add_disposer(dispose)
        return dispose

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._services:
            return self._services[key]
        if self.parent is not None:
            return self.parent.get(key, default)
        return default

    def require(self, key: str) -> Any:
        value = self.get(key)
        if value is None:
            raise LookupError(f"required service is unavailable: {key}")
        return value

    def add_disposer(self, disposer: Disposer) -> Disposer:
        self._ensure_open()
        self._disposers.append(disposer)
        return disposer

    async def effect(self, setup: Callable[[], MaybeValue], description: str = "") -> Any:
        """Run a setup function and register its returned disposer."""

        self._ensure_open()
        try:
            result = await _resolve(setup())
        except Exception as exc:
            suffix = f" ({description})" if description else ""
            raise RuntimeError(f"context effect failed{suffix}") from exc
        if callable(result):
            self.add_disposer(result)
        return result

    def subscribe(self, event: str, handler: Callable[..., Any], *, priority: int = 0) -> Disposer:
        disposer = self.events.subscribe(event, handler, priority=priority)
        self.add_disposer(disposer)
        return disposer

    async def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        failures: list[Exception] = []
        while self._disposers:
            disposer = self._disposers.pop()
            try:
                await _resolve(disposer())
            except Exception as exc:
                failures.append(exc)
        self._services.clear()
        if failures:
            raise ExceptionGroup("context disposal failed", failures)

    def _ensure_open(self) -> None:
        if self._disposed:
            raise RuntimeError("context is disposed")
