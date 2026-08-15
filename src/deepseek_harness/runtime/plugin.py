"""Plugin mounting and lifecycle management."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .context import Context

PluginApply = Callable[[Context], Any | Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class PluginSpec:
    """A named plugin with explicit service prerequisites."""

    name: str
    apply: PluginApply
    requires: tuple[str, ...] = ()


class PluginRuntime:
    """Own the root context and mounted plugin scopes."""

    def __init__(self) -> None:
        self.root = Context()
        self._plugins: dict[str, Context] = {}

    async def install(self, plugin: PluginSpec) -> Context:
        if plugin.name in self._plugins:
            raise RuntimeError(f"plugin already installed: {plugin.name}")
        missing = [key for key in plugin.requires if self.root.get(key) is None]
        if missing:
            raise LookupError(
                f"plugin {plugin.name} requires unavailable services: {', '.join(missing)}"
            )
        scope = self.root.child()
        try:
            result = plugin.apply(scope)
            if inspect.isawaitable(result):
                result = await result
            if callable(result):
                scope.add_disposer(result)
        except Exception:
            await scope.dispose()
            raise
        self._plugins[plugin.name] = scope
        return scope

    async def uninstall(self, name: str) -> None:
        scope = self._plugins.pop(name, None)
        if scope is not None:
            await scope.dispose()

    async def dispose(self) -> None:
        failures: list[Exception] = []
        for name in reversed(tuple(self._plugins)):
            try:
                await self.uninstall(name)
            except Exception as exc:
                failures.append(exc)
        try:
            await self.root.dispose()
        except Exception as exc:
            failures.append(exc)
        if failures:
            raise ExceptionGroup("plugin runtime disposal failed", failures)

    def installed(self) -> tuple[str, ...]:
        return tuple(self._plugins)
