"""Scoped model-facing tool registry."""

from __future__ import annotations

import inspect
import json
import math
from asyncio import wait_for
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ..errors import ToolError
from ..llm.types import ToolSchema


@dataclass(frozen=True, slots=True)
class ToolContext:
    session_id: str
    cwd: str
    call_id: str | None = None


@dataclass(frozen=True, slots=True)
class ToolResult:
    text: str
    is_error: bool = False
    meta: dict[str, Any] | None = None


ToolExecutor = Callable[[dict[str, Any], ToolContext], ToolResult | Awaitable[ToolResult]]
ToolResultTransformer = Callable[
    [str, ToolContext, ToolResult], ToolResult | Awaitable[ToolResult]
]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    execute: ToolExecutor
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.timeout_seconds is not None and (
            self.timeout_seconds <= 0 or not math.isfinite(self.timeout_seconds)
        ):
            raise ValueError("tool timeout_seconds must be finite and positive")

    def schema(self) -> ToolSchema:
        return ToolSchema(self.name, self.description, self.parameters)


class ToolRegistry:
    """Register, expose, and execute tools with strict call-name lookup."""

    def __init__(self, result_transformer: ToolResultTransformer | None = None) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._result_transformer = result_transformer

    def register(self, definition: ToolDefinition) -> Callable[[], None]:
        if not definition.name or definition.name in self._tools:
            raise ToolError(f"duplicate or empty tool name: {definition.name!r}")
        self._tools[definition.name] = definition
        active = True

        def dispose() -> None:
            nonlocal active
            if active:
                active = False
                if self._tools.get(definition.name) is definition:
                    self._tools.pop(definition.name, None)

        return dispose

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def schemas(self) -> tuple[ToolSchema, ...]:
        return tuple(definition.schema() for definition in self._tools.values())

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    async def execute(self, name: str, raw_arguments: str, context: ToolContext) -> ToolResult:
        definition = self._tools.get(name)
        if definition is None:
            return ToolResult(f'tool "{name}" is not available', is_error=True)
        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as exc:
            return ToolResult(f"invalid JSON arguments for {name}: {exc.msg}", is_error=True)
        if not isinstance(arguments, dict):
            return ToolResult(f"arguments for {name} must be a JSON object", is_error=True)
        try:
            result = definition.execute(arguments, context)
            if inspect.isawaitable(result):
                if definition.timeout_seconds is None:
                    result = await result
                else:
                    result = await wait_for(result, definition.timeout_seconds)
            if not isinstance(result, ToolResult):
                raise TypeError("tool executor must return ToolResult")
            if self._result_transformer is not None:
                transformed = self._result_transformer(name, context, result)
                if inspect.isawaitable(transformed):
                    transformed = await transformed
                if not isinstance(transformed, ToolResult):
                    raise TypeError("tool result transformer must return ToolResult")
                result = transformed
            return result
        except TimeoutError:
            timeout = definition.timeout_seconds
            rendered = f"tool {name} timed out after {timeout:g}s"
            return ToolResult(rendered, is_error=True, meta={"code": "TOOL_TIMEOUT"})
        except Exception as exc:
            code = getattr(exc, "code", None)
            return ToolResult(
                f"{type(exc).__name__}: {exc}",
                is_error=True,
                meta={"code": code} if isinstance(code, str) else None,
            )
