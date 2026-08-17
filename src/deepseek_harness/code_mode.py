"""Python Code Mode runtime and the model-facing ``run_code`` tool.

The TypeScript Web/headless bundles expose a language-aware ``run_code`` seam.
This first Python backend executes an async function body with a deliberately
small builtin surface, bridges calls through the existing ToolRegistry, and
keeps the result bounded.  It is a containment/convenience boundary, not an
OS security boundary; a future subprocess backend can implement the same
``CodeRuntime`` contract without changing the tool schema.
"""

from __future__ import annotations

import ast
import asyncio
import json
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from .models import JsonValue, assert_json_value
from .tools.registry import ToolContext, ToolDefinition, ToolRegistry, ToolResult

CodeFailureKind = Literal["exception", "timeout", "output-limit", "aborted"]


@dataclass(frozen=True, slots=True)
class CodeRunFailure:
    kind: CodeFailureKind
    message: str

    def to_dict(self) -> dict[str, JsonValue]:
        return {"kind": self.kind, "message": self.message}


@dataclass(frozen=True, slots=True)
class CodeRunResult:
    logs: tuple[str, ...]
    value: JsonValue | None = None
    error: CodeRunFailure | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {"logs": list(self.logs)}
        if self.value is not None:
            result["value"] = self.value
        if self.error is not None:
            result["error"] = self.error.to_dict()
        return result


@dataclass(frozen=True, slots=True)
class CodeRuntimeConfig:
    """Validated execution and result budgets."""

    max_wall_seconds: float = 600.0
    max_output_bytes: int = 67_108_864

    def __post_init__(self) -> None:
        if self.max_wall_seconds <= 0:
            raise ValueError("max_wall_seconds must be positive")
        if (
            isinstance(self.max_output_bytes, bool)
            or not isinstance(self.max_output_bytes, int)
            or self.max_output_bytes < 4
        ):
            raise ValueError("max_output_bytes must be at least 4")


class ToolCallError(RuntimeError):
    """Raised inside a code program when a nested tool returns an error."""

    def __init__(self, tool_name: str, message: str) -> None:
        super().__init__(message)
        self.tool_name = tool_name

    @property
    def toolName(self) -> str:  # noqa: N802 - matches the Python SDK contract from DSH.
        return self.tool_name


class _ToolCallable:
    def __init__(
        self,
        name: str,
        registry: ToolRegistry,
        context: ToolContext,
        next_call_id: Callable[[], str],
    ) -> None:
        self.name = name
        self.registry = registry
        self.context = context
        self.next_call_id = next_call_id

    async def __call__(self, arguments: dict[str, Any]) -> str:
        if not isinstance(arguments, dict):
            raise ToolCallError(self.name, "tool arguments must be a JSON object")
        try:
            encoded = json.dumps(arguments, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ToolCallError(self.name, f"tool arguments must be lossless JSON: {exc}") from exc
        result = await self.registry.execute(
            self.name,
            encoded,
            ToolContext(
                self.context.session_id,
                self.context.cwd,
                self.next_call_id(),
            ),
        )
        if result.is_error:
            raise ToolCallError(self.name, result.text)
        return result.text


class _ToolProxy:
    def __init__(self, registry: ToolRegistry, context: ToolContext) -> None:
        self.registry = registry
        self.context = context
        self._counter = 0

    def _next_call_id(self) -> str:
        self._counter += 1
        parent = self.context.call_id or "run-code"
        return f"{parent}:code:{self._counter}"

    def __getitem__(self, name: str) -> _ToolCallable:
        if not isinstance(name, str) or not name:
            raise ToolCallError(str(name), "tool name must be a non-empty string")
        if name == "run_code":
            raise ToolCallError(name, "run_code cannot call itself")
        return _ToolCallable(name, self.registry, self.context, self._next_call_id)

    def __getattr__(self, name: str) -> _ToolCallable:
        if name.startswith("_"):
            raise AttributeError(name)
        return self[name]


class CodeRuntime:
    """Execute one Python async function body against a ToolRegistry."""

    def __init__(self, config: CodeRuntimeConfig | None = None) -> None:
        self.config = config or CodeRuntimeConfig()

    async def run(
        self,
        program: str,
        *,
        registry: ToolRegistry,
        context: ToolContext,
    ) -> CodeRunResult:
        if not isinstance(program, str) or not program.strip():
            return CodeRunResult((), error=CodeRunFailure("exception", "code must be non-empty"))
        logs: list[str] = []

        def capture_print(*values: object, sep: str = " ", end: str = "\n") -> None:
            if not isinstance(sep, str) or not isinstance(end, str):
                raise TypeError("print sep and end must be strings")
            rendered = sep.join(str(value) for value in values)
            if end.endswith("\n"):
                rendered += end.rstrip("\n")
            else:
                rendered += end
            logs.append(rendered)

        async def execute() -> JsonValue | None:
            proxy = _ToolProxy(registry, context)
            source = "async def __dsh_main__():\n" + textwrap.indent(program, "    ") + "\n"
            try:
                tree = ast.parse(source, filename="<run_code>", mode="exec")
                self._validate_tree(tree)
                namespace: dict[str, Any] = {
                    "__builtins__": self._safe_builtins(capture_print),
                    "asyncio": _asyncio_facade(),
                    "print": capture_print,
                    "tools": proxy,
                    "ToolCallError": ToolCallError,
                }
                exec(compile(tree, "<run_code>", "exec"), namespace, namespace)
                value = await namespace["__dsh_main__"]()
            except ToolCallError:
                raise
            except Exception as exc:
                raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc
            assert_json_value(value)
            return value

        try:
            value = await asyncio.wait_for(execute(), self.config.max_wall_seconds)
        except TimeoutError:
            return self._bounded_result(
                logs,
                error=CodeRunFailure(
                    "timeout", f"code execution timed out after {self.config.max_wall_seconds:g}s"
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return self._bounded_result(
                logs,
                error=CodeRunFailure("exception", str(exc) or "code execution failed"),
            )
        return self._bounded_result(logs, value=value)

    def _bounded_result(
        self,
        logs: list[str],
        *,
        value: JsonValue | None = None,
        error: CodeRunFailure | None = None,
    ) -> CodeRunResult:
        candidate = CodeRunResult(tuple(logs), value=value, error=error)
        encoded = json.dumps(candidate.to_dict(), ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) <= self.config.max_output_bytes:
            return candidate
        message = f"outer output exceeded {self.config.max_output_bytes} bytes"
        retained: list[str] = []
        for log in logs:
            trial = CodeRunResult(
                tuple(retained + [log]), error=CodeRunFailure("output-limit", message)
            )
            if (
                len(json.dumps(trial.to_dict(), ensure_ascii=False).encode("utf-8"))
                > self.config.max_output_bytes
            ):
                break
            retained.append(log)
        return CodeRunResult(
            tuple(retained),
            error=CodeRunFailure("output-limit", message),
        )

    @staticmethod
    def _validate_tree(tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise RuntimeError("imports are not available in run_code")
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                raise RuntimeError(f"dunder name {node.id!r} is not available in run_code")
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise RuntimeError(f"dunder attribute {node.attr!r} is not available in run_code")

    @staticmethod
    def _safe_builtins(capture_print: Callable[..., None]) -> dict[str, Any]:
        names = (
            "all",
            "any",
            "bool",
            "dict",
            "enumerate",
            "Exception",
            "float",
            "int",
            "isinstance",
            "len",
            "list",
            "max",
            "min",
            "range",
            "repr",
            "set",
            "sorted",
            "str",
            "sum",
            "tuple",
            "TypeError",
            "ValueError",
            "zip",
        )
        import builtins

        return {name: getattr(builtins, name) for name in names} | {"print": capture_print}


def _asyncio_facade() -> Any:
    return type(
        "AsyncioFacade",
        (),
        {"gather": staticmethod(asyncio.gather), "sleep": staticmethod(asyncio.sleep)},
    )()


def render_code_sdk(registry: ToolRegistry) -> str:
    """Render concise Python Code Mode guidance for the current tool catalog."""

    lines = [
        "## Writing code for run_code",
        "",
        "run_code executes the body of an async Python function. Top-level await and return work.",
        "Call tools with `await tools.name(args)`; failed calls raise `ToolCallError`.",
        "Use print(...) and/or return a lossless JSON value. "
        "Intermediate tool results stay inside the run.",
        "",
        "Available tools:",
    ]
    for schema in sorted(registry.all_schemas(), key=lambda item: item.name):
        if schema.name == "run_code":
            continue
        lines.append(f"- `{schema.name}`: {schema.description}")
    return "\n".join(lines)


def install_code_tool(
    registry: ToolRegistry,
    *,
    config: CodeRuntimeConfig | None = None,
) -> Callable[[], None]:
    """Register the additive Python ``run_code`` model tool."""

    runtime = CodeRuntime(config)

    async def execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
        code = args.get("code")
        description = args.get("description")
        if not isinstance(code, str) or not code.strip():
            return ToolResult("code must be a non-empty string", is_error=True)
        if not isinstance(description, str) or not description.strip():
            return ToolResult("description must be a non-empty string", is_error=True)
        result = await runtime.run(code, registry=registry, context=context)
        text_parts = list(result.logs)
        if result.value is not None:
            text_parts.append(
                result.value
                if isinstance(result.value, str)
                else json.dumps(result.value, ensure_ascii=False, indent=2)
            )
        if result.error is not None:
            text_parts.append(f"run_code failed [{result.error.kind}]: {result.error.message}")
        text = "\n".join(text_parts) or "(run_code completed with no output)"
        return ToolResult(
            text,
            is_error=result.error is not None,
            meta={
                "code": "CODE_RUN_FAILED",
                "kind": result.error.kind,
            }
            if result.error is not None
            else None,
        )

    return registry.register(
        ToolDefinition(
            name="run_code",
            description=(
                "Execute a Python program against the available tools. The code is the body "
                "of an async function; top-level await and return work."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The body of an async Python function.",
                    },
                    "description": {
                        "type": "string",
                        "description": "A short description of what the program does.",
                    },
                },
                "required": ["code", "description"],
                "additionalProperties": False,
            },
            execute=execute,
        )
    )


__all__ = [
    "CodeFailureKind",
    "CodeRunFailure",
    "CodeRunResult",
    "CodeRuntime",
    "CodeRuntimeConfig",
    "ToolCallError",
    "install_code_tool",
    "render_code_sdk",
]
