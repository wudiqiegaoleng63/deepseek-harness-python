"""Small local capability set used by the first native runtime slice."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .policy import WorkspacePolicy
from .registry import ToolDefinition, ToolRegistry, ToolResult


def install_builtin_tools(
    registry: ToolRegistry,
    policy: WorkspacePolicy,
    *,
    enable_shell: bool = False,
) -> list[Callable[[], None]]:
    """Install read/write/list tools and optionally an explicitly unconfined shell."""

    disposers = [
        registry.register(
            ToolDefinition(
                name="read_file",
                description="Read a UTF-8 text file inside the workspace.",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "max_bytes": {"type": "integer"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
                execute=lambda args, ctx: _read_file(args, policy),
            )
        ),
        registry.register(
            ToolDefinition(
                name="write_file",
                description="Write UTF-8 text to a file inside the workspace.",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
                execute=lambda args, ctx: _write_file(args, policy),
            )
        ),
        registry.register(
            ToolDefinition(
                name="list_files",
                description="List files and directories below a workspace path.",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "max_entries": {"type": "integer"}},
                    "additionalProperties": False,
                },
                execute=lambda args, ctx: _list_files(args, policy),
            )
        ),
    ]
    if enable_shell:
        disposers.append(
            registry.register(
                ToolDefinition(
                    name="run_bash",
                    description="Run a bash command. Only available in danger-full-access mode.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "timeout_seconds": {"type": "number"},
                        },
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                    execute=lambda args, ctx: _run_bash(args, policy),
                )
            )
        )
    return disposers


async def _read_file(args: dict[str, Any], policy: WorkspacePolicy) -> ToolResult:
    path = policy.assert_readable(str(args.get("path", "")))
    max_bytes = int(args.get("max_bytes", 32_000))
    if max_bytes <= 0:
        return ToolResult("max_bytes must be positive", is_error=True)
    data = await asyncio.to_thread(path.read_bytes)
    truncated = len(data) > max_bytes
    text = data[:max_bytes].decode("utf-8", errors="replace")
    if truncated:
        text += f"\n[truncated after {max_bytes} bytes]"
    return ToolResult(text)


async def _write_file(args: dict[str, Any], policy: WorkspacePolicy) -> ToolResult:
    path = policy.assert_writable(str(args.get("path", "")))
    content = args.get("content")
    if not isinstance(content, str):
        return ToolResult("content must be a string", is_error=True)
    await asyncio.to_thread(_write_text, path, content)
    return ToolResult(f"wrote {len(content.encode('utf-8'))} bytes to {path}")


async def _list_files(args: dict[str, Any], policy: WorkspacePolicy) -> ToolResult:
    path = policy.assert_readable(str(args.get("path", ".")))
    max_entries = int(args.get("max_entries", 200))
    if max_entries <= 0:
        return ToolResult("max_entries must be positive", is_error=True)
    entries = await asyncio.to_thread(lambda: sorted(path.iterdir(), key=lambda item: item.name))
    visible = entries[:max_entries]
    result = "\n".join(
        f"{item.relative_to(policy.root)}{'/' if item.is_dir() else ''}" for item in visible
    )
    if len(entries) > max_entries:
        result += f"\n[truncated after {max_entries} entries]"
    return ToolResult(result)


async def _run_bash(args: dict[str, Any], policy: WorkspacePolicy) -> ToolResult:
    policy.assert_shell_allowed()
    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        return ToolResult("command must be a non-empty string", is_error=True)
    timeout = float(args.get("timeout_seconds", 60))
    if timeout <= 0:
        return ToolResult("timeout_seconds must be positive", is_error=True)
    process = await asyncio.create_subprocess_exec(
        "/bin/bash",
        "-lc",
        command,
        cwd=str(policy.root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        return ToolResult(f"command timed out after {timeout:g}s", is_error=True)
    output = (stdout + stderr).decode("utf-8", errors="replace")
    if process.returncode:
        return ToolResult(f"exit code {process.returncode}\n{output}", is_error=True)
    return ToolResult(output or "command completed successfully")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
