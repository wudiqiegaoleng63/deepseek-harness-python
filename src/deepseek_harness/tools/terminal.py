"""Model-facing persistent terminal tools."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from ..jobs import JobHandle, JobOutcome, JobRegistry
from ..terminal import (
    ALLOWED_SIGNALS,
    TerminalSessionService,
    _SendOperation,
    bound_terminal_text,
)
from .policy import WorkspacePolicy
from .registry import ToolContext, ToolDefinition, ToolRegistry, ToolResult

MAX_RESULT_BYTES = 256 * 1024
TRUNCATED = "\n[output truncated]"


def install_terminal_tools(
    registry: ToolRegistry,
    terminals: TerminalSessionService,
    policy: WorkspacePolicy,
    *,
    jobs: JobRegistry | None = None,
    owner_session: str | None = None,
    max_result_bytes: int = MAX_RESULT_BYTES,
) -> list[Callable[[], None]]:
    """Install the six TS-compatible terminal tools for one session registry."""

    if isinstance(max_result_bytes, bool) or not isinstance(max_result_bytes, int):
        raise ValueError("max_result_bytes must be an integer")
    if max_result_bytes < 64:
        raise ValueError("max_result_bytes must be at least 64")

    def owner(context: ToolContext) -> str:
        if owner_session is not None and context.session_id != owner_session:
            raise PermissionError("terminal tool context does not belong to the registered session")
        return context.session_id

    def session_id(args: dict[str, Any]) -> str:
        value = args.get("sessionId")
        if not isinstance(value, str) or not value:
            raise ValueError("sessionId must be a non-empty string")
        return value

    async def terminal_open(args: dict[str, Any], context: ToolContext) -> ToolResult:
        policy.assert_shell_allowed()
        raw_type = args.get("type")
        if not isinstance(raw_type, str) or not raw_type:
            raise ValueError("type must be a non-empty string")
        name = args.get("name")
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError("name must be a non-empty string")
        raw_cwd = args.get("cwd")
        if raw_cwd is not None and not isinstance(raw_cwd, str):
            raise TypeError("cwd must be a string")
        cwd = policy.assert_readable(raw_cwd or context.cwd)
        if not cwd.is_dir():
            raise ValueError(f"terminal cwd is not a directory: {cwd}")
        result = await terminals.spawn(
            owner(context), type=raw_type, name=name, cwd=cwd
        )
        motd = result.get("motd") or "(no startup output)"
        label = result["sessionId"] if name is None else f"{result['sessionId']} ({name})"
        text = f"started terminal session {label} [type: {raw_type}]\n{motd}"
        return ToolResult(bound_terminal_text(text, max_result_bytes), meta=result)

    async def terminal_send(args: dict[str, Any], context: ToolContext) -> ToolResult:
        policy.assert_shell_allowed()
        target = session_id(args)
        text = args.get("text")
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        submit = args.get("submit", True)
        if not isinstance(submit, bool):
            raise TypeError("submit must be a boolean")
        background = args.get("run_in_background", False)
        if not isinstance(background, bool):
            raise TypeError("run_in_background must be a boolean")
        current_owner = owner(context)
        if background:
            if jobs is None:
                raise RuntimeError("background terminal sends require the job registry")
            cancelled = False

            async def start() -> JobHandle:
                operation = terminals.start_send(
                    current_owner, target, text=text, submit=submit
                )

                async def finish() -> JobOutcome:
                    nonlocal cancelled
                    try:
                        result = await asyncio.shield(operation.done)
                    except Exception as exc:
                        return JobOutcome("failed", str(exc))
                    detail = _send_detail(result)
                    return JobOutcome("killed" if cancelled else "completed", detail)

                def cancel(_reason: str | None = None) -> None:
                    nonlocal cancelled
                    cancelled = True
                    operation.cancel()

                return JobHandle(
                    cancel=cancel,
                    done=finish(),
                    read_output=lambda: _render_send_read(operation),
                )

            job_id = await jobs.start(
                kind="pty-send",
                label=f"{target}: {text or '(input)'}",
                owner_session=current_owner,
                starter=start,
            )
            value = {"kind": "background", "jobId": job_id}
            return ToolResult(f"started background job {job_id}", meta=value)

        operation = terminals.start_send(current_owner, target, text=text, submit=submit)
        result = await operation.join()
        value = {"kind": "foreground", **result}
        return ToolResult(_render_send(result, max_result_bytes), meta=value)

    async def terminal_read(args: dict[str, Any], context: ToolContext) -> ToolResult:
        policy.assert_shell_allowed()
        target = session_id(args)
        offset = args.get("offset", 0)
        count = args.get("count", 500)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer")
        value = terminals.read(owner(context), target, offset=offset, count=count)
        return ToolResult(_render_read(value, max_result_bytes), meta=value)

    async def terminal_list(_args: dict[str, Any], context: ToolContext) -> ToolResult:
        policy.assert_shell_allowed()
        value = terminals.list(owner(context))
        return ToolResult(_render_list(value, max_result_bytes), meta={"sessions": value})

    async def terminal_signal(args: dict[str, Any], context: ToolContext) -> ToolResult:
        policy.assert_shell_allowed()
        target = session_id(args)
        signal_name = args.get("signal")
        if not isinstance(signal_name, str) or signal_name not in ALLOWED_SIGNALS:
            raise ValueError(
                "signal must be one of SIGINT, SIGTERM, SIGKILL, SIGTSTP, or SIGHUP"
            )
        value = await terminals.signal(owner(context), target, signal_name)
        return ToolResult(
            f"delivered {signal_name} to foreground process group {value['targetPgid']}",
            meta=value,
        )

    async def terminal_close(args: dict[str, Any], context: ToolContext) -> ToolResult:
        policy.assert_shell_allowed()
        target = session_id(args)
        closed = await terminals.kill(owner(context), target)
        value = {"sessionId": target, "outcome": "closed" if closed else "already-closing"}
        text = (
            f"closed terminal session {target}"
            if closed
            else f"terminal session {target} was already closing"
        )
        return ToolResult(text, meta=value)

    definitions = [
        ToolDefinition(
            name="terminal_open",
            description=(
                "Create a persistent, owner-isolated terminal session from a registered "
                "backend type. Use this for shell or REPL state that must survive across calls."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": 'Usually "shell".'},
                    "name": {"type": "string"},
                    "cwd": {"type": "string"},
                },
                "required": ["type"],
                "additionalProperties": False,
            },
            execute=terminal_open,
        ),
        ToolDefinition(
            name="terminal_send",
            description=(
                "Send text to a persistent terminal. Enter is submitted by default; "
                "background mode returns a job id for job_output/job_kill."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "sessionId": {"type": "string"},
                    "text": {"type": "string"},
                    "submit": {"type": "boolean"},
                    "run_in_background": {"type": "boolean"},
                },
                "required": ["sessionId", "text"],
                "additionalProperties": False,
            },
            execute=terminal_send,
        ),
        ToolDefinition(
            name="terminal_read",
            description="Read a bounded page of retained output from a persistent terminal.",
            parameters={
                "type": "object",
                "properties": {
                    "sessionId": {"type": "string"},
                    "offset": {"type": "integer"},
                    "count": {"type": "integer"},
                },
                "required": ["sessionId"],
                "additionalProperties": False,
            },
            execute=terminal_read,
        ),
        ToolDefinition(
            name="terminal_list",
            description="List persistent terminal sessions owned by the current session.",
            parameters={"type": "object", "additionalProperties": False},
            execute=terminal_list,
        ),
        ToolDefinition(
            name="terminal_signal",
            description="Send an allowed signal to a persistent terminal foreground process group.",
            parameters={
                "type": "object",
                "properties": {
                    "sessionId": {"type": "string"},
                    "signal": {"type": "string", "enum": sorted(ALLOWED_SIGNALS)},
                },
                "required": ["sessionId", "signal"],
                "additionalProperties": False,
            },
            execute=terminal_signal,
        ),
        ToolDefinition(
            name="terminal_close",
            description="Close a persistent terminal and wait for its owned process tree to exit.",
            parameters={
                "type": "object",
                "properties": {"sessionId": {"type": "string"}},
                "required": ["sessionId"],
                "additionalProperties": False,
            },
            execute=terminal_close,
        ),
    ]
    return [registry.register(definition) for definition in definitions]


def _send_detail(result: dict[str, Any]) -> str:
    status = result.get("sessionStatus")
    if isinstance(status, dict) and status.get("kind") == "running":
        return f"wait: {result.get('waitReason', 'unknown')}"
    if isinstance(status, dict):
        exit_code = status.get("exitCode")
        detail = exit_code if exit_code is not None else status.get("signal")
        return f"session exited: {detail if detail is not None else 'unknown'}"
    return "terminal send completed"


def _render_send(result: dict[str, Any], max_bytes: int) -> str:
    output = result.get("viewport") or "(no new output)"
    status = result.get("sessionStatus", {})
    if isinstance(status, dict) and status.get("kind") == "running":
        rendered_status = "running"
    elif isinstance(status, dict):
        rendered_status = (
            f"exited code={status.get('exitCode')} signal={status.get('signal')}"
        )
    else:
        rendered_status = "unknown"
    text = f"{output}\n[wait: {result.get('waitReason')}]\n[session: {rendered_status}]"
    if result.get("truncated") is True and TRUNCATED not in text:
        text += TRUNCATED
    return bound_terminal_text(text, max_bytes)


def _render_send_read(operation: _SendOperation) -> str:
    text, truncated = operation.read_output()
    if truncated:
        separator = "" if not text or text.endswith("\n") else "\n"
        text += f"{separator}{TRUNCATED.lstrip()}"
    return text


def _render_read(result: dict[str, Any], max_bytes: int) -> str:
    text = result.get("text") or "(no retained output)"
    suffix = (
        f"\n[lines: {result.get('lineBegin')}-{result.get('lineEnd')} "
        f"of {result.get('totalLines')}]"
    )
    if result.get("truncated") is True:
        suffix += TRUNCATED
    return bound_terminal_text(f"{text}{suffix}", max_bytes)


def _render_list(sessions: list[dict[str, Any]], max_bytes: int) -> str:
    if not sessions:
        return "(no terminal sessions)"
    rows: list[str] = []
    for session in sessions:
        name = f" ({session['name']})" if "name" in session else ""
        status = session.get("status", {})
        if isinstance(status, dict) and status.get("kind") == "running":
            rendered_status = "running"
        elif isinstance(status, dict):
            rendered_status = (
                f"exited code={status.get('exitCode')} signal={status.get('signal')}"
            )
        else:
            rendered_status = "unknown"
        pid = f" pid={session['pid']}" if "pid" in session else ""
        rows.append(f"{session['sessionId']}{name} [{session['type']}] {rendered_status}{pid}")
    return bound_terminal_text("\n".join(rows), max_bytes)


__all__ = ["MAX_RESULT_BYTES", "install_terminal_tools"]
