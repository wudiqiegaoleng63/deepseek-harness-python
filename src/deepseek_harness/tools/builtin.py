"""Small local capability set used by the first native runtime slice."""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..jobs import JobRegistry, start_bash_process
from .policy import PermissionMode, WorkspacePolicy
from .registry import ToolDefinition, ToolRegistry, ToolResult


def install_builtin_tools(
    registry: ToolRegistry,
    policy: WorkspacePolicy,
    *,
    enable_shell: bool = False,
    jobs: JobRegistry | None = None,
) -> list[Callable[[], None]]:
    """Install file tools and the shell/job tools for a local agent.

    ``run_bash`` is retained as a compatibility alias for the early Python
    runtime.  ``bash`` is the canonical DSH tool name and adds the background
    job controls used by the TS frontend.
    """

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
        for name in ("bash", "run_bash"):
            disposers.append(
                registry.register(
                    ToolDefinition(
                        name=name,
                        description=(
                            "Execute a bash command and return stdout/stderr. "
                            "Set run_in_background to true for long-running work; "
                            "read it with job_output and stop it with job_kill."
                        ),
                        parameters={
                            "type": "object",
                            "properties": {
                                "command": {"type": "string"},
                                "description": {"type": "string"},
                                "timeoutMs": {"type": "number"},
                                "timeout_seconds": {"type": "number"},
                                "workdir": {"type": "string"},
                                "run_in_background": {"type": "boolean"},
                            },
                            "required": ["command"],
                            "additionalProperties": False,
                        },
                        execute=lambda args, ctx: _run_bash(args, policy, jobs, ctx.session_id),
                    )
                )
            )
        disposers.extend(_install_job_tools(registry, jobs))
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

    def display_path(item: Path) -> str:
        if policy.mode is PermissionMode.DANGER_FULL_ACCESS:
            try:
                shown = item.relative_to(policy.root)
            except ValueError:
                shown = item
        else:
            shown = item.relative_to(policy.root)
        return f"{shown}{'/' if item.is_dir() else ''}"

    result = "\n".join(display_path(item) for item in visible)
    if len(entries) > max_entries:
        result += f"\n[truncated after {max_entries} entries]"
    return ToolResult(result)


async def _run_bash(
    args: dict[str, Any],
    policy: WorkspacePolicy,
    jobs: JobRegistry | None,
    owner_session: str,
) -> ToolResult:
    policy.assert_shell_allowed()
    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        return ToolResult("command must be a non-empty string", is_error=True)
    timeout_value = args.get("timeoutMs", args.get("timeout_seconds", 60))
    timeout = float(timeout_value) if isinstance(timeout_value, (int, float)) else 60.0
    if timeout <= 0:
        return ToolResult("timeout_seconds must be positive", is_error=True)
    raw_workdir = args.get("workdir", ".")
    if not isinstance(raw_workdir, str):
        return ToolResult("workdir must be a string", is_error=True)
    workdir = policy.assert_readable(raw_workdir)
    if args.get("run_in_background") is True:
        if jobs is None:
            return ToolResult(
                "background jobs unavailable; load the jobs capability and job tools",
                is_error=True,
            )
        try:
            job_id = await jobs.start(
                kind="bash",
                label=command,
                owner_session=owner_session,
                starter=lambda: start_bash_process(command, cwd=workdir),
            )
        except Exception as exc:
            return ToolResult(f"could not start background command: {exc}", is_error=True)
        return ToolResult(
            f"started background job {job_id}\nRead output with job_output; stop it with job_kill."
        )
    executable = shutil.which("bash") or shutil.which("sh")
    if executable is None:
        return ToolResult("a bash-compatible shell is not installed", is_error=True)
    process = await asyncio.create_subprocess_exec(
        executable,
        "-lc",
        command,
        cwd=str(workdir),
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


def _install_job_tools(
    registry: ToolRegistry,
    jobs: JobRegistry | None,
) -> list[Callable[[], None]]:
    """Install the model-facing job controller tools."""

    def require_jobs() -> JobRegistry:
        if jobs is None:
            raise RuntimeError("background jobs are not configured")
        return jobs

    disposers = [
        registry.register(
            ToolDefinition(
                name="job_output",
                description=(
                    "Read incremental output from a background job. Set wait to true "
                    "only when blocked on the result."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string"},
                        "wait": {"type": "boolean"},
                        "timeout_ms": {"type": "number"},
                    },
                    "required": ["job_id"],
                    "additionalProperties": False,
                },
                execute=lambda args, ctx: _job_output(args, require_jobs(), ctx.session_id),
            )
        ),
        registry.register(
            ToolDefinition(
                name="job_list",
                description="List the current session's background jobs and statuses.",
                parameters={"type": "object", "additionalProperties": False},
                execute=lambda args, ctx: _job_list(require_jobs(), ctx.session_id),
            )
        ),
        registry.register(
            ToolDefinition(
                name="job_kill",
                description="Request cancellation of a running background job.",
                parameters={
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["job_id"],
                    "additionalProperties": False,
                },
                execute=lambda args, ctx: _job_kill(args, require_jobs(), ctx.session_id),
            )
        ),
    ]
    return disposers


async def _job_output(args: dict[str, Any], jobs: JobRegistry, session_id: str) -> ToolResult:
    job_id = args.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        return ToolResult("job_id must be a non-empty string", is_error=True)
    if args.get("wait") is True:
        timeout = args.get("timeout_ms", 30_000)
        if not isinstance(timeout, (int, float)):
            return ToolResult("timeout_ms must be a number", is_error=True)
        try:
            await jobs.wait(job_id, min(float(timeout), 600_000), session_id)
        except Exception as exc:
            return ToolResult(f"could not wait for job: {exc}", is_error=True)
    try:
        result = jobs.read(job_id, session_id)
    except Exception as exc:
        return ToolResult(f"could not read job: {exc}", is_error=True)
    text = result.text or "(no new output)"
    if result.snapshot.detail:
        text += f"\n[{result.snapshot.detail}]"
    text += f"\n[status: {result.snapshot.status}]"
    return ToolResult(text)


async def _job_list(jobs: JobRegistry, session_id: str) -> ToolResult:
    snapshots = jobs.list(session_id)
    if not snapshots:
        return ToolResult("(no background jobs)")
    return ToolResult(
        "\n".join(f"{job.id} [{job.kind}] {job.status} — {job.label}" for job in snapshots)
    )


async def _job_kill(args: dict[str, Any], jobs: JobRegistry, session_id: str) -> ToolResult:
    job_id = args.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        return ToolResult("job_id must be a non-empty string", is_error=True)
    reason = args.get("reason")
    if reason is not None and not isinstance(reason, str):
        return ToolResult("reason must be a string", is_error=True)
    try:
        outcome = jobs.kill(job_id, session_id, reason)
        snapshot = jobs.get(job_id, session_id)
    except Exception as exc:
        return ToolResult(f"could not kill job: {exc}", is_error=True)
    if outcome == "already-finished":
        return ToolResult(f"job {job_id} had already finished [{snapshot.status}]")
    return ToolResult(f"requested cancellation of job {job_id} [{snapshot.status}]")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
