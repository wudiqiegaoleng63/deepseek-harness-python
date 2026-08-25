"""Small local capability set used by the first native runtime slice."""

from __future__ import annotations

import asyncio
import fnmatch
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..jobs import JobRegistry, start_bash_process
from ..sandbox import SandboxExecutionPolicy, SandboxProvider
from .policy import PermissionMode, WorkspacePolicy
from .registry import ToolDefinition, ToolRegistry, ToolResult


def install_builtin_tools(
    registry: ToolRegistry,
    policy: WorkspacePolicy,
    *,
    enable_shell: bool = False,
    jobs: JobRegistry | None = None,
    sandbox: SandboxProvider | None = None,
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
    disposers.extend(_install_filesystem_tools(registry, policy))
    if enable_shell:
        disposers.extend(install_shell_tools(registry, policy, jobs=jobs, sandbox=sandbox))
    return disposers


def install_shell_tools(
    registry: ToolRegistry,
    policy: WorkspacePolicy,
    *,
    jobs: JobRegistry | None = None,
    sandbox: SandboxProvider | None = None,
) -> list[Callable[[], None]]:
    """Install shell and background-job tools for a live permission mode.

    The host can add/remove this capability when a session switches between
    permission presets.  The executor still checks ``WorkspacePolicy`` on
    every call, so registration is not a security boundary.  In
    ``workspace-write`` mode the sandbox provider must be available; commands
    are then wrapped with it exactly like the TS runtime.
    """

    disposers: list[Callable[[], None]] = []
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
                    execute=lambda args, ctx: _run_bash(
                        args, policy, jobs, ctx.session_id, sandbox
                    ),
                )
            )
        )
    disposers.extend(_install_job_tools(registry, jobs))
    return disposers


def _install_filesystem_tools(
    registry: ToolRegistry,
    policy: WorkspacePolicy,
) -> list[Callable[[], None]]:
    """Install the canonical DSH read/write/edit/search tool names."""

    return [
        registry.register(
            ToolDefinition(
                name="read",
                description="Read a UTF-8 text file with line-numbered content.",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "offset": {"type": "integer"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["file_path"],
                    "additionalProperties": False,
                },
                execute=lambda args, ctx: _read_window(args, policy),
            )
        ),
        registry.register(
            ToolDefinition(
                name="write",
                description="Create or fully replace a UTF-8 text file.",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["file_path", "content"],
                    "additionalProperties": False,
                },
                execute=lambda args, ctx: _write_canonical(args, policy),
            )
        ),
        registry.register(
            ToolDefinition(
                name="edit",
                description=(
                    "Replace an exact string in a UTF-8 file; use replace_all for repeated matches."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                        "replace_all": {"type": "boolean"},
                    },
                    "required": ["file_path", "old_string", "new_string"],
                    "additionalProperties": False,
                },
                execute=lambda args, ctx: _edit_file(args, policy),
            )
        ),
        registry.register(
            ToolDefinition(
                name="glob",
                description="Find files by glob pattern, excluding VCS metadata directories.",
                parameters={
                    "type": "object",
                    "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
                execute=lambda args, ctx: _glob_files(args, policy),
            )
        ),
        registry.register(
            ToolDefinition(
                name="grep",
                description="Search file contents with a regular expression.",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                        "include": {"type": "string"},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
                execute=lambda args, ctx: _grep_files(args, policy),
            )
        ),
        registry.register(
            ToolDefinition(
                name="str_replace_editor",
                description=(
                    "View, create, replace or insert text in a file using "
                    "Claude-compatible editor commands."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "enum": ["view", "create", "str_replace", "insert"],
                        },
                        "path": {"type": "string"},
                        "file_text": {"type": "string"},
                        "insert_line": {"type": "integer"},
                        "new_str": {"type": "string"},
                        "old_str": {"type": "string"},
                        "view_range": {"type": "array"},
                    },
                    "required": ["command", "path"],
                    "additionalProperties": False,
                },
                execute=lambda args, ctx: _str_replace_editor(args, policy),
            )
        ),
    ]


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
    sandbox: SandboxProvider | None = None,
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
    executable = shutil.which("bash") or shutil.which("sh")
    if executable is None:
        return ToolResult("a bash-compatible shell is not installed", is_error=True)
    argv: list[str] = [executable, "-lc", command]
    if policy.mode is PermissionMode.WORKSPACE_WRITE:
        if sandbox is None or not sandbox.is_available():
            return ToolResult(
                "shell in workspace-write requires the bubblewrap sandbox; "
                "install bwrap or switch to danger-full-access",
                is_error=True,
            )
        argv = sandbox.confine(
            argv,
            SandboxExecutionPolicy("workspace-write", str(policy.root)),
        )
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
                starter=lambda: start_bash_process(command, cwd=workdir, argv=argv),
            )
        except Exception as exc:
            return ToolResult(f"could not start background command: {exc}", is_error=True)
        return ToolResult(
            f"started background job {job_id}\nRead output with job_output; stop it with job_kill."
        )
    process = await asyncio.create_subprocess_exec(
        *argv,
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


async def _read_window(args: dict[str, Any], policy: WorkspacePolicy) -> ToolResult:
    file_path = args.get("file_path")
    if not isinstance(file_path, str) or not file_path.strip():
        return ToolResult("file_path must be a non-empty string", is_error=True)
    offset = args.get("offset", 1)
    limit = args.get("limit", 2_000)
    if not isinstance(offset, int) or offset < 1 or not isinstance(limit, int) or limit < 1:
        return ToolResult("offset and limit must be positive integers", is_error=True)
    path = policy.assert_readable(file_path)
    try:
        text = await asyncio.to_thread(path.read_text, encoding="utf-8")
    except UnicodeDecodeError:
        return ToolResult(f"file is not valid UTF-8: {path}", is_error=True)
    except OSError as exc:
        return ToolResult(str(exc), is_error=True)
    lines = text.splitlines()
    selected = lines[offset - 1 : offset - 1 + limit]
    numbered = "\n".join(f"{number}: {line}" for number, line in enumerate(selected, offset))
    if not numbered:
        numbered = "(no lines in requested range)"
    return ToolResult(
        f"<path>{path}</path>\n<type>file</type>\n<content>\n{numbered}\n</content>\n"
        f"[lines {offset}-{min(len(lines), offset + len(selected) - 1)} of {len(lines)}]"
    )


async def _write_canonical(args: dict[str, Any], policy: WorkspacePolicy) -> ToolResult:
    file_path = args.get("file_path")
    content = args.get("content")
    if not isinstance(file_path, str) or not file_path.strip():
        return ToolResult("file_path must be a non-empty string", is_error=True)
    if not isinstance(content, str):
        return ToolResult("content must be a string", is_error=True)
    path = policy.assert_writable(file_path)
    existed = await asyncio.to_thread(path.exists)
    await asyncio.to_thread(_write_text, path, content)
    operation = "update" if existed else "create"
    return ToolResult(
        f"<path>{path}</path>\n<type>file</type>\n<content>\n{operation.title()}d file\n</content>"
    )


async def _edit_file(args: dict[str, Any], policy: WorkspacePolicy) -> ToolResult:
    file_path = args.get("file_path")
    old_string = args.get("old_string")
    new_string = args.get("new_string")
    replace_all = args.get("replace_all", False)
    if not isinstance(file_path, str) or not file_path.strip():
        return ToolResult("file_path must be a non-empty string", is_error=True)
    if not isinstance(old_string, str) or not old_string:
        return ToolResult("old_string must be a non-empty string", is_error=True)
    if not isinstance(new_string, str) or old_string == new_string:
        return ToolResult("new_string must be a different string", is_error=True)
    if not isinstance(replace_all, bool):
        return ToolResult("replace_all must be a boolean", is_error=True)
    path = policy.assert_writable(file_path)
    try:
        original = await asyncio.to_thread(path.read_text, encoding="utf-8")
    except OSError as exc:
        return ToolResult(str(exc), is_error=True)
    matches = original.count(old_string)
    if matches == 0:
        return ToolResult("old_string was not found in the file", is_error=True)
    if matches > 1 and not replace_all:
        return ToolResult(
            f"old_string appeared {matches} times; provide a more specific string "
            "or set replace_all=true",
            is_error=True,
        )
    updated = original.replace(old_string, new_string, -1 if replace_all else 1)
    await asyncio.to_thread(_write_text, path, updated)
    return ToolResult(f"The file {path} has been updated successfully ({matches} replacement(s)).")


async def _glob_files(args: dict[str, Any], policy: WorkspacePolicy) -> ToolResult:
    pattern = args.get("pattern")
    raw_root = args.get("path", ".")
    if not isinstance(pattern, str) or not pattern.strip():
        return ToolResult("pattern must be a non-empty string", is_error=True)
    if not isinstance(raw_root, str) or not raw_root.strip():
        return ToolResult("path must be a non-empty string", is_error=True)
    root = policy.assert_readable(raw_root)
    try:
        paths = await asyncio.to_thread(_collect_glob, root, pattern)
    except OSError as exc:
        return ToolResult(str(exc), is_error=True)
    return ToolResult("\n".join(paths) if paths else "No files found")


def _collect_glob(root: Path, pattern: str) -> list[str]:
    result: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or any(
            part in {".git", ".svn", ".hg", ".bzr", ".jj", ".sl"} for part in path.parts
        ):
            continue
        relative = path.relative_to(root).as_posix()
        matched = (
            fnmatch.fnmatch(path.name, pattern)
            if "/" not in pattern
            else fnmatch.fnmatch(relative, pattern)
        )
        if matched:
            result.append(relative)
    return sorted(result)


async def _grep_files(args: dict[str, Any], policy: WorkspacePolicy) -> ToolResult:
    pattern = args.get("pattern")
    raw_root = args.get("path", ".")
    include = args.get("include")
    max_results = args.get("max_results", 200)
    if not isinstance(pattern, str) or not pattern:
        return ToolResult("pattern must be a non-empty string", is_error=True)
    if not isinstance(raw_root, str) or not raw_root.strip():
        return ToolResult("path must be a non-empty string", is_error=True)
    if include is not None and (
        not isinstance(include, str) or not include.strip() or include.startswith("!")
    ):
        return ToolResult("include must be one positive glob", is_error=True)
    if not isinstance(max_results, int) or max_results < 1:
        return ToolResult("max_results must be a positive integer", is_error=True)
    try:
        expression = re.compile(pattern)
    except re.error as exc:
        return ToolResult(f"invalid regular expression: {exc}", is_error=True)
    root = policy.assert_readable(raw_root)
    try:
        matches = await asyncio.to_thread(_collect_grep, root, expression, include, max_results)
    except OSError as exc:
        return ToolResult(str(exc), is_error=True)
    if not matches:
        return ToolResult("No matches found")
    suffix = f"\n[showing {len(matches)} matches]"
    return ToolResult("\n".join(matches) + suffix)


def _collect_grep(
    root: Path,
    expression: re.Pattern[str],
    include: str | None,
    max_results: int,
) -> list[str]:
    candidates = [root] if root.is_file() else list(root.rglob("*"))
    matches: list[str] = []
    for path in candidates:
        if len(matches) >= max_results or not path.is_file():
            continue
        if any(part in {".git", ".svn", ".hg", ".bzr", ".jj", ".sl"} for part in path.parts):
            continue
        if include is not None and not fnmatch.fnmatch(path.name, include):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        display = path.name if root.is_file() else path.relative_to(root).as_posix()
        for line_number, line in enumerate(text.splitlines(), 1):
            if expression.search(line):
                matches.append(f"{display}:{line_number}:{line}")
                if len(matches) >= max_results:
                    break
    return matches


async def _str_replace_editor(args: dict[str, Any], policy: WorkspacePolicy) -> ToolResult:
    command = args.get("command")
    raw_path = args.get("path")
    if command not in {"view", "create", "str_replace", "insert"}:
        return ToolResult("command must be view, create, str_replace, or insert", is_error=True)
    if not isinstance(raw_path, str) or not raw_path.strip():
        return ToolResult("path must be a non-empty string", is_error=True)
    if command == "view":
        path = policy.assert_readable(raw_path)
        if path.is_dir():
            entries = await asyncio.to_thread(lambda: sorted(item.name for item in path.iterdir()))
            return ToolResult("\n".join(entries) or "(empty directory)")
        view_range = args.get("view_range")
        if view_range is not None and (
            not isinstance(view_range, list)
            or len(view_range) not in {2}
            or not all(isinstance(item, int) for item in view_range)
        ):
            return ToolResult("view_range must be [start_line, end_line]", is_error=True)
        read_args: dict[str, Any] = {"file_path": raw_path}
        if isinstance(view_range, list):
            read_args["offset"] = max(1, view_range[0])
            if view_range[1] != -1:
                read_args["limit"] = max(1, view_range[1] - view_range[0] + 1)
        return await _read_window(read_args, policy)
    if command == "create":
        return await _write_canonical(
            {"file_path": raw_path, "content": args.get("file_text", "")}, policy
        )
    if command == "str_replace":
        return await _edit_file(
            {
                "file_path": raw_path,
                "old_string": args.get("old_str"),
                "new_string": args.get("new_str", ""),
            },
            policy,
        )
    insert_line = args.get("insert_line")
    new_str = args.get("new_str")
    if not isinstance(insert_line, int) or insert_line < 0 or not isinstance(new_str, str):
        return ToolResult("insert requires a non-negative insert_line and new_str", is_error=True)
    path = policy.assert_writable(raw_path)
    try:
        original = await asyncio.to_thread(path.read_text, encoding="utf-8")
    except OSError as exc:
        return ToolResult(str(exc), is_error=True)
    lines = original.splitlines(keepends=True)
    index = min(insert_line, len(lines))
    lines.insert(index, new_str if new_str.endswith("\n") else f"{new_str}\n")
    await asyncio.to_thread(_write_text, path, "".join(lines))
    return ToolResult(f"The file {path} has been updated successfully.")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
