"""Out-of-process subagent client for the native Claude Code CLI.

Runs one fresh ``claude --print --output-format stream-json`` invocation in the
delegating session's workspace and accepts exactly one answer: the final
``result`` message must report subtype ``success``, must not be marked as an
error, and must carry non-blank text.  Everything else — another subtype, an
error-marked success, a blank answer, a missing result, or a process failure —
is a failed run, so a partial product session is never surfaced as an answer.

This mirrors the TS ``@deepseek-ai/dsh-subagent-claude-code`` provider, which
drives the same native CLI through the official Agent SDK; the Python host
speaks the CLI's own stream-json wire, so it needs no product SDK dependency.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .child_process import (
    DEFAULT_DISPOSE_EOF_GRACE_MS,
    DEFAULT_DISPOSE_GRACE_MS,
    child_environment,
    dispose_child_process,
)

#: The tool a delegated child must never use: nobody is there to answer it.
DISALLOWED_TOOLS = ("AskUserQuestion",)

SubagentStopReason = Literal["completed", "max-tokens", "refusal", "aborted", "error"]


class ClaudeCodeError(RuntimeError):
    """The Claude Code child failed to produce an answer."""


class ClaudeCodeStartupCancelled(ClaudeCodeError):
    """The run was cancelled before the child process was published."""

    def __init__(self, message: str = "subagent request was aborted before Claude Code started"):
        super().__init__(message)


def successful_result(message: Mapping[str, Any]) -> str:
    """Strictly derive the only CLI result that can complete a run.

    A ``result`` message qualifies only when it reports subtype ``success``,
    carries ``is_error: false``, and contains non-blank text; the caller turns
    any raised detail into a failed run.
    """

    subtype = message.get("subtype")
    is_error = message.get("is_error")
    answer = message.get("result")
    if (
        subtype != "success"
        or is_error is True
        or not isinstance(answer, str)
        or not answer.strip()
    ):
        errors = message.get("errors")
        detail = (
            "success result was marked as an error or contained no answer"
            if subtype == "success"
            else (
                "; ".join(str(item) for item in errors)
                if isinstance(errors, list) and errors
                else str(subtype)
            )
        )
        raise ClaudeCodeError(f"Claude Code failed: {detail}")
    return answer


def read_cli_events(lines: list[str]) -> str:
    """Consume one complete CLI stream and require a strict success result."""

    answer: str | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            message = json.loads(stripped)
        except ValueError:
            continue
        if not isinstance(message, dict) or message.get("type") != "result":
            continue
        answer = successful_result(message)
    if answer is None:
        raise ClaudeCodeError("Claude Code ended without a result")
    return answer


@dataclass(frozen=True, slots=True)
class ClaudeCodeRunSpec:
    """Resolved spawn spec for one Claude Code child process."""

    command: str = "claude"
    cwd: str = "."
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    model: str | None = None
    #: Extra CLI flags the deployment owns (for example a permission mode).
    extra_args: tuple[str, ...] = ()
    dispose_eof_grace_ms: int = DEFAULT_DISPOSE_EOF_GRACE_MS
    dispose_grace_ms: int = DEFAULT_DISPOSE_GRACE_MS


@dataclass(frozen=True, slots=True)
class ClaudeCodeSubagentConfig:
    """Deployment config for driving the native Claude Code CLI as a subagent.

    The child is a fresh product session in its own process: it inherits the
    parent workspace and the host's native settings, but not the parent
    conversation, persona, or tool policy.
    """

    command: str = "claude"
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: str | None = None
    provider_name: str = "claude-code"
    model: str | None = None
    extra_args: tuple[str, ...] = ()
    dispose_eof_grace_ms: int = DEFAULT_DISPOSE_EOF_GRACE_MS
    dispose_grace_ms: int = DEFAULT_DISPOSE_GRACE_MS

    def __post_init__(self) -> None:
        if not self.command.strip():
            raise ValueError("claude-code subagent command must be a non-empty string")
        if not self.provider_name.strip():
            raise ValueError("claude-code subagent provider name must be a non-empty string")
        if self.model is not None and not self.model.strip():
            raise ValueError("claude-code subagent model must be a non-empty string")
        if self.dispose_eof_grace_ms <= 0 or self.dispose_grace_ms <= 0:
            raise ValueError("claude-code subagent dispose graces must be positive")

    def spec_for(self, parent_cwd: str | None) -> ClaudeCodeRunSpec:
        """Resolve the spawn spec for one delegation."""

        raw = self.cwd if self.cwd is not None else parent_cwd
        if not raw:
            raise ValueError(
                "claude-code subagent requires a working directory: set its cwd or "
                "delegate from a session with one"
            )
        resolved = Path(raw).expanduser()
        resolved = (
            (Path.cwd() / resolved).resolve() if not resolved.is_absolute() else resolved.resolve()
        )
        if not resolved.is_dir():
            raise ValueError(f"claude-code working directory is not a directory: {resolved}")
        return ClaudeCodeRunSpec(
            command=self.command,
            args=self.args,
            cwd=str(resolved),
            env=self.env,
            model=self.model,
            extra_args=self.extra_args,
            dispose_eof_grace_ms=self.dispose_eof_grace_ms,
            dispose_grace_ms=self.dispose_grace_ms,
        )


@dataclass(frozen=True, slots=True)
class ClaudeCodeRunResult:
    """One finished Claude Code invocation."""

    output: str
    stop_reason: SubagentStopReason


def cli_arguments(spec: ClaudeCodeRunSpec) -> list[str]:
    """The fixed non-interactive flag set plus the deployment's own flags.

    The child gets no session persistence and cannot ask a question, matching
    the provider contract: an unattended run either answers or fails.
    """

    arguments = [
        *spec.args,
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
        "--no-session-persistence",
        "--disallowed-tools",
        *DISALLOWED_TOOLS,
    ]
    if spec.model is not None:
        arguments.extend(["--model", spec.model])
    arguments.extend(spec.extra_args)
    return arguments


class ClaudeCodeRun:
    """One started Claude Code child, owning its result and teardown."""

    def __init__(
        self,
        run_id: str,
        process: asyncio.subprocess.Process,
        spec: ClaudeCodeRunSpec,
        cancel_event: asyncio.Event | None,
        on_error: Any = None,
    ) -> None:
        self.id = run_id
        self._process = process
        self._spec = spec
        self._cancel_event = cancel_event
        self._on_error = on_error
        self._cancelled = False
        self._loop = asyncio.get_running_loop()
        self._settled: asyncio.Future[None] = self._loop.create_future()
        self._disposal: asyncio.Task[None] | None = None
        self._lines: list[str] = []
        self._reader = self._loop.create_task(self._read_stream())
        self._result_task = self._loop.create_task(self._run())

    def cancel(self) -> None:
        """Request cancellation: the result settles ``aborted`` without waiting."""

        if self._cancelled:
            return
        self._cancelled = True
        if not self._settled.done():
            self._settled.set_result(None)

    async def result(self) -> ClaudeCodeRunResult:
        return await self._result_task

    @property
    def settled(self) -> bool:
        return self._result_task.done()

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    async def _read_stream(self) -> None:
        stdout = self._process.stdout
        if stdout is None:
            return
        with contextlib.suppress(Exception):
            while True:
                raw = await stdout.readline()
                if not raw:
                    return
                self._lines.append(raw.decode("utf-8", errors="replace"))

    async def _run(self) -> ClaudeCodeRunResult:
        try:
            done, _ = await asyncio.wait(
                {self._reader, self._settled}, return_when=asyncio.FIRST_COMPLETED
            )
            if self._reader not in done:
                return ClaudeCodeRunResult("", "aborted")
            await self._process.wait()
            if self._cancelled:
                return ClaudeCodeRunResult("", "aborted")
            return ClaudeCodeRunResult(read_cli_events(self._lines), "completed")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._cancelled:
                return ClaudeCodeRunResult("", "aborted")
            if self._on_error is not None:
                with contextlib.suppress(Exception):
                    self._on_error(exc, "error")
            return ClaudeCodeRunResult("", "error")
        finally:
            if not self._reader.done():
                self._reader.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._reader

    async def dispose(self) -> None:
        """Idempotent teardown: EOF → SIGTERM → SIGKILL over the process tree."""

        if self._disposal is None:
            self._disposal = self._loop.create_task(self._dispose_once())
        await self._disposal

    async def _dispose_once(self) -> None:
        self.cancel()
        await dispose_child_process(
            self._process, self._spec.dispose_eof_grace_ms, self._spec.dispose_grace_ms
        )
        with contextlib.suppress(Exception):
            await self._result_task


async def start_claude_run(
    prompt: str,
    *,
    spec: ClaudeCodeRunSpec,
    cancel_event: asyncio.Event | None = None,
    on_error: Any = None,
) -> ClaudeCodeRun:
    """Spawn one Claude Code child, hand it the task, and publish a run.

    The task travels on stdin rather than argv, so a long delegation cannot hit
    the platform's argument limit.
    """

    if cancel_event is not None and cancel_event.is_set():
        raise ClaudeCodeStartupCancelled()
    try:
        process = await asyncio.create_subprocess_exec(
            spec.command,
            *cli_arguments(spec),
            cwd=spec.cwd,
            env=child_environment(spec.env),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,
            start_new_session=True,
        )
    except OSError as exc:
        raise ClaudeCodeError(f"Claude Code failed to spawn: {exc}") from exc

    cancelled = False
    try:
        assert process.stdin is not None
        process.stdin.write(prompt.encode("utf-8"))
        await process.stdin.drain()
        process.stdin.close()
    except BaseException:
        cancelled = cancel_event is not None and cancel_event.is_set()
        await dispose_child_process(process, spec.dispose_eof_grace_ms, spec.dispose_grace_ms)
        if cancelled:
            raise ClaudeCodeStartupCancelled() from None
        raise

    return ClaudeCodeRun(str(uuid.uuid4()), process, spec, cancel_event, on_error)


__all__ = [
    "DISALLOWED_TOOLS",
    "ClaudeCodeError",
    "ClaudeCodeRun",
    "ClaudeCodeRunResult",
    "ClaudeCodeRunSpec",
    "ClaudeCodeStartupCancelled",
    "ClaudeCodeSubagentConfig",
    "cli_arguments",
    "read_cli_events",
    "successful_result",
]
