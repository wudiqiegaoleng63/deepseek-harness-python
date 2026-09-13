"""Out-of-process ACP subagent client.

Drives one child Agent Client Protocol server in a fresh subprocess: spawn,
``initialize``, ``session/new``, one ``session/prompt`` whose committed assistant
text becomes the run output, then a cooperative disposal ladder (stdin EOF →
SIGTERM → SIGKILL) that resolves only at whole-process quiescence.  Mirrors the
TS ``@deepseek-ai/dsh-subagent-acp`` run module, including its stop-reason
vocabulary, permission policy, and environment scrub.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .child_process import (
    DEFAULT_DISPOSE_EOF_GRACE_MS,
    DEFAULT_DISPOSE_GRACE_MS,
    child_environment,
    dispose_child_process,
)

PROTOCOL_VERSION = 1


PermissionPolicy = Literal["allow", "reject"]
SubagentStopReason = Literal["completed", "max-tokens", "refusal", "aborted", "error"]


class AcpClientError(RuntimeError):
    """The ACP child failed a protocol exchange or its transport closed."""


class AcpStartupCancelled(AcpClientError):
    """The run was cancelled before its ACP session was published."""

    def __init__(self, message: str = "subagent request was aborted before the ACP child started"):
        super().__init__(message)


def acp_stop_reason(reason: Any) -> SubagentStopReason:
    """Map an ACP stop reason to the harness subagent vocabulary.

    ``max_turn_requests`` — the child hit its turn-request budget — has no
    harness equivalent and means the task did not finish cleanly, so it and any
    unknown terminal reason map to ``error`` rather than a silent ``completed``.
    """

    if reason == "end_turn":
        return "completed"
    if reason == "max_tokens":
        return "max-tokens"
    if reason == "refusal":
        return "refusal"
    if reason == "cancelled":
        return "aborted"
    return "error"


def acp_content_text(content: Any) -> str:
    """Collect the text of an ACP content block (non-text blocks contribute nothing)."""

    if isinstance(content, dict) and content.get("type") == "text":
        text = content.get("text")
        return text if isinstance(text, str) else ""
    return ""


def to_acp_prompt(blocks: Sequence[Any]) -> list[dict[str, Any]]:
    """Translate harness prompt blocks into ACP blocks (text only)."""

    rendered: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                rendered.append({"type": "text", "text": text})
    return rendered


@dataclass(frozen=True, slots=True)
class AcpRunSpec:
    """Resolved spawn spec for one ACP child process."""

    command: str
    cwd: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    permission: PermissionPolicy = "reject"
    dispose_eof_grace_ms: int = DEFAULT_DISPOSE_EOF_GRACE_MS
    dispose_grace_ms: int = DEFAULT_DISPOSE_GRACE_MS


@dataclass(frozen=True, slots=True)
class AcpRunResult:
    """One finished ACP child turn."""

    output: str
    stop_reason: SubagentStopReason


@dataclass(frozen=True, slots=True)
class AcpSubagentConfig:
    """Deployment config for driving an external ACP agent as a subagent.

    The child is a fresh remote session (`inheritsParentContext: false`), so
    this provider offers one-shot foreground runs only; continuable or
    background delegation stays with the in-process provider.
    """

    command: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    permission: PermissionPolicy = "reject"
    #: Working-directory override; the parent session's cwd is used when absent.
    cwd: str | None = None
    provider_name: str = "acp"
    dispose_eof_grace_ms: int = DEFAULT_DISPOSE_EOF_GRACE_MS
    dispose_grace_ms: int = DEFAULT_DISPOSE_GRACE_MS

    def __post_init__(self) -> None:
        if not self.command.strip():
            raise ValueError("acp subagent command must be a non-empty string")
        if not self.provider_name.strip():
            raise ValueError("acp subagent provider name must be a non-empty string")
        if self.permission not in {"allow", "reject"}:
            raise ValueError("acp subagent permission must be allow or reject")
        if self.dispose_eof_grace_ms <= 0 or self.dispose_grace_ms <= 0:
            raise ValueError("acp subagent dispose graces must be positive")

    def spec_for(self, parent_cwd: str | None) -> AcpRunSpec:
        """Resolve the spawn spec for one delegation.

        The child runs in the configured override when set, else the delegating
        parent session's workspace — never this process's own cwd, because one
        harness process serves sessions from many workspaces.
        """

        raw = self.cwd if self.cwd is not None else parent_cwd
        if not raw:
            raise ValueError(
                "acp subagent requires a working directory: set its cwd or delegate "
                "from a session with one"
            )
        resolved = Path(raw).expanduser()
        if not resolved.is_absolute():
            resolved = (Path.cwd() / resolved).resolve()
        else:
            resolved = resolved.resolve()
        if not resolved.is_dir():
            raise ValueError(f"acp subagent working directory is not a directory: {resolved}")
        return AcpRunSpec(
            command=self.command,
            args=self.args,
            cwd=str(resolved),
            env=self.env,
            permission=self.permission,
            dispose_eof_grace_ms=self.dispose_eof_grace_ms,
            dispose_grace_ms=self.dispose_grace_ms,
        )


class _AcpConnection:
    """Line-delimited JSON-RPC over one child's stdio streams."""

    def __init__(self, process: asyncio.subprocess.Process, permission: PermissionPolicy) -> None:
        self._process = process
        self._permission = permission
        self._pending: dict[str, asyncio.Future[Any]] = {}
        self._ids = itertools.count(1)
        self._write_lock = asyncio.Lock()
        self._reader: asyncio.Task[None] | None = None
        self._closed = False
        self._partial: list[str] = []

    @property
    def output(self) -> str:
        """Committed assistant text collected so far."""

        return "".join(self._partial)

    def start(self) -> None:
        self._reader = asyncio.get_running_loop().create_task(self._read_loop())

    async def request(self, method: str, params: dict[str, Any]) -> Any:
        if self._closed:
            raise AcpClientError(f"ACP child transport is closed before {method}")
        request_id = f"c-{next(self._ids)}"
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._write(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            )
            return await future
        finally:
            self._pending.pop(request_id, None)

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        await self._write({"jsonrpc": "2.0", "method": method, "params": params})

    async def aclose(self) -> None:
        self._closed = True
        for future in tuple(self._pending.values()):
            if not future.done():
                future.set_exception(AcpClientError("ACP child transport closed"))
        self._pending.clear()
        reader, self._reader = self._reader, None
        if reader is not None:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader

    async def _write(self, frame: dict[str, Any]) -> None:
        payload = (json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        stdin = self._process.stdin
        if stdin is None or self._closed:
            raise AcpClientError("ACP child stdin is unavailable")
        async with self._write_lock:
            try:
                stdin.write(payload)
                await stdin.drain()
            except (BrokenPipeError, ConnectionResetError, RuntimeError) as exc:
                raise AcpClientError(f"ACP child transport failed: {exc}") from exc

    async def _read_loop(self) -> None:
        stdout = self._process.stdout
        if stdout is None:
            return
        try:
            while True:
                raw = await stdout.readline()
                if not raw:
                    break
                try:
                    message = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if not isinstance(message, dict):
                    continue
                await self._route(message)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        finally:
            for future in tuple(self._pending.values()):
                if not future.done():
                    future.set_exception(AcpClientError("ACP child closed its protocol stream"))

    async def _route(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        if not isinstance(method, str):
            future = self._pending.get(str(message.get("id")))
            if future is None or future.done():
                return
            error = message.get("error")
            if isinstance(error, dict):
                future.set_exception(
                    AcpClientError(str(error.get("message") or "ACP child returned an error"))
                )
            else:
                future.set_result(message.get("result"))
            return
        if "id" not in message:
            self._on_notification(method, message.get("params"))
            return
        # A request from the child: only the permission channel is part of this
        # client's contract, and its answer must be machine policy.
        if method == "session/request_permission":
            params = message.get("params")
            result = {
                "outcome": self._permission_answer(params if isinstance(params, dict) else {})
            }
        else:
            result = {}
        await self._write({"jsonrpc": "2.0", "id": message.get("id"), "result": result})

    def _on_notification(self, method: str, params: Any) -> None:
        if method != "session/update" or not isinstance(params, dict):
            return
        update = params.get("update")
        if not isinstance(update, dict) or update.get("sessionUpdate") != "agent_message_chunk":
            # Thoughts, tool calls, plans, and titles are consumed but never
            # surfaced: the subagent returns only its final answer.
            return
        self._partial.append(acp_content_text(update.get("content")))

    def _permission_answer(self, params: dict[str, Any]) -> dict[str, Any]:
        """Auto-answer a child permission request by the configured policy."""

        if self._permission == "allow":
            options = params.get("options")
            if isinstance(options, list):
                for option in options:
                    if not isinstance(option, dict):
                        continue
                    if option.get("kind") in {"allow_once", "allow_always"}:
                        return {"outcome": "selected", "optionId": option.get("optionId")}
        # No allow option was offered (or this client rejects): answer cancelled
        # so the child does not proceed.
        return {"outcome": "cancelled"}


class AcpRun:
    """One started ACP child, owning its result and teardown."""

    def __init__(
        self,
        run_id: str,
        process: asyncio.subprocess.Process,
        connection: _AcpConnection,
        session_id: str,
        prompt: Sequence[Any],
        spec: AcpRunSpec,
        cancel_event: asyncio.Event | None,
        on_error: Any = None,
    ) -> None:
        self.id = run_id
        self._process = process
        self._connection = connection
        self._session_id = session_id
        self._prompt = list(prompt)
        self._spec = spec
        self._cancel_event = cancel_event
        self._on_error = on_error
        self._cancelled = False
        self._loop = asyncio.get_running_loop()
        self._settled: asyncio.Future[None] = self._loop.create_future()
        self._disposal: asyncio.Task[None] | None = None
        self._cancel_task: asyncio.Task[None] | None = None
        self._result_task = self._loop.create_task(self._run_prompt())

    def cancel(self) -> None:
        """Request cancellation: the result settles ``aborted`` without waiting."""

        if self._cancelled:
            return
        self._cancelled = True
        if not self._settled.done():
            self._settled.set_result(None)
        # Best-effort ACP cancel; process teardown remains authoritative.
        self._cancel_task = self._loop.create_task(self._request_remote_cancel())

    async def _request_remote_cancel(self) -> None:
        with contextlib.suppress(Exception):
            await self._connection.notify("session/cancel", {"sessionId": self._session_id})

    async def result(self) -> AcpRunResult:
        """Await the child's answer, preserving any text folded before a cancel."""

        return await self._result_task

    @property
    def returncode(self) -> int | None:
        """The child's exit status, or ``None`` while it is still running."""

        return self._process.returncode

    async def _run_prompt(self) -> AcpRunResult:
        prompt = self._loop.create_task(
            self._connection.request(
                "session/prompt",
                {
                    "sessionId": self._session_id,
                    "prompt": to_acp_prompt(self._prompt),
                },
            )
        )
        cancel_wait = self._settled
        try:
            done, _ = await asyncio.wait({prompt, cancel_wait}, return_when=asyncio.FIRST_COMPLETED)
            if prompt in done:
                response = prompt.result()
                reason = response.get("stopReason") if isinstance(response, dict) else None
                return AcpRunResult(self._connection.output, acp_stop_reason(reason))
            return AcpRunResult(self._connection.output, "aborted")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # A transport failure is flattened into the result, but the original
            # fault is preserved for diagnostics rather than silently lost.
            if self._cancelled:
                return AcpRunResult(self._connection.output, "aborted")
            if self._on_error is not None:
                with contextlib.suppress(Exception):
                    self._on_error(exc, "error")
            return AcpRunResult(self._connection.output, "error")
        finally:
            # The cancellation future is shared state owned by cancel(); only
            # this turn's own request is torn down here.
            if not prompt.done():
                prompt.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await prompt

    async def dispose(self) -> None:
        """Idempotent teardown: ACP cancel, then EOF → SIGTERM → SIGKILL."""

        if self._disposal is None:
            self._disposal = self._loop.create_task(self._dispose_once())
        await self._disposal

    async def _dispose_once(self) -> None:
        self.cancel()
        await dispose_child_process(
            self._process, self._spec.dispose_eof_grace_ms, self._spec.dispose_grace_ms
        )
        await self._connection.aclose()
        if self._cancel_task is not None:
            with contextlib.suppress(Exception):
                await self._cancel_task
        # The torn-down transport settles any in-flight prompt; awaiting here
        # keeps the run's result attached to its process lifetime.
        with contextlib.suppress(Exception):
            await self._result_task


async def start_acp_run(
    prompt: Sequence[Any],
    *,
    spec: AcpRunSpec,
    cancel_event: asyncio.Event | None = None,
    on_error: Any = None,
) -> AcpRun:
    """Spawn an ACP child, establish its session, and publish a run handle.

    A spawn, initialization, or new-session failure reaps the still-private
    process before raising, so a failed start never leaves an orphan.
    """

    if cancel_event is not None and cancel_event.is_set():
        raise AcpStartupCancelled()
    env = child_environment(spec.env)
    try:
        process = await asyncio.create_subprocess_exec(
            spec.command,
            *spec.args,
            cwd=spec.cwd,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,
            start_new_session=True,
        )
    except OSError as exc:
        raise AcpClientError(f"ACP child failed to spawn: {exc}") from exc

    connection = _AcpConnection(process, spec.permission)
    connection.start()
    cancelled = False
    try:
        startup = asyncio.get_running_loop().create_task(_open_session(connection, spec.cwd))
        exit_wait = asyncio.get_running_loop().create_task(process.wait())
        racers: set[asyncio.Task[Any]] = {startup, exit_wait}
        cancel_wait: asyncio.Task[Any] | None = None
        if cancel_event is not None:
            cancel_wait = asyncio.get_running_loop().create_task(cancel_event.wait())
            racers.add(cancel_wait)
        try:
            done, _ = await asyncio.wait(racers, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in racers:
                if not task.done():
                    task.cancel()
        if cancel_wait is not None and cancel_wait in done:
            cancelled = True
            raise AcpStartupCancelled()
        if exit_wait in done:
            # A clean exit must never be mistaken for a ready session.
            raise AcpClientError(
                f"ACP child exited before its session started (code {process.returncode})"
            )
        session_id = startup.result()
    except BaseException:
        await connection.aclose()
        await dispose_child_process(process, spec.dispose_eof_grace_ms, spec.dispose_grace_ms)
        if cancelled:
            raise AcpStartupCancelled() from None
        raise

    run = AcpRun(
        str(uuid.uuid4()), process, connection, session_id, prompt, spec, cancel_event, on_error
    )
    return run


async def _open_session(connection: _AcpConnection, cwd: str) -> str:
    await connection.request(
        "initialize",
        {
            "protocolVersion": PROTOCOL_VERSION,
            # Advertise no optional client capabilities: the child self-serves.
            "clientCapabilities": {},
        },
    )
    session = await connection.request("session/new", {"cwd": cwd, "mcpServers": []})
    session_id = session.get("sessionId") if isinstance(session, dict) else None
    if not isinstance(session_id, str) or not session_id:
        raise AcpClientError("ACP child published without a session id")
    return session_id


__all__ = [
    "AcpClientError",
    "AcpRun",
    "AcpRunResult",
    "AcpRunSpec",
    "AcpStartupCancelled",
    "AcpSubagentConfig",
    "DEFAULT_DISPOSE_EOF_GRACE_MS",
    "DEFAULT_DISPOSE_GRACE_MS",
    "PROTOCOL_VERSION",
    "acp_content_text",
    "acp_stop_reason",
    "start_acp_run",
    "to_acp_prompt",
]
