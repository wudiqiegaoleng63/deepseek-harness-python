"""Out-of-process subagent client for a child DeepSeek Harness SDK runtime.

Spawns a complete harness runtime in a fresh subprocess and drives it over the
stdio SDK JSON-RPC protocol: ``initialize``, one ``session/prompt``, and the
child's durable session events, from which the answer is read.  It is the
sibling of the ACP client and mirrors the TS ``@deepseek-ai/dsh-subagent-dsh-sdk``
provider: the same workspace resolution, environment scrub, and disposal ladder,
with a different wire and a child that is a full peer harness.
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
    scrubbed_child_env,
)

#: Default bound on the protocol ``shutdown`` exchange during disposal.
DEFAULT_SHUTDOWN_TIMEOUT_MS = 1_000

SubagentStopReason = Literal["completed", "max-tokens", "refusal", "aborted", "error"]


class SdkClientError(RuntimeError):
    """The child runtime failed a protocol exchange or its transport closed."""


class SdkStartupCancelled(SdkClientError):
    """The run was cancelled before its child session was published."""

    def __init__(self, message: str = "subagent request was aborted before the SDK child started"):
        super().__init__(message)


def turn_end_to_subagent_reason(kind: Any) -> SubagentStopReason:
    """Map a child harness turn ending to the subagent vocabulary.

    Everything the child can report without finishing cleanly — an error, an
    interrupted turn, a disposal, or a future variant — maps to ``error`` so a
    partial answer is never reported as success.
    """

    if kind == "completed":
        return "completed"
    if kind == "max-tokens":
        return "max-tokens"
    if kind == "aborted":
        return "aborted"
    return "error"


def assistant_text(events: Sequence[dict[str, Any]]) -> str:
    """Read a child's answer from its durable session events.

    The last complete non-empty ``assistant/message`` wins (an empty-content
    message that merely records usage is skipped); otherwise the accumulated
    streamed text is the answer.
    """

    answer = ""
    streamed: list[str] = []
    for event in events:
        kind = event.get("type")
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        if kind == "assistant/message":
            message = data.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list):
                text = "".join(
                    block["text"]
                    for block in content
                    if isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                )
                if text:
                    answer = text
        elif kind == "assistant/chunk":
            chunk = data.get("chunk")
            if (
                isinstance(chunk, dict)
                and chunk.get("kind") == "text"
                and isinstance(chunk.get("text"), str)
            ):
                streamed.append(chunk["text"])
    return answer or "".join(streamed)


@dataclass(frozen=True, slots=True)
class SdkRunSpec:
    """Resolved spawn spec for one child harness runtime."""

    command: str
    cwd: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    provider: str = "deepseek-official"
    model: str = "deepseek-v4-flash"
    max_tokens: int | None = None
    shutdown_timeout_ms: int = DEFAULT_SHUTDOWN_TIMEOUT_MS
    dispose_eof_grace_ms: int = DEFAULT_DISPOSE_EOF_GRACE_MS
    dispose_grace_ms: int = DEFAULT_DISPOSE_GRACE_MS


@dataclass(frozen=True, slots=True)
class SdkSubagentConfig:
    """Deployment config for driving a peer harness as a subagent.

    The child is a fresh runtime with its own session store, model route, and
    tools, so this provider offers one-shot foreground runs only.
    """

    command: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: str | None = None
    provider_name: str = "dsh-sdk"
    provider: str = "deepseek-official"
    model: str = "deepseek-v4-flash"
    max_tokens: int | None = None
    shutdown_timeout_ms: int = DEFAULT_SHUTDOWN_TIMEOUT_MS
    dispose_eof_grace_ms: int = DEFAULT_DISPOSE_EOF_GRACE_MS
    dispose_grace_ms: int = DEFAULT_DISPOSE_GRACE_MS

    def __post_init__(self) -> None:
        if not self.command.strip():
            raise ValueError("sdk subagent command must be a non-empty string")
        if not self.provider_name.strip():
            raise ValueError("sdk subagent provider name must be a non-empty string")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("sdk subagent max_tokens must be positive")
        if self.shutdown_timeout_ms <= 0:
            raise ValueError("sdk subagent shutdown timeout must be positive")
        if self.dispose_eof_grace_ms <= 0 or self.dispose_grace_ms <= 0:
            raise ValueError("sdk subagent dispose graces must be positive")

    def spec_for(self, parent_cwd: str | None) -> SdkRunSpec:
        """Resolve the spawn spec for one delegation."""

        raw = self.cwd if self.cwd is not None else parent_cwd
        if not raw:
            raise ValueError(
                "sdk subagent requires a working directory: set its cwd or delegate "
                "from a session with one"
            )
        resolved = Path(raw).expanduser()
        resolved = (
            (Path.cwd() / resolved).resolve() if not resolved.is_absolute() else resolved.resolve()
        )
        if not resolved.is_dir():
            raise ValueError(f"sdk subagent working directory is not a directory: {resolved}")
        return SdkRunSpec(
            command=self.command,
            args=self.args,
            cwd=str(resolved),
            env=self.env,
            provider=self.provider,
            model=self.model,
            max_tokens=self.max_tokens,
            shutdown_timeout_ms=self.shutdown_timeout_ms,
            dispose_eof_grace_ms=self.dispose_eof_grace_ms,
            dispose_grace_ms=self.dispose_grace_ms,
        )


@dataclass(frozen=True, slots=True)
class SdkRunResult:
    """One finished child runtime activity."""

    output: str
    stop_reason: SubagentStopReason


class _SdkConnection:
    """Line-delimited JSON-RPC over one child runtime's stdio streams."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process = process
        self._pending: dict[str, asyncio.Future[Any]] = {}
        self._ids = itertools.count(1)
        self._write_lock = asyncio.Lock()
        self._reader: asyncio.Task[None] | None = None
        self._closed = False
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._status: dict[str, str] = {}

    @property
    def events(self) -> dict[str, list[dict[str, Any]]]:
        return self._events

    def start(self) -> None:
        self._reader = asyncio.get_running_loop().create_task(self._read_loop())

    async def request(self, method: str, params: dict[str, Any]) -> Any:
        if self._closed:
            raise SdkClientError(f"child runtime transport is closed before {method}")
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

    async def aclose(self) -> None:
        self._closed = True
        for future in tuple(self._pending.values()):
            if not future.done():
                future.set_exception(SdkClientError("child runtime transport closed"))
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
            raise SdkClientError("child runtime stdin is unavailable")
        async with self._write_lock:
            try:
                stdin.write(payload)
                await stdin.drain()
            except (BrokenPipeError, ConnectionResetError, RuntimeError) as exc:
                raise SdkClientError(f"child runtime transport failed: {exc}") from exc

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
                if isinstance(message, dict):
                    self._route(message)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        finally:
            for future in tuple(self._pending.values()):
                if not future.done():
                    future.set_exception(SdkClientError("child runtime closed its protocol stream"))

    def _route(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        if isinstance(method, str):
            params = message.get("params")
            if method == "session.event" and isinstance(params, dict):
                session_id = params.get("sessionId")
                event = params.get("event")
                if isinstance(session_id, str) and isinstance(event, dict):
                    self._events.setdefault(session_id, []).append(event)
            return
        future = self._pending.get(str(message.get("id")))
        if future is None or future.done():
            return
        error = message.get("error")
        if isinstance(error, dict):
            data = error.get("data")
            detail = data.get("message") if isinstance(data, dict) else None
            future.set_exception(
                SdkClientError(str(detail or error.get("message") or "child runtime error"))
            )
        else:
            future.set_result(message.get("result"))


class SdkRun:
    """One started child runtime activity, owning its result and teardown."""

    def __init__(
        self,
        run_id: str,
        process: asyncio.subprocess.Process,
        connection: _SdkConnection,
        session_id: str,
        prompt: str,
        spec: SdkRunSpec,
        cancel_event: asyncio.Event | None,
        on_error: Any = None,
    ) -> None:
        self.id = run_id
        self.session_id = session_id
        self._process = process
        self._connection = connection
        self._prompt = prompt
        self._spec = spec
        self._cancel_event = cancel_event
        self._on_error = on_error
        self._cancelled = False
        self._loop = asyncio.get_running_loop()
        self._settled: asyncio.Future[None] = self._loop.create_future()
        self._disposal: asyncio.Task[None] | None = None
        self._result_task = self._loop.create_task(self._run_prompt())

    def cancel(self) -> None:
        """Request cancellation: the result settles ``aborted`` without waiting.

        The SDK wire has no prompt-level cancel, so the local settlement is the
        authoritative one; process teardown follows from ``dispose``.
        """

        if self._cancelled:
            return
        self._cancelled = True
        if not self._settled.done():
            self._settled.set_result(None)

    async def result(self) -> SdkRunResult:
        """Await the child's answer, preserving text folded before a cancel."""

        return await self._result_task

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    def _answer(self) -> tuple[str, Any]:
        events = self._connection.events.get(self.session_id, [])
        reason: Any = None
        for event in reversed(events):
            if event.get("type") == "turn/end":
                data = event.get("data")
                if isinstance(data, dict):
                    ended = data.get("reason")
                    reason = ended.get("kind") if isinstance(ended, dict) else None
                break
        return assistant_text(events), reason

    async def _run_prompt(self) -> SdkRunResult:
        prompt = self._loop.create_task(
            self._connection.request(
                "session/prompt",
                {
                    "sessionId": self.session_id,
                    "contentBlocks": [{"type": "text", "text": self._prompt}],
                },
            )
        )
        try:
            done, _ = await asyncio.wait(
                {prompt, self._settled}, return_when=asyncio.FIRST_COMPLETED
            )
            if prompt not in done:
                text, _ = self._answer()
                return SdkRunResult(text, "aborted")
            prompt.result()
            # The activity's durable turn ending decides the stop reason; the
            # prompt response only proves the turn was accepted.
            text, reason = self._answer()
            if self._cancelled:
                return SdkRunResult(text, "aborted")
            return SdkRunResult(text, turn_end_to_subagent_reason(reason))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            text, _ = self._answer()
            if self._cancelled:
                return SdkRunResult(text, "aborted")
            if self._on_error is not None:
                with contextlib.suppress(Exception):
                    self._on_error(exc, "error")
            return SdkRunResult(text, "error")
        finally:
            if not prompt.done():
                prompt.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await prompt

    async def dispose(self) -> None:
        """Idempotent teardown: bounded protocol shutdown, then the exit ladder."""

        if self._disposal is None:
            self._disposal = self._loop.create_task(self._dispose_once())
        await self._disposal

    async def _dispose_once(self) -> None:
        self.cancel()
        with contextlib.suppress(Exception):
            async with asyncio.timeout(self._spec.shutdown_timeout_ms / 1000):
                await self._connection.request("shutdown", {})
        await self._connection.aclose()
        await dispose_child_process(
            self._process, self._spec.dispose_eof_grace_ms, self._spec.dispose_grace_ms
        )
        with contextlib.suppress(Exception):
            await self._result_task


async def start_sdk_run(
    prompt: str,
    *,
    spec: SdkRunSpec,
    cancel_event: asyncio.Event | None = None,
    on_error: Any = None,
) -> SdkRun:
    """Spawn a child harness runtime, complete its handshake, and publish a run."""

    if cancel_event is not None and cancel_event.is_set():
        raise SdkStartupCancelled()
    try:
        process = await asyncio.create_subprocess_exec(
            spec.command,
            *spec.args,
            cwd=spec.cwd,
            env=child_environment(spec.env),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,
            start_new_session=True,
        )
    except OSError as exc:
        raise SdkClientError(f"child runtime failed to spawn: {exc}") from exc

    connection = _SdkConnection(process)
    connection.start()
    cancelled = False
    try:
        startup = asyncio.get_running_loop().create_task(_open_session(connection, spec))
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
            raise SdkStartupCancelled()
        if exit_wait in done:
            raise SdkClientError(
                f"child runtime exited before its session started (code {process.returncode})"
            )
        session_id = startup.result()
    except BaseException:
        await connection.aclose()
        await dispose_child_process(process, spec.dispose_eof_grace_ms, spec.dispose_grace_ms)
        if cancelled:
            raise SdkStartupCancelled() from None
        raise

    return SdkRun(
        uuid.uuid4().hex, process, connection, session_id, prompt, spec, cancel_event, on_error
    )


async def _open_session(connection: _SdkConnection, spec: SdkRunSpec) -> str:
    request: dict[str, Any] = {
        "cwd": spec.cwd,
        "provider": spec.provider,
        "model": spec.model,
    }
    if spec.max_tokens is not None:
        request["maxTokens"] = spec.max_tokens
    await connection.request("initialize", request)
    # The child runtime keys sessions by the id the parent sends, so the run
    # mints one in its own namespace for this activity only.
    return f"sdk-{uuid.uuid4().hex}"


__all__ = [
    "DEFAULT_SHUTDOWN_TIMEOUT_MS",
    "SdkClientError",
    "SdkRun",
    "SdkRunResult",
    "SdkRunSpec",
    "SdkStartupCancelled",
    "SdkSubagentConfig",
    "assistant_text",
    "scrubbed_child_env",
    "turn_end_to_subagent_reason",
]
