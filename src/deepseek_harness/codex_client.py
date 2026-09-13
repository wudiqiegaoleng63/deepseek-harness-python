"""Out-of-process subagent client for the Codex ``app-server`` wire.

Spawns ``codex app-server --stdio``, performs the initialize handshake, creates
one ephemeral thread, submits one text task, and accepts only the answer of
that thread's own turn: the latest ``agentMessage`` with phase
``final_answer``, falling back to the latest unphased message, while commentary
is never an answer.  A completed turn without a non-blank answer, any other
terminal status, an unknown server request, or a protocol violation fails the
run; a context-window failure maps to ``max-tokens``.

This mirrors the TS ``@deepseek-ai/dsh-subagent-codex`` provider, including its
unattended policy for the requests Codex can raise while nobody is watching.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
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

CLIENT_INFO = {
    "name": "deepseek-harness",
    "title": "DeepSeek Harness",
    "version": "0.0.1",
}

#: Terminal turn statuses the app-server may report.
TERMINAL_STATUSES = ("completed", "interrupted", "failed")

SubagentStopReason = Literal["completed", "max-tokens", "refusal", "aborted", "error"]


class CodexError(RuntimeError):
    """The Codex app-server failed a protocol exchange or produced no answer."""


class CodexStartupCancelled(CodexError):
    """The run was cancelled before its ephemeral thread was published."""

    def __init__(self, message: str = "subagent request was aborted before Codex started"):
        super().__init__(message)


def unattended_decision(params: Mapping[str, Any]) -> str:
    """Pick the non-approval decision an unattended provider may take.

    ``cancel`` is preferred when the request offers it, ``decline`` otherwise;
    an app-server that offers no decision list at all falls back to
    ``decline``.  Neither choice ever grants the requested action.
    """

    available = params.get("availableDecisions")
    if not isinstance(available, list):
        return "decline"
    if "cancel" in available:
        return "cancel"
    if "decline" in available:
        return "decline"
    return "decline"


def context_window_exceeded(turn: Mapping[str, Any]) -> bool:
    """Whether a terminal turn failed because the child ran out of context."""

    error = turn.get("error")
    return isinstance(error, dict) and error.get("codexErrorInfo") == "contextWindowExceeded"


def select_answer(final_answer: str | None, unphased: str | None) -> str:
    """The best non-commentary answer observed so far, preserving bytes."""

    selected = final_answer if final_answer is not None else unphased
    if selected is None or not selected.strip():
        return ""
    return selected


@dataclass(frozen=True, slots=True)
class CodexRunSpec:
    """Resolved spawn spec for one Codex app-server child."""

    command: str = "codex"
    cwd: str = "."
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    dispose_eof_grace_ms: int = DEFAULT_DISPOSE_EOF_GRACE_MS
    dispose_grace_ms: int = DEFAULT_DISPOSE_GRACE_MS


@dataclass(frozen=True, slots=True)
class CodexSubagentConfig:
    """Deployment config for driving Codex as a subagent.

    The child runs its own ephemeral product thread in the delegating session's
    workspace; the parent conversation, persona, and tool policy do not cross
    the process boundary.
    """

    command: str = "codex"
    args: tuple[str, ...] = ("app-server", "--stdio")
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: str | None = None
    provider_name: str = "codex"
    dispose_eof_grace_ms: int = DEFAULT_DISPOSE_EOF_GRACE_MS
    dispose_grace_ms: int = DEFAULT_DISPOSE_GRACE_MS

    def __post_init__(self) -> None:
        if not self.command.strip():
            raise ValueError("codex subagent command must be a non-empty string")
        if not self.provider_name.strip():
            raise ValueError("codex subagent provider name must be a non-empty string")
        if self.dispose_eof_grace_ms <= 0 or self.dispose_grace_ms <= 0:
            raise ValueError("codex subagent dispose graces must be positive")

    def spec_for(self, parent_cwd: str | None) -> CodexRunSpec:
        """Resolve the spawn spec for one delegation."""

        raw = self.cwd if self.cwd is not None else parent_cwd
        if not raw:
            raise ValueError(
                "codex subagent requires a working directory: set its cwd or "
                "delegate from a session with one"
            )
        resolved = Path(raw).expanduser()
        resolved = (
            (Path.cwd() / resolved).resolve() if not resolved.is_absolute() else resolved.resolve()
        )
        if not resolved.is_dir():
            raise ValueError(f"codex subagent working directory is not a directory: {resolved}")
        return CodexRunSpec(
            command=self.command,
            args=self.args,
            cwd=str(resolved),
            env=self.env,
            dispose_eof_grace_ms=self.dispose_eof_grace_ms,
            dispose_grace_ms=self.dispose_grace_ms,
        )


@dataclass(frozen=True, slots=True)
class CodexRunResult:
    """One finished Codex turn."""

    output: str
    stop_reason: SubagentStopReason


class _CodexConnection:
    """JSON-RPC over one app-server child, including its server requests."""

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        on_notification: Any,
        on_server_request: Any,
    ) -> None:
        self._process = process
        self._on_notification = on_notification
        self._on_server_request = on_server_request
        self._pending: dict[str, asyncio.Future[Any]] = {}
        self._ids = itertools.count(1)
        self._write_lock = asyncio.Lock()
        self._reader: asyncio.Task[None] | None = None
        self._closed = False
        self._fatal: asyncio.Future[BaseException] = asyncio.get_running_loop().create_future()

    @property
    def fatal(self) -> asyncio.Future[BaseException]:
        """Settles when the transport failed for any reason.

        A protocol failure must fail the run, not only the exchange that
        tripped it, so the run races this against its own turn ending.
        """

        return self._fatal

    def start(self) -> None:
        self._reader = asyncio.get_running_loop().create_task(self._read_loop())

    async def request(self, method: str, params: dict[str, Any]) -> Any:
        if self._closed:
            raise CodexError(f"Codex transport is closed before {method}")
        request_id = f"c-{next(self._ids)}"
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._write(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            )
            done, _ = await asyncio.wait({future, self._fatal}, return_when=asyncio.FIRST_COMPLETED)
            if future in done:
                return future.result()
            raise self._fatal.result()
        finally:
            self._pending.pop(request_id, None)

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        frame: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            frame["params"] = params
        await self._write(frame)

    async def aclose(self) -> None:
        self._closed = True
        self._fail(CodexError("Codex transport closed"))
        reader, self._reader = self._reader, None
        if reader is not None:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader

    def _fail(self, error: BaseException) -> None:
        if not self._fatal.done():
            self._fatal.set_result(error)
        for future in tuple(self._pending.values()):
            if not future.done():
                future.set_exception(error)
        self._pending.clear()

    async def _write(self, frame: dict[str, Any]) -> None:
        payload = (json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        stdin = self._process.stdin
        if stdin is None or self._closed:
            raise CodexError("Codex stdin is unavailable")
        async with self._write_lock:
            try:
                stdin.write(payload)
                await stdin.drain()
            except (BrokenPipeError, ConnectionResetError, RuntimeError) as exc:
                raise CodexError(f"Codex transport failed: {exc}") from exc

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
                    await self._route(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail(CodexError(f"Codex protocol stream failed: {exc}"))
            return
        self._fail(CodexError("Codex app-server closed its protocol stream"))

    async def _route(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        if isinstance(method, str):
            params = message.get("params")
            params = params if isinstance(params, dict) else {}
            if "id" in message:
                try:
                    result = self._on_server_request(method, params)
                except Exception as exc:
                    self._fail(exc)
                    await self._write(
                        {
                            "jsonrpc": "2.0",
                            "id": message.get("id"),
                            "error": {"code": -32601, "message": str(exc)},
                        }
                    )
                    return
                await self._write({"jsonrpc": "2.0", "id": message.get("id"), "result": result})
                return
            self._on_notification(method, params)
            return
        future = self._pending.get(str(message.get("id")))
        if future is None or future.done():
            return
        error = message.get("error")
        if isinstance(error, dict):
            future.set_exception(CodexError(str(error.get("message") or "Codex returned an error")))
        else:
            future.set_result(message.get("result"))


class CodexRun:
    """One started Codex thread/turn, owning its result and teardown."""

    def __init__(
        self,
        run_id: str,
        process: asyncio.subprocess.Process,
        connection: _CodexConnection,
        thread_id: str,
        prompt: str,
        spec: CodexRunSpec,
        cancel_event: asyncio.Event | None,
        on_error: Any = None,
    ) -> None:
        self.id = run_id
        self.thread_id = thread_id
        self._process = process
        self._connection = connection
        self._prompt = prompt
        self._spec = spec
        self._cancel_event = cancel_event
        self._on_error = on_error
        self._cancelled = False
        self._loop = asyncio.get_running_loop()
        self._settled: asyncio.Future[None] = self._loop.create_future()
        self._turn_id: str | None = None
        self._terminal: asyncio.Future[dict[str, Any]] = self._loop.create_future()
        self._early: list[tuple[str, dict[str, Any]]] = []
        self._final_answer: str | None = None
        self._unphased: str | None = None
        self._disposal: asyncio.Task[None] | None = None
        self._terminal.add_done_callback(self._observe_terminal)
        self._result_task = self._loop.create_task(self.run_turn(prompt))

    # ------------------------------------------------------------------ output

    def answer(self) -> str:
        """The best non-commentary answer observed so far."""

        return select_answer(self._final_answer, self._unphased)

    # ------------------------------------------------------------ notification

    def on_notification(self, method: str, params: dict[str, Any]) -> None:
        if method not in {"turn/started", "item/completed", "turn/completed"}:
            return
        if self._turn_id is None:
            # The app-server may emit turn notifications before the turn/start
            # response names the turn; replay them once it does.
            self._early.append((method, params))
            return
        self._handle_notification(method, params)

    def _handle_notification(self, method: str, params: dict[str, Any]) -> None:
        if params.get("threadId") != self.thread_id:
            return
        if method == "turn/started":
            return
        if method == "item/completed":
            # Item notifications name their turn directly; only the terminal
            # notification carries a whole turn object.
            if params.get("turnId") != self._turn_id:
                return
            item = params.get("item")
            if not isinstance(item, dict) or item.get("type") != "agentMessage":
                return
            text = item.get("text")
            if not isinstance(text, str):
                self._fail(CodexError("Codex returned an invalid agent message"))
                return
            phase = item.get("phase")
            if phase == "final_answer":
                self._final_answer = text
            elif phase is None:
                self._unphased = text
            elif phase != "commentary":
                self._fail(CodexError(f"Codex returned an unknown agent message phase {phase!r}"))
            return
        turn = params.get("turn")
        if not isinstance(turn, dict) or str(turn.get("id")) != self._turn_id:
            return
        status = turn.get("status")
        if status not in TERMINAL_STATUSES:
            self._fail(CodexError(f"Codex returned invalid terminal turn status {status!r}"))
            return
        if not self._terminal.done():
            self._terminal.set_result(params)

    def _fail(self, error: BaseException) -> None:
        if not self._terminal.done():
            self._terminal.set_exception(error)

    @staticmethod
    def _observe_terminal(future: asyncio.Future[dict[str, Any]]) -> None:
        # The failure is delivered through run_turn; retrieving it here keeps a
        # protocol violation from surfacing as an unhandled-future warning when
        # the run already settled.
        if not future.cancelled():
            future.exception()

    # ---------------------------------------------------------- server request

    def on_server_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Answer one app-server request with the unattended policy."""

        self._require_own_thread(params, nullable_turn=method == "mcpServer/elicitation/request")
        if method in {
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
        }:
            return {"decision": unattended_decision(params)}
        if method == "item/permissions/requestApproval":
            return {"permissions": {}, "scope": "turn"}
        if method == "item/tool/requestUserInput":
            return {"answers": {}}
        if method == "mcpServer/elicitation/request":
            return {"action": "decline", "content": None, "_meta": None}
        raise CodexError(f"unsupported Codex app-server request {method!r}")

    def _require_own_thread(self, params: dict[str, Any], *, nullable_turn: bool) -> None:
        if params.get("threadId") != self.thread_id:
            raise CodexError("Codex request referenced another thread")
        turn_id = params.get("turnId")
        if nullable_turn and turn_id is None:
            return
        if not isinstance(turn_id, str) or (self._turn_id is not None and turn_id != self._turn_id):
            raise CodexError("Codex request referenced another turn")

    # ---------------------------------------------------------------- turn run

    async def run_turn(self, prompt: str) -> CodexRunResult:  # noqa: D401 - internal driver
        prompt_task = self._loop.create_task(
            self._connection.request(
                "turn/start",
                {
                    "threadId": self.thread_id,
                    "input": [{"type": "text", "text": prompt, "text_elements": []}],
                },
            )
        )
        try:
            done, _ = await asyncio.wait(
                {prompt_task, self._settled}, return_when=asyncio.FIRST_COMPLETED
            )
            if prompt_task not in done:
                return CodexRunResult(self.answer(), "aborted")
            response = prompt_task.result()
            turn = response.get("turn") if isinstance(response, dict) else None
            if not isinstance(turn, dict) or not isinstance(turn.get("id"), str):
                raise CodexError("Codex turn/start did not name a turn")
            self._turn_id = str(turn["id"])
            for method, params in self._early:
                self._handle_notification(method, params)
            self._early.clear()

            done, _ = await asyncio.wait(
                {self._terminal, self._connection.fatal, self._settled},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if self._connection.fatal in done:
                raise self._connection.fatal.result()
            if self._terminal not in done and self._settled in done:
                self._interrupt()
                return CodexRunResult(self.answer(), "aborted")
            completed = await self._terminal
            if self._cancelled:
                return CodexRunResult(self.answer(), "aborted")
            return self._settle(completed)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._cancelled:
                return CodexRunResult(self.answer(), "aborted")
            if self._on_error is not None:
                with contextlib.suppress(Exception):
                    self._on_error(exc, "error")
            return CodexRunResult(self.answer(), "error")
        finally:
            if not prompt_task.done():
                prompt_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await prompt_task

    def _settle(self, completed: Mapping[str, Any]) -> CodexRunResult:
        turn = completed.get("turn")
        turn = turn if isinstance(turn, dict) else {}
        if context_window_exceeded(turn):
            # The turn did not finish the task, but a context overflow is a
            # token limit rather than a generic failure.
            return CodexRunResult(self.answer(), "max-tokens")
        if turn.get("status") != "completed":
            raise CodexError(f"Codex turn ended with status {turn.get('status')!r}")
        output = self.answer()
        if not output.strip():
            raise CodexError("Codex completed without a final answer")
        return CodexRunResult(output, "completed")

    # --------------------------------------------------------------- lifecycle

    def cancel(self) -> None:
        """Request cancellation: the result settles ``aborted`` without waiting."""

        if self._cancelled:
            return
        self._cancelled = True
        if not self._settled.done():
            self._settled.set_result(None)
        self._interrupt()

    def _interrupt(self) -> None:
        if self._turn_id is None:
            return
        task = self._loop.create_task(self._interrupt_remote())
        task.add_done_callback(lambda _task: None)

    async def _interrupt_remote(self) -> None:
        with contextlib.suppress(Exception):
            await self._connection.request(
                "turn/interrupt", {"threadId": self.thread_id, "turnId": self._turn_id}
            )

    async def result(self) -> CodexRunResult:
        return await self._result_task

    @property
    def settled(self) -> bool:
        return self._result_task.done()

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    async def dispose(self) -> None:
        """Idempotent teardown: interrupt, close the wire, then reap the child."""

        if self._disposal is None:
            self._disposal = self._loop.create_task(self._dispose_once())
        await self._disposal

    async def _dispose_once(self) -> None:
        self.cancel()
        await self._connection.aclose()
        await dispose_child_process(
            self._process, self._spec.dispose_eof_grace_ms, self._spec.dispose_grace_ms
        )
        with contextlib.suppress(Exception):
            await self._result_task


async def start_codex_run(
    prompt: str,
    *,
    spec: CodexRunSpec,
    cancel_event: asyncio.Event | None = None,
    on_error: Any = None,
) -> CodexRun:
    """Spawn Codex, complete the handshake, and publish an ephemeral-thread run."""

    if cancel_event is not None and cancel_event.is_set():
        raise CodexStartupCancelled()
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
        raise CodexError(f"Codex failed to spawn: {exc}") from exc

    holder: dict[str, Any] = {}

    def on_notification(method: str, params: dict[str, Any]) -> None:
        run = holder.get("run")
        if run is not None:
            run.on_notification(method, params)

    def on_server_request(method: str, params: dict[str, Any]) -> dict[str, Any]:
        run = holder.get("run")
        if run is None:
            # A request before the run is published cannot be attributed to a
            # thread; refuse it rather than acting on unknown state.
            raise CodexError(f"Codex requested {method!r} before its thread started")
        return run.on_server_request(method, params)

    connection = _CodexConnection(process, on_notification, on_server_request)
    connection.start()
    cancelled = False
    try:
        startup = asyncio.get_running_loop().create_task(_open_thread(connection, spec.cwd))
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
            raise CodexStartupCancelled()
        if exit_wait in done:
            raise CodexError(f"Codex exited before its thread started (code {process.returncode})")
        thread_id = startup.result()
    except BaseException:
        await connection.aclose()
        await dispose_child_process(process, spec.dispose_eof_grace_ms, spec.dispose_grace_ms)
        if cancelled:
            raise CodexStartupCancelled() from None
        raise

    run = CodexRun(
        uuid.uuid4().hex, process, connection, thread_id, prompt, spec, cancel_event, on_error
    )
    holder["run"] = run
    return run


async def _open_thread(connection: _CodexConnection, cwd: str) -> str:
    await connection.request(
        "initialize",
        {
            "clientInfo": CLIENT_INFO,
            "capabilities": {"experimentalApi": False, "requestAttestation": False},
        },
    )
    await connection.notify("initialized")
    response = await connection.request("thread/start", {"cwd": cwd, "ephemeral": True})
    thread = response.get("thread") if isinstance(response, dict) else None
    if not isinstance(thread, dict):
        raise CodexError("Codex did not return a thread")
    thread_id = thread.get("id")
    if not isinstance(thread_id, str) or not thread_id:
        raise CodexError("Codex returned a thread without an id")
    if thread.get("ephemeral") is not True:
        raise CodexError("Codex did not create an ephemeral thread")
    return thread_id


__all__ = [
    "CLIENT_INFO",
    "CodexError",
    "CodexRun",
    "CodexRunResult",
    "CodexRunSpec",
    "CodexStartupCancelled",
    "CodexSubagentConfig",
    "TERMINAL_STATUSES",
    "context_window_exceeded",
    "select_answer",
    "start_codex_run",
    "unattended_decision",
]
