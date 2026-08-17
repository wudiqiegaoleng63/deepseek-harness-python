"""Native synchronous SDK built on the same Python host service as the Web UI."""

from __future__ import annotations

import asyncio
import os
import threading
import uuid
from collections.abc import Callable, Coroutine
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from .compaction import CompactionPolicy
from .llm.adapter import LlmAdapter
from .llm.types import RetryPolicy
from .tools import PermissionMode
from .web.service import HarnessService

JsonObject = dict[str, Any]
AdapterFactory = Callable[[str], LlmAdapter]
EventCallback = Callable[[JsonObject], None]
T = TypeVar("T")


@dataclass(slots=True)
class DeepSeekHarnessConfig:
    """Configuration for an in-process native Harness runtime."""

    provider: str = "deepseek-official"
    model: str = "deepseek-v4-flash"
    max_tokens: int | None = None
    cwd: str | os.PathLike[str] | None = None
    session_root: str | os.PathLike[str] | None = None
    permission_mode: PermissionMode = PermissionMode.WORKSPACE_WRITE
    base_url: str | None = None
    api_key: str | None = None
    request_timeout_seconds: float | None = 300.0
    shutdown_timeout_seconds: float = 5.0
    retry_policy: RetryPolicy | None = None
    compaction_policy: CompactionPolicy | None = None
    adapter_factory: AdapterFactory | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.request_timeout_seconds is not None and self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive or None")
        if self.shutdown_timeout_seconds <= 0:
            raise ValueError("shutdown_timeout_seconds must be positive")


@dataclass(frozen=True, slots=True)
class RunResult:
    session_id: str
    final_response: str
    finish_reason: str | None
    events: tuple[JsonObject, ...]


class DeepSeekHarness:
    """Reusable synchronous facade over a background asyncio host."""

    def __init__(self, config: DeepSeekHarnessConfig | None = None, **kwargs: Any) -> None:
        if config is not None and kwargs:
            raise TypeError("pass either DeepSeekHarnessConfig or keyword options, not both")
        self.config = config or DeepSeekHarnessConfig(**kwargs)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runtime: HarnessService | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._closed = True

    def __enter__(self) -> DeepSeekHarness:
        self.start()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()

    def start(self) -> None:
        if not self._closed:
            return
        self._closed = False
        self._ready.clear()
        self._startup_error = None
        self._thread = threading.Thread(target=self._thread_main, name="dsh-python-runtime")
        self._thread.start()
        if not self._ready.wait(self.config.shutdown_timeout_seconds):
            self._closed = True
            raise TimeoutError("DeepSeek Harness runtime did not start in time")
        if self._startup_error is not None:
            error = self._startup_error
            self.close()
            raise RuntimeError("DeepSeek Harness runtime failed to start") from error

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        loop = self._loop
        thread = self._thread
        if loop is not None and loop.is_running():
            try:
                self._call(self._shutdown(), timeout=self.config.shutdown_timeout_seconds)
            except Exception:
                loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=self.config.shutdown_timeout_seconds)
        self._loop = None
        self._runtime = None
        self._thread = None

    def run(
        self,
        input: str | list[JsonObject],
        *,
        session_id: str | None = None,
        on_event: EventCallback | None = None,
    ) -> RunResult:
        """Run one turn, reusing the durable session when ``session_id`` repeats."""

        self.start()
        content = [{"type": "text", "text": input}] if isinstance(input, str) else input
        return self._call(
            self._run(
                content,
                session_id or f"session-{uuid.uuid4().hex}",
                on_event,
            ),
            timeout=self.config.request_timeout_seconds,
        )

    def start_session(self, session_id: str | None = None) -> Session:
        self.start()
        return Session(self, session_id or f"session-{uuid.uuid4().hex}")

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            runtime = HarnessService(
                self.config.session_root
                or os.getenv("DSH_SESSION_ROOT", "~/.deepseek_harness_python/sessions"),
                cwd=self.config.cwd,
                model=self.config.model,
                permission_mode=self.config.permission_mode,
                adapter_factory=self.config.adapter_factory,
                retry_policy=self.config.retry_policy,
                compaction_policy=self.config.compaction_policy,
            )
            self._runtime = runtime
            loop.run_until_complete(self._configure(runtime))
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            loop.close()
            return
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            if not loop.is_closed():
                loop.close()

    async def _configure(self, runtime: HarnessService) -> None:
        patch: JsonObject = {}
        if self.config.max_tokens is not None:
            patch["maxTokens"] = self.config.max_tokens
        if self.config.base_url is not None:
            patch["baseURL"] = self.config.base_url
        if patch:
            await runtime.settings.update("llm-deepseek", patch)
        if self.config.api_key is not None:
            await runtime.credentials.set("DEEPSEEK_API_KEY", self.config.api_key)

    async def _shutdown(self) -> None:
        runtime = self._runtime
        if runtime is not None:
            await runtime.dispose()
        loop = asyncio.get_running_loop()
        loop.call_soon(loop.stop)

    async def _run(
        self,
        content: list[JsonObject],
        session_id: str,
        on_event: EventCallback | None,
    ) -> RunResult:
        runtime = self._require_runtime()
        handle = await runtime.create_session(
            session_id=session_id,
            cwd=str(Path(self.config.cwd or Path.cwd()).expanduser().resolve()),
        )
        first_seq = handle.session.seq
        await runtime.prompt(session_id, content, include_message_id=True)
        task = handle.task
        if task is not None:
            await task
        events = tuple(event.to_dict() for event in handle.session.events if event.seq >= first_seq)
        if on_event is not None:
            for event in events:
                on_event(event)
        return RunResult(
            session_id,
            self._final_response(events),
            self._finish_reason(events),
            events,
        )

    def _call(self, coroutine: Coroutine[Any, Any, T], *, timeout: float | None = None) -> T:
        loop = self._loop
        if loop is None or not loop.is_running():
            coroutine.close()
            raise RuntimeError("DeepSeek Harness runtime is not running")
        future: Future[T] = asyncio.run_coroutine_threadsafe(coroutine, loop)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            # Propagate cancellation into the event-loop thread so a timed-out
            # request cannot keep mutating the session after the caller has
            # already observed the timeout.
            future.cancel()
            raise

    def _require_runtime(self) -> HarnessService:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("DeepSeek Harness runtime is not initialized")
        return runtime

    @staticmethod
    def _final_response(events: tuple[JsonObject, ...]) -> str:
        for event in reversed(events):
            if event.get("type") != "assistant/message":
                continue
            data = event.get("data")
            message = data.get("message") if isinstance(data, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            return "".join(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return ""

    @staticmethod
    def _finish_reason(events: tuple[JsonObject, ...]) -> str | None:
        for event in reversed(events):
            if event.get("type") != "turn/end":
                continue
            data = event.get("data")
            reason = data.get("reason") if isinstance(data, dict) else None
            kind = reason.get("kind") if isinstance(reason, dict) else None
            return kind if isinstance(kind, str) else None
        return None


class Session:
    """A stable SDK handle for repeatedly continuing one durable session."""

    def __init__(self, harness: DeepSeekHarness, session_id: str) -> None:
        self.harness = harness
        self.id = session_id

    def run(
        self,
        input: str | list[JsonObject],
        *,
        on_event: EventCallback | None = None,
    ) -> RunResult:
        return self.harness.run(input, session_id=self.id, on_event=on_event)


__all__ = ["DeepSeekHarness", "DeepSeekHarnessConfig", "RunResult", "Session"]
