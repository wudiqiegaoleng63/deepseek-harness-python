"""Synchronous client for a subprocess-backed Harness SDK runtime.

The wire protocol is shared with the TypeScript SDK: one JSON-RPC frame per
line on stdin/stdout, with session and subagent lifecycle notifications flowing
independently of request responses.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]
NotificationCallback = Callable[[JsonObject], None]
NotificationFilter = Callable[[JsonObject], bool]


class TransportClosedError(RuntimeError):
    """The SDK runtime process exited before answering a request."""


class RequestTimeoutError(TimeoutError):
    """A JSON-RPC request or notification wait exceeded its deadline."""


class SdkProtocolError(ValueError):
    """The SDK runtime returned an invalid protocol shape."""


class JsonRpcResponseError(RuntimeError):
    """The SDK runtime returned a JSON-RPC error response."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


@dataclass(frozen=True, slots=True)
class ProcessRunResult:
    """One high-level SDK activity interval."""

    session_id: str
    final_response: str
    events: tuple[JsonObject, ...]
    notifications: tuple[JsonObject, ...]


_TERMINAL = object()


class NotificationSubscription:
    """Blocking subscription over notifications matching one client filter."""

    def __init__(self, owner: HarnessClient, filter_: NotificationFilter | None) -> None:
        self._owner = owner
        self._filter = filter_
        self._queue: queue.Queue[JsonObject | object] = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False
        self._failure: BaseException | None = None

    def next(self, timeout: float | None = None) -> JsonObject:
        """Wait for the next matching notification."""

        with self._lock:
            failure = self._failure
            closed = self._closed
        if failure is not None and self._queue.empty():
            raise failure
        if closed and self._queue.empty():
            raise TransportClosedError("notification subscription is closed")
        try:
            value = self._queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise RequestTimeoutError("notification wait timed out") from exc
        if value is _TERMINAL:
            with self._lock:
                failure = self._failure
            if failure is not None:
                raise failure from None
            raise TransportClosedError("notification subscription is closed")
        assert isinstance(value, dict)
        return value

    def try_next(self) -> JsonObject | None:
        """Return one queued notification, or ``None`` when none is ready."""

        try:
            value = self._queue.get_nowait()
        except queue.Empty:
            with self._lock:
                failure = self._failure
            if failure is not None:
                raise failure from None
            return None
        if value is _TERMINAL:
            with self._lock:
                failure = self._failure
            if failure is not None:
                raise failure
            raise TransportClosedError("notification subscription is closed")
        assert isinstance(value, dict)
        return value

    def close(self) -> None:
        """Detach and discard queued notifications."""

        self._owner._remove_subscription(self)
        with self._lock:
            self._closed = True
            self._failure = TransportClosedError("notification subscription closed")
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._queue.put(_TERMINAL)

    def __iter__(self) -> Iterator[JsonObject]:
        while True:
            yield self.next()

    def _push(self, notification: JsonObject) -> None:
        with self._lock:
            if self._closed or self._failure is not None:
                return
            filter_ = self._filter
        if filter_ is not None:
            try:
                if not filter_(notification):
                    return
            except BaseException as exc:
                self._owner._remove_subscription(self)
                self._fail(exc)
                return
        with self._lock:
            if self._closed or self._failure is not None:
                return
            self._queue.put(notification)

    def _fail(self, error: BaseException) -> None:
        with self._lock:
            if self._failure is not None:
                return
            self._failure = error
            self._queue.put(_TERMINAL)


class HarnessClient:
    """Low-level line JSON-RPC client that owns one runtime subprocess."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: dict[str, str] | None = None,
        request_timeout_seconds: float | None = 300.0,
        shutdown_timeout_seconds: float = 5.0,
    ) -> None:
        if not command:
            raise ValueError("SDK runtime command cannot be empty")
        if request_timeout_seconds is not None and request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive or None")
        if shutdown_timeout_seconds <= 0:
            raise ValueError("shutdown_timeout_seconds must be positive")
        self.command = tuple(command)
        self.cwd = str(Path(cwd).expanduser().resolve()) if cwd is not None else None
        self.env = env
        self.request_timeout_seconds = request_timeout_seconds
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self._process: subprocess.Popen[bytes] | None = None
        self._response_queue: queue.Queue[JsonObject | object] = queue.Queue()
        self._backlog: deque[JsonObject] = deque()
        self._notification_backlog: deque[JsonObject] = deque(maxlen=10_000)
        self._write_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._subscriptions: set[NotificationSubscription] = set()
        self._session_parents: dict[str, str] = {}
        self._next_id = 0
        self._closed = False
        self._closing = False
        self._stderr_tail: deque[str] = deque(maxlen=400)
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None

    def start(self) -> None:
        """Spawn the runtime lazily, or return when it is already live."""

        if self._closed:
            raise TransportClosedError("DeepSeek Harness runtime client is closed")
        process = self._process
        if process is not None:
            if process.poll() is None:
                return
            raise self._closed_error("DeepSeek Harness runtime is not running")
        try:
            process = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                env=self.env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise TransportClosedError(f"DeepSeek Harness runtime failed to start: {exc}") from exc
        self._process = process
        assert process.stdout is not None
        assert process.stderr is not None
        self._reader_thread = threading.Thread(
            target=self._read_stdout,
            args=(process.stdout,),
            name="dsh-sdk-stdout",
            daemon=True,
        )
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(process.stderr,),
            name="dsh-sdk-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    def initialize(
        self,
        *,
        cwd: str | os.PathLike[str],
        provider: str = "deepseek-official",
        model: str = "deepseek-v4-flash",
        max_tokens: int | None = None,
    ) -> JsonObject:
        params: JsonObject = {
            "cwd": str(Path(cwd).expanduser().resolve()),
            "provider": provider,
            "model": model,
        }
        if max_tokens is not None:
            params["maxTokens"] = max_tokens
        result = self.request("initialize", params)
        server_info = result.get("serverInfo")
        if (
            not isinstance(server_info, dict)
            or not isinstance(server_info.get("name"), str)
            or not isinstance(server_info.get("version"), str)
        ):
            raise SdkProtocolError("initialize returned no server identity")
        return result

    def prompt(self, session_id: str, content_blocks: list[JsonObject]) -> str:
        result = self.request(
            "session/prompt",
            {"sessionId": session_id, "contentBlocks": content_blocks},
        )
        message_id = result.get("messageId")
        if not isinstance(message_id, str):
            raise SdkProtocolError("session/prompt returned no message id")
        return message_id

    def request(self, method: str, params: JsonObject | None = None) -> JsonObject:
        """Send one request and wait for its matching response."""

        self.start()
        with self._request_lock:
            request_id = self._next_id
            self._next_id += 1
            self._write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": {} if params is None else params,
                }
            )
            deadline = self._deadline(self.request_timeout_seconds)
            while True:
                frame = self._next_response(deadline)
                if frame.get("id") != request_id:
                    self._backlog.append(frame)
                    continue
                error = frame.get("error")
                if isinstance(error, dict):
                    code = error.get("code")
                    if not isinstance(code, int) or isinstance(code, bool):
                        code = -32603
                    message = error.get("message")
                    raise JsonRpcResponseError(
                        code,
                        message if isinstance(message, str) else "SDK request failed",
                        error.get("data"),
                    )
                result = frame.get("result")
                if not isinstance(result, dict):
                    raise SdkProtocolError(f"{method} returned a non-object result")
                return result

    def subscribe(self, filter_: NotificationFilter | None = None) -> NotificationSubscription:
        """Subscribe to future server notifications matching ``filter_``."""

        subscription = NotificationSubscription(self, filter_)
        with self._state_lock:
            if self._closed:
                subscription._fail(self._closed_error("DeepSeek Harness runtime is closed"))
            else:
                self._subscriptions.add(subscription)
        return subscription

    def subscribe_session_tree(self, session_id: str) -> NotificationSubscription:
        """Subscribe to a session and descendants discovered via subagent edges."""

        def matches(notification: JsonObject) -> bool:
            method = notification.get("method")
            params = notification.get("params")
            if not isinstance(params, dict):
                return False
            if method in {"subagent.started", "subagent.finished"}:
                parent_id = params.get("parentSessionId")
                child_id = params.get("childSessionId")
                return (
                    isinstance(parent_id, str)
                    and self._is_descendant(parent_id, session_id)
                ) or child_id == session_id
            related_id = params.get("sessionId")
            return isinstance(related_id, str) and self._is_descendant(related_id, session_id)

        return self.subscribe(matches)

    def drain_notifications(self) -> list[JsonObject]:
        """Drain notifications received since the last call."""

        with self._state_lock:
            values = list(self._notification_backlog)
            self._notification_backlog.clear()
        return values

    def close(self) -> None:
        """Best-effort shutdown followed by stdin EOF and process reaping."""

        with self._close_lock:
            if self._closed:
                return
            self._closing = True
            process = self._process
            if process is None:
                self._closed = True
                self._fail_subscriptions(
                    TransportClosedError("DeepSeek Harness runtime client is closed")
                )
                return
            if process.poll() is None:
                old_timeout = self.request_timeout_seconds
                self.request_timeout_seconds = self.shutdown_timeout_seconds
                try:
                    self.request("shutdown", {})
                except Exception:
                    pass
                finally:
                    self.request_timeout_seconds = old_timeout
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=self.shutdown_timeout_seconds)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=self.shutdown_timeout_seconds)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            self._closed = True
            self._fail_subscriptions(self._closed_error("DeepSeek Harness runtime closed"))

    def _write(self, frame: JsonObject) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise TransportClosedError("SDK runtime is not running")
        payload = (json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        with self._write_lock:
            try:
                process.stdin.write(payload)
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise self._closed_error("SDK runtime stdin is closed") from exc

    def _next_response(self, deadline: float | None) -> JsonObject:
        if self._backlog:
            return self._backlog.popleft()
        timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
        if timeout == 0:
            raise RequestTimeoutError("SDK request timed out")
        try:
            value = self._response_queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise RequestTimeoutError("SDK request timed out") from exc
        if value is _TERMINAL:
            raise self._closed_error("SDK runtime exited before answering")
        assert isinstance(value, dict)
        return value

    def _read_stdout(self, stream: Any) -> None:
        for line in iter(stream.readline, b""):
            try:
                frame = json.loads(line)
            except (TypeError, ValueError):
                continue
            if not isinstance(frame, dict):
                continue
            if isinstance(frame.get("method"), str) and "id" not in frame:
                self._dispatch_notification(frame)
            else:
                self._response_queue.put(frame)
        self._response_queue.put(_TERMINAL)
        self._fail_subscriptions(self._closed_error("SDK runtime stdout closed"))

    def _read_stderr(self, stream: Any) -> None:
        for line in iter(stream.readline, b""):
            text = line.decode("utf-8", errors="replace").strip()
            if text:
                self._stderr_tail.append(text)

    def _dispatch_notification(self, notification: JsonObject) -> None:
        method = notification.get("method")
        params = notification.get("params")
        if method == "subagent.started" and isinstance(params, dict):
            parent_id = params.get("parentSessionId")
            child_id = params.get("childSessionId")
            if isinstance(parent_id, str) and isinstance(child_id, str) and parent_id != child_id:
                with self._state_lock:
                    self._session_parents[child_id] = parent_id
        with self._state_lock:
            self._notification_backlog.append(notification)
            subscriptions = tuple(self._subscriptions)
        for subscription in subscriptions:
            subscription._push(notification)

    def _remove_subscription(self, subscription: NotificationSubscription) -> None:
        with self._state_lock:
            self._subscriptions.discard(subscription)

    def _fail_subscriptions(self, error: BaseException) -> None:
        with self._state_lock:
            subscriptions = tuple(self._subscriptions)
        for subscription in subscriptions:
            subscription._fail(error)

    def _is_descendant(self, session_id: str, root_id: str) -> bool:
        visited: set[str] = set()
        current = session_id
        while current not in visited:
            if current == root_id:
                return True
            visited.add(current)
            parent = self._session_parents.get(current)
            if parent is None:
                return False
            current = parent
        return False

    def _deadline(self, timeout: float | None) -> float | None:
        return None if timeout is None else time.monotonic() + timeout

    @staticmethod
    def _remaining(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())

    def _closed_error(self, prefix: str) -> TransportClosedError:
        detail = "\n".join(self._stderr_tail)
        process = self._process
        code = process.poll() if process is not None else None
        suffix = f"\nexit code: {code}" if code is not None else ""
        if detail:
            suffix += f"\nstderr tail:\n{detail}"
        return TransportClosedError(f"{prefix}{suffix}")


class DeepSeekHarnessProcess:
    """High-level synchronous SDK that drives a separate Python runtime."""

    def __init__(
        self,
        *,
        command: Sequence[str] | None = None,
        cwd: str | os.PathLike[str] | None = None,
        provider: str = "deepseek-official",
        model: str = "deepseek-v4-flash",
        max_tokens: int | None = None,
        env: dict[str, str] | None = None,
        request_timeout_seconds: float | None = 300.0,
        shutdown_timeout_seconds: float = 5.0,
    ) -> None:
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.cwd = str(Path(cwd or Path.cwd()).expanduser().resolve())
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self._launch = {
            "command": tuple(
                command or (sys.executable, "-m", "deepseek_harness.cli", "sdk-server")
            ),
            "cwd": cwd,
            "env": env,
            "request_timeout_seconds": request_timeout_seconds,
            "shutdown_timeout_seconds": shutdown_timeout_seconds,
        }
        self.client = self._new_client()
        self._initialized = False
        self._closed = False

    def _new_client(self) -> HarnessClient:
        return HarnessClient(**self._launch)

    def start(self) -> None:
        """Start the child and complete the initialize handshake."""

        if self._closed:
            raise TransportClosedError("DeepSeek Harness process SDK is closed")
        if self._initialized:
            return
        try:
            self.client.initialize(
                cwd=self.cwd,
                provider=self.provider,
                model=self.model,
                max_tokens=self.max_tokens,
            )
        except Exception:
            self.client.close()
            if not self._closed:
                self.client = self._new_client()
            raise
        self._initialized = True

    def run(
        self,
        input: str | list[JsonObject],
        *,
        session_id: str | None = None,
        on_notification: NotificationCallback | None = None,
    ) -> ProcessRunResult:
        """Queue one prompt and collect its activity through the next idle state."""

        self.start()
        target = session_id or f"session-{uuid.uuid4().hex}"
        content = [{"type": "text", "text": input}] if isinstance(input, str) else input
        subscription = self.client.subscribe_session_tree(target)
        try:
            message_id = self.client.prompt(target, content)
            deadline = self.client._deadline(self.client.request_timeout_seconds)
            notifications: list[JsonObject] = []
            events: list[JsonObject] = []
            received = False
            while True:
                notification = subscription.next(timeout=self.client._remaining(deadline))
                if not received:
                    if not self._is_inbox_receipt(notification, target, message_id):
                        continue
                    received = True
                self._consume_notification(
                    notification,
                    target,
                    events,
                    notifications,
                    on_notification,
                )
                if self._is_idle(notification, target):
                    break
        finally:
            subscription.close()
        return ProcessRunResult(
            target,
            self._final_response(events),
            tuple(events),
            tuple(notifications),
        )

    def session(self, session_id: str | None = None) -> ProcessSession:
        """Return a stable handle for repeatedly continuing one session."""

        return ProcessSession(self, session_id or f"session-{uuid.uuid4().hex}")

    def start_session(self, session_id: str | None = None) -> ProcessSession:
        """Compatibility alias for the in-process SDK's session factory."""

        return self.session(session_id)

    def close(self) -> None:
        """Close the child process; this instance is terminal after close."""

        if self._closed:
            return
        self._closed = True
        self.client.close()

    def __enter__(self) -> DeepSeekHarnessProcess:
        self.start()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    @staticmethod
    def _is_inbox_receipt(notification: JsonObject, session_id: str, message_id: str) -> bool:
        if notification.get("method") != "session.event":
            return False
        params = notification.get("params")
        if not isinstance(params, dict) or params.get("sessionId") != session_id:
            return False
        event = params.get("event")
        if not isinstance(event, dict) or event.get("type") != "agent/inbox/spliced":
            return False
        data = event.get("data")
        inserted = data.get("inserted") if isinstance(data, dict) else None
        return isinstance(inserted, list) and any(
            isinstance(message, dict) and message.get("id") == message_id
            for message in inserted
        )

    @staticmethod
    def _is_idle(notification: JsonObject, session_id: str) -> bool:
        if notification.get("method") != "session.status":
            return False
        params = notification.get("params")
        return (
            isinstance(params, dict)
            and params.get("sessionId") == session_id
            and params.get("status") == "idle"
        )

    @classmethod
    def _consume_notification(
        cls,
        notification: JsonObject,
        session_id: str,
        events: list[JsonObject],
        notifications: list[JsonObject],
        on_notification: NotificationCallback | None,
    ) -> None:
        notifications.append(notification)
        if on_notification is not None:
            on_notification(notification)
        if notification.get("method") != "session.event":
            return
        params = notification.get("params")
        if not isinstance(params, dict) or params.get("sessionId") != session_id:
            return
        event = params.get("event")
        cls._validate_event(event)
        assert isinstance(event, dict)
        events.append(event)

    @staticmethod
    def _validate_event(event: Any) -> None:
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise SdkProtocolError(f"session.event carried no event envelope: {event!r}")
        if event.get("type") != "assistant/message":
            return
        data = event.get("data")
        message = data.get("message") if isinstance(data, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list) or not all(
            isinstance(block, dict) and isinstance(block.get("type"), str) for block in content
        ):
            raise SdkProtocolError(f"assistant/message carried malformed content: {event!r}")

    @staticmethod
    def _final_response(events: list[JsonObject]) -> str:
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


class ProcessSession:
    """Stable session handle backed by a :class:`DeepSeekHarnessProcess`."""

    def __init__(self, harness: DeepSeekHarnessProcess, session_id: str) -> None:
        self.harness = harness
        self.id = session_id

    def run(
        self,
        input: str | list[JsonObject],
        *,
        on_notification: NotificationCallback | None = None,
    ) -> ProcessRunResult:
        return self.harness.run(input, session_id=self.id, on_notification=on_notification)


__all__ = [
    "DeepSeekHarnessProcess",
    "HarnessClient",
    "JsonRpcResponseError",
    "NotificationSubscription",
    "ProcessSession",
    "ProcessRunResult",
    "RequestTimeoutError",
    "SdkProtocolError",
    "TransportClosedError",
]
