"""Out-of-process JSON-RPC protocol for embedding a Harness runtime.

The transport is intentionally small and line-oriented so a caller can launch
``dsh-python sdk-server`` from any language.  The protocol mirrors the TS SDK:
``initialize``, ``session/prompt``, and ``shutdown`` requests plus streamed
session and agent lifecycle notifications.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .llm import DeepSeekAdapter
from .web.service import HarnessService

SDK_SERVER_NAME = "deepseek-harness-sdk-runtime"
SDK_SERVER_VERSION = "0.1.0.dev0"

JsonObject = dict[str, Any]
NotificationSink = Callable[[str, JsonObject], Awaitable[None]]
ServiceFactory = Callable[[JsonObject], HarnessService | Awaitable[HarnessService]]


class JsonRpcResponseError(Exception):
    """An error returned by a JSON-RPC peer."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


class SdkProtocolError(ValueError):
    """A malformed SDK request or response shape."""


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SdkProtocolError(f"{field} must be a positive integer")
    return value


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SdkProtocolError(f"{field} must be a non-empty string")
    return value


class HarnessSdkJsonRpcServer:
    """Serve SDK requests over a caller-owned notification sink."""

    def __init__(
        self,
        service_factory: ServiceFactory,
        *,
        notify: NotificationSink | None = None,
        max_tokens_as_success: bool = False,
    ) -> None:
        self._service_factory = service_factory
        self._notify_sink = notify
        self._max_tokens_as_success = max_tokens_as_success
        self._service: HarnessService | None = None
        self._cwd = ""
        self._provider = "deepseek-official"
        self._model = "deepseek-v4-flash"
        self._max_tokens: int | None = None
        self._forwarders: set[asyncio.Task[None]] = set()
        self._parents: dict[str, str] = {}
        self._finished_children: set[str] = set()
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None

    def set_notification_sink(self, notify: NotificationSink | None) -> None:
        """Replace the notification sink before initialization."""

        if self._service is not None:
            raise RuntimeError("notification sink cannot change after initialize")
        self._notify_sink = notify

    async def handle_request(self, method: str, params: Any = None) -> JsonObject:
        """Dispatch one protocol request and return its JSON object result."""

        if method == "initialize":
            return await self.initialize(params)
        if method == "session/prompt":
            return await self.prompt(params)
        if method == "shutdown":
            return await self.shutdown()
        raise JsonRpcResponseError(-32601, f"unknown DeepSeek Harness SDK method: {method}")

    async def initialize(self, params: Any) -> JsonObject:
        if self._service is not None:
            raise SdkProtocolError("SDK runtime is already initialized")
        if not isinstance(params, dict):
            raise SdkProtocolError("initialize params must be an object")
        cwd = Path(_required_string(params.get("cwd"), "initialize.cwd")).expanduser().resolve()
        if not cwd.is_dir():
            raise SdkProtocolError(f"initialize.cwd is not a directory: {cwd}")
        provider = _required_string(params.get("provider"), "initialize.provider")
        model = _required_string(params.get("model"), "initialize.model")
        max_tokens = params.get("maxTokens")
        if max_tokens is not None:
            max_tokens = _positive_int(max_tokens, "initialize.maxTokens")
        normalized: JsonObject = {
            "cwd": str(cwd),
            "provider": provider,
            "model": model,
        }
        if max_tokens is not None:
            normalized["maxTokens"] = max_tokens
        service = self._service_factory(normalized)
        if inspect.isawaitable(service):
            service = await service
        self._service = service
        self._cwd = str(cwd)
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens
        if max_tokens is not None:
            await service.settings.update("llm-deepseek", {"maxTokens": max_tokens})
        self._start_forwarders()
        return {"serverInfo": {"name": SDK_SERVER_NAME, "version": SDK_SERVER_VERSION}}

    async def prompt(self, params: Any) -> JsonObject:
        service = self._require_service()
        if self._closing:
            raise SdkProtocolError("SDK runtime is shutting down")
        if not isinstance(params, dict):
            raise SdkProtocolError("session/prompt params must be an object")
        session_id = _required_string(params.get("sessionId"), "session/prompt.sessionId")
        content = params.get("contentBlocks")
        if not isinstance(content, list):
            raise SdkProtocolError("session/prompt.contentBlocks must be an array")
        await service.create_session(
            session_id=session_id,
            cwd=self._cwd,
            model_selection={"provider": self._provider, "model": self._model},
        )
        value = await service.prompt(session_id, content, include_message_id=True)
        message_id = value.get("messageId")
        if not isinstance(message_id, str):
            raise RuntimeError("session/prompt did not return a message id")
        return {"messageId": message_id}

    async def shutdown(self) -> JsonObject:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close(), name="dsh-sdk-shutdown")
        await self._close_task
        return {}

    async def close(self) -> None:
        """Close the server when stdin reaches EOF or the host exits."""

        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close(), name="dsh-sdk-close")
        await self._close_task

    def _require_service(self) -> HarnessService:
        if self._service is None:
            raise SdkProtocolError("SDK runtime must be initialized first")
        return self._service

    def _start_forwarders(self) -> None:
        service = self._require_service()
        for kind in ("mux", "host"):
            task = asyncio.create_task(self._forward(kind, service), name=f"dsh-sdk-{kind}")
            self._forwarders.add(task)
            task.add_done_callback(self._forwarders.discard)

    async def _forward(self, kind: str, service: HarnessService) -> None:
        try:
            async for frame in service.stream(kind):
                if kind == "mux":
                    await self._forward_mux(frame)
                else:
                    await self._forward_host(frame)
        except asyncio.CancelledError:
            raise

    async def _forward_mux(self, frame: JsonObject) -> None:
        if frame.get("type") != "session/event":
            return
        session_id = frame.get("sessionId")
        event = frame.get("event")
        if not isinstance(session_id, str) or not isinstance(event, dict):
            return
        await self._notify("session.event", {"sessionId": session_id, "event": event})
        if event.get("type") != "turn/end" or session_id not in self._parents:
            return
        if session_id in self._finished_children:
            return
        self._finished_children.add(session_id)
        data = event.get("data")
        reason = data.get("reason") if isinstance(data, dict) else None
        stop_reason = reason.get("kind") if isinstance(reason, dict) else "error"
        if not isinstance(stop_reason, str):
            stop_reason = "error"
        status = (
            "ok"
            if stop_reason == "completed"
            or (stop_reason == "max-tokens" and self._max_tokens_as_success)
            else "error"
        )
        await self._notify(
            "subagent.finished",
            {
                "provider": "python-in-process",
                "agentId": session_id,
                "parentSessionId": self._parents[session_id],
                "childSessionId": session_id,
                "status": status,
                "stopReason": stop_reason,
            },
        )

    async def _forward_host(self, frame: JsonObject) -> None:
        kind = frame.get("type")
        if kind == "host/session-status":
            session_id = frame.get("sessionId")
            running = frame.get("running")
            if isinstance(session_id, str) and isinstance(running, bool):
                await self._notify(
                    "session.status",
                    {"sessionId": session_id, "status": "running" if running else "idle"},
                )
            return
        if kind != "host/session-added":
            return
        session_id = frame.get("sessionId")
        parent_id = frame.get("parentSessionId")
        if not isinstance(session_id, str) or not isinstance(parent_id, str):
            return
        self._parents[session_id] = parent_id
        await self._notify(
            "subagent.started",
            {"parentSessionId": parent_id, "childSessionId": session_id},
        )

    async def _notify(self, method: str, params: JsonObject) -> None:
        notify = self._notify_sink
        if notify is not None:
            await notify(method, params)

    async def _close(self) -> None:
        self._closing = True
        tasks = tuple(self._forwarders)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        service = self._service
        self._service = None
        if service is not None:
            await service.dispose()


async def serve_stdio(
    server: HarnessSdkJsonRpcServer,
    *,
    input_stream: Any = None,
    output_stream: Any = None,
) -> None:
    """Run a JSON-RPC server over line-buffered binary stdio streams."""

    source = input_stream or sys.stdin.buffer
    target = output_stream or sys.stdout.buffer
    write_lock = asyncio.Lock()
    shutdown_requested = False

    async def write(frame: JsonObject) -> None:
        payload = (json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        async with write_lock:
            await asyncio.to_thread(target.write, payload)
            await asyncio.to_thread(target.flush)

    async def notify(method: str, params: JsonObject) -> None:
        await write({"jsonrpc": "2.0", "method": method, "params": params})

    server.set_notification_sink(notify)
    try:
        while not shutdown_requested:
            raw = await asyncio.to_thread(source.readline)
            if not raw:
                break
            try:
                message = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
                continue
            method = message.get("method")
            if not isinstance(method, str):
                continue
            request_id = message.get("id")
            params = message.get("params")
            try:
                result = await server.handle_request(method, params)
            except JsonRpcResponseError as exc:
                if request_id is not None:
                    await write(
                        {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {"code": exc.code, "message": str(exc), "data": exc.data},
                        }
                    )
            except Exception as exc:
                if request_id is not None:
                    await write(
                        {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {"code": -32603, "message": str(exc)},
                        }
                    )
            else:
                if request_id is not None:
                    await write({"jsonrpc": "2.0", "id": request_id, "result": result})
            shutdown_requested = method == "shutdown"
    finally:
        await server.close()


async def default_sdk_service(
    params: JsonObject,
    *,
    session_root: str | os.PathLike[str] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 120.0,
) -> HarnessService:
    """Create the stock SDK server service from an initialize request."""

    def adapter_factory(_model: str) -> DeepSeekAdapter:
        return DeepSeekAdapter(api_key=api_key, base_url=base_url, timeout=timeout)

    return HarnessService(
        session_root
        or os.getenv("DSH_SESSION_ROOT", "~/.deepseek_harness_python/sessions"),
        cwd=Path(str(params["cwd"])),
        model=str(params["model"]),
        adapter_factory=adapter_factory,
    )


async def run_sdk_server(
    *,
    session_root: str | os.PathLike[str] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 120.0,
) -> None:
    """Launch the stock stdio server used by the CLI entry point."""

    async def factory(params: JsonObject) -> HarnessService:
        return await default_sdk_service(
            params,
            session_root=session_root,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    await serve_stdio(HarnessSdkJsonRpcServer(factory))


__all__ = [
    "HarnessSdkJsonRpcServer",
    "JsonRpcResponseError",
    "SDK_SERVER_NAME",
    "SDK_SERVER_VERSION",
    "SdkProtocolError",
    "default_sdk_service",
    "run_sdk_server",
    "serve_stdio",
]
