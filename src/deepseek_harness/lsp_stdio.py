"""Optional stdio language-server provider for the LSP capability seam.

This is intentionally a generic adapter rather than a language-server catalog.  A deployment
supplies an executable and extension map; the provider lazily owns one JSON-RPC process per
workspace, transiently opens the requested UTF-8 document, runs one read-only query, and closes it.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import LspError
from .lsp import (
    LspHover,
    LspHoverResult,
    LspLocation,
    LspLocationsResult,
    LspOperation,
    LspPosition,
    LspProvider,
    LspProviderQuery,
    LspQueryResult,
    LspRange,
    LspRuntime,
)

DEFAULT_MAX_MESSAGE_BYTES = 16_000_000
DEFAULT_MAX_DOCUMENT_BYTES = 4_000_000
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 5.0
DEFAULT_KILL_GRACE_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class LspStdioServerConfig:
    """Launch and protocol bounds for one local language server."""

    command: str
    extension_to_language: Mapping[str, str]
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    initialization_options: Any = None
    configuration: Any = None
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES
    max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES
    shutdown_timeout_seconds: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS
    kill_grace_seconds: float = DEFAULT_KILL_GRACE_SECONDS

    def __post_init__(self) -> None:
        if not isinstance(self.command, str) or not self.command.strip():
            raise ValueError("LSP stdio command must be a non-empty string")
        if not self.extension_to_language:
            raise ValueError("LSP stdio extension_to_language must not be empty")
        if any(not isinstance(value, str) for value in self.args):
            raise ValueError("LSP stdio args must contain strings")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.env.items()
        ):
            raise ValueError("LSP stdio env must map strings to strings")
        for name, value in (
            ("max_message_bytes", self.max_message_bytes),
            ("max_document_bytes", self.max_document_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("shutdown_timeout_seconds", self.shutdown_timeout_seconds),
            ("kill_grace_seconds", self.kill_grace_seconds),
        ):
            if value <= 0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and positive")


class LspStdioProvider(LspProvider):
    """A pooled, workspace-scoped JSON-RPC stdio LSP provider."""

    def __init__(self, provider_id: str, config: LspStdioServerConfig) -> None:
        self.id = provider_id
        self.extension_to_language = dict(config.extension_to_language)
        self._config = config
        self._instances: dict[str, _LspInstance] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._disposed = False

    async def query(self, request: LspProviderQuery) -> LspQueryResult:
        if self._disposed:
            raise LspError("lsp-stdio provider is disposed", code="LSP_DISPOSED")
        workspace, source_path, source_text = _read_workspace_source(
            request.workspace_root,
            request.file_path,
            self._config.max_document_bytes,
        )
        workspace_key = str(workspace)
        lock = self._locks.setdefault(workspace_key, asyncio.Lock())
        async with lock:
            if self._disposed:
                raise LspError("lsp-stdio provider is disposed", code="LSP_DISPOSED")
            instance = self._instances.get(workspace_key)
            if instance is None or instance.dead:
                instance = await self._create_instance(workspace)
                self._instances[workspace_key] = instance
            try:
                return await instance.query(
                    request,
                    source_path.as_uri(),
                    source_text,
                )
            except LspError as exc:
                if exc.code not in {"LSP_TRANSPORT", "LSP_PROCESS_EXITED"}:
                    raise
                await instance.dispose()
                if self._instances.get(workspace_key) is instance:
                    self._instances.pop(workspace_key, None)
                replacement = await self._create_instance(workspace)
                self._instances[workspace_key] = replacement
                return await replacement.query(request, source_path.as_uri(), source_text)

    async def _create_instance(self, workspace: Path) -> _LspInstance:
        if self._disposed:
            raise LspError("lsp-stdio provider is disposed", code="LSP_DISPOSED")
        instance = _LspInstance(workspace, self._config)
        try:
            await instance.start()
        except Exception:
            await instance.dispose()
            raise
        return instance

    async def dispose_all(self) -> None:
        self._disposed = True
        instances = tuple(self._instances.values())
        self._instances.clear()
        await asyncio.gather(
            *(instance.dispose() for instance in instances), return_exceptions=True
        )


def install_lsp_stdio_providers(
    runtime: LspRuntime,
    servers: Mapping[str, LspStdioServerConfig],
) -> list[Callable[[], None]]:
    """Register a validated server table atomically and return its disposers."""

    if not servers:
        raise ValueError("lsp-stdio servers must contain at least one entry")
    providers = [LspStdioProvider(provider_id, config) for provider_id, config in servers.items()]
    disposers: list[Callable[[], None]] = []
    try:
        for provider in providers:
            disposers.append(runtime.register_provider(provider))
    except Exception:
        for dispose in reversed(disposers):
            dispose()
        raise
    return disposers


class _LspInstance:
    def __init__(self, workspace: Path, config: LspStdioServerConfig) -> None:
        self.workspace = workspace
        self.config = config
        self.workspace_uri = workspace.as_uri()
        self.process: asyncio.subprocess.Process | None = None
        self.capabilities: dict[str, Any] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._next_id = 1
        self._closed = False
        self._dispose_lock = asyncio.Lock()

    @property
    def dead(self) -> bool:
        process = self.process
        return self._closed or process is None or process.returncode is not None

    async def start(self) -> None:
        environment = os.environ.copy()
        environment.update(self.config.env)
        try:
            self.process = await asyncio.create_subprocess_exec(
                self.config.command,
                *self.config.args,
                cwd=str(self.workspace),
                env=environment,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise LspError(f"failed to start LSP server: {exc}", code="LSP_PROCESS_START") from exc
        self._reader_task = asyncio.create_task(self._read_loop())
        initialize = await self._request(
            "initialize",
            {
                "processId": None,
                "rootUri": self.workspace_uri,
                "workspaceFolders": [
                    {"uri": self.workspace_uri, "name": self.workspace.name or "workspace"}
                ],
                "capabilities": {
                    "general": {"positionEncodings": ["utf-16"]},
                    "workspace": {"configuration": True, "workspaceFolders": True},
                    "textDocument": {
                        "definition": {"linkSupport": True},
                        "implementation": {"linkSupport": True},
                        "hover": {"contentFormat": ["markdown", "plaintext"]},
                    },
                },
                "initializationOptions": self.config.initialization_options,
            },
        )
        if not isinstance(initialize, dict):
            raise LspError("LSP initialize result was malformed", code="LSP_MALFORMED_RESPONSE")
        raw_capabilities = initialize.get("capabilities")
        if not isinstance(raw_capabilities, dict):
            raise LspError(
                "LSP initialize result had no capabilities",
                code="LSP_MALFORMED_RESPONSE",
            )
        self.capabilities = raw_capabilities
        await self._notify("initialized", {})

    async def query(
        self,
        request: LspProviderQuery,
        document_uri: str,
        source_text: str,
    ) -> LspQueryResult:
        self._assert_live()
        if not _supports_operation(self.capabilities, request.operation):
            raise LspError(
                f"LSP server does not support {request.operation}",
                code="LSP_UNSUPPORTED_OPERATION",
            )
        if not _supports_transient_open(self.capabilities.get("textDocumentSync")):
            raise LspError(
                "LSP server does not support transient document open/close",
                code="LSP_UNSUPPORTED_SERVER",
            )
        text_document = {"uri": document_uri, "languageId": request.language_id, "version": 1}
        await self._notify(
            "textDocument/didOpen",
            {"textDocument": {**text_document, "text": source_text}},
        )
        try:
            params: dict[str, Any] = {
                "textDocument": {"uri": document_uri},
                "position": {
                    "line": request.position.line,
                    "character": request.position.character,
                },
            }
            if request.operation == "findReferences":
                params["context"] = {"includeDeclaration": True}
            payload = await self._request(_request_method(request.operation), params)
            if request.operation == "hover":
                return LspHoverResult("hover", normalize_hover(payload))
            return LspLocationsResult(
                "locations",
                tuple(normalize_locations(payload)),
                self.workspace_uri,
            )
        finally:
            try:
                await self._notify("textDocument/didClose", {"textDocument": {"uri": document_uri}})
            except LspError:
                await self.dispose()
                raise

    def _assert_live(self) -> None:
        if self.dead:
            raise LspError("LSP server process exited", code="LSP_PROCESS_EXITED")

    async def _request(self, method: str, params: Any) -> Any:
        self._assert_live_for_write()
        request_id = self._next_id
        self._next_id += 1
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = future
        try:
            await self._send(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            )
            return await future
        finally:
            self._pending.pop(request_id, None)

    async def _notify(self, method: str, params: Any) -> None:
        self._assert_live_for_write()
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _assert_live_for_write(self) -> None:
        if self.dead or self.process is None or self.process.stdin is None:
            raise LspError("LSP server transport is unavailable", code="LSP_TRANSPORT")

    async def _send(self, message: dict[str, Any]) -> None:
        process = self.process
        if process is None or process.stdin is None:
            raise LspError("LSP server transport is unavailable", code="LSP_TRANSPORT")
        body = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(body) > self.config.max_message_bytes:
            raise LspError(
                "outgoing LSP message exceeds the configured limit", code="LSP_MESSAGE_TOO_LARGE"
            )
        frame = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body
        async with self._write_lock:
            try:
                process.stdin.write(frame)
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionError) as exc:
                self._fail_pending(LspError("LSP server transport failed", code="LSP_TRANSPORT"))
                raise LspError("LSP server transport failed", code="LSP_TRANSPORT") from exc

    async def _read_loop(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        try:
            while True:
                headers = await process.stdout.readuntil(b"\r\n\r\n")
                length = _content_length(headers)
                if length > self.config.max_message_bytes:
                    raise LspError(
                        "incoming LSP message exceeds the configured limit",
                        code="LSP_MESSAGE_TOO_LARGE",
                    )
                body = await process.stdout.readexactly(length)
                try:
                    message = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise LspError(
                        "LSP server returned malformed JSON", code="LSP_MALFORMED_RESPONSE"
                    ) from exc
                await self._handle_message(message)
        except asyncio.CancelledError:
            raise
        except (asyncio.IncompleteReadError, ConnectionError):
            self._fail_pending(LspError("LSP server process exited", code="LSP_PROCESS_EXITED"))
        except LspError as exc:
            self._fail_pending(exc)
        finally:
            self._closed = True

    async def _handle_message(self, message: Any) -> None:
        if not isinstance(message, dict):
            raise LspError("LSP server message was not an object", code="LSP_MALFORMED_RESPONSE")
        if "id" in message and ("result" in message or "error" in message):
            request_id = message.get("id")
            if isinstance(request_id, int):
                future = self._pending.get(request_id)
                if future is not None and not future.done():
                    error = message.get("error")
                    if isinstance(error, dict):
                        future.set_exception(
                            LspError(
                                str(error.get("message", "LSP server request failed")),
                                code="LSP_SERVER_ERROR",
                            )
                        )
                    else:
                        future.set_result(message.get("result"))
            return
        method = message.get("method")
        if isinstance(method, str) and "id" in message:
            asyncio.create_task(
                self._answer_server_request(message["id"], method, message.get("params"))
            )

    async def _answer_server_request(self, request_id: Any, method: str, params: Any) -> None:
        if method == "workspace/configuration":
            items = params.get("items", []) if isinstance(params, dict) else []
            result = (
                [self.config.configuration for _item in items] if isinstance(items, list) else []
            )
            await self._send_response(request_id, result)
            return
        if method == "workspace/workspaceFolders":
            await self._send_response(
                request_id,
                [{"uri": self.workspace_uri, "name": self.workspace.name or "workspace"}],
            )
            return
        if method in {
            "client/registerCapability",
            "client/unregisterCapability",
            "window/workDoneProgress/create",
        }:
            await self._send_response(request_id, None)
            return
        if method == "workspace/applyEdit":
            await self._send_response(
                request_id, {"applied": False, "failureReason": "read-only LSP host"}
            )
            return
        await self._send_error(request_id, -32601, f"unsupported LSP server request: {method}")

    async def _send_response(self, request_id: Any, result: Any) -> None:
        await self._send({"jsonrpc": "2.0", "id": request_id, "result": result})

    async def _send_error(self, request_id: Any, code: int, message: str) -> None:
        await self._send(
            {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
        )

    def _fail_pending(self, error: Exception) -> None:
        for future in tuple(self._pending.values()):
            if not future.done():
                future.set_exception(error)

    async def dispose(self) -> None:
        async with self._dispose_lock:
            if self._closed and self.process is None:
                return
            process = self.process
            if process is None:
                self._closed = True
                return
            try:
                if process.returncode is None and process.stdin is not None:
                    try:
                        await asyncio.wait_for(
                            self._request("shutdown", None), self.config.shutdown_timeout_seconds
                        )
                        await self._notify("exit", None)
                    except Exception:
                        pass
                if process.returncode is None:
                    try:
                        await asyncio.wait_for(process.wait(), self.config.kill_grace_seconds)
                    except TimeoutError:
                        process.terminate()
                        try:
                            await asyncio.wait_for(process.wait(), self.config.kill_grace_seconds)
                        except TimeoutError:
                            process.kill()
                            await process.wait()
            finally:
                self._closed = True
                self._fail_pending(LspError("LSP server disposed", code="LSP_DISPOSED"))
                reader = self._reader_task
                if (
                    reader is not None
                    and not reader.done()
                    and reader is not asyncio.current_task()
                ):
                    reader.cancel()
                    await asyncio.gather(reader, return_exceptions=True)
                self.process = None


def _content_length(headers: bytes) -> int:
    for line in headers.decode("ascii", errors="replace").splitlines():
        name, separator, value = line.partition(":")
        if separator and name.casefold() == "content-length":
            try:
                length = int(value.strip())
            except ValueError as exc:
                raise LspError(
                    "LSP Content-Length was invalid", code="LSP_MALFORMED_RESPONSE"
                ) from exc
            if length < 0:
                raise LspError("LSP Content-Length was negative", code="LSP_MALFORMED_RESPONSE")
            return length
    raise LspError("LSP response omitted Content-Length", code="LSP_MALFORMED_RESPONSE")


def _request_method(operation: LspOperation) -> str:
    return {
        "goToDefinition": "textDocument/definition",
        "findReferences": "textDocument/references",
        "goToImplementation": "textDocument/implementation",
        "hover": "textDocument/hover",
    }[operation]


def _supports_operation(capabilities: Mapping[str, Any], operation: LspOperation) -> bool:
    key = {
        "goToDefinition": "definitionProvider",
        "findReferences": "referencesProvider",
        "goToImplementation": "implementationProvider",
        "hover": "hoverProvider",
    }[operation]
    value = capabilities.get(key)
    return value is not None and value is not False


def _supports_transient_open(value: Any) -> bool:
    if isinstance(value, int) and not isinstance(value, bool):
        return value in {1, 2}
    return isinstance(value, dict) and value.get("openClose") is True


def _read_workspace_source(
    workspace_root: str,
    file_path: str,
    max_document_bytes: int,
) -> tuple[Path, Path, str]:
    workspace = Path(workspace_root).expanduser().resolve()
    if not workspace.is_dir():
        raise LspError(f"workspace is not a directory: {workspace}", code="LSP_WORKSPACE_REQUIRED")
    requested = Path(file_path).expanduser()
    target = (requested if requested.is_absolute() else workspace / requested).resolve()
    try:
        target.relative_to(workspace)
    except ValueError as exc:
        raise LspError(
            "LSP source is outside the workspace", code="LSP_SOURCE_OUTSIDE_WORKSPACE"
        ) from exc
    if not target.is_file():
        raise LspError(f"LSP source is not a regular file: {target}", code="LSP_SOURCE_INVALID")
    try:
        data = target.read_bytes()
    except OSError as exc:
        raise LspError(f"failed to read LSP source: {exc}", code="LSP_SOURCE_READ") from exc
    if len(data) > max_document_bytes:
        raise LspError(
            "LSP source exceeds the configured document limit", code="LSP_DOCUMENT_TOO_LARGE"
        )
    try:
        return workspace, target, data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LspError("LSP source is not valid UTF-8", code="LSP_SOURCE_ENCODING") from exc


def normalize_locations(payload: Any) -> tuple[LspLocation, ...]:
    """Normalize Location and LocationLink response forms."""

    if payload is None:
        return ()
    elements = payload if isinstance(payload, list) else [payload]
    locations: list[LspLocation] = []
    for element in elements:
        if not isinstance(element, dict):
            raise LspError(
                "LSP navigation result contained a non-object entry", code="LSP_MALFORMED_RESPONSE"
            )
        if isinstance(element.get("targetUri"), str):
            uri = element["targetUri"]
            raw_range = element.get("targetSelectionRange")
        else:
            uri = element.get("uri")
            raw_range = element.get("range")
        if not isinstance(uri, str):
            raise LspError(
                "LSP navigation result omitted a location URI", code="LSP_MALFORMED_RESPONSE"
            )
        locations.append(LspLocation(uri, _range(raw_range)))
    return tuple(locations)


def normalize_hover(payload: Any) -> LspHover | None:
    """Normalize MarkupContent, MarkedString, and arrays into the seam hover type."""

    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise LspError("LSP hover result was not an object", code="LSP_MALFORMED_RESPONSE")
    contents = _hover_contents(payload.get("contents"))
    if not contents:
        return None
    raw_range = payload.get("range")
    return LspHover(contents, None if raw_range is None else _range(raw_range))


def _range(value: Any) -> LspRange:
    if not isinstance(value, dict):
        raise LspError("LSP response contained a malformed range", code="LSP_MALFORMED_RESPONSE")
    return LspRange(_position(value.get("start")), _position(value.get("end")))


def _position(value: Any) -> LspPosition:
    if not isinstance(value, dict):
        raise LspError("LSP response contained a malformed position", code="LSP_MALFORMED_RESPONSE")
    line = value.get("line")
    character = value.get("character")
    if (
        isinstance(line, bool)
        or isinstance(character, bool)
        or not isinstance(line, int)
        or not isinstance(character, int)
        or line < 0
        or character < 0
    ):
        raise LspError("LSP response contained an invalid position", code="LSP_MALFORMED_RESPONSE")
    return LspPosition(line, character)


def _hover_contents(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n\n".join(_marked_string(item) for item in value)
    if isinstance(value, dict):
        if value.get("kind") in {"markdown", "plaintext"} and isinstance(value.get("value"), str):
            return value["value"]
        if isinstance(value.get("language"), str) and isinstance(value.get("value"), str):
            return _fenced(value["language"], value["value"])
    raise LspError("LSP hover contents were malformed", code="LSP_MALFORMED_RESPONSE")


def _marked_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if (
        isinstance(value, dict)
        and isinstance(value.get("language"), str)
        and isinstance(value.get("value"), str)
    ):
        return _fenced(value["language"], value["value"])
    raise LspError(
        "LSP hover contents contained a malformed MarkedString", code="LSP_MALFORMED_RESPONSE"
    )


def _fenced(language: str, value: str) -> str:
    return f"```{language}\n{value}\n```"


__all__ = [
    "DEFAULT_KILL_GRACE_SECONDS",
    "DEFAULT_MAX_DOCUMENT_BYTES",
    "DEFAULT_MAX_MESSAGE_BYTES",
    "DEFAULT_SHUTDOWN_TIMEOUT_SECONDS",
    "LspStdioProvider",
    "LspStdioServerConfig",
    "install_lsp_stdio_providers",
    "normalize_hover",
    "normalize_locations",
]
