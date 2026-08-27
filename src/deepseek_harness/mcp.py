"""MCP client bridge mirroring the TS ``dsh-mcp-client`` stdio transport.

Connects to an external Model Context Protocol server over stdio and registers
its tools on a :class:`ToolRegistry` under server-qualified public names
(``mcp__<serverName>__<rawName>``).  The raw name is only ever sent on the
wire; the public name is never parsed back.  One shared client per server name
serves every session registry, matching the TS host-plane semantics.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .tools.registry import ToolContext, ToolDefinition, ToolRegistry, ToolResult

MCP_PROTOCOL_VERSION = "2024-11-05"
DEFAULT_TOOL_CALL_TIMEOUT_SECONDS = 60.0
MAX_PUBLIC_NAME_LENGTH = 64
SERVER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
INVALID_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]")
_HASH_LENGTH = 12


class McpError(RuntimeError):
    """An MCP transport or protocol failure."""


@dataclass(slots=True)
class McpStdioConfig:
    """Configuration for one stdio MCP server connection."""

    server_name: str
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""
    tool_call_timeout_seconds: float = DEFAULT_TOOL_CALL_TIMEOUT_SECONDS
    fail_on_startup_error: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.server_name, str) or not SERVER_NAME_PATTERN.match(
            self.server_name
        ):
            raise ValueError(
                "mcp-client: serverName must match [A-Za-z0-9_-]{1,32}: "
                f"{self.server_name!r}"
            )
        if not isinstance(self.command, str) or not self.command:
            raise ValueError("mcp-client: command must be a non-empty string")
        if self.tool_call_timeout_seconds <= 0:
            raise ValueError("mcp-client: toolCallTimeoutMs must be positive")


def public_tool_name(server_name: str, raw_name: str) -> str:
    """Deterministic model-facing name within the DeepSeek function contract."""

    base = f"mcp__{server_name}__{raw_name}"
    normalized = INVALID_NAME_CHARS.sub("_", base)
    if len(normalized) <= MAX_PUBLIC_NAME_LENGTH and normalized == base:
        return normalized
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:_HASH_LENGTH]
    keep = MAX_PUBLIC_NAME_LENGTH - _HASH_LENGTH - 1
    return f"{normalized[:keep]}-{digest}"


@dataclass(slots=True)
class McpTool:
    name: str
    description: str
    input_schema: dict[str, Any]


class McpStdioClient:
    """One JSON-RPC connection to an MCP server child process over stdio."""

    def __init__(self, config: McpStdioConfig) -> None:
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def connected(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def start(self) -> None:
        env = os.environ.copy()
        env.update(self.config.env)
        cwd = self.config.cwd or None
        try:
            self._process = await asyncio.create_subprocess_exec(
                self.config.command,
                *self.config.args,
                cwd=cwd,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError as exc:
            raise McpError(f"mcp-client({self.config.server_name}): spawn failed: {exc}") from exc
        assert self._process is not None and self._process.stdout is not None
        self._reader_task = asyncio.create_task(
            self._read_loop(), name=f"dsh-mcp-read-{self.config.server_name}"
        )
        await self._request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "deepseek-harness-python", "version": "0.1.0"},
            },
        )
        await self._notify("notifications/initialized")

    async def _read_loop(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        while True:
            line = await self._process.stdout.readline()
            if not line:
                break
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if not isinstance(message, dict):
                continue
            self._dispatch(message)
        for future in self._pending.values():
            if not future.done():
                future.set_exception(
                    McpError(f"mcp-client({self.config.server_name}): connection closed")
                )
        self._pending.clear()

    def _dispatch(self, message: dict[str, Any]) -> None:
        if isinstance(message.get("id"), int) and "result" in message or "error" in message:
            future = self._pending.pop(int(message["id"]), None)
            if future is None or future.done():
                return
            if "error" in message:
                error = message.get("error")
                code = error.get("code") if isinstance(error, dict) else None
                text = error.get("message") if isinstance(error, dict) else str(error)
                future.set_exception(
                    McpError(f"mcp-client({self.config.server_name}): {code}: {text}")
                )
            else:
                future.set_result(message.get("result"))

    async def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        if self._process is None or self._process.stdin is None:
            raise McpError(f"mcp-client({self.config.server_name}): not started")
        self._next_id += 1
        request_id = self._next_id
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        line = json.dumps(message, ensure_ascii=False) + "\n"
        assert self._process.stdin is not None
        self._process.stdin.write(line.encode("utf-8"))
        await self._process.stdin.drain()
        return await future

    async def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        if self._process is None or self._process.stdin is None:
            raise McpError(f"mcp-client({self.config.server_name}): not started")
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        assert self._process.stdin is not None
        self._process.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
        await self._process.stdin.drain()

    async def list_tools(self, *, timeout_seconds: float | None = 30.0) -> list[McpTool]:
        tools: list[McpTool] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {} if cursor is None else {"cursor": cursor}
            result = await asyncio.wait_for(
                self._request("tools/list", params), timeout_seconds
            )
            if not isinstance(result, dict):
                raise McpError(f"mcp-client({self.config.server_name}): malformed tools/list")
            for raw in result.get("tools", []):
                if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
                    continue
                schema = raw.get("inputSchema")
                description = raw.get("description")
                tools.append(
                    McpTool(
                        name=raw["name"],
                        description=description if isinstance(description, str) else "",
                        input_schema=schema if isinstance(schema, dict) else {"type": "object"},
                    )
                )
            next_cursor = result.get("nextCursor")
            cursor = next_cursor if isinstance(next_cursor, str) and next_cursor else None
            if cursor is None:
                return tools

    async def call_tool(
        self,
        raw_name: str,
        arguments: dict[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> ToolResult:
        budget = timeout_seconds or self.config.tool_call_timeout_seconds
        try:
            result = await asyncio.wait_for(
                self._request("tools/call", {"name": raw_name, "arguments": arguments}),
                budget,
            )
        except TimeoutError:
            return ToolResult(
                f"mcp tool {raw_name} timed out after {budget:g}s", is_error=True
            )
        except (McpError, asyncio.CancelledError):
            raise
        return render_mcp_result(raw_name, result)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._reader_task is not None:
            self._reader_task.cancel()
            await asyncio.gather(self._reader_task, return_exceptions=True)
            self._reader_task = None
        process = self._process
        self._process = None
        if process is None:
            return
        if process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), 5)
            except TimeoutError:
                process.kill()
                await process.wait()


def render_mcp_result(raw_name: str, result: Any) -> ToolResult:
    """Fold an MCP ``tools/call`` result into a model-facing ToolResult."""

    if not isinstance(result, dict):
        return ToolResult(f"mcp tool {raw_name} returned a malformed result", is_error=True)
    is_error = result.get("isError") is True
    parts: list[str] = []
    content = result.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
    rendered = "\n".join(parts) if parts else "(mcp tool returned no text content)"
    return ToolResult(rendered, is_error=is_error)


async def connect_mcp_server(
    config: McpStdioConfig,
) -> tuple[McpStdioClient, list[McpTool]]:
    """Connect, shake hands, and discover the initial tool list."""

    client = McpStdioClient(config)
    try:
        await client.start()
        tools = await client.list_tools()
    except Exception:
        await client.close()
        raise
    return client, tools


def install_mcp_tools(
    registry: ToolRegistry,
    client: McpStdioClient,
    tools: list[McpTool],
    *,
    resolve_context: Callable[[], ToolContext] | None = None,
) -> list[Callable[[], None]]:
    """Register the server's current tools under public names on one registry."""

    def make_executor(raw_name: str, tool_name: str) -> Any:
        async def execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
            del context
            try:
                return await client.call_tool(raw_name, args)
            except McpError as exc:
                return ToolResult(f"{tool_name} failed: {exc}", is_error=True)

        return execute

    disposers: list[Callable[[], None]] = []
    for tool in tools:
        public = public_tool_name(client.config.server_name, tool.name)
        disposers.append(
            registry.register(
                ToolDefinition(
                    name=public,
                    description=tool.description
                    or f"MCP tool {tool.name} from server {client.config.server_name}",
                    parameters=tool.input_schema,
                    execute=make_executor(tool.name, public),
                    timeout_seconds=client.config.tool_call_timeout_seconds,
                )
            )
        )
    return disposers


async def install_mcp_server(
    registry: ToolRegistry,
    config: McpStdioConfig,
) -> tuple[McpStdioClient | _UnavailableClient, list[Callable[[], None]]]:
    """Connect to one server and register its tools; containment per config.

    A failed startup with ``fail_on_startup_error`` rejects; otherwise the
    server contributes no tools this generation and the caller may retry.
    """

    try:
        client, tools = await connect_mcp_server(config)
    except Exception:
        if config.fail_on_startup_error:
            raise
        return _UnavailableClient(config), []
    disposers = install_mcp_tools(registry, client, tools)
    return client, disposers


class _UnavailableClient:
    """Stand-in for a contained startup failure; closes as a no-op."""

    def __init__(self, config: McpStdioConfig) -> None:
        self.config = config

    async def close(self) -> None:
        return None


__all__ = [
    "DEFAULT_TOOL_CALL_TIMEOUT_SECONDS",
    "MAX_PUBLIC_NAME_LENGTH",
    "MCP_PROTOCOL_VERSION",
    "McpError",
    "McpStdioClient",
    "McpStdioConfig",
    "McpTool",
    "connect_mcp_server",
    "install_mcp_server",
    "install_mcp_tools",
    "public_tool_name",
    "render_mcp_result",
]
