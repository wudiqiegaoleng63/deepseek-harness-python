from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from deepseek_harness.llm.types import StreamChunk
from deepseek_harness.mcp import (
    McpError,
    McpStdioConfig,
    public_tool_name,
    render_mcp_result,
)
from deepseek_harness.sandbox import UnavailableSandbox
from deepseek_harness.tools import ToolRegistry
from deepseek_harness.web import HarnessService

_SERVER_SCRIPT = """import json, sys

tools = [
    {
        "name": "echo",
        "description": "Echo the provided text back",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {"name": "boom", "description": "Always fails", "inputSchema": {"type": "object"}},
]

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    message = json.loads(line)
    method = message.get("method")
    if message.get("id") is None:
        continue
    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "demo", "version": "1.0"},
        }
    elif method == "tools/list":
        result = {"tools": tools}
    elif method == "tools/call":
        name = message["params"]["name"]
        if name == "boom":
            result = {
                "content": [{"type": "text", "text": "detonated"}],
                "isError": True,
            }
        else:
            args = message["params"].get("arguments", {})
            result = {
                "content": [{"type": "text", "text": f"echo: {args.get('text', '')}"}]
            }
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}), flush=True)
"""


def _write_server(tmp_path: Path) -> Path:
    script = tmp_path / "mcp_server.py"
    script.write_text(_SERVER_SCRIPT, encoding="utf-8")
    return script


def test_public_tool_name_matches_deepseek_contract() -> None:
    assert public_tool_name("demo", "echo") == "mcp__demo__echo"
    lossy = public_tool_name("demo", "my tool.v2")
    assert lossy.startswith("mcp__demo__my_tool_v2-")
    assert len(lossy) <= 64
    assert lossy != public_tool_name("demo", "my_tool_v2")
    long = public_tool_name("demo", "x" * 200)
    assert len(long) == 64


def test_render_mcp_result_folds_text_and_error() -> None:
    rendered = render_mcp_result(
        "echo", {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    )
    assert rendered.text == "a\nb" and not rendered.is_error
    failed = render_mcp_result(
        "boom", {"content": [{"type": "text", "text": "x"}], "isError": True}
    )
    assert failed.is_error
    malformed = render_mcp_result("boom", None)
    assert malformed.is_error


def test_stdio_client_handshake_list_and_call(tmp_path: Path) -> None:
    async def scenario() -> None:
        from deepseek_harness.mcp import connect_mcp_server

        script = _write_server(tmp_path)
        config = McpStdioConfig(
            server_name="demo",
            command=sys.executable,
            args=(str(script),),
            tool_call_timeout_seconds=10,
        )
        client, tools = await connect_mcp_server(config)
        registry = ToolRegistry()
        try:
            assert [tool.name for tool in tools] == ["echo", "boom"]
            assert "Echo the provided text" in tools[0].description
            from deepseek_harness.mcp import install_mcp_tools

            install_mcp_tools(registry, client, tools)
            assert "mcp__demo__echo" in registry.names()
            result = await registry.execute(
                "mcp__demo__echo",
                json.dumps({"text": "hello"}),
                _context("demo-session", tmp_path),
            )
            assert not result.is_error
            assert result.text == "echo: hello"

            failed = await registry.execute(
                "mcp__demo__boom", "{}", _context("demo-session", tmp_path)
            )
            assert failed.is_error
            assert "detonated" in failed.text
        finally:
            await client.close()
        assert not client.connected

    asyncio.run(scenario())


def test_service_registers_mcp_tools_and_cleans_up(tmp_path: Path) -> None:
    async def scenario() -> None:
        script = _write_server(tmp_path)
        config = McpStdioConfig(
            server_name="demo",
            command=sys.executable,
            args=(str(script),),
            tool_call_timeout_seconds=10,
        )
        service = HarnessService(
            tmp_path / "state",
            cwd=tmp_path,
            adapter_factory=lambda _model: _Adapter(),
            sandbox_provider=UnavailableSandbox(),
            mcp_servers=(config,),
        )
        handle = await service.create_session(session_id="mcp", cwd=str(tmp_path))
        try:
            registry = service._tool_registries[handle.session.id]
            assert "mcp__demo__echo" in registry.names()
            result = await registry.execute(
                "mcp__demo__echo",
                json.dumps({"text": "via service"}),
                _context(handle.session.id, tmp_path),
            )
            assert result.text == "echo: via service"
        finally:
            await service.dispose()
        assert service._mcp_clients == {}

    asyncio.run(scenario())


def test_contained_startup_failure_contributes_no_tools(tmp_path: Path) -> None:
    async def scenario() -> None:
        config = McpStdioConfig(
            server_name="missing",
            command="/does/not/exist/mcp-server",
            tool_call_timeout_seconds=5,
        )
        service = HarnessService(
            tmp_path / "state",
            cwd=tmp_path,
            adapter_factory=lambda _model: _Adapter(),
            sandbox_provider=UnavailableSandbox(),
            mcp_servers=(config,),
        )
        handle = await service.create_session(session_id="contained", cwd=str(tmp_path))
        try:
            registry = service._tool_registries[handle.session.id]
            assert not [
                name for name in registry.names() if name.startswith("mcp__")
            ]
        finally:
            await service.dispose()

    asyncio.run(scenario())


def test_fail_on_startup_error_rejects(tmp_path: Path) -> None:
    async def scenario() -> None:
        config = McpStdioConfig(
            server_name="strict",
            command="/does/not/exist/mcp-server",
            fail_on_startup_error=True,
        )
        service = HarnessService(
            tmp_path / "state",
            cwd=tmp_path,
            adapter_factory=lambda _model: _Adapter(),
            sandbox_provider=UnavailableSandbox(),
            mcp_servers=(config,),
        )
        with pytest.raises((McpError, OSError)):
            await service.create_session(session_id="strict", cwd=str(tmp_path))
        await service.dispose()

    asyncio.run(scenario())


def _context(session_id: str, tmp_path: Path):
    from deepseek_harness.tools import ToolContext

    return ToolContext(session_id, str(tmp_path))


class _Adapter:
    def stream(self, request):
        del request

        async def chunks():
            yield StreamChunk(kind="text", text="done")
            yield StreamChunk(kind="done", finish_reason="stop")

        return chunks()

    async def aclose(self) -> None:
        return None
