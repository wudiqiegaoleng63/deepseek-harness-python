from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from deepseek_harness.code_mode import (
    CodeRuntime,
    CodeRuntimeConfig,
    ToolCallError,
    install_code_tool,
    render_code_sdk,
)
from deepseek_harness.llm.types import LlmRequest, StreamChunk
from deepseek_harness.tools.registry import ToolContext, ToolDefinition, ToolRegistry, ToolResult
from deepseek_harness.web import HarnessService


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="Echo one text value.",
            parameters={"type": "object"},
            execute=lambda args, _ctx: ToolResult(str(args.get("text", ""))),
        )
    )
    return registry


def test_python_code_runtime_bridges_tools_prints_and_returns_json() -> None:
    async def scenario() -> None:
        result = await CodeRuntime().run(
            "value = await tools.echo({'text': 'from tool'})\n"
            "print('answer', value)\n"
            "return {'value': value, 'count': 2}",
            registry=_registry(),
            context=ToolContext("code-session", ".", "call-1"),
        )
        assert result.error is None
        assert result.logs == ("answer from tool",)
        assert result.value == {"value": "from tool", "count": 2}

    asyncio.run(scenario())


def test_python_code_runtime_reports_tool_errors_imports_timeouts_and_output_limits() -> None:
    async def scenario() -> None:
        registry = _registry()
        tool_error = await CodeRuntime().run(
            "try:\n"
            "    await tools.missing({})\n"
            "except ToolCallError as exc:\n"
            "    print(exc.toolName)",
            registry=registry,
            context=ToolContext("code-errors", "."),
        )
        assert tool_error.error is None
        assert tool_error.logs == ("missing",)

        imported = await CodeRuntime().run(
            "import os\nreturn 1",
            registry=registry,
            context=ToolContext("code-errors", "."),
        )
        assert imported.error is not None
        assert "imports are not available" in imported.error.message

        timed_out = await CodeRuntime(CodeRuntimeConfig(max_wall_seconds=0.01)).run(
            "await asyncio.sleep(1)",
            registry=registry,
            context=ToolContext("code-errors", "."),
        )
        assert timed_out.error is not None
        assert timed_out.error.kind == "timeout"

        limited = await CodeRuntime(CodeRuntimeConfig(max_output_bytes=80)).run(
            "print('x' * 200)",
            registry=registry,
            context=ToolContext("code-errors", "."),
        )
        assert limited.error is not None
        assert limited.error.kind == "output-limit"

    asyncio.run(scenario())


def test_run_code_tool_and_sdk_are_registered_with_the_expected_surface() -> None:
    async def scenario() -> None:
        registry = _registry()
        install_code_tool(registry, config=CodeRuntimeConfig(max_wall_seconds=2))
        result = await registry.execute(
            "run_code",
            json.dumps(
                {
                    "code": "return await tools.echo({'text': 'ok'})",
                    "description": "Echo a value",
                }
            ),
            ToolContext("code-tool", ".", "outer"),
        )
        assert result.text == "ok"
        assert not result.is_error
        sdk = render_code_sdk(registry)
        assert "## Writing code for run_code" in sdk
        assert "`echo`: Echo one text value." in sdk
        assert "`run_code`" not in sdk.split("Available tools:", 1)[1]

    asyncio.run(scenario())


def test_service_code_mode_exposes_only_run_code_to_the_model(tmp_path) -> None:
    class Adapter:
        def stream(self, request: LlmRequest) -> AsyncIterator[StreamChunk]:
            del request

            async def chunks() -> AsyncIterator[StreamChunk]:
                yield StreamChunk(kind="text", text="done")
                yield StreamChunk(kind="done", finish_reason="stop")

            return chunks()

        async def aclose(self) -> None:
            return None

    async def scenario() -> None:
        service = HarnessService(
            tmp_path / "state",
            cwd=tmp_path,
            tools_mode="code",
            adapter_factory=lambda _model: Adapter(),
        )
        handle = await service.create_session(session_id="code-mode", cwd=str(tmp_path))
        assert handle.agent.tools.schemas()[0].name == "run_code"
        assert set(handle.agent.tools.names()) > {"run_code"}
        system_prompt = handle.agent.system_prompt
        assert callable(system_prompt)
        assert "## Writing code for run_code" in system_prompt()
        await service.dispose()

    asyncio.run(scenario())


def test_tool_call_error_keeps_the_python_sdk_name() -> None:
    error = ToolCallError("read", "failed")
    assert error.tool_name == "read"
    assert error.toolName == "read"
