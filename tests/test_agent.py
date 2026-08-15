from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from deepseek_harness.agent import Agent
from deepseek_harness.llm.types import LlmRequest, StreamChunk
from deepseek_harness.session import Session
from deepseek_harness.tools.registry import ToolDefinition, ToolRegistry, ToolResult


class ScriptedAdapter:
    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, request: LlmRequest) -> AsyncIterator[StreamChunk]:
        if self.calls == 0:
            self.calls += 1
            yield StreamChunk(
                kind="tool-call-delta", index=0, call_id="call-1", name="lookup", arguments="{}"
            )
            yield StreamChunk(kind="done", finish_reason="tool_calls")
            return
        self.calls += 1
        yield StreamChunk(kind="text", text="done")
        yield StreamChunk(kind="done", finish_reason="stop")

    async def aclose(self) -> None:
        return None


def test_agent_runs_tool_call_then_continues_model_turn() -> None:
    async def scenario() -> None:
        session = Session("session-agent")
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                "lookup",
                "Return a scripted value.",
                {"type": "object", "additionalProperties": False},
                lambda args, context: ToolResult("value"),
            )
        )
        agent = Agent(session, ScriptedAdapter(), tools=registry)
        result = await agent.run("start")
        assert result.final_response == "done"
        assert result.finish_reason == "completed"
        assert [event.type for event in result.events] == [
            "turn/start",
            "user/message",
            "step/start",
            "assistant/chunk",
            "assistant/chunk",
            "assistant/message",
            "step/end",
            "tool/call",
            "tool/result",
            "step/start",
            "assistant/chunk",
            "assistant/chunk",
            "assistant/message",
            "step/end",
            "turn/end",
        ]
        assert [message.text for message in session.derive_messages()] == [
            "start",
            "",
            "value",
            "done",
        ]
        await agent.dispose()

    asyncio.run(scenario())
