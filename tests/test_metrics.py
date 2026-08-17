from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from deepseek_harness.agent import Agent
from deepseek_harness.llm.types import LlmRequest, StreamChunk, ToolSchema
from deepseek_harness.metrics import (
    context_breakdown_projection,
    context_pressure_projection,
    session_stats_projection,
    token_usage_projection,
)
from deepseek_harness.models import Message, TextContent
from deepseek_harness.session import Session, SessionEvent
from deepseek_harness.web import HarnessService


def event(seq: int, time: int, event_type: str, data: dict) -> SessionEvent:
    return SessionEvent(seq, time, event_type, data)


def test_metrics_folds_usage_pressure_and_session_wall_times() -> None:
    events = [
        event(0, 100, "request/context", {"provider": "test", "model": "m", "contextWindow": 100}),
        event(1, 110, "step/start", {"turn": 1, "step": 1}),
        event(
            2,
            120,
            "assistant/chunk",
            {"turn": 1, "step": 1, "chunk": {"kind": "text", "text": "hi"}},
        ),
        event(3, 125, "assistant/chunk", {
            "turn": 1,
            "step": 1,
            "chunk": {"kind": "done", "usage": {"inputTokens": 20, "outputTokens": 3}},
        }),
        event(4, 130, "assistant/message", {
            "turn": 1,
            "step": 1,
            "usage": {"inputTokens": 20, "outputTokens": 3},
            "message": {"role": "assistant", "content": [], "source": {"kind": "assistant"}},
        }),
        event(5, 140, "tool/call", {"callId": "call-1", "name": "test", "arguments": "{}"}),
        event(6, 150, "tool/result", {
            "message": {
                "role": "tool",
                "content": [{"type": "tool-result", "callId": "call-1", "text": "ok"}],
                "source": {"kind": "tool"},
            },
        }),
        event(7, 160, "step/end", {"turn": 1, "step": 1}),
        event(8, 170, "turn/end", {"turn": 1, "reason": {"kind": "completed"}}),
    ]

    assert token_usage_projection(events) == {
        "uncachedInputTokens": 20,
        "outputTokens": 3,
        "cacheReadTokens": 0,
        "cacheWriteTokens": 0,
    }
    assert context_pressure_projection(events) == {
        "pressureTokens": 20,
        "contextWindow": 100,
    }
    assert session_stats_projection(events) == {
        "turns": 1,
        "steps": 1,
        "llmMs": 20,
        "toolMs": 10,
        "ttftMs": 10,
        "ttftSteps": 1,
        "decodeMs": 10,
        "decodeTokens": 3,
    }


def test_context_breakdown_prices_system_tools_and_surface() -> None:
    messages = (Message("user", (TextContent("hello"),), {"kind": "user"}),)
    value = context_breakdown_projection(
        messages,
        system="system prompt",
        tools=(ToolSchema("read", "read a file", {"type": "object"}),),
    )
    assert value["systemTokens"] > 0
    assert value["toolsTokens"] > 0
    assert value["messageTokens"] > 0


def test_agent_persists_provider_usage_on_assistant_message() -> None:
    class UsageAdapter:
        async def stream(self, request: LlmRequest) -> AsyncIterator[StreamChunk]:
            del request
            yield StreamChunk(kind="text", text="answer")
            yield StreamChunk(
                kind="done",
                finish_reason="stop",
                usage={"inputTokens": 8, "outputTokens": 2},
            )

        async def aclose(self) -> None:
            return None

    async def scenario() -> None:
        session = Session("metrics-agent")
        agent = Agent(session, UsageAdapter())
        await agent.run("question")
        assistant = next(event for event in session.events if event.type == "assistant/message")
        assert assistant.data["usage"] == {"inputTokens": 8, "outputTokens": 2}
        assert any(event.type == "request/header" for event in session.events)
        assert any(event.type == "request/context" for event in session.events)
        assert token_usage_projection(list(session.events)) == {
            "uncachedInputTokens": 8,
            "outputTokens": 2,
            "cacheReadTokens": 0,
            "cacheWriteTokens": 0,
        }
        await agent.dispose()

    asyncio.run(scenario())


def test_web_history_exposes_metric_projections_for_the_shared_ui(tmp_path) -> None:
    class UsageAdapter:
        async def stream(self, request: LlmRequest) -> AsyncIterator[StreamChunk]:
            del request
            yield StreamChunk(kind="text", text="answer")
            yield StreamChunk(
                kind="done",
                finish_reason="stop",
                usage={"inputTokens": 8, "outputTokens": 2},
            )

        async def aclose(self) -> None:
            return None

    async def scenario() -> None:
        service = HarnessService(
            tmp_path / "state",
            cwd=tmp_path,
            adapter_factory=lambda _model: UsageAdapter(),
        )
        handle = await service.create_session(session_id="metrics-web", cwd=str(tmp_path))
        await handle.agent.run("question")
        history = await service.history(handle.session.id)
        values = history["projections"]["values"]
        assert values["tokenUsage"]["outputTokens"] == 2
        assert values["contextPressure"]["pressureTokens"] == 8
        assert values["contextPressure"]["contextWindow"] == 1_000_000
        assert values["contextBreakdown"]["messageTokens"] > 0
        assert values["sessionStats"]["steps"] == 1
        await service.dispose()

    asyncio.run(scenario())
