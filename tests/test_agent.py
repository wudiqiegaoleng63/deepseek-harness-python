from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from deepseek_harness.agent import Agent
from deepseek_harness.compaction import CompactionPolicy
from deepseek_harness.errors import ConfigurationError, LlmError
from deepseek_harness.llm.types import LlmRequest, RetryPolicy, StreamChunk
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
            "request/header",
            "request/context",
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


class RetryAdapter:
    def __init__(self, failure: Exception) -> None:
        self.failure = failure
        self.calls = 0

    async def stream(self, request: LlmRequest) -> AsyncIterator[StreamChunk]:
        del request
        self.calls += 1
        if self.calls == 1:
            raise self.failure
        yield StreamChunk(kind="text", text="recovered")
        yield StreamChunk(kind="done", finish_reason="stop")

    async def aclose(self) -> None:
        return None


def test_agent_retries_transient_llm_failures_with_durable_events() -> None:
    async def scenario() -> None:
        session = Session("session-retry")
        adapter = RetryAdapter(LlmError("temporary outage", code="SERVER", status=503))
        agent = Agent(
            session,
            adapter,
            retry_policy=RetryPolicy(
                initial_delay_seconds=0.001,
                max_delay_seconds=0.001,
                jitter_ratio=0,
            ),
        )
        result = await agent.run("retry this")

        assert adapter.calls == 2
        assert result.final_response == "recovered"
        assert result.finish_reason == "completed"
        retry = [event for event in result.events if event.type == "llm/retry"]
        assert len(retry) == 1
        assert retry[0].data["failure"] == {
            "message": "temporary outage",
            "code": "SERVER",
            "status": 503,
        }
        assert any(event.type == "llm/retry-started" for event in result.events)
        await agent.dispose()

    asyncio.run(scenario())


def test_agent_does_not_retry_configuration_failures() -> None:
    async def scenario() -> None:
        session = Session("session-no-retry")
        adapter = RetryAdapter(ConfigurationError("missing key"))
        agent = Agent(session, adapter)
        result = await agent.run("do not retry")

        assert adapter.calls == 1
        assert result.finish_reason == "error"
        end = next(event for event in result.events if event.type == "turn/end")
        reason = end.data["reason"]
        assert isinstance(reason, dict)
        failure = reason["failure"]
        assert isinstance(failure, dict)
        assert failure["code"] == "CONFIGURATION"
        assert not any(event.type == "llm/retry" for event in result.events)
        await agent.dispose()

    asyncio.run(scenario())


def test_retry_policy_validates_and_honors_provider_delay() -> None:
    with pytest.raises(ValueError, match="positive"):
        RetryPolicy(initial_delay_seconds=0)
    with pytest.raises(ValueError, match="unique"):
        RetryPolicy(retryable_codes=("SERVER", "SERVER"))

    policy = RetryPolicy(initial_delay_seconds=0.5, max_delay_seconds=2, jitter_ratio=0)
    assert policy.delay_seconds(1, retry_after_ms=1250) == 1.25
    assert policy.delay_seconds(3, random_value=0) == 2


class RecordingAdapter:
    def __init__(self) -> None:
        self.requests: list[tuple[str, ...]] = []

    async def stream(self, request: LlmRequest) -> AsyncIterator[StreamChunk]:
        self.requests.append(tuple(message.text for message in request.messages))
        yield StreamChunk(kind="text", text="ack")
        yield StreamChunk(kind="done", finish_reason="stop")

    async def aclose(self) -> None:
        return None


def test_agent_compacts_long_history_with_durable_checkpoint() -> None:
    async def scenario() -> None:
        session = Session("session-compaction")
        adapter = RecordingAdapter()
        agent = Agent(
            session,
            adapter,
            compaction_policy=CompactionPolicy(
                context_window_tokens=300,
                threshold_ratio=0.6,
                retain_ratio=None,
                retain_tokens=70,
                summary_max_chars=300,
            ),
        )

        for index in range(3):
            await agent.run(f"request-{index}: " + ("context " * 90))

        event_types = [event.type for event in session.events]
        assert event_types.count("compaction/start") >= 1
        assert event_types.count("compaction/summary") == event_types.count("compaction/end")
        for summary_index, event_type in enumerate(event_types):
            if event_type != "compaction/summary":
                continue
            assert event_types[summary_index : summary_index + 3] == [
                "compaction/summary",
                "user/message",
                "compaction/end",
            ]

        projected = session.derive_messages()
        checkpoints = [
            message for message in projected if message.source.get("kind") == "compaction"
        ]
        assert len(checkpoints) == 1
        assert "<compacted-summary>" in checkpoints[0].text
        assert "request-0" in checkpoints[0].text
        assert len(projected) < 7
        assert any("request-2" in request for request in adapter.requests[-1])

        await agent.dispose()

    asyncio.run(scenario())
