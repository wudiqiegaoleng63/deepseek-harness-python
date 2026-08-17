from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from deepseek_harness.agent import Agent
from deepseek_harness.checkpoint import SessionCheckpointPolicy
from deepseek_harness.llm.types import LlmRequest, StreamChunk
from deepseek_harness.models import Message
from deepseek_harness.session import Session
from deepseek_harness.tools.registry import ToolDefinition, ToolRegistry, ToolResult


class CheckpointAdapter:
    def __init__(self, checkpoints: list[tuple[str, int]]) -> None:
        self.calls = 0
        self.checkpoints = checkpoints

    async def stream(self, request: LlmRequest) -> AsyncIterator[StreamChunk]:
        del request
        self.calls += 1
        if self.calls == 1:
            yield StreamChunk(
                kind="tool-call-delta",
                index=0,
                call_id="call-1",
                name="lookup",
                arguments="{}",
            )
            yield StreamChunk(kind="done", finish_reason="tool_calls")
            return
        yield StreamChunk(kind="text", text="done")
        yield StreamChunk(kind="done", finish_reason="stop")

    async def aclose(self) -> None:
        return None


def test_checkpoint_policy_flushes_before_model_and_tool_side_effects() -> None:
    async def scenario() -> None:
        checkpoints: list[tuple[str, int]] = []
        session = Session("checkpoint-session")

        async def flush(current: Session) -> None:
            checkpoints.append((current.events[-1].type, current.seq))

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                "lookup",
                "Return a value.",
                {"type": "object", "additionalProperties": False},
                lambda _args, _context: ToolResult("value"),
            )
        )
        agent = Agent(
            session,
            CheckpointAdapter(checkpoints),
            tools=registry,
            checkpoint_policy=SessionCheckpointPolicy(flush),
        )

        result = await agent.run(Message("user", (), {"kind": "user"}))

        assert result.final_response == "done"
        assert [event_type for event_type, _seq in checkpoints] == [
            "step/start",
            "tool/call",
            "step/start",
        ]
        await agent.dispose()

    asyncio.run(scenario())


def test_checkpoint_failure_is_fail_closed_before_model_dispatch() -> None:
    async def scenario() -> None:
        calls = 0

        class Adapter:
            async def stream(self, request: LlmRequest) -> AsyncIterator[StreamChunk]:
                nonlocal calls
                del request
                calls += 1
                yield StreamChunk(kind="text", text="must not run")

            async def aclose(self) -> None:
                return None

        async def fail(_session: Session) -> None:
            raise OSError("disk unavailable")

        agent = Agent(
            Session("checkpoint-failure"),
            Adapter(),
            checkpoint_policy=SessionCheckpointPolicy(fail),
        )
        with pytest.raises(OSError, match="disk unavailable"):
            await agent.run("request")
        assert calls == 0
        await agent.dispose()

    asyncio.run(scenario())
