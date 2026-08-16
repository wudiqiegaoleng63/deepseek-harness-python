from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import cast

from deepseek_harness.cli import _run_headless
from deepseek_harness.llm.adapter import LlmAdapter
from deepseek_harness.llm.types import LlmRequest, StreamChunk
from deepseek_harness.tools import PermissionMode


class FullRuntimeAdapter:
    def __init__(self) -> None:
        self.calls = 0
        self.max_tokens: int | None = None

    def stream(self, request: LlmRequest) -> AsyncIterator[StreamChunk]:
        async def chunks() -> AsyncIterator[StreamChunk]:
            self.max_tokens = request.config.max_tokens
            if self.calls == 0:
                self.calls += 1
                yield StreamChunk(
                    kind="tool-call-delta",
                    index=0,
                    call_id="call-goal",
                    name="get_goal",
                    arguments=json.dumps({}),
                )
                yield StreamChunk(kind="done", finish_reason="tool_calls")
                return
            yield StreamChunk(kind="text", text="headless full runtime")
            yield StreamChunk(kind="done", finish_reason="stop")

        return chunks()

    async def aclose(self) -> None:
        return None


def test_headless_uses_the_full_service_tool_runtime(tmp_path) -> None:
    async def scenario() -> None:
        adapter = FullRuntimeAdapter()
        result = await _run_headless(
            "inspect the current goal",
            model="test-model",
            cwd=tmp_path,
            session_root=tmp_path / "sessions",
            max_tokens=321,
            permission_mode=PermissionMode.READ_ONLY,
            adapter_factory=lambda _model: cast(LlmAdapter, adapter),
        )
        assert result.final_response == "headless full runtime"
        assert result.finish_reason == "completed"
        assert adapter.max_tokens == 321
        assert any(
            event.type == "tool/call"
            and event.data.get("name") == "get_goal"
            for event in result.events
        )

    asyncio.run(scenario())
