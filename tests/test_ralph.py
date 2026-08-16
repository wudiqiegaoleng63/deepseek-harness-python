from __future__ import annotations

import asyncio
import json

from deepseek_harness.llm.types import LlmRequest, StreamChunk
from deepseek_harness.tools.registry import ToolContext
from deepseek_harness.web import HarnessService


class RalphAdapter:
    def __init__(self, counter: dict[str, int]) -> None:
        self.counter = counter

    def stream(self, request: LlmRequest):
        del request
        self.counter["round"] += 1
        round_number = self.counter["round"]
        status = "complete" if round_number == 2 else "continue"
        report = {
            "status": status,
            "summary": f"round {round_number} verified the workspace",
            "evidence": ["workspace inspected"] if status == "complete" else [],
            "nextSteps": [] if status == "complete" else ["continue the implementation"],
            "blocker": "",
        }

        async def chunks():
            yield StreamChunk(kind="text", text=json.dumps(report))
            yield StreamChunk(kind="done", finish_reason="stop")

        return chunks()

    async def aclose(self) -> None:
        return None


def test_ralph_runs_fresh_children_and_records_bounded_handoffs(tmp_path) -> None:
    async def scenario() -> None:
        counter = {"round": 0}
        service = HarnessService(
            tmp_path / "sessions",
            cwd=tmp_path,
            adapter_factory=lambda _model: RalphAdapter(counter),
        )
        await service.dispatch("session.create", {"sessionId": "ralph-root", "cwd": str(tmp_path)})
        handle = await service.get_session("ralph-root")
        handle.session.append("turn/start", {"turn": 1})
        handle.session.append(
            "user/message",
            {
                "message": {
                    "id": "human-message",
                    "role": "user",
                    "content": [{"type": "text", "text": "run Ralph"}],
                    "source": {"kind": "user"},
                }
            },
        )
        result = await service._tool_registries["ralph-root"].execute(
            "ralph",
            json.dumps({"objective": "finish the feature", "maxRounds": 3}),
            ToolContext("ralph-root", str(tmp_path)),
        )
        assert not result.is_error
        assert "Ralph worker reported completion after 2 rounds." in result.text
        assert result.meta is not None
        assert result.meta["agentsStarted"] == 2
        children = [
            event.data["childSessionId"]
            for event in handle.session.events
            if event.type == "subagent/started"
        ]
        assert len(children) == 2
        assert len({str(child) for child in children}) == 2
        event_types = [event.type for event in handle.session.events]
        assert "tool-workflow/run-start" in event_types
        assert "tool-workflow/run-end" in event_types
        await service.dispose()

    asyncio.run(scenario())
