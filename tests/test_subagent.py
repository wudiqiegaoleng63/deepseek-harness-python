from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import cast

from deepseek_harness.llm.adapter import LlmAdapter
from deepseek_harness.llm.types import LlmRequest, StreamChunk
from deepseek_harness.tools.registry import ToolContext
from deepseek_harness.web import HarnessService


class RepeatingAdapter:
    def __init__(self, text: str = "child answer") -> None:
        self.text = text

    async def stream(self, _request: LlmRequest) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(kind="text", text=self.text)
        yield StreamChunk(kind="done", finish_reason="stop")

    async def aclose(self) -> None:
        return None


def test_subagent_foreground_and_fork_preserve_durable_child_identity(tmp_path) -> None:
    async def scenario() -> None:
        service = HarnessService(
            tmp_path / "sessions",
            cwd=tmp_path,
            adapter_factory=lambda _model: cast(LlmAdapter, RepeatingAdapter()),
        )
        await service.dispatch("session.create", {"sessionId": "parent", "cwd": str(tmp_path)})
        registry = service._tool_registries["parent"]
        context = ToolContext("parent", str(tmp_path))

        foreground = await registry.execute(
            "subagent",
            json.dumps(
                {
                    "description": "answer directly",
                    "prompt": "Return a concise answer.",
                    "run_in_background": False,
                }
            ),
            context,
        )
        assert not foreground.is_error
        assert "child answer" in foreground.text
        assert foreground.meta is not None
        foreground_id = str(foreground.meta["subagentId"])
        foreground_session = await service.get_session(foreground_id)
        assert foreground_session.session.header.origin == "subagent"
        assert any(
            event.type == "subagent/descriptor" for event in foreground_session.session.events
        )

        await service.dispatch(
            "session.prompt",
            {
                "sessionId": "parent",
                "mode": "queue",
                "content": [{"type": "text", "text": "establish fork context"}],
            },
        )
        parent = await service.get_session("parent")
        assert parent.task is not None
        await parent.task

        forked = await registry.execute(
            "subagent_fork",
            json.dumps(
                {
                    "description": "review this conversation",
                    "prompt": "Summarize the inherited context.",
                    "run_in_background": False,
                }
            ),
            context,
        )
        assert not forked.is_error
        assert forked.meta is not None
        fork_id = str(forked.meta["subagentId"])
        fork_session = await service.get_session(fork_id)
        assert (fork_session.session.header.seed_length or 0) > 0
        assert any(event.type == "user/message" for event in fork_session.session.events)
        await service.dispose()

    asyncio.run(scenario())


def test_subagent_background_is_continuable_and_visible_to_control_tools(tmp_path) -> None:
    async def scenario() -> None:
        service = HarnessService(
            tmp_path / "sessions",
            cwd=tmp_path,
            adapter_factory=lambda _model: cast(
                LlmAdapter, RepeatingAdapter("background answer")
            ),
        )
        await service.dispatch("session.create", {"sessionId": "parent", "cwd": str(tmp_path)})
        registry = service._tool_registries["parent"]
        context = ToolContext("parent", str(tmp_path))

        started = await registry.execute(
            "subagent",
            json.dumps({"description": "background work", "prompt": "Work independently."}),
            context,
        )
        assert not started.is_error
        assert started.meta is not None
        child_id = str(started.meta["subagentId"])
        job_id = str(started.meta["jobId"])
        terminal = await service.jobs.wait(job_id, 2_000, "parent")
        assert terminal.status == "completed"
        output = service.jobs.read(job_id, "parent")
        assert "background answer" in output.text

        listed = await registry.execute("list_agents", "{}", context)
        assert not listed.is_error
        assert child_id in listed.text
        assert "background work" in listed.text

        sent = await registry.execute(
            "send_message",
            json.dumps({"subagent_id": child_id, "message": "Now provide a follow-up."}),
            context,
        )
        assert not sent.is_error
        child = await service.get_session(child_id)
        assert child.task is not None
        await child.task
        history = await service.history(child_id)
        assert any(
            event["event"]["type"] == "user/message"
            and event["event"]["data"]["message"]["content"][0]["text"]
            == "Now provide a follow-up."
            for event in history["events"]
        )
        await service.dispose()

    asyncio.run(scenario())
