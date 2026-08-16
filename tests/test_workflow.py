from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import cast

from deepseek_harness.llm.adapter import LlmAdapter
from deepseek_harness.llm.types import LlmRequest, StreamChunk
from deepseek_harness.tools.registry import ToolContext
from deepseek_harness.web import HarnessService
from deepseek_harness.workflow import (
    WorkflowChildResult,
    WorkflowError,
    run_workflow,
)


class WorkflowAdapter:
    def stream(self, _request: LlmRequest) -> AsyncIterator[StreamChunk]:
        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(kind="text", text="child answer")
            yield StreamChunk(kind="done", finish_reason="stop")

        return chunks()

    async def aclose(self) -> None:
        return None


def test_workflow_runner_executes_parallel_pipeline_and_events() -> None:
    async def scenario() -> None:
        events: list[tuple[str, dict[str, object]]] = []
        started = 0

        async def sink(kind: str, data: dict[str, object]) -> None:
            events.append((kind, data))

        async def agent_runner(prompt, label, options, on_started):
            nonlocal started
            started += 1
            child_id = f"child-{started}"
            await on_started(child_id)
            return WorkflowChildResult(child_id, True, f"done:{prompt}")

        result = await run_workflow(
            script=(
                "phase('audit'); log('starting'); "
                "const values = await parallel([() => agent('a'), () => agent('b')]); "
                "const piped = await pipeline([1, 2], async (value) => value + 1, "
                "async (value) => value * 2); "
                "return { argsValue: args.kind, values, piped };"
            ),
            meta={"name": "audit", "description": "test workflow"},
            args={"kind": "unit"},
            agent_runner=agent_runner,
            event_sink=sink,
        )
        assert result.stop_reason == "completed"
        assert result.agents_started == 2
        assert result.value == {
            "argsValue": "unit",
            "values": ["done:a", "done:b"],
            "piped": [4, 6],
        }
        event_kinds = [kind for kind, _data in events]
        assert event_kinds[:2] == ["phase", "log"]
        assert event_kinds.count("agent-start") == 2
        assert event_kinds.count("agent-end") == 2
        assert all(
            data["childId"] in {"child-1", "child-2"}
            for kind, data in events
            if kind == "agent-start"
        )

    asyncio.run(scenario())


def test_workflow_runner_maps_child_failures_to_null_and_fatal_hooks_to_error() -> None:
    async def scenario() -> None:
        async def failed_agent(prompt, label, options, on_started):
            await on_started("failed-child")
            return WorkflowChildResult("failed-child", False, error="child failed")

        failed = await run_workflow(
            script="return await parallel([() => agent('will be null')]);",
            meta={"name": "failures", "description": "test failures"},
            args=None,
            agent_runner=failed_agent,
            event_sink=lambda _kind, _data: asyncio.sleep(0),
        )
        assert failed.stop_reason == "completed"
        assert failed.value == [None]

        fatal = await run_workflow(
            script="return await parallel([() => agent('bad', { nope: true })]);",
            meta={"name": "fatal", "description": "test fatal"},
            args=None,
            agent_runner=failed_agent,
            event_sink=lambda _kind, _data: asyncio.sleep(0),
        )
        assert fatal.stop_reason == "error"
        assert fatal.error is not None
        assert "not recognized" in fatal.error

    asyncio.run(scenario())


def test_workflow_runner_rejects_unsupported_schema_before_child_start() -> None:
    async def scenario() -> None:
        called = False

        async def agent_runner(prompt, label, options, on_started):
            nonlocal called
            called = True
            raise AssertionError("unsupported schema must not start a child")

        result = await run_workflow(
            script=(
                "return await agent('structured', { schema: { type: 'object', "
                "properties: { value: { type: 'string', pattern: 'x' } } } });"
            ),
            meta={"name": "schema", "description": "test schema"},
            args=None,
            agent_runner=agent_runner,
            event_sink=lambda _kind, _data: asyncio.sleep(0),
        )
        assert result.stop_reason == "error"
        assert result.error is not None
        assert "unsupported field" in result.error
        assert called is False

    asyncio.run(scenario())


def test_workflow_error_has_machine_code() -> None:
    error = WorkflowError("bad", "META_INVALID")
    assert error.code == "META_INVALID"


def test_workflow_tool_runs_fresh_agents_and_records_events(tmp_path) -> None:
    async def scenario() -> None:
        service = HarnessService(
            tmp_path / "sessions",
            cwd=tmp_path,
            adapter_factory=lambda _model: cast(LlmAdapter, WorkflowAdapter()),
        )
        await service.dispatch(
            "session.create", {"sessionId": "workflow-root", "cwd": str(tmp_path)}
        )
        handle = await service.get_session("workflow-root")
        handle.session.append("turn/start", {"turn": 1})
        handle.session.append(
            "user/message",
            {
                "message": {
                    "id": "human-message",
                    "role": "user",
                    "content": [{"type": "text", "text": "run a workflow"}],
                    "source": {"kind": "user"},
                }
            },
        )
        result = await service._tool_registries["workflow-root"].execute(
            "workflow",
            json.dumps(
                {
                    "script": (
                        "phase('workers'); "
                        "const values = await parallel(["
                        "() => agent('first', { label: 'first' }), "
                        "() => agent('second', { label: 'second' })"
                        "]); return { values };"
                    ),
                    "meta": {"name": "fan-out", "description": "parallel worker test"},
                }
            ),
            ToolContext("workflow-root", str(tmp_path)),
        )
        assert not result.is_error
        assert "workflow \"fan-out\" completed (2 agents)." in result.text
        assert result.meta is not None
        assert result.meta["agentsStarted"] == 2
        assert result.meta["result"] == {"values": ["child answer", "child answer"]}

        events = handle.session.events
        event_types = [event.type for event in events]
        assert "tool-workflow/run-start" in event_types
        assert "tool-workflow/phase" in event_types
        assert event_types.count("tool-workflow/agent-start") == 2
        assert event_types.count("tool-workflow/agent-end") == 2
        assert event_types[-1] == "tool-workflow/run-end"
        child_ids = {
            str(event.data["childId"])
            for event in events
            if event.type == "tool-workflow/agent-start"
        }
        assert len(child_ids) == 2
        for child_id in child_ids:
            child = await service.get_session(child_id)
            assert child.session.header.parent_session == "workflow-root"
        await service.dispose()

    asyncio.run(scenario())
