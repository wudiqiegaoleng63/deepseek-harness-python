from __future__ import annotations

import asyncio
import json

from deepseek_harness.tools.registry import ToolContext
from deepseek_harness.web import HarnessService


def test_todo_write_replaces_projection_and_clears_at_next_turn(tmp_path) -> None:
    async def scenario() -> None:
        service = HarnessService(tmp_path / "sessions", cwd=tmp_path)
        await service.dispatch(
            "session.create", {"sessionId": "todo-session", "cwd": str(tmp_path)}
        )
        registry = service._tool_registries["todo-session"]
        context = ToolContext("todo-session", str(tmp_path))

        first = await registry.execute(
            "todo_write",
            json.dumps(
                {
                    "todos": [
                        {"content": "inspect the runtime", "status": "in_progress"},
                        {"content": "publish the result", "status": "pending"},
                    ]
                }
            ),
            context,
        )
        assert not first.is_error
        assert "1 pending, 1 in progress, 0 completed" in first.text
        history = await service.history("todo-session")
        assert history["projections"]["values"]["todos"] == [
            {"content": "inspect the runtime", "status": "in_progress"},
            {"content": "publish the result", "status": "pending"},
        ]

        second = await registry.execute(
            "todo_write",
            json.dumps({"todos": [{"content": "publish the result", "status": "completed"}]}),
            context,
        )
        assert not second.is_error
        history = await service.history("todo-session")
        assert history["projections"]["values"]["todos"] == [
            {"content": "publish the result", "status": "completed"}
        ]

        handle = await service.get_session("todo-session")
        event = handle.session.append("turn/start", {"turn": 2})
        service._publish_event(handle.session.id, event)
        history = await service.history("todo-session")
        assert history["projections"]["values"]["todos"] is None

        duplicate = await registry.execute(
            "todo_write",
            json.dumps(
                {
                    "todos": [
                        {"content": "same", "status": "pending"},
                        {"content": "same", "status": "completed"},
                    ]
                }
            ),
            context,
        )
        assert duplicate.is_error
        assert "duplicate" in duplicate.text
        await service.dispose()

    asyncio.run(scenario())
