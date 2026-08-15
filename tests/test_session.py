from __future__ import annotations

import asyncio

from deepseek_harness.models import Message, TextContent, create_user_message
from deepseek_harness.session import JsonlSessionStore, Session


def test_session_projects_model_history_and_round_trips_jsonl(tmp_path) -> None:
    async def scenario() -> None:
        session = Session(
            "session-test", header=Session.header_for("session-test", cwd=str(tmp_path))
        )
        user = create_user_message("hello")
        assistant = Message("assistant", (TextContent("world"),), {"kind": "assistant"})
        session.append("turn/start", {"turn": 1})
        session.append("user/message", {"message": user.to_dict()})
        session.append("assistant/message", {"turn": 1, "step": 1, "message": assistant.to_dict()})
        session.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})

        assert [message.text for message in session.derive_messages()] == ["hello", "world"]
        assert session.last_turn_reason() == "completed"

        store = JsonlSessionStore(tmp_path / "sessions")
        await store.save(session)
        loaded = await store.load(session.id)
        assert loaded.header.cwd == str(tmp_path)
        assert loaded.events == session.events
        assert await store.list_ids() == (session.id,)

    asyncio.run(scenario())
