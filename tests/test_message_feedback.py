from __future__ import annotations

import asyncio

from deepseek_harness.models import Message, TextContent
from deepseek_harness.web import HarnessService


def test_message_feedback_remote_uses_compare_and_set_and_survives_restart(tmp_path) -> None:
    async def scenario() -> None:
        root = tmp_path / "sessions"
        service = HarnessService(root, cwd=tmp_path)
        await service.dispatch("session.create", {"sessionId": "feedback", "cwd": str(tmp_path)})
        handle = await service.get_session("feedback")
        assistant = Message("assistant", (TextContent("answer"),), {"kind": "assistant"})
        handle.session.append("assistant/message", {"message": assistant.to_dict()})
        await service.store.save(handle.session)

        empty = await service.dispatch("messageFeedback/list", {"sessionId": "feedback"})
        assert empty == {"ok": True, "value": {"items": []}}
        created = await service.dispatch(
            "messageFeedback/put",
            {
                "sessionId": "feedback",
                "messageId": assistant.id,
                "rating": "positive",
                "note": "clear answer",
                "ifVersion": None,
            },
        )
        assert created["ok"] is True
        item = created["value"]
        assert item["messageId"] == assistant.id
        version = item["version"]

        retry = await service.dispatch(
            "messageFeedback/put",
            {
                "sessionId": "feedback",
                "messageId": assistant.id,
                "rating": "positive",
                "note": "clear answer",
                "ifVersion": version,
            },
        )
        assert retry == created

        conflict = await service.dispatch(
            "messageFeedback/put",
            {
                "sessionId": "feedback",
                "messageId": assistant.id,
                "rating": "negative",
                "ifVersion": None,
            },
        )
        assert conflict["ok"] is False
        assert conflict["error"]["code"] == "version-conflict"

        updated = await service.dispatch(
            "messageFeedback/put",
            {
                "sessionId": "feedback",
                "messageId": assistant.id,
                "rating": "negative",
                "ifVersion": version,
            },
        )
        assert updated["ok"] is True
        await service.dispose()

        restarted = HarnessService(root, cwd=tmp_path)
        listed = await restarted.dispatch("messageFeedback/list", {"sessionId": "feedback"})
        assert listed["value"]["items"][0]["rating"] == "negative"
        deleted = await restarted.dispatch(
            "messageFeedback/delete",
            {
                "sessionId": "feedback",
                "messageId": assistant.id,
                "ifVersion": updated["value"]["version"],
            },
        )
        assert deleted == {"ok": True, "value": {"absent": True}}
        missing = await restarted.dispatch(
            "messageFeedback/put",
            {
                "sessionId": "feedback",
                "messageId": "does-not-exist",
                "rating": "positive",
                "ifVersion": None,
            },
        )
        assert missing["error"]["code"] == "target-not-found"
        await restarted.dispose()

    asyncio.run(scenario())
