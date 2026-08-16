from __future__ import annotations

import asyncio
import json

from deepseek_harness.tools.registry import ToolContext
from deepseek_harness.web import HarnessService


def test_ask_user_question_tool_round_trips_through_pending_mux_protocol(tmp_path) -> None:
    async def scenario() -> None:
        service = HarnessService(tmp_path / "sessions", cwd=tmp_path)
        await service.dispatch(
            "session.create", {"sessionId": "question-session", "cwd": str(tmp_path)}
        )
        registry = service._tool_registries["question-session"]
        task = asyncio.create_task(
            registry.execute(
                "ask_user_question",
                json.dumps(
                    {
                        "questions": [
                            {
                                "id": "mode",
                                "question": "Which mode should I use?",
                                "header": "Choose mode",
                                "options": [
                                    {"label": "safe", "description": "Read-only checks."},
                                    {"label": "fast"},
                                ],
                            }
                        ]
                    }
                ),
                ToolContext("question-session", str(tmp_path)),
            )
        )
        while not service._pending_questions:
            await asyncio.sleep(0)
        rpc_id, pending = next(iter(service._pending_questions.items()))
        assert "multiSelect" not in pending.questions[0]
        assert pending.questions[0]["options"][0]["label"] == "safe"
        response = await service.respond(
            {
                "type": "client-response",
                "rpcId": rpc_id,
                "result": {
                    "ok": True,
                    "value": {
                        "sessionId": "question-session",
                        "answer": {"answers": [{"id": "mode", "selected": ["safe"]}]},
                    },
                },
            }
        )
        assert response == {"accepted": True}
        result = await task
        assert not result.is_error
        assert json.loads(result.text) == {
            "answers": [{"id": "mode", "selected": ["safe"]}]
        }
        await service.dispose()

    asyncio.run(scenario())
