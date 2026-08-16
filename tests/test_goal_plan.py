from __future__ import annotations

import asyncio
import json

from deepseek_harness.models import create_user_message
from deepseek_harness.tools.registry import ToolContext
from deepseek_harness.web import HarnessService


def _open_human_turn(service: HarnessService, session_id: str) -> None:
    session = service._handles[session_id].session
    session.append("turn/start", {"turn": 1})
    session.append(
        "user/message",
        {"message": create_user_message("keep working", source={"kind": "user"}).to_dict()},
    )


def test_model_goal_tools_use_cas_revisions_and_project_full_views(tmp_path) -> None:
    async def scenario() -> None:
        service = HarnessService(tmp_path / "sessions", cwd=tmp_path)
        await service.dispatch("session.create", {"sessionId": "goal-tools", "cwd": str(tmp_path)})
        registry = service._tool_registries["goal-tools"]
        context = ToolContext("goal-tools", str(tmp_path))
        _open_human_turn(service, "goal-tools")

        created = await registry.execute(
            "create_goal",
            json.dumps({"objective": "ship the feature", "max_goal_rounds": 4}),
            context,
        )
        assert not created.is_error
        created_value = json.loads(created.text)
        assert created_value["goal"] == {
            "id": created_value["goal"]["id"],
            "revision": 1,
            "objective": "ship the feature",
            "phase": "active",
            "roundsStarted": 0,
            "maxGoalRounds": 4,
        }
        assert created_value["activation"] == "armed"

        edited = await registry.execute(
            "update_goal",
            json.dumps(
                {
                    "goal_id": created_value["goal"]["id"],
                    "revision": 1,
                    "action": "edit",
                    "objective": "ship the feature today",
                }
            ),
            context,
        )
        assert not edited.is_error
        edited_value = json.loads(edited.text)
        assert edited_value["goal"]["revision"] == 2

        stale = await registry.execute(
            "update_goal",
            json.dumps(
                {
                    "goal_id": created_value["goal"]["id"],
                    "revision": 1,
                    "action": "complete",
                }
            ),
            context,
        )
        assert stale.is_error
        assert stale.meta == {"code": "GOAL_STALE_REVISION"}

        blocked = await registry.execute(
            "update_goal",
            json.dumps(
                {
                    "goal_id": created_value["goal"]["id"],
                    "revision": 2,
                    "action": "blocked",
                    "blocked_reason": "waiting for a human choice",
                }
            ),
            context,
        )
        assert not blocked.is_error
        blocked_value = json.loads(blocked.text)
        assert blocked_value["goal"]["phase"] == "blocked"
        assert blocked_value["goal"]["blockedReason"] == {
            "code": "model-reported",
            "message": "waiting for a human choice",
        }
        assert blocked_value["activation"] == "disarmed"
        await service.dispose()

    asyncio.run(scenario())


def test_plan_command_projection_and_review_tool_share_question_protocol(tmp_path) -> None:
    async def scenario() -> None:
        service = HarnessService(tmp_path / "sessions", cwd=tmp_path)
        await service.dispatch("session.create", {"sessionId": "plan-tools", "cwd": str(tmp_path)})
        registry = service._tool_registries["plan-tools"]
        context = ToolContext("plan-tools", str(tmp_path))

        command = await service.prompt(
            "plan-tools",
            [{"type": "text", "text": "/plan"}],
        )
        assert command["command"]["kind"] == "success"
        assert (await service.history("plan-tools"))["projections"]["values"]["plan"] == {
            "active": True,
            "pending": False,
        }
        handle = await service.get_session("plan-tools")
        system_prompt = handle.agent.system_prompt
        assert callable(system_prompt)
        assert "Plan mode is active" in system_prompt()

        task = asyncio.create_task(
            registry.execute(
                "exit_plan_mode",
                json.dumps({"plan": "# Ship it\n\n1. Test the change."}),
                context,
            )
        )
        while not service._pending_questions:
            await asyncio.sleep(0)
        rpc_id, pending = next(iter(service._pending_questions.items()))
        assert pending.questions[0]["detail"] == "# Ship it\n\n1. Test the change."
        assert pending.questions[0]["intent"] == {"kind": "plan-review", "approve": "Approve"}
        await service.respond(
            {
                "type": "client-response",
                "rpcId": rpc_id,
                "result": {
                    "ok": True,
                    "value": {
                        "sessionId": "plan-tools",
                        "answer": {"answers": [{"id": "plan-review", "selected": ["Approve"]}]},
                    },
                },
            }
        )
        approved = await task
        assert not approved.is_error
        assert approved.meta == {"approved": True}
        assert (await service.history("plan-tools"))["projections"]["values"]["plan"] == {
            "active": False,
            "pending": False,
        }
        await service.dispose()

    asyncio.run(scenario())
