from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator

import pytest

from deepseek_harness.attachments import AttachmentStore
from deepseek_harness.goals import GoalManager
from deepseek_harness.llm.types import LlmRequest, StreamChunk
from deepseek_harness.session import Session
from deepseek_harness.web import HarnessService
from deepseek_harness.web.service import ApiFault

PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="


class ContractAdapter:
    def stream(self, request: LlmRequest) -> AsyncIterator[StreamChunk]:
        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(kind="text", text="ok")
            yield StreamChunk(kind="done", finish_reason="stop")

        return chunks()

    async def aclose(self) -> None:
        return None


def test_attachment_store_is_content_addressed_and_round_trips(tmp_path) -> None:
    store = AttachmentStore(tmp_path)
    data = base64.b64decode(PNG)
    ref = store.save(data, "image/png", name=r"C:\tmp\photo.png")
    assert ref.attachment_id.startswith("sha256:")
    assert ref.width == 1 and ref.height == 1
    assert ref.name == "photo.png"
    assert store.read(ref).data == data
    assert store.save(data, "image/png").attachment_id == ref.attachment_id


def test_goal_manager_persists_cas_revisions_in_session_events() -> None:
    session = Session("goal-test")
    manager = GoalManager(default_max_goal_rounds=3)
    created = manager.create(session, " ship it ", None)
    assert created["revision"] == 1
    edited = manager.edit(session, created, "ship it today", None)
    assert edited["revision"] == 2
    paused = manager.transition(session, "pause", edited)
    resumed = manager.transition(session, "resume", paused)
    assert resumed["revision"] == 4
    projection = manager.fold(session).projection()
    assert projection is not None
    goal = projection["goal"]
    assert isinstance(goal, dict)
    assert goal["phase"] == "active"
    manager.clear(session, resumed)
    assert manager.fold(session).projection() is None


def test_rpc_domains_cover_workspace_goal_attachment_fork_and_subagent(tmp_path) -> None:
    async def scenario() -> None:
        service = HarnessService(
            tmp_path / "state",
            cwd=tmp_path,
            adapter_factory=lambda _model: ContractAdapter(),
        )
        workspace = await service.dispatch("workspace.create", {"path": str(tmp_path)})
        workspace_id = workspace["workspace"]["workspaceId"]
        created = await service.dispatch(
            "session.create",
            {"workspaceId": workspace_id, "sessionId": "contract-session"},
        )
        session_id = created["sessionId"]

        goal = await service.dispatch(
            "goal.create", {"sessionId": session_id, "objective": "test the host"}
        )
        assert goal["ref"]["revision"] == 1

        prompted = await service.dispatch(
            "session.prompt",
            {
                "sessionId": session_id,
                "mode": "queue",
                "content": [{"type": "text", "text": "hello"}],
            },
        )
        assert prompted == {"accepted": True}
        handle = await service.get_session(session_id)
        assert handle.task is not None
        await handle.task

        image_prompt = await service.dispatch(
            "session.prompt",
            {
                "sessionId": session_id,
                "mode": "queue",
                "content": [{"type": "image", "mediaType": "image/png", "data": PNG}],
            },
        )
        assert image_prompt == {"accepted": True}
        handle = await service.get_session(session_id)
        assert handle.task is not None
        await handle.task
        image_id: str | None = None
        for event in handle.session.events:
            if event.type != "user/message":
                continue
            message = event.data.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list) or not content or not isinstance(content[0], dict):
                continue
            block = content[0]
            attachment = block.get("attachment")
            candidate = attachment.get("attachmentId") if isinstance(attachment, dict) else None
            if block.get("type") == "image" and isinstance(candidate, str):
                image_id = candidate
                break
        assert image_id is not None
        attachment = await service.dispatch(
            "session.attachment",
            {"sessionId": session_id, "attachmentId": image_id},
        )
        assert attachment["data"] == PNG

        forked = await service.dispatch("session.fork", {"sessionId": session_id})
        child_id = forked["sessionId"]
        listed = await service.dispatch("subagent.list", {"parentSessionId": session_id})
        assert any(entry["id"] == child_id for entry in listed["entries"])
        child_history = await service.dispatch(
            "subagent.history",
            {
                "parentSessionId": session_id,
                "childSessionId": child_id,
                "mode": "continuable",
            },
        )
        assert child_history["events"]

        await service.dispose()

    asyncio.run(scenario())


def test_settings_and_credentials_survive_service_restart(tmp_path) -> None:
    async def scenario() -> None:
        first = HarnessService(tmp_path / "state", cwd=tmp_path)
        updated = await first.dispatch(
            "settings.update", {"ns": "ui-theme", "patch": {"mode": "dark"}}
        )
        assert updated["value"]["mode"] == "dark"
        await first.dispatch("credentials.set", {"ref": "TEST_DSH_KEY", "value": "secret"})
        await first.dispose()

        second = HarnessService(tmp_path / "state", cwd=tmp_path)
        described = await second.dispatch("settings.describe", {})
        theme = next(item for item in described["namespaces"] if item["ns"] == "ui-theme")
        assert theme["value"]["mode"] == "dark"
        credentials = await second.dispatch("credentials.describe", {"refs": ["TEST_DSH_KEY"]})
        assert credentials["credentials"]["TEST_DSH_KEY"]["configured"] is True
        await second.dispose()

    asyncio.run(scenario())


def test_skill_list_discovers_project_frontmatter_and_filters_model_only_skills(tmp_path) -> None:
    skills_root = tmp_path / ".agents" / "skills"
    (skills_root / "review").mkdir(parents=True)
    (skills_root / "review" / "SKILL.md").write_text(
        "---\n"
        "name: review\n"
        "description: Review changed files\n"
        "whenToUse: before merging\n"
        "---\n\nInstructions\n",
        encoding="utf-8",
    )
    (skills_root / "private").mkdir()
    (skills_root / "private" / "SKILL.md").write_text(
        "---\nname: private\ndescription: Private helper\nuser-invocable: false\n---\n",
        encoding="utf-8",
    )

    async def scenario() -> None:
        service = HarnessService(tmp_path / "state", cwd=tmp_path)
        await service.dispatch(
            "session.create",
            {"sessionId": "skill-session", "cwd": str(tmp_path)},
        )
        value = await service.dispatch("skill.list", {"sessionId": "skill-session"})
        assert value["skills"] == [
            {
                "name": "review",
                "description": "Review changed files",
                "whenToUse": "before merging",
                "modelInvocable": True,
            }
        ]
        await service.dispose()

    asyncio.run(scenario())


def test_rpc_validation_and_projection_baseline_match_wire_contract(tmp_path) -> None:
    async def scenario() -> None:
        service = HarnessService(tmp_path / "state", cwd=tmp_path)
        await service.dispatch("session.create", {"sessionId": "validation", "cwd": str(tmp_path)})
        history = await service.dispatch("session.history", {"sessionId": "validation"})
        assert history["projections"]["asOfSeq"] == -1
        assert history["projections"]["values"]["imageLimits"]["maxImageBytes"] > 0

        with pytest.raises(ApiFault, match="500 characters"):
            await service.dispatch("session.search", {"query": "x" * 501})
        with pytest.raises(ApiFault, match="text content"):
            await service.dispatch(
                "session.prompt",
                {
                    "sessionId": "validation",
                    "mode": "queue",
                    "content": [{"type": "text", "text": 42}],
                },
            )
        await service.dispose()

    asyncio.run(scenario())
