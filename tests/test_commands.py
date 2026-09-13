from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

from deepseek_harness.llm.adapter import LlmAdapter
from deepseek_harness.llm.types import LlmRequest, StreamChunk
from deepseek_harness.web import HarnessService


class QuietAdapter:
    async def stream(self, _request: LlmRequest) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(kind="text", text="model answer")

    async def aclose(self) -> None:
        return None


def service(tmp_path: Path, **kwargs: object) -> HarnessService:
    return HarnessService(
        tmp_path / "sessions",
        cwd=tmp_path,
        adapter_factory=lambda _model: cast(LlmAdapter, QuietAdapter()),
        **kwargs,  # type: ignore[arg-type]
    )


def command_text(result: dict[str, object]) -> str:
    command = result["command"]
    assert isinstance(command, dict)
    return str(command["text"])


def command_kind(result: dict[str, object]) -> str:
    command = result["command"]
    assert isinstance(command, dict)
    return str(command["kind"])


# ----------------------------------------------------------------------- /goal


def test_goal_command_creates_shows_and_edits(tmp_path: Path) -> None:
    async def scenario() -> None:
        harness = service(tmp_path)
        await harness.dispatch("session.create", {"sessionId": "s", "cwd": str(tmp_path)})
        handle = await harness.get_session("s")

        created = await harness.prompt("s", [{"type": "text", "text": "/goal ship the release"}])
        assert command_kind(created) == "success"
        assert command_text(created) == (
            "Goal created\n"
            "Status: active\n"
            "Objective: ship the release\n"
            "Rounds: 0/256\n"
            "Activation: armed\n"
            "\n"
            "Commands: /goal edit <objective>, /goal pause, /goal clear"
        )

        shown = await harness.prompt("s", [{"type": "text", "text": "/goal"}])
        assert command_text(shown).startswith("Goal\nStatus: active\n")

        edited = await harness.prompt("s", [{"type": "text", "text": "/goal edit ship it sooner"}])
        assert command_text(edited).startswith("Goal updated\n")
        assert "Objective: ship it sooner" in command_text(edited)

        # The command exchanges are durable and model-free.
        assert [event.type for event in handle.session.events].count("command/done") == 3
        assert not any(
            event.type == "user/message" and "goal" in str(event.data)
            for event in handle.session.events
        )
        await harness.dispose()

    asyncio.run(scenario())


def test_goal_command_pause_resume_and_clear(tmp_path: Path) -> None:
    async def scenario() -> None:
        harness = service(tmp_path)
        await harness.dispatch("session.create", {"sessionId": "s", "cwd": str(tmp_path)})
        await harness.prompt("s", [{"type": "text", "text": "/goal watch the build"}])

        paused = await harness.prompt("s", [{"type": "text", "text": "/goal PAUSE"}])
        assert command_text(paused) == (
            "Goal paused\n"
            "Status: paused\n"
            "Objective: watch the build\n"
            "Rounds: 0/256\n"
            "Activation: disarmed\n"
            "\n"
            "Commands: /goal edit <objective>, /goal resume, /goal clear"
        )
        resumed = await harness.prompt("s", [{"type": "text", "text": "/goal resume"}])
        assert command_text(resumed).startswith("Goal resumed\n")
        assert "Activation: armed" in command_text(resumed)

        cleared = await harness.prompt("s", [{"type": "text", "text": "/goal clear"}])
        assert command_text(cleared) == "Goal cleared."
        empty = await harness.prompt("s", [{"type": "text", "text": "/goal clear"}])
        assert command_text(empty) == "No goal to clear."
        idle = await harness.prompt("s", [{"type": "text", "text": "/goal"}])
        assert command_text(idle).startswith("No goal is currently set.\nUsage: /goal")
        await harness.dispose()

    asyncio.run(scenario())


def test_goal_command_control_words_are_literal_objectives_within_a_sentence(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        harness = service(tmp_path)
        await harness.dispatch("session.create", {"sessionId": "s", "cwd": str(tmp_path)})
        result = await harness.prompt(
            "s", [{"type": "text", "text": "/goal pause after verification"}]
        )
        assert "Objective: pause after verification" in command_text(result)
        assert "Status: active" in command_text(result)

        replaced = await harness.prompt("s", [{"type": "text", "text": "/goal a new aim"}])
        # An unfinished goal is never replaced without an explicit clear.
        assert command_kind(replaced) == "error"
        assert command_text(replaced) == (
            "A goal is already active. Use /goal edit <objective> to change it "
            "or /goal clear before replacing it."
        )
        awaiting = await harness.prompt("s", [{"type": "text", "text": "/goal edit"}])
        assert command_kind(awaiting) == "error"
        assert command_text(awaiting).startswith("Goal editing requires a replacement objective.")
        await harness.dispose()

    asyncio.run(scenario())


# ------------------------------------------------------------------- /feedback


def test_feedback_command_records_log_only_text(tmp_path: Path, monkeypatch) -> None:
    async def scenario() -> None:
        monkeypatch.setenv("DSH_HOME", str(tmp_path / "home"))
        harness = service(tmp_path)
        await harness.dispatch("session.create", {"sessionId": "s", "cwd": str(tmp_path)})
        handle = await harness.get_session("s")

        result = await harness.prompt(
            "s", [{"type": "text", "text": "/feedback the plan mode felt slow"}]
        )
        assert command_kind(result) == "success"
        recorded = next(event for event in handle.session.events if event.type == "feedback/record")
        assert recorded.data == {"text": "the plan mode felt slow"}
        user_id = (tmp_path / "home" / ".anonymous-user-id").read_text(encoding="utf-8").strip()
        assert command_text(result) == (
            f"Feedback recorded for session s\n"
            f"Anonymous user: {user_id}. Session sharing is not configured."
        )
        # The input is not duplicated into the command record.
        run = next(event for event in handle.session.events if event.type == "command/run")
        assert "args" not in run.data

        empty = await harness.prompt("s", [{"type": "text", "text": "/feedback   "}])
        assert command_kind(empty) == "error"
        assert command_text(empty) == "Feedback text is required. Usage: /feedback <text>"
        assert len([e for e in handle.session.events if e.type == "feedback/record"]) == 1
        await harness.dispose()

    asyncio.run(scenario())


def test_feedback_text_that_looks_like_a_command_is_still_feedback(tmp_path: Path) -> None:
    async def scenario() -> None:
        harness = service(tmp_path)
        await harness.dispatch("session.create", {"sessionId": "s", "cwd": str(tmp_path)})
        handle = await harness.get_session("s")
        await harness.prompt("s", [{"type": "text", "text": "/feedback /plan felt slow"}])
        recorded = next(event for event in handle.session.events if event.type == "feedback/record")
        assert recorded.data == {"text": "/plan felt slow"}
        await harness.dispose()

    asyncio.run(scenario())


def test_anonymous_user_id_is_stable_and_home_scoped(tmp_path: Path) -> None:
    async def scenario() -> None:
        from deepseek_harness.identity import get_or_create_anonymous_user_id

        home = tmp_path / "home"
        first = get_or_create_anonymous_user_id(home)
        assert get_or_create_anonymous_user_id(home) == first
        # A different home is a different identity; deleting the file mints a
        # fresh one on the next process, which this call models directly.
        other = get_or_create_anonymous_user_id(tmp_path / "other-home")
        assert other != first
        assert (tmp_path / "other-home" / ".anonymous-user-id").read_text().strip() == other

    asyncio.run(scenario())


def test_unknown_commands_still_fail(tmp_path: Path) -> None:
    async def scenario() -> None:
        harness = service(tmp_path)
        await harness.dispatch("session.create", {"sessionId": "s", "cwd": str(tmp_path)})
        try:
            await harness.prompt("s", [{"type": "text", "text": "/nope"}])
        except Exception as exc:
            assert "unknown command" in str(exc)
        else:
            raise AssertionError("an unknown command must be rejected")
        await harness.dispose()

    asyncio.run(scenario())
