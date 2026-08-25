from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from deepseek_harness.schedule import (
    MIN_EVERY_INTERVAL_SECONDS,
    ScheduleInputError,
    ScheduleLogError,
    ScheduleManager,
    ScheduleRuntime,
    fold_schedules,
    parse_at_input,
)
from deepseek_harness.session import Session
from deepseek_harness.tools import ToolContext
from deepseek_harness.web import HarnessService


def _now() -> datetime:
    return datetime.now(UTC)


def test_schedule_create_list_delete_round_trip() -> None:
    session = Session("sched-round")
    manager = ScheduleManager()
    now = _now()
    created = manager.create(
        session,
        "check the build",
        after_seconds=30,
        now=now,
    )
    assert created["kind"] == "after"
    assert created["afterSeconds"] == 30
    assert created["state"] == "scheduled"
    assert created["deliveryMode"] == "session-local"
    assert manager.list(session, now=now)[0]["id"] == "sched-1"

    assert manager.delete(session, "sched-1") == {"id": "sched-1", "deleted": True}
    assert manager.delete(session, "sched-1") == {
        "id": "sched-1",
        "deleted": False,
        "code": "schedule_not_found",
    }
    assert manager.list(session, now=now) == []


def test_schedule_validation_rejects_bad_selectors() -> None:
    session = Session("sched-validate")
    manager = ScheduleManager()
    now = _now()
    for kwargs in (
        {"after_seconds": 0},
        {"every_seconds": MIN_EVERY_INTERVAL_SECONDS - 1},
        {"after_seconds": 10, "at": "2099-01-01T00:00:00Z"},
        {},
    ):
        try:
            manager.create(session, "prompt", now=now, **kwargs)
        except ScheduleInputError as exc:
            assert exc.code in {"invalid_rule", "invalid_selector", "frequency_too_high"}
        else:
            raise AssertionError(f"expected rejection for {kwargs}")
    try:
        manager.create(session, "   ", after_seconds=10, now=now)
    except ScheduleInputError as exc:
        assert exc.code == "invalid_prompt"
    else:
        raise AssertionError("expected empty prompt rejection")


def test_schedule_at_parsing_accepts_offsets_and_zones() -> None:
    now = datetime(2026, 8, 25, 12, 0, 0, tzinfo=UTC)
    parsed = parse_at_input("2026-08-25T20:30:00+08:00", now=now)
    assert parsed.hour == 12 and parsed.minute == 30 and parsed.tzinfo is UTC
    local = parse_at_input(
        {"date": "2026-08-25", "time": "20:30:00", "time_zone": "Asia/Shanghai"},
        now=now,
    )
    assert local == parsed
    for value in (
        "2026-08-25T20:00:00",
        "2026-08-25 20:00:00Z",
        {"date": "2026-08-25", "time": "20:00:00"},
        {"date": "2026-08-25", "time": "20:00:00", "time_zone": "Mars/Olympus"},
    ):
        try:
            parse_at_input(value, now=now)
        except ScheduleInputError:
            continue
        raise AssertionError(f"expected rejection for {value}")
    try:
        parse_at_input("2020-01-01T00:00:00Z", now=now)
    except ScheduleInputError as exc:
        assert exc.code == "not_future"
    else:
        raise AssertionError("expected past rejection")


def test_schedule_fold_rejects_corrupt_streams() -> None:
    session = Session("sched-fold")
    manager = ScheduleManager()
    manager.create(session, "once", after_seconds=10, now=_now())
    events = [event for event in session.events if event.type == "schedule/change"]
    duplicate = dict(events[0].data)
    session.append("schedule/change", duplicate)
    try:
        fold_schedules(session.events)
    except ScheduleLogError as exc:
        assert exc.code == "corrupt_schedule_log"
    else:
        raise AssertionError("expected corrupt log rejection")


def test_schedule_runtime_delivers_one_shot_and_every_batch() -> None:
    async def scenario() -> None:
        session = Session("sched-runtime")
        manager = ScheduleManager()
        # Anchor the rule in the past so its first occurrence is overdue now.
        now = _now() - timedelta(seconds=600)
        manager.create(session, "standup", every_seconds=300, now=now)
        sent: list[tuple[str, str]] = []
        persisted: list[str] = []
        runtime = ScheduleRuntime(
            manager,
            sessions=lambda: [session],
            persist=lambda value: persisted.append(value.id) or asyncio.sleep(0),
            send=lambda session_id, text: sent.append((session_id, text)) or asyncio.sleep(0),
        )
        await runtime.tick()

        assert len(sent) == 1
        assert sent[0][0] == "sched-runtime"
        assert "[SCHEDULE REMINDER BATCH]" in sent[0][1]
        payload = json.loads(sent[0][1].split("reminders_json: ", 1)[1])
        assert payload[0]["schedule_id"] == "sched-1"
        assert payload[0]["reminder_prompt"] == "standup"
        # One pre-dispatch checkpoint and one post-append barrier, matching TS.
        assert persisted == ["sched-runtime", "sched-runtime"]

        # The dispatch advanced the anchor past the accepted time; the record
        # is no longer due until the next interval elapses.
        _, every = manager.due(session, now=now + timedelta(seconds=1))
        assert every == []

    asyncio.run(scenario())


def test_service_registers_schedule_tools_for_root_sessions(tmp_path: Path) -> None:
    async def scenario() -> None:
        class Adapter:
            def stream(self, request):
                del request

                async def chunks():
                    yield StreamChunk(kind="text", text="done")
                    yield StreamChunk(kind="done", finish_reason="stop")

                return chunks()

            async def aclose(self) -> None:
                return None

        from deepseek_harness.llm.types import StreamChunk

        service = HarnessService(
            tmp_path / "state",
            cwd=tmp_path,
            adapter_factory=lambda _model: Adapter(),
        )
        handle = await service.create_session(session_id="schedule-root", cwd=str(tmp_path))
        try:
            registry = service._tool_registries[handle.session.id]
            assert {"schedule_create", "schedule_list", "schedule_delete"} <= set(registry.names())
            created = await registry.execute(
                "schedule_create",
                json.dumps({"prompt": "ping", "after_seconds": 1}),
                ToolContext("schedule-root", str(tmp_path)),
            )
            value = json.loads(created.text)
            assert value["id"] == "sched-1"
            listed = await registry.execute(
                "schedule_list", "{}", ToolContext("schedule-root", ".")
            )
            assert [item["id"] for item in json.loads(listed.text)] == ["sched-1"]
        finally:
            await service.dispose()

    asyncio.run(scenario())
