from __future__ import annotations

import asyncio

from deepseek_harness.compaction import CompactionPolicy
from deepseek_harness.models import create_user_message
from deepseek_harness.web import HarnessService


def test_manual_compact_command_records_lifecycle_and_accounting(tmp_path) -> None:
    async def scenario() -> None:
        service = HarnessService(
            tmp_path / "state",
            cwd=tmp_path,
            compaction_policy=CompactionPolicy(
                context_window_tokens=10_000,
                threshold_ratio=0.2,
                retain_ratio=None,
                retain_tokens=500,
                summary_max_chars=2_000,
            ),
        )
        handle = await service.create_session(session_id="manual-compact", cwd=str(tmp_path))
        for index in range(5):
            message = create_user_message(f"history-{index}: " + ("context " * 600))
            handle.session.append("user/message", {"message": message.to_dict()})

        result = await service.prompt(
            handle.session.id,
            [{"type": "text", "text": "/compact"}],
        )

        command = result["command"]
        assert command["kind"] == "success"
        assert str(command["text"]).startswith("Compacted ")
        assert isinstance(command["sourceEventSeq"], int)
        event_types = [event.type for event in handle.session.events]
        assert event_types[0:1] == ["user/message"]
        assert event_types[-1] == "command/done"
        assert event_types[5] == "command/run"
        assert event_types[-2] == "compaction/end"
        starts = [event for event in handle.session.events if event.type == "compaction/start"]
        summaries = [event for event in handle.session.events if event.type == "compaction/summary"]
        ends = [event for event in handle.session.events if event.type == "compaction/end"]
        assert len(starts) == len(summaries) == len(ends) == 1
        assert starts[0].data["trigger"] == "manual"
        assert starts[0].data["turn"] is None
        assert summaries[0].seq == command["sourceEventSeq"]
        assert ends[0].data["summarySeq"] == summaries[0].seq
        assert handle.session.derive_messages()

        await service.dispose()

    asyncio.run(scenario())


def test_manual_compact_command_handles_no_history_arguments_and_busy_sessions(tmp_path) -> None:
    async def scenario() -> None:
        service = HarnessService(tmp_path / "state", cwd=tmp_path)
        handle = await service.create_session(session_id="manual-compact-errors", cwd=str(tmp_path))

        empty = await service.prompt(
            handle.session.id,
            [{"type": "text", "text": "/compact"}],
        )
        assert empty["command"] == {"kind": "success", "text": "No compactable history yet."}

        rejected = await service.prompt(
            handle.session.id,
            [{"type": "text", "text": "/compact now"}],
        )
        assert rejected["command"] == {
            "kind": "error",
            "text": "Usage: /compact (no arguments)",
        }

        handle.session.append("turn/start", {"turn": 1})
        busy = await service.prompt(
            handle.session.id,
            [{"type": "text", "text": "/compact"}],
        )
        assert busy["command"] == {
            "kind": "error",
            "text": (
                "Compaction is unavailable because this process has an active compaction, "
                "or the agent is not idle."
            ),
        }
        handle.session.append("turn/end", {"turn": 1, "reason": {"kind": "aborted"}})
        command_events = [
            event.type
            for event in handle.session.events
            if event.type in {"command/run", "command/done"}
        ]
        assert command_events == [
            "command/run",
            "command/done",
            "command/run",
            "command/done",
            "command/run",
            "command/done",
        ]

        await service.dispose()

    asyncio.run(scenario())
