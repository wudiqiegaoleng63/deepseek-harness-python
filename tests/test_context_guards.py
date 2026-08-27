from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from deepseek_harness.guards import RepeatToolGuard
from deepseek_harness.llm.types import StreamChunk
from deepseek_harness.sandbox import UnavailableSandbox
from deepseek_harness.session import Session
from deepseek_harness.session_query import SessionSearchIndex, documents_from_messages
from deepseek_harness.time_context import TimeContextInjector
from deepseek_harness.tools import ToolContext
from deepseek_harness.web import HarnessService


class _Adapter:
    def stream(self, request):
        del request

        async def chunks():
            yield StreamChunk(kind="text", text="done")
            yield StreamChunk(kind="done", finish_reason="stop")

        return chunks()

    async def aclose(self) -> None:
        return None


def test_repeat_guard_escalates_and_resets_on_user_turn() -> None:
    guard = RepeatToolGuard()
    arguments = {"path": "a.txt", "content": "x"}
    assert guard.observe("s", "write_file", arguments) is None
    assert guard.observe("s", "write_file", arguments) is None
    gentle = guard.observe("s", "write_file", arguments)
    assert gentle is not None and "repeating the exact same tool call" in gentle
    assert guard.observe("s", "write_file", arguments) is None
    detailed = guard.observe("s", "write_file", arguments)
    assert detailed is not None and "consecutive_calls: 5" in detailed
    assert '"path": "a.txt"' in detailed or '"path":"a.txt"' in detailed

    # Argument order does not matter for the chain key: the run continues to
    # the next configured threshold (8) across reordered arguments.
    reordered = {"content": "x", "path": "a.txt"}
    assert guard.observe("s", "write_file", reordered) is None
    assert guard.observe("s", "write_file", reordered) is None
    escalated = guard.observe("s", "write_file", reordered)
    assert escalated is not None and "consecutive_calls: 8" in escalated

    # A different call restarts at one; a user turn resets the chain.
    assert guard.observe("s", "read_file", {"path": "b.txt"}) is None
    guard.reset("s")
    assert guard.observe("s", "write_file", arguments) is None


def test_repeat_guard_wildcards_and_validation() -> None:
    guard = RepeatToolGuard(include=("mcp_*",), exclude=("mcp_secret*",))
    assert guard.observe("s", "mcp_search", {"q": 1}) is None
    assert guard._tracked("mcp_search") is True
    assert guard._tracked("mcp_secret_lookup") is False
    assert guard._tracked("bash") is False
    for bad in ((), (1,), (3, 3)):
        try:
            RepeatToolGuard(thresholds=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected rejection for {bad}")


def test_time_context_injector_renders_and_throttles() -> None:
    injector = TimeContextInjector(time_zone="Asia/Shanghai", refresh_interval_seconds=60)
    session = Session("time-s")
    now = datetime.now(UTC)
    first = injector.message_text(session, now=now)
    assert first is not None
    assert "turn 0, step 1" in first
    assert "Elapsed since the preceding model-visible message: unavailable." in first

    # Mirror the real flow: the injected reading lands as a durable
    # plugin-sourced user message, which drives the refresh throttle.
    session.append(
        "user/message",
        {
            "message": {
                "role": "user",
                "content": [],
                "source": {"kind": "plugin", "plugin": "time-context"},
            }
        },
    )
    assert injector.message_text(session, now=now + timedelta(seconds=10)) is None

    session.append("turn/start", {"turn": 1})
    session.append("user/message", {"message": {"role": "user", "content": []}})
    later = injector.message_text(session, now=now + timedelta(seconds=90))
    assert later is not None
    assert "turn 1, step 1" in later
    assert "Elapsed since the preceding model-visible message" in later

    try:
        TimeContextInjector(time_zone="Mars/Olympus")
    except ValueError as exc:
        assert "invalid IANA timeZone" in str(exc)
    else:
        raise AssertionError("expected invalid zone rejection")


def test_session_search_index_round_trip(tmp_path: Path) -> None:
    index = SessionSearchIndex(tmp_path / "query.db")
    try:
        assert index.available
        index.index_session(
            "s1",
            documents_from_messages(
                [
                    ("user", "please fix the login bug"),
                    ("assistant", "fixed the login bug in auth.py"),
                ]
            ),
        )
        index.index_session(
            "s2",
            documents_from_messages([("user", "write docs for the parser")]),
        )
        assert index.session_count() == 2
        hits = index.search("login bug")
        assert [item["sessionId"] for item in hits] == ["s1"]
        assert "login" in hits[0]["snippet"].casefold()
        index.remove_session("s1")
        assert index.search("login bug") == []
        assert index.search("") == []
    finally:
        index.close()


def test_service_wires_guard_query_and_time_context(tmp_path: Path) -> None:
    async def scenario() -> None:
        service = HarnessService(
            tmp_path / "state",
            cwd=tmp_path,
            adapter_factory=lambda _model: _Adapter(),
            sandbox_provider=UnavailableSandbox(),
            session_query_path="auto",
            time_context_zone="UTC",
        )
        handle = await service.create_session(session_id="wired", cwd=str(tmp_path))
        context = ToolContext(handle.session.id, str(tmp_path))
        try:
            registry = service._tool_registries[handle.session.id]
            arguments = json.dumps({"path": "same.txt", "content": "x"})
            await registry.execute("write_file", arguments, context)
            await registry.execute("write_file", arguments, context)
            third = await registry.execute("write_file", arguments, context)
            assert "repeating the exact same tool call" in third.text
            assert (tmp_path / "same.txt").exists()

            # The query index answers across persisted content.
            assert service.session_query is not None
            handle.session.append(
                "user/message",
                {
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": "please write same.txt"}],
                    }
                },
            )
            service._index_session(handle.session)
            assert service.session_query.search("same.txt")
        finally:
            await service.dispose()

    asyncio.run(scenario())
