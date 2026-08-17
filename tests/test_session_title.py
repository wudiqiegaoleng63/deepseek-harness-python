from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from deepseek_harness.llm.types import LlmRequest, StreamChunk
from deepseek_harness.models import create_user_message
from deepseek_harness.session import Session
from deepseek_harness.session_title import (
    SessionTitleConfig,
    SessionTitleInvalidError,
    SessionTitleService,
    fallback_session_title,
    fold_session_title,
    normalize_session_title,
    truncate_title_utf8,
)


def test_title_normalization_removes_terminal_controls_and_respects_utf8_budget() -> None:
    raw = "\x1b[31m  Build\t the\nthing \u200f now  \x1b[0m"
    assert normalize_session_title(raw, 80) == "Build the thing now"
    assert fallback_session_title("one two three four five six", 5, 80) == "one two three four five"
    assert truncate_title_utf8("😀😀", 5) == "😀"
    assert truncate_title_utf8("😀", 3) == ""


def test_session_title_service_fallback_is_durable_and_explicit_rename_is_pinned() -> None:
    session = Session("title-session")
    service = SessionTitleService(
        SessionTitleConfig(fallback_max_words=5, fallback_max_bytes=40, max_title_bytes=80)
    )
    first = session.append(
        "user/message",
        {"message": create_user_message("Explain append-only logs clearly").to_dict()},
    )

    generated = service.on_user_message(session, first)

    assert generated is not None
    fallback = fold_session_title(session)
    assert fallback is not None
    assert fallback.title == "Explain append-only logs clearly"
    assert generated.data["messageSeqs"] == [first.seq]
    second = session.append(
        "user/message",
        {"message": create_user_message("a later prompt").to_dict()},
    )
    assert service.on_user_message(session, second) is None

    renamed = service.rename(session, "\x1b[32m  Hand picked title  \x1b[0m")
    assert renamed.data["source"] == {"kind": "user"}
    snapshot = fold_session_title(Session.from_jsonl(session.to_jsonl()))
    assert snapshot is not None
    assert snapshot.title == "Hand picked title"

    with pytest.raises(SessionTitleInvalidError, match="visible"):
        service.rename(session, "\x1b[31m\x1b[0m")


def test_session_title_service_ignores_non_human_messages_and_can_refresh_fallback() -> None:
    session = Session("title-refresh")
    service = SessionTitleService()
    assistant = session.append(
        "user/message",
        {
            "message": {
                **create_user_message("not human", source={"kind": "assistant"}).to_dict(),
            }
        },
    )
    assert service.on_user_message(session, assistant) is None
    human = session.append(
        "user/message",
        {"message": create_user_message("Derive a fresh title").to_dict()},
    )
    assert service.on_user_message(session, human) is not None
    refreshed = service.refresh_fallback(session)
    assert refreshed is not None
    snapshot = fold_session_title(session)
    assert snapshot is not None
    assert snapshot.source == {"kind": "fallback"}


def test_harness_service_auto_titles_first_prompt_and_exposes_projection(tmp_path) -> None:
    class Adapter:
        async def stream(self, request: LlmRequest) -> AsyncIterator[StreamChunk]:
            del request
            yield StreamChunk(kind="text", text="done")
            yield StreamChunk(kind="done", finish_reason="stop")

        async def aclose(self) -> None:
            return None

    async def scenario() -> None:
        from deepseek_harness.web import HarnessService

        service = HarnessService(
            tmp_path / "state",
            cwd=tmp_path,
            adapter_factory=lambda _model: Adapter(),
        )
        handle = await service.create_session(session_id="title-service", cwd=str(tmp_path))
        await service.prompt(handle.session.id, [{"type": "text", "text": "Inspect the API"}])
        assert handle.task is not None
        await handle.task
        summaries = await service.list_sessions()
        assert summaries[0]["title"] == "Inspect the API"
        history = await service.history(handle.session.id)
        assert history["projections"]["values"]["title"] == "Inspect the API"
        await service.dispose()

    asyncio.run(scenario())
