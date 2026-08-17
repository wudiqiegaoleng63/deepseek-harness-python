from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from deepseek_harness.llm.types import LlmRequest, StreamChunk
from deepseek_harness.session import Session
from deepseek_harness.session_title import fold_session_title
from deepseek_harness.session_title_llm import (
    SESSION_TITLE_TIMEOUT_CODE,
    SessionTitleLlmConfig,
    SessionTitleLlmError,
    generate_session_title,
)


class RecordingAdapter:
    def __init__(self, chunks: tuple[StreamChunk, ...]) -> None:
        self.chunks = chunks
        self.requests: list[LlmRequest] = []
        self.closed = False

    async def stream(self, request: LlmRequest) -> AsyncIterator[StreamChunk]:
        self.requests.append(request)
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class HangingAdapter:
    async def stream(self, request: LlmRequest) -> AsyncIterator[StreamChunk]:
        del request
        await asyncio.sleep(10)
        yield StreamChunk(kind="text", text="late")

    async def aclose(self) -> None:
        return None


def test_first_prompt_title_request_is_bounded_and_auditable() -> None:
    async def scenario() -> None:
        session = Session("title-llm")
        adapter = RecordingAdapter(
            (
                StreamChunk(kind="text", text="  Understand the API  "),
                StreamChunk(kind="done", finish_reason="stop"),
            )
        )
        result = await generate_session_title(
            session,
            ((7, "请解释这个 API"),),
            route_provider="deepseek-official",
            route_model="deepseek-v4-flash",
            adapter_factory=lambda _model: adapter,
            config=SessionTitleLlmConfig(max_input_bytes=4096, max_output_tokens=32),
        )

        assert result.title == "Understand the API"
        assert result.message_seqs == (7,)
        assert adapter.closed
        assert len(adapter.requests) == 1
        request = adapter.requests[0]
        assert request.purpose == "session-title"
        assert request.session_id == "title-llm"
        assert request.config.thinking == "disabled"
        assert request.config.reasoning_effort == "off"
        audit = session.events[-1]
        assert audit.type == "session/title-llm-request"
        assert audit.data["messageSeqs"] == [7]
        assert audit.data["maxTokens"] == 32

    asyncio.run(scenario())


def test_title_generation_keeps_request_failure_separate_from_fallback() -> None:
    async def scenario() -> None:
        session = Session("title-llm-failure")
        with pytest.raises(SessionTitleLlmError, match="exceeding"):
            await generate_session_title(
                session,
                ((1, "x" * 100),),
                route_provider="provider",
                route_model="model",
                adapter_factory=lambda _model: RecordingAdapter(()),
                config=SessionTitleLlmConfig(max_input_bytes=16),
            )
        assert not session.events

    asyncio.run(scenario())


def test_title_generation_timeout_has_stable_code() -> None:
    async def scenario() -> None:
        with pytest.raises(SessionTitleLlmError) as error:
            await generate_session_title(
                Session("title-timeout"),
                ((1, "wait"),),
                route_provider="provider",
                route_model="model",
                adapter_factory=lambda _model: HangingAdapter(),
                config=SessionTitleLlmConfig(timeout_seconds=0.01),
            )
        assert error.value.code == SESSION_TITLE_TIMEOUT_CODE

    asyncio.run(scenario())


def test_harness_can_run_first_prompt_title_provider_asynchronously(tmp_path) -> None:
    class Adapter:
        async def stream(self, request: LlmRequest) -> AsyncIterator[StreamChunk]:
            if request.purpose == "session-title":
                yield StreamChunk(kind="text", text="Model generated title")
            else:
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
            session_title_llm=SessionTitleLlmConfig(timeout_seconds=1),
        )
        handle = await service.create_session(session_id="title-provider", cwd=str(tmp_path))
        await service.prompt(
            handle.session.id,
            [{"type": "text", "text": "Inspect the API"}],
        )
        assert handle.task is not None
        await handle.task

        async def wait_for_provider_title() -> None:
            for _ in range(100):
                title = fold_session_title(handle.session)
                if title is not None and title.source.get("kind") == "provider":
                    return
                await asyncio.sleep(0.01)
            raise AssertionError("model-backed title did not arrive")

        await wait_for_provider_title()
        title = fold_session_title(handle.session)
        assert title is not None
        assert title.title == "Model generated title"
        assert any(event.type == "session/title-llm-request" for event in handle.session.events)
        await service.dispose()

    asyncio.run(scenario())
