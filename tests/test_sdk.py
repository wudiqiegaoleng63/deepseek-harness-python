from __future__ import annotations

from collections.abc import AsyncIterator

from deepseek_harness import DeepSeekHarness, DeepSeekHarnessConfig
from deepseek_harness.llm.types import LlmRequest, StreamChunk


class SdkAdapter:
    def stream(self, request: LlmRequest) -> AsyncIterator[StreamChunk]:
        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(kind="text", text="sdk response")
            yield StreamChunk(kind="done", finish_reason="stop")

        return chunks()

    async def aclose(self) -> None:
        return None


def test_native_sdk_reuses_durable_sessions(tmp_path) -> None:
    events: list[dict[str, object]] = []
    with DeepSeekHarness(
        DeepSeekHarnessConfig(
            cwd=tmp_path,
            session_root=tmp_path / "state",
            adapter_factory=lambda _model: SdkAdapter(),
        )
    ) as harness:
        first = harness.run("first", session_id="sdk-session", on_event=events.append)
        second = harness.start_session("sdk-session").run("second")

    assert first.session_id == second.session_id == "sdk-session"
    assert first.final_response == second.final_response == "sdk response"
    assert first.finish_reason == second.finish_reason == "completed"
    assert any(event.get("type") == "assistant/message" for event in events)
