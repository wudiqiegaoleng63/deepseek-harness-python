"""Optional first-prompt model-backed session title generation."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .llm.adapter import LlmAdapter
from .llm.types import LlmCallConfig, LlmRequest, StreamChunk
from .models import JsonValue, create_user_message
from .session import Session
from .session_title import normalize_session_title

TITLE_PROVIDER_ID = "session-title-first-prompt-llm"
SESSION_TITLE_TIMEOUT_CODE = "SESSION_TITLE_TIMEOUT"


@dataclass(frozen=True, slots=True)
class SessionTitleLlmConfig:
    """Bounded policy for one auxiliary first-prompt title request."""

    target_words: int = 5
    target_cjk_characters: int = 10
    max_input_bytes: int = 4096
    max_output_tokens: int = 64
    timeout_seconds: float = 60.0
    provider: str | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("target_words", self.target_words),
            ("target_cjk_characters", self.target_cjk_characters),
            ("max_input_bytes", self.max_input_bytes),
            ("max_output_tokens", self.max_output_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.timeout_seconds <= 0 or not math.isfinite(self.timeout_seconds):
            raise ValueError("timeout_seconds must be finite and positive")
        if (self.provider is None) != (self.model is None):
            raise ValueError("provider and model must be supplied together")
        if self.provider is not None and not self.provider.strip():
            raise ValueError("provider override must be non-empty")
        if self.model is not None and not self.model.strip():
            raise ValueError("model override must be non-empty")


@dataclass(frozen=True, slots=True)
class SessionTitleLlmResult:
    title: str
    message_seqs: tuple[int, ...]
    provider: str
    model: str


class SessionTitleLlmError(RuntimeError):
    """Expected failure from the bounded auxiliary title request."""

    def __init__(self, message: str, *, code: str = "SESSION_TITLE_LLM_ERROR") -> None:
        super().__init__(message)
        self.code = code


AdapterFactory = Callable[[str], LlmAdapter]


async def generate_session_title(
    session: Session,
    messages: Sequence[tuple[int, str]],
    *,
    route_provider: str,
    route_model: str,
    adapter_factory: AdapterFactory,
    config: SessionTitleLlmConfig,
) -> SessionTitleLlmResult:
    """Generate a title from a fixed first-message snapshot and record its request."""

    if not messages:
        raise SessionTitleLlmError("at least one source message is required")
    if not route_provider.strip() or not route_model.strip():
        raise SessionTitleLlmError("title generation requires a non-empty provider and model")

    selected = [{"seq": seq, "text": text} for seq, text in messages]
    framed = (
        "Generate the session title from this JSON array of human messages:\n"
        + json.dumps(selected, ensure_ascii=False, separators=(",", ":"))
    )
    input_bytes = len(framed.encode("utf-8"))
    if input_bytes > config.max_input_bytes:
        raise SessionTitleLlmError(
            f"input is {input_bytes} bytes, exceeding max_input_bytes {config.max_input_bytes}"
        )

    provider = config.provider or route_provider
    model = config.model or route_model
    system = (
        "Create a concise title for an AI coding-assistant session from the supplied human "
        "messages.\n"
        "Return only the title on one line, in plain text of natural language, with no quotes, "
        "prefix, explanation, Markdown, XML, or terminal control codes. No code is allowed.\n"
        f"Use the language of the messages. Aim for about {config.target_words} words in "
        f"non-CJK languages or {config.target_cjk_characters} CJK characters."
    )
    auxiliary_message = create_user_message(
        framed,
        source={"kind": "plugin", "plugin": "dsh-session-title-llm"},
    )
    request_data: dict[str, JsonValue] = {
        "titleProvider": TITLE_PROVIDER_ID,
        "messageSeqs": [seq for seq, _text in messages],
        "route": {"provider": provider, "model": model},
        "system": system,
        "messages": [auxiliary_message.to_dict()],
        "maxTokens": config.max_output_tokens,
    }
    session.append("session/title-llm-request", request_data)

    request = LlmRequest(
        messages=(auxiliary_message,),
        config=LlmCallConfig(
            provider=provider,
            model=model,
            max_tokens=config.max_output_tokens,
            thinking="disabled",
            reasoning_effort="off",
        ),
        system=system,
        purpose="session-title",
        session_id=session.id,
    )
    adapter = adapter_factory(model)
    text_parts: list[str] = []
    finish_reason: str | None = None
    try:
        try:
            async with asyncio.timeout(config.timeout_seconds):
                async for chunk in adapter.stream(request):
                    _consume_title_chunk(chunk, text_parts)
                    if chunk.kind == "done":
                        finish_reason = chunk.finish_reason or "stop"
        except TimeoutError as exc:
            raise SessionTitleLlmError(
                "session title generation timed out",
                code=SESSION_TITLE_TIMEOUT_CODE,
            ) from exc
    finally:
        await adapter.aclose()

    if finish_reason in {"length", "max_tokens", "max-tokens"}:
        raise SessionTitleLlmError("title output reached max_output_tokens")
    if finish_reason in {"tool_calls", "tool-call"}:
        raise SessionTitleLlmError("title model unexpectedly requested a tool")
    if finish_reason not in {None, "stop", "completed"}:
        raise SessionTitleLlmError(f"unsupported title finish reason: {finish_reason}")
    title = normalize_session_title(" ".join(text_parts), 1_000_000)
    if not title:
        raise SessionTitleLlmError("title model produced no text")
    return SessionTitleLlmResult(title, tuple(seq for seq, _text in messages), provider, model)


def _consume_title_chunk(chunk: StreamChunk, text_parts: list[str]) -> None:
    if chunk.kind == "text":
        text_parts.append(chunk.text)
    elif chunk.kind == "tool-call-delta":
        raise SessionTitleLlmError("title output must contain text only")


__all__ = [
    "SESSION_TITLE_TIMEOUT_CODE",
    "TITLE_PROVIDER_ID",
    "SessionTitleLlmConfig",
    "SessionTitleLlmError",
    "SessionTitleLlmResult",
    "generate_session_title",
]
