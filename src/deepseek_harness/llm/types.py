"""Provider-neutral request and stream types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from ..errors import LlmFailure
from ..models import JsonValue, Message


@dataclass(frozen=True, slots=True)
class ToolSchema:
    name: str
    description: str
    parameters: dict[str, JsonValue]

    def to_openai(self) -> dict[str, JsonValue]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True, slots=True)
class LlmCallConfig:
    provider: str = "deepseek-official"
    model: str = "deepseek-v4-flash"
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    thinking: Literal["enabled", "disabled"] | None = None
    reasoning_effort: Literal["off", "high", "max"] | None = None


RetryMode = Literal["normal", "always"]
DEFAULT_RETRYABLE_CODES = (
    "EMPTY_RESPONSE",
    "RATE_LIMIT",
    "SERVER",
    "TIMEOUT",
    "TRANSPORT",
)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded or unbounded exponential retry policy for one model route."""

    mode: RetryMode = "normal"
    max_retries: int = 2
    retryable_codes: tuple[str, ...] = DEFAULT_RETRYABLE_CODES
    initial_delay_seconds: float = 0.5
    max_delay_seconds: float = 10.0
    jitter_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.mode not in {"normal", "always"}:
            raise ValueError("retry mode must be normal or always")
        if isinstance(self.max_retries, bool) or self.max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if not isinstance(self.max_retries, int):
            raise ValueError("max_retries must be a non-negative integer")
        if not self.retryable_codes or len(set(self.retryable_codes)) != len(self.retryable_codes):
            raise ValueError("retryable_codes must be non-empty and unique")
        if any(not isinstance(code, str) or not code for code in self.retryable_codes):
            raise ValueError("retryable_codes must contain non-empty strings")
        if self.initial_delay_seconds <= 0 or self.max_delay_seconds <= 0:
            raise ValueError("retry delays must be positive")
        if self.initial_delay_seconds > self.max_delay_seconds:
            raise ValueError("initial_delay_seconds must not exceed max_delay_seconds")
        if not 0 <= self.jitter_ratio <= 1:
            raise ValueError("jitter_ratio must be between 0 and 1")

    def allows(self, failure: LlmFailure, retry: int) -> bool:
        """Return whether ``retry`` (1-based) is admitted for this failure."""

        if retry <= 0:
            return False
        if self.mode == "always":
            return True
        return retry <= self.max_retries and failure.code in self.retryable_codes

    def delay_seconds(
        self,
        retry: int,
        retry_after_ms: int | None = None,
        random_value: float | None = None,
    ) -> float:
        """Calculate provider-aware exponential backoff with bounded jitter."""

        if retry_after_ms is not None and 0 < retry_after_ms <= self.max_delay_seconds * 1000:
            return retry_after_ms / 1000
        exponent = min(max(retry - 1, 0), 1024)
        exponential = min(self.initial_delay_seconds * 2**exponent, self.max_delay_seconds)
        sample = 0.5 if random_value is None else max(0.0, min(1.0, random_value))
        multiplier = 1 - self.jitter_ratio + 2 * self.jitter_ratio * sample
        return min(exponential * multiplier, self.max_delay_seconds)


@dataclass(frozen=True, slots=True)
class LlmRequest:
    messages: tuple[Message, ...]
    config: LlmCallConfig
    system: str | None = None
    tools: tuple[ToolSchema, ...] = ()
    purpose: Literal["conversation", "compaction", "session-title"] = "conversation"
    session_id: str | None = None


StreamKind: TypeAlias = Literal["text", "reasoning", "tool-call-delta", "done"]


@dataclass(frozen=True, slots=True)
class StreamChunk:
    kind: StreamKind
    text: str = ""
    index: int = 0
    call_id: str | None = None
    name: str | None = None
    arguments: str = ""
    finish_reason: str | None = None
    usage: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind}
        for key, value in (
            ("text", self.text),
            ("index", self.index),
            ("callId", self.call_id),
            ("name", self.name),
            ("arguments", self.arguments),
            ("finishReason", self.finish_reason),
            ("usage", self.usage),
        ):
            if value not in (None, "", 0) or key == "index":
                result[key] = value
        return result
