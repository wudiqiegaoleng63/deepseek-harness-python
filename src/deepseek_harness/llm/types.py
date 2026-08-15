"""Provider-neutral request and stream types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

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


@dataclass(frozen=True, slots=True)
class LlmRequest:
    messages: tuple[Message, ...]
    config: LlmCallConfig
    system: str | None = None
    tools: tuple[ToolSchema, ...] = ()


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
