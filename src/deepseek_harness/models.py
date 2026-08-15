"""Model-facing message and content vocabulary.

The wire representation deliberately stays lossless JSON.  Known blocks get
typed Python wrappers while unknown blocks remain round-trippable through
``RawContent`` so newer TS logs do not get silently discarded by an older
Python reader.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class TextContent:
    text: str
    type: Literal["text"] = field(default="text", init=False)

    def to_dict(self) -> dict[str, JsonValue]:
        return {"type": self.type, "text": self.text}


@dataclass(frozen=True, slots=True)
class ReasoningContent:
    text: str
    type: Literal["reasoning"] = field(default="reasoning", init=False)

    def to_dict(self) -> dict[str, JsonValue]:
        return {"type": self.type, "text": self.text}


@dataclass(frozen=True, slots=True)
class ToolCallContent:
    call_id: str
    name: str
    arguments: str
    type: Literal["tool-call"] = field(default="tool-call", init=False)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "type": self.type,
            "callId": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
        }


@dataclass(frozen=True, slots=True)
class ToolResultContent:
    call_id: str
    text: str
    type: Literal["tool-result"] = field(default="tool-result", init=False)

    def to_dict(self) -> dict[str, JsonValue]:
        return {"type": self.type, "callId": self.call_id, "text": self.text}


@dataclass(frozen=True, slots=True)
class ImageContent:
    """Durable image reference with optional in-process upload bytes.

    The encoded bytes are deliberately omitted from the session event.  The
    optional ``data`` field only bridges the upload admission path to the
    current model request; a resumed session reads the content-addressed
    object through the attachment service.
    """

    attachment: dict[str, JsonValue]
    data: str | None = field(default=None, repr=False, compare=False)
    type: Literal["image"] = field(default="image", init=False)

    def to_dict(self) -> dict[str, JsonValue]:
        return {"type": self.type, "attachment": self.attachment}


@dataclass(frozen=True, slots=True)
class RawContent:
    raw_type: str
    data: dict[str, Any]

    @property
    def type(self) -> str:
        return self.raw_type

    def to_dict(self) -> dict[str, JsonValue]:
        value = dict(self.data)
        value["type"] = self.raw_type
        return value


ContentBlock: TypeAlias = (
    TextContent | ReasoningContent | ToolCallContent | ToolResultContent | ImageContent | RawContent
)
Role: TypeAlias = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: tuple[ContentBlock, ...]
    source: dict[str, JsonValue]
    id: str = field(default_factory=lambda: f"message-{uuid.uuid4().hex}")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "role": self.role,
            "content": [block.to_dict() for block in self.content],
            "source": self.source,
        }

    @property
    def text(self) -> str:
        return "".join(
            block.text
            for block in self.content
            if isinstance(block, (TextContent, ReasoningContent, ToolResultContent))
        )


def create_user_message(text: str, *, source: dict[str, JsonValue] | None = None) -> Message:
    return Message(
        role="user",
        content=(TextContent(text),),
        source=source or {"kind": "user"},
    )


def create_tool_message(
    call_id: str, text: str, *, source: dict[str, JsonValue] | None = None
) -> Message:
    return Message(
        role="tool",
        content=(ToolResultContent(call_id, text),),
        source=source or {"kind": "tool"},
    )


def content_from_dict(value: Any) -> ContentBlock:
    if not isinstance(value, dict) or not isinstance(value.get("type"), str):
        raise ValueError("content block must be an object with a string type")
    kind = value["type"]
    if kind == "text":
        return TextContent(str(value.get("text", "")))
    if kind == "reasoning":
        return ReasoningContent(str(value.get("text", "")))
    if kind == "tool-call":
        return ToolCallContent(
            call_id=str(value.get("callId", "")),
            name=str(value.get("name", "")),
            arguments=str(value.get("arguments", "{}")),
        )
    if kind == "tool-result":
        return ToolResultContent(
            call_id=str(value.get("callId", "")),
            text=str(value.get("text", "")),
        )
    if kind == "image":
        attachment = value.get("attachment")
        if not isinstance(attachment, dict):
            raise ValueError("image content must contain an attachment object")
        return ImageContent(dict(attachment))
    return RawContent(kind, dict(value))


def message_from_dict(value: Any) -> Message:
    if not isinstance(value, dict):
        raise ValueError("message must be an object")
    role = value.get("role")
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError(f"unsupported message role: {role!r}")
    raw_content = value.get("content", [])
    if not isinstance(raw_content, list):
        raise ValueError("message content must be an array")
    source = value.get("source", {"kind": role})
    if not isinstance(source, dict):
        raise ValueError("message source must be an object")
    return Message(
        role=role,
        content=tuple(content_from_dict(item) for item in raw_content),
        source=dict(source),
        id=str(value.get("id") or f"message-{uuid.uuid4().hex}"),
    )


def assert_json_value(value: Any) -> None:
    """Reject NaN, bytes, and other values that cannot survive a session log."""

    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not lossless JSON") from exc


def now_millis() -> int:
    return int(time.time() * 1000)
