"""Append-only session model."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..models import (
    ImageContent,
    JsonValue,
    Message,
    assert_json_value,
    message_from_dict,
    now_millis,
)

SESSION_FORMAT_VERSION = 0
SessionListener = Callable[["SessionEvent"], Any]


@dataclass(frozen=True, slots=True)
class SessionHeader:
    id: str
    created_at: int = field(default_factory=now_millis)
    cwd: str | None = None
    parent_session: str | None = None
    seed_length: int | None = None
    origin: str | None = None
    agent_preset: str | None = None
    model_selection: dict[str, JsonValue] | None = None
    version: int = SESSION_FORMAT_VERSION

    def to_dict(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {
            "id": self.id,
            "createdAt": self.created_at,
            "version": self.version,
        }
        for key, value in (
            ("cwd", self.cwd),
            ("parentSession", self.parent_session),
            ("seedLength", self.seed_length),
            ("origin", self.origin),
            ("agentPreset", self.agent_preset),
            ("modelSelection", self.model_selection),
        ):
            if value is not None:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SessionHeader:
        return cls(
            id=str(value["id"]),
            created_at=int(value["createdAt"]),
            cwd=value.get("cwd"),
            parent_session=value.get("parentSession"),
            seed_length=value.get("seedLength"),
            origin=value.get("origin"),
            agent_preset=value.get("agentPreset"),
            model_selection=value.get("modelSelection"),
            version=int(value.get("version", SESSION_FORMAT_VERSION)),
        )


@dataclass(frozen=True, slots=True)
class SessionEvent:
    seq: int
    time: int
    type: str
    data: dict[str, JsonValue]

    def to_dict(self) -> dict[str, JsonValue]:
        return {"seq": self.seq, "time": self.time, "type": self.type, "data": self.data}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SessionEvent:
        data = value.get("data")
        if not isinstance(data, dict):
            raise ValueError("session event data must be an object")
        return cls(
            seq=int(value["seq"]),
            time=int(value["time"]),
            type=str(value["type"]),
            data=copy.deepcopy(data),
        )


class Session:
    """In-memory append-only log and model-history projection."""

    def __init__(
        self,
        session_id: str,
        *,
        header: SessionHeader | None = None,
        events: list[SessionEvent] | None = None,
    ) -> None:
        self.header = header or SessionHeader(session_id)
        if self.header.id != session_id:
            raise ValueError("session id and header id must match")
        self._events = list(events or [])
        self._listeners: list[SessionListener] = []
        self._attachment_data: dict[str, str] = {}
        self._validate_sequence()

    @property
    def id(self) -> str:
        return self.header.id

    @staticmethod
    def header_for(
        session_id: str,
        *,
        cwd: str | None = None,
        parent_session: str | None = None,
        origin: str | None = None,
        agent_preset: str | None = None,
        model_selection: dict[str, JsonValue] | None = None,
    ) -> SessionHeader:
        """Create a fresh header through one stable construction point."""

        return SessionHeader(
            session_id,
            cwd=cwd,
            parent_session=parent_session,
            origin=origin,
            agent_preset=agent_preset,
            model_selection=model_selection,
        )

    @property
    def events(self) -> tuple[SessionEvent, ...]:
        return tuple(self._events)

    @property
    def seq(self) -> int:
        return len(self._events)

    def listen(self, listener: SessionListener) -> Callable[[], None]:
        self._listeners.append(listener)
        active = True

        def dispose() -> None:
            nonlocal active
            if not active:
                return
            active = False
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

        return dispose

    def append(self, event_type: str, data: dict[str, JsonValue]) -> SessionEvent:
        if not event_type:
            raise ValueError("session event type cannot be empty")
        assert_json_value(data)
        event = SessionEvent(
            seq=self.seq, time=now_millis(), type=event_type, data=copy.deepcopy(data)
        )
        self._events.append(event)
        for listener in tuple(self._listeners):
            listener(event)
        return event

    def derive_messages(self) -> tuple[Message, ...]:
        messages: list[Message] = []
        for event in self._events:
            if event.type in {"user/message", "assistant/message", "tool/result"}:
                raw = event.data.get("message")
                if isinstance(raw, dict):
                    message = message_from_dict(raw)
                    content = tuple(
                        ImageContent(
                            block.attachment,
                            self._attachment_data.get(str(block.attachment.get("attachmentId"))),
                        )
                        if isinstance(block, ImageContent)
                        else block
                        for block in message.content
                    )
                    messages.append(Message(message.role, content, message.source, message.id))
        return tuple(messages)

    def register_attachment_data(self, attachment_id: str, data: str) -> None:
        """Make an admitted upload available to the current model request."""

        if attachment_id:
            self._attachment_data[attachment_id] = data

    def last_turn_reason(self) -> str | None:
        for event in reversed(self._events):
            if event.type == "turn/end":
                reason = event.data.get("reason")
                if isinstance(reason, dict):
                    kind = reason.get("kind")
                    if isinstance(kind, str):
                        return kind
        return None

    def to_jsonl(self) -> str:
        lines = [{"kind": "header", "header": self.header.to_dict()}]
        lines.extend(event.to_dict() for event in self._events)
        return "".join(
            json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n" for line in lines
        )

    @classmethod
    def from_jsonl(cls, text: str) -> Session:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not rows or rows[0].get("kind") != "header":
            raise ValueError("session JSONL is missing its header")
        header = SessionHeader.from_dict(rows[0]["header"])
        events = [SessionEvent.from_dict(row) for row in rows[1:]]
        return cls(header.id, header=header, events=events)

    def _validate_sequence(self) -> None:
        for expected, event in enumerate(self._events):
            if event.seq != expected:
                raise ValueError(f"session event sequence is not contiguous at {expected}")
