"""Durable session titles with deterministic first-prompt fallback."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass

from .models import JsonValue, message_from_dict
from .session import Session, SessionEvent

_OSC_SEQUENCE = re.compile(r"(?:\x1b\]|\x9d)(?:(?!\x07|\x1b\\)[\s\S])*(?:\x07|\x1b\\|$)")
_CSI_SEQUENCE = re.compile(r"(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]")
_ESC_SEQUENCE = re.compile(r"\x1b[@-_]")
_CONTROL_CHARACTER = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_DIRECTIONAL_CONTROL = re.compile(
    r"[\u200b\u200e\u200f\u202a-\u202e\u2060-\u2064\u2066-\u206f\ufeff]"
)


@dataclass(frozen=True, slots=True)
class SessionTitleConfig:
    fallback_max_words: int = 5
    fallback_max_bytes: int = 40
    max_title_bytes: int = 80

    def __post_init__(self) -> None:
        for name, value in (
            ("fallback_max_words", self.fallback_max_words),
            ("fallback_max_bytes", self.fallback_max_bytes),
            ("max_title_bytes", self.max_title_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.fallback_max_bytes > self.max_title_bytes:
            raise ValueError("fallback_max_bytes must not exceed max_title_bytes")


@dataclass(frozen=True, slots=True)
class SessionTitleSnapshot:
    title: str
    message_seqs: tuple[int, ...]
    source: dict[str, JsonValue]
    event_seq: int
    updated_at: int


class SessionTitleInvalidError(ValueError):
    """Raised when an explicit title contains no visible text."""

    code = "title-invalid"


def truncate_title_utf8(value: str, max_bytes: int) -> str:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if len(value.encode("utf-8")) <= max_bytes:
        return value
    output: list[str] = []
    used = 0
    for character in value:
        size = len(character.encode("utf-8"))
        if used + size > max_bytes:
            break
        output.append(character)
        used += size
    return "".join(output)


def normalize_session_title(value: str, max_bytes: int) -> str:
    cleaned = _clean_title_text(value)
    return truncate_title_utf8(cleaned, max_bytes).rstrip()


def fallback_session_title(value: str, max_words: int, max_bytes: int) -> str:
    if isinstance(max_words, bool) or not isinstance(max_words, int) or max_words <= 0:
        raise ValueError("max_words must be a positive integer")
    cleaned = _clean_title_text(value)
    words = cleaned.split(" ")[:max_words]
    return truncate_title_utf8(" ".join(words), max_bytes).rstrip()


def fold_session_title(session: Session) -> SessionTitleSnapshot | None:
    for event in reversed(session.events):
        if event.type != "session/title":
            continue
        raw_title = event.data.get("title")
        if not isinstance(raw_title, str) or not raw_title:
            continue
        raw_seqs = event.data.get("messageSeqs", [])
        message_seqs = tuple(
            item for item in raw_seqs if isinstance(item, int) and not isinstance(item, bool)
        ) if isinstance(raw_seqs, list) else ()
        raw_source = event.data.get("source", {"kind": "fallback"})
        source: dict[str, JsonValue]
        if isinstance(raw_source, dict):
            source = copy.deepcopy(raw_source)
        else:
            source = {"kind": "fallback"}
        return SessionTitleSnapshot(raw_title, message_seqs, source, event.seq, event.time)
    return None


class SessionTitleService:
    """Own title normalization and first eligible human-message fallback."""

    def __init__(self, config: SessionTitleConfig | None = None) -> None:
        self.config = config or SessionTitleConfig()

    def get(self, session: Session) -> SessionTitleSnapshot | None:
        return fold_session_title(session)

    def on_user_message(self, session: Session, event: SessionEvent) -> SessionEvent | None:
        if event.type != "user/message":
            return None
        raw_message = event.data.get("message")
        if not isinstance(raw_message, dict):
            return None
        raw_source = raw_message.get("source")
        if not isinstance(raw_source, dict) or raw_source.get("kind") != "user":
            return None
        if self.get(session) is not None:
            return None
        message = message_from_dict(raw_message)
        title = fallback_session_title(
            message.text,
            self.config.fallback_max_words,
            self.config.fallback_max_bytes,
        )
        if not title:
            return None
        return session.append(
            "session/title",
            {
                "title": title,
                "messageSeqs": [event.seq],
                "source": {"kind": "fallback"},
            },
        )

    def rename(self, session: Session, raw_title: str) -> SessionEvent:
        title = normalize_session_title(raw_title, self.config.max_title_bytes)
        if not title:
            raise SessionTitleInvalidError("session title must contain visible characters")
        return session.append(
            "session/title",
            {"title": title, "messageSeqs": [], "source": {"kind": "user"}},
        )

    def refresh_fallback(self, session: Session) -> SessionEvent | None:
        """Explicitly re-derive a fallback from the first eligible user message."""

        for event in session.events:
            if event.type != "user/message":
                continue
            raw_message = event.data.get("message")
            if not isinstance(raw_message, dict):
                continue
            source = raw_message.get("source")
            if not isinstance(source, dict) or source.get("kind") != "user":
                continue
            message = message_from_dict(raw_message)
            title = fallback_session_title(
                message.text,
                self.config.fallback_max_words,
                self.config.fallback_max_bytes,
            )
            if not title:
                return None
            return session.append(
                "session/title",
                {"title": title, "messageSeqs": [event.seq], "source": {"kind": "fallback"}},
            )
        return None


def _clean_title_text(value: str) -> str:
    value = _OSC_SEQUENCE.sub("", value)
    value = _CSI_SEQUENCE.sub("", value)
    value = _ESC_SEQUENCE.sub("", value)
    value = _CONTROL_CHARACTER.sub("", value)
    value = _DIRECTIONAL_CONTROL.sub("", value)
    return re.sub(r"\s+", " ", value).strip()


__all__ = [
    "SessionTitleConfig",
    "SessionTitleInvalidError",
    "SessionTitleService",
    "SessionTitleSnapshot",
    "fallback_session_title",
    "fold_session_title",
    "normalize_session_title",
    "truncate_title_utf8",
]
