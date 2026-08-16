"""Event-sourced whole-list todo state for the model and web projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from .models import JsonValue
from .session import Session, SessionEvent

TodoStatus = Literal["pending", "in_progress", "completed"]
TodoItem = dict[str, JsonValue]


class TodoError(Exception):
    """A model-facing todo validation failure."""


@dataclass(frozen=True, slots=True)
class TodoWrite:
    """Normalized write result and its durable session event."""

    todos: list[TodoItem]
    counts: dict[str, int]
    event: SessionEvent

    def value(self) -> dict[str, JsonValue]:
        return cast(
            dict[str, JsonValue],
            {
            "todos": self.todos,
            "counts": {
                "pending": self.counts["pending"],
                "inProgress": self.counts["in_progress"],
                "completed": self.counts["completed"],
            },
            },
        )


class TodoManager:
    """Validate writes and replay the latest standing todo list."""

    def __init__(self, *, allow_parallel_in_progress: bool = True) -> None:
        self.allow_parallel_in_progress = allow_parallel_in_progress

    def write(self, session: Session, raw: Any) -> TodoWrite:
        todos = self._normalize(raw)
        active = sum(item["status"] == "in_progress" for item in todos)
        if not self.allow_parallel_in_progress and active > 1:
            raise TodoError(
                f"at most one task may be in_progress (got {active})"
            )
        counts = {
            "pending": sum(item["status"] == "pending" for item in todos),
            "in_progress": active,
            "completed": sum(item["status"] == "completed" for item in todos),
        }
        event = session.append("todo/write", {"todos": cast(JsonValue, todos)})
        return TodoWrite(todos, counts, event)

    def fold(self, session: Session) -> list[TodoItem] | None:
        current: list[TodoItem] | None = None
        for event in session.events:
            if event.type == "turn/start":
                current = None
            elif event.type == "todo/write":
                decoded = self._decode_event(event)
                if decoded is not None:
                    current = decoded
        return current

    @classmethod
    def _normalize(cls, raw: Any) -> list[TodoItem]:
        if not isinstance(raw, list):
            raise TodoError("todos must be an array")
        normalized: list[TodoItem] = []
        seen: set[str] = set()
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise TodoError(f"todo item {index} must be an object")
            if set(item) != {"content", "status"}:
                raise TodoError(f"todo item {index} has unknown or missing fields")
            content = item.get("content")
            status = item.get("status")
            if not isinstance(content, str) or not content.strip():
                raise TodoError("todo content must be a non-empty string")
            content = content.strip()
            if content in seen:
                raise TodoError(f"duplicate todo content: {content!r}")
            if status not in {"pending", "in_progress", "completed"}:
                raise TodoError(f"invalid todo status: {status!r}")
            seen.add(content)
            normalized.append({"content": content, "status": status})
        return normalized

    @classmethod
    def _decode_event(cls, event: SessionEvent) -> list[TodoItem] | None:
        raw = event.data.get("todos")
        try:
            return cls._normalize(raw)
        except TodoError:
            return None


__all__ = ["TodoError", "TodoItem", "TodoManager", "TodoStatus", "TodoWrite"]
