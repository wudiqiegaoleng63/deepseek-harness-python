"""Lifecycle-bound per-message feedback with a small durable sidecar."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from .models import now_millis
from .session import Session


class MessageFeedbackManager:
    """Implement the TS messageFeedback Remote contract without a new database.

    Feedback is intentionally kept outside the Session event log.  The row is
    fenced by the Session's creation timestamp and cwd, so reusing a session id
    cannot expose a previous lifecycle's ratings.
    """

    def __init__(self, root: str | os.PathLike[str], *, max_note_bytes: int = 8192) -> None:
        if max_note_bytes < 1:
            raise ValueError("max_note_bytes must be positive")
        self.path = Path(root).expanduser().resolve() / "message-feedback.json"
        self.max_note_bytes = max_note_bytes
        self._rows: dict[str, dict[str, Any]] = self._load()
        self._lock = asyncio.Lock()

    async def list(self, session: Session) -> dict[str, Any]:
        async with self._lock:
            row = self._current_row(session)
            return {"ok": True, "value": {"items": list(row["items"]) if row else []}}

    async def put(
        self,
        session: Session,
        message_id: str,
        rating: str,
        note: str | None,
        if_version: str | None,
    ) -> dict[str, Any]:
        resolved_note = self._resolve_note(note)
        if resolved_note[0] is False:
            return {"ok": False, "error": resolved_note[1]}
        if rating not in {"positive", "negative"}:
            return {
                "ok": False,
                "error": {"code": "invalid-rating", "rating": rating},
            }
        async with self._lock:
            if not self._has_message(session, message_id):
                return {
                    "ok": False,
                    "error": {
                        "code": "target-not-found",
                        "sessionId": session.id,
                        "messageId": message_id,
                    },
                }
            row = self._current_row(session)
            items = list(row["items"]) if row else []
            index = next(
                (index for index, item in enumerate(items) if item["messageId"] == message_id),
                -1,
            )
            existing = items[index] if index >= 0 else None
            expected = existing["version"] if existing is not None else None
            if if_version != expected:
                return {
                    "ok": False,
                    "error": {
                        "code": "version-conflict",
                        "current": existing,
                    },
                }
            if (
                existing is not None
                and existing["rating"] == rating
                and existing.get("note") == resolved_note[1]
            ):
                return {"ok": True, "value": dict(existing)}
            now = now_millis()
            item: dict[str, Any] = {
                "messageId": message_id,
                "rating": rating,
                "version": str(uuid.uuid4()),
                "createdAt": existing["createdAt"] if existing else now,
                "updatedAt": max(now, existing["updatedAt"]) if existing else now,
            }
            if resolved_note[1] is not None:
                item["note"] = resolved_note[1]
            if index >= 0:
                items[index] = item
            else:
                items.append(item)
            self._rows[session.id] = {"session": self._identity(session), "items": items}
            self._save()
            return {"ok": True, "value": dict(item)}

    async def delete(
        self,
        session: Session,
        message_id: str,
        if_version: str | None,
    ) -> dict[str, Any]:
        async with self._lock:
            row = self._current_row(session)
            items = list(row["items"]) if row else []
            index = next(
                (index for index, item in enumerate(items) if item["messageId"] == message_id),
                -1,
            )
            if index < 0:
                return {"ok": True, "value": {"absent": True}}
            existing = items[index]
            if if_version != existing["version"]:
                return {
                    "ok": False,
                    "error": {"code": "version-conflict", "current": dict(existing)},
                }
            items.pop(index)
            self._rows[session.id] = {"session": self._identity(session), "items": items}
            self._save()
            return {"ok": True, "value": {"absent": True}}

    def _current_row(self, session: Session) -> dict[str, Any] | None:
        row = self._rows.get(session.id)
        if row is None or row.get("session") != self._identity(session):
            return None
        return row

    @staticmethod
    def _identity(session: Session) -> dict[str, Any]:
        identity: dict[str, Any] = {"createdAt": session.header.created_at}
        if session.header.cwd is not None:
            identity["cwd"] = session.header.cwd
        return identity

    @staticmethod
    def _has_message(session: Session, message_id: str) -> bool:
        return any(
            message.role == "assistant" and message.id == message_id
            for message in session.derive_messages()
        )

    def _resolve_note(self, note: str | None) -> tuple[bool, str | dict[str, Any] | None]:
        if note is None:
            return True, None
        if not isinstance(note, str):
            return False, {"code": "note-invalid"}
        if not note.strip():
            return False, {"code": "note-blank"}
        actual_bytes = len(note.encode("utf-8"))
        if actual_bytes > self.max_note_bytes:
            return False, {
                "code": "note-too-large",
                "maxBytes": self.max_note_bytes,
                "actualBytes": actual_bytes,
            }
        return True, note

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid message feedback sidecar: {self.path}") from exc
        if not isinstance(raw, dict):
            raise ValueError("message feedback sidecar must contain an object")
        return {
            key: value
            for key, value in raw.items()
            if isinstance(key, str) and isinstance(value, dict)
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
                json.dump(self._rows, stream, ensure_ascii=False, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)


__all__ = ["MessageFeedbackManager"]
