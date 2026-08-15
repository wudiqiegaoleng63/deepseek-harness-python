"""Atomic JSONL session persistence."""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from pathlib import Path

from ..models import JsonValue
from .model import Session

_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class JsonlSessionStore:
    """Store one lossless Session log per file."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, session_id: str) -> Path:
        if not _SESSION_ID.fullmatch(session_id):
            raise ValueError(f"invalid session id: {session_id!r}")
        return self.root / f"{session_id}.jsonl"

    async def save(self, session: Session) -> None:
        path = self.path_for(session.id)
        text = session.to_jsonl()
        await asyncio.to_thread(self._write_atomic, path, text)

    async def load(self, session_id: str) -> Session:
        path = self.path_for(session_id)
        try:
            text = await asyncio.to_thread(path.read_text, encoding="utf-8")
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"session does not exist: {session_id}") from exc
        return Session.from_jsonl(text)

    async def create(
        self,
        session_id: str,
        *,
        cwd: str | None = None,
        parent_session: str | None = None,
        origin: str | None = None,
        agent_preset: str | None = None,
        model_selection: dict[str, JsonValue] | None = None,
    ) -> Session:
        session = Session(
            session_id,
            header=Session.header_for(
                session_id,
                cwd=cwd,
                parent_session=parent_session,
                origin=origin,
                agent_preset=agent_preset,
                model_selection=model_selection,
            ),
        )
        path = self.path_for(session_id)
        if path.exists():
            raise FileExistsError(f"session already exists: {session_id}")
        await self.save(session)
        return session

    async def list_ids(self) -> tuple[str, ...]:
        return tuple(sorted(path.stem for path in self.root.glob("*.jsonl")))

    @staticmethod
    def _write_atomic(path: Path, text: str) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
