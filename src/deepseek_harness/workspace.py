"""Durable workspace registry mirroring the DSH workspace domain."""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .persistence import JsonStateStore


class WorkspaceError(Exception):
    """Base class for registry operations that need a wire-level error."""


class WorkspaceNotFound(WorkspaceError):
    def __init__(self, workspace_id: str) -> None:
        super().__init__(f'workspace "{workspace_id}" not found')
        self.workspace_id = workspace_id


class WorkspaceNameConflict(WorkspaceError):
    def __init__(self, title: str) -> None:
        super().__init__(f"workspace title is already in use: {title}")
        self.title = title


class WorkspaceMoveInvalid(WorkspaceError):
    def __init__(
        self, workspace_id: str, session_id: str, before_session_id: str | None = None
    ) -> None:
        super().__init__(f'session "{session_id}" is not accounted by workspace "{workspace_id}"')
        self.workspace_id = workspace_id
        self.session_id = session_id
        self.before_session_id = before_session_id


class WorkspaceInvalidPath(WorkspaceError):
    def __init__(self, path: str) -> None:
        super().__init__(f"workspace path is not a directory: {path}")
        self.path = path


@dataclass(frozen=True, slots=True)
class WorkspaceView:
    workspace_id: str
    path: str
    title: str
    session_ids: tuple[str, ...]
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspaceId": self.workspace_id,
            "path": self.path,
            "title": self.title,
            "sessionIds": list(self.session_ids),
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> WorkspaceView:
        return cls(
            workspace_id=str(value["workspaceId"]),
            path=str(value["path"]),
            title=str(value["title"]),
            session_ids=tuple(str(item) for item in value.get("sessionIds", [])),
            created_at=str(value["createdAt"]),
            updated_at=str(value["updatedAt"]),
        )


class WorkspaceRegistry:
    """Ordered workspace records plus the global archive set."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._store = JsonStateStore(
            Path(root).expanduser().resolve() / "workspaces.json",
            default={"version": 1, "workspaces": [], "archivedSessionIds": []},
        )
        self._workspaces: list[WorkspaceView] = []
        self._archived: set[str] = set()
        self._loaded = False
        self._lock = asyncio.Lock()

    async def list(self) -> tuple[tuple[WorkspaceView, ...], tuple[str, ...]]:
        async with self._lock:
            await self._ensure_loaded()
            return tuple(self._workspaces), tuple(sorted(self._archived))

    async def create(self, raw_path: str) -> tuple[WorkspaceView, bool]:
        path = self._canonical_directory(raw_path)
        async with self._lock:
            await self._ensure_loaded()
            existing = next((item for item in self._workspaces if item.path == str(path)), None)
            if existing is not None:
                return existing, False
            title = path.name or str(path)
            now = _now()
            workspace = WorkspaceView(
                workspace_id=f"workspace-{uuid.uuid4().hex}",
                path=str(path),
                title=title,
                session_ids=(),
                created_at=now,
                updated_at=now,
            )
            self._workspaces.append(workspace)
            await self._save()
            return workspace, True

    async def get(self, workspace_id: str) -> WorkspaceView:
        async with self._lock:
            await self._ensure_loaded()
            return self._get_unlocked(workspace_id)

    async def attach_session(self, workspace_id: str, session_id: str) -> WorkspaceView:
        async with self._lock:
            await self._ensure_loaded()
            current = self._get_unlocked(workspace_id)
            if session_id in current.session_ids:
                return current
            updated = replace(
                current, session_ids=(session_id, *current.session_ids), updated_at=_now()
            )
            self._replace(updated)
            await self._save()
            return updated

    async def attach_by_path(self, path: str, session_id: str) -> WorkspaceView | None:
        canonical = str(Path(path).expanduser().resolve())
        async with self._lock:
            await self._ensure_loaded()
            current = next((item for item in self._workspaces if item.path == canonical), None)
            if current is None or session_id in current.session_ids:
                return current
            updated = replace(
                current, session_ids=(session_id, *current.session_ids), updated_at=_now()
            )
            self._replace(updated)
            await self._save()
            return updated

    async def rename(self, workspace_id: str, title: str) -> WorkspaceView:
        title = title.strip()
        if not title:
            raise ValueError("workspace title cannot be empty")
        async with self._lock:
            await self._ensure_loaded()
            current = self._get_unlocked(workspace_id)
            if title != current.title and any(item.title == title for item in self._workspaces):
                raise WorkspaceNameConflict(title)
            if title == current.title:
                return current
            updated = replace(current, title=title, updated_at=_now())
            self._replace(updated)
            await self._save()
            return updated

    async def delete(self, workspace_id: str) -> None:
        async with self._lock:
            await self._ensure_loaded()
            self._get_unlocked(workspace_id)
            self._workspaces = [
                item for item in self._workspaces if item.workspace_id != workspace_id
            ]
            await self._save()

    async def insert_before(
        self, workspace_id: str, before_workspace_id: str | None
    ) -> tuple[str, ...]:
        async with self._lock:
            await self._ensure_loaded()
            current = self._get_unlocked(workspace_id)
            if before_workspace_id == workspace_id:
                return tuple(item.workspace_id for item in self._workspaces)
            if before_workspace_id is not None:
                self._get_unlocked(before_workspace_id)
            self._workspaces = [
                item for item in self._workspaces if item.workspace_id != workspace_id
            ]
            if before_workspace_id is None:
                self._workspaces.append(current)
            else:
                index = next(
                    index
                    for index, item in enumerate(self._workspaces)
                    if item.workspace_id == before_workspace_id
                )
                self._workspaces.insert(index, current)
            await self._save()
            return tuple(item.workspace_id for item in self._workspaces)

    async def insert_session_before(
        self,
        workspace_id: str,
        session_id: str,
        before_session_id: str | None,
    ) -> WorkspaceView:
        async with self._lock:
            await self._ensure_loaded()
            current = self._get_unlocked(workspace_id)
            if session_id not in current.session_ids:
                raise WorkspaceMoveInvalid(workspace_id, session_id, before_session_id)
            if before_session_id == session_id:
                return current
            if before_session_id is not None and before_session_id not in current.session_ids:
                raise WorkspaceMoveInvalid(workspace_id, session_id, before_session_id)
            remaining = [item for item in current.session_ids if item != session_id]
            if before_session_id is None:
                remaining.append(session_id)
            else:
                index = remaining.index(before_session_id)
                remaining.insert(index, session_id)
            updated = replace(current, session_ids=tuple(remaining), updated_at=_now())
            self._replace(updated)
            await self._save()
            return updated

    async def archive(self, session_id: str) -> tuple[str, ...]:
        async with self._lock:
            await self._ensure_loaded()
            self._archived.add(session_id)
            await self._save()
            return tuple(sorted(self._archived))

    async def owns_session(self, session_id: str) -> WorkspaceView | None:
        async with self._lock:
            await self._ensure_loaded()
            return next(
                (item for item in self._workspaces if session_id in item.session_ids),
                None,
            )

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        value = await self._store.load()
        rows = value.get("workspaces", [])
        self._workspaces = [WorkspaceView.from_dict(row) for row in rows if isinstance(row, dict)]
        archived = value.get("archivedSessionIds", [])
        self._archived = {str(item) for item in archived if isinstance(item, str)}
        self._loaded = True

    async def _save(self) -> None:
        await self._store.save(
            {
                "version": 1,
                "workspaces": [item.to_dict() for item in self._workspaces],
                "archivedSessionIds": sorted(self._archived),
            }
        )

    def _get_unlocked(self, workspace_id: str) -> WorkspaceView:
        for item in self._workspaces:
            if item.workspace_id == workspace_id:
                return item
        raise WorkspaceNotFound(workspace_id)

    def _replace(self, workspace: WorkspaceView) -> None:
        self._workspaces = [
            workspace if item.workspace_id == workspace.workspace_id else item
            for item in self._workspaces
        ]

    @staticmethod
    def _canonical_directory(raw_path: str) -> Path:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise WorkspaceInvalidPath(str(path))
        return path


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "WorkspaceError",
    "WorkspaceInvalidPath",
    "WorkspaceMoveInvalid",
    "WorkspaceNameConflict",
    "WorkspaceNotFound",
    "WorkspaceRegistry",
    "WorkspaceView",
]
