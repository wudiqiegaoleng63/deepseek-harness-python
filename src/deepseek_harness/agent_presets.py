"""Small, file-backed agent-preset roster for the host API."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .persistence import JsonStateStore

_PRESET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class AgentPresetError(Exception):
    def __init__(self, message: str, code: str, preset: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.preset = preset


@dataclass(frozen=True, slots=True)
class AgentPreset:
    id: str
    trust: str
    is_default: bool
    content: str
    name: str | None = None
    description: str | None = None
    broken: str | None = None

    def entry(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "trust": self.trust,
            "isDefault": self.is_default,
        }
        for key, value in (
            ("name", self.name),
            ("description", self.description),
            ("broken", self.broken),
        ):
            if value is not None:
                result[key] = value
        return result


class AgentPresetRegistry:
    """User presets are copy-only and never expose arbitrary file paths."""

    def __init__(self, root: str | Path) -> None:
        self._store = JsonStateStore(
            Path(root).expanduser().resolve() / "agent-presets.json",
            default={"version": 1, "presets": []},
        )
        self._presets: dict[str, AgentPreset] = {}
        self._loaded = False
        self._lock = asyncio.Lock()

    async def list(self) -> tuple[AgentPreset, ...]:
        async with self._lock:
            await self._ensure_loaded()
            return tuple(self._presets.values())

    async def get(self, preset_id: str) -> AgentPreset:
        async with self._lock:
            await self._ensure_loaded()
            preset = self._presets.get(preset_id)
            if preset is None:
                raise AgentPresetError(
                    f'agent preset "{preset_id}" was not found',
                    "agent-preset-not-found",
                    preset_id,
                )
            return preset

    async def copy(self, source_id: str, target_id: str, name: str | None = None) -> AgentPreset:
        self._validate_id(target_id)
        async with self._lock:
            await self._ensure_loaded()
            source = self._presets.get(source_id)
            if source is None:
                raise AgentPresetError(
                    f'agent preset "{source_id}" was not found',
                    "agent-preset-not-found",
                    source_id,
                )
            if target_id in self._presets:
                raise AgentPresetError(
                    f'agent preset "{target_id}" already exists',
                    "agent-preset-invalid",
                    target_id,
                )
            preset = AgentPreset(
                target_id,
                "user",
                False,
                source.content,
                name.strip() if isinstance(name, str) and name.strip() else None,
                source.description,
            )
            self._presets[target_id] = preset
            await self._save()
            return preset

    async def remove(self, preset_id: str) -> None:
        async with self._lock:
            await self._ensure_loaded()
            preset = self._presets.get(preset_id)
            if preset is None:
                raise AgentPresetError(
                    f'agent preset "{preset_id}" was not found',
                    "agent-preset-not-found",
                    preset_id,
                )
            if preset.trust != "user":
                raise AgentPresetError(
                    f'agent preset "{preset_id}" is read-only',
                    "agent-preset-read-only",
                    preset_id,
                )
            del self._presets[preset_id]
            await self._save()

    async def open_document(self, preset_id: str) -> dict[str, Any]:
        preset = await self.get(preset_id)
        if preset.trust != "user":
            raise AgentPresetError(
                f'agent preset "{preset_id}" is read-only',
                "agent-preset-read-only",
                preset_id,
            )
        return {"opened": False, "path": str(self._store.path)}

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        raw = await self._store.load()
        rows = raw.get("presets", [])
        if isinstance(rows, list):
            for item in rows:
                if not isinstance(item, dict):
                    continue
                preset_id = item.get("id")
                content = item.get("content")
                if not isinstance(preset_id, str) or not _PRESET_ID.fullmatch(preset_id):
                    continue
                if not isinstance(content, str):
                    continue
                self._presets[preset_id] = AgentPreset(
                    preset_id,
                    "user",
                    bool(item.get("isDefault", False)),
                    content,
                    item.get("name") if isinstance(item.get("name"), str) else None,
                    item.get("description") if isinstance(item.get("description"), str) else None,
                )
        self._loaded = True

    async def _save(self) -> None:
        await self._store.save(
            {
                "version": 1,
                "presets": [
                    {
                        "id": preset.id,
                        "isDefault": preset.is_default,
                        "content": preset.content,
                        **({"name": preset.name} if preset.name is not None else {}),
                        **(
                            {"description": preset.description}
                            if preset.description is not None
                            else {}
                        ),
                    }
                    for preset in self._presets.values()
                ],
            }
        )

    @staticmethod
    def _validate_id(value: str) -> None:
        if not _PRESET_ID.fullmatch(value):
            raise AgentPresetError(
                "agent preset id must contain only letters, numbers, '.', '_' or '-'",
                "agent-preset-invalid",
                value,
            )


__all__ = ["AgentPreset", "AgentPresetError", "AgentPresetRegistry"]
