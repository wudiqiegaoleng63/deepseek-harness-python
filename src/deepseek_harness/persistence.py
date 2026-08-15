"""Small atomic JSON state stores used by host-owned registries.

Session transcripts have their own JSONL store because they are append-only.
Configuration and registry state is instead represented as one versioned JSON
snapshot.  Keeping the two persistence shapes separate makes corruption and
recovery policy explicit at the call site.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class JsonStateStore:
    """Read and atomically replace one JSON object on disk."""

    def __init__(self, path: str | os.PathLike[str], *, default: dict[str, Any]) -> None:
        self.path = Path(path).expanduser().resolve()
        self.default = default
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def load(self) -> dict[str, Any]:
        try:
            raw = await asyncio.to_thread(self.path.read_text, encoding="utf-8")
        except FileNotFoundError:
            return _copy_object(self.default)
        try:
            value = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid JSON state file: {self.path}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSON state root must be an object: {self.path}")
        return value

    async def save(self, value: dict[str, Any]) -> None:
        encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        await asyncio.to_thread(self._save_sync, encoded)

    def _save_sync(self, encoded: str) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)


def _copy_object(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, ensure_ascii=False))


__all__ = ["JsonStateStore"]
