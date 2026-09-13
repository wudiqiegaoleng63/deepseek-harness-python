"""Harness-home identity for feedback and telemetry acknowledgements.

The anonymous user id is a random UUID persisted as a bare line in
``.anonymous-user-id`` inside the harness home (``$DSH_HOME`` else ``~/.dsh``).
It is scoped to that home, carries no account or machine identity, and lives
exactly as long as the file does: deleting it mints a fresh identity on the
next call.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

#: File inside the harness home storing the id as a bare UUID line.
ANONYMOUS_USER_ID_FILE = ".anonymous-user-id"

#: Memoized per resolved path so one process touches the disk once, while a
#: file deleted mid-run keeps the process's established identity.
_CACHE: dict[str, str] = {}


def resolve_dsh_home(home: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the harness home: an explicit value, else ``$DSH_HOME``, else ``~/.dsh``."""

    if home is not None:
        return Path(home).expanduser()
    configured = os.environ.get("DSH_HOME")
    return Path(configured).expanduser() if configured else Path("~/.dsh").expanduser()


def get_or_create_anonymous_user_id(home: str | os.PathLike[str] | None = None) -> str:
    """Return this harness home's anonymous user id, minting and persisting it once."""

    path = resolve_dsh_home(home) / ANONYMOUS_USER_ID_FILE
    key = str(path)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    existing = ""
    try:
        existing = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        existing = ""
    except OSError:
        existing = ""
    if existing:
        _CACHE[key] = existing
        return existing
    value = str(uuid.uuid4())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value + "\n", encoding="utf-8")
    except OSError:
        # An unwritable home degrades to a process-local identity rather than
        # failing the command that asked for it.
        pass
    _CACHE[key] = value
    return value


__all__ = [
    "ANONYMOUS_USER_ID_FILE",
    "get_or_create_anonymous_user_id",
    "resolve_dsh_home",
]
