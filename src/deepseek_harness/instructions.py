"""Bounded, durable workspace instruction loading.

The TypeScript base bundle loads ``AGENTS.md``/``CLAUDE.md`` instructions into
the model-visible session surface.  This module keeps the same high-value
contract for the Python host: discovery is rooted at the workspace, content is
bounded by UTF-8 bytes, and each rendered baseline carries replayable source
metadata so a changed or removed file produces a new durable context message.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .models import JsonValue, Message, TextContent
from .session import Session

DEFAULT_MAX_INSTRUCTION_BYTES = 65_536
DEFAULT_MAX_INSTRUCTION_SOURCE_BYTES = 1_048_576
DEFAULT_PROJECT_ROOT_MARKERS = (".git",)
DEFAULT_INSTRUCTION_FILE_CANDIDATES = ("AGENTS.md", "CLAUDE.md")
DEFAULT_LOCAL_INSTRUCTION_FILE_CANDIDATES = ("AGENTS.local.md", "CLAUDE.local.md")

_SYSTEM_REMINDER_OPEN = "<system-reminder>"
_SYSTEM_REMINDER_CLOSE = "</system-reminder>"
_BASELINE_INTRO = (
    "The following workspace instructions may be relevant to your work. "
    "Use them as guidance when applicable. More specific instructions take precedence "
    "over broader ones. They do not override system, developer, or direct user instructions."
)
_REPLACEMENT_INTRO = (
    "This complete workspace instruction baseline replaces all earlier workspace "
    f"instruction baselines. {_BASELINE_INTRO}"
)
_EMPTY_REPLACEMENT_INTRO = (
    "This complete workspace instruction baseline replaces all earlier workspace "
    "instruction baselines. No workspace instructions are currently active."
)


@dataclass(frozen=True, slots=True)
class InstructionFile:
    """One instruction file that survived discovery and source-size checks."""

    path: Path
    display_path: str
    scope: str
    content: str
    digest: str


class WorkspaceInstructionLoader:
    """Discover and render workspace instructions for one session.

    The loader intentionally does not retain instruction prose between calls.
    It rescans the bounded instruction chain before each model step, which makes
    edits made through the normal file tools visible on the following step
    without introducing a filesystem watcher.
    """

    def __init__(
        self,
        *,
        dsh_home: str | os.PathLike[str] | None = None,
        max_bytes: int = DEFAULT_MAX_INSTRUCTION_BYTES,
        max_source_bytes: int = DEFAULT_MAX_INSTRUCTION_SOURCE_BYTES,
        project_root_markers: Iterable[str] = DEFAULT_PROJECT_ROOT_MARKERS,
        instruction_file_candidates: Iterable[str] = DEFAULT_INSTRUCTION_FILE_CANDIDATES,
        local_instruction_file_candidates: Iterable[str] = (
            DEFAULT_LOCAL_INSTRUCTION_FILE_CANDIDATES
        ),
    ) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        if (
            isinstance(max_source_bytes, bool)
            or not isinstance(max_source_bytes, int)
            or max_source_bytes <= 0
        ):
            raise ValueError("max_source_bytes must be a positive integer")
        self.max_bytes = max_bytes
        self.max_source_bytes = max_source_bytes
        self.project_root_markers = self._normalize_candidates(project_root_markers)
        self.instruction_file_candidates = self._normalize_candidates(instruction_file_candidates)
        self.local_instruction_file_candidates = self._normalize_candidates(
            local_instruction_file_candidates
        )
        configured_home = dsh_home or os.environ.get("DSH_HOME")
        self.dsh_home = (
            Path(configured_home).expanduser() if configured_home else Path.home() / ".dsh"
        )

    async def prepare(self, session: Session) -> Message | None:
        """Return a new baseline message when the visible baseline is stale."""

        cwd = Path(session.header.cwd or Path.cwd()).expanduser().resolve()
        files = self._discover(cwd)
        current_identity = self._identity(cwd, files)
        previous = self._visible_baseline(session)
        if previous is not None and previous[0] == current_identity:
            return None
        if not files and previous is None:
            return None

        changes = self._changes(previous[1] if previous is not None else (), files)
        text = self._render(files, replacement=previous is not None)
        source: dict[str, JsonValue] = {
            "kind": "agent-instructions",
            "form": "instructions",
            "baseline": True,
            "baselineIdentity": current_identity,
            "changes": changes,
        }
        return Message(
            role="user",
            content=(TextContent(text),),
            source=source,
        )

    def _discover(self, cwd: Path) -> tuple[InstructionFile, ...]:
        root = self._project_root(cwd)
        result: list[InstructionFile] = []

        global_path = self.dsh_home / "AGENTS.md"
        global_file = self._read_file(
            global_path,
            display_path="~/.dsh/AGENTS.md"
            if self.dsh_home == Path.home() / ".dsh"
            else "$DSH_HOME/AGENTS.md",
            scope="user-global\x00AGENTS.md",
        )
        if global_file is not None:
            result.append(global_file)

        seen_by_directory: dict[Path, set[str]] = {}
        for directory in self._ancestor_chain(root, cwd):
            seen = seen_by_directory.setdefault(directory, set())
            candidates = (
                *self.instruction_file_candidates,
                *self.local_instruction_file_candidates,
            )
            for name in candidates:
                path = directory / name
                display_path = os.path.relpath(path, root)
                if display_path == ".":
                    display_path = path.name
                scope_dir = os.path.relpath(directory, root)
                if scope_dir == ".":
                    scope_dir = "."
                loaded = self._read_file(
                    path,
                    display_path=display_path,
                    scope=f"{scope_dir}\x00{name}",
                )
                if loaded is None:
                    continue
                trimmed = loaded.content.strip()
                if trimmed in seen:
                    continue
                seen.add(trimmed)
                result.append(loaded)
        return tuple(result)

    def _read_file(self, path: Path, *, display_path: str, scope: str) -> InstructionFile | None:
        try:
            if not path.is_file() or path.stat().st_size > self.max_source_bytes:
                return None
            raw = path.read_bytes()
            if len(raw) > self.max_source_bytes:
                return None
            content = raw.decode("utf-8")
        except (OSError, UnicodeError):
            return None
        if not content.strip():
            return None
        return InstructionFile(
            path=path,
            display_path=display_path,
            scope=scope,
            content=content,
            digest=hashlib.sha1(raw).hexdigest(),
        )

    def _project_root(self, cwd: Path) -> Path:
        current = cwd
        while True:
            if any((current / marker).exists() for marker in self.project_root_markers):
                return current
            if current.parent == current:
                return cwd
            current = current.parent

    @staticmethod
    def _ancestor_chain(root: Path, cwd: Path) -> tuple[Path, ...]:
        chain: list[Path] = []
        current = cwd
        while current != root:
            chain.append(current)
            if current.parent == current:
                break
            current = current.parent
        chain.append(root)
        return tuple(reversed(chain))

    @staticmethod
    def _normalize_candidates(candidates: Iterable[str]) -> tuple[str, ...]:
        normalized: list[str] = []
        for candidate in candidates:
            if not candidate or candidate in {".", ".."} or "/" in candidate or "\\" in candidate:
                continue
            normalized.append(candidate)
        return tuple(normalized)

    def _identity(self, cwd: Path, files: tuple[InstructionFile, ...]) -> str:
        root = self._project_root(cwd)
        payload = {
            "projectRoot": os.path.relpath(root, cwd),
            "markers": self.project_root_markers,
            "candidates": self.instruction_file_candidates,
            "localCandidates": self.local_instruction_file_candidates,
            "maxBytes": self.max_bytes,
            "maxSourceBytes": self.max_source_bytes,
            "files": [(file.scope, file.display_path, file.digest) for file in files],
        }
        return hashlib.sha1(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest()

    @staticmethod
    def _visible_baseline(
        session: Session,
    ) -> tuple[str, tuple[dict[str, JsonValue], ...]] | None:
        for message in reversed(session.derive_messages()):
            source = message.source
            if source.get("kind") != "agent-instructions" or source.get("baseline") is not True:
                continue
            identity = source.get("baselineIdentity")
            raw_changes = source.get("changes")
            if not isinstance(identity, str) or not isinstance(raw_changes, list):
                continue
            changes = tuple(item for item in raw_changes if isinstance(item, dict))
            return identity, changes
        return None

    @staticmethod
    def _changes(
        previous: tuple[dict[str, JsonValue], ...],
        current: tuple[InstructionFile, ...],
    ) -> list[JsonValue]:
        old = {
            str(item["scope"]): item
            for item in previous
            if isinstance(item.get("scope"), str) and isinstance(item.get("path"), str)
        }
        changes: list[JsonValue] = []
        current_scopes: set[str] = set()
        for file in current:
            current_scopes.add(file.scope)
            old_item = old.get(file.scope)
            action = "set" if old_item is None else "replace"
            if old_item is not None and old_item.get("digest") == file.digest:
                action = "set"
            changes.append(
                {
                    "action": action,
                    "scope": file.scope,
                    "path": file.display_path,
                    "digest": file.digest,
                }
            )
        for scope, item in old.items():
            if scope not in current_scopes:
                changes.append(
                    {
                        "action": "remove",
                        "scope": scope,
                        "path": item["path"],
                    }
                )
        return changes

    def _render(self, files: tuple[InstructionFile, ...], *, replacement: bool) -> str:
        intro = _REPLACEMENT_INTRO if replacement and files else (
            _EMPTY_REPLACEMENT_INTRO if replacement else _BASELINE_INTRO
        )
        sections = [
            f"Instructions from: {self._escape(file.display_path)}\n\n{self._escape(file.content)}"
            for file in files
        ]
        prefix = f"{_SYSTEM_REMINDER_OPEN}\n{intro}\n\n"
        suffix = f"\n{_SYSTEM_REMINDER_CLOSE}"
        if not sections:
            return self._truncate_utf8(prefix + suffix, self.max_bytes)
        budget = self.max_bytes - len((prefix + suffix).encode("utf-8"))
        if budget <= 0:
            return self._truncate_utf8(prefix + suffix, self.max_bytes)

        selected: list[str] = []
        for section in reversed(sections):
            separator_bytes = 2 if selected else 0
            encoded = section.encode("utf-8")
            if len(encoded) + separator_bytes <= budget:
                selected.append(section)
                budget -= len(encoded) + separator_bytes
                continue
            if not selected and budget > 0:
                selected.append(self._truncate_utf8(section, budget))
                budget = 0
        selected.reverse()
        return prefix + ("\n\n".join(selected) if selected else "") + suffix

    @staticmethod
    def _escape(value: str) -> str:
        return value.replace(_SYSTEM_REMINDER_CLOSE, "<\\/system-reminder>")

    @staticmethod
    def _truncate_utf8(value: str, max_bytes: int) -> str:
        raw = value.encode("utf-8")
        if len(raw) <= max_bytes:
            return value
        return raw[:max_bytes].decode("utf-8", errors="ignore")


__all__ = [
    "DEFAULT_INSTRUCTION_FILE_CANDIDATES",
    "DEFAULT_LOCAL_INSTRUCTION_FILE_CANDIDATES",
    "DEFAULT_MAX_INSTRUCTION_BYTES",
    "DEFAULT_MAX_INSTRUCTION_SOURCE_BYTES",
    "DEFAULT_PROJECT_ROOT_MARKERS",
    "InstructionFile",
    "WorkspaceInstructionLoader",
]
