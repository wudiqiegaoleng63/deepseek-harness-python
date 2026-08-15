"""Filesystem-backed skill discovery for the host ``skill.list`` surface."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class SkillSummary:
    name: str
    description: str
    when_to_use: str | None = None
    model_invocable: bool = True
    user_invocable: bool = True

    def to_wire(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "modelInvocable": self.model_invocable,
        }
        if self.when_to_use is not None:
            result["whenToUse"] = self.when_to_use
        return result


class SkillRegistry:
    """Discover project and user ``SKILL.md`` files without loading bodies."""

    async def list(self, cwd: str | os.PathLike[str]) -> tuple[SkillSummary, ...]:
        return await asyncio.to_thread(self._list_sync, Path(cwd))

    @classmethod
    def _list_sync(cls, cwd: Path) -> tuple[SkillSummary, ...]:
        project_root = cls._project_root(cwd.expanduser().resolve())
        roots = (
            project_root / ".dsh" / "skills",
            project_root / ".agents" / "skills",
            project_root / ".claude" / "skills",
            Path(os.getenv("DSH_HOME", "~/.dsh")).expanduser() / "skills",
            Path(os.getenv("DSH_AGENTS_HOME", "~/.agents")).expanduser() / "skills",
        )
        # Higher-ranked roots override lower-ranked roots, while the stable
        # traversal order makes duplicate resolution deterministic.
        discovered: dict[str, tuple[int, SkillSummary]] = {}
        for rank, root in enumerate(roots):
            for path in cls._skill_files(root):
                skill = cls._parse(path)
                if skill is None or not skill.user_invocable:
                    continue
                current = discovered.get(skill.name)
                if current is None or rank >= current[0]:
                    discovered[skill.name] = (rank, skill)
        return tuple(
            skill for _, skill in sorted(discovered.values(), key=lambda item: item[1].name)
        )

    @staticmethod
    def _project_root(cwd: Path) -> Path:
        for candidate in (cwd, *cwd.parents):
            if (candidate / ".git").exists():
                return candidate
        return cwd

    @staticmethod
    def _skill_files(root: Path) -> tuple[Path, ...]:
        if not root.is_dir():
            return ()
        files: list[Path] = []
        try:
            entries = sorted(root.iterdir(), key=lambda item: item.name.casefold())
        except OSError:
            return ()
        for entry in entries:
            if entry.is_dir():
                candidate = entry / "SKILL.md"
                if candidate.is_file():
                    files.append(candidate)
            elif entry.is_file() and entry.suffix.casefold() == ".md":
                files.append(entry)
        return tuple(files)

    @staticmethod
    def _parse(path: Path) -> SkillSummary | None:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        if not text.startswith("---"):
            return None
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return None
        end = next((index for index in range(1, len(lines)) if lines[index].strip() == "---"), None)
        if end is None:
            return None
        try:
            frontmatter = yaml.safe_load("\n".join(lines[1:end]))
        except yaml.YAMLError:
            return None
        if not isinstance(frontmatter, dict):
            return None
        fallback_name = path.parent.name if path.name == "SKILL.md" else path.stem
        name = frontmatter.get("name", fallback_name)
        description = frontmatter.get("description")
        if not isinstance(name, str) or not name.strip():
            return None
        if not isinstance(description, str) or not description.strip():
            return None
        when_to_use = frontmatter.get("whenToUse")
        if when_to_use is not None and not isinstance(when_to_use, str):
            when_to_use = None
        model_invocable = frontmatter.get("disable-model-invocation") is not True
        if isinstance(frontmatter.get("modelInvocable"), bool):
            model_invocable = frontmatter["modelInvocable"]
        user_invocable = frontmatter.get("user-invocable") is not False
        if isinstance(frontmatter.get("userInvocable"), bool):
            user_invocable = frontmatter["userInvocable"]
        return SkillSummary(
            name.strip(),
            description.strip(),
            when_to_use.strip() if isinstance(when_to_use, str) and when_to_use.strip() else None,
            model_invocable,
            user_invocable,
        )


__all__ = ["SkillRegistry", "SkillSummary"]
