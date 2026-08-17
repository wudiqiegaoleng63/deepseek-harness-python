"""Initial local workspace policy seam.

The policy deliberately does not pretend to be an OS sandbox.  It enforces
path boundaries in Python now; a later platform provider will wrap child
processes with Landlock/ACL restrictions before enabling untrusted shell work.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class PermissionMode(StrEnum):
    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    DANGER_FULL_ACCESS = "danger-full-access"


@dataclass(slots=True)
class WorkspacePolicy:
    root: Path
    mode: PermissionMode = PermissionMode.WORKSPACE_WRITE
    extra_read_roots: Sequence[Path] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.expanduser().resolve())
        object.__setattr__(
            self,
            "extra_read_roots",
            tuple(path.expanduser().resolve() for path in self.extra_read_roots),
        )

    def resolve(self, raw_path: str) -> Path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        return candidate.resolve(strict=False)

    def assert_readable(self, raw_path: str) -> Path:
        path = self.resolve(raw_path)
        if self.mode is PermissionMode.DANGER_FULL_ACCESS:
            return path
        if not self._inside(path, self.root) and not any(
            self._inside(path, extra_root) for extra_root in self.extra_read_roots
        ):
            raise PermissionError(f"path escapes the workspace: {path}")
        return path

    def assert_writable(self, raw_path: str) -> Path:
        if self.mode is PermissionMode.READ_ONLY:
            raise PermissionError("the current permission mode is read-only")
        path = self.resolve(raw_path)
        if self.mode is not PermissionMode.DANGER_FULL_ACCESS:
            self._assert_inside(path)
        return path

    def assert_shell_allowed(self) -> None:
        if self.mode is PermissionMode.READ_ONLY:
            raise PermissionError("shell execution is disabled in read-only mode")
        if self.mode is PermissionMode.WORKSPACE_WRITE:
            raise PermissionError(
                "shell execution requires a platform sandbox or danger-full-access mode"
            )

    def set_mode(self, mode: PermissionMode) -> None:
        """Change the live session mode after a durable permission event."""

        if not isinstance(mode, PermissionMode):
            raise TypeError("mode must be a PermissionMode")
        self.mode = mode

    def _assert_inside(self, path: Path) -> None:
        if not self._inside(path, self.root):
            raise PermissionError(f"path escapes the workspace: {path}")

    @staticmethod
    def _inside(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
        except ValueError:
            return False
        return True
