"""Local process-confinement seam mirroring the TS sandbox family.

The TypeScript runtime wraps every confined shell and terminal argv with a
platform backend before spawning.  This module provides the bubblewrap backend
for Linux with the same profile shape.  The seam fails closed: a confined
execution without a usable backend is an explicit error, never a silent
fallback.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

SandboxMode = Literal["read-only", "workspace-write"]


class SandboxUnavailableError(RuntimeError):
    """A confined execution was requested without a usable sandbox backend."""


@dataclass(frozen=True, slots=True)
class SandboxExecutionPolicy:
    """File-effect policy for one confined execution."""

    mode: SandboxMode
    workspace_root: str


class SandboxProvider(Protocol):
    """Wraps one argv so its file effects honor the policy."""

    def is_available(self) -> bool: ...

    def confine(
        self,
        argv: Sequence[str],
        policy: SandboxExecutionPolicy,
    ) -> list[str]: ...


class BubblewrapSandbox:
    """bubblewrap backend matching the TS ``bwrapProfileArgs`` profile.

    Everything is bound read-only; ``workspace-write`` additionally mounts a
    private ``/tmp`` and binds the workspace root read-write.  Network and
    process visibility are deliberately outside this vocabulary, matching the
    TS contract.
    """

    def __init__(self, executable: str = "bwrap") -> None:
        self._executable = executable

    def is_available(self) -> bool:
        return shutil.which(self._executable) is not None

    def confine(
        self,
        argv: Sequence[str],
        policy: SandboxExecutionPolicy,
    ) -> list[str]:
        executable = shutil.which(self._executable)
        if executable is None:
            raise SandboxUnavailableError(
                f'workspace-write confinement requires "{self._executable}" '
                "(bubblewrap); install it or switch to danger-full-access"
            )
        args = [
            executable,
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--die-with-parent",
        ]
        if policy.mode == "workspace-write":
            args += [
                "--tmpfs",
                "/tmp",
                "--bind",
                policy.workspace_root,
                policy.workspace_root,
            ]
        return [*args, "--", *argv]


class UnavailableSandbox:
    """Provider that never confines; disables the sandbox explicitly."""

    def is_available(self) -> bool:
        return False

    def confine(
        self,
        argv: Sequence[str],
        policy: SandboxExecutionPolicy,
    ) -> list[str]:
        del argv, policy
        raise SandboxUnavailableError("the sandbox is disabled by configuration")


__all__ = [
    "BubblewrapSandbox",
    "SandboxExecutionPolicy",
    "SandboxMode",
    "SandboxProvider",
    "SandboxUnavailableError",
    "UnavailableSandbox",
]
