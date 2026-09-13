"""Child-process environment and teardown rules shared by out-of-process backends.

Every harness child starts from the same scrubbed environment and is torn down
through the same cooperative ladder, so the ACP and SDK subagent providers
cannot drift apart on credentials or on orphaned processes.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import signal
from collections.abc import Mapping

#: EOF grace for child flush and nested-process teardown; wider than the signal grace.
DEFAULT_DISPOSE_EOF_GRACE_MS = 6_000

#: Default POSIX grace between SIGTERM and SIGKILL on dispose.
DEFAULT_DISPOSE_GRACE_MS = 3_000

#: Credential-shaped ambient names never reach a child implicitly.
SENSITIVE_ENV_PATTERN = re.compile(r"KEY|PASSWORD|SECRET|TOKEN", re.IGNORECASE)

#: Ambient names the harness owns; a child gets them only through explicit extras.
DSH_ENV_PREFIX = "DSH_"


def scrubbed_child_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the ambient environment minus credentials and harness-owned names.

    A deliberately supplied entry survives because explicit extras are merged
    after the scrub, so a child's own key or a deployment ``DSH_*`` fact can be
    forwarded on purpose while ambient ones never leak implicitly.
    """

    source = os.environ if env is None else env
    return {
        name: value
        for name, value in source.items()
        if not SENSITIVE_ENV_PATTERN.search(name) and not name.upper().startswith(DSH_ENV_PREFIX)
    }


def child_environment(extras: Mapping[str, str] | None = None) -> dict[str, str]:
    """The scrubbed parent environment with explicit extras layered on top."""

    env = scrubbed_child_env()
    if extras:
        env.update({str(name): str(value) for name, value in extras.items()})
    return env


def _signal_tree(process: asyncio.subprocess.Process, sig: int) -> None:
    try:
        os.killpg(os.getpgid(process.pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        with contextlib.suppress(ProcessLookupError, OSError):
            if sig == signal.SIGKILL:
                process.kill()
            else:
                process.terminate()


async def _wait_for_exit(process: asyncio.subprocess.Process, seconds: float) -> bool:
    try:
        async with asyncio.timeout(seconds):
            await process.wait()
    except TimeoutError:
        return False
    return True


async def dispose_child_process(
    process: asyncio.subprocess.Process | None,
    eof_grace_ms: float = DEFAULT_DISPOSE_EOF_GRACE_MS,
    grace_ms: float = DEFAULT_DISPOSE_GRACE_MS,
) -> None:
    """Cooperative teardown ladder: stdin EOF, then SIGTERM → grace → SIGKILL.

    Resolves only at whole-process quiescence.  A child that already exited (or
    a failed spawn with no process) needs no teardown.
    """

    if process is None or process.returncode is not None:
        return
    if process.stdin is not None:
        with contextlib.suppress(BrokenPipeError, ConnectionResetError, RuntimeError, OSError):
            process.stdin.close()
    if await _wait_for_exit(process, eof_grace_ms / 1000):
        return
    # terminate() owns the bounded SIGTERM→SIGKILL timer; the final wait is the
    # process owner's unbounded exit proof, not a second derived grace.
    _signal_tree(process, signal.SIGTERM)
    if await _wait_for_exit(process, grace_ms / 1000):
        return
    _signal_tree(process, signal.SIGKILL)
    await process.wait()


__all__ = [
    "DEFAULT_DISPOSE_EOF_GRACE_MS",
    "DEFAULT_DISPOSE_GRACE_MS",
    "DSH_ENV_PREFIX",
    "SENSITIVE_ENV_PATTERN",
    "child_environment",
    "dispose_child_process",
    "scrubbed_child_env",
]
