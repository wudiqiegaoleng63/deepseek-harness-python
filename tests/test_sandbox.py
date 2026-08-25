from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest

from deepseek_harness.jobs import JobRegistry
from deepseek_harness.sandbox import (
    BubblewrapSandbox,
    SandboxExecutionPolicy,
    SandboxUnavailableError,
    UnavailableSandbox,
)
from deepseek_harness.tools import (
    PermissionMode,
    ToolContext,
    ToolRegistry,
    WorkspacePolicy,
    install_shell_tools,
)

BWRAP_AVAILABLE = shutil.which("bwrap") is not None
requires_bwrap = pytest.mark.skipif(not BWRAP_AVAILABLE, reason="bubblewrap is not installed")


def test_bwrap_confine_profile_matches_ts_shape(tmp_path: Path) -> None:
    sandbox = BubblewrapSandbox()
    argv = sandbox.confine(
        ["bash", "-i"],
        SandboxExecutionPolicy("workspace-write", str(tmp_path)),
    )
    assert argv[0].endswith("bwrap")
    assert argv[-3] == "--"
    assert argv[-2:] == ["bash", "-i"]
    assert "--ro-bind" in argv
    assert "--die-with-parent" in argv
    index = argv.index("--tmpfs")
    assert argv[index + 1] == "/tmp"
    bind = argv.index("--bind")
    assert argv[bind + 1 : bind + 3] == [str(tmp_path), str(tmp_path)]

    read_only = sandbox.confine(["bash", "-c", ":"], SandboxExecutionPolicy("read-only", "/"))
    assert "--tmpfs" not in read_only
    assert "--bind" not in read_only


def test_unavailable_providers_fail_closed() -> None:
    sandbox = UnavailableSandbox()
    assert sandbox.is_available() is False
    with pytest.raises(SandboxUnavailableError, match="disabled"):
        sandbox.confine(
            ["bash"],
            SandboxExecutionPolicy("workspace-write", "/tmp"),
        )


def test_workspace_write_bash_requires_sandbox(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = ToolRegistry()
        disposers = install_shell_tools(
            registry,
            WorkspacePolicy(tmp_path, PermissionMode.WORKSPACE_WRITE),
            sandbox=UnavailableSandbox(),
        )
        try:
            result = await registry.execute(
                "bash", '{"command":"echo hi"}', ToolContext("sandbox-bash", str(tmp_path))
            )
            assert result.is_error
            assert "bwrap" in result.text
        finally:
            for dispose in reversed(disposers):
                dispose()

    asyncio.run(scenario())


@requires_bwrap
def test_workspace_write_bash_runs_confined(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = ToolRegistry()
        jobs = JobRegistry()
        disposers = install_shell_tools(
            registry,
            WorkspacePolicy(tmp_path, PermissionMode.WORKSPACE_WRITE),
            jobs=jobs,
            sandbox=BubblewrapSandbox(),
        )
        context = ToolContext("sandbox-bash", str(tmp_path))
        try:
            inside = await registry.execute(
                "bash",
                json.dumps(
                    {
                        "command": "printf inside > ok.txt && cat ok.txt",
                    }
                ),
                context,
            )
            assert not inside.is_error
            assert inside.text.strip() == "inside"

            outside = await registry.execute(
                "bash",
                json.dumps({"command": "touch /usr/lib/dsh-sandbox-denied && echo touched"}),
                context,
            )
            assert outside.is_error
            assert not Path("/usr/lib/dsh-sandbox-denied").exists()

            background = await registry.execute(
                "bash",
                json.dumps({"command": "printf bg-done", "run_in_background": True}),
                context,
            )
            assert not background.is_error
            job_id = str(background.text.split()[3])
            output = await registry.execute(
                "job_output",
                json.dumps({"job_id": job_id, "wait": True, "timeout_ms": 3000}),
                context,
            )
            assert "bg-done" in output.text
        finally:
            await jobs.close()
            for dispose in reversed(disposers):
                dispose()

    asyncio.run(scenario())
