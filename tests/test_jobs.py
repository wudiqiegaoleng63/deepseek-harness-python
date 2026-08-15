from __future__ import annotations

import asyncio
from collections.abc import Awaitable

from deepseek_harness.jobs import JobHandle, JobOutcome, JobRegistry
from deepseek_harness.tools import PermissionMode, WorkspacePolicy, install_builtin_tools
from deepseek_harness.tools.registry import ToolContext, ToolRegistry


def test_job_registry_enforces_owner_and_tracks_terminal_state() -> None:
    async def scenario() -> None:
        changed: list[str | None] = []
        done = asyncio.Event()
        cancelled: list[str | None] = []

        async def producer() -> JobOutcome:
            await done.wait()
            return JobOutcome("completed", "exit code: 0", "final")

        registry = JobRegistry(on_changed=changed.append)
        job_id = await registry.start(
            kind="test",
            label="unit job",
            owner_session="owner-a",
            starter=lambda: _handle(producer(), cancelled),
        )
        assert registry.get(job_id, "owner-a").status == "running"
        try:
            registry.get(job_id, "owner-b")
        except PermissionError:
            pass
        else:
            raise AssertionError("foreign session accessed an owned job")

        assert registry.kill(job_id, "owner-a", "test stop") == "requested"
        assert cancelled == ["test stop"]
        done.set()
        terminal = await registry.wait(job_id, 1_000, "owner-a")
        assert terminal.status == "completed"
        assert terminal.detail == "exit code: 0"
        assert registry.read(job_id, "owner-a").text == "final"
        assert changed[:1] == ["owner-a"]
        assert changed[-1] == "owner-a"
        await registry.close()

    asyncio.run(scenario())


def test_bash_background_job_is_readable_through_model_tools(tmp_path) -> None:
    async def scenario() -> None:
        registry = ToolRegistry()
        jobs = JobRegistry()
        disposers = install_builtin_tools(
            registry,
            WorkspacePolicy(tmp_path, PermissionMode.DANGER_FULL_ACCESS),
            enable_shell=True,
            jobs=jobs,
        )
        context = ToolContext("session-shell", str(tmp_path))
        try:
            started = await registry.execute(
                "bash",
                '{"command":"printf first; sleep 0.05; printf second", "run_in_background":true}',
                context,
            )
            assert not started.is_error
            job_id = started.text.split()[3]
            listed = await registry.execute("job_list", "{}", context)
            assert job_id in listed.text
            output = await registry.execute(
                "job_output",
                f'{{"job_id":"{job_id}","wait":true,"timeout_ms":2000}}',
                context,
            )
            assert not output.is_error
            assert "first" in output.text
            assert "second" in output.text
            assert "[status: completed]" in output.text
        finally:
            for dispose in reversed(disposers):
                dispose()
            await jobs.close()

    asyncio.run(scenario())


async def _handle(done: Awaitable[JobOutcome], cancelled: list[str | None]) -> JobHandle:
    return JobHandle(cancel=lambda reason: cancelled.append(reason), done=done)
