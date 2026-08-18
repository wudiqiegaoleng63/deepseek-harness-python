from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest

from deepseek_harness.jobs import JobRegistry
from deepseek_harness.llm.types import LlmRequest, StreamChunk
from deepseek_harness.terminal import PtyUnsupportedError, TerminalConfig, TerminalSessionService
from deepseek_harness.tools import (
    PermissionMode,
    ToolContext,
    ToolRegistry,
    WorkspacePolicy,
    install_builtin_tools,
    install_terminal_tools,
)
from deepseek_harness.web import HarnessService


class _Adapter:
    def stream(self, request: LlmRequest) -> AsyncIterator[StreamChunk]:
        del request

        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(kind="text", text="done")
            yield StreamChunk(kind="done", finish_reason="stop")

        return chunks()

    async def aclose(self) -> None:
        return None


def _config(**overrides: object) -> TerminalConfig:
    values: dict[str, object] = {
        "idle_seconds": 0.15,
        "send_timeout_seconds": 2.0,
        "startup_timeout_seconds": 2.0,
        "close_grace_seconds": 0.5,
    }
    values.update(overrides)
    return TerminalConfig(**values)  # type: ignore[arg-type]


def _tools(
    tmp_path: Path,
    terminals: TerminalSessionService,
    *,
    owner: str,
    jobs: JobRegistry | None = None,
) -> tuple[ToolRegistry, list[Callable[[], None]]]:
    registry = ToolRegistry()
    disposers = install_terminal_tools(
        registry,
        terminals,
        WorkspacePolicy(tmp_path, PermissionMode.DANGER_FULL_ACCESS),
        jobs=jobs,
        owner_session=owner,
    )
    return registry, disposers


def test_terminal_persists_shell_state_and_reads_bounded_pages(tmp_path: Path) -> None:
    async def scenario() -> None:
        terminals = TerminalSessionService(_config())
        registry, disposers = _tools(tmp_path, terminals, owner="session-a")
        context = ToolContext("session-a", str(tmp_path))
        try:
            opened = await registry.execute(
                "terminal_open",
                json.dumps({"type": "shell", "name": "main"}),
                context,
            )
            assert not opened.is_error
            assert opened.meta is not None
            assert opened.meta["sessionId"] == "pty-1"
            assert opened.meta["status"] == {"kind": "running"}

            first = await registry.execute(
                "terminal_send",
                json.dumps({"sessionId": "pty-1", "text": "pwd"}),
                context,
            )
            assert not first.is_error
            assert str(tmp_path) in first.text

            changed = await registry.execute(
                "terminal_send",
                json.dumps({"sessionId": "pty-1", "text": "cd /tmp"}),
                context,
            )
            assert not changed.is_error
            second = await registry.execute(
                "terminal_send",
                json.dumps({"sessionId": "pty-1", "text": "pwd"}),
                context,
            )
            assert "/tmp" in second.text

            listed = await registry.execute("terminal_list", "{}", context)
            assert listed.meta is not None
            listed_sessions = listed.meta["sessions"]
            assert isinstance(listed_sessions, list)
            assert listed_sessions[0]["sessionId"] == opened.meta["sessionId"]
            assert listed_sessions[0]["name"] == "main"
            page = await registry.execute(
                "terminal_read",
                json.dumps({"sessionId": "pty-1", "offset": 0, "count": 2}),
                context,
            )
            assert page.meta is not None
            assert page.meta["totalLines"] >= 2
            assert page.meta["lineBegin"] == 0
            assert page.meta["lineEnd"] <= 2
        finally:
            await terminals.close_all()
            for dispose in reversed(disposers):
                dispose()

    asyncio.run(scenario())


def test_terminal_owner_isolation_and_duplicate_names(tmp_path: Path) -> None:
    async def scenario() -> None:
        terminals = TerminalSessionService(_config())
        first, first_disposers = _tools(tmp_path, terminals, owner="one")
        second, second_disposers = _tools(tmp_path, terminals, owner="two")
        one = ToolContext("one", str(tmp_path))
        two = ToolContext("two", str(tmp_path))
        try:
            opened = await first.execute(
                "terminal_open", '{"type":"shell","name":"main"}', one
            )
            assert not opened.is_error
            duplicate = await first.execute(
                "terminal_open", '{"type":"shell","name":"main"}', one
            )
            assert duplicate.is_error
            foreign = await second.execute(
                "terminal_read", '{"sessionId":"pty-1"}', two
            )
            assert foreign.is_error
            listed = await second.execute("terminal_list", "{}", two)
            assert listed.meta == {"sessions": []}
        finally:
            await terminals.close_all()
            for dispose in (*first_disposers, *second_disposers):
                dispose()

    asyncio.run(scenario())


def test_terminal_rejects_concurrent_send_and_supports_signals(tmp_path: Path) -> None:
    async def scenario() -> None:
        terminals = TerminalSessionService(_config())
        registry, disposers = _tools(tmp_path, terminals, owner="signals")
        context = ToolContext("signals", str(tmp_path))
        try:
            await registry.execute("terminal_open", '{"type":"shell"}', context)
            operation = terminals.start_send("signals", "pty-1", text="sleep 5", submit=True)
            with pytest.raises(Exception, match="active send"):
                terminals.start_send("signals", "pty-1", text="pwd", submit=True)
            await asyncio.sleep(0.1)
            signalled = await registry.execute(
                "terminal_signal",
                '{"sessionId":"pty-1","signal":"SIGINT"}',
                context,
            )
            assert not signalled.is_error
            assert signalled.meta is not None
            assert signalled.meta["delivered"] is True
            await asyncio.wait_for(operation.done, timeout=2)
            rejected = await registry.execute(
                "terminal_signal",
                '{"sessionId":"pty-1","signal":"SIGKILL"}',
                context,
            )
            assert rejected.is_error
            assert "terminal_close" in rejected.text
        finally:
            await terminals.close_all()
            for dispose in reversed(disposers):
                dispose()

    asyncio.run(scenario())


def test_terminal_does_not_treat_nested_prompt_as_shell_readiness(tmp_path: Path) -> None:
    async def scenario() -> None:
        terminals = TerminalSessionService(_config(idle_seconds=0.1))
        registry, disposers = _tools(tmp_path, terminals, owner="nested")
        context = ToolContext("nested", str(tmp_path))
        try:
            await registry.execute("terminal_open", '{"type":"shell"}', context)
            result = await registry.execute(
                "terminal_send",
                '{"sessionId":"pty-1","text":"bash -i"}',
                context,
            )
            assert not result.is_error
            assert result.meta is not None
            assert result.meta["waitReason"] == "inferred_idle"
        finally:
            await terminals.close_all()
            for dispose in reversed(disposers):
                dispose()

    asyncio.run(scenario())


def test_terminal_send_and_read_output_are_bounded(tmp_path: Path) -> None:
    async def scenario() -> None:
        terminals = TerminalSessionService(_config(max_read_bytes=128, scrollback_bytes=256))
        registry, disposers = _tools(tmp_path, terminals, owner="bounded")
        context = ToolContext("bounded", str(tmp_path))
        try:
            await registry.execute("terminal_open", '{"type":"shell"}', context)
            result = await registry.execute(
                "terminal_send",
                json.dumps({"sessionId": "pty-1", "text": "yes x | head -c 2000"}),
                context,
            )
            assert not result.is_error
            assert result.meta is not None
            assert result.meta["truncated"] is True
            assert len(result.text.encode()) <= 256 * 1024
            read = await registry.execute(
                "terminal_read", '{"sessionId":"pty-1","count":500}', context
            )
            assert read.meta is not None
            assert read.meta["truncated"] is True
            assert len(read.meta["text"].encode()) <= 128
        finally:
            await terminals.close_all()
            for dispose in reversed(disposers):
                dispose()

    asyncio.run(scenario())


def test_background_terminal_send_reuses_job_registry(tmp_path: Path) -> None:
    async def scenario() -> None:
        terminals = TerminalSessionService(_config())
        jobs = JobRegistry()
        registry = ToolRegistry()
        disposers = [
            *install_builtin_tools(
                registry,
                WorkspacePolicy(tmp_path, PermissionMode.DANGER_FULL_ACCESS),
                enable_shell=True,
                jobs=jobs,
            ),
            *install_terminal_tools(
                registry,
                terminals,
                WorkspacePolicy(tmp_path, PermissionMode.DANGER_FULL_ACCESS),
                jobs=jobs,
                owner_session="background",
            ),
        ]
        context = ToolContext("background", str(tmp_path))
        try:
            await registry.execute("terminal_open", '{"type":"shell"}', context)
            started = await registry.execute(
                "terminal_send",
                '{"sessionId":"pty-1","text":"sleep 5","run_in_background":true}',
                context,
            )
            assert not started.is_error
            job_id = started.meta["jobId"] if started.meta else ""
            killed = await registry.execute(
                "job_kill", json.dumps({"job_id": job_id}), context
            )
            assert not killed.is_error
            output = await registry.execute(
                "job_output",
                json.dumps({"job_id": job_id, "wait": True, "timeout_ms": 2000}),
                context,
            )
            assert not output.is_error
            assert "status: killed" in output.text
        finally:
            await terminals.close_all()
            await jobs.close()
            for dispose in reversed(disposers):
                dispose()

    asyncio.run(scenario())


def test_service_only_registers_terminals_in_danger_mode(tmp_path: Path) -> None:
    async def scenario() -> None:
        service = HarnessService(
            tmp_path / "state",
            cwd=tmp_path,
            adapter_factory=lambda _model: _Adapter(),
        )
        handle = await service.create_session(session_id="permission", cwd=str(tmp_path))
        try:
            registry = service._tool_registries[handle.session.id]
            assert "terminal_open" not in registry.names()
            await service.dispatch(
                "permission.set",
                {"sessionId": "permission", "preset": "danger-full-access"},
            )
            assert {
                "terminal_open",
                "terminal_send",
                "terminal_read",
                "terminal_list",
                "terminal_signal",
                "terminal_close",
            } <= set(registry.names())
            opened = await registry.execute(
                "terminal_open", '{"type":"shell"}', ToolContext("permission", str(tmp_path))
            )
            assert not opened.is_error
        finally:
            await service.dispose()

    asyncio.run(scenario())


def test_service_dispose_closes_terminal_processes(tmp_path: Path) -> None:
    async def scenario() -> None:
        service = HarnessService(
            tmp_path / "state",
            cwd=tmp_path,
            permission_mode=PermissionMode.DANGER_FULL_ACCESS,
            adapter_factory=lambda _model: _Adapter(),
        )
        handle = await service.create_session(session_id="dispose", cwd=str(tmp_path))
        registry = service._tool_registries[handle.session.id]
        opened = await registry.execute(
            "terminal_open", '{"type":"shell"}', ToolContext("dispose", str(tmp_path))
        )
        assert opened.meta is not None
        pid = int(opened.meta["pid"])
        await service.dispose()
        assert service.terminals.list("dispose") == []
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)

    asyncio.run(scenario())


def test_service_dispose_waits_for_pending_terminal_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import deepseek_harness.terminal as terminal_module

        started = asyncio.Event()
        release = asyncio.Event()
        original_spawn = terminal_module._PtySession.spawn

        async def delayed_spawn(
            cls: object,
            cwd: Path,
            config: TerminalConfig,
            session_id: str,
            owner: str,
        ) -> object:
            del cls
            started.set()
            await release.wait()
            return await original_spawn(cwd, config, session_id, owner)

        monkeypatch.setattr(
            terminal_module._PtySession,
            "spawn",
            classmethod(delayed_spawn),
        )
        terminals = TerminalSessionService(_config())
        pending = asyncio.create_task(
            terminals.spawn("pending", type="shell", cwd=tmp_path)
        )
        await started.wait()
        disposing = asyncio.create_task(terminals.close_all())
        await asyncio.sleep(0)
        assert not disposing.done()
        release.set()
        with pytest.raises(Exception, match="disposing"):
            await pending
        await disposing
        assert terminals.list("pending") == []

    asyncio.run(scenario())


def test_unsupported_platform_has_clear_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import deepseek_harness.terminal as terminal_module

        monkeypatch.setattr(terminal_module, "pty_supported", lambda: False)
        with pytest.raises(PtyUnsupportedError, match="require a POSIX platform"):
            await TerminalSessionService().spawn(
                "unsupported", type="shell", cwd=tmp_path
            )

    asyncio.run(scenario())
