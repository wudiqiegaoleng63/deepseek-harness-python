from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest

from deepseek_harness.jobs import JobRegistry
from deepseek_harness.llm.types import LlmRequest, StreamChunk
from deepseek_harness.sandbox import BubblewrapSandbox, UnavailableSandbox
from deepseek_harness.terminal import PtyUnsupportedError, TerminalConfig, TerminalSessionService
from deepseek_harness.tools import (
    PermissionMode,
    ToolContext,
    ToolRegistry,
    ToolResult,
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
            assert first.meta is not None
            assert first.meta["waitReason"] == "stdin_read"

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


def test_terminal_exact_probe_settles_repl_stdin_read(tmp_path: Path) -> None:
    async def scenario() -> None:
        terminals = TerminalSessionService(_config(idle_seconds=5.0))
        registry, disposers = _tools(tmp_path, terminals, owner="repl")
        context = ToolContext("repl", str(tmp_path))
        try:
            await registry.execute("terminal_open", '{"type":"shell"}', context)
            result = await registry.execute(
                "terminal_send",
                '{"sessionId":"pty-1","text":"bash -i"}',
                context,
            )
            assert not result.is_error
            assert result.meta is not None
            # The nested interactive bash blocks on a tty read; the exact
            # probe settles stdin_read instead of waiting for silence.
            assert result.meta["waitReason"] == "stdin_read"
        finally:
            await terminals.close_all()
            for dispose in reversed(disposers):
                dispose()

    asyncio.run(scenario())


def test_terminal_long_running_command_is_not_reported_as_stdin_read(tmp_path: Path) -> None:
    async def scenario() -> None:
        terminals = TerminalSessionService(_config(idle_seconds=0.15))
        registry, disposers = _tools(tmp_path, terminals, owner="long")
        context = ToolContext("long", str(tmp_path))
        try:
            await registry.execute("terminal_open", '{"type":"shell"}', context)
            result = await registry.execute(
                "terminal_send",
                '{"sessionId":"pty-1","text":"sleep 1"}',
                context,
            )
            assert not result.is_error
            assert result.meta is not None
            # sleep blocks in a timer, never in a tty read, so the exact
            # probe must not claim stdin readiness; silence settles idle.
            assert result.meta["waitReason"] == "inferred_idle"
        finally:
            await terminals.close_all()
            for dispose in reversed(disposers):
                dispose()

    asyncio.run(scenario())


def test_terminal_sets_window_size_for_tty_programs(tmp_path: Path) -> None:
    async def scenario() -> None:
        terminals = TerminalSessionService(_config())
        registry, disposers = _tools(tmp_path, terminals, owner="winsize")
        context = ToolContext("winsize", str(tmp_path))
        try:
            await registry.execute("terminal_open", '{"type":"shell"}', context)
            result = await registry.execute(
                "terminal_send",
                '{"sessionId":"pty-1","text":"stty size"}',
                context,
            )
            assert not result.is_error
            assert "40 160" in result.text
        finally:
            await terminals.close_all()
            for dispose in reversed(disposers):
                dispose()

    asyncio.run(scenario())


def test_terminal_input_read_blocks_in_exact_stdin_probe(tmp_path: Path) -> None:
    async def scenario() -> None:
        terminals = TerminalSessionService(_config(idle_seconds=5.0))
        registry, disposers = _tools(tmp_path, terminals, owner="stdin")
        context = ToolContext("stdin", str(tmp_path))
        try:
            await registry.execute("terminal_open", '{"type":"shell"}', context)
            result = await registry.execute(
                "terminal_send",
                """{"sessionId":"pty-1","text":"python3 -c 'input()'"}""",
                context,
            )
            assert not result.is_error
            assert result.meta is not None
            # A python process blocked in read(0) is genuinely waiting for
            # input; the probe settles long before the idle bound could fire.
            assert result.meta["waitReason"] == "stdin_read"
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
            sandbox_provider=UnavailableSandbox(),
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
            *,
            sandbox: object = None,
            sandbox_policy: object = None,
        ) -> object:
            del cls, sandbox, sandbox_policy
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


def test_service_registers_workspace_write_terminals_under_sandbox(tmp_path: Path) -> None:
    async def scenario() -> None:
        service = HarnessService(
            tmp_path / "state",
            cwd=tmp_path,
            adapter_factory=lambda _model: _Adapter(),
            sandbox_provider=BubblewrapSandbox(),
        )
        handle = await service.create_session(session_id="sandboxed", cwd=str(tmp_path))
        try:
            registry = service._tool_registries[handle.session.id]
            assert "terminal_open" in registry.names()
            opened = await registry.execute(
                "terminal_open", '{"type":"shell"}', ToolContext("sandboxed", str(tmp_path))
            )
            assert not opened.is_error
            sent = await registry.execute(
                "terminal_send",
                '{"sessionId":"pty-1","text":"pwd"}',
                ToolContext("sandboxed", str(tmp_path)),
            )
            assert not sent.is_error
            assert str(tmp_path) in sent.text
        finally:
            await service.dispose()

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


def test_terminal_natural_exit_releases_fd_and_rejects_signals(tmp_path: Path) -> None:
    async def scenario() -> None:
        terminals = TerminalSessionService(_config())
        registry, disposers = _tools(tmp_path, terminals, owner="natural")
        context = ToolContext("natural", str(tmp_path))
        try:
            opened = await registry.execute("terminal_open", '{"type":"shell"}', context)
            assert opened.meta is not None
            pid = int(opened.meta["pid"])
            sent = await registry.execute(
                "terminal_send", '{"sessionId":"pty-1","text":"exit"}', context
            )
            assert sent.meta is not None
            assert sent.meta["waitReason"] == "session_exit"
            assert sent.meta["sessionStatus"]["kind"] == "exited"
            record = terminals._sessions["pty-1"]
            assert record.session._master_closed is True
            listed = await registry.execute("terminal_list", "{}", context)
            assert listed.meta is not None
            assert listed.meta["sessions"][0]["status"]["kind"] == "exited"
            rejected = await registry.execute(
                "terminal_signal", '{"sessionId":"pty-1","signal":"SIGINT"}', context
            )
            assert rejected.is_error
            assert "exited" in rejected.text
            closed = await registry.execute("terminal_close", '{"sessionId":"pty-1"}', context)
            assert closed.meta == {"sessionId": "pty-1", "outcome": "closed"}
            assert terminals.list("natural") == []
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
        finally:
            for dispose in reversed(disposers):
                dispose()

    asyncio.run(scenario())


def test_sanitizer_joins_split_utf8_and_bounds_pending() -> None:
    from deepseek_harness.terminal import _TerminalSanitizer

    sanitizer = _TerminalSanitizer()
    raw = "界".encode()
    first, _ = sanitizer.feed(raw[:1])
    second, _ = sanitizer.feed(raw[1:])
    assert "�" not in first + second
    assert "界" in first + second

    bounded = _TerminalSanitizer()
    head, _ = bounded.feed(b"\x1b]133;D;0")
    assert head == ""
    bulk, _ = bounded.feed(b"x" * 70_000)
    assert bulk == ""
    assert len(bounded._pending.encode()) <= 65_536
    tail, _ = bounded.feed(b"\x07after")
    assert "after" in tail


def test_terminal_enforces_owner_session_quota(tmp_path: Path) -> None:
    async def scenario() -> None:
        terminals = TerminalSessionService(_config(max_sessions_per_owner=1))
        registry, disposers = _tools(tmp_path, terminals, owner="quota")
        context = ToolContext("quota", str(tmp_path))
        try:
            first = await registry.execute("terminal_open", '{"type":"shell"}', context)
            assert not first.is_error
            second = await registry.execute("terminal_open", '{"type":"shell"}', context)
            assert second.is_error
            assert "limit" in second.text
        finally:
            await terminals.close_all()
            for dispose in reversed(disposers):
                dispose()

    asyncio.run(scenario())


def test_close_owner_rejects_late_spawn_publication(
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
            *,
            sandbox: object = None,
            sandbox_policy: object = None,
        ) -> object:
            del cls, sandbox, sandbox_policy
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
            terminals.spawn("late", type="shell", cwd=tmp_path)
        )
        await started.wait()
        closing = asyncio.create_task(terminals.close_owner("late"))
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()
        with pytest.raises(Exception, match="disposed"):
            await pending
        await closing
        assert terminals.list("late") == []

    asyncio.run(scenario())


def test_cancelled_foreground_send_interrupts_and_frees_slot(tmp_path: Path) -> None:
    async def scenario() -> None:
        terminals = TerminalSessionService(_config(idle_seconds=0.1))
        registry, disposers = _tools(tmp_path, terminals, owner="cancel")
        context = ToolContext("cancel", str(tmp_path))
        spam = tmp_path / "spam.py"
        spam.write_text(
            "import time\nwhile True:\n    print('x', flush=True)\n    time.sleep(0.05)\n",
            encoding="utf-8",
        )
        try:
            await registry.execute("terminal_open", '{"type":"shell"}', context)
            task = asyncio.create_task(
                registry.execute(
                    "terminal_send",
                    json.dumps({"sessionId": "pty-1", "text": f"python3 {spam}"}),
                    context,
                )
            )
            await asyncio.sleep(0.2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            # The cancelled send interrupts the foreground command and frees
            # the slot once the shell prompt returns; retry until then.
            follow: ToolResult | None = None
            for _ in range(50):
                follow = await registry.execute(
                    "terminal_send", '{"sessionId":"pty-1","text":"echo freed"}', context
                )
                if not follow.is_error:
                    break
                await asyncio.sleep(0.1)
            assert follow is not None and not follow.is_error
            assert "freed" in follow.text
        finally:
            await terminals.close_all()
            for dispose in reversed(disposers):
                dispose()

    asyncio.run(scenario())


def test_background_cancel_escalates_past_trapped_sigint(tmp_path: Path) -> None:
    async def scenario() -> None:
        terminals = TerminalSessionService(
            _config(cancel_grace_seconds=0.3, idle_seconds=0.1)
        )
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
                owner_session="escalate",
            ),
        ]
        context = ToolContext("escalate", str(tmp_path))
        script = tmp_path / "ignore-int.py"
        script.write_text(
            "import signal, time\n"
            "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            "while True:\n"
            "    print('x', flush=True)\n"
            "    time.sleep(0.05)\n",
            encoding="utf-8",
        )
        try:
            await registry.execute("terminal_open", '{"type":"shell"}', context)
            started = await registry.execute(
                "terminal_send",
                json.dumps(
                    {
                        "sessionId": "pty-1",
                        "text": f"python3 {script}",
                        "run_in_background": True,
                    }
                ),
                context,
            )
            assert started.meta is not None
            job_id = str(started.meta["jobId"])
            killed = await registry.execute(
                "job_kill", json.dumps({"job_id": job_id}), context
            )
            assert not killed.is_error
            output = await asyncio.wait_for(
                registry.execute(
                    "job_output",
                    json.dumps({"job_id": job_id, "wait": True, "timeout_ms": 5000}),
                    context,
                ),
                timeout=8,
            )
            assert "[status: killed" in output.text
        finally:
            await terminals.close_all()
            await jobs.close()
            for dispose in reversed(disposers):
                dispose()

    asyncio.run(scenario())
