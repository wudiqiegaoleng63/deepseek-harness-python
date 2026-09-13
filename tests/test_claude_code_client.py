from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from deepseek_harness.claude_code_client import (
    ClaudeCodeError,
    ClaudeCodeRunSpec,
    ClaudeCodeStartupCancelled,
    cli_arguments,
    read_cli_events,
    start_claude_run,
    successful_result,
)

FIXTURE = Path(__file__).parent / "claude_child_fixture.py"


def spec(tmp_path: Path, extra_env: dict[str, str] | None = None, **overrides: object):
    env = {name: value for name, value in os.environ.items() if name.startswith("MOCK_")}
    env.update(extra_env or {})
    values: dict[str, object] = {
        "command": sys.executable,
        "args": (str(FIXTURE),),
        "cwd": str(tmp_path),
        "env": env,
        "dispose_eof_grace_ms": 2_000,
    }
    values.update(overrides)
    return ClaudeCodeRunSpec(**values)  # type: ignore[arg-type]


async def wait_for(predicate, timeout: float = 10.0) -> None:
    for _ in range(int(timeout / 0.02)):
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition never became true")


# ------------------------------------------------------------------ result rules


def test_successful_result_requires_a_strict_success() -> None:
    assert (
        successful_result({"subtype": "success", "is_error": False, "result": "answer"}) == "answer"
    )
    for message in (
        {"subtype": "success", "is_error": True, "result": "answer"},
        {"subtype": "success", "is_error": False, "result": "   "},
        {"subtype": "success", "is_error": False},
        {"subtype": "error_max_turns", "errors": ["hit the turn cap"], "result": ""},
        {"subtype": "error_during_execution", "errors": [], "result": "partial"},
    ):
        try:
            successful_result(message)
        except ClaudeCodeError:
            continue
        raise AssertionError(f"{message} must not complete a run")


def test_read_cli_events_requires_a_result_message() -> None:
    lines = [
        '{"type":"system","subtype":"init"}',
        '{"type":"assistant","message":{"content":[{"type":"text","text":"working"}]}}',
        '{"type":"result","subtype":"success","is_error":false,"result":"final answer"}',
        "not json",
    ]
    assert read_cli_events(lines) == "final answer"
    try:
        read_cli_events(['{"type":"system","subtype":"init"}'])
    except ClaudeCodeError as exc:
        assert "without a result" in str(exc)
    else:
        raise AssertionError("a stream without a result must fail")


def test_cli_arguments_carry_the_fixed_non_interactive_flags() -> None:
    arguments = cli_arguments(
        ClaudeCodeRunSpec(
            command="claude", cwd=".", model="opus", extra_args=("--permission-mode", "plan")
        )
    )
    assert arguments[:2] == ["--print", "--output-format"]
    assert "stream-json" in arguments
    assert "--no-session-persistence" in arguments
    assert "AskUserQuestion" in arguments
    assert arguments[-4:] == ["--model", "opus", "--permission-mode", "plan"]


# ------------------------------------------------------------------- one run


def test_run_returns_the_final_answer(tmp_path: Path) -> None:
    async def scenario() -> None:
        run = await start_claude_run(
            "do the task", spec=spec(tmp_path, {"MOCK_CLAUDE_RESULT": "claude says hi"})
        )
        try:
            result = await run.result()
            assert (result.output, result.stop_reason) == ("claude says hi", "completed")
        finally:
            await run.dispose()

    asyncio.run(scenario())


def test_run_sends_the_task_on_stdin_with_the_fixed_flags(tmp_path: Path) -> None:
    async def scenario() -> None:
        stdin_file = tmp_path / "stdin"
        args_file = tmp_path / "args"
        run = await start_claude_run(
            "the whole task",
            spec=spec(
                tmp_path,
                {
                    "MOCK_CLAUDE_STDIN_FILE": str(stdin_file),
                    "MOCK_CLAUDE_ARGS_FILE": str(args_file),
                },
            ),
        )
        try:
            await run.result()
            assert stdin_file.read_text(encoding="utf-8") == "the whole task"
            arguments = args_file.read_text(encoding="utf-8")
            assert "--print" in arguments
            assert "--output-format" in arguments
        finally:
            await run.dispose()

    asyncio.run(scenario())


def test_run_fails_on_every_non_success_result(tmp_path: Path) -> None:
    async def scenario() -> None:
        for env, expected in (
            ({"MOCK_CLAUDE_SUBTYPE": "error_max_turns", "MOCK_CLAUDE_ERRORS": "turn cap"}, "error"),
            ({"MOCK_CLAUDE_IS_ERROR": "1"}, "error"),
            ({"MOCK_CLAUDE_RESULT": "   "}, "error"),
            ({"MOCK_CLAUDE_NO_RESULT": "1"}, "error"),
        ):
            reasons: list[str] = []

            def record(_error: object, reason: str, sink: list[str] = reasons) -> None:
                sink.append(reason)

            run = await start_claude_run(
                "do the task",
                spec=spec(tmp_path, env),
                on_error=record,
            )
            try:
                result = await run.result()
                assert (result.output, result.stop_reason) == ("", expected)
                assert reasons == ["error"]
            finally:
                await run.dispose()

    asyncio.run(scenario())


def test_run_reports_error_when_the_child_exits_non_zero(tmp_path: Path) -> None:
    async def scenario() -> None:
        run = await start_claude_run("do the task", spec=spec(tmp_path, {"MOCK_CLAUDE_CRASH": "1"}))
        try:
            assert (await run.result()).stop_reason == "error"
        finally:
            await run.dispose()
        assert run.returncode == 9

    asyncio.run(scenario())


# ------------------------------------------------------------------ lifecycle


def test_cancel_settles_aborted_without_an_answer(tmp_path: Path) -> None:
    async def scenario() -> None:
        ready = tmp_path / "ready"
        run = await start_claude_run(
            "do the task",
            spec=spec(
                tmp_path,
                {"MOCK_CLAUDE_HANG": "1", "MOCK_CLAUDE_READY_FILE": str(ready)},
                dispose_eof_grace_ms=200,
            ),
        )
        await wait_for(ready.exists)
        run.cancel()
        result = await asyncio.wait_for(run.result(), 5)
        # Cancellation produces no final result, so no partial text is claimed.
        assert (result.output, result.stop_reason) == ("", "aborted")
        await run.dispose()
        assert run.returncode is not None

    asyncio.run(scenario())


def test_startup_cancelled_before_spawn_is_rejected(tmp_path: Path) -> None:
    async def scenario() -> None:
        cancel = asyncio.Event()
        cancel.set()
        try:
            await start_claude_run("do the task", spec=spec(tmp_path), cancel_event=cancel)
        except ClaudeCodeStartupCancelled:
            return
        raise AssertionError("a pre-aborted request must not start")

    asyncio.run(scenario())


def test_spawn_failure_surfaces_without_a_process(tmp_path: Path) -> None:
    async def scenario() -> None:
        try:
            await start_claude_run(
                "do the task", spec=spec(tmp_path, command=str(tmp_path / "missing-claude"))
            )
        except ClaudeCodeError as exc:
            assert "spawn" in str(exc)
            return
        raise AssertionError("a missing executable must fail the start")

    asyncio.run(scenario())


def test_dispose_is_idempotent_and_reaps_the_child(tmp_path: Path) -> None:
    async def scenario() -> None:
        run = await start_claude_run("do the task", spec=spec(tmp_path))
        await run.result()
        await asyncio.gather(run.dispose(), run.dispose())
        assert run.returncode is not None

    asyncio.run(scenario())
