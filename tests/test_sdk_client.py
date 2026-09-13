from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from deepseek_harness.sdk_client import (
    SdkClientError,
    SdkRunSpec,
    SdkStartupCancelled,
    assistant_text,
    start_sdk_run,
    turn_end_to_subagent_reason,
)

FIXTURE = Path(__file__).parent / "sdk_child_fixture.py"


def spec(
    tmp_path: Path, extra_env: dict[str, str] | None = None, **overrides: object
) -> SdkRunSpec:
    env = {name: value for name, value in os.environ.items() if name.startswith("MOCK_")}
    env.update(extra_env or {})
    values: dict[str, object] = {
        "command": sys.executable,
        "args": (str(FIXTURE),),
        "cwd": str(tmp_path),
        "env": env,
        "dispose_eof_grace_ms": 2_000,
        "shutdown_timeout_ms": 1_000,
    }
    values.update(overrides)
    return SdkRunSpec(**values)  # type: ignore[arg-type]


async def wait_for(predicate, timeout: float = 10.0) -> None:
    for _ in range(int(timeout / 0.02)):
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition never became true")


# ------------------------------------------------------------------- pure codec


def test_turn_end_to_subagent_reason_reads_the_child_vocabulary() -> None:
    assert turn_end_to_subagent_reason("completed") == "completed"
    assert turn_end_to_subagent_reason("max-tokens") == "max-tokens"
    assert turn_end_to_subagent_reason("aborted") == "aborted"
    # Every unclean ending is a failure, never a silent success.
    assert turn_end_to_subagent_reason("error") == "error"
    assert turn_end_to_subagent_reason("max-steps") == "error"
    assert turn_end_to_subagent_reason(None) == "error"


def test_assistant_text_prefers_the_last_non_empty_message() -> None:
    chunks = [
        {"type": "assistant/chunk", "data": {"chunk": {"kind": "text", "text": "streamed "}}},
        {"type": "assistant/chunk", "data": {"chunk": {"kind": "text", "text": "text"}}},
    ]
    assert assistant_text(chunks) == "streamed text"
    committed = [
        *chunks,
        {
            "type": "assistant/message",
            "data": {"message": {"content": [{"type": "text", "text": "first"}]}},
        },
        {
            "type": "assistant/message",
            "data": {"message": {"content": [{"type": "text", "text": "final"}]}},
        },
    ]
    assert assistant_text(committed) == "final"
    # An empty-content message that only records usage is skipped.
    usage_only = [
        {
            "type": "assistant/message",
            "data": {"message": {"content": []}, "usage": {"totalTokens": 1}},
        },
        *chunks,
    ]
    assert assistant_text(usage_only) == "streamed text"
    assert assistant_text([]) == ""


# ---------------------------------------------------------------------- a run


def test_run_reads_the_answer_from_child_events(tmp_path: Path) -> None:
    async def scenario() -> None:
        run = await start_sdk_run(
            "do the task", spec=spec(tmp_path, {"MOCK_SDK_TEXT": "child answer"})
        )
        try:
            assert run.id and run.session_id
            result = await run.result()
            assert (result.output, result.stop_reason) == ("child answer", "completed")
        finally:
            await run.dispose()

    asyncio.run(scenario())


def test_run_uses_the_child_message_over_streamed_text(tmp_path: Path) -> None:
    async def scenario() -> None:
        run = await start_sdk_run(
            "do the task",
            spec=spec(
                tmp_path, {"MOCK_SDK_EMPTY_MESSAGE": "1", "MOCK_SDK_TEXT": "streamed fallback"}
            ),
        )
        try:
            # The committed message is empty, so the streamed text is the answer.
            assert (await run.result()).output == "streamed fallback"
        finally:
            await run.dispose()

    asyncio.run(scenario())


def test_run_maps_unclean_turn_endings(tmp_path: Path) -> None:
    async def scenario() -> None:
        for kind, expected in (
            ("max-tokens", "max-tokens"),
            ("aborted", "aborted"),
            ("error", "error"),
            ("disposed", "error"),
        ):
            run = await start_sdk_run(
                "do the task", spec=spec(tmp_path, {"MOCK_SDK_TURN_KIND": kind})
            )
            try:
                assert (await run.result()).stop_reason == expected
            finally:
                await run.dispose()

    asyncio.run(scenario())


def test_run_reports_error_when_the_child_never_ends_its_turn(tmp_path: Path) -> None:
    async def scenario() -> None:
        run = await start_sdk_run(
            "do the task",
            spec=spec(tmp_path, {"MOCK_SDK_NO_TURN_END": "1", "MOCK_SDK_TEXT": "partial"}),
        )
        try:
            result = await run.result()
            # Committed text survives, but an activity without a clean ending
            # is never reported as success.
            assert (result.output, result.stop_reason) == ("partial", "error")
        finally:
            await run.dispose()

    asyncio.run(scenario())


# ------------------------------------------------------------------ lifecycle


def test_cancel_settles_aborted_and_keeps_partial_output(tmp_path: Path) -> None:
    async def scenario() -> None:
        ready = tmp_path / "ready"
        run = await start_sdk_run(
            "do the task",
            spec=spec(
                tmp_path,
                {
                    "MOCK_SDK_HANG": "1",
                    "MOCK_SDK_TEXT": "partial answer",
                    "MOCK_SDK_READY_FILE": str(ready),
                },
                dispose_eof_grace_ms=200,
            ),
        )
        await wait_for(ready.exists)
        run.cancel()
        result = await asyncio.wait_for(run.result(), 5)
        assert (result.output, result.stop_reason) == ("partial answer", "aborted")
        await run.dispose()

    asyncio.run(scenario())


def test_startup_cancelled_before_handshake_reaps_the_child(tmp_path: Path) -> None:
    async def scenario() -> None:
        cancel = asyncio.Event()
        cancel.set()
        try:
            await start_sdk_run("do the task", spec=spec(tmp_path), cancel_event=cancel)
        except SdkStartupCancelled:
            return
        raise AssertionError("a pre-aborted request must not start")

    asyncio.run(scenario())


def test_child_that_exits_before_handshake_rejects(tmp_path: Path) -> None:
    async def scenario() -> None:
        try:
            await start_sdk_run(
                "do the task",
                spec=spec(tmp_path, command=sys.executable, args=("-c", "raise SystemExit(2)")),
            )
        except SdkClientError:
            return
        raise AssertionError("a dead child must not publish a run")

    asyncio.run(scenario())


def test_spawn_failure_surfaces_without_a_process(tmp_path: Path) -> None:
    async def scenario() -> None:
        try:
            await start_sdk_run(
                "do the task", spec=spec(tmp_path, command=str(tmp_path / "missing"))
            )
        except SdkClientError as exc:
            assert "spawn" in str(exc)
            return
        raise AssertionError("a missing executable must fail the start")

    asyncio.run(scenario())


def test_crashed_child_flattens_into_an_error_result(tmp_path: Path) -> None:
    async def scenario() -> None:
        errors: list[tuple[BaseException, str]] = []
        run = await start_sdk_run(
            "do the task",
            spec=spec(tmp_path, {"MOCK_SDK_CRASH_ON_PROMPT": "1"}),
            on_error=lambda error, reason: errors.append((error, reason)),
        )
        try:
            assert (await run.result()).stop_reason == "error"
            assert errors and errors[0][1] == "error"
        finally:
            await run.dispose()

    asyncio.run(scenario())


def test_dispose_uses_the_protocol_shutdown_before_any_signal(tmp_path: Path) -> None:
    async def scenario() -> None:
        marker = tmp_path / "shutdown"
        run = await start_sdk_run(
            "do the task", spec=spec(tmp_path, {"MOCK_SDK_SHUTDOWN_FILE": str(marker)})
        )
        await run.result()
        await run.dispose()
        assert marker.read_text(encoding="utf-8") == "shutdown"
        assert run.returncode == 0

    asyncio.run(scenario())


def test_dispose_escalates_for_an_eof_deaf_child(tmp_path: Path) -> None:
    async def scenario() -> None:
        marker = tmp_path / "sigterm"
        ready = tmp_path / "armed"
        run = await start_sdk_run(
            "do the task",
            spec=spec(
                tmp_path,
                {
                    "MOCK_SDK_IGNORE_EOF": "1",
                    "MOCK_SDK_SIGTERM_FILE": str(marker),
                    "MOCK_SDK_READY_FILE": str(ready),
                    "MOCK_SDK_HANG": "1",
                },
                shutdown_timeout_ms=200,
                dispose_eof_grace_ms=200,
            ),
        )
        # The fixture arms its signal handlers before the prompt arrives, so
        # wait for the turn itself to be in flight (the second marker value).
        await wait_for(lambda: ready.exists() and ready.read_text(encoding="utf-8") == "ready")
        await run.dispose()
        assert marker.read_text(encoding="utf-8") == "sigterm"

    asyncio.run(scenario())


def test_dispose_is_idempotent(tmp_path: Path) -> None:
    async def scenario() -> None:
        run = await start_sdk_run("do the task", spec=spec(tmp_path))
        await run.result()
        await asyncio.gather(run.dispose(), run.dispose())
        assert run.returncode is not None

    asyncio.run(scenario())
