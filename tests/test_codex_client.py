from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from deepseek_harness.codex_client import (
    CodexError,
    CodexRunSpec,
    CodexStartupCancelled,
    context_window_exceeded,
    select_answer,
    start_codex_run,
    unattended_decision,
)

FIXTURE = Path(__file__).parent / "codex_child_fixture.py"


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
    return CodexRunSpec(**values)  # type: ignore[arg-type]


async def wait_for(predicate, timeout: float = 10.0) -> None:
    for _ in range(int(timeout / 0.02)):
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition never became true")


# ------------------------------------------------------------------- pure rules


def test_unattended_decision_never_grants_access() -> None:
    assert unattended_decision({"availableDecisions": ["accept", "cancel"]}) == "cancel"
    assert unattended_decision({"availableDecisions": ["decline"]}) == "decline"
    # An app-server that offers no decision list at all is declined.
    assert unattended_decision({}) == "decline"
    assert unattended_decision({"availableDecisions": "accept"}) == "decline"


def test_context_window_exceeded_reads_the_error_info() -> None:
    assert context_window_exceeded({"error": {"codexErrorInfo": "contextWindowExceeded"}})
    assert not context_window_exceeded({"error": {"codexErrorInfo": "other"}})
    assert not context_window_exceeded({})


def test_select_answer_prefers_the_final_phase() -> None:
    assert select_answer("final", "unphased") == "final"
    assert select_answer(None, "unphased") == "unphased"
    # A blank selected answer is no answer: the fallback is positional (used
    # only when no final-phase message exists), not a blank-message rescue.
    assert select_answer("  ", "unphased") == ""
    assert select_answer(None, None) == ""


# ---------------------------------------------------------------------- a run


def test_run_returns_the_final_answer(tmp_path: Path) -> None:
    async def scenario() -> None:
        run = await start_codex_run(
            "do the task", spec=spec(tmp_path, {"MOCK_CODEX_ANSWER": "codex says hi"})
        )
        try:
            result = await run.result()
            assert (result.output, result.stop_reason) == ("codex says hi", "completed")
        finally:
            await run.dispose()

    asyncio.run(scenario())


def test_run_ignores_commentary_and_prefers_the_final_answer(tmp_path: Path) -> None:
    async def scenario() -> None:
        run = await start_codex_run(
            "do the task",
            spec=spec(
                tmp_path,
                {
                    "MOCK_CODEX_COMMENTARY": "thinking out loud",
                    "MOCK_CODEX_UNPHASED": "unphased answer",
                    "MOCK_CODEX_ANSWER": "the final answer",
                },
            ),
        )
        try:
            assert (await run.result()).output == "the final answer"
        finally:
            await run.dispose()

        unphased = await start_codex_run(
            "do the task",
            spec=spec(
                tmp_path,
                {
                    "MOCK_CODEX_COMMENTARY": "thinking out loud",
                    "MOCK_CODEX_UNPHASED": "unphased answer",
                },
            ),
        )
        try:
            # Without a final phase the unphased message is the fallback, and
            # commentary never competes with it.
            assert (await unphased.result()).output == "unphased answer"
        finally:
            await unphased.dispose()

    asyncio.run(scenario())


def test_run_accepts_answers_that_arrive_before_the_turn_response(tmp_path: Path) -> None:
    async def scenario() -> None:
        run = await start_codex_run(
            "do the task",
            spec=spec(tmp_path, {"MOCK_CODEX_EARLY": "1", "MOCK_CODEX_ANSWER": "early answer"}),
        )
        try:
            result = await run.result()
            assert (result.output, result.stop_reason) == ("early answer", "completed")
        finally:
            await run.dispose()

    asyncio.run(scenario())


def test_run_maps_terminal_statuses_and_context_overflow(tmp_path: Path) -> None:
    async def scenario() -> None:
        for env, expected in (
            ({"MOCK_CODEX_STATUS": "failed"}, "error"),
            ({"MOCK_CODEX_STATUS": "interrupted"}, "error"),
            (
                {
                    "MOCK_CODEX_STATUS": "failed",
                    "MOCK_CODEX_ERROR_INFO": "contextWindowExceeded",
                },
                "max-tokens",
            ),
        ):
            errors: list[str] = []

            def record(_error: object, reason: str, sink: list[str] = errors) -> None:
                sink.append(reason)

            run = await start_codex_run("do the task", spec=spec(tmp_path, env), on_error=record)
            try:
                result = await run.result()
                assert result.stop_reason == expected
                if expected == "error":
                    assert errors == ["error"]
            finally:
                await run.dispose()

    asyncio.run(scenario())


def test_run_fails_when_a_completed_turn_has_no_answer(tmp_path: Path) -> None:
    async def scenario() -> None:
        run = await start_codex_run(
            "do the task",
            spec=spec(tmp_path, {"MOCK_CODEX_ANSWER": "   ", "MOCK_CODEX_COMMENTARY": "only talk"}),
        )
        try:
            assert (await run.result()).stop_reason == "error"
        finally:
            await run.dispose()

    asyncio.run(scenario())


def test_run_fails_a_non_ephemeral_thread(tmp_path: Path) -> None:
    async def scenario() -> None:
        try:
            await start_codex_run(
                "do the task", spec=spec(tmp_path, {"MOCK_CODEX_NOT_EPHEMERAL": "1"})
            )
        except CodexError as exc:
            assert "ephemeral" in str(exc)
            return
        raise AssertionError("a persisted thread must not be accepted")

    asyncio.run(scenario())


# ------------------------------------------------------------- server requests


def test_run_answers_approvals_without_granting(tmp_path: Path) -> None:
    async def scenario() -> None:
        requests_file = tmp_path / "requests"
        run = await start_codex_run(
            "do the task",
            spec=spec(
                tmp_path,
                {
                    "MOCK_CODEX_REQUEST": "command",
                    "MOCK_CODEX_DECISIONS": "accept;cancel",
                    "MOCK_CODEX_REQUESTS_FILE": str(requests_file),
                    "MOCK_CODEX_ANSWER": "done after asking",
                },
            ),
        )
        try:
            result = await run.result()
            assert (result.output, result.stop_reason) == ("done after asking", "completed")
        finally:
            await run.dispose()
        recorded = [json.loads(line) for line in requests_file.read_text().splitlines()]
        assert recorded == [{"jsonrpc": "2.0", "id": "server-1", "result": {"decision": "cancel"}}]

    asyncio.run(scenario())


def test_run_answers_permissions_user_input_and_elicitation(tmp_path: Path) -> None:
    async def scenario() -> None:
        # A permission request is answered and the turn still finishes.
        requests_file = tmp_path / "requests-permissions"
        run = await start_codex_run(
            "do the task",
            spec=spec(
                tmp_path,
                {
                    "MOCK_CODEX_REQUEST": "permissions",
                    "MOCK_CODEX_REQUESTS_FILE": str(requests_file),
                    "MOCK_CODEX_ANSWER": "answered without permissions",
                },
            ),
        )
        try:
            result = await run.result()
            assert (result.output, result.stop_reason) == (
                "answered without permissions",
                "completed",
            )
        finally:
            await run.dispose()
        assert [json.loads(line) for line in requests_file.read_text().splitlines()] == [
            {"jsonrpc": "2.0", "id": "server-1", "result": {"permissions": {}, "scope": "turn"}}
        ]

        # User input and elicitation are declined with no answers; the fixture
        # leaves the turn running, so the parent settles it by disposing.
        for kind, expected in (
            ("userInput", {"answers": {}}),
            ("elicitation", {"action": "decline", "content": None, "_meta": None}),
        ):
            kind_file = tmp_path / f"requests-{kind}"
            pending = await start_codex_run(
                "do the task",
                spec=spec(
                    tmp_path,
                    {"MOCK_CODEX_REQUEST": kind, "MOCK_CODEX_REQUESTS_FILE": str(kind_file)},
                    dispose_eof_grace_ms=200,
                ),
            )
            try:
                for _ in range(100):
                    if kind_file.exists():
                        break
                    await asyncio.sleep(0.02)
                assert not pending.settled
            finally:
                await pending.dispose()
            assert [json.loads(line) for line in kind_file.read_text().splitlines()] == [
                {"jsonrpc": "2.0", "id": "server-1", "result": expected}
            ]

    asyncio.run(scenario())


def test_run_fails_an_unknown_server_request(tmp_path: Path) -> None:
    async def scenario() -> None:
        run = await start_codex_run(
            "do the task", spec=spec(tmp_path, {"MOCK_CODEX_REQUEST": "unknown"})
        )
        try:
            assert (await run.result()).stop_reason == "error"
        finally:
            await run.dispose()

    asyncio.run(scenario())


# ------------------------------------------------------------------ lifecycle


def test_cancel_settles_aborted(tmp_path: Path) -> None:
    async def scenario() -> None:
        ready = tmp_path / "ready"
        run = await start_codex_run(
            "do the task",
            spec=spec(
                tmp_path,
                {"MOCK_CODEX_HANG": "1", "MOCK_CODEX_READY_FILE": str(ready)},
                dispose_eof_grace_ms=200,
            ),
        )
        await wait_for(ready.exists)
        run.cancel()
        result = await asyncio.wait_for(run.result(), 5)
        # Cancellation is local and authoritative, and the answer the child had
        # already emitted is preserved rather than discarded.
        assert result.stop_reason == "aborted"
        assert result.output == "codex child answer"
        await run.dispose()
        assert run.returncode is not None

    asyncio.run(scenario())


def test_startup_cancelled_before_spawn_is_rejected(tmp_path: Path) -> None:
    async def scenario() -> None:
        cancel = asyncio.Event()
        cancel.set()
        try:
            await start_codex_run("do the task", spec=spec(tmp_path), cancel_event=cancel)
        except CodexStartupCancelled:
            return
        raise AssertionError("a pre-aborted request must not start")

    asyncio.run(scenario())


def test_child_that_exits_before_its_thread_rejects(tmp_path: Path) -> None:
    async def scenario() -> None:
        try:
            await start_codex_run(
                "do the task", spec=spec(tmp_path, command=sys.executable, args=("-c", "pass"))
            )
        except CodexError:
            return
        raise AssertionError("a dead app-server must not publish a run")

    asyncio.run(scenario())


def test_dispose_is_idempotent_and_reaps_the_child(tmp_path: Path) -> None:
    async def scenario() -> None:
        run = await start_codex_run("do the task", spec=spec(tmp_path))
        await run.result()
        await asyncio.gather(run.dispose(), run.dispose())
        assert run.returncode is not None

    asyncio.run(scenario())
