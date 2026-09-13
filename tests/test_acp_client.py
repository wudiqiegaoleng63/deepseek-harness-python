from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from deepseek_harness.acp_client import (
    DEFAULT_DISPOSE_EOF_GRACE_MS,
    AcpClientError,
    AcpRunSpec,
    AcpStartupCancelled,
    acp_content_text,
    acp_stop_reason,
    dispose_acp_child,
    scrubbed_child_env,
    start_acp_run,
    to_acp_prompt,
)

FIXTURE = Path(__file__).parent / "acp_child_fixture.py"


def spec(tmp_path: Path, **overrides: object) -> AcpRunSpec:
    values: dict[str, object] = {
        "command": sys.executable,
        "args": (str(FIXTURE),),
        "cwd": str(tmp_path),
        "dispose_eof_grace_ms": 2_000,
    }
    values.update(overrides)
    return AcpRunSpec(**values)  # type: ignore[arg-type]


def child_spec(
    tmp_path: Path,
    extra_env: dict[str, str] | None = None,
    **overrides: object,
) -> AcpRunSpec:
    """A fixture child whose env carries this process's MOCK_* knobs."""

    env = {name: value for name, value in os.environ.items() if name.startswith("MOCK_")}
    env.update(extra_env or {})
    return spec(tmp_path, env=env, **overrides)


async def wait_for(predicate, timeout: float = 10.0) -> None:
    for _ in range(int(timeout / 0.02)):
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition never became true")


# ------------------------------------------------------------------- pure codec


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("end_turn", "completed"),
        ("max_tokens", "max-tokens"),
        ("refusal", "refusal"),
        ("cancelled", "aborted"),
        ("max_turn_requests", "error"),
        ("some_future_reason", "error"),
        (None, "error"),
    ],
)
def test_acp_stop_reason_maps_the_wire_vocabulary(reason: object, expected: str) -> None:
    assert acp_stop_reason(reason) == expected


def test_acp_content_text_and_prompt_translation() -> None:
    assert acp_content_text({"type": "text", "text": "hi"}) == "hi"
    assert acp_content_text({"type": "image", "data": ""}) == ""
    assert acp_content_text(None) == ""
    assert to_acp_prompt(
        [
            {"type": "text", "text": "keep"},
            {"type": "image", "data": "..."},
            {"type": "resource_link", "name": "a", "uri": "file:///a"},
            {"type": "text", "text": " tail"},
        ]
    ) == [{"type": "text", "text": "keep"}, {"type": "text", "text": " tail"}]


def test_scrubbed_child_env_drops_credentials_and_harness_names() -> None:
    parent = {
        "PATH": "/usr/bin",
        "HOME": "/home/x",
        "DEEPSEEK_API_KEY": "secret",
        "TF_TOKEN": "x",
        "db_password": "x",
        "DSH_SESSION_ROOT": "/sessions",
        "dsh_home": "/dsh",
        "HTTPS_PROXY": "http://proxy",
    }
    scrubbed = scrubbed_child_env(parent)
    assert scrubbed == {"PATH": "/usr/bin", "HOME": "/home/x", "HTTPS_PROXY": "http://proxy"}


def test_explicit_env_survives_the_scrub(tmp_path: Path) -> None:
    async def scenario() -> None:
        run = await start_acp_run(
            [{"type": "text", "text": "go"}],
            spec=child_spec(
                tmp_path,
                extra_env={
                    "MOCK_TEXT": "unused",
                    "MOCK_ECHO_ENV": "DEEPSEEK_API_KEY",
                    "DEEPSEEK_API_KEY": "child-own-key",
                    "TF_TOKEN": "explicit-token",
                },
            ),
        )
        try:
            assert (await run.result()).output == "child-own-key"
        finally:
            await run.dispose()

    asyncio.run(scenario())


# ------------------------------------------------------------------- one run


def test_run_collects_streamed_output_and_stop_reason(tmp_path: Path) -> None:
    async def scenario() -> None:
        run = await start_acp_run(
            [{"type": "text", "text": "go"}],
            spec=child_spec(tmp_path, extra_env={"MOCK_TEXT": "child says hi"}),
        )
        try:
            assert run.id
            result = await run.result()
            assert (result.output, result.stop_reason) == ("child says hi", "completed")
        finally:
            await run.dispose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("mock_stop", "expected"),
    [("max_tokens", "max-tokens"), ("refusal", "refusal"), ("max_turn_requests", "error")],
)
def test_run_maps_child_stop_reasons(tmp_path: Path, mock_stop: str, expected: str) -> None:
    async def scenario() -> None:
        run = await start_acp_run(
            [{"type": "text", "text": "go"}],
            spec=child_spec(tmp_path, extra_env={"MOCK_STOP": mock_stop}),
        )
        try:
            assert (await run.result()).stop_reason == expected
        finally:
            await run.dispose()

    asyncio.run(scenario())


def test_run_consumes_non_message_updates(tmp_path: Path) -> None:
    async def scenario() -> None:
        run = await start_acp_run(
            [{"type": "text", "text": "go"}],
            spec=child_spec(tmp_path, extra_env={"MOCK_TEXT": "answer", "MOCK_THOUGHT": "1"}),
        )
        try:
            # The thought is consumed but never accumulated into the answer.
            assert (await run.result()).output == "answer"
        finally:
            await run.dispose()

    asyncio.run(scenario())


def test_run_uses_the_parent_session_cwd_for_process_and_session(tmp_path: Path) -> None:
    async def scenario() -> None:
        run = await start_acp_run(
            [{"type": "text", "text": "go"}],
            spec=child_spec(tmp_path, extra_env={"MOCK_ECHO_CWD": "1", "MOCK_TEXT": "unused"}),
        )
        try:
            output = (await run.result()).output
        finally:
            await run.dispose()
        process_cwd, session_cwd = output.splitlines()
        # The child's own cwd and the workspace it was told about agree.
        assert os.path.realpath(process_cwd) == os.path.realpath(str(tmp_path))
        assert session_cwd == str(tmp_path)

    asyncio.run(scenario())


# ------------------------------------------------------------------ permission


def test_permission_default_rejects_and_child_settles_aborted(tmp_path: Path) -> None:
    async def scenario() -> None:
        run = await start_acp_run(
            [{"type": "text", "text": "go"}],
            spec=child_spec(tmp_path, extra_env={"MOCK_PERMISSION": "1"}),
        )
        try:
            # A reject-shaped answer makes the child answer `cancelled`.
            assert (await run.result()).stop_reason == "aborted"
        finally:
            await run.dispose()

    asyncio.run(scenario())


def test_permission_allow_selects_the_first_allow_option(tmp_path: Path) -> None:
    async def scenario() -> None:
        run = await start_acp_run(
            [{"type": "text", "text": "go"}],
            spec=child_spec(
                tmp_path, permission="allow", extra_env={"MOCK_PERMISSION": "1", "MOCK_TEXT": "ok"}
            ),
        )
        try:
            result = await run.result()
            assert (result.output, result.stop_reason) == ("ok", "completed")
        finally:
            await run.dispose()

    asyncio.run(scenario())


def test_permission_allow_falls_back_to_cancelled_without_an_allow_option(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        run = await start_acp_run(
            [{"type": "text", "text": "go"}],
            spec=child_spec(
                tmp_path,
                permission="allow",
                extra_env={"MOCK_PERMISSION": "1", "MOCK_NO_ALLOW": "1"},
            ),
        )
        try:
            assert (await run.result()).stop_reason == "aborted"
        finally:
            await run.dispose()

    asyncio.run(scenario())


# ------------------------------------------------------------------ lifecycle


def test_cancel_settles_aborted_without_child_cooperation(tmp_path: Path) -> None:
    async def scenario() -> None:
        ready = tmp_path / "ready"
        run = await start_acp_run(
            [{"type": "text", "text": "go"}],
            spec=child_spec(
                tmp_path,
                extra_env={
                    "MOCK_HANG": "1",
                    "MOCK_IGNORE_CANCEL": "1",
                    "MOCK_READY_FILE": str(ready),
                    "MOCK_TEXT": "partial answer",
                },
            ),
        )
        await wait_for(ready.exists)
        run.cancel()
        result = await asyncio.wait_for(run.result(), 5)
        # The child never resolves, but the run settles and keeps its text.
        assert (result.output, result.stop_reason) == ("partial answer", "aborted")
        await run.dispose()

    asyncio.run(scenario())


def test_startup_cancelled_before_session_reaps_the_child(tmp_path: Path) -> None:
    async def scenario() -> None:
        cancel = asyncio.Event()
        cancel.set()
        with pytest.raises(AcpStartupCancelled):
            await start_acp_run(
                [{"type": "text", "text": "go"}],
                spec=child_spec(tmp_path),
                cancel_event=cancel,
            )

    asyncio.run(scenario())


def test_missing_session_id_rejects_and_reaps(tmp_path: Path) -> None:
    async def scenario() -> None:
        with pytest.raises(AcpClientError, match="session id"):
            await start_acp_run(
                [{"type": "text", "text": "go"}],
                spec=child_spec(tmp_path, extra_env={"MOCK_MISSING_SESSION_ID": "1"}),
            )

    asyncio.run(scenario())


def test_child_that_exits_before_its_session_rejects(tmp_path: Path) -> None:
    async def scenario() -> None:
        with pytest.raises(AcpClientError):
            await start_acp_run(
                [{"type": "text", "text": "go"}],
                spec=spec(tmp_path, command=sys.executable, args=("-c", "raise SystemExit(3)")),
            )

    asyncio.run(scenario())


def test_spawn_failure_surfaces_without_a_process(tmp_path: Path) -> None:
    async def scenario() -> None:
        with pytest.raises(AcpClientError, match="spawn"):
            await start_acp_run(
                [{"type": "text", "text": "go"}],
                spec=spec(tmp_path, command=str(tmp_path / "missing-binary")),
            )

    asyncio.run(scenario())


def test_crashed_child_flattens_into_an_error_result(tmp_path: Path) -> None:
    async def scenario() -> None:
        errors: list[tuple[BaseException, str]] = []
        run = await start_acp_run(
            [{"type": "text", "text": "go"}],
            spec=child_spec(tmp_path, extra_env={"MOCK_CRASH_ON_PROMPT": "1"}),
            on_error=lambda error, reason: errors.append((error, reason)),
        )
        # The transport failure is flattened into the result, never a rejection.
        assert (await run.result()).stop_reason == "error"
        assert errors and errors[0][1] == "error"
        await run.dispose()

    asyncio.run(scenario())


# -------------------------------------------------------------------- dispose


def test_dispose_tier_one_cooperative_eof_exit(tmp_path: Path) -> None:
    async def scenario() -> None:
        flushed = tmp_path / "flushed"
        ready = tmp_path / "ready"
        run = await start_acp_run(
            [{"type": "text", "text": "go"}],
            spec=child_spec(
                tmp_path,
                extra_env={
                    "MOCK_FLUSH_ON_EOF": str(flushed),
                    "MOCK_FLUSH_DELAY_MS": "150",
                    "MOCK_HANG": "1",
                    "MOCK_IGNORE_CANCEL": "1",
                    "MOCK_READY_FILE": str(ready),
                },
            ),
        )
        await wait_for(ready.exists)
        await run.dispose()
        # The EOF window outlasted the child's own flush; no signal was needed,
        # and the cancel that preceded teardown settled the pending turn.
        assert flushed.read_text(encoding="utf-8") == "flushed"
        assert (await run.result()).stop_reason == "aborted"
        assert run.returncode == 0

    asyncio.run(scenario())


def test_dispose_tier_two_sigterm_for_an_eof_deaf_child(tmp_path: Path) -> None:
    async def scenario() -> None:
        sigterm = tmp_path / "sigterm"
        run = await start_acp_run(
            [{"type": "text", "text": "go"}],
            spec=child_spec(
                tmp_path,
                extra_env={
                    "MOCK_IGNORE_EOF": "1",
                    "MOCK_SIGTERM_FILE": str(sigterm),
                    "MOCK_READY_FILE": str(tmp_path / "armed"),
                },
            ),
        )
        await run.result()
        await run.dispose()
        assert sigterm.read_text(encoding="utf-8") == "sigterm"

    asyncio.run(scenario())


def test_dispose_tier_three_sigkill_for_a_term_trapping_child(tmp_path: Path) -> None:
    async def scenario() -> None:
        ready = tmp_path / "trap-armed"
        run = await start_acp_run(
            [{"type": "text", "text": "go"}],
            spec=child_spec(
                tmp_path,
                extra_env={"MOCK_TRAP_SIGTERM": "1", "MOCK_READY_FILE": str(ready)},
                dispose_eof_grace_ms=200,
            ),
        )
        await wait_for(ready.exists)
        await run.result()
        await run.dispose()
        assert run.returncode == -9  # SIGKILL escalation proof

    asyncio.run(scenario())


def test_dispose_acp_child_ignores_a_missing_or_exited_process(tmp_path: Path) -> None:
    async def scenario() -> None:
        await dispose_acp_child(None)
        process = await asyncio.create_subprocess_exec(sys.executable, "-c", "pass")
        await process.wait()
        await dispose_acp_child(process)

    asyncio.run(scenario())


def test_dispose_is_idempotent(tmp_path: Path) -> None:
    async def scenario() -> None:
        run = await start_acp_run([{"type": "text", "text": "go"}], spec=child_spec(tmp_path))
        await run.result()
        await asyncio.gather(run.dispose(), run.dispose())
        assert DEFAULT_DISPOSE_EOF_GRACE_MS > 0

    asyncio.run(scenario())
