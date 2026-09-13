"""Scripted Codex app-server used as a subprocess by the codex_client tests.

Speaks the app-server JSON-RPC wire and is driven entirely by environment
variables:

- ``MOCK_CODEX_ANSWER`` — the ``final_answer`` agent message's text.
- ``MOCK_CODEX_UNPHASED`` — an unphased agent message, the compatibility fallback.
- ``MOCK_CODEX_COMMENTARY`` — a commentary message, which is never an answer.
- ``MOCK_CODEX_STATUS`` — the terminal turn status (default ``completed``).
- ``MOCK_CODEX_ERROR_INFO`` — ``error.codexErrorInfo`` on the terminal turn.
- ``MOCK_CODEX_NO_TERMINAL`` — if ``1``, never emit ``turn/completed``.
- ``MOCK_CODEX_EARLY`` — if ``1``, emit the item before the ``turn/start`` response.
- ``MOCK_CODEX_HANG`` — accept the turn and never finish.
- ``MOCK_CODEX_REQUEST`` — issue a server request first: ``command``,
  ``file``, ``permissions``, ``userInput``, ``elicitation``, or ``unknown``.
- ``MOCK_CODEX_DECISIONS`` — ``;``-joined ``availableDecisions`` for an
  approval request.
- ``MOCK_CODEX_REQUESTS_FILE`` — records every server request the client sent
  back, so a test can assert the unattended answers.
- ``MOCK_CODEX_THREAD_ID`` — the thread id to publish (default ``thread-1``).
- ``MOCK_CODEX_READY_FILE`` — touched once a turn is in flight.
- ``MOCK_CODEX_EXIT_CODE`` — exit with this code instead of answering.
- ``MOCK_CODEX_IGNORE_EOF`` / ``MOCK_CODEX_TRAP_SIGTERM`` /
  ``MOCK_CODEX_SIGTERM_FILE`` — teardown-ladder behaviours.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from typing import Any


def _send(frame: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(frame, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _record(path: str | None, value: str) -> None:
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(value)


THREAD_ID = os.environ.get("MOCK_CODEX_THREAD_ID", "thread-1")
TURN_ID = "turn-1"


def _notify(method: str, params: dict[str, Any]) -> None:
    _send({"method": method, "params": params})


def _answer_item(text: str | None, phase: str | None) -> None:
    if text is None:
        return
    _notify(
        "item/completed",
        {
            "threadId": THREAD_ID,
            "turnId": TURN_ID,
            "item": {"type": "agentMessage", "text": text, "phase": phase},
        },
    )


def _terminal() -> None:
    turn: dict[str, Any] = {
        "id": TURN_ID,
        "status": os.environ.get("MOCK_CODEX_STATUS", "completed"),
    }
    error_info = os.environ.get("MOCK_CODEX_ERROR_INFO")
    if error_info:
        turn["error"] = {"codexErrorInfo": error_info}
    _notify("turn/completed", {"threadId": THREAD_ID, "turn": turn})


def _emit_answers() -> None:
    _answer_item(os.environ.get("MOCK_CODEX_COMMENTARY"), "commentary")
    _answer_item(os.environ.get("MOCK_CODEX_UNPHASED"), None)
    answer = os.environ.get("MOCK_CODEX_ANSWER")
    if answer is None and os.environ.get("MOCK_CODEX_UNPHASED") is None:
        # A plain run needs some answer; an unphased-only run must not get one.
        answer = "codex child answer"
    _answer_item(answer, "final_answer")


def _handle_request(message: dict[str, Any]) -> bool:
    """Handle one client request. Returns False when the process should exit."""
    method = message.get("method")
    request_id = message.get("id")
    if method == "initialize":
        _send(
            {
                "id": request_id,
                "result": {
                    "userAgent": "mock-codex/0",
                    "codexHome": "/tmp",
                    "platformFamily": "unix",
                    "platformOs": "linux",
                },
            }
        )
        return True
    if method == "initialized":
        return True
    if method == "thread/start":
        ephemeral = bool((message.get("params") or {}).get("ephemeral")) and (
            os.environ.get("MOCK_CODEX_NOT_EPHEMERAL") != "1"
        )
        _send(
            {
                "id": request_id,
                "result": {"thread": {"id": THREAD_ID, "ephemeral": ephemeral}},
            }
        )
        return True
    if method == "turn/start":
        _handle_turn_start(message)
        return True
    if method == "turn/interrupt":
        _send({"id": request_id, "result": {}})
        return True
    return True


def _handle_turn_start(message: dict[str, Any]) -> None:
    request_id = message.get("id")
    exit_code = os.environ.get("MOCK_CODEX_EXIT_CODE")
    if exit_code:
        raise SystemExit(int(exit_code))
    request_kind = os.environ.get("MOCK_CODEX_REQUEST")
    if request_kind:
        # The turn is accepted first; the request is raised while it runs.
        _send({"id": request_id, "result": {"turn": {"id": TURN_ID}}})
        _server_request(request_kind)
        return  # the turn continues once the answer arrives
    if os.environ.get("MOCK_CODEX_EARLY") == "1":
        _emit_answers()
        _notify("turn/started", {"threadId": THREAD_ID, "turn": {"id": TURN_ID}})
    _send({"id": request_id, "result": {"turn": {"id": TURN_ID}}})
    if os.environ.get("MOCK_CODEX_EARLY") != "1":
        _notify("turn/started", {"threadId": THREAD_ID, "turn": {"id": TURN_ID}})
        _emit_answers()
    _record(os.environ.get("MOCK_CODEX_READY_FILE"), "ready")
    if os.environ.get("MOCK_CODEX_HANG") == "1":
        while True:
            time.sleep(0.05)
    if os.environ.get("MOCK_CODEX_NO_TERMINAL") == "1":
        return
    _terminal()


def _server_request(kind: str) -> None:
    params: dict[str, Any] = {"threadId": THREAD_ID, "turnId": TURN_ID}
    decisions = os.environ.get("MOCK_CODEX_DECISIONS")
    if decisions:
        params["availableDecisions"] = decisions.split(";")
    method = {
        "command": "item/commandExecution/requestApproval",
        "file": "item/fileChange/requestApproval",
        "permissions": "item/permissions/requestApproval",
        "userInput": "item/tool/requestUserInput",
        "elicitation": "mcpServer/elicitation/request",
        "unknown": "item/unknown/request",
    }[kind]
    _send({"id": "server-1", "method": method, "params": params})


def _install_signal_handlers() -> None:
    if os.environ.get("MOCK_CODEX_TRAP_SIGTERM") == "1":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        _record(os.environ.get("MOCK_CODEX_READY_FILE"), "trap-armed")
        return
    if os.environ.get("MOCK_CODEX_IGNORE_EOF") == "1":
        sigterm_file = os.environ.get("MOCK_CODEX_SIGTERM_FILE")

        def on_term(_signum: int, _frame: object) -> None:
            _record(sigterm_file, "sigterm")
            os._exit(0)

        signal.signal(signal.SIGTERM, on_term)
        _record(os.environ.get("MOCK_CODEX_READY_FILE"), "ignore-eof-armed")


def main() -> None:
    _install_signal_handlers()
    requests: list[dict[str, Any]] = []
    while True:
        line = sys.stdin.readline()
        if not line:
            if os.environ.get("MOCK_CODEX_IGNORE_EOF") == "1" or (
                os.environ.get("MOCK_CODEX_TRAP_SIGTERM") == "1"
            ):
                while True:
                    time.sleep(0.05)
            return
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if not isinstance(message, dict):
            continue
        if "method" not in message:
            # A response to a server request: record it and finish the turn.
            requests.append(message)
            _record(
                os.environ.get("MOCK_CODEX_REQUESTS_FILE"),
                "\n".join(json.dumps(item, sort_keys=True) for item in requests),
            )
            if os.environ.get("MOCK_CODEX_REQUEST") in {"command", "file", "permissions"}:
                _emit_answers()
                _terminal()
            continue
        if not _handle_request(message):
            return
        if message.get("method") == "turn/start" and os.environ.get("MOCK_CODEX_REQUEST"):
            continue


if __name__ == "__main__":
    main()
