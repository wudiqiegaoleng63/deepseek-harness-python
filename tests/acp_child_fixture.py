"""Scripted ACP agent used as a subprocess by the acp_client tests.

Speaks the agent side of ACP over stdio, driven entirely by environment
variables so no model or network is involved:

- ``MOCK_TEXT`` — assistant text streamed as one ``agent_message_chunk``.
- ``MOCK_ECHO_ENV`` — if set to a variable NAME, stream that variable's value
  (or ``<NAME unset>``) instead, so a test can assert what env reached the child.
- ``MOCK_ECHO_CWD`` — if ``1``, stream the process cwd and the announced session
  cwd instead of ``MOCK_TEXT``.
- ``MOCK_STOP`` — the ACP stop reason returned from ``session/prompt``.
- ``MOCK_HANG`` — if ``1``, ``session/prompt`` waits for ``session/cancel``.
- ``MOCK_IGNORE_CANCEL`` — with ``MOCK_HANG``, never resolve on cancel.
- ``MOCK_PERMISSION`` — if ``1``, ask the client to approve before answering.
- ``MOCK_NO_ALLOW`` — with ``MOCK_PERMISSION``, offer only reject-shaped options.
- ``MOCK_THOUGHT`` — if ``1``, emit a non-message update before the answer.
- ``MOCK_READY_FILE`` — touched once the prompt handler is in flight.
- ``MOCK_MISSING_SESSION_ID`` — if ``1``, return an empty ``session/new`` result.
- ``MOCK_CRASH_ON_PROMPT`` / ``MOCK_CRASH_ON_CANCEL`` — exit hard at that point.
- ``MOCK_FLUSH_ON_EOF`` — on stdin EOF, touch this path after
  ``MOCK_FLUSH_DELAY_MS`` and exit on its own (models a child whose durable
  flush needs an EOF window before any signal).
- ``MOCK_IGNORE_EOF`` — stay alive past EOF; exit on SIGTERM, touching
  ``MOCK_SIGTERM_FILE`` when set (proof the graceful-signal rung fired).
- ``MOCK_TRAP_SIGTERM`` — ignore SIGTERM entirely, forcing the SIGKILL rung.
"""

from __future__ import annotations

import itertools
import json
import os
import signal
import sys
import time
import uuid
from typing import Any

IDS = itertools.count(1)


def _write(frame: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(frame, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _touch(path: str | None, value: str = "ready") -> None:
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(value)


def _streamed_text() -> str:
    echo_env = os.environ.get("MOCK_ECHO_ENV")
    if echo_env:
        return os.environ.get(echo_env) or f"<{echo_env} unset>"
    return os.environ.get("MOCK_TEXT", "mock child answer")


def _option_set() -> list[dict[str, str]]:
    if os.environ.get("MOCK_NO_ALLOW") == "1":
        return [{"optionId": "no", "name": "Reject", "kind": "reject_once"}]
    return [
        {"optionId": "yes", "name": "Allow", "kind": "allow_once"},
        {"optionId": "no", "name": "Reject", "kind": "reject_once"},
    ]


def _stay_alive() -> None:
    """Hold the process open so disposal must escalate beyond stdin EOF."""
    while True:
        time.sleep(0.05)


def _on_eof() -> None:
    flush_path = os.environ.get("MOCK_FLUSH_ON_EOF")
    if flush_path:
        time.sleep(float(os.environ.get("MOCK_FLUSH_DELAY_MS", "150")) / 1000)
        _touch(flush_path, "flushed")
        os._exit(0)
    if os.environ.get("MOCK_IGNORE_EOF") == "1" or os.environ.get("MOCK_TRAP_SIGTERM") == "1":
        _stay_alive()


def _install_signal_handlers() -> None:
    if os.environ.get("MOCK_TRAP_SIGTERM") == "1":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        _touch(os.environ.get("MOCK_READY_FILE"), "trap-armed")
        return
    if os.environ.get("MOCK_IGNORE_EOF") == "1":
        sigterm_file = os.environ.get("MOCK_SIGTERM_FILE")

        def on_term(_signum: int, _frame: object) -> None:
            _touch(sigterm_file, "sigterm")
            os._exit(0)

        signal.signal(signal.SIGTERM, on_term)
        _touch(os.environ.get("MOCK_READY_FILE"), "ignore-eof-armed")


class Child:
    def __init__(self) -> None:
        self.session_cwd = ""
        self.session_id = ""
        self.pending_prompt_id: str | None = None
        self.pending_permission_id: str | None = None
        self.prompt_request_id: str | None = None
        self.prompt_params: dict[str, Any] = {}
        self.cancelled = False

    # ------------------------------------------------------------------ requests
    def handle(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        params = message.get("params") or {}
        if method == "initialize":
            self._result(
                message,
                {
                    "protocolVersion": 1,
                    "agentCapabilities": {
                        "promptCapabilities": {
                            "image": False,
                            "audio": False,
                            "embeddedContext": False,
                        }
                    },
                    "authMethods": [],
                },
            )
        elif method == "session/new":
            self.session_cwd = str(params.get("cwd", ""))
            if os.environ.get("MOCK_MISSING_SESSION_ID") == "1":
                self._result(message, {})
            else:
                self.session_id = os.environ.get("MOCK_SESSION_ID") or uuid.uuid4().hex
                self._result(message, {"sessionId": self.session_id})
        elif method == "session/prompt":
            self._prompt(message, params)
        elif method == "session/cancel":
            self._cancel()
        elif (
            self.pending_permission_id is not None
            and str(message.get("id")) == self.pending_permission_id
        ):
            self._permission_answer(message.get("result"))

    def _result(self, message: dict[str, Any], result: dict[str, Any]) -> None:
        _write({"jsonrpc": "2.0", "id": message.get("id"), "result": result})

    def _prompt(self, message: dict[str, Any], params: dict[str, Any]) -> None:
        if os.environ.get("MOCK_CRASH_ON_PROMPT") == "1":
            os._exit(1)
        self.prompt_request_id = str(message.get("id"))
        self.prompt_params = params
        if os.environ.get("MOCK_PERMISSION") == "1":
            self.pending_permission_id = f"m-{next(IDS)}"
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": self.pending_permission_id,
                    "method": "session/request_permission",
                    "params": {
                        "sessionId": params.get("sessionId"),
                        "toolCall": {"toolCallId": "mock-call", "title": "mock side effect"},
                        "options": _option_set(),
                    },
                }
            )
            return
        self._answer(params)

    def _permission_answer(self, result: Any) -> None:
        self.pending_permission_id = None
        outcome = (result or {}).get("outcome") if isinstance(result, dict) else None
        if isinstance(outcome, dict) and outcome.get("outcome") == "cancelled":
            self._result({"id": self.prompt_request_id}, {"stopReason": "cancelled"})
            self.prompt_request_id = None
            return
        self._answer(self.prompt_params)

    def _answer(self, params: dict[str, Any]) -> None:
        session_id = params.get("sessionId") or self.session_id
        if os.environ.get("MOCK_THOUGHT") == "1":
            _write(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "agent_thought_chunk",
                            "content": {"type": "text", "text": "thinking..."},
                        },
                    },
                }
            )
        if os.environ.get("MOCK_ECHO_CWD") == "1":
            streamed = f"{os.getcwd()}\n{self.session_cwd}"
        else:
            streamed = _streamed_text()
        _write(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": session_id,
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": streamed},
                    },
                },
            }
        )
        _touch(os.environ.get("MOCK_READY_FILE"))
        if os.environ.get("MOCK_HANG") == "1":
            self.pending_prompt_id = self.prompt_request_id
            if self.cancelled:
                self._settle_cancelled()
            return
        self._result(
            {"id": self.prompt_request_id},
            {"stopReason": os.environ.get("MOCK_STOP", "end_turn")},
        )
        self.prompt_request_id = None

    def _settle_cancelled(self) -> None:
        if self.pending_prompt_id is None:
            return
        self._result({"id": self.pending_prompt_id}, {"stopReason": "cancelled"})
        self.pending_prompt_id = None

    def _cancel(self) -> None:
        if os.environ.get("MOCK_CRASH_ON_CANCEL") == "1":
            os._exit(1)
        self.cancelled = True
        if os.environ.get("MOCK_IGNORE_CANCEL") != "1":
            self._settle_cancelled()


def main() -> None:
    _install_signal_handlers()
    child = Child()
    while True:
        line = sys.stdin.readline()
        if not line:
            _on_eof()
            return
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if isinstance(message, dict):
            child.handle(message)


if __name__ == "__main__":
    main()
