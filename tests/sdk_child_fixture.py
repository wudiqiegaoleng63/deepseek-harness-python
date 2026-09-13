"""Scripted harness SDK runtime used as a subprocess by the sdk_client tests.

Speaks the SDK stdio JSON-RPC protocol (``initialize``, ``session/prompt``,
``shutdown``, plus ``session.event`` notifications) and is driven entirely by
environment variables, so no model or network is involved:

- ``MOCK_SDK_TEXT`` — assistant text streamed as one ``assistant/chunk`` text
  delta and committed as one ``assistant/message``.
- ``MOCK_SDK_TURN_KIND`` — the ``turn/end`` reason kind (default ``completed``).
- ``MOCK_SDK_NO_TURN_END`` — if ``1``, commit the message without a turn ending.
- ``MOCK_SDK_EMPTY_MESSAGE`` — if ``1``, commit an empty assistant message
  (usage only) before streaming, so only the streamed text remains as output.
- ``MOCK_SDK_HANG`` — if ``1``, accept the prompt but never finish the turn.
- ``MOCK_SDK_CRASH_ON_PROMPT`` — if ``1``, exit hard when prompted.
- ``MOCK_SDK_SHUTDOWN_FILE`` — touched when the protocol ``shutdown`` arrives
  (proof dispose used the graceful protocol rung before any signal).
- ``MOCK_SDK_IGNORE_EOF`` — stay alive past stdin EOF; exit on SIGTERM,
  touching ``MOCK_SDK_SIGTERM_FILE`` when set.
- ``MOCK_SDK_TRAP_SIGTERM`` — ignore SIGTERM entirely, forcing SIGKILL.
- ``MOCK_SDK_READY_FILE`` — touched once a prompt is in flight.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from typing import Any


def _write(frame: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(frame, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _touch(path: str | None, value: str = "ready") -> None:
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(value)


def _event(session_id: str, type_: str, data: dict[str, Any]) -> None:
    _write(
        {
            "jsonrpc": "2.0",
            "method": "session.event",
            "params": {
                "sessionId": session_id,
                "event": {"seq": _event.seq, "time": 0, "type": type_, "data": data},
            },
        }
    )
    _event.seq += 1


_event.seq = 0


def _handle_prompt(message: dict[str, Any], params: dict[str, Any]) -> None:
    if os.environ.get("MOCK_SDK_CRASH_ON_PROMPT") == "1":
        os._exit(1)
    session_id = str(params.get("sessionId", "session"))
    text = os.environ.get("MOCK_SDK_TEXT", "sdk child answer")
    _event(session_id, "turn/start", {"turn": 0})
    _event(
        session_id,
        "user/message",
        {"message": {"id": "u-1", "role": "user", "content": [{"type": "text", "text": "go"}]}},
    )
    _event(
        session_id,
        "assistant/chunk",
        {"turn": 0, "step": 1, "chunk": {"kind": "text", "text": text}},
    )
    if os.environ.get("MOCK_SDK_EMPTY_MESSAGE") == "1":
        _event(
            session_id,
            "assistant/message",
            {"turn": 0, "step": 1, "message": {"role": "assistant", "content": []}},
        )
    _event(
        session_id,
        "assistant/message",
        {
            "turn": 0,
            "step": 1,
            "message": {
                "role": "assistant",
                "content": []
                if os.environ.get("MOCK_SDK_EMPTY_MESSAGE") == "1"
                else [{"type": "text", "text": text}],
            },
        },
    )
    _touch(os.environ.get("MOCK_SDK_READY_FILE"))
    if os.environ.get("MOCK_SDK_HANG") == "1":
        # A busy runtime never reads further input, so neither a shutdown
        # request nor a cancellation can reach it.
        while True:
            time.sleep(0.05)
    if os.environ.get("MOCK_SDK_NO_TURN_END") != "1":
        _event(
            session_id,
            "turn/end",
            {"turn": 0, "reason": {"kind": os.environ.get("MOCK_SDK_TURN_KIND", "completed")}},
        )
    _write({"jsonrpc": "2.0", "id": message.get("id"), "result": {"messageId": "m-1"}})


def _install_signal_handlers() -> None:
    if os.environ.get("MOCK_SDK_TRAP_SIGTERM") == "1":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        _touch(os.environ.get("MOCK_SDK_READY_FILE"), "trap-armed")
        return
    if os.environ.get("MOCK_SDK_IGNORE_EOF") == "1":
        sigterm_file = os.environ.get("MOCK_SDK_SIGTERM_FILE")

        def on_term(_signum: int, _frame: object) -> None:
            _touch(sigterm_file, "sigterm")
            os._exit(0)

        signal.signal(signal.SIGTERM, on_term)
        _touch(os.environ.get("MOCK_SDK_READY_FILE"), "ignore-eof-armed")


def main() -> None:
    _install_signal_handlers()
    while True:
        line = sys.stdin.readline()
        if not line:
            if os.environ.get("MOCK_SDK_IGNORE_EOF") == "1" or (
                os.environ.get("MOCK_SDK_TRAP_SIGTERM") == "1"
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
        method = message.get("method")
        params = message.get("params") or {}
        if method == "initialize":
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": message.get("id"),
                    "result": {"serverInfo": {"name": "mock-sdk-runtime", "version": "0"}},
                }
            )
        elif method == "session/prompt":
            _handle_prompt(message, params)
        elif method == "shutdown":
            _touch(os.environ.get("MOCK_SDK_SHUTDOWN_FILE"), "shutdown")
            _write({"jsonrpc": "2.0", "id": message.get("id"), "result": {}})
            return


if __name__ == "__main__":
    main()
