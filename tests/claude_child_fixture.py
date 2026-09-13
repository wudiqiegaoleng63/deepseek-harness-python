"""Scripted Claude Code CLI used as a subprocess by the claude_code tests.

Emits the CLI's ``--output-format stream-json`` wire (one JSON object per line)
and is driven entirely by environment variables:

- ``MOCK_CLAUDE_RESULT`` — the final ``result`` message's text.
- ``MOCK_CLAUDE_SUBTYPE`` — its subtype (default ``success``).
- ``MOCK_CLAUDE_IS_ERROR`` — set to ``1`` to mark the success result as errored.
- ``MOCK_CLAUDE_NO_RESULT`` — set to ``1`` to end the stream without a result.
- ``MOCK_CLAUDE_ERRORS`` — ``;``-joined error details for a failed subtype.
- ``MOCK_CLAUDE_STREAM_TEXT`` — assistant text streamed before the result.
- ``MOCK_CLAUDE_HANG`` — accept the task and never finish.
- ``MOCK_CLAUDE_CRASH`` — exit non-zero before answering.
- ``MOCK_CLAUDE_READY_FILE`` — touched once the task has been read.
- ``MOCK_CLAUDE_ARGS_FILE`` — the file the received argv is written to, so a
  test can assert the flags the provider passed.
- ``MOCK_CLAUDE_STDIN_FILE`` — the file the received task text is written to.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any


def _write(frame: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(frame, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _record(path: str | None, value: str) -> None:
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(value)


def main() -> None:
    _record(os.environ.get("MOCK_CLAUDE_ARGS_FILE"), "\n".join(sys.argv[1:]))
    task = sys.stdin.read()
    _record(os.environ.get("MOCK_CLAUDE_STDIN_FILE"), task)
    _record(os.environ.get("MOCK_CLAUDE_READY_FILE"), "ready")

    if os.environ.get("MOCK_CLAUDE_CRASH") == "1":
        sys.stderr.write("mock claude crashed\n")
        raise SystemExit(9)

    _write({"type": "system", "subtype": "init", "session_id": "mock-session"})
    streamed = os.environ.get("MOCK_CLAUDE_STREAM_TEXT")
    if streamed:
        _write(
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": streamed}]},
            }
        )

    if os.environ.get("MOCK_CLAUDE_HANG") == "1":
        while True:
            time.sleep(0.05)

    if os.environ.get("MOCK_CLAUDE_NO_RESULT") == "1":
        return

    subtype = os.environ.get("MOCK_CLAUDE_SUBTYPE", "success")
    result: dict[str, Any] = {
        "type": "result",
        "subtype": subtype,
        "is_error": os.environ.get("MOCK_CLAUDE_IS_ERROR") == "1",
    }
    if subtype == "success":
        result["result"] = os.environ.get("MOCK_CLAUDE_RESULT", "claude child answer")
    else:
        errors = os.environ.get("MOCK_CLAUDE_ERRORS")
        result["errors"] = errors.split(";") if errors else []
        result["result"] = os.environ.get("MOCK_CLAUDE_RESULT", "")
    _write(result)


if __name__ == "__main__":
    main()
