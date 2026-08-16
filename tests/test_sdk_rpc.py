from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from collections.abc import AsyncIterator
from io import BytesIO
from pathlib import Path
from typing import cast

import pytest

from deepseek_harness.llm.adapter import LlmAdapter
from deepseek_harness.llm.types import LlmRequest, StreamChunk
from deepseek_harness.sdk_process import (
    DeepSeekHarnessProcess,
    JsonRpcResponseError,
    RequestTimeoutError,
)
from deepseek_harness.sdk_rpc import HarnessSdkJsonRpcServer, serve_stdio
from deepseek_harness.web import HarnessService


class RpcAdapter:
    def stream(self, _request: LlmRequest) -> AsyncIterator[StreamChunk]:
        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(kind="text", text="rpc answer")
            yield StreamChunk(kind="done", finish_reason="stop")

        return chunks()

    async def aclose(self) -> None:
        return None


def test_sdk_rpc_server_forwards_runtime_notifications(tmp_path: Path) -> None:
    async def scenario() -> None:
        service = HarnessService(
            tmp_path / "sessions",
            cwd=tmp_path,
            adapter_factory=lambda _model: cast(LlmAdapter, RpcAdapter()),
        )
        notifications: list[dict[str, object]] = []

        async def notify(method: str, params: dict[str, object]) -> None:
            notifications.append({"method": method, "params": params})

        server = HarnessSdkJsonRpcServer(lambda _params: service, notify=notify)
        initialized = await server.handle_request(
            "initialize",
            {
                "cwd": str(tmp_path),
                "provider": "deepseek-official",
                "model": "test-model",
                "maxTokens": 123,
            },
        )
        assert initialized["serverInfo"]["name"] == "deepseek-harness-sdk-runtime"  # type: ignore[index]
        assert service.settings.get_value_sync("llm-deepseek")["maxTokens"] == 123

        result = await server.handle_request(
            "session/prompt",
            {
                "sessionId": "rpc-session",
                "contentBlocks": [{"type": "text", "text": "hello"}],
            },
        )
        assert isinstance(result["messageId"], str)
        handle = await service.get_session("rpc-session")
        assert handle.task is not None
        await handle.task
        for _ in range(100):
            if any(
                item["method"] == "session.status"
                and item["params"]["status"] == "idle"  # type: ignore[index]
                for item in notifications
            ):
                break
            await asyncio.sleep(0.001)

        methods = [str(item["method"]) for item in notifications]
        assert "session.event" in methods
        assert "session.status" in methods
        assert any(
            item["method"] == "session.event"
            and item["params"]["event"]["type"] == "assistant/message"  # type: ignore[index]
            for item in notifications
        )
        await server.handle_request("shutdown", {})

    asyncio.run(scenario())


def test_sdk_stdio_server_writes_json_rpc_responses(tmp_path: Path) -> None:
    async def scenario() -> None:
        service = HarnessService(tmp_path / "sessions", cwd=tmp_path)
        incoming = BytesIO(
            (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "cwd": str(tmp_path),
                            "provider": "deepseek-official",
                            "model": "test-model",
                        },
                    }
                )
                + "\n"
                + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": {}})
                + "\n"
            ).encode()
        )
        outgoing = BytesIO()
        server = HarnessSdkJsonRpcServer(lambda _params: service)
        await serve_stdio(server, input_stream=incoming, output_stream=outgoing)
        frames = [json.loads(line) for line in outgoing.getvalue().splitlines()]
        assert {frame["id"] for frame in frames} == {1, 2}
        assert all("error" not in frame for frame in frames)

    asyncio.run(scenario())


def test_sdk_process_collects_owned_activity_and_session_tree(tmp_path: Path) -> None:
    runtime = tmp_path / "fake_runtime.py"
    runtime.write_text(
        textwrap.dedent(
            """
            import json
            import sys

            sequence = 0

            def write(frame):
                sys.stdout.write(json.dumps(frame) + "\\n")
                sys.stdout.flush()

            def event(session_id, kind, data):
                global sequence
                write({"jsonrpc": "2.0", "method": "session.event", "params": {
                    "sessionId": session_id,
                    "event": {"type": kind, "seq": sequence, "time": 0, "data": data},
                }})
                sequence += 1

            for line in sys.stdin:
                request = json.loads(line)
                method = request.get("method")
                request_id = request.get("id")
                params = request.get("params") or {}
                if method == "initialize":
                    write({"jsonrpc": "2.0", "id": request_id, "result": {
                        "serverInfo": {"name": "deepseek-harness-sdk-runtime", "version": "fake"}
                    }})
                elif method == "session/prompt":
                    session_id = params["sessionId"]
                    message_id = "fake-user-1"
                    event(session_id, "agent/inbox/spliced", {"inserted": [{"id": message_id}]})
                    write({"jsonrpc": "2.0", "method": "session.status", "params": {
                        "sessionId": session_id, "status": "running"
                    }})
                    child_id = session_id + "-child"
                    write({"jsonrpc": "2.0", "method": "subagent.started", "params": {
                        "parentSessionId": session_id, "childSessionId": child_id
                    }})
                    event(child_id, "assistant/message", {"message": {
                        "content": [{"type": "text", "text": "child"}]
                    }})
                    write({"jsonrpc": "2.0", "method": "subagent.finished", "params": {
                        "parentSessionId": session_id, "childSessionId": child_id,
                        "agentId": child_id, "provider": "fake", "status": "ok",
                        "stopReason": "completed"
                    }})
                    event(session_id, "assistant/message", {"message": {
                        "content": [{"type": "text", "text": "process answer"}]
                    }})
                    write({"jsonrpc": "2.0", "method": "session.status", "params": {
                        "sessionId": session_id, "status": "idle"
                    }})
                    write({"jsonrpc": "2.0", "id": request_id, "result": {"messageId": message_id}})
                elif method == "shutdown":
                    write({"jsonrpc": "2.0", "id": request_id, "result": {}})
                    break
            """
        ),
        encoding="utf-8",
    )
    seen: list[dict[str, object]] = []
    with DeepSeekHarnessProcess(
        command=(sys.executable, str(runtime)),
        cwd=tmp_path,
        request_timeout_seconds=5,
    ) as harness:
        result = harness.run("hello", session_id="process-root", on_notification=seen.append)

    assert result.session_id == "process-root"
    assert result.final_response == "process answer"
    assert any(event["type"] == "assistant/message" for event in result.events)
    assert any(item["method"] == "subagent.finished" for item in result.notifications)
    assert len(seen) == len(result.notifications)


def test_sdk_process_preserves_rpc_errors_and_protocol_validation(tmp_path: Path) -> None:
    runtime = tmp_path / "error_runtime.py"
    runtime.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    if request['method'] == 'initialize':\n"
        "        result = {'serverInfo': {'name': 'x', 'version': 'y'}}\n"
        "        frame = {'jsonrpc': '2.0', 'id': request['id'], 'result': result}\n"
        "        print(json.dumps(frame), flush=True)\n"
        "    elif request['method'] == 'session/prompt':\n"
        "        error = {'code': 9, 'message': 'no prompt'}\n"
        "        frame = {'jsonrpc': '2.0', 'id': request['id'], 'error': error}\n"
        "        print(json.dumps(frame), flush=True)\n"
        "    elif request['method'] == 'shutdown':\n"
        "        frame = {'jsonrpc': '2.0', 'id': request['id'], 'result': {}}\n"
        "        print(json.dumps(frame), flush=True)\n"
        "        break\n",
        encoding="utf-8",
    )
    harness = DeepSeekHarnessProcess(
        command=(sys.executable, str(runtime)), cwd=tmp_path, request_timeout_seconds=1
    )
    harness.start()
    with pytest.raises(JsonRpcResponseError, match="no prompt") as caught:
        harness.client.prompt("root", [{"type": "text", "text": "hello"}])
    assert caught.value.code == 9
    harness.close()

    with pytest.raises(ValueError, match="positive"):
        DeepSeekHarnessProcess(cwd=tmp_path, max_tokens=0)


def test_sdk_process_request_timeout(tmp_path: Path) -> None:
    runtime = tmp_path / "hang_runtime.py"
    runtime.write_text(
        "import json, sys, time\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    if request['method'] == 'initialize':\n"
        "        result = {'serverInfo': {'name': 'x', 'version': 'y'}}\n"
        "        frame = {'jsonrpc': '2.0', 'id': request['id'], 'result': result}\n"
        "        print(json.dumps(frame), flush=True)\n"
        "    elif request['method'] == 'session/prompt':\n"
        "        while True: time.sleep(1)\n",
        encoding="utf-8",
    )
    harness = DeepSeekHarnessProcess(
        command=(sys.executable, str(runtime)), cwd=tmp_path, request_timeout_seconds=0.05
    )
    harness.start()
    with pytest.raises(RequestTimeoutError):
        harness.client.prompt("root", [{"type": "text", "text": "hello"}])
    harness.close()


def test_sdk_rpc_rejects_unknown_method(tmp_path: Path) -> None:
    async def scenario() -> None:
        service = HarnessService(tmp_path / "sessions", cwd=tmp_path)
        server = HarnessSdkJsonRpcServer(lambda _params: service)
        with pytest.raises(Exception, match="unknown"):
            await server.handle_request("unknown", {})
        await server.close()

    asyncio.run(scenario())
