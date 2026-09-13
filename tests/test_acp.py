from __future__ import annotations

import asyncio
import json
import queue
from collections.abc import AsyncIterator
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import pytest

from deepseek_harness.acp import (
    HarnessAcpServer,
    permission_outcome,
    prompt_has_unsupported_content,
    prompt_outcome,
    render_prompt_blocks,
    turn_end_to_stop_reason,
)
from deepseek_harness.llm.adapter import LlmAdapter
from deepseek_harness.llm.types import LlmRequest, StreamChunk
from deepseek_harness.sdk_rpc import serve_stdio
from deepseek_harness.session import SessionEvent
from deepseek_harness.web import HarnessService


class TextAdapter:
    def __init__(self, text: str = "acp answer") -> None:
        self.text = text

    def stream(self, _request: LlmRequest) -> AsyncIterator[StreamChunk]:
        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(kind="text", text=self.text)
            yield StreamChunk(kind="done", finish_reason="stop")

        return chunks()

    async def aclose(self) -> None:
        return None


def make_server(tmp_path: Path, adapter: object | None = None) -> HarnessAcpServer:
    service = HarnessService(
        tmp_path / "sessions",
        cwd=tmp_path,
        adapter_factory=lambda _model: cast(LlmAdapter, adapter or TextAdapter()),
    )
    return HarnessAcpServer(service)


class PushedInput:
    def __init__(self) -> None:
        self.lines: queue.Queue[bytes] = queue.Queue()

    def push(self, frame: dict[str, object]) -> None:
        self.lines.put((json.dumps(frame) + "\n").encode())

    def close(self) -> None:
        self.lines.put(b"")

    def readline(self) -> bytes:
        return self.lines.get()


# ---------------------------------------------------------------- pure codecs


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ({"kind": "completed"}, "end_turn"),
        ({"kind": "max-tokens"}, "max_tokens"),
        ({"kind": "aborted"}, "end_turn"),
        ({"kind": "blocked"}, "end_turn"),
        ({"kind": "error", "error": {"code": "UNKNOWN", "message": "failed"}}, "end_turn"),
        ({"kind": "interrupted"}, "cancelled"),
        (None, "end_turn"),
    ],
)
def test_turn_end_to_stop_reason_matches_ts_bridge(
    reason: dict[str, object] | None, expected: str
) -> None:
    assert turn_end_to_stop_reason(reason) == expected


def test_render_prompt_blocks_drops_unsupported_and_keeps_baseline() -> None:
    assert render_prompt_blocks([{"type": "text", "text": "first second"}]) == "first second"
    assert (
        render_prompt_blocks(
            [
                {"type": "text", "text": "summarize"},
                {"type": "resource_link", "name": "notes.txt", "uri": "file:///tmp/notes.txt"},
            ]
        )
        == 'summarize\n[resource_link name="notes.txt" uri="file:///tmp/notes.txt"]\n'
    )
    # Unsupported inline payloads are dropped from the conversion itself.
    assert render_prompt_blocks([{"type": "image", "data": "", "mimeType": "image/png"}]) == ""
    assert render_prompt_blocks(None) == ""


def test_prompt_has_unsupported_content_rejects_beyond_baseline() -> None:
    assert not prompt_has_unsupported_content([{"type": "text", "text": "ok"}])
    assert not prompt_has_unsupported_content(
        [{"type": "resource_link", "name": "a", "uri": "file:///a"}]
    )
    assert prompt_has_unsupported_content([{"type": "image", "data": ""}])
    assert prompt_has_unsupported_content("not-a-list") is False


# ------------------------------------------------------------- server methods


def test_acp_initialize_advertises_text_only_automation(tmp_path: Path) -> None:
    async def scenario() -> None:
        server = make_server(tmp_path)
        response = await server.handle_request(
            "initialize",
            {"protocolVersion": 1, "clientCapabilities": {}},
        )
        assert response == {
            "protocolVersion": 1,
            "agentInfo": {
                "name": "deepseek-harness-acp",
                "version": "0.0.1",
            },
            "agentCapabilities": {
                "promptCapabilities": {
                    "audio": False,
                    "embeddedContext": False,
                    "image": False,
                },
            },
            "authMethods": [],
        }
        assert await server.handle_request("authenticate", {"methodId": "unused"}) == {}
        with pytest.raises(Exception, match="unknown ACP method"):
            await server.handle_request("unknown/method", {})
        await server.close()

    asyncio.run(scenario())


def test_acp_session_new_validates_automation_contract(tmp_path: Path) -> None:
    async def scenario() -> None:
        server = make_server(tmp_path)
        with pytest.raises(Exception, match="absolute path"):
            await server.handle_request("session/new", {"cwd": "relative"})
        with pytest.raises(Exception, match="additionalDirectories"):
            await server.handle_request(
                "session/new",
                {"cwd": str(tmp_path), "mcpServers": [], "additionalDirectories": ["/other"]},
            )
        with pytest.raises(Exception, match="mcpServers"):
            await server.handle_request(
                "session/new",
                {
                    "cwd": str(tmp_path),
                    "mcpServers": [{"name": "fs", "command": "node"}],
                },
            )
        created = await server.handle_request(
            "session/new", {"cwd": str(tmp_path), "mcpServers": []}
        )
        session_id = created["sessionId"]
        handle = await server.service.get_session(session_id)
        assert str(handle.session.header.cwd) == str(tmp_path)
        await server.close()

    asyncio.run(scenario())


def test_acp_prompt_settles_with_committed_chunks(tmp_path: Path) -> None:
    async def scenario() -> None:
        updates: list[dict[str, object]] = []

        async def notify(method: str, params: dict[str, object]) -> None:
            updates.append({"method": method, **params})  # type: ignore[dict-item]

        server = make_server(tmp_path)
        server.set_notification_sink(notify)
        created = await server.handle_request("session/new", {"cwd": str(tmp_path)})
        session_id = str(created["sessionId"])
        result = await server.handle_request(
            "session/prompt", {"sessionId": session_id, "prompt": [{"type": "text", "text": "go"}]}
        )
        assert result == {"stopReason": "end_turn"}
        assert updates == [
            {
                "method": "session/update",
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "acp answer"},
                },
            }
        ]
        await server.close()

    asyncio.run(scenario())


def test_acp_prompt_validation_runs_before_a_turn_starts(tmp_path: Path) -> None:
    async def scenario() -> None:
        server = make_server(tmp_path)
        created = await server.handle_request("session/new", {"cwd": str(tmp_path)})
        session_id = str(created["sessionId"])
        with pytest.raises(Exception, match="empty prompt"):
            await server.handle_request(
                "session/prompt",
                {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "  "}],
                },
            )
        with pytest.raises(Exception, match="only text and resource_link"):
            await server.handle_request(
                "session/prompt",
                {"sessionId": session_id, "prompt": [{"type": "image", "data": ""}]},
            )
        handle = await server.service.get_session(session_id)
        assert not any(event.type == "turn/start" for event in handle.session.events)
        with pytest.raises(Exception, match="unknown session"):
            await server.handle_request(
                "session/prompt",
                {"sessionId": "missing", "prompt": [{"type": "text", "text": "go"}]},
            )
        await server.close()

    asyncio.run(scenario())


def test_acp_resource_links_reach_the_model_as_text(tmp_path: Path) -> None:
    async def scenario() -> None:
        seen: list[LlmRequest] = []

        class RecordingAdapter(TextAdapter):
            def stream(self, request: LlmRequest) -> AsyncIterator[StreamChunk]:
                seen.append(request)
                return super().stream(request)

        server = make_server(tmp_path, RecordingAdapter())
        created = await server.handle_request("session/new", {"cwd": str(tmp_path)})
        session_id = str(created["sessionId"])
        await server.handle_request(
            "session/prompt",
            {
                "sessionId": session_id,
                "prompt": [
                    {"type": "text", "text": "summarize"},
                    {"type": "resource_link", "name": "notes.txt", "uri": "file:///tmp/notes.txt"},
                ],
            },
        )
        content = [block.to_dict() for block in seen[-1].messages[-1].content]
        assert content == [
            {
                "type": "text",
                "text": 'summarize\n[resource_link name="notes.txt" uri="file:///tmp/notes.txt"]\n',
            }
        ]
        await server.close()

    asyncio.run(scenario())


# ---------------------------------------------------------------- cancellation


class HangingAdapter:
    def stream(self, _request: LlmRequest) -> AsyncIterator[StreamChunk]:
        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(kind="text", text="starting")
            await asyncio.Event().wait()  # never set; the cancel must break it

            yield StreamChunk(kind="done", finish_reason="stop")

        return chunks()

    async def aclose(self) -> None:
        return None


def test_acp_discarded_prompt_admission_settles_cancelled(tmp_path: Path) -> None:
    async def scenario() -> None:
        server = make_server(tmp_path)
        created = await server.handle_request("session/new", {"cwd": str(tmp_path)})
        session_id = str(created["sessionId"])

        async def discarded(session_id: str, content: object, **kwargs: object) -> object:
            # The harness accepts the message but no turn ever claims it, which
            # leaves a turnless slot: admission discarded, no turn/start.
            return {"accepted": True, "messageId": "message-never-recorded"}

        server.service.prompt = discarded  # type: ignore[method-assign]
        result = await server.handle_request(
            "session/prompt",
            {"sessionId": session_id, "prompt": [{"type": "text", "text": "go"}]},
        )
        assert result == {"stopReason": "cancelled"}
        await server.close()

    asyncio.run(scenario())


def test_prompt_outcome_correlates_admission_and_turn() -> None:
    def event(seq: int, type_: str, data: dict[str, Any]) -> SessionEvent:
        return SessionEvent(seq=seq, time=seq, type=type_, data=data)

    events = [
        event(0, "turn/start", {"turn": 0}),
        event(1, "user/message", {"message": {"id": "message-a"}}),
        event(2, "turn/end", {"turn": 0, "reason": {"kind": "completed"}}),
        event(3, "turn/start", {"turn": 1}),
        event(4, "user/message", {"message": {"id": "message-b"}}),
        event(5, "turn/end", {"turn": 1, "reason": {"kind": "error", "error": {}}}),
    ]
    # Each prompt correlates with the turn its own message was admitted into,
    # not with a neighbouring turn.
    assert prompt_outcome(events, 0, "message-a") == (True, {"kind": "completed"})
    assert prompt_outcome(events, 0, "message-b") == (True, {"kind": "error", "error": {}})
    # A message that never reached the session is a discarded admission.
    assert prompt_outcome(events, 0, "message-missing") == (False, None)
    # Only events at or after the prompt's own sequence are considered.
    assert prompt_outcome(events, 3, "message-a") == (False, None)


def test_acp_cancel_interrupts_a_running_prompt(tmp_path: Path) -> None:
    async def scenario() -> None:
        server = make_server(tmp_path, HangingAdapter())
        created = await server.handle_request("session/new", {"cwd": str(tmp_path)})
        session_id = str(created["sessionId"])
        pending = asyncio.ensure_future(
            server.handle_request(
                "session/prompt",
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "go"}]},
            )
        )
        for _ in range(500):
            handle = await server.service.get_session(session_id)
            if handle.agent.status == "running":
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("the prompt never started running")
        cancelled = await server.handle_request("session/cancel", {"sessionId": session_id})
        assert cancelled == {}
        assert await pending == {"stopReason": "cancelled"}
        # An unknown-id cancellation stays a no-op.
        assert await server.handle_request("session/cancel", {"sessionId": "missing"}) == {}
        await server.close()

    asyncio.run(scenario())


def test_acp_cancel_interrupts_autonomous_work_without_prompt(tmp_path: Path) -> None:
    async def scenario() -> None:
        server = make_server(tmp_path, HangingAdapter())
        created = await server.handle_request("session/new", {"cwd": str(tmp_path)})
        session_id = str(created["sessionId"])
        handle = await server.service.get_session(session_id)
        await server.service.prompt(
            session_id,
            [{"type": "text", "text": "autonomous work"}],
        )
        for _ in range(500):
            if handle.agent.status == "running":
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("autonomous work never started")
        assert await server.handle_request("session/cancel", {"sessionId": session_id}) == {}
        await handle.agent.when_idle()
        assert handle.agent.status == "idle"
        assert session_id not in server._pending
        await server.close()

    asyncio.run(scenario())


def test_acp_close_cancels_an_inflight_prompt(tmp_path: Path) -> None:
    async def scenario() -> None:
        server = make_server(tmp_path, HangingAdapter())
        created = await server.handle_request("session/new", {"cwd": str(tmp_path)})
        session_id = str(created["sessionId"])
        pending = asyncio.create_task(
            server.handle_request(
                "session/prompt",
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "go"}]},
            )
        )
        await asyncio.sleep(0)
        assert session_id in server._pending
        await server.close()
        assert await pending == {"stopReason": "cancelled"}

    asyncio.run(scenario())


def test_acp_idle_cancel_does_not_poison_the_next_prompt(tmp_path: Path) -> None:
    async def scenario() -> None:
        server = make_server(tmp_path)
        created = await server.handle_request("session/new", {"cwd": str(tmp_path)})
        session_id = str(created["sessionId"])
        await server.handle_request("session/cancel", {"sessionId": session_id})
        result = await server.handle_request(
            "session/prompt", {"sessionId": session_id, "prompt": [{"type": "text", "text": "go"}]}
        )
        assert result == {"stopReason": "end_turn"}
        await server.close()

    asyncio.run(scenario())


# ------------------------------------------------------------------ permission


def test_permission_outcome_only_grants_the_advertised_choice() -> None:
    assert (
        permission_outcome({"outcome": {"outcome": "selected", "optionId": "allow-once"}})
        == "allowed-once"
    )
    assert (
        permission_outcome({"outcome": {"outcome": "selected", "optionId": "reject-once"}})
        == "rejected"
    )
    assert permission_outcome(
        {"outcome": {"outcome": "selected", "optionId": "unknown-grant"}}
    ) == ("rejected")
    assert permission_outcome({"outcome": {"outcome": "cancelled"}}) == "cancelled"
    # Malformed or absent answers fail closed.
    assert permission_outcome(None) == "rejected"
    assert permission_outcome({"outcome": "selected"}) == "rejected"


def test_acp_answers_bridge_owned_approval_requests(tmp_path: Path) -> None:
    async def scenario() -> None:
        requests: list[dict[str, Any]] = []
        answers: list[Any] = [{"outcome": {"outcome": "selected", "optionId": "allow-once"}}]

        async def request(method: str, params: dict[str, Any]) -> Any:
            requests.append({"method": method, **params})
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer

        server = make_server(tmp_path)
        server.set_request_sink(request)
        created = await server.handle_request("session/new", {"cwd": str(tmp_path)})
        session_id = str(created["sessionId"])

        approval = asyncio.ensure_future(
            server.service.request_approval(
                session_id,
                "write_file",
                approval_id="approval-1",
                call_id="call-9",
            )
        )
        for _ in range(500):
            if requests:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("the bridge never asked the client for permission")

        assert requests == [
            {
                "method": "session/request_permission",
                "sessionId": session_id,
                "toolCall": {"toolCallId": "call-9"},
                "options": [
                    {"optionId": "allow-once", "name": "Allow once", "kind": "allow_once"},
                    {"optionId": "reject-once", "name": "Reject", "kind": "reject_once"},
                ],
            }
        ]
        assert await approval == "allowed-once"

        # A rejection and a cancellation map straight through.
        answers.append({"outcome": {"outcome": "selected", "optionId": "reject-once"}})
        rejected = asyncio.ensure_future(
            server.service.request_approval(
                session_id, "write_file", approval_id="approval-2", call_id="call-10"
            )
        )
        assert await rejected == "rejected"

        answers.append({"outcome": {"outcome": "cancelled"}})
        cancelled = asyncio.ensure_future(
            server.service.request_approval(
                session_id, "write_file", approval_id="approval-3", call_id="call-11"
            )
        )
        assert await cancelled == "cancelled"

        # A client-side failure never grants access.
        answers.append(ValueError("client gone"))
        unavailable = asyncio.ensure_future(
            server.service.request_approval(
                session_id, "write_file", approval_id="approval-4", call_id="call-12"
            )
        )
        assert await unavailable == "unavailable"
        await server.close()

    asyncio.run(scenario())


def test_acp_leaves_foreign_and_untargeted_approvals_to_a_ui(tmp_path: Path) -> None:
    async def scenario() -> None:
        requests: list[str] = []

        async def request(method: str, params: dict[str, Any]) -> Any:
            requests.append(method)
            return {"outcome": {"outcome": "selected", "optionId": "allow-once"}}

        server = make_server(tmp_path)
        server.set_request_sink(request)
        created = await server.handle_request("session/new", {"cwd": str(tmp_path)})
        session_id = str(created["sessionId"])
        foreign = await server.service.create_session(session_id="foreign", cwd=str(tmp_path))

        # An approval for a session this connection does not own stays with the
        # UI answerer, as does one that carries no tool call id.
        unowned = asyncio.ensure_future(
            server.service.request_approval(
                "foreign", "write_file", approval_id="approval-1", call_id="call-9"
            )
        )
        untargeted = asyncio.ensure_future(
            server.service.request_approval(session_id, "write_file", approval_id="approval-2")
        )
        await asyncio.sleep(0.05)
        assert requests == []
        assert (
            await server.service.resolve_approval("foreign", "approval-1", "allowed-once") is True
        )
        assert (
            await server.service.resolve_approval(session_id, "approval-2", "allowed-once") is True
        )
        assert await unowned == "allowed-once"
        assert await untargeted == "allowed-once"
        assert await server.service.resolve_approval(session_id, "missing", "rejected") is False
        del foreign
        await server.close()

    asyncio.run(scenario())


def test_acp_disposed_bridge_rejects_new_sessions_and_prompts(tmp_path: Path) -> None:
    async def scenario() -> None:
        server = make_server(tmp_path)
        created = await server.handle_request("session/new", {"cwd": str(tmp_path)})
        session_id = str(created["sessionId"])
        await server.close()
        with pytest.raises(Exception, match="disposed"):
            await server.handle_request("session/new", {"cwd": str(tmp_path)})
        with pytest.raises(Exception, match="disposed"):
            await server.handle_request(
                "session/prompt",
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "go"}]},
            )

    asyncio.run(scenario())


def test_acp_keeps_sessions_isolated(tmp_path: Path) -> None:
    async def scenario() -> None:
        updates: list[dict[str, Any]] = []

        async def notify(method: str, params: dict[str, Any]) -> None:
            updates.append({"method": method, **params})

        server = make_server(tmp_path, HangingAdapter())
        server.set_notification_sink(notify)
        first = str(
            (await server.handle_request("session/new", {"cwd": str(tmp_path)}))["sessionId"]
        )
        second = str(
            (await server.handle_request("session/new", {"cwd": str(tmp_path)}))["sessionId"]
        )
        assert first != second

        # Each session owns an independent prompt slot and cancellation path.
        running = asyncio.ensure_future(
            server.handle_request(
                "session/prompt", {"sessionId": first, "prompt": [{"type": "text", "text": "A"}]}
            )
        )
        for _ in range(500):
            if (await server.service.get_session(first)).agent.status == "running":
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("session A never started running")

        with pytest.raises(Exception, match="already in flight"):
            await server.handle_request(
                "session/prompt", {"sessionId": first, "prompt": [{"type": "text", "text": "A2"}]}
            )
        assert await server.handle_request("session/cancel", {"sessionId": first}) == {}
        assert await running == {"stopReason": "cancelled"}
        # Cancelling A must not cancel B or settle A's prompt as B's outcome.
        assert (await server.service.get_session(second)).agent.status == "idle"
        assert not any(update.get("sessionId") == second for update in updates)
        await server.close()

    asyncio.run(scenario())


# -------------------------------------------------------------- stdio wiring


def test_acp_stdio_serves_initialize_and_shutdown_round_trip(tmp_path: Path) -> None:
    async def scenario() -> None:
        incoming = BytesIO(
            (
                json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
                + "\n"
                + json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "session/new",
                        "params": {"cwd": str(tmp_path), "mcpServers": []},
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "unknown",
                        "params": {},
                    }
                )
                + "\n"
            ).encode()
        )
        outgoing = BytesIO()
        service = HarnessService(
            tmp_path / "sessions",
            cwd=tmp_path,
            adapter_factory=lambda _model: cast(LlmAdapter, TextAdapter()),
        )
        await serve_stdio(HarnessAcpServer(service), input_stream=incoming, output_stream=outgoing)
        frames = [json.loads(line) for line in outgoing.getvalue().splitlines()]
        by_id = {frame["id"]: frame for frame in frames}
        assert by_id[1]["result"]["protocolVersion"] == 1
        assert isinstance(by_id[2]["result"]["sessionId"], str)
        assert by_id[3]["error"]["code"] == -32601
        assert all(frame["id"] != 4 for frame in frames)

    asyncio.run(scenario())


def test_acp_stdio_reads_cancel_while_prompt_is_in_flight(tmp_path: Path) -> None:
    async def scenario() -> None:
        incoming = PushedInput()
        outgoing = BytesIO()
        server = make_server(tmp_path, HangingAdapter())
        serving = asyncio.create_task(
            serve_stdio(server, input_stream=incoming, output_stream=outgoing)
        )

        async def response(request_id: int) -> dict[str, object]:
            for _ in range(500):
                for line in outgoing.getvalue().splitlines():
                    frame = json.loads(line)
                    if frame.get("id") == request_id:
                        return frame
                await asyncio.sleep(0.01)
            raise AssertionError(f"request {request_id} did not receive a response")

        incoming.push({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        initialize = await response(1)
        assert initialize["result"]["protocolVersion"] == 1  # type: ignore[index]

        incoming.push(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/new",
                "params": {"cwd": str(tmp_path)},
            }
        )
        created = await response(2)
        session_id = str(created["result"]["sessionId"])  # type: ignore[index]
        incoming.push(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "go"}],
                },
            }
        )
        for _ in range(500):
            handle = await server.service.get_session(session_id)
            if handle.agent.status == "running":
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("the prompt never started running")

        # This is a JSON-RPC notification, not a request, and must be read
        # while the prompt request is still waiting for the agent to quiesce.
        incoming.push(
            {
                "jsonrpc": "2.0",
                "method": "session/cancel",
                "params": {"sessionId": session_id},
            }
        )
        prompt_response = await response(3)
        assert prompt_response["result"] == {"stopReason": "cancelled"}
        incoming.close()
        await serving


def test_acp_stdio_shapes_session_update_notification(tmp_path: Path) -> None:
    async def scenario() -> None:
        incoming = PushedInput()
        outgoing = BytesIO()
        server = make_server(tmp_path)
        serving = asyncio.create_task(
            serve_stdio(server, input_stream=incoming, output_stream=outgoing)
        )

        async def response(request_id: int) -> dict[str, object]:
            for _ in range(500):
                for line in outgoing.getvalue().splitlines():
                    frame = json.loads(line)
                    if frame.get("id") == request_id:
                        return frame
                await asyncio.sleep(0.01)
            raise AssertionError(f"request {request_id} did not receive a response")

        incoming.push({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        await response(1)
        incoming.push(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/new",
                "params": {"cwd": str(tmp_path)},
            }
        )
        created = await response(2)
        session_id = str(created["result"]["sessionId"])  # type: ignore[index]
        incoming.push(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "go"}],
                },
            }
        )
        assert (await response(3))["result"] == {"stopReason": "end_turn"}

        update: dict[str, object] | None = None
        for _ in range(500):
            for line in outgoing.getvalue().splitlines():
                frame = json.loads(line)
                if frame.get("method") == "session/update":
                    update = frame
                    break
            if update is not None:
                break
            await asyncio.sleep(0.01)
        assert update == {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "acp answer"},
                },
            },
        }
        incoming.close()
        await serving

    asyncio.run(scenario())


def test_acp_stdio_routes_permission_request_and_response(tmp_path: Path) -> None:
    async def scenario() -> None:
        incoming = PushedInput()
        outgoing = BytesIO()
        server = make_server(tmp_path)
        serving = asyncio.create_task(
            serve_stdio(server, input_stream=incoming, output_stream=outgoing)
        )

        def frames() -> list[dict[str, Any]]:
            return [json.loads(line) for line in outgoing.getvalue().splitlines()]

        async def response(request_id: object) -> dict[str, Any]:
            for _ in range(500):
                for frame in frames():
                    if frame.get("id") == request_id:
                        return frame
                await asyncio.sleep(0.01)
            raise AssertionError(f"request {request_id} did not receive a response")

        incoming.push({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        await response(1)
        incoming.push(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/new",
                "params": {"cwd": str(tmp_path)},
            }
        )
        created = await response(2)
        session_id = str(created["result"]["sessionId"])

        approval = asyncio.ensure_future(
            server.service.request_approval(
                session_id, "write_file", approval_id="approval-1", call_id="call-9"
            )
        )
        permission: dict[str, Any] | None = None
        for _ in range(500):
            permission = next(
                (f for f in frames() if f.get("method") == "session/request_permission"), None
            )
            if permission is not None:
                break
            await asyncio.sleep(0.01)
        assert permission is not None
        assert str(permission["id"]).startswith("srv-")
        assert permission["params"]["toolCall"] == {"toolCallId": "call-9"}

        # The client's response is a frame without a method and is routed back
        # to the bridge's outstanding request.
        incoming.push(
            {
                "jsonrpc": "2.0",
                "id": permission["id"],
                "result": {"outcome": {"outcome": "selected", "optionId": "allow-once"}},
            }
        )
        assert await approval == "allowed-once"
        incoming.close()
        await serving

    asyncio.run(scenario())


def test_acp_stdio_settles_pending_permission_on_disconnect(tmp_path: Path) -> None:
    async def scenario() -> None:
        incoming = PushedInput()
        outgoing = BytesIO()
        server = make_server(tmp_path)
        serving = asyncio.create_task(
            serve_stdio(server, input_stream=incoming, output_stream=outgoing)
        )

        incoming.push({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        for _ in range(500):
            if any(json.loads(line).get("id") == 1 for line in outgoing.getvalue().splitlines()):
                break
            await asyncio.sleep(0.01)
        incoming.push(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/new",
                "params": {"cwd": str(tmp_path)},
            }
        )
        for _ in range(500):
            if any(json.loads(line).get("id") == 2 for line in outgoing.getvalue().splitlines()):
                break
            await asyncio.sleep(0.01)
        created = next(
            json.loads(line)
            for line in outgoing.getvalue().splitlines()
            if json.loads(line).get("id") == 2
        )
        session_id = str(created["result"]["sessionId"])

        approval = asyncio.ensure_future(
            server.service.request_approval(
                session_id, "write_file", approval_id="approval-1", call_id="call-9"
            )
        )
        for _ in range(500):
            if any(
                json.loads(line).get("method") == "session/request_permission"
                for line in outgoing.getvalue().splitlines()
            ):
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("the bridge never asked the client for permission")

        # The client disconnects without answering. Teardown settles the
        # approval, and no grant is ever inferred from a missing answer.
        incoming.close()
        await serving
        assert await approval == "cancelled"

    asyncio.run(scenario())
