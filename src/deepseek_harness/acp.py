"""Automation-only Agent Client Protocol server over JSON-RPC stdio.

Programmatic clients create fresh harness agents, send text prompts, collect
committed assistant text, and cancel work — the same contract as the TS
``dsh-acp`` bridge.  This is a transport adapter, not a UI layer: no session
resume, transcript replay, editor capabilities, or live token deltas.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .sdk_rpc import JsonRpcResponseError
from .session import SessionEvent
from .web.service import ApprovalOutcome, HarnessService

__all__ = [
    "AcpProtocolError",
    "HarnessAcpServer",
    "PROTOCOL_VERSION",
    "permission_outcome",
    "prompt_has_unsupported_content",
    "prompt_outcome",
    "render_prompt_blocks",
    "run_acp_server",
    "turn_end_to_stop_reason",
]

PROTOCOL_VERSION = 1


class AcpProtocolError(JsonRpcResponseError, ValueError):
    """A request violated the automation protocol contract."""

    def __init__(self, message: str, data: Any = None) -> None:
        super().__init__(-32602, message, data)


def render_prompt_blocks(blocks: Any) -> str:
    """Concatenate text blocks verbatim; baseline resource links flatten.

    Unsupported blocks (image, audio, embedded resources) are dropped from the
    text conversion; ``prompt_has_unsupported_content`` rejects them before a
    turn starts so nothing is silently lost.
    """

    if not isinstance(blocks, list):
        return ""
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif block_type == "resource_link":
            name = (
                json.dumps(block["name"], ensure_ascii=False)
                if isinstance(block.get("name"), str)
                else '""'
            )
            uri = (
                json.dumps(block["uri"], ensure_ascii=False)
                if isinstance(block.get("uri"), str)
                else '""'
            )
            parts.append(f"\n[resource_link name={name} uri={uri}]\n")
    return "".join(parts)


def prompt_has_unsupported_content(blocks: Any) -> bool:
    """Whether a prompt carries content beyond the ACP baseline."""

    if not isinstance(blocks, list):
        return False
    return any(
        not isinstance(block, dict) or block.get("type") not in {"text", "resource_link"}
        for block in blocks
    )


def turn_end_to_stop_reason(reason: Any) -> str:
    """Map a harness turn ending to ACP's terminal reason vocabulary."""

    kind = reason.get("kind") if isinstance(reason, dict) else None
    if kind == "max-tokens":
        return "max_tokens"
    if kind == "interrupted":
        return "cancelled"
    return "end_turn"


def permission_outcome(response: Any) -> ApprovalOutcome:
    """Map an ACP permission response to a harness approval outcome.

    Only the two advertised one-shot choices grant anything: a cancellation is
    propagated, any other answer (including an unknown option) is a rejection,
    and an absent or malformed response never grants access.
    """

    outcome = response.get("outcome") if isinstance(response, dict) else None
    if isinstance(outcome, dict):
        kind = outcome.get("outcome")
        if kind == "cancelled":
            return "cancelled"
        if kind == "selected" and outcome.get("optionId") == "allow-once":
            return "allowed-once"
    return "rejected"


def prompt_outcome(
    events: Sequence[SessionEvent], first_seq: int, message_id: str | None
) -> tuple[bool, Any]:
    """Correlate one queued prompt with admission and its turn ending.

    Returns whether the prompt was admitted (its user message reached the
    session) and the reason of the turn that claimed it.  A prompt whose
    admission was discarded — removed before a turn claimed it — is a turnless
    slot, which the automation contract settles as ``cancelled``.
    """

    relevant = [event for event in events if event.seq >= first_seq]
    admitted = message_id is None
    target_turn: int | None = None
    active_turn: int | None = None
    for event in relevant:
        if event.type == "turn/start":
            value = event.data.get("turn")
            active_turn = value if isinstance(value, int) else None
        elif event.type == "user/message" and message_id is not None:
            message = event.data.get("message")
            if isinstance(message, dict) and message.get("id") == message_id:
                admitted = True
                target_turn = active_turn
                break
    if not admitted:
        # No turn can be correlated with a prompt the session never accepted.
        return False, None
    for event in reversed(relevant):
        if event.type != "turn/end":
            continue
        if target_turn is None or event.data.get("turn") == target_turn:
            return admitted, event.data.get("reason")
    return admitted, None


class HarnessAcpServer:
    """Serve the automation-only ACP method set over one HarnessService."""

    def __init__(self, service: HarnessService) -> None:
        self.service = service
        self.notify: Any = None
        self.request: Any = None
        self._sessions: set[str] = set()
        self._handles: dict[str, Any] = {}
        self._pending: set[str] = set()
        self._cancelled: set[str] = set()
        self._closed = False
        self._notify_tasks: set[asyncio.Task[None]] = set()
        self._approval_tasks: set[asyncio.Task[None]] = set()
        self._forwarder: asyncio.Task[None] | None = None

    def set_notification_sink(self, notify: Any) -> None:
        self.notify = notify

    def set_request_sink(self, request: Any) -> None:
        """Attach the transport's outbound-request sink and start forwarding."""

        self.request = request
        if request is not None and self._forwarder is None:
            self._forwarder = asyncio.get_running_loop().create_task(
                self._forward_approvals(), name="dsh-acp-approvals"
            )

    async def handle_request(self, method: str, params: Any = None) -> dict[str, Any]:
        params = params if isinstance(params, dict) else {}
        if method == "initialize":
            return {
                "protocolVersion": PROTOCOL_VERSION,
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
        if method == "authenticate":
            return {}
        if method == "session/new":
            return await self._session_new(params)
        if method == "session/prompt":
            return await self._session_prompt(params)
        if method == "session/cancel":
            return self._session_cancel(params)
        raise JsonRpcResponseError(-32601, f"unknown ACP method: {method}")

    async def close(self) -> None:
        # Closing the connection is equivalent to cancelling all bridge-owned
        # prompts. Mark them before disposing the service so their awaiters
        # resolve with ACP's cancellation reason rather than ordinary end_turn.
        self._closed = True
        self._cancelled.update(self._pending)
        for session_id in self._pending:
            handle = self._handles.get(session_id)
            if handle is not None:
                handle.queue.clear()
        if self._forwarder is not None:
            self._forwarder.cancel()
            await asyncio.gather(self._forwarder, return_exceptions=True)
            self._forwarder = None
        tasks = tuple(self._approval_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            self._approval_tasks.clear()
        await self.service.dispose()
        tasks = tuple(self._notify_tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            self._notify_tasks.clear()
        self._sessions.clear()
        self._handles.clear()
        # A prompt coroutine owns cleanup of its pending/cancelled markers. Do
        # not clear those sets while service.dispose() is waking that coroutine.
        if not self._pending:
            self._cancelled.clear()

    async def _session_new(self, params: dict[str, Any]) -> dict[str, Any]:
        self._assert_open()
        raw_cwd = params.get("cwd")
        if not isinstance(raw_cwd, str) or not raw_cwd:
            raise AcpProtocolError("session/new requires an absolute cwd")
        cwd = Path(raw_cwd)
        if not cwd.is_absolute():
            raise AcpProtocolError(f"cwd must be an absolute path: {raw_cwd}")
        cwd = cwd.expanduser()
        mcp_servers = params.get("mcpServers") or []
        extra_dirs = params.get("additionalDirectories") or []
        if mcp_servers:
            raise AcpProtocolError("mcpServers is not supported")
        if extra_dirs:
            raise AcpProtocolError("additionalDirectories is not supported")
        handle = await self.service.create_session(cwd=str(cwd))
        self._sessions.add(handle.session.id)
        self._handles[handle.session.id] = handle
        handle.agent.subscribe(lambda event: self._on_agent_event(handle.session.id, event))
        return {"sessionId": handle.session.id}

    async def _session_prompt(self, params: dict[str, Any]) -> dict[str, Any]:
        self._assert_open()
        session_id = params.get("sessionId")
        if not isinstance(session_id, str) or not session_id:
            raise AcpProtocolError("session/prompt requires a sessionId")
        if session_id not in self._sessions:
            raise AcpProtocolError(f"unknown session: {session_id}")
        if session_id in self._pending:
            raise AcpProtocolError(f"a prompt is already in flight for this session: {session_id}")
        blocks = params.get("prompt")
        if prompt_has_unsupported_content(blocks):
            raise AcpProtocolError("only text and resource_link prompt content is supported")
        text = render_prompt_blocks(blocks)
        if not text.strip():
            raise AcpProtocolError("empty prompt")
        handle = await self.service.get_session(session_id)
        first_seq = handle.session.seq
        self._pending.add(session_id)
        try:
            accepted = await self.service.prompt(
                session_id,
                [{"type": "text", "text": text}],
                include_message_id=True,
            )
            message_id = accepted.get("messageId")
            if not isinstance(message_id, str):
                message_id = None
            while True:
                await handle.agent.when_idle()
                if not handle.queue and (handle.task is None or handle.task.done()):
                    break
                await asyncio.sleep(0.02)

            admitted, reason_data = prompt_outcome(handle.session.events, first_seq, message_id)
            if session_id in self._cancelled:
                reason = "cancelled"
            elif not admitted:
                # The prompt never reached a turn: its admission was discarded,
                # so the contract reports cancellation rather than quiescence.
                reason = "cancelled"
            elif isinstance(reason_data, dict) and reason_data.get("kind") == "error":
                error = reason_data.get("error")
                detail = error.get("message") if isinstance(error, dict) else None
                raise JsonRpcResponseError(
                    -32603,
                    f"turn failed: {detail if isinstance(detail, str) else 'unknown error'}",
                )
            else:
                # The bridge settles token-limited turns at whole-agent idle;
                # ACP therefore receives end_turn rather than max_tokens here.
                reason = (
                    "end_turn"
                    if isinstance(reason_data, dict) and reason_data.get("kind") == "max-tokens"
                    else turn_end_to_stop_reason(reason_data)
                )
            return {"stopReason": reason}
        finally:
            self._pending.discard(session_id)
            self._cancelled.discard(session_id)

    def _assert_open(self) -> None:
        if self._closed:
            raise JsonRpcResponseError(-32603, "the ACP bridge has been disposed")

    def _session_cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = params.get("sessionId")
        if not isinstance(session_id, str) or not session_id:
            return {}
        # An idle cancel must not affect the next prompt, but a known session
        # may have autonomous queued work that the client still owns.
        handle = self._handles.get(session_id)
        if handle is None:
            return {}
        if session_id in self._pending:
            self._cancelled.add(session_id)
        task = handle.task
        if task is not None and not task.done():
            task.cancel()
        # The queue runner can be cancelled before its first scheduling turn;
        # remove queued work as well or the prompt waiter would wait forever.
        handle.queue.clear()
        return {}

    async def _forward_approvals(self) -> None:
        """Answer approval requests for bridge-owned sessions by machine policy."""

        try:
            async for frame in self.service.stream("mux"):
                if frame.get("type") != "approval/requested":
                    continue
                task = asyncio.get_running_loop().create_task(self._ask_permission(frame))
                self._approval_tasks.add(task)
                task.add_done_callback(self._approval_tasks.discard)
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    async def _ask_permission(self, frame: dict[str, Any]) -> None:
        session_id = frame.get("sessionId")
        approval_id = frame.get("approvalId")
        call_id = frame.get("callId")
        # Only bridge-owned sessions, and only requests carrying the tool call
        # id an ACP client can address, are answered here; everything else is
        # left to a UI answerer.
        if (
            self.request is None
            or not isinstance(session_id, str)
            or session_id not in self._sessions
            or not isinstance(approval_id, str)
            or not isinstance(call_id, str)
        ):
            return
        params = {
            "sessionId": session_id,
            "toolCall": {"toolCallId": call_id},
            "options": [
                {"optionId": "allow-once", "name": "Allow once", "kind": "allow_once"},
                {"optionId": "reject-once", "name": "Reject", "kind": "reject_once"},
            ],
        }
        try:
            response = await self.request("session/request_permission", params)
        except asyncio.CancelledError:
            raise
        except Exception:
            outcome: ApprovalOutcome = "unavailable"
        else:
            outcome = permission_outcome(response)
        await self.service.resolve_approval(session_id, approval_id, outcome)

    def _on_agent_event(self, session_id: str, event: SessionEvent) -> None:
        if self.notify is None:
            return
        if event.type == "assistant/message":
            message = event.data.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                return
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text = block.get("text")
                    if isinstance(text, str) and text:
                        task = asyncio.get_running_loop().create_task(
                            self._emit_chunk(session_id, text)
                        )
                        self._notify_tasks.add(task)
                        task.add_done_callback(self._notification_done)
                elif block_type == "image":
                    attachment = block.get("attachment")
                    attachment_id = (
                        attachment.get("attachmentId") if isinstance(attachment, dict) else None
                    )
                    if isinstance(attachment_id, str):
                        task = asyncio.get_running_loop().create_task(
                            self._emit_chunk(
                                session_id,
                                f"[image attachment {attachment_id}]",
                            )
                        )
                        self._notify_tasks.add(task)
                        task.add_done_callback(self._notification_done)
            # Presentation and trace data stay off the automation wire.
            return

    def _notification_done(self, task: asyncio.Task[None]) -> None:
        self._notify_tasks.discard(task)
        if not task.cancelled():
            # Retrieve sink failures so a disconnected client cannot produce an
            # unhandled-task warning or affect the owning turn.
            task.exception()

    async def _emit_chunk(self, session_id: str, text: str) -> None:
        if self.notify is None:
            return
        await self.notify(
            "session/update",
            {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": text},
                },
            },
        )


async def run_acp_server(
    *,
    cwd: str,
    model: str = "deepseek-v4-flash",
    session_root: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 120.0,
    input_stream: Any = None,
    output_stream: Any = None,
) -> None:
    """Serve one automation ACP connection over stdio streams."""

    from deepseek_harness.llm import DeepSeekAdapter
    from deepseek_harness.sdk_rpc import serve_stdio

    def adapter_factory(_model: str) -> DeepSeekAdapter:
        return DeepSeekAdapter(api_key=api_key, base_url=base_url, timeout=timeout)

    service = HarnessService(
        session_root or os.getenv("DSH_SESSION_ROOT", "~/.deepseek_harness_python/sessions"),
        cwd=cwd,
        model=model,
        adapter_factory=adapter_factory,
    )
    await serve_stdio(
        HarnessAcpServer(service),
        input_stream=input_stream,
        output_stream=output_stream,
    )
