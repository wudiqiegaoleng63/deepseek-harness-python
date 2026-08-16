"""Default Agent loop over the Session, LLM, and Tool seams."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from ..llm.adapter import LlmAdapter
from ..llm.types import LlmCallConfig, LlmRequest, StreamChunk
from ..models import (
    JsonValue,
    Message,
    TextContent,
    ToolCallContent,
    create_tool_message,
    create_user_message,
)
from ..session import Session, SessionEvent
from ..tools.registry import ToolContext, ToolRegistry

AgentStatus = Literal["idle", "running", "disposed"]
EventListener = Callable[[SessionEvent], Any | Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class RunResult:
    session_id: str
    final_response: str
    finish_reason: str | None
    events: tuple[SessionEvent, ...]


@dataclass(slots=True)
class _ToolCallAccumulator:
    index: int
    call_id: str = ""
    name: str = ""
    arguments: str = ""


class Agent:
    """One owned Agent and its serial turn driver."""

    def __init__(
        self,
        session: Session,
        llm: LlmAdapter,
        *,
        tools: ToolRegistry | None = None,
        config: LlmCallConfig | None = None,
        system_prompt: str | Callable[[], str] | None = None,
        max_steps: int = 64,
    ) -> None:
        self.session = session
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.config = config or LlmCallConfig()
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.status: AgentStatus = "idle"
        self._listeners: list[EventListener] = []
        self._run_lock = asyncio.Lock()
        self._idle = asyncio.Event()
        self._idle.set()
        self._turn = self._next_turn_number()
        self._disposed = False

    def subscribe(self, listener: EventListener) -> Callable[[], None]:
        self._listeners.append(listener)
        active = True

        def dispose() -> None:
            nonlocal active
            if not active:
                return
            active = False
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

        return dispose

    async def run(self, prompt: str | Message) -> RunResult:
        """Queue and fully drain one ordinary user turn."""

        if self._disposed:
            raise RuntimeError("agent is disposed")
        async with self._run_lock:
            self._idle.clear()
            self.status = "running"
            first_seq = self.session.seq
            try:
                return await self._run_turn(prompt, first_seq)
            finally:
                self.status = "idle"
                self._idle.set()

    async def when_idle(self) -> None:
        await self._idle.wait()

    async def dispose(self) -> None:
        self._disposed = True
        await self.when_idle()
        self.status = "disposed"
        close = getattr(self.llm, "aclose", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result

    async def _run_turn(self, prompt: str | Message, first_seq: int) -> RunResult:
        turn = self._turn
        self._turn += 1
        await self._append("turn/start", {"turn": turn})
        user_message = prompt if isinstance(prompt, Message) else create_user_message(prompt)
        await self._append("user/message", {"message": user_message.to_dict()})
        final_text = ""
        finish_reason: str | None = None

        try:
            for step in range(1, self.max_steps + 1):
                await self._append("step/start", {"turn": turn, "step": step})
                text_parts: list[str] = []
                reasoning_parts: list[str] = []
                calls: dict[int, _ToolCallAccumulator] = {}
                request = LlmRequest(
                    messages=self.session.derive_messages(),
                    config=self.config,
                    system=(
                        self.system_prompt()
                        if callable(self.system_prompt)
                        else self.system_prompt
                    ),
                    tools=self.tools.schemas(),
                )
                try:
                    async for chunk in self.llm.stream(request):
                        await self._append(
                            "assistant/chunk",
                            {"turn": turn, "step": step, "chunk": chunk.to_dict()},
                        )
                        self._consume_chunk(chunk, text_parts, reasoning_parts, calls)
                        if chunk.kind == "done" and chunk.finish_reason:
                            finish_reason = self._normalize_finish_reason(chunk.finish_reason)
                except Exception as exc:
                    await self._append("step/end", {"turn": turn, "step": step})
                    reason = {"kind": "error", "error": {"code": "LLM_ERROR", "message": str(exc)}}
                    await self._append("turn/end", {"turn": turn, "reason": reason})
                    return self._result(first_seq, final_text, "error")

                blocks: list[Any] = []
                if reasoning_parts:
                    blocks.append({"type": "reasoning", "text": "".join(reasoning_parts)})
                if text_parts:
                    final_text = "".join(text_parts)
                    blocks.append(TextContent(final_text))
                for call in sorted(calls.values(), key=lambda item: item.index):
                    if not call.call_id:
                        call.call_id = f"call-{call.index}"
                    blocks.append(ToolCallContent(call.call_id, call.name, call.arguments))
                assistant = Message(
                    role="assistant",
                    content=tuple(self._block_from_dict_or_value(block) for block in blocks),
                    source={"kind": "assistant"},
                )
                await self._append(
                    "assistant/message",
                    {"turn": turn, "step": step, "message": assistant.to_dict()},
                )
                await self._append("step/end", {"turn": turn, "step": step})

                if calls:
                    for call in sorted(calls.values(), key=lambda item: item.index):
                        await self._append(
                            "tool/call",
                            {
                                "turn": turn,
                                "step": step,
                                "callId": call.call_id,
                                "name": call.name,
                                "arguments": call.arguments,
                            },
                        )
                        result = await self.tools.execute(
                            call.name,
                            call.arguments,
                            ToolContext(
                                session_id=self.session.id, cwd=self.session.header.cwd or "."
                            ),
                        )
                        tool_message = create_tool_message(call.call_id, result.text)
                        data: dict[str, JsonValue] = {
                            "turn": turn,
                            "step": step,
                            "message": tool_message.to_dict(),
                        }
                        if result.is_error:
                            data["error"] = {"name": "ToolError", "code": "TOOL_EXECUTION_FAILED"}
                        if result.meta is not None:
                            data["meta"] = result.meta
                        await self._append("tool/result", data)
                    continue

                terminal_reason = finish_reason or "completed"
                await self._append(
                    "turn/end",
                    {"turn": turn, "reason": {"kind": terminal_reason}},
                )
                return self._result(first_seq, final_text, terminal_reason)

            await self._append("turn/end", {"turn": turn, "reason": {"kind": "max-steps"}})
            return self._result(first_seq, final_text, "max-steps")
        except asyncio.CancelledError:
            await self._append("turn/end", {"turn": turn, "reason": {"kind": "aborted"}})
            raise

    async def _append(self, event_type: str, data: dict[str, JsonValue]) -> SessionEvent:
        event = self.session.append(event_type, data)
        for listener in tuple(self._listeners):
            result = listener(event)
            if inspect.isawaitable(result):
                await result
        return event

    @staticmethod
    def _consume_chunk(
        chunk: StreamChunk,
        text_parts: list[str],
        reasoning_parts: list[str],
        calls: dict[int, _ToolCallAccumulator],
    ) -> None:
        if chunk.kind == "text":
            text_parts.append(chunk.text)
        elif chunk.kind == "reasoning":
            reasoning_parts.append(chunk.text)
        elif chunk.kind == "tool-call-delta":
            call = calls.setdefault(chunk.index, _ToolCallAccumulator(chunk.index))
            if chunk.call_id:
                call.call_id = chunk.call_id
            if chunk.name:
                call.name = chunk.name
            call.arguments += chunk.arguments

    @staticmethod
    def _normalize_finish_reason(reason: str) -> str:
        if reason in {"length", "max_tokens"}:
            return "max-tokens"
        if reason in {"tool_calls", "tool-call"}:
            return "completed"
        if reason in {"stop", "completed"}:
            return "completed"
        return reason

    @staticmethod
    def _block_from_dict_or_value(value: Any) -> Any:
        if isinstance(value, (TextContent, ToolCallContent)):
            return value
        if isinstance(value, dict) and value.get("type") == "reasoning":
            from ..models import ReasoningContent

            return ReasoningContent(str(value.get("text", "")))
        return value

    def _result(self, first_seq: int, text: str, reason: str | None) -> RunResult:
        return RunResult(
            session_id=self.session.id,
            final_response=text,
            finish_reason=reason,
            events=tuple(event for event in self.session.events if event.seq >= first_seq),
        )

    def _next_turn_number(self) -> int:
        turns: list[int] = []
        for event in self.session.events:
            turn = event.data.get("turn")
            if event.type == "turn/end" and isinstance(turn, int):
                turns.append(turn)
        return max(turns, default=0) + 1
