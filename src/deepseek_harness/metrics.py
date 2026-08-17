"""Replayable usage, context-pressure, and session-statistic projections.

These folds intentionally read the durable session log instead of keeping
process-local counters.  A resumed session therefore exposes the same values
to the shared frontend, while provider usage remains optional for adapters
that do not report it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .llm.types import ToolSchema
from .models import (
    ContentBlock,
    ImageContent,
    Message,
    RawContent,
    ReasoningContent,
    TextContent,
    ToolCallContent,
    ToolResultContent,
)
from .session import SessionEvent

CHARS_PER_TOKEN = 4
BLOCK_OVERHEAD = 4
ROLE_OVERHEAD = 4


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _first_int(value: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        candidate = _nonnegative_int(value.get(key))
        if candidate is not None:
            return candidate
    return None


def normalize_usage(value: object) -> dict[str, int] | None:
    """Normalize camelCase or OpenAI-style usage into disjoint DSH buckets."""

    if not isinstance(value, Mapping):
        return None
    input_tokens = _first_int(value, "inputTokens", "input_tokens")
    output_tokens = _first_int(value, "outputTokens", "output_tokens")
    prompt_tokens = _first_int(value, "prompt_tokens")
    completion_tokens = _first_int(value, "completion_tokens")
    cache_read = _first_int(
        value,
        "cacheReadTokens",
        "cache_read_tokens",
        "prompt_cache_hit_tokens",
    )
    details = value.get("prompt_tokens_details")
    if cache_read is None and isinstance(details, Mapping):
        cache_read = _first_int(details, "cached_tokens", "cacheReadTokens")
    cache_write = _first_int(value, "cacheWriteTokens", "cache_write_tokens")
    if input_tokens is None and prompt_tokens is not None:
        input_tokens = max(0, prompt_tokens - (cache_read or 0))
    if output_tokens is None and completion_tokens is not None:
        output_tokens = completion_tokens
    if input_tokens is None and output_tokens is None:
        return None
    return {
        "inputTokens": input_tokens or 0,
        "outputTokens": output_tokens or 0,
        **({"cacheReadTokens": cache_read} if cache_read is not None else {}),
        **({"cacheWriteTokens": cache_write} if cache_write is not None else {}),
    }


def _usage_from_event(event: SessionEvent) -> dict[str, int] | None:
    if event.type == "assistant/chunk":
        chunk = event.data.get("chunk")
        if isinstance(chunk, Mapping):
            return normalize_usage(chunk.get("usage"))
    if event.type == "assistant/message":
        return normalize_usage(event.data.get("usage"))
    return None


def token_usage_projection(events: Sequence[SessionEvent]) -> dict[str, int]:
    """Accumulate usage, replacing an earlier sample from the same step."""

    totals = {
        "uncachedInputTokens": 0,
        "outputTokens": 0,
        "cacheReadTokens": 0,
        "cacheWriteTokens": 0,
    }
    last_key: tuple[int, int] | None = None
    last_buckets: dict[str, int] | None = None
    for event in events:
        usage = _usage_from_event(event)
        turn = event.data.get("turn")
        step = event.data.get("step")
        if usage is None or not isinstance(turn, int) or not isinstance(step, int):
            continue
        buckets = {
            "uncachedInputTokens": usage.get("inputTokens", 0),
            "outputTokens": usage.get("outputTokens", 0),
            "cacheReadTokens": usage.get("cacheReadTokens", 0),
            "cacheWriteTokens": usage.get("cacheWriteTokens", 0),
        }
        previous = last_buckets if last_key == (turn, step) else None
        for name in totals:
            totals[name] += buckets[name] - (previous.get(name, 0) if previous else 0)
        last_key = (turn, step)
        last_buckets = buckets
    return totals


def context_pressure_projection(events: Sequence[SessionEvent]) -> dict[str, int]:
    """Return the newest provider prompt pressure and advertised capacity."""

    pressure: int | None = None
    context_window: int | None = None
    for event in events:
        if event.type == "request/context":
            candidate = _nonnegative_int(event.data.get("contextWindow"))
            if candidate is not None and candidate > 0:
                context_window = candidate
            elif "contextWindow" in event.data:
                context_window = None
        usage = _usage_from_event(event)
        if usage is not None:
            pressure = (
                usage.get("inputTokens", 0)
                + usage.get("cacheReadTokens", 0)
                + usage.get("cacheWriteTokens", 0)
            )
    return {
        **({"pressureTokens": pressure} if pressure is not None else {}),
        **({"contextWindow": context_window} if context_window is not None else {}),
    }


def _price_text(value: str) -> int:
    return math.ceil(len(value) / CHARS_PER_TOKEN)


def estimate_content_tokens(content: Sequence[ContentBlock]) -> int:
    """Price content using the same fixed-density heuristic as DSH."""

    total = 0
    for block in content:
        if isinstance(block, (TextContent, ReasoningContent)):
            total += _price_text(block.text) + BLOCK_OVERHEAD
        elif isinstance(block, ToolCallContent):
            total += _price_text(block.name) + _price_text(block.arguments) + BLOCK_OVERHEAD
        elif isinstance(block, ToolResultContent):
            total += _price_text(block.text) + BLOCK_OVERHEAD
        elif isinstance(block, ImageContent):
            total += BLOCK_OVERHEAD + _price_text(json.dumps(block.to_dict(), ensure_ascii=False))
        elif isinstance(block, RawContent):
            total += BLOCK_OVERHEAD + _price_text(json.dumps(block.to_dict(), ensure_ascii=False))
    return total


def estimate_message_surface_tokens(message: Message) -> int:
    return estimate_content_tokens(message.content) + ROLE_OVERHEAD


def context_breakdown_projection(
    messages: Sequence[Message],
    *,
    system: str | None,
    tools: Sequence[ToolSchema],
) -> dict[str, int]:
    """Estimate system, tool-schema, and current conversation composition."""

    system_tokens = 0 if system is None else _price_text(system) + ROLE_OVERHEAD
    tools_tokens = 0
    if tools:
        encoded = json.dumps(
            [tool.to_openai() for tool in tools],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        tools_tokens = _price_text(encoded) + BLOCK_OVERHEAD
    return {
        "systemTokens": system_tokens,
        "toolsTokens": tools_tokens,
        "messageTokens": sum(estimate_message_surface_tokens(message) for message in messages),
    }


def _is_token_delta(chunk: Mapping[str, Any]) -> bool:
    kind = chunk.get("kind") or chunk.get("type")
    if kind in {"text", "reasoning"}:
        return isinstance(chunk.get("text"), str) and bool(chunk["text"])
    if kind == "tool-call-delta":
        return any(
            isinstance(chunk.get(key), str) and bool(chunk[key])
            for key in ("arguments", "name", "callId")
        )
    return False


def _tool_result_call_id(event: SessionEvent) -> str | None:
    message = event.data.get("message")
    if not isinstance(message, Mapping):
        return None
    source = message.get("source")
    if isinstance(source, Mapping):
        source_call_id = source.get("callId")
        if isinstance(source_call_id, str):
            return source_call_id
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, Mapping):
                block_call_id = block.get("callId")
                if isinstance(block_call_id, str):
                    return block_call_id
    return None


def session_stats_projection(events: Sequence[SessionEvent]) -> dict[str, int]:
    """Fold turn/step counts and matched model/tool wall times."""

    totals = {
        "turns": 0,
        "steps": 0,
        "llmMs": 0,
        "toolMs": 0,
        "ttftMs": 0,
        "ttftSteps": 0,
        "decodeMs": 0,
        "decodeTokens": 0,
    }
    last_turn: int | None = None
    open_step: tuple[int, int, int, int | None] | None = None
    pending_calls: dict[str, int] = {}
    for event in events:
        if event.type == "step/start":
            turn = event.data.get("turn")
            step = event.data.get("step")
            if isinstance(turn, int) and isinstance(step, int):
                open_step = (turn, step, event.time, None)
        elif event.type == "assistant/chunk" and open_step is not None:
            turn = event.data.get("turn")
            step = event.data.get("step")
            chunk = event.data.get("chunk")
            if (
                isinstance(turn, int)
                and isinstance(step, int)
                and isinstance(chunk, Mapping)
                and (turn, step) == open_step[:2]
                and open_step[3] is None
                and _is_token_delta(chunk)
            ):
                open_step = (open_step[0], open_step[1], open_step[2], event.time)
        elif event.type == "assistant/message" and open_step is not None:
            turn = event.data.get("turn")
            step = event.data.get("step")
            if (
                not isinstance(turn, int)
                or not isinstance(step, int)
                or (turn, step) != open_step[:2]
            ):
                continue
            totals["llmMs"] += max(0, event.time - open_step[2])
            first_token = open_step[3]
            if first_token is not None:
                totals["ttftMs"] += max(0, first_token - open_step[2])
                totals["ttftSteps"] += 1
                usage = normalize_usage(event.data.get("usage"))
                if usage is not None:
                    totals["decodeMs"] += max(0, event.time - first_token)
                    totals["decodeTokens"] += usage.get("outputTokens", 0)
            open_step = None
        elif event.type == "tool/call":
            call_id = event.data.get("callId")
            if isinstance(call_id, str):
                pending_calls[call_id] = event.time
        elif event.type == "tool/result":
            call_id = _tool_result_call_id(event)
            dispatched = pending_calls.pop(call_id, None) if call_id is not None else None
            if dispatched is not None:
                totals["toolMs"] += max(0, event.time - dispatched)
        elif event.type == "step/end":
            turn = event.data.get("turn")
            if isinstance(turn, int):
                if turn != last_turn:
                    totals["turns"] += 1
                last_turn = turn
            totals["steps"] += 1
            open_step = None
        elif event.type == "turn/end":
            pending_calls.clear()
    return totals


__all__ = [
    "context_breakdown_projection",
    "context_pressure_projection",
    "estimate_content_tokens",
    "estimate_message_surface_tokens",
    "normalize_usage",
    "session_stats_projection",
    "token_usage_projection",
]
