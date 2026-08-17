"""Replayable context-window protection for the Python Agent loop.

The TypeScript runtime exposes compaction as a durable session capability.  The
Python runtime keeps the same important boundary: the append-only log remains
complete, while ``Session.derive_messages()`` exposes a bounded checkpoint plus
the recent tail to the next model request.

This first backend deliberately uses a deterministic local checkpoint instead
of a second LLM call.  It is therefore safe to use in tests, offline sessions,
and deployments that only configure one provider.  The event shape leaves room
for a provider-backed summarizer later without changing the session projection.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .llm.types import ToolSchema
from .models import (
    JsonValue,
    Message,
    RawContent,
    TextContent,
    ToolCallContent,
    ToolResultContent,
)
from .session import Session, SessionEvent

if TYPE_CHECKING:
    from .tool_result_pruner import ToolResultPruner

CompactionTrigger = Literal["pressure", "context-overflow", "manual"]
ManualCompactionErrorCode = Literal[
    "busy", "cancelled", "changed", "summary", "commit", "persistence"
]
CompactionAppender = Callable[[str, dict[str, JsonValue]], Awaitable[SessionEvent]]

CHECKPOINT_PREFIX = (
    "This is an automatically generated checkpoint condensing an earlier span of the "
    "conversation to free up context. Treat the captured context as established "
    "background and build on it without restating it. Continue the task directly from "
    "the messages that follow, without acknowledging this checkpoint."
)


class ManualCompactionError(RuntimeError):
    """An expected failure from an argument-free manual compaction request."""

    def __init__(self, code: ManualCompactionErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class CompactionPolicy:
    """Token-pressure and retention policy for one Agent route.

    Token counts use a stable JSON-character heuristic.  A provider-specific
    tokenizer can replace the estimator later; the policy and durable event
    contract do not depend on that implementation detail.
    """

    context_window_tokens: int = 1_000_000
    threshold_ratio: float = 0.8
    retain_ratio: float | None = 0.16
    retain_tokens: int | None = None
    chars_per_token: float = 4.0
    summary_max_chars: int = 12_000
    minimum_replaced_messages: int = 2
    minimum_retained_messages: int = 2
    enabled: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.context_window_tokens, bool)
            or not isinstance(self.context_window_tokens, int)
            or self.context_window_tokens <= 0
        ):
            raise ValueError("context_window_tokens must be a positive integer")
        if not 0 < self.threshold_ratio <= 1:
            raise ValueError("threshold_ratio must be greater than 0 and at most 1")
        if self.retain_ratio is not None and self.retain_tokens is not None:
            raise ValueError("retain_ratio and retain_tokens are mutually exclusive")
        if self.retain_ratio is None and self.retain_tokens is None:
            raise ValueError("one of retain_ratio or retain_tokens must be configured")
        if self.retain_ratio is not None and not 0 <= self.retain_ratio < self.threshold_ratio:
            raise ValueError("retain_ratio must be non-negative and below threshold_ratio")
        if self.retain_tokens is not None and (
            isinstance(self.retain_tokens, bool)
            or not isinstance(self.retain_tokens, int)
            or self.retain_tokens < 0
        ):
            raise ValueError("retain_tokens must be a non-negative integer")
        if self.retain_tokens is not None and self.retain_tokens >= self.threshold_tokens:
            raise ValueError("retain_tokens must be below threshold tokens")
        if self.chars_per_token <= 0 or not math.isfinite(self.chars_per_token):
            raise ValueError("chars_per_token must be finite and positive")
        if self.summary_max_chars <= 0:
            raise ValueError("summary_max_chars must be positive")
        if self.minimum_replaced_messages < 1:
            raise ValueError("minimum_replaced_messages must be positive")
        if self.minimum_retained_messages < 1:
            raise ValueError("minimum_retained_messages must be positive")

    @property
    def threshold_tokens(self) -> int:
        return max(1, math.floor(self.context_window_tokens * self.threshold_ratio))

    @property
    def retention_tokens(self) -> int:
        if self.retain_tokens is not None:
            return self.retain_tokens
        assert self.retain_ratio is not None
        return math.floor(self.context_window_tokens * self.retain_ratio)


@dataclass(frozen=True, slots=True)
class CompactionResult:
    """Durable coordinates and measurements for one completed compaction."""

    compaction_id: str
    start_seq: int
    summary_seq: int
    end_seq: int
    replaced_message_count: int
    retained_message_count: int
    replaced_token_count: int
    estimated_tokens_before: int
    estimated_tokens_after: int


def estimate_json_tokens(value: object, *, chars_per_token: float = 4.0) -> int:
    """Estimate tokens for a JSON-compatible request fragment."""

    if chars_per_token <= 0 or not math.isfinite(chars_per_token):
        raise ValueError("chars_per_token must be finite and positive")
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    return max(1, math.ceil(len(encoded) / chars_per_token))


def estimate_message_tokens(message: Message, *, chars_per_token: float = 4.0) -> int:
    """Estimate the serialized cost of one model message."""

    return estimate_json_tokens(message.to_dict(), chars_per_token=chars_per_token)


def estimate_messages_tokens(
    messages: Sequence[Message], *, chars_per_token: float = 4.0
) -> int:
    """Estimate the serialized cost of a message sequence."""

    return sum(
        estimate_message_tokens(message, chars_per_token=chars_per_token) for message in messages
    )


def estimate_request_overhead(
    system: str | None,
    tools: Sequence[ToolSchema],
    *,
    chars_per_token: float = 4.0,
) -> int:
    """Estimate system and tool-schema tokens not represented by session messages."""

    overhead = estimate_json_tokens(
        {"system": system or "", "tools": [tool.to_openai() for tool in tools]},
        chars_per_token=chars_per_token,
    )
    return overhead


async def compact_if_needed(
    session: Session,
    policy: CompactionPolicy,
    append: CompactionAppender,
    *,
    overhead_tokens: int = 0,
    trigger: CompactionTrigger = "pressure",
    turn: int | None = None,
    pruner: ToolResultPruner | None = None,
) -> CompactionResult | None:
    """Replace an old message prefix with a durable checkpoint when pressured.

    The function is intentionally serialized by the Agent's turn lock.  It
    computes and validates the replacement before appending ``start`` so a
    no-op never leaves a stale compaction marker in the session.
    """

    if not policy.enabled:
        return None
    if overhead_tokens < 0:
        raise ValueError("overhead_tokens must be non-negative")

    messages = session.derive_messages()
    estimated_before = overhead_tokens + estimate_messages_tokens(
        messages, chars_per_token=policy.chars_per_token
    )
    if estimated_before <= policy.threshold_tokens:
        return None

    if pruner is not None:
        pruned = pruner.prune_session(session)
        if pruned.pruned:
            messages = session.derive_messages()
            estimated_before = overhead_tokens + estimate_messages_tokens(
                messages, chars_per_token=policy.chars_per_token
            )
            if estimated_before <= policy.threshold_tokens:
                return None

    cut = _select_cut(messages, policy)
    if cut is None:
        return None
    replaced = messages[:cut]
    retained = messages[cut:]
    summary_text = _build_local_summary(replaced, policy.summary_max_chars)
    checkpoint_text = _frame_checkpoint(summary_text)
    checkpoint = Message(
        role="user",
        content=(TextContent(checkpoint_text),),
        source={"kind": "compaction", "compactionId": f"pending-{uuid.uuid4().hex}"},
    )
    estimated_after = overhead_tokens + estimate_messages_tokens(
        (checkpoint, *retained), chars_per_token=policy.chars_per_token
    )
    if estimated_after >= estimated_before:
        return None

    compaction_id = f"compaction-{uuid.uuid4().hex}"
    checkpoint = Message(
        role="user",
        content=(TextContent(checkpoint_text),),
        source={"kind": "compaction", "compactionId": compaction_id},
    )
    start_data: dict[str, JsonValue] = {
        "compactionId": compaction_id,
        "trigger": trigger,
        "estimatedTokensBefore": estimated_before,
        "thresholdTokens": policy.threshold_tokens,
        "retainTokens": policy.retention_tokens,
        "replacedMessageCount": len(replaced),
        "retainedMessageCount": len(retained),
    }
    if turn is not None:
        start_data["turn"] = turn
    else:
        start_data["turn"] = None
    start_event = await append("compaction/start", start_data)

    summary_data: dict[str, JsonValue] = {
        "compactionId": compaction_id,
        "trigger": trigger,
        "summary": summary_text,
        "message": checkpoint.to_dict(),
        "shadowedMessageIds": [message.id for message in replaced],
        "shadowedMessageCount": len(replaced),
        "shadowedTokenCount": estimate_messages_tokens(
            replaced, chars_per_token=policy.chars_per_token
        ),
    }
    if turn is not None:
        summary_data["turn"] = turn
    else:
        summary_data["turn"] = None
    summary_event = await append("compaction/summary", summary_data)
    await append(
        "user/message",
        {
            "message": checkpoint.to_dict(),
            "surfaceOp": {"op": "replace", "compactionId": compaction_id},
            "sourceEventSeqs": [start_event.seq, summary_event.seq],
        },
    )
    end_event = await append(
        "compaction/end",
        {
            "compactionId": compaction_id,
            "trigger": trigger,
            "status": "completed",
            "summarySeq": summary_event.seq,
            "estimatedTokensBefore": estimated_before,
            "estimatedTokensAfter": estimated_after,
            "replacedMessageCount": len(replaced),
            "retainedMessageCount": len(retained),
            "startSeq": start_event.seq,
        },
    )
    return CompactionResult(
        compaction_id=compaction_id,
        start_seq=start_event.seq,
        summary_seq=summary_event.seq,
        end_seq=end_event.seq,
        replaced_message_count=len(replaced),
        retained_message_count=len(retained),
        replaced_token_count=estimate_messages_tokens(
            replaced, chars_per_token=policy.chars_per_token
        ),
        estimated_tokens_before=estimated_before,
        estimated_tokens_after=estimated_after,
    )


def _select_cut(messages: Sequence[Message], policy: CompactionPolicy) -> int | None:
    """Choose an old-prefix boundary without splitting a tool exchange."""

    if len(messages) <= policy.minimum_replaced_messages:
        return None

    retain_budget = policy.retention_tokens
    tail_count = 0
    tail_tokens = 0
    for message in reversed(messages):
        message_tokens = estimate_message_tokens(message, chars_per_token=policy.chars_per_token)
        if (
            tail_count >= policy.minimum_retained_messages
            and tail_tokens + message_tokens > retain_budget
        ):
            break
        tail_count += 1
        tail_tokens += message_tokens
        if tail_count >= policy.minimum_retained_messages and tail_tokens >= retain_budget:
            break

    tail_count = max(tail_count, policy.minimum_retained_messages)
    tail_count = min(tail_count, len(messages) - policy.minimum_replaced_messages)
    cut = len(messages) - tail_count
    while cut > 0 and _is_tool_result_message(messages[cut]):
        cut -= 1
    if cut < policy.minimum_replaced_messages or len(messages) - cut < 1:
        return None
    return cut


def _is_tool_result_message(message: Message) -> bool:
    return any(isinstance(block, ToolResultContent) for block in message.content)


def _build_local_summary(messages: Sequence[Message], max_chars: int) -> str:
    """Create a bounded, privacy-preserving-enough transcript checkpoint."""

    lines: list[str] = []
    used = 0
    omitted = 0
    for message in messages:
        line = _message_summary_line(message)
        remaining = max_chars - used
        if remaining <= 0:
            omitted += 1
            continue
        if len(line) > remaining:
            if remaining <= 24:
                omitted += 1
                continue
            line = line[: remaining - 1].rstrip() + "…"
        lines.append(line)
        used += len(line) + 1
    if omitted:
        lines.append(f"- ({omitted} earlier message(s) omitted from the bounded checkpoint)")
    if not lines:
        return "- (no textual content in the replaced span)"
    return "\n".join(lines)


def _message_summary_line(message: Message) -> str:
    role = message.role
    text = message.text.strip().replace("\n", " ")
    if message.source.get("kind") == "compaction":
        text = _checkpoint_body(text).replace("\n", " ")
    tool_calls = [block for block in message.content if isinstance(block, ToolCallContent)]
    raw_blocks = [block for block in message.content if isinstance(block, RawContent)]
    if tool_calls:
        calls = ", ".join(call.name or "unnamed-tool" for call in tool_calls)
        text = f"{text} [tool call: {calls}]" if text else f"[tool call: {calls}]"
    if raw_blocks and not text:
        text = f"[{len(raw_blocks)} unsupported content block(s)]"
    if not text and any(isinstance(block, ToolResultContent) for block in message.content):
        text = "[tool result]"
    return f"- {role}: {text or '(empty message)'}"


def _checkpoint_body(text: str) -> str:
    start = text.find("<compacted-summary>")
    end = text.find("</compacted-summary>", start + 1)
    if start < 0 or end < 0:
        return text
    return text[start + len("<compacted-summary>") : end].strip()


def _frame_checkpoint(summary: str) -> str:
    return f"{CHECKPOINT_PREFIX}\n\n<compacted-summary>\n{summary}\n</compacted-summary>"


__all__ = [
    "CHECKPOINT_PREFIX",
    "ManualCompactionError",
    "ManualCompactionErrorCode",
    "CompactionPolicy",
    "CompactionResult",
    "CompactionTrigger",
    "compact_if_needed",
    "estimate_json_tokens",
    "estimate_message_tokens",
    "estimate_messages_tokens",
    "estimate_request_overhead",
]
