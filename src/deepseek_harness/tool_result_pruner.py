"""Replay-safe, model-free pruning for oversized tool results."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, replace

from .compaction import estimate_message_tokens
from .models import ContentBlock, Message, TextContent, ToolResultContent
from .session import Session

PRUNE_MARKER = "\n\n[... tool result middle pruned ...]\n\n"


@dataclass(frozen=True, slots=True)
class ToolResultPruneConfig:
    """Character budgets for deterministic head/middle/tail pruning."""

    threshold_chars: int = 8_192
    head_chars: int = 4_096
    tail_chars: int = 1_024

    def __post_init__(self) -> None:
        _assert_positive_integer("threshold_chars", self.threshold_chars)
        _assert_nonnegative_integer("head_chars", self.head_chars)
        _assert_nonnegative_integer("tail_chars", self.tail_chars)
        emitted = self.head_chars + len(PRUNE_MARKER) + self.tail_chars
        if emitted > self.threshold_chars:
            raise ValueError(
                "head_chars + marker + tail_chars "
                f"({emitted}) must be at most threshold_chars ({self.threshold_chars})"
            )


@dataclass(frozen=True, slots=True)
class PrunedEntry:
    """Coordinates and size accounting for one replacement."""

    original_seq: int
    replacement_seq: int
    call_id: str
    chars_before: int
    chars_after: int


@dataclass(frozen=True, slots=True)
class PruneResult:
    """Aggregate result from one stable-surface pruning pass."""

    pruned: tuple[PrunedEntry, ...]
    chars_removed: int


class ToolResultPruner:
    """Rewrite only the current model surface while retaining the full log."""

    def __init__(self, config: ToolResultPruneConfig | None = None) -> None:
        self.config = config or DEFAULT_TOOL_RESULT_PRUNE_CONFIG

    @staticmethod
    def measure_content(blocks: Sequence[ContentBlock]) -> int:
        """Count Unicode code points in text-bearing tool-result blocks."""

        return sum(
            len(block.text)
            for block in blocks
            if isinstance(block, (TextContent, ToolResultContent))
        )

    def prune_content(self, blocks: Sequence[ContentBlock]) -> tuple[ContentBlock, ...] | None:
        """Keep a bounded head and tail, replacing the middle with a marker."""

        total_chars = self.measure_content(blocks)
        if total_chars <= self.config.threshold_chars:
            return None

        removed_start = self.config.head_chars
        removed_end = total_chars - self.config.tail_chars
        pruned: list[ContentBlock] = []
        consumed = 0
        marker_inserted = False

        for block in blocks:
            if not isinstance(block, (TextContent, ToolResultContent)):
                pruned.append(block)
                continue

            points = list(block.text)
            block_start = consumed
            block_end = block_start + len(points)
            head_end = min(len(points), max(0, removed_start - block_start))
            tail_start = min(len(points), max(0, removed_end - block_start))
            intersects_removed = block_start < removed_end and block_end > removed_start
            marker = PRUNE_MARKER if intersects_removed and not marker_inserted else ""
            if marker:
                marker_inserted = True
            text = "".join(points[:head_end]) + marker + "".join(points[tail_start:])
            if text:
                pruned.append(replace(block, text=text))
            consumed = block_end

        if not marker_inserted:
            raise RuntimeError("tool-result prune failed to locate the removed text span")
        chars_after = self.measure_content(pruned)
        if chars_after > self.config.threshold_chars or chars_after >= total_chars:
            raise RuntimeError("tool-result prune replacement must shrink within the budget")
        return tuple(pruned)

    def prune_session(self, session: Session) -> PruneResult:
        """Prune every oversized tool result in one stable surface snapshot."""

        candidates = tuple(
            node
            for node in session.current_surface()
            if node.event.type == "tool/result"
        )
        entries: list[PrunedEntry] = []
        chars_removed = 0

        for node in candidates:
            content = self.prune_content(node.message.content)
            if content is None:
                continue
            call_id = _tool_call_id(node.message)
            if call_id is None:
                continue
            chars_before = self.measure_content(node.message.content)
            chars_after = self.measure_content(content)
            replacement_message = Message(
                node.message.role,
                content,
                node.message.source,
                node.message.id,
            )
            replacement_data = copy.deepcopy(node.event.data)
            replacement_data["message"] = replacement_message.to_dict()
            replacement_data["surfaceOp"] = {
                "op": "replace",
                "start": node.event.seq,
                "end": node.event.seq,
            }
            replacement_data["sourceEventSeqs"] = [node.event.seq]
            session.append(
                "compaction/prune",
                {
                    "shadowedRange": {"start": node.event.seq, "end": node.event.seq},
                    "shadowedSeqs": [node.event.seq],
                    "shadowedTokenCount": estimate_message_tokens(node.message),
                },
            )
            replacement = session.append("tool/result", replacement_data)
            entries.append(
                PrunedEntry(
                    original_seq=node.event.seq,
                    replacement_seq=replacement.seq,
                    call_id=call_id,
                    chars_before=chars_before,
                    chars_after=chars_after,
                )
            )
            chars_removed += chars_before - chars_after

        return PruneResult(tuple(entries), chars_removed)


def _tool_call_id(message: Message) -> str | None:
    for block in message.content:
        if isinstance(block, ToolResultContent) and block.call_id:
            return block.call_id
    value = message.source.get("callId")
    return value if isinstance(value, str) and value else None


def _assert_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _assert_nonnegative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


DEFAULT_TOOL_RESULT_PRUNE_CONFIG = ToolResultPruneConfig()


__all__ = [
    "DEFAULT_TOOL_RESULT_PRUNE_CONFIG",
    "PRUNE_MARKER",
    "PruneResult",
    "PrunedEntry",
    "ToolResultPruneConfig",
    "ToolResultPruner",
]
