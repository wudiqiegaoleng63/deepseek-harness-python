from __future__ import annotations

import pytest

from deepseek_harness import (
    PRUNE_MARKER,
    ToolResultPruneConfig,
    ToolResultPruner,
)
from deepseek_harness.models import (
    ReasoningContent,
    TextContent,
    ToolResultContent,
    create_tool_message,
)
from deepseek_harness.session import Session


def test_tool_result_prune_config_validates_budgets_and_unicode_content() -> None:
    config = ToolResultPruneConfig(threshold_chars=50, head_chars=4, tail_chars=3)
    pruner = ToolResultPruner(config)
    result = pruner.prune_content(
        [TextContent("😀" * 60), ReasoningContent("keep this rich block")]
    )

    assert result is not None
    assert result[0] == TextContent("😀" * 4 + PRUNE_MARKER + "😀" * 3)
    assert result[1] == ReasoningContent("keep this rich block")
    assert pruner.measure_content(result) <= 50

    with pytest.raises(ValueError, match="at most threshold_chars"):
        ToolResultPruneConfig(threshold_chars=20, head_chars=10, tail_chars=10)
    with pytest.raises(ValueError, match="positive integer"):
        ToolResultPruneConfig(threshold_chars=0)


def test_tool_result_pruner_replaces_current_surface_and_replays_losslessly() -> None:
    session = Session("prune-session")
    message = create_tool_message(
        "call-1",
        "x" * 100,
        source={"kind": "tool", "callId": "call-1"},
    )
    original = session.append(
        "tool/result",
        {
            "turn": 1,
            "step": 1,
            "message": message.to_dict(),
            "isError": True,
            "meta": {"future": {"nested": True}},
        },
    )
    session.append("turn/start", {"turn": 2})
    pruner = ToolResultPruner(
        ToolResultPruneConfig(threshold_chars=50, head_chars=4, tail_chars=3)
    )

    result = pruner.prune_session(session)

    assert len(result.pruned) == 1
    entry = result.pruned[0]
    assert entry.original_seq == original.seq
    assert entry.call_id == "call-1"
    assert entry.chars_after <= 50
    assert session.events[original.seq].data["message"] == message.to_dict()
    replacement = session.events[entry.replacement_seq]
    assert replacement.type == "tool/result"
    assert replacement.data["meta"] == {"future": {"nested": True}}
    assert replacement.data["surfaceOp"] == {
        "op": "replace",
        "start": original.seq,
        "end": original.seq,
    }
    assert session.current_surface()[0].seq == entry.replacement_seq
    assert PRUNE_MARKER in session.derive_messages()[0].text

    replay = Session.from_jsonl(session.to_jsonl())
    assert replay.derive_messages() == session.derive_messages()
    assert pruner.prune_session(session).pruned == ()


def test_tool_result_pruner_skips_short_results_and_preserves_non_text_blocks() -> None:
    session = Session("prune-multiple")
    short = create_tool_message("short", "small")
    session.append("tool/result", {"message": short.to_dict()})
    rich = create_tool_message("rich", "A" * 100)
    rich = rich.__class__(
        rich.role,
        (ToolResultContent("rich", "A" * 100), ReasoningContent("private")),
        rich.source,
        rich.id,
    )
    session.append("tool/result", {"message": rich.to_dict()})
    pruner = ToolResultPruner(
        ToolResultPruneConfig(threshold_chars=50, head_chars=4, tail_chars=3)
    )

    result = pruner.prune_session(session)

    assert [entry.call_id for entry in result.pruned] == ["rich"]
    projected = session.derive_messages()
    assert projected[0].text == "small"
    assert projected[1].content[1] == ReasoningContent("private")
