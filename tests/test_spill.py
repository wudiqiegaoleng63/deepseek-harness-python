from __future__ import annotations

import asyncio
import os
from pathlib import Path

from deepseek_harness.spill import (
    LocalSpillStore,
    SaveTextSpill,
    SpillOwner,
    SpillPolicy,
    SpillRef,
    SpillSource,
)
from deepseek_harness.tools.registry import ToolContext, ToolDefinition, ToolRegistry, ToolResult


def test_local_spill_store_writes_private_session_scoped_files(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = LocalSpillStore(tmp_path / "spill")
        ref = await store.save_text(
            SaveTextSpill(
                owner=SpillOwner("session/with traversal"),
                source=SpillSource("grep", "call-1"),
                suggested_name="../result:?.txt",
                content="完整 output\n",
            )
        )
        path = Path(ref.locator)
        assert path.read_text(encoding="utf-8") == "完整 output\n"
        assert path.parent == store.session_root("session/with traversal")
        assert path.parent.parent == store.root
        assert os.stat(store.root).st_mode & 0o777 == 0o700
        assert os.stat(path.parent).st_mode & 0o777 == 0o700
        assert os.stat(path).st_mode & 0o777 == 0o600
        assert "/../" not in ref.locator
        assert "/" not in path.name
        assert "~002F" in path.name and "~003A" in path.name and "~003F" in path.name
        assert ref.bytes == len("完整 output\n".encode())

    asyncio.run(scenario())


class RecordingSpillStore:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.saved: list[SaveTextSpill] = []

    async def save_text(self, input: SaveTextSpill) -> SpillRef:
        if self.fail:
            raise OSError("disk full")
        self.saved.append(input)
        return SpillRef("/spill/result.txt", len(input.content.encode("utf-8")), "use read")


def test_spill_policy_bounds_successful_results_and_preserves_best_effort_fallbacks() -> None:
    async def scenario() -> None:
        store = RecordingSpillStore()
        policy = SpillPolicy(store, max_inline_bytes=200)
        result = await policy.transform(
            "grep",
            ToolContext("session-1", ".", call_id="call-1"),
            ToolResult("HEAD-" + "x" * 1_500 + "-TAIL", meta={"count": 1}),
        )
        assert not result.is_error
        assert result.meta == {"count": 1}
        assert len(result.text.encode("utf-8")) <= 200
        assert "HEAD-" in result.text
        assert "-TAIL" in result.text
        assert "Full formatted result stored at: /spill/result.txt" in result.text
        assert store.saved[0].source == SpillSource("grep", "call-1")
        assert store.saved[0].content.startswith("HEAD-")

        small = ToolResult("tiny")
        assert await policy.transform("grep", ToolContext("s", "."), small) == small
        read = ToolResult("x" * 1_000)
        assert await policy.transform("read", ToolContext("s", "."), read) == read
        error = ToolResult("x" * 1_000, is_error=True)
        assert await policy.transform("grep", ToolContext("s", "."), error) == error

        failing = SpillPolicy(RecordingSpillStore(fail=True), max_inline_bytes=10)
        original = ToolResult("x" * 100)
        assert await failing.transform("grep", ToolContext("s", "."), original) == original

    asyncio.run(scenario())


def test_tool_registry_applies_spill_transformer_after_tool_execution() -> None:
    async def scenario() -> None:
        store = RecordingSpillStore()
        registry = ToolRegistry(
            result_transformer=SpillPolicy(store, max_inline_bytes=100).transform
        )
        registry.register(
            ToolDefinition(
                "big",
                "big",
                {"type": "object", "additionalProperties": False},
                lambda _args, _context: ToolResult("z" * 500),
            )
        )
        result = await registry.execute("big", "{}", ToolContext("session-1", ".", "call-9"))
        assert len(result.text.encode("utf-8")) <= 100
        assert store.saved[0].source.call_id == "call-9"

    asyncio.run(scenario())
