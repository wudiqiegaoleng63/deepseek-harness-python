"""Session-scoped local storage and best-effort tool-result spill policy."""

from __future__ import annotations

import asyncio
import hashlib
import os
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .tools.registry import ToolContext, ToolResult


@dataclass(frozen=True, slots=True)
class SpillOwner:
    session_id: str


@dataclass(frozen=True, slots=True)
class SpillSource:
    tool_name: str
    call_id: str
    label: str = "result"


@dataclass(frozen=True, slots=True)
class SaveTextSpill:
    owner: SpillOwner
    source: SpillSource
    suggested_name: str
    content: str


@dataclass(frozen=True, slots=True)
class SpillRef:
    locator: str
    bytes: int
    retrieval_hint: str


class SpillStore(Protocol):
    async def save_text(self, input: SaveTextSpill) -> SpillRef:
        """Persist the complete input text and return its opaque locator."""

        ...


class LocalSpillStore:
    """Private, session-scoped files with exclusive owner-only writes."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve()

    def session_root(self, session_id: str) -> Path:
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]
        return self.root / f"session-{digest}"

    async def save_text(self, input: SaveTextSpill) -> SpillRef:
        if not input.owner.session_id:
            raise ValueError("spill owner session_id must be non-empty")
        if not input.source.tool_name or not input.source.call_id:
            raise ValueError("spill source tool_name and call_id must be non-empty")
        path = await asyncio.to_thread(
            _save_text_file,
            self.root,
            input.owner.session_id,
            input.suggested_name,
            input.content,
        )
        return SpillRef(
            locator=str(path),
            bytes=len(input.content.encode("utf-8")),
            retrieval_hint="Use read with offset/limit, or grep this path to search within it.",
        )


ToolResultTransformer = Callable[
    [str, ToolContext, ToolResult], ToolResult | Awaitable[ToolResult]
]


class SpillPolicy:
    """Replace oversized successful plain-text results with a bounded recovery notice."""

    def __init__(self, store: SpillStore, *, max_inline_bytes: int | None = 50_000) -> None:
        if max_inline_bytes is not None and (
            isinstance(max_inline_bytes, bool)
            or not isinstance(max_inline_bytes, int)
            or max_inline_bytes < 0
        ):
            raise ValueError("max_inline_bytes must be a non-negative integer or None")
        self.store = store
        self.max_inline_bytes = max_inline_bytes

    async def transform(
        self,
        tool_name: str,
        context: ToolContext,
        result: ToolResult,
    ) -> ToolResult:
        cap = self.max_inline_bytes
        if cap is None or result.is_error or tool_name == "read":
            return result
        total_bytes = len(result.text.encode("utf-8"))
        if total_bytes <= cap:
            return result
        try:
            ref = await self.store.save_text(
                SaveTextSpill(
                    owner=SpillOwner(context.session_id),
                    source=SpillSource(
                        tool_name,
                        context.call_id or f"tool-{context.session_id}",
                        "result",
                    ),
                    suggested_name=f"{tool_name}.txt",
                    content=result.text,
                )
            )
        except Exception:
            return result

        reserve_notice = _spill_notice(total_bytes, ref)
        preview_budget = max(0, cap - len((reserve_notice + "\n\n").encode("utf-8")))
        preview, omitted = _head_tail_preview(result.text, preview_budget)
        notice = _spill_notice(omitted, ref)
        replacement = f"{preview}\n\n{notice}" if preview else notice
        replacement_bytes = len(replacement.encode("utf-8"))
        if replacement_bytes > cap or replacement_bytes >= total_bytes:
            return result
        return ToolResult(replacement, meta=result.meta)


def _spill_notice(omitted_bytes: int, ref: SpillRef) -> str:
    return (
        f"(Omitted {omitted_bytes} bytes. Full formatted result stored at: "
        f"{ref.locator}. {ref.retrieval_hint})"
    )


def _head_tail_preview(text: str, budget: int) -> tuple[str, int]:
    raw = text.encode("utf-8")
    if len(raw) <= budget:
        return text, 0
    head_budget = (budget + 1) // 2
    tail_budget = budget // 2
    head = raw[:head_budget].decode("utf-8", errors="ignore")
    tail = raw[-tail_budget:].decode("utf-8", errors="ignore") if tail_budget else ""
    kept_bytes = len(head.encode("utf-8")) + len(tail.encode("utf-8"))
    return head + tail, len(raw) - kept_bytes


def _save_text_file(
    root: Path,
    session_id: str,
    suggested_name: str,
    content: str,
) -> Path:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        root.chmod(0o700)
    except OSError:
        pass
    session_dir = root / f"session-{hashlib.sha256(session_id.encode('utf-8')).hexdigest()[:12]}"
    session_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        session_dir.chmod(0o700)
    except OSError:
        pass
    filename = f"{secrets.token_hex(6)}-{_encode_segment(suggested_name)}"
    path = session_dir / filename
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            descriptor = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    return path


def _encode_segment(raw: str) -> str:
    if raw == "":
        return "~"
    if raw == ".":
        return "~002E"
    if raw == "..":
        return "~002E~002E"
    output: list[str] = []
    for char in raw:
        if char.isascii() and (char.isalnum() or char in "._-"):
            output.append(char)
        else:
            output.append(f"~{ord(char):04X}")
    return "".join(output)


__all__ = [
    "LocalSpillStore",
    "SaveTextSpill",
    "SpillOwner",
    "SpillPolicy",
    "SpillRef",
    "SpillSource",
    "SpillStore",
    "ToolResultTransformer",
]
