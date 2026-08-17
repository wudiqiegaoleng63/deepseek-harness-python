"""Semantic durability boundaries for model requests and tool dispatch."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .session import Session

CheckpointCallback = Callable[[Session], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class SessionCheckpointPolicy:
    """Flush a session before an external model or tool side effect."""

    flush: CheckpointCallback

    async def before_model_request(self, session: Session) -> None:
        await self.flush(session)

    async def before_tool_dispatch(self, session: Session) -> None:
        await self.flush(session)


__all__ = ["CheckpointCallback", "SessionCheckpointPolicy"]
