"""The provider-neutral LLM adapter protocol."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from .types import LlmRequest, StreamChunk


class LlmAdapter(Protocol):
    """A model provider that yields normalized stream chunks."""

    def stream(self, request: LlmRequest) -> AsyncIterator[StreamChunk]:
        """Start one request and yield provider-neutral chunks."""
        ...

    async def aclose(self) -> None:
        """Release provider resources."""
