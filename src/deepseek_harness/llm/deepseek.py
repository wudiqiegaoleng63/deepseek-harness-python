"""DeepSeek OpenAI-compatible streaming adapter."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..errors import ConfigurationError, LlmError
from ..models import ImageContent, Message, TextContent, ToolCallContent, ToolResultContent
from .types import LlmRequest, StreamChunk


class DeepSeekAdapter:
    """Adapter for DeepSeek's chat-completions-compatible endpoint."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = (
            base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
        ).rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def stream(self, request: LlmRequest) -> AsyncIterator[StreamChunk]:
        if not self.api_key:
            raise ConfigurationError("DEEPSEEK_API_KEY is required for the DeepSeek adapter")
        payload: dict[str, Any] = {
            "model": request.config.model,
            "messages": self._messages(request),
            "stream": True,
        }
        if request.system:
            payload["messages"].insert(0, {"role": "system", "content": request.system})
        if request.tools:
            payload["tools"] = [tool.to_openai() for tool in request.tools]
        for key, value in (
            ("max_tokens", request.config.max_tokens),
            ("temperature", request.config.temperature),
            ("top_p", request.config.top_p),
        ):
            if value is not None:
                payload[key] = value
        if request.config.thinking is not None:
            payload["thinking"] = {"type": request.config.thinking}
        if request.config.reasoning_effort not in {None, "off"}:
            payload["reasoning_effort"] = request.config.reasoning_effort

        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise LlmError(
                        f"DeepSeek request failed ({response.status_code}): {body[:1000]}"
                    )
                saw_done = False
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        saw_done = True
                        break
                    if not raw:
                        continue
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise LlmError("DeepSeek returned malformed SSE JSON") from exc
                    for chunk in self._chunks(data):
                        yield chunk
                if not saw_done:
                    yield StreamChunk(kind="done", finish_reason="stop")
        except httpx.HTTPError as exc:
            raise LlmError(f"DeepSeek transport failed: {exc}") from exc

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @staticmethod
    def _messages(request: LlmRequest) -> list[dict[str, Any]]:
        return [DeepSeekAdapter._message(message) for message in request.messages]

    @staticmethod
    def _message(message: Message) -> dict[str, Any]:
        if message.role == "tool":
            block = next(
                (item for item in message.content if isinstance(item, ToolResultContent)), None
            )
            return {
                "role": "tool",
                "tool_call_id": block.call_id if block else message.source.get("callId", ""),
                "content": message.text,
            }
        if message.role == "assistant":
            calls = [item for item in message.content if isinstance(item, ToolCallContent)]
            result: dict[str, Any] = {"role": "assistant", "content": message.text or None}
            if calls:
                result["tool_calls"] = [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                    for call in calls
                ]
            return result
        return {"role": message.role, "content": DeepSeekAdapter._content(message)}

    @staticmethod
    def _content(message: Message) -> str | list[dict[str, Any]]:
        images = [item for item in message.content if isinstance(item, ImageContent)]
        if not images:
            return message.text
        parts: list[dict[str, Any]] = []
        for block in message.content:
            if isinstance(block, TextContent):
                parts.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageContent):
                if block.data:
                    url = (
                        f"data:{block.attachment.get('mediaType', 'image/png')};base64,{block.data}"
                    )
                    parts.append({"type": "image_url", "image_url": {"url": url}})
                else:
                    parts.append(
                        {
                            "type": "text",
                            "text": (
                                f"[image attachment {block.attachment.get('attachmentId', '')}]"
                            ),
                        }
                    )
        return parts

    @staticmethod
    def _chunks(data: dict[str, Any]) -> list[StreamChunk]:
        chunks: list[StreamChunk] = []
        usage = data.get("usage")
        for choice in data.get("choices", []):
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if isinstance(text, str) and text:
                chunks.append(StreamChunk(kind="text", text=text))
            reasoning = delta.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                chunks.append(StreamChunk(kind="reasoning", text=reasoning))
            for index, call in enumerate(delta.get("tool_calls") or []):
                function = call.get("function") or {}
                chunks.append(
                    StreamChunk(
                        kind="tool-call-delta",
                        index=int(call.get("index", index)),
                        call_id=call.get("id"),
                        name=function.get("name"),
                        arguments=function.get("arguments", ""),
                    )
                )
            finish = choice.get("finish_reason")
            if finish is not None:
                chunks.append(StreamChunk(kind="done", finish_reason=str(finish), usage=usage))
        return chunks
