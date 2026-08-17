from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from deepseek_harness.errors import LlmError
from deepseek_harness.llm import DeepSeekAdapter
from deepseek_harness.llm.types import LlmCallConfig, LlmRequest, ToolSchema
from deepseek_harness.models import create_user_message


def test_deepseek_adapter_parses_openai_compatible_sse() -> None:
    async def scenario() -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            chunks = [
                {"choices": [{"delta": {"content": "hello"}}]},
                {"choices": [{"delta": {"reasoning_content": "thinking"}}]},
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-1",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": '{"path":"',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": '"README.md"}'},
                                    }
                                ]
                            }
                        }
                    ]
                },
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            ]
            body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            body += "data: [DONE]\n\n"
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=body.encode(),
                request=request,
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        adapter = DeepSeekAdapter(api_key="test-key", client=client)
        request = LlmRequest(
            messages=(create_user_message("inspect"),),
            config=LlmCallConfig(model="deepseek-chat"),
            system="You are a coding agent.",
            tools=(
                ToolSchema(
                    "read_file",
                    "Read a file.",
                    {"type": "object", "properties": {"path": {"type": "string"}}},
                ),
            ),
        )

        chunks = [chunk async for chunk in adapter.stream(request)]
        assert [chunk.kind for chunk in chunks] == [
            "text",
            "reasoning",
            "tool-call-delta",
            "tool-call-delta",
            "done",
        ]
        assert chunks[0].text == "hello"
        assert chunks[1].text == "thinking"
        assert chunks[2].name == "read_file"
        assert chunks[2].arguments == '{"path":"'
        assert chunks[3].arguments == '"README.md"}'
        assert chunks[4].finish_reason == "tool_calls"
        messages = captured["messages"]
        assert isinstance(messages, list)
        assert messages[0] == {"role": "system", "content": "You are a coding agent."}
        assert messages[1]["role"] == "user"
        assert captured["stream"] is True
        tools = captured["tools"]
        assert isinstance(tools, list)
        assert tools[0]["function"]["name"] == "read_file"
        await adapter.aclose()

    asyncio.run(scenario())


def test_deepseek_adapter_classifies_rate_limit_and_retry_after() -> None:
    async def scenario() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                headers={"retry-after": "1.5", "x-request-id": "req-1"},
                json={"error": {"message": "slow down", "type": "rate_limit"}},
                request=request,
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        adapter = DeepSeekAdapter(api_key="test-key", client=client)
        request = LlmRequest(
            messages=(create_user_message("hello"),),
            config=LlmCallConfig(model="deepseek-chat"),
        )
        with pytest.raises(LlmError) as caught:
            _ = [chunk async for chunk in adapter.stream(request)]
        assert caught.value.code == "RATE_LIMIT"
        assert caught.value.status == 429
        assert caught.value.retry_after_ms == 1500
        assert caught.value.request_id == "req-1"
        await adapter.aclose()

    asyncio.run(scenario())


def test_deepseek_adapter_disables_thinking_for_session_title_requests() -> None:
    async def scenario() -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            body = 'data: {"choices":[{"delta":{"content":"title"}}]}\n\n'
            body += 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            body += "data: [DONE]\n\n"
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=body.encode(),
                request=request,
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        adapter = DeepSeekAdapter(api_key="test-key", client=client)
        request = LlmRequest(
            messages=(create_user_message("title this"),),
            config=LlmCallConfig(
                model="deepseek-v4-flash",
                thinking="enabled",
                reasoning_effort="max",
            ),
            purpose="session-title",
        )
        _ = [chunk async for chunk in adapter.stream(request)]
        assert captured["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in captured
        await adapter.aclose()

    asyncio.run(scenario())
