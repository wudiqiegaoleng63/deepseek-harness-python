from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx

from deepseek_harness.llm.types import LlmRequest, StreamChunk
from deepseek_harness.web import HarnessService, create_app


class ScriptedAdapter:
    def stream(self, request: LlmRequest) -> AsyncIterator[StreamChunk]:
        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(kind="text", text="hello from python")
            yield StreamChunk(kind="done", finish_reason="stop")

        return chunks()

    async def aclose(self) -> None:
        return None


def test_fastapi_rpc_and_session_lifecycle(tmp_path) -> None:
    async def scenario() -> None:
        service = HarnessService(
            tmp_path / "sessions",
            cwd=tmp_path,
            adapter_factory=lambda _model: ScriptedAdapter(),
        )
        app = create_app(service=service)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            health = await client.get("/health")
            assert health.json()["product"] == "DeepSeek Harness Python"

            created = await client.post(
                "/api/session.create",
                json={
                    "type": "client-request",
                    "rpcId": "rpc-create",
                    "method": "session.create",
                    "payload": {"cwd": str(tmp_path), "sessionId": "session-web"},
                },
            )
            create_body = created.json()
            assert create_body["result"]["ok"] is True
            assert create_body["result"]["value"] == {"sessionId": "session-web"}

            prompted = await client.post(
                "/api/session.prompt",
                json={
                    "type": "client-request",
                    "rpcId": "rpc-prompt",
                    "method": "session.prompt",
                    "payload": {
                        "sessionId": "session-web",
                        "mode": "queue",
                        "content": [{"type": "text", "text": "hello"}],
                    },
                },
            )
            assert prompted.json()["result"]["value"] == {"accepted": True}

            handle = await service.get_session("session-web")
            task = handle.task
            assert task is not None
            await task
            history = await client.post(
                "/api/session.history",
                json={
                    "type": "client-request",
                    "rpcId": "rpc-history",
                    "method": "session.history",
                    "payload": {"sessionId": "session-web"},
                },
            )
            events = history.json()["result"]["value"]["events"]
            assert any(item["event"]["type"] == "assistant/message" for item in events)

        await service.dispose()

    asyncio.run(scenario())


def test_fastapi_serves_python_branded_frontend_and_boot_graph(tmp_path) -> None:
    async def scenario() -> None:
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text(
            "<html><head><title>DeepSeek Harness Python</title></head><body></body></html>",
            encoding="utf-8",
        )
        (dist / "boot.json").write_text(
            '{"rev":"test","entries":[{"id":"plugin","url":"/plugins/plugin/client.js","rev":"x"}]}',
            encoding="utf-8",
        )
        plugin = dist / "plugins" / "plugin"
        plugin.mkdir(parents=True)
        (plugin / "client.js").write_text("window.testPlugin = true", encoding="utf-8")

        service = HarnessService(tmp_path / "sessions", cwd=tmp_path)
        app = create_app(service=service, web_dist=dist)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            index = await client.get("/")
            assert "window.__DSH_BOOT__" in index.text
            assert '"rev":"test"' in index.text
            bundle = await client.get("/plugins/plugin/client.js")
            assert bundle.text == "window.testPlugin = true"
        await service.dispose()

    asyncio.run(scenario())
