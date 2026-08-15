from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from io import BytesIO
from zipfile import ZipFile

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


def test_session_export_download_contains_root_and_descendants(tmp_path) -> None:
    async def scenario() -> None:
        service = HarnessService(
            tmp_path / "sessions",
            cwd=tmp_path,
            adapter_factory=lambda _model: ScriptedAdapter(),
        )
        app = create_app(service=service)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            await service.dispatch(
                "session.create",
                {"sessionId": "export-root", "cwd": str(tmp_path)},
            )
            await service.dispatch(
                "session.prompt",
                {
                    "sessionId": "export-root",
                    "content": [{"type": "text", "text": "export me"}],
                },
            )
            root = await service.get_session("export-root")
            assert root.task is not None
            await root.task
            await service.dispatch("session.fork", {"sessionId": "export-root"})

            head = await client.head(
                "/api/session.export",
                params={"sessionId": "export-root", "includeDescendants": "true"},
            )
            assert head.status_code == 200
            assert head.headers["content-type"] == "application/zip"
            assert "dsh-session-export-root.zip" in head.headers["content-disposition"]

            response = await client.get(
                "/api/session.export",
                params={"sessionId": "export-root", "includeDescendants": "true"},
            )
            assert response.status_code == 200
            with ZipFile(BytesIO(response.content)) as archive:
                names = set(archive.namelist())
                assert "session.jsonl" in names
                assert any(name.startswith("subagents/") for name in names)
                assert b"export me" in archive.read("session.jsonl")

            missing = await client.get(
                "/api/session.export", params={"sessionId": "does-not-exist"}
            )
            assert missing.status_code == 404
        await service.dispose()

    asyncio.run(scenario())


def test_respond_resolves_approval_and_question_requests(tmp_path) -> None:
    async def scenario() -> None:
        service = HarnessService(tmp_path / "sessions", cwd=tmp_path)
        await service.dispatch("session.create", {"sessionId": "interactive", "cwd": str(tmp_path)})
        app = create_app(service=service)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            approval_task = asyncio.create_task(
                service.request_approval(
                    "interactive",
                    "write_file",
                    approval_id="approval-1",
                    reason="the model wants to modify a file",
                )
            )
            while not service._pending_approvals:
                await asyncio.sleep(0)
            approval_rpc = next(iter(service._pending_approvals))
            approval_response = await client.post(
                "/api/respond",
                json={
                    "type": "client-response",
                    "rpcId": approval_rpc,
                    "result": {
                        "ok": True,
                        "value": {
                            "sessionId": "interactive",
                            "approvalId": "approval-1",
                            "outcome": "allowed-once",
                        },
                    },
                },
            )
            assert approval_response.json() == {"accepted": True}
            assert await approval_task == "allowed-once"

            question_task = asyncio.create_task(
                service.request_question(
                    "interactive",
                    [
                        {
                            "id": "color",
                            "question": "Which color?",
                            "options": [{"label": "blue"}, {"label": "green"}],
                        }
                    ],
                )
            )
            while not service._pending_questions:
                await asyncio.sleep(0)
            question_rpc = next(iter(service._pending_questions))
            question_response = await client.post(
                "/api/respond",
                json={
                    "type": "client-response",
                    "rpcId": question_rpc,
                    "result": {
                        "ok": True,
                        "value": {
                            "sessionId": "interactive",
                            "answer": {"answers": [{"id": "color", "selected": ["blue"]}]},
                        },
                    },
                },
            )
            assert question_response.json() == {"accepted": True}
            assert await question_task == {"answers": [{"id": "color", "selected": ["blue"]}]}
        await service.dispose()

    asyncio.run(scenario())
