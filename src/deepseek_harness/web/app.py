"""FastAPI application exposing the DSH-compatible HTTP and event carriers."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from ..llm.adapter import LlmAdapter
from ..tools import PermissionMode
from .service import ApiFault, HarnessService

AdapterFactory = Callable[[str], LlmAdapter]


def create_app(
    *,
    session_root: str | os.PathLike[str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    model: str = "deepseek-v4-flash",
    permission_mode: PermissionMode = PermissionMode.WORKSPACE_WRITE,
    service: HarnessService | None = None,
    web_dist: str | os.PathLike[str] | None = None,
    adapter_factory: AdapterFactory | None = None,
) -> FastAPI:
    runtime = service or HarnessService(
        session_root or os.getenv("DSH_SESSION_ROOT", "~/.deepseek_harness_python/sessions"),
        cwd=cwd,
        model=model,
        permission_mode=permission_mode,
        adapter_factory=adapter_factory,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        await runtime.dispose()

    app = FastAPI(title="DeepSeek Harness Python", version="0.1.0.dev0", lifespan=lifespan)
    app.state.harness = runtime

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "product": "DeepSeek Harness Python"}

    @app.get("/api/events.mux")
    async def mux_events() -> StreamingResponse:
        return _sse_response(runtime.stream("mux"))

    @app.get("/api/events.host")
    async def host_events() -> StreamingResponse:
        return _sse_response(runtime.stream("host"))

    @app.websocket("/api/events.mux")
    async def mux_websocket(socket: WebSocket) -> None:
        await _websocket_stream(socket, runtime.stream("mux"))

    @app.websocket("/api/events.host")
    async def host_websocket(socket: WebSocket) -> None:
        await _websocket_stream(socket, runtime.stream("host"))

    @app.api_route("/api/session.export", methods=["GET", "HEAD"], include_in_schema=False)
    async def session_export(request: Request) -> Response:
        session_id = request.query_params.get("sessionId")
        include_descendants = request.query_params.get("includeDescendants")
        if not session_id:
            return PlainTextResponse("sessionId is required", status_code=400)
        if include_descendants not in {None, "true", "false"}:
            return PlainTextResponse(
                "includeDescendants must be true or false",
                status_code=400,
            )
        try:
            data = await runtime.export_zip(
                session_id,
                include_descendants=include_descendants == "true",
            )
        except ApiFault as exc:
            status = 404 if exc.code == "session-not-found" else 500
            return PlainTextResponse(str(exc), status_code=status)
        headers = {
            "Content-Disposition": f'attachment; filename="{runtime.export_filename(session_id)}"'
        }
        if request.method == "HEAD":
            return Response(status_code=200, media_type="application/zip", headers=headers)
        return Response(content=data, media_type="application/zip", headers=headers)

    @app.post("/api/{method:path}", response_model=None)
    async def unary(request: Request, method: str) -> JSONResponse | PlainTextResponse:
        media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if media_type != "application/json":
            return PlainTextResponse("content type must be application/json", status_code=415)
        try:
            body = await request.json()
        except ValueError:
            return PlainTextResponse("body is not JSON", status_code=400)
        if not isinstance(body, dict):
            return _error_response(
                "invalid-request", "invalid client-request message", "bad-request"
            )
        if method == "respond":
            if body.get("type") != "client-response" or not isinstance(body.get("rpcId"), str):
                return JSONResponse({"accepted": False, "reason": "bad-response"})
            result = body.get("result")
            if not isinstance(result, dict):
                return JSONResponse({"accepted": False, "reason": "bad-response"})
            return JSONResponse(await runtime.respond(body))
        rpc_id = body.get("rpcId")
        if not isinstance(rpc_id, str):
            rpc_id = str(uuid.uuid4())
        if body.get("type") != "client-request" or body.get("method") != method:
            return _error_response(rpc_id, "invalid client-request message", "bad-request")
        payload = body.get("payload")
        if not isinstance(payload, dict):
            return _error_response(rpc_id, "request payload must be an object", "bad-request")
        try:
            value = await runtime.dispatch(method, payload)
        except ApiFault as exc:
            return _error_response(rpc_id, str(exc), exc.code, exc.details)
        except Exception as exc:
            return _error_response(rpc_id, str(exc), "internal")
        return JSONResponse(
            {
                "type": "server-response",
                "rpcId": rpc_id,
                "result": {"ok": True, "value": value},
            }
        )

    frontend_value = web_dist or os.getenv("DSH_WEB_DIST")
    frontend = Path(frontend_value).expanduser().resolve() if frontend_value else None
    if frontend is not None and frontend.is_dir():

        @app.get("/", include_in_schema=False)
        async def frontend_index() -> HTMLResponse:
            return HTMLResponse(_render_frontend_index(frontend))

        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    else:

        @app.get("/", include_in_schema=False)
        async def missing_frontend() -> PlainTextResponse:
            return PlainTextResponse(
                "DeepSeek Harness Python API is running. "
                "Build the DSH frontend and set DSH_WEB_DIST."
            )

    return app


def _error_response(
    rpc_id: str,
    message: str,
    code: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    if details is None:
        details = {"issues": []} if code == "bad-request" else {}
    return JSONResponse(
        {
            "type": "server-response",
            "rpcId": rpc_id,
            "result": {
                "ok": False,
                "error": {"code": code, "message": message, "details": details},
            },
        }
    )


async def _websocket_stream(
    socket: WebSocket,
    frames: AsyncIterator[dict[str, Any]],
) -> None:
    """Send DSH server-request envelopes over a browser WebSocket downlink."""

    await socket.accept()
    try:
        async for frame in frames:
            await socket.send_json(
                {
                    "type": "server-request",
                    "rpcId": str(uuid.uuid4()),
                    "method": frame.get("type", "stream/error"),
                    "payload": frame,
                }
            )
    except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
        # A cancelled browser connection can close between two sends.  The
        # service generator's finally block still removes its subscriber.
        return


def _sse_response(frames: AsyncIterator[dict[str, Any]]) -> StreamingResponse:
    async def body() -> AsyncIterator[str]:
        yield ": connected\n\n"
        async for frame in frames:
            full = {
                "type": "server-request",
                "rpcId": str(uuid.uuid4()),
                "method": frame.get("type", "stream/error"),
                "payload": frame,
            }
            yield f"data: {json.dumps(full, ensure_ascii=False, separators=(',', ':'))}\n\n"

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _render_frontend_index(frontend: Path) -> str:
    index = (frontend / "index.html").read_text(encoding="utf-8")
    graph_path = frontend / "boot.json"
    graph: Any = {"rev": "python", "entries": []}
    if graph_path.is_file():
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
    manifest = json.dumps(graph, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    script = f"<script>window.__DSH_BOOT__ = {manifest}</script>"
    if "<head>" in index:
        return index.replace("<head>", f"<head>{script}", 1)
    return script + index


app = create_app()

__all__ = ["app", "create_app"]
