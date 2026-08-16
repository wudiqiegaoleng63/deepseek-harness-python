"""Command-line entry points for the native Python runtime."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Callable
from pathlib import Path

import typer

from .errors import HarnessError
from .llm import DeepSeekAdapter, LlmAdapter
from .tools import PermissionMode
from .web import HarnessService

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command("headless")
def headless(
    task: str = typer.Argument(..., help="One task to send to the agent."),
    model: str = typer.Option("deepseek-v4-flash", "--model", help="DeepSeek model id."),
    cwd: Path = typer.Option(Path.cwd(), "--cwd", help="Workspace directory."),
    session_root: Path | None = typer.Option(
        None, "--session-root", help="JSONL session directory."
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        envvar="DEEPSEEK_BASE_URL",
        help="DeepSeek-compatible API base URL.",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        envvar="DEEPSEEK_API_KEY",
        help="API key; defaults to DEEPSEEK_API_KEY.",
    ),
    request_timeout_seconds: float = typer.Option(
        120.0,
        "--request-timeout",
        min=1.0,
        help="Provider request timeout in seconds.",
    ),
    max_tokens: int | None = typer.Option(None, "--max-tokens", min=1),
    permission_mode: PermissionMode = typer.Option(
        PermissionMode.WORKSPACE_WRITE,
        "--permission-mode",
        help="Workspace policy for local tools.",
    ),
) -> None:
    """Run one task and print the final assistant response."""

    try:
        result = asyncio.run(
            _run_headless(
                task,
                model=model,
                cwd=cwd,
                session_root=session_root,
                base_url=base_url,
                api_key=api_key,
                request_timeout_seconds=request_timeout_seconds,
                max_tokens=max_tokens,
                permission_mode=permission_mode,
            )
        )
    except HarnessError as exc:
        typer.echo(f"dsh-python: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(result.final_response)
    if result.finish_reason not in {"completed", "max-tokens"}:
        raise typer.Exit(code=1)


@app.command("version")
def version() -> None:
    """Print the Python runtime version."""

    typer.echo("DeepSeek Harness Python 0.1.0.dev0")


@app.command("sdk-server")
def sdk_server(
    session_root: Path | None = typer.Option(
        None, "--session-root", help="JSONL session directory."
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        envvar="DEEPSEEK_BASE_URL",
        help="DeepSeek-compatible API base URL.",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        envvar="DEEPSEEK_API_KEY",
        help="API key; defaults to DEEPSEEK_API_KEY.",
    ),
    request_timeout_seconds: float = typer.Option(
        120.0,
        "--request-timeout",
        min=1.0,
        help="Provider request timeout in seconds.",
    ),
) -> None:
    """Run the newline-delimited JSON-RPC SDK runtime on stdio."""

    from .sdk_rpc import run_sdk_server

    try:
        asyncio.run(
            run_sdk_server(
                session_root=session_root,
                api_key=api_key,
                base_url=base_url,
                timeout=request_timeout_seconds,
            )
        )
    except KeyboardInterrupt as exc:
        raise typer.Exit(code=130) from exc


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(3080, "--port", min=1, max=65535, help="Bind port."),
    cwd: Path = typer.Option(Path.cwd(), "--cwd", help="Default workspace directory."),
    session_root: Path | None = typer.Option(
        None, "--session-root", help="JSONL session directory."
    ),
    web_dist: Path | None = typer.Option(
        None, "--web-dist", help="Built frontend directory to serve."
    ),
    model: str = typer.Option("deepseek-v4-flash", "--model", help="Default DeepSeek model."),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        envvar="DEEPSEEK_BASE_URL",
        help="DeepSeek-compatible API base URL.",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        envvar="DEEPSEEK_API_KEY",
        help="API key; defaults to DEEPSEEK_API_KEY.",
    ),
    request_timeout_seconds: float = typer.Option(
        120.0,
        "--request-timeout",
        min=1.0,
        help="Provider request timeout in seconds.",
    ),
    permission_mode: PermissionMode = typer.Option(
        PermissionMode.WORKSPACE_WRITE,
        "--permission-mode",
        help="Workspace policy for local tools.",
    ),
) -> None:
    """Serve the web UI API and live event streams."""

    import uvicorn

    from .web import create_app

    def configured_adapter(_selected_model: str) -> LlmAdapter:
        return DeepSeekAdapter(
            api_key=api_key,
            base_url=base_url,
            timeout=request_timeout_seconds,
        )

    uvicorn.run(
        create_app(
            session_root=session_root,
            cwd=cwd,
            model=model,
            permission_mode=permission_mode,
            web_dist=web_dist,
            adapter_factory=configured_adapter,
        ),
        host=host,
        port=port,
    )


async def _run_headless(
    task: str,
    *,
    model: str,
    cwd: Path,
    session_root: Path | None,
    base_url: str | None = None,
    api_key: str | None = None,
    request_timeout_seconds: float = 120.0,
    max_tokens: int | None,
    permission_mode: PermissionMode,
    adapter_factory: Callable[[str], LlmAdapter] | None = None,
):
    workspace = cwd.expanduser().resolve()
    if not workspace.is_dir():
        raise HarnessError(f"workspace directory does not exist: {workspace}")
    if request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be positive")

    if adapter_factory is None:
        def configured_adapter(_selected_model: str) -> LlmAdapter:
            return DeepSeekAdapter(
                api_key=api_key,
                base_url=base_url,
                timeout=request_timeout_seconds,
            )

        adapter_factory = configured_adapter
    service = HarnessService(
        session_root
        or Path(os.getenv("DSH_SESSION_ROOT", "~/.deepseek_harness_python/sessions")),
        cwd=workspace,
        model=model,
        permission_mode=permission_mode,
        adapter_factory=adapter_factory,
    )
    try:
        if max_tokens is not None:
            await service.settings.update("llm-deepseek", {"maxTokens": max_tokens})
        handle = await service.create_session(
            session_id=f"session-{uuid.uuid4().hex}",
            cwd=str(workspace),
        )
        result = await handle.agent.run(task)
        await service.store.save(handle.session)
        return result
    finally:
        await service.dispose()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
