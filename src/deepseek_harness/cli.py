"""Command-line entry points for the native Python runtime."""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

import typer

from .agent import Agent
from .errors import HarnessError
from .llm import DeepSeekAdapter, LlmCallConfig
from .session import JsonlSessionStore
from .tools import PermissionMode, WorkspacePolicy, install_builtin_tools
from .tools.registry import ToolRegistry

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command("headless")
def headless(
    task: str = typer.Argument(..., help="One task to send to the agent."),
    model: str = typer.Option("deepseek-chat", "--model", help="DeepSeek model id."),
    cwd: Path = typer.Option(Path.cwd, "--cwd", help="Workspace directory."),
    session_root: Path | None = typer.Option(
        None, "--session-root", help="JSONL session directory."
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


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(3080, "--port", min=1, max=65535, help="Bind port."),
    cwd: Path = typer.Option(Path.cwd, "--cwd", help="Default workspace directory."),
    session_root: Path | None = typer.Option(
        None, "--session-root", help="JSONL session directory."
    ),
    web_dist: Path | None = typer.Option(
        None, "--web-dist", help="Built frontend directory to serve."
    ),
    model: str = typer.Option("deepseek-chat", "--model", help="Default DeepSeek model."),
    permission_mode: PermissionMode = typer.Option(
        PermissionMode.WORKSPACE_WRITE,
        "--permission-mode",
        help="Workspace policy for local tools.",
    ),
) -> None:
    """Serve the web UI API and live event streams."""

    import uvicorn

    from .web import create_app

    uvicorn.run(
        create_app(
            session_root=session_root,
            cwd=cwd,
            model=model,
            permission_mode=permission_mode,
            web_dist=web_dist,
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
    max_tokens: int | None,
    permission_mode: PermissionMode,
):
    workspace = cwd.expanduser().resolve()
    if not workspace.is_dir():
        raise HarnessError(f"workspace directory does not exist: {workspace}")
    root = session_root or Path(
        os.getenv("DSH_SESSION_ROOT", "~/.deepseek_harness_python/sessions")
    )
    store = JsonlSessionStore(root)
    session = await store.create(f"session-{uuid.uuid4().hex}", cwd=str(workspace))
    adapter = DeepSeekAdapter()
    registry = ToolRegistry()
    policy = WorkspacePolicy(workspace, permission_mode)
    disposers = install_builtin_tools(registry, policy)
    agent = Agent(
        session,
        adapter,
        tools=registry,
        config=LlmCallConfig(model=model, max_tokens=max_tokens),
        system_prompt=(
            "You are a coding agent powered by the DeepSeek model. "
            f"Your working directory is {workspace}."
        ),
    )
    try:
        result = await agent.run(task)
        await store.save(session)
        return result
    finally:
        for dispose in reversed(disposers):
            dispose()
        await agent.dispose()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
