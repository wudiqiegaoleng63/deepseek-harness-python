from __future__ import annotations

import asyncio

from deepseek_harness.tools import PermissionMode, WorkspacePolicy, install_builtin_tools
from deepseek_harness.tools.registry import ToolContext, ToolRegistry


def test_builtin_file_tools_are_workspace_bounded(tmp_path) -> None:
    async def scenario() -> None:
        registry = ToolRegistry()
        policy = WorkspacePolicy(tmp_path, PermissionMode.WORKSPACE_WRITE)
        disposers = install_builtin_tools(registry, policy)
        context = ToolContext("session-test", str(tmp_path))
        try:
            write = await registry.execute(
                "write_file", '{"path":"a.txt","content":"hello"}', context
            )
            assert not write.is_error
            read = await registry.execute("read_file", '{"path":"a.txt"}', context)
            assert read.text == "hello"
            escaped = await registry.execute("read_file", '{"path":"../outside.txt"}', context)
            assert escaped.is_error
        finally:
            for dispose in reversed(disposers):
                dispose()

    asyncio.run(scenario())


def test_read_only_policy_rejects_writes(tmp_path) -> None:
    async def scenario() -> None:
        registry = ToolRegistry()
        disposers = install_builtin_tools(
            registry,
            WorkspacePolicy(tmp_path, PermissionMode.READ_ONLY),
        )
        try:
            result = await registry.execute(
                "write_file",
                '{"path":"a.txt","content":"blocked"}',
                ToolContext("session-test", str(tmp_path)),
            )
            assert result.is_error
            assert not (tmp_path / "a.txt").exists()
        finally:
            for dispose in reversed(disposers):
                dispose()

    asyncio.run(scenario())


def test_danger_full_access_listing_can_render_paths_outside_workspace(tmp_path) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        registry = ToolRegistry()
        disposers = install_builtin_tools(
            registry,
            WorkspacePolicy(workspace, PermissionMode.DANGER_FULL_ACCESS),
        )
        try:
            result = await registry.execute(
                "list_files",
                f'{{"path":"{tmp_path}"}}',
                ToolContext("session-test", str(workspace)),
            )
            assert not result.is_error
            assert "outside.txt" in result.text
        finally:
            for dispose in reversed(disposers):
                dispose()

    asyncio.run(scenario())
