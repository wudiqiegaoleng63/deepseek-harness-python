from __future__ import annotations

import asyncio
from math import inf

from deepseek_harness.tools import PermissionMode, WorkspacePolicy, install_builtin_tools
from deepseek_harness.tools.registry import ToolContext, ToolDefinition, ToolRegistry, ToolResult


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


def test_canonical_filesystem_tools_cover_read_edit_glob_grep_and_editor(tmp_path) -> None:
    async def scenario() -> None:
        source = tmp_path / "src"
        source.mkdir()
        (source / "one.py").write_text("alpha\nbeta\nalpha\n", encoding="utf-8")
        (source / "two.txt").write_text("gamma\n", encoding="utf-8")
        registry = ToolRegistry()
        disposers = install_builtin_tools(
            registry,
            WorkspacePolicy(tmp_path, PermissionMode.WORKSPACE_WRITE),
        )
        context = ToolContext("session-test", str(tmp_path))
        try:
            read = await registry.execute(
                "read", '{"file_path":"src/one.py","offset":2,"limit":1}', context
            )
            assert not read.is_error
            assert "2: beta" in read.text

            edited = await registry.execute(
                "edit",
                '{"file_path":"src/one.py","old_string":"beta","new_string":"delta"}',
                context,
            )
            assert not edited.is_error
            assert "delta" in (source / "one.py").read_text(encoding="utf-8")

            glob = await registry.execute("glob", '{"pattern":"*.py","path":"src"}', context)
            assert glob.text == "one.py"
            grep = await registry.execute(
                "grep", '{"pattern":"alpha","path":"src","include":"*.py"}', context
            )
            assert "one.py:1:alpha" in grep.text

            created = await registry.execute(
                "str_replace_editor",
                '{"command":"create","path":"src/new.txt","file_text":"created"}',
                context,
            )
            assert not created.is_error
            viewed = await registry.execute(
                "str_replace_editor",
                '{"command":"view","path":"src/new.txt"}',
                context,
            )
            assert "1: created" in viewed.text
        finally:
            for dispose in reversed(disposers):
                dispose()

    asyncio.run(scenario())


def test_tool_registry_enforces_cancellable_definition_timeout() -> None:
    async def slow(_args, _context):
        await asyncio.sleep(1)
        return ToolResult("late")

    with_timeout = ToolRegistry()
    with_timeout.register(
        ToolDefinition(
            "slow",
            "A deliberately slow test tool.",
            {"type": "object", "additionalProperties": False},
            slow,
            timeout_seconds=0.001,
        )
    )

    async def scenario() -> None:
        result = await with_timeout.execute("slow", "{}", ToolContext("s", "."))
        assert result.is_error
        assert result.meta == {"code": "TOOL_TIMEOUT"}
        assert "timed out" in result.text

    asyncio.run(scenario())
    try:
        ToolDefinition(
            "invalid",
            "invalid",
            {"type": "object"},
            slow,
            timeout_seconds=inf,
        )
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("non-finite tool timeouts must be rejected")
