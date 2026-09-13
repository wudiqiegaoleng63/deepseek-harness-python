from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import cast

from deepseek_harness.llm.adapter import LlmAdapter
from deepseek_harness.llm.types import LlmRequest, StreamChunk
from deepseek_harness.tools.registry import ToolContext
from deepseek_harness.web import HarnessService


class RepeatingAdapter:
    def __init__(self, text: str = "child answer") -> None:
        self.text = text

    async def stream(self, _request: LlmRequest) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(kind="text", text=self.text)
        yield StreamChunk(kind="done", finish_reason="stop")

    async def aclose(self) -> None:
        return None


def test_subagent_foreground_and_fork_preserve_durable_child_identity(tmp_path) -> None:
    async def scenario() -> None:
        service = HarnessService(
            tmp_path / "sessions",
            cwd=tmp_path,
            adapter_factory=lambda _model: cast(LlmAdapter, RepeatingAdapter()),
        )
        await service.dispatch("session.create", {"sessionId": "parent", "cwd": str(tmp_path)})
        registry = service._tool_registries["parent"]
        context = ToolContext("parent", str(tmp_path))

        foreground = await registry.execute(
            "subagent",
            json.dumps(
                {
                    "description": "answer directly",
                    "prompt": "Return a concise answer.",
                    "run_in_background": False,
                }
            ),
            context,
        )
        assert not foreground.is_error
        assert "child answer" in foreground.text
        assert foreground.meta is not None
        foreground_id = str(foreground.meta["subagentId"])
        foreground_session = await service.get_session(foreground_id)
        assert foreground_session.session.header.origin == "subagent"
        assert any(
            event.type == "subagent/descriptor" for event in foreground_session.session.events
        )

        await service.dispatch(
            "session.prompt",
            {
                "sessionId": "parent",
                "mode": "queue",
                "content": [{"type": "text", "text": "establish fork context"}],
            },
        )
        parent = await service.get_session("parent")
        assert parent.task is not None
        await parent.task

        forked = await registry.execute(
            "subagent_fork",
            json.dumps(
                {
                    "description": "review this conversation",
                    "prompt": "Summarize the inherited context.",
                    "run_in_background": False,
                }
            ),
            context,
        )
        assert not forked.is_error
        assert forked.meta is not None
        fork_id = str(forked.meta["subagentId"])
        fork_session = await service.get_session(fork_id)
        assert (fork_session.session.header.seed_length or 0) > 0
        assert any(event.type == "user/message" for event in fork_session.session.events)
        await service.dispose()

    asyncio.run(scenario())


def test_subagent_background_is_continuable_and_visible_to_control_tools(tmp_path) -> None:
    async def scenario() -> None:
        service = HarnessService(
            tmp_path / "sessions",
            cwd=tmp_path,
            adapter_factory=lambda _model: cast(LlmAdapter, RepeatingAdapter("background answer")),
        )
        await service.dispatch("session.create", {"sessionId": "parent", "cwd": str(tmp_path)})
        registry = service._tool_registries["parent"]
        context = ToolContext("parent", str(tmp_path))

        started = await registry.execute(
            "subagent",
            json.dumps({"description": "background work", "prompt": "Work independently."}),
            context,
        )
        assert not started.is_error
        assert started.meta is not None
        child_id = str(started.meta["subagentId"])
        job_id = str(started.meta["jobId"])
        terminal = await service.jobs.wait(job_id, 2_000, "parent")
        assert terminal.status == "completed"
        output = service.jobs.read(job_id, "parent")
        assert "background answer" in output.text

        listed = await registry.execute("list_agents", "{}", context)
        assert not listed.is_error
        assert child_id in listed.text
        assert "background work" in listed.text

        sent = await registry.execute(
            "send_message",
            json.dumps({"subagent_id": child_id, "message": "Now provide a follow-up."}),
            context,
        )
        assert not sent.is_error
        child = await service.get_session(child_id)
        assert child.task is not None
        await child.task
        history = await service.history(child_id)
        assert any(
            event["event"]["type"] == "user/message"
            and event["event"]["data"]["message"]["content"][0]["text"]
            == "Now provide a follow-up."
            for event in history["events"]
        )
        await service.dispose()

    asyncio.run(scenario())


def acp_service(tmp_path, extra_env=None):
    """A harness whose `subagent` tool can delegate to the fixture ACP child."""

    import os
    import sys
    from pathlib import Path

    from deepseek_harness.acp_client import AcpSubagentConfig

    env = {name: value for name, value in os.environ.items() if name.startswith("MOCK_")}
    env.update(extra_env or {})
    return HarnessService(
        tmp_path / "sessions",
        cwd=tmp_path,
        adapter_factory=lambda _model: cast(LlmAdapter, RepeatingAdapter()),
        acp_subagent=AcpSubagentConfig(
            command=sys.executable,
            args=(str(Path(__file__).parent / "acp_child_fixture.py"),),
            env=env,
            dispose_eof_grace_ms=2_000,
        ),
    )


def test_acp_subagent_delegates_foreground_through_the_tool(tmp_path) -> None:
    async def scenario() -> None:
        service = acp_service(tmp_path, {"MOCK_TEXT": "remote child answer"})
        await service.dispatch("session.create", {"sessionId": "parent", "cwd": str(tmp_path)})
        registry = service._tool_registries["parent"]
        result = await registry.execute(
            "subagent",
            json.dumps(
                {
                    "description": "ask the remote agent",
                    "prompt": "Answer directly.",
                    "agent": "acp",
                }
            ),
            ToolContext("parent", str(tmp_path)),
        )
        assert not result.is_error
        assert "remote child answer" in result.text
        assert result.meta is not None
        assert result.meta["provider"] == "acp"
        assert result.meta["finishReason"] == "completed"
        assert result.meta["subagentId"]
        # The out-of-process child owns no local session.
        assert not list((tmp_path / "sessions").glob(f"**/{result.meta['subagentId']}*"))
        await service.dispose()

    asyncio.run(scenario())


def test_acp_subagent_rejects_unknown_provider_fork_and_background(tmp_path) -> None:
    async def scenario() -> None:
        service = acp_service(tmp_path)
        await service.dispatch("session.create", {"sessionId": "parent", "cwd": str(tmp_path)})
        registry = service._tool_registries["parent"]
        context = ToolContext("parent", str(tmp_path))

        unknown = await registry.execute(
            "subagent",
            json.dumps({"description": "x", "prompt": "y", "agent": "spawn"}),
            context,
        )
        assert unknown.is_error
        assert "unknown subagent provider" in unknown.text

        forked = await registry.execute(
            "subagent_fork",
            json.dumps({"description": "x", "prompt": "y", "agent": "acp"}),
            context,
        )
        assert forked.is_error
        assert "inherits no parent context" in forked.text

        # The out-of-process provider is one-shot: a background call names a
        # fresh remote session that no follow-up tool could address, so it is
        # rejected rather than silently downgraded.
        background = await registry.execute(
            "subagent",
            json.dumps(
                {"description": "x", "prompt": "y", "agent": "acp", "run_in_background": True}
            ),
            context,
        )
        assert background.is_error
        assert "foreground" in background.text
        await service.dispose()

    asyncio.run(scenario())


def test_acp_subagent_maps_a_failed_child_to_an_error_result(tmp_path) -> None:
    async def scenario() -> None:
        service = acp_service(tmp_path, {"MOCK_STOP": "max_turn_requests", "MOCK_TEXT": "partial"})
        await service.dispatch("session.create", {"sessionId": "parent", "cwd": str(tmp_path)})
        registry = service._tool_registries["parent"]
        result = await registry.execute(
            "subagent",
            json.dumps({"description": "x", "prompt": "y", "agent": "acp"}),
            ToolContext("parent", str(tmp_path)),
        )
        assert result.is_error
        assert result.meta is not None
        assert result.meta["finishReason"] == "error"
        assert "partial" in result.text
        await service.dispose()

    asyncio.run(scenario())


def test_acp_subagent_tool_is_hidden_without_configuration(tmp_path) -> None:
    async def scenario() -> None:
        service = HarnessService(
            tmp_path / "sessions",
            cwd=tmp_path,
            adapter_factory=lambda _model: cast(LlmAdapter, RepeatingAdapter()),
        )
        await service.dispatch("session.create", {"sessionId": "parent", "cwd": str(tmp_path)})
        registry = service._tool_registries["parent"]
        schema = next(item for item in registry.schemas() if item.name == "subagent")
        properties = schema.parameters["properties"]
        assert isinstance(properties, dict)
        assert "agent" not in properties
        await service.dispose()

    asyncio.run(scenario())


def sdk_service(tmp_path, extra_env=None):
    """A harness whose `subagent` tool can delegate to the fixture SDK runtime."""

    import os
    import sys
    from pathlib import Path

    from deepseek_harness.sdk_client import SdkSubagentConfig

    env = {name: value for name, value in os.environ.items() if name.startswith("MOCK_")}
    env.update(extra_env or {})
    return HarnessService(
        tmp_path / "sessions",
        cwd=tmp_path,
        adapter_factory=lambda _model: cast(LlmAdapter, RepeatingAdapter()),
        sdk_subagent=SdkSubagentConfig(
            command=sys.executable,
            args=(str(Path(__file__).parent / "sdk_child_fixture.py"),),
            env=env,
            shutdown_timeout_ms=1_000,
            dispose_eof_grace_ms=2_000,
        ),
    )


def test_sdk_subagent_delegates_foreground_through_the_tool(tmp_path) -> None:
    async def scenario() -> None:
        service = sdk_service(tmp_path, {"MOCK_SDK_TEXT": "peer harness answer"})
        await service.dispatch("session.create", {"sessionId": "parent", "cwd": str(tmp_path)})
        registry = service._tool_registries["parent"]
        result = await registry.execute(
            "subagent",
            json.dumps(
                {
                    "description": "ask the peer harness",
                    "prompt": "Answer directly.",
                    "agent": "dsh-sdk",
                }
            ),
            ToolContext("parent", str(tmp_path)),
        )
        assert not result.is_error
        assert "peer harness answer" in result.text
        assert result.meta is not None
        assert result.meta["provider"] == "dsh-sdk"
        assert result.meta["finishReason"] == "completed"
        await service.dispose()

    asyncio.run(scenario())


def test_sdk_subagent_maps_an_unclean_child_turn_to_an_error(tmp_path) -> None:
    async def scenario() -> None:
        service = sdk_service(tmp_path, {"MOCK_SDK_TURN_KIND": "error", "MOCK_SDK_TEXT": "partial"})
        await service.dispatch("session.create", {"sessionId": "parent", "cwd": str(tmp_path)})
        registry = service._tool_registries["parent"]
        result = await registry.execute(
            "subagent",
            json.dumps({"description": "x", "prompt": "y", "agent": "dsh-sdk"}),
            ToolContext("parent", str(tmp_path)),
        )
        assert result.is_error
        assert result.meta is not None
        assert result.meta["finishReason"] == "error"
        assert "partial" in result.text
        await service.dispose()

    asyncio.run(scenario())


def test_both_remote_providers_are_offered_by_name(tmp_path) -> None:
    async def scenario() -> None:
        import os
        import sys
        from pathlib import Path

        from deepseek_harness.acp_client import AcpSubagentConfig
        from deepseek_harness.sdk_client import SdkSubagentConfig

        fixtures = Path(__file__).parent
        env = {name: value for name, value in os.environ.items() if name.startswith("MOCK_")}
        service = HarnessService(
            tmp_path / "sessions",
            cwd=tmp_path,
            adapter_factory=lambda _model: cast(LlmAdapter, RepeatingAdapter()),
            acp_subagent=AcpSubagentConfig(
                command=sys.executable,
                args=(str(fixtures / "acp_child_fixture.py"),),
                env=env,
                dispose_eof_grace_ms=2_000,
            ),
            sdk_subagent=SdkSubagentConfig(
                command=sys.executable,
                args=(str(fixtures / "sdk_child_fixture.py"),),
                env=env,
                dispose_eof_grace_ms=2_000,
            ),
        )
        await service.dispatch("session.create", {"sessionId": "parent", "cwd": str(tmp_path)})
        registry = service._tool_registries["parent"]
        schema = next(item for item in registry.schemas() if item.name == "subagent")
        properties = schema.parameters["properties"]
        assert isinstance(properties, dict)
        agent_schema = properties["agent"]
        assert isinstance(agent_schema, dict)
        assert agent_schema["enum"] == ["acp", "dsh-sdk"]
        await service.dispose()

    asyncio.run(scenario())


def claude_service(tmp_path, extra_env=None):
    """A harness whose `subagent` tool can delegate to the fixture Claude CLI."""

    import os
    import sys
    from pathlib import Path

    from deepseek_harness.claude_code_client import ClaudeCodeSubagentConfig

    env = {name: value for name, value in os.environ.items() if name.startswith("MOCK_")}
    env.update(extra_env or {})
    return HarnessService(
        tmp_path / "sessions",
        cwd=tmp_path,
        adapter_factory=lambda _model: cast(LlmAdapter, RepeatingAdapter()),
        claude_code_subagent=ClaudeCodeSubagentConfig(
            command=sys.executable,
            args=(str(Path(__file__).parent / "claude_child_fixture.py"),),
            env=env,
            dispose_eof_grace_ms=2_000,
        ),
    )


def test_claude_code_subagent_delegates_the_task_and_returns_the_answer(tmp_path) -> None:
    async def scenario() -> None:
        service = claude_service(tmp_path, {"MOCK_CLAUDE_RESULT": "claude finished it"})
        await service.dispatch("session.create", {"sessionId": "parent", "cwd": str(tmp_path)})
        registry = service._tool_registries["parent"]
        result = await registry.execute(
            "subagent",
            json.dumps(
                {
                    "description": "hand off to claude",
                    "prompt": "Do the whole task.",
                    "agent": "claude-code",
                }
            ),
            ToolContext("parent", str(tmp_path)),
        )
        assert not result.is_error
        assert "claude finished it" in result.text
        assert result.meta is not None
        assert result.meta["provider"] == "claude-code"
        assert result.meta["finishReason"] == "completed"
        await service.dispose()

    asyncio.run(scenario())


def test_claude_code_subagent_fails_a_non_success_result(tmp_path) -> None:
    async def scenario() -> None:
        service = claude_service(
            tmp_path,
            {"MOCK_CLAUDE_SUBTYPE": "error_max_turns", "MOCK_CLAUDE_ERRORS": "turn cap"},
        )
        await service.dispatch("session.create", {"sessionId": "parent", "cwd": str(tmp_path)})
        registry = service._tool_registries["parent"]
        result = await registry.execute(
            "subagent",
            json.dumps({"description": "x", "prompt": "y", "agent": "claude-code"}),
            ToolContext("parent", str(tmp_path)),
        )
        assert result.is_error
        assert result.meta is not None
        assert result.meta["finishReason"] == "error"
        await service.dispose()

    asyncio.run(scenario())


def test_claude_code_subagent_is_listed_while_running(tmp_path) -> None:
    async def scenario() -> None:
        import os
        import sys
        from pathlib import Path

        from deepseek_harness.claude_code_client import ClaudeCodeSubagentConfig

        ready = tmp_path / "ready"
        env = {name: value for name, value in os.environ.items() if name.startswith("MOCK_")}
        env.update({"MOCK_CLAUDE_HANG": "1", "MOCK_CLAUDE_READY_FILE": str(ready)})
        service = HarnessService(
            tmp_path / "sessions",
            cwd=tmp_path,
            adapter_factory=lambda _model: cast(LlmAdapter, RepeatingAdapter()),
            claude_code_subagent=ClaudeCodeSubagentConfig(
                command=sys.executable,
                args=(str(Path(__file__).parent / "claude_child_fixture.py"),),
                env=env,
                dispose_eof_grace_ms=200,
            ),
        )
        await service.dispatch("session.create", {"sessionId": "parent", "cwd": str(tmp_path)})
        registry = service._tool_registries["parent"]
        context = ToolContext("parent", str(tmp_path))
        pending = asyncio.ensure_future(
            registry.execute(
                "subagent",
                json.dumps({"description": "long job", "prompt": "Work.", "agent": "claude-code"}),
                context,
            )
        )
        for _ in range(500):
            if ready.exists():
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError("the delegated child never started")

        listed = await registry.execute("list_agents", "{}", context)
        assert not listed.is_error
        assert "long job" in listed.text
        assert "claude-code" in listed.text

        await service.dispose()
        result = await pending
        assert result.is_error

    asyncio.run(scenario())


def codex_service(tmp_path, extra_env=None):
    """A harness whose `subagent` tool can delegate to the fixture Codex server."""

    import os
    import sys
    from pathlib import Path

    from deepseek_harness.codex_client import CodexSubagentConfig

    env = {name: value for name, value in os.environ.items() if name.startswith("MOCK_")}
    env.update(extra_env or {})
    return HarnessService(
        tmp_path / "sessions",
        cwd=tmp_path,
        adapter_factory=lambda _model: cast(LlmAdapter, RepeatingAdapter()),
        codex_subagent=CodexSubagentConfig(
            command=sys.executable,
            args=(str(Path(__file__).parent / "codex_child_fixture.py"),),
            env=env,
            dispose_eof_grace_ms=2_000,
        ),
    )


def test_codex_subagent_delegates_and_returns_the_final_answer(tmp_path) -> None:
    async def scenario() -> None:
        service = codex_service(tmp_path, {"MOCK_CODEX_ANSWER": "codex handled it"})
        await service.dispatch("session.create", {"sessionId": "parent", "cwd": str(tmp_path)})
        registry = service._tool_registries["parent"]
        result = await registry.execute(
            "subagent",
            json.dumps({"description": "hand off to codex", "prompt": "Do it.", "agent": "codex"}),
            ToolContext("parent", str(tmp_path)),
        )
        assert not result.is_error
        assert "codex handled it" in result.text
        assert result.meta is not None
        assert result.meta["provider"] == "codex"
        assert result.meta["finishReason"] == "completed"
        await service.dispose()

    asyncio.run(scenario())


def test_codex_subagent_fails_a_turn_without_an_answer(tmp_path) -> None:
    async def scenario() -> None:
        service = codex_service(
            tmp_path, {"MOCK_CODEX_ANSWER": "  ", "MOCK_CODEX_COMMENTARY": "only thinking"}
        )
        await service.dispatch("session.create", {"sessionId": "parent", "cwd": str(tmp_path)})
        registry = service._tool_registries["parent"]
        result = await registry.execute(
            "subagent",
            json.dumps({"description": "x", "prompt": "y", "agent": "codex"}),
            ToolContext("parent", str(tmp_path)),
        )
        assert result.is_error
        assert result.meta is not None
        assert result.meta["finishReason"] == "error"
        await service.dispose()

    asyncio.run(scenario())


def test_all_four_remote_providers_are_offered_by_name(tmp_path) -> None:
    async def scenario() -> None:
        import os
        import sys
        from pathlib import Path

        from deepseek_harness.acp_client import AcpSubagentConfig
        from deepseek_harness.claude_code_client import ClaudeCodeSubagentConfig
        from deepseek_harness.codex_client import CodexSubagentConfig
        from deepseek_harness.sdk_client import SdkSubagentConfig

        fixtures = Path(__file__).parent
        env = {name: value for name, value in os.environ.items() if name.startswith("MOCK_")}
        service = HarnessService(
            tmp_path / "sessions",
            cwd=tmp_path,
            adapter_factory=lambda _model: cast(LlmAdapter, RepeatingAdapter()),
            acp_subagent=AcpSubagentConfig(
                command=sys.executable,
                args=(str(fixtures / "acp_child_fixture.py"),),
                env=env,
                dispose_eof_grace_ms=2_000,
            ),
            sdk_subagent=SdkSubagentConfig(
                command=sys.executable,
                args=(str(fixtures / "sdk_child_fixture.py"),),
                env=env,
                dispose_eof_grace_ms=2_000,
            ),
            claude_code_subagent=ClaudeCodeSubagentConfig(
                command=sys.executable,
                args=(str(fixtures / "claude_child_fixture.py"),),
                env=env,
                dispose_eof_grace_ms=2_000,
            ),
            codex_subagent=CodexSubagentConfig(
                command=sys.executable,
                args=(str(fixtures / "codex_child_fixture.py"),),
                env=env,
                dispose_eof_grace_ms=2_000,
            ),
        )
        await service.dispatch("session.create", {"sessionId": "parent", "cwd": str(tmp_path)})
        registry = service._tool_registries["parent"]
        schema = next(item for item in registry.schemas() if item.name == "subagent")
        properties = schema.parameters["properties"]
        assert isinstance(properties, dict)
        agent_schema = properties["agent"]
        assert isinstance(agent_schema, dict)
        assert agent_schema["enum"] == ["acp", "dsh-sdk", "claude-code", "codex"]
        await service.dispose()

    asyncio.run(scenario())
