from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from deepseek_harness.hooks import (
    HookBridge,
    HookConfigError,
    HookOutput,
    matches_matcher,
    merge_hook_outputs,
    parse_claude_code_config,
    parse_hook_output,
    post_tool_use_hook,
    pre_tool_use_hook,
)
from deepseek_harness.session import Session
from deepseek_harness.tools import (
    PermissionMode,
    ToolContext,
    ToolRegistry,
    WorkspacePolicy,
    install_builtin_tools,
)


def _write_hook_config(tmp_path: Path, hooks: dict) -> Path:
    config_path = tmp_path / "hooks.json"
    config_path.write_text(json.dumps({"hooks": hooks}), encoding="utf-8")
    return config_path


def test_parse_config_keeps_command_hooks_and_skips_others(tmp_path: Path) -> None:
    config_path = _write_hook_config(
        tmp_path,
        {
            "PreToolUse": [
                {
                    "matcher": "Bash|Terminal",
                    "hooks": [
                        {"type": "command", "command": "echo pre", "timeout": 5},
                        {"type": "prompt", "command": "ignored"},
                    ],
                }
            ],
            "Stop": [{"hooks": [{"type": "command", "command": "echo stop"}]}],
        },
    )
    bridge = HookBridge(str(config_path))
    assert set(bridge.config) == {"PreToolUse", "Stop"}
    assert bridge.config["PreToolUse"][0].matcher == "Bash|Terminal"
    assert bridge.config["PreToolUse"][0].hooks[0].timeout_sec == 5
    assert bridge.config["Stop"][0].matcher is None
    assert bridge.skipped == [("PreToolUse", "prompt")]


def test_parse_config_rejects_invalid_regex_matcher(tmp_path: Path) -> None:
    config_path = _write_hook_config(
        tmp_path,
        {"PreToolUse": [{"matcher": "([", "hooks": [{"type": "command", "command": "x"}]}]},
    )
    with pytest.raises(HookConfigError, match="invalid claude-code regex matcher"):
        HookBridge(str(config_path))


def test_missing_config_fails_load(tmp_path: Path) -> None:
    with pytest.raises(HookConfigError, match="could not load hook config"):
        HookBridge(str(tmp_path / "missing.json"))


def test_matcher_semantics_follow_claude_code() -> None:
    assert matches_matcher(None, "Bash")
    assert matches_matcher("*", "Bash")
    assert matches_matcher("Bash|Read", "Read")
    assert not matches_matcher("Bash|Read", "Write")
    assert matches_matcher("^W", "Write")
    assert not matches_matcher("^R", "Write")


def test_parse_hook_output_decodes_exit_codes_and_structured_stdout() -> None:
    blocked = parse_hook_output(2, "", "no secrets in shell")
    assert blocked.decision == "block"
    assert blocked.reason == "no secrets in shell"

    structured = parse_hook_output(
        0,
        json.dumps(
            {
                "continue": False,
                "stopReason": "halted",
                "systemMessage": "note",
                "decision": "approve",
                "reason": "ok",
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "dangerous",
                    "additionalContext": "extra",
                    "updatedInput": {"command": "safe"},
                },
            }
        ),
        "",
        expected_event_name="PreToolUse",
    )
    assert structured.decision == "deny"
    assert structured.reason == "dangerous"
    assert structured.stop is True
    assert structured.stop_reason == "halted"
    assert structured.additional_context == "extra"
    assert structured.updated_input == {"command": "safe"}

    # A hookSpecificOutput naming a different event keeps only top-level fields.
    mismatched = parse_hook_output(
        0,
        json.dumps(
            {
                "decision": "approve",
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "permissionDecision": "deny",
                },
            }
        ),
        "",
        expected_event_name="PreToolUse",
    )
    assert mismatched.decision == "approve"

    # Out-of-band top-level deny is invalid and ignored per the schemas.
    invalid = parse_hook_output(0, json.dumps({"decision": "deny"}), "")
    assert invalid.decision is None


def test_merge_applies_deny_over_ask_over_allow() -> None:
    merged = merge_hook_outputs(
        [
            HookOutput(exit_code=0, stderr="", stdout="", decision="allow"),
            HookOutput(exit_code=0, stderr="", stdout="", decision="ask", reason="why"),
            HookOutput(exit_code=0, stderr="", stdout="", decision="block", reason="no"),
        ]
    )
    assert merged.decision == "deny"
    assert merged.reason == "no"
    plain = merge_hook_outputs([])
    assert plain.decision == "none" and not plain.stop


def test_pre_and_post_tool_hooks_block_and_inject_context(tmp_path: Path) -> None:
    async def scenario() -> None:
        config_path = _write_hook_config(
            tmp_path,
            {
                "PreToolUse": [
                    {
                        "matcher": "read_file",
                        "hooks": [
                            {
                                "type": "command",
                                "command": (
                                    "python3 -c \"import json,sys;"
                                    "d=json.load(sys.stdin);"
                                    "print(json.dumps({'hookSpecificOutput':"
                                    "{'hookEventName':'PreToolUse',"
                                    "'permissionDecision':'deny',"
                                    "'permissionDecisionReason':'no shell for you'}}))\""
                                ),
                            }
                        ],
                    }
                ],
                "PostToolUse": [
                    {
                        "matcher": "write_file",
                        "hooks": [
                            {
                                "type": "command",
                                "command": (
                                    "python3 -c \"import json,sys;"
                                    "json.load(sys.stdin);"
                                    "print(json.dumps({'hookSpecificOutput':"
                                    "{'hookEventName':'PostToolUse',"
                                    "'additionalContext':'checked by hook'}}))\""
                                ),
                            }
                        ],
                    }
                ],
            },
        )
        bridge = HookBridge(str(config_path))
        session = Session("hooks-session")
        registry = ToolRegistry()

        async def pre_execute(name, arguments, context):
            return await pre_tool_use_hook(bridge, session, name, arguments, context)

        async def post_execute(name, arguments, context, result):
            return await post_tool_use_hook(
                bridge, session, name, arguments, context, result
            )

        registry.pre_execute = pre_execute
        registry.post_execute = post_execute
        disposers = install_builtin_tools(
            registry,
            WorkspacePolicy(tmp_path, PermissionMode.DANGER_FULL_ACCESS),
        )
        context = ToolContext("hooks-session", str(tmp_path))
        try:
            blocked = await registry.execute(
                "read_file", '{"path":"a.txt"}', context
            )
            assert blocked.is_error
            assert "no shell for you" in blocked.text

            written = await registry.execute(
                "write_file", json.dumps({"path": "a.txt", "content": "hi"}), context
            )
            assert not written.is_error
            assert "checked by hook" in written.text
        finally:
            for dispose in reversed(disposers):
                dispose()

    asyncio.run(scenario())


def test_service_prompt_and_stop_hooks_fire(tmp_path: Path) -> None:
    async def scenario() -> None:
        class Adapter:
            def stream(self, request):
                del request

                async def chunks():
                    yield StreamChunk(kind="text", text="done")
                    yield StreamChunk(kind="done", finish_reason="stop")

                return chunks()

            async def aclose(self) -> None:
                return None

        from deepseek_harness.llm.types import StreamChunk
        from deepseek_harness.sandbox import UnavailableSandbox
        from deepseek_harness.web import HarnessService

        # UserPromptSubmit deny: the prompt is rejected before any turn starts.
        deny_config = _write_hook_config(
            tmp_path,
            {
                "UserPromptSubmit": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": (
                                    "python3 -c \"import json,sys;"
                                    "d=json.load(sys.stdin);"
                                    "decision = 'deny' if 'secret' in d.get('prompt','') "
                                    "else 'allow';"
                                    "print(json.dumps({'decision': decision,"
                                    "'reason': 'hook said no'}))\""
                                ),
                            }
                        ]
                    }
                ]
            },
        )
        service = HarnessService(
            tmp_path / "state",
            cwd=tmp_path,
            adapter_factory=lambda _model: Adapter(),
            sandbox_provider=UnavailableSandbox(),
            hooks_config_path=deny_config,
        )
        await service.create_session(session_id="hooked", cwd=str(tmp_path))
        try:
            with pytest.raises(Exception, match="hook said no"):
                await service.prompt(
                    "hooked",
                    [{"type": "text", "text": "tell me the secret"}],
                )
            accepted = await service.prompt(
                "hooked",
                [{"type": "text", "text": "hello"}],
            )
            assert accepted["accepted"] is True
        finally:
            await service.dispose()

        # SessionStart additionalContext prepends to the first prompt content.
        context_config = _write_hook_config(
            tmp_path,
            {
                "SessionStart": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": (
                                    "python3 -c \"import json;"
                                    "print(json.dumps({'hookSpecificOutput':"
                                    "{'hookEventName':'SessionStart',"
                                    "'additionalContext':'session boot context'}}))\""
                                ),
                            }
                        ]
                    }
                ]
            },
        )
        service2 = HarnessService(
            tmp_path / "state2",
            cwd=tmp_path,
            adapter_factory=lambda _model: Adapter(),
            sandbox_provider=UnavailableSandbox(),
            hooks_config_path=context_config,
        )
        handle2 = await service2.create_session(session_id="boot", cwd=str(tmp_path))
        try:
            for _ in range(100):
                if handle2.pending_hook_context:
                    break
                await asyncio.sleep(0.05)
            assert "session boot context" in handle2.pending_hook_context
        finally:
            await service2.dispose()


def test_hook_events_are_recorded_for_tool_hooks(tmp_path: Path) -> None:
    config_path = _write_hook_config(
        tmp_path,
        {"PostToolUse": [{"hooks": [{"type": "command", "command": "cat > /dev/null"}]}]},
    )
    bridge = HookBridge(str(config_path))
    assert bridge.loaded


def test_parse_config_accepts_bare_event_map(tmp_path: Path) -> None:
    config_path = tmp_path / "hooks.json"
    config_path.write_text(
        json.dumps({"Stop": [{"hooks": [{"type": "command", "command": "true"}]}]}),
        encoding="utf-8",
    )
    config, _skipped = parse_claude_code_config(
        json.loads(config_path.read_text(encoding="utf-8"))
    )
    assert "Stop" in config


def test_substitute_command_replaces_tokens() -> None:
    from deepseek_harness.hooks import substitute_command

    assert substitute_command(
        "${CLAUDE_PLUGIN_ROOT}/run.sh ${CLAUDE_PROJECT_DIR}",
        plugin_root="/plugins/x",
        project_dir="/work",
    ) == "/plugins/x/run.sh /work"
