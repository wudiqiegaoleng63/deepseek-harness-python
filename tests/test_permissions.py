from __future__ import annotations

import asyncio

from deepseek_harness.permissions import (
    CUSTOM_PERMISSION_PRESET,
    PermissionPresetManager,
)
from deepseek_harness.sandbox import UnavailableSandbox
from deepseek_harness.session import Session
from deepseek_harness.tools import PermissionMode
from deepseek_harness.tools.registry import ToolContext
from deepseek_harness.web import HarnessService


def test_permission_preset_fold_and_projection_match_shared_contract() -> None:
    manager = PermissionPresetManager()
    session = Session("permission-pure")

    assert manager.current(
        list(session.events),
        default_mode=PermissionMode.WORKSPACE_WRITE,
        default_approval="ask",
    ) == "workspace-write"
    session.append("sandbox/mode", {"mode": "read-only"})
    projection = manager.projection(
        list(session.events),
        default_mode=PermissionMode.WORKSPACE_WRITE,
        default_approval="ask",
    ).to_dict()
    assert projection["currentValue"] == CUSTOM_PERMISSION_PRESET
    options = projection["options"]
    assert isinstance(options, list)
    last = options[-1]
    assert isinstance(last, dict)
    assert last["value"] == CUSTOM_PERMISSION_PRESET

    changes = manager.change_events(
        list(session.events),
        "danger-full-access",
        default_mode=PermissionMode.WORKSPACE_WRITE,
        default_approval="ask",
    )
    assert changes == (
        ("permission/preset", {"preset": "danger-full-access"}),
        ("sandbox/mode", {"mode": "danger-full-access"}),
        ("approval/policy", {"policy": "never"}),
    )


def test_service_permission_switch_projects_persists_and_reconfigures_shell(tmp_path) -> None:
    async def scenario() -> None:
        state = tmp_path / "state"
        # Pin the sandbox off so the workspace-write assertions below describe
        # the no-sandbox deployment; sandbox-enabled behavior has its own tests.
        service = HarnessService(state, cwd=tmp_path, sandbox_provider=UnavailableSandbox())
        handle = await service.create_session(session_id="permission-service", cwd=str(tmp_path))
        assert "bash" not in service._tool_registries[handle.session.id].names()
        initial = (await service.history(handle.session.id))["projections"]["values"]
        assert initial["permissions"]["currentValue"] == "workspace-write"

        result = await service.dispatch(
            "permission.set",
            {"sessionId": handle.session.id, "preset": "danger-full-access"},
        )
        assert result["permissions"]["currentValue"] == "danger-full-access"
        assert "bash" in service._tool_registries[handle.session.id].names()
        shell_result = await service._tool_registries[handle.session.id].execute(
            "bash",
            '{"command":"printf permission-switch"}',
            ToolContext(handle.session.id, str(tmp_path)),
        )
        assert shell_result.text == "permission-switch"

        await service.dispatch(
            "permission.set",
            {"sessionId": handle.session.id, "preset": "workspace-write"},
        )
        assert "bash" not in service._tool_registries[handle.session.id].names()
        assert [event.type for event in handle.session.events if event.type in {
            "permission/preset", "sandbox/mode", "approval/policy"
        }] == [
            "permission/preset",
            "sandbox/mode",
            "approval/policy",
            "permission/preset",
            "sandbox/mode",
            "approval/policy",
        ]
        await service.dispose()

        resumed = HarnessService(state, cwd=tmp_path, sandbox_provider=UnavailableSandbox())
        resumed_handle = await resumed.get_session(handle.session.id)
        assert resumed_handle.policy.mode is PermissionMode.WORKSPACE_WRITE
        assert "bash" not in resumed._tool_registries[handle.session.id].names()
        await resumed.dispose()

    asyncio.run(scenario())


def test_permission_command_and_never_approval_are_auditable(tmp_path) -> None:
    async def scenario() -> None:
        service = HarnessService(tmp_path / "state", cwd=tmp_path)
        handle = await service.create_session(session_id="permission-command", cwd=str(tmp_path))
        command = await service.prompt(
            handle.session.id,
            [{"type": "text", "text": "/permission danger-full-access"}],
        )
        assert command["command"]["text"] == "preset danger-full-access"
        assert handle.session.events[-1].type == "command/done"

        outcome = await service.request_approval(
            handle.session.id,
            "bash",
            approval_id="approval-never",
        )
        assert outcome == "rejected"
        assert not service._pending_approvals
        assert [event.type for event in handle.session.events][-2:] == [
            "approval/asked",
            "approval/decided",
        ]
        await service.dispose()

    asyncio.run(scenario())
