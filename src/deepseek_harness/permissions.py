"""Event-sourced permission presets shared by the web host and tools.

The TypeScript runtime keeps the user-facing preset separate from the two
enforcement knobs it controls: sandbox mode and approval policy.  Keeping the
same separation here makes a session restart a pure replay operation and lets
the web projection expose one stable select value without hiding the durable
policy events.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from .models import JsonValue
from .session import SessionEvent
from .tools.policy import PermissionMode

ApprovalPolicy = Literal["ask", "never"]
APPROVAL_POLICIES: tuple[ApprovalPolicy, ...] = ("ask", "never")
CUSTOM_PERMISSION_PRESET = "custom"


@dataclass(frozen=True, slots=True)
class PermissionPreset:
    """One selectable sandbox/approval bundle."""

    mode: PermissionMode
    approval: ApprovalPolicy
    name: str | None = None
    description: str | None = None


DEFAULT_PERMISSION_PRESETS: Mapping[str, PermissionPreset] = {
    "workspace-write": PermissionPreset(
        PermissionMode.WORKSPACE_WRITE,
        "ask",
        description=(
            "Write inside the workspace and permitted temporary directories; "
            "wider retries require approval."
        ),
    ),
    "danger-full-access": PermissionPreset(
        PermissionMode.DANGER_FULL_ACCESS,
        "never",
        description="Full file access without approval prompts.",
    ),
}


@dataclass(frozen=True, slots=True)
class PermissionKnobs:
    """Last durable value of each permission knob, before defaults are applied."""

    preset: str | None = None
    mode: PermissionMode | None = None
    approval: ApprovalPolicy | None = None


@dataclass(frozen=True, slots=True)
class PermissionProjection:
    """JSON-compatible select state sent to the shared frontend."""

    options: tuple[dict[str, str], ...]
    current_value: str

    def to_dict(self) -> dict[str, object]:
        return {
            "options": [dict(option) for option in self.options],
            "currentValue": self.current_value,
        }


class PermissionPresetManager:
    """Fold and write the permission preset contract for one deployment."""

    def __init__(
        self,
        presets: Mapping[str, PermissionPreset] | None = None,
    ) -> None:
        selected = dict(presets or DEFAULT_PERMISSION_PRESETS)
        if not selected:
            raise ValueError("permission preset table cannot be empty")
        if CUSTOM_PERMISSION_PRESET in selected:
            raise ValueError(f"{CUSTOM_PERMISSION_PRESET!r} is reserved for derived state")
        for name, spec in selected.items():
            if not name.strip():
                raise ValueError("permission preset names cannot be empty")
            if not isinstance(spec, PermissionPreset):
                raise TypeError(f"permission preset {name!r} must be a PermissionPreset")
            if spec.approval not in APPROVAL_POLICIES:
                raise ValueError(f"unknown approval policy for preset {name!r}")
        self._presets = selected

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._presets)

    def resolve(self, name: str) -> PermissionPreset:
        try:
            return self._presets[name]
        except KeyError as exc:
            raise ValueError(
                f'unknown permission preset "{name}" (available: {", ".join(self.names)})'
            ) from exc

    @staticmethod
    def fold(events: tuple[SessionEvent, ...] | list[SessionEvent]) -> PermissionKnobs:
        """Fold the last valid value of each durable permission knob."""

        state = PermissionKnobs()
        for event in events:
            if event.type == "permission/preset":
                value = event.data.get("preset")
                if isinstance(value, str) and value:
                    state = PermissionKnobs(value, state.mode, state.approval)
            elif event.type == "sandbox/mode":
                value = event.data.get("mode")
                try:
                    mode = PermissionMode(value) if isinstance(value, str) else None
                except ValueError:
                    mode = None
                if mode is not None:
                    state = PermissionKnobs(state.preset, mode, state.approval)
            elif event.type == "approval/policy":
                value = event.data.get("policy")
                if value in APPROVAL_POLICIES:
                    state = PermissionKnobs(state.preset, state.mode, value)
        return state

    def current(
        self,
        events: tuple[SessionEvent, ...] | list[SessionEvent],
        *,
        default_mode: PermissionMode,
        default_approval: ApprovalPolicy,
    ) -> str:
        state = self.fold(events)
        mode = state.mode or default_mode
        approval = state.approval or default_approval

        def matches(spec: PermissionPreset) -> bool:
            return spec.mode is mode and spec.approval == approval

        if state.preset is not None:
            selected = self._presets.get(state.preset)
            if selected is not None and matches(selected):
                return state.preset
        for name, spec in self._presets.items():
            if matches(spec):
                return name
        return CUSTOM_PERMISSION_PRESET

    def projection(
        self,
        events: tuple[SessionEvent, ...] | list[SessionEvent],
        *,
        default_mode: PermissionMode,
        default_approval: ApprovalPolicy,
    ) -> PermissionProjection:
        current = self.current(
            events,
            default_mode=default_mode,
            default_approval=default_approval,
        )
        options = tuple(
            {
                "value": name,
                "name": spec.name or name,
                **({"description": spec.description} if spec.description else {}),
            }
            for name, spec in self._presets.items()
        )
        if current == CUSTOM_PERMISSION_PRESET:
            options += (
                {
                    "value": CUSTOM_PERMISSION_PRESET,
                    "name": "Custom",
                    "description": (
                        "Current sandbox and approval settings do not match a preset."
                    ),
                },
            )
        return PermissionProjection(options, current)

    def change_events(
        self,
        events: tuple[SessionEvent, ...] | list[SessionEvent],
        name: str,
        *,
        default_mode: PermissionMode,
        default_approval: ApprovalPolicy,
    ) -> tuple[tuple[str, dict[str, JsonValue]], ...]:
        """Return the minimal durable event sequence for selecting ``name``."""

        spec = self.resolve(name)
        state = self.fold(events)
        current = self.current(
            events,
            default_mode=default_mode,
            default_approval=default_approval,
        )
        changes: list[tuple[str, dict[str, JsonValue]]] = []
        if current != name:
            changes.append(("permission/preset", {"preset": name}))
        if spec.mode is not (state.mode or default_mode):
            changes.append(("sandbox/mode", {"mode": spec.mode.value}))
        if spec.approval != (state.approval or default_approval):
            changes.append(("approval/policy", {"policy": spec.approval}))
        return tuple(changes)


__all__ = [
    "APPROVAL_POLICIES",
    "ApprovalPolicy",
    "CUSTOM_PERMISSION_PRESET",
    "DEFAULT_PERMISSION_PRESETS",
    "PermissionKnobs",
    "PermissionPreset",
    "PermissionPresetManager",
    "PermissionProjection",
]
