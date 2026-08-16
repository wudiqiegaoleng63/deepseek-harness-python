"""Event-sourced plan-mode state shared by the web host and model tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .session import Session


@dataclass(frozen=True, slots=True)
class PlanFold:
    """The logged mode and a pending command selection, if one exists."""

    active: bool
    wanted: bool | None

    @property
    def pending(self) -> bool:
        return self.wanted is not None and self.wanted != self.active

    def to_dict(self) -> dict[str, bool]:
        return {"active": self.active, "pending": self.pending}


def fold(session: Session) -> PlanFold:
    """Fold the last ``plan/mode`` and ``/plan`` command records."""

    active = False
    wanted: bool | None = None
    for event in session.events:
        if event.type == "command/run" and event.data.get("name") == "plan":
            args = event.data.get("args")
            if isinstance(args, str):
                wanted = args.strip() != "off"
        elif event.type == "plan/mode":
            value = event.data.get("active")
            if isinstance(value, bool):
                active = value
                wanted = None
    return PlanFold(active, wanted)


def has_open_turn(session: Session) -> bool:
    """Whether the log currently contains an unclosed ``turn/start``."""

    open_turn = False
    for event in session.events:
        if event.type == "turn/start":
            open_turn = True
        elif event.type == "turn/end":
            open_turn = False
    return open_turn


def command_target(args: Any) -> bool | None:
    """Decode a command's raw plan argument into its requested mode."""

    if not isinstance(args, str):
        return None
    return args.strip() != "off"


__all__ = ["PlanFold", "command_target", "fold", "has_open_turn"]
