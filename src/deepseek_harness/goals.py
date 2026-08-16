"""Event-sourced goal mutations and the ``goal`` session projection."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from .models import JsonValue, now_millis
from .session import Session

GoalPhase = Literal["active", "paused", "blocked", "complete"]


class GoalError(Exception):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class GoalSnapshot:
    id: str
    revision: int
    objective: str
    phase: GoalPhase
    max_goal_rounds: int
    blocked_reason: dict[str, JsonValue] | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {
            "id": self.id,
            "revision": self.revision,
            "objective": self.objective,
            "phase": self.phase,
            "maxGoalRounds": self.max_goal_rounds,
        }
        if self.blocked_reason is not None:
            result["blockedReason"] = self.blocked_reason
        return result


@dataclass(frozen=True, slots=True)
class GoalFold:
    goal: GoalSnapshot | None
    rounds_started: int
    created_at: int | None
    updated_at: int | None
    last_ref: dict[str, JsonValue] | None

    def projection(self) -> dict[str, JsonValue] | None:
        if self.goal is None or self.created_at is None or self.updated_at is None:
            return None
        return {
            "goal": self.goal.to_dict(),
            "roundsStarted": self.rounds_started,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }


class GoalManager:
    """Pure replay plus CAS-guarded writes against a Session log."""

    def __init__(self, *, default_max_goal_rounds: int = 256) -> None:
        self.default_max_goal_rounds = default_max_goal_rounds

    def fold(self, session: Session) -> GoalFold:
        goal: GoalSnapshot | None = None
        rounds_started = 0
        created_at: int | None = None
        updated_at: int | None = None
        last_ref: dict[str, JsonValue] | None = None
        for event in session.events:
            if event.type != "goal/change":
                continue
            raw = event.data
            if raw.get("kind") != "goal/change":
                continue
            operation = raw.get("operation")
            if operation == "clear":
                cleared = raw.get("cleared")
                if isinstance(cleared, dict):
                    revision = cleared.get("revision")
                    last_ref = {
                        "id": str(cleared.get("id", "")),
                        "revision": revision if isinstance(revision, int) else 0,
                    }
                goal = None
                created_at = None
                updated_at = None
                rounds_started = 0
                continue
            raw_goal = raw.get("goal")
            if not isinstance(raw_goal, dict):
                continue
            goal = self._decode_snapshot(raw_goal)
            rounds = raw.get("roundsStarted", 0)
            rounds_started = int(rounds) if isinstance(rounds, int) else 0
            created = raw.get("createdAt")
            updated = raw.get("updatedAt")
            created_at = int(created) if isinstance(created, int) else None
            updated_at = int(updated) if isinstance(updated, int) else None
            last_ref = {"id": goal.id, "revision": goal.revision}
        return GoalFold(goal, rounds_started, created_at, updated_at, last_ref)

    def view(
        self,
        session: Session,
        *,
        activation: Literal["armed", "disarmed"] = "disarmed",
    ) -> dict[str, JsonValue] | None:
        """Return the model-facing live view for the current goal.

        ``activation`` is deliberately supplied by the host.  It is process
        local and must never be written into the durable ``goal/change``
        payload, matching the TypeScript goal service's separation between
        replay state and continuation authority.
        """

        projection = self.fold(session).projection()
        if projection is None:
            return None
        raw_goal = projection.get("goal")
        if not isinstance(raw_goal, dict):
            return None
        view = dict(raw_goal)
        for key in ("roundsStarted", "createdAt", "updatedAt"):
            value = projection.get(key)
            if value is not None:
                view[key] = value
        view["activation"] = activation
        return view

    def create(
        self, session: Session, objective: str, max_goal_rounds: int | None
    ) -> dict[str, JsonValue]:
        current = self.fold(session).goal
        if current is not None and current.phase != "complete":
            raise GoalError(
                f'goal "{current.id}" already exists with phase "{current.phase}"',
                "GOAL_ALREADY_EXISTS",
            )
        normalized = self._objective(objective)
        rounds = self._rounds(max_goal_rounds)
        now = now_millis()
        goal = GoalSnapshot(f"goal-{uuid.uuid4().hex}", 1, normalized, "active", rounds)
        self._append_snapshot(session, "create", goal, 0, now, now)
        return {"id": goal.id, "revision": goal.revision}

    def edit(
        self,
        session: Session,
        ref: dict[str, Any],
        objective: str | None,
        max_goal_rounds: int | None,
    ) -> dict[str, JsonValue]:
        if objective is None and max_goal_rounds is None:
            raise GoalError(
                "goal edit requires objective and/or maxGoalRounds", "GOAL_INVALID_EDIT"
            )
        folded = self._expect(session, ref)
        current = folded.goal
        assert current is not None
        goal = GoalSnapshot(
            current.id,
            current.revision + 1,
            self._objective(objective) if objective is not None else current.objective,
            current.phase,
            self._rounds(max_goal_rounds)
            if max_goal_rounds is not None
            else current.max_goal_rounds,
            current.blocked_reason,
        )
        self._append_snapshot(
            session,
            "edit",
            goal,
            folded.rounds_started,
            folded.created_at or now_millis(),
            self._next_time(folded),
        )
        return {"id": goal.id, "revision": goal.revision}

    def transition(
        self,
        session: Session,
        operation: Literal["pause", "resume", "complete"],
        ref: dict[str, Any],
    ) -> dict[str, JsonValue]:
        folded = self._expect(session, ref)
        current = folded.goal
        assert current is not None
        allowed: dict[str, tuple[GoalPhase, ...]] = {
            "pause": ("active",),
            "resume": ("active", "paused", "blocked"),
            "complete": ("active", "paused", "blocked"),
        }
        if current.phase not in allowed[operation]:
            raise GoalError(
                f'cannot {operation} goal "{current.id}" from phase "{current.phase}"',
                "GOAL_INVALID_TRANSITION",
            )
        if operation == "resume" and folded.rounds_started >= current.max_goal_rounds:
            raise GoalError(
                f'goal "{current.id}" exhausted its round budget', "GOAL_INVALID_TRANSITION"
            )
        phase: GoalPhase = (
            "paused" if operation == "pause" else "active" if operation == "resume" else "complete"
        )
        goal = GoalSnapshot(
            current.id,
            current.revision + 1,
            current.objective,
            phase,
            current.max_goal_rounds,
        )
        self._append_snapshot(
            session,
            operation,
            goal,
            folded.rounds_started,
            folded.created_at or now_millis(),
            self._next_time(folded),
        )
        return {"id": goal.id, "revision": goal.revision}

    def block(
        self,
        session: Session,
        ref: dict[str, Any],
        reason: dict[str, Any],
    ) -> dict[str, JsonValue]:
        """Mark an active goal blocked with a durable policy explanation."""

        folded = self._expect(session, ref)
        current = folded.goal
        assert current is not None
        if current.phase != "active":
            raise GoalError(
                f'cannot block goal "{current.id}" from phase "{current.phase}"',
                "GOAL_INVALID_TRANSITION",
            )
        code = reason.get("code")
        message = reason.get("message")
        if (
            not isinstance(code, str)
            or re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", code) is None
            or not isinstance(message, str)
            or not message.strip()
        ):
            raise GoalError(
                "goal block reason requires a lower-kebab-case code and a non-empty message",
                "GOAL_INVALID_BLOCK_REASON",
            )
        goal = GoalSnapshot(
            current.id,
            current.revision + 1,
            current.objective,
            "blocked",
            current.max_goal_rounds,
            {"code": code, "message": message.strip()},
        )
        self._append_snapshot(
            session,
            "block",
            goal,
            folded.rounds_started,
            folded.created_at or now_millis(),
            self._next_time(folded),
        )
        return {"id": goal.id, "revision": goal.revision}

    def clear(self, session: Session, ref: dict[str, Any]) -> None:
        folded = self._expect(session, ref)
        current = folded.goal
        assert current is not None
        tombstone = {"id": current.id, "revision": current.revision + 1}
        session.append(
            "goal/change",
            {
                "kind": "goal/change",
                "version": 1,
                "operation": "clear",
                "cleared": tombstone,
                "clearedAt": self._next_time(folded),
            },
        )

    def _append_snapshot(
        self,
        session: Session,
        operation: str,
        goal: GoalSnapshot,
        rounds_started: int,
        created_at: int,
        updated_at: int,
    ) -> None:
        session.append(
            "goal/change",
            {
                "kind": "goal/change",
                "version": 1,
                "operation": operation,
                "goal": goal.to_dict(),
                "roundsStarted": rounds_started,
                "createdAt": created_at,
                "updatedAt": updated_at,
            },
        )

    def _expect(self, session: Session, ref: dict[str, Any]) -> GoalFold:
        folded = self.fold(session)
        current = folded.goal
        if current is None:
            raise GoalError("no current goal", "GOAL_NOT_FOUND")
        if ref.get("id") != current.id or ref.get("revision") != current.revision:
            raise GoalError("goal reference is stale", "GOAL_STALE_REVISION")
        return folded

    def _rounds(self, value: int | None) -> int:
        resolved = self.default_max_goal_rounds if value is None else value
        if isinstance(resolved, bool) or not isinstance(resolved, int) or resolved < 1:
            raise GoalError("maxGoalRounds must be a positive integer", "GOAL_INVALID_MAX_ROUNDS")
        return resolved

    @staticmethod
    def _objective(value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise GoalError("goal objective must be a non-empty string", "GOAL_INVALID_OBJECTIVE")
        return normalized

    @staticmethod
    def _next_time(folded: GoalFold) -> int:
        return max(now_millis(), folded.updated_at or 0)

    @staticmethod
    def _decode_snapshot(value: dict[str, Any]) -> GoalSnapshot:
        phase = value.get("phase")
        if phase not in {"active", "paused", "blocked", "complete"}:
            raise ValueError("goal snapshot has an invalid phase")
        reason = value.get("blockedReason")
        return GoalSnapshot(
            str(value.get("id", "")),
            int(value.get("revision", 0)),
            str(value.get("objective", "")),
            phase,
            int(value.get("maxGoalRounds", 0)),
            dict(reason) if isinstance(reason, dict) else None,
        )


__all__ = ["GoalError", "GoalFold", "GoalManager", "GoalSnapshot"]
