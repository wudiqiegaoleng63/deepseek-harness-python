"""Advisory repeat-call detection mirroring the TS ``repeat-tool-reminder``.

The guard enriches tool results with logged model context without vetoing or
rewriting calls: consecutive identical calls trigger a gentle first reminder,
then detailed escalation naming the tool, run length, and canonical arguments.
A user interjection resets the chain — repetition across it is not a loop.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

DEFAULT_THRESHOLDS: tuple[int, ...] = (3, 5, 8)
DEFAULT_ARGUMENTS_PREVIEW_CHARS = 500

GENTLE_REMINDER = (
    "You are repeating the exact same tool call with identical arguments. "
    "Carefully analyze the previous result before calling again: if the task is "
    "not complete, try a different approach or different arguments instead of "
    "repeating the call."
)


def detailed_reminder(tool_name: str, count: int, canonical_arguments: str) -> str:
    return (
        "Repeated tool call detected:\n"
        f"- tool: {tool_name}\n"
        f"- consecutive_calls: {count}\n"
        f"- arguments: {canonical_arguments}\n"
        "The repeated calls are not making progress. Do not call this tool with "
        "these exact arguments again. Inspect the latest result and choose a "
        "different action, different arguments, or finish the task if enough "
        "evidence has been gathered."
    )


def _sort_json_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_sort_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _sort_json_value(value[key]) for key in sorted(value)}
    return value


def canonicalize(arguments: Any) -> str:
    return json.dumps(_sort_json_value(arguments), ensure_ascii=False, sort_keys=True)


def _wildcard_to_regex(pattern: str) -> re.Pattern[str]:
    escaped = pattern.replace("*", "\0")
    escaped = re.escape(escaped).replace("\0", ".*")
    return re.compile(rf"^{escaped}$")


def _preview_arguments(canonical: str, cap: int) -> str:
    if len(canonical) <= cap:
        return canonical
    return f"{canonical[:cap]}… (+{len(canonical) - cap} more chars)"


def validate_thresholds(values: tuple[int, ...]) -> tuple[int, ...]:
    if not values:
        raise ValueError("repeat-tool-reminder: `thresholds` must not be empty")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(
                f"repeat-tool-reminder: invalid threshold {value} — "
                "every threshold must be an integer >= 2"
            )
    if len(set(values)) != len(values):
        raise ValueError("repeat-tool-reminder: `thresholds` must not contain duplicates")
    return tuple(sorted(values))


@dataclass(slots=True)
class _Chain:
    key: str
    count: int


@dataclass(slots=True)
class RepeatToolGuard:
    """Per-session consecutive-repeat chains with threshold escalation."""

    thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    arguments_preview_chars: int = DEFAULT_ARGUMENTS_PREVIEW_CHARS
    _chains: dict[str, _Chain] = field(default_factory=dict, init=False)
    _include: tuple[re.Pattern[str], ...] = field(default=(), init=False)
    _exclude: tuple[re.Pattern[str], ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        self.thresholds = validate_thresholds(self.thresholds)
        if (
            isinstance(self.arguments_preview_chars, bool)
            or not isinstance(self.arguments_preview_chars, int)
            or self.arguments_preview_chars < 1
        ):
            raise ValueError(
                "repeat-tool-reminder: invalid argumentsPreviewChars "
                f"{self.arguments_preview_chars} — must be an integer >= 1"
            )
        self._include = tuple(_wildcard_to_regex(pattern) for pattern in self.include)
        self._exclude = tuple(_wildcard_to_regex(pattern) for pattern in self.exclude)

    def reset(self, session_id: str) -> None:
        self._chains.pop(session_id, None)

    def forget_session(self, session_id: str) -> None:
        self.reset(session_id)

    def _tracked(self, tool_name: str) -> bool:
        if self._include and not any(pattern.match(tool_name) for pattern in self._include):
            return False
        return not any(pattern.match(tool_name) for pattern in self._exclude)

    def observe(
        self,
        session_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str | None:
        """Advance the chain for one attempt; return the reminder text at a threshold."""

        if not self._tracked(tool_name):
            return None
        canonical = canonicalize(arguments)
        key = json.dumps([tool_name, canonical], ensure_ascii=False)
        chain = self._chains.get(session_id)
        count = chain.count + 1 if chain is not None and chain.key == key else 1
        self._chains[session_id] = _Chain(key=key, count=count)
        if count not in self.thresholds:
            return None
        if count == self.thresholds[0]:
            return GENTLE_REMINDER
        return detailed_reminder(
            tool_name, count, _preview_arguments(canonical, self.arguments_preview_chars)
        )


__all__ = [
    "DEFAULT_ARGUMENTS_PREVIEW_CHARS",
    "DEFAULT_THRESHOLDS",
    "GENTLE_REMINDER",
    "RepeatToolGuard",
    "canonicalize",
    "detailed_reminder",
    "validate_thresholds",
]
