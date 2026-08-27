"""Opt-in request clock context mirroring the TS ``dsh-time-context``.

Eligible model steps gain a durable time reading.  The TS plugin runs as a
prepended pre-step listener with browser-zone resolution; this port renders
the same text shape and refresh throttling.  Injection throttling reads the
durable ``user/message`` records this module stamps, so a resumed session
keeps its cadence.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .session import Session

_PLUGIN_NAME = "time-context"


def format_timestamp(moment: datetime, zone: str) -> str:
    try:
        localized = moment.astimezone(ZoneInfo(zone))
    except (ZoneInfoNotFoundError, KeyError, ValueError, OSError):
        localized = moment
    return localized.strftime("%Y-%m-%d %H:%M:%S %Z").strip()


def _event_ms(event_time: int | None) -> int | None:
    return event_time if isinstance(event_time, int) and event_time > 0 else None


def _last_message_time(session: Session) -> int | None:
    for event in reversed(session.events):
        if event.type in {"user/message", "assistant/message", "tool/result"}:
            return _event_ms(event.time)
    return None


def _last_injection_ms(session: Session) -> int | None:
    for event in reversed(session.events):
        if event.type != "user/message":
            continue
        message = event.data.get("message")
        source = message.get("source") if isinstance(message, dict) else None
        if (
            isinstance(source, dict)
            and source.get("kind") == "plugin"
            and source.get("plugin") == _PLUGIN_NAME
        ):
            return _event_ms(event.time)
    return None


def open_turn(session: Session) -> int:
    for event in reversed(session.events):
        if event.type == "turn/start":
            turn = event.data.get("turn")
            if isinstance(turn, int):
                return turn
    return 0


def step_in_turn(session: Session, turn: int) -> int:
    started = False
    steps = 0
    for event in session.events:
        if event.type == "turn/start":
            started = event.data.get("turn") == turn
            continue
        if started and event.type == "step/start":
            steps += 1
    return steps + 1


def format_duration(elapsed_ms: int) -> str:
    seconds = max(0, elapsed_ms) // 1000
    days, seconds = divmod(seconds, 86_400)
    hours, seconds = divmod(seconds, 3_600)
    minutes, seconds = divmod(seconds, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


class TimeContextInjector:
    """Builds the durable time reading for eligible steps, throttled per session."""

    def __init__(
        self,
        *,
        time_zone: str | None = None,
        refresh_interval_seconds: float = 0.0,
    ) -> None:
        if refresh_interval_seconds < 0:
            raise ValueError("time-context: refresh interval must be non-negative")
        zone = time_zone
        if zone is None:
            local = datetime.now().astimezone().tzinfo
            zone = str(getattr(local, "key", local)) if local is not None else "UTC"
        try:
            ZoneInfo(zone)
        except (ZoneInfoNotFoundError, KeyError, ValueError, OSError) as exc:
            raise ValueError(f"time-context: invalid IANA timeZone {zone!r}") from exc
        self.time_zone = zone
        self.refresh_interval_seconds = refresh_interval_seconds

    def message_text(
        self,
        session: Session,
        *,
        now: datetime | None = None,
    ) -> str | None:
        """Return the time-context text for the upcoming step, or None when throttled."""

        now = now or datetime.now(UTC)
        now_ms = int(now.timestamp() * 1000)
        last = _last_injection_ms(session)
        if (
            last is not None
            and self.refresh_interval_seconds > 0
            and now_ms - last < self.refresh_interval_seconds * 1000
        ):
            return None
        turn = open_turn(session)
        step = step_in_turn(session, turn)
        if step <= 1:
            previous = _last_message_time(session)
            baseline = "model-visible message"
        else:
            previous = last
            baseline = "step context"
        elapsed = (
            "unavailable" if previous is None else format_duration(now_ms - previous)
        )
        return (
            f"Time sampled while preparing turn {turn}, step {step}: "
            f"{format_timestamp(now, self.time_zone)}\n"
            f"Elapsed since the preceding {baseline}: {elapsed}."
        )

    @staticmethod
    def source() -> dict[str, Any]:
        return {"kind": "plugin", "plugin": _PLUGIN_NAME}


__all__ = ["TimeContextInjector", "format_duration", "format_timestamp"]
