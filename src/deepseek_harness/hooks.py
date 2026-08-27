"""Claude Code command-hook bridge mirroring the TS ``dsh-hook-protocol``.

Unmodified ``hooks.json`` command hooks run at the harness interception
points.  The protocol owns matcher semantics, outcome decoding, and the
most-restrictive merge; the bridge owns Claude payloads, environment,
substitution, and decision mapping.  Non-command hook types are parsed and
skipped with a diagnostic, never run.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .session import Session, SessionEvent
from .tools.registry import ToolContext, ToolResult

CLAUDE_EVENTS: tuple[str, ...] = (
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "Stop",
    "SubagentStart",
    "SubagentStop",
)
DEFAULT_HOOK_TIMEOUT_MS = 600_000
DEFAULT_STDERR_SUMMARY_MAX_CHARS = 500
BLOCKING_EXIT_CODE = 2
_CLAUDE_LITERAL = re.compile(r"^[A-Za-z0-9_|]+$")
_plugin_handler_counter = 0


class HookConfigError(ValueError):
    """A hook configuration is invalid (for example a broken regex matcher)."""


@dataclass(frozen=True, slots=True)
class CommandHook:
    command: str
    timeout_sec: float | None = None


@dataclass(frozen=True, slots=True)
class MatcherGroup:
    matcher: str | None
    hooks: tuple[CommandHook, ...]


@dataclass(slots=True)
class HookOutput:
    exit_code: int | None
    stderr: str
    stdout: str
    stop: bool | None = None
    stop_reason: str | None = None
    decision: str | None = None
    reason: str | None = None
    hook_event_name: str | None = None
    additional_context: str | None = None
    system_message: str | None = None
    updated_input: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class MergedHookOutcome:
    decision: str = "none"
    reason: str | None = None
    stop: bool = False
    stop_reason: str | None = None
    additional_context: tuple[str, ...] = ()
    system_messages: tuple[str, ...] = ()


def is_match_all(matcher: str | None) -> bool:
    return matcher is None or matcher == "" or matcher == "*"


def matcher_diagnostic(matcher: str | None) -> str | None:
    if is_match_all(matcher):
        return None
    assert matcher is not None
    if _CLAUDE_LITERAL.match(matcher):
        return None
    try:
        re.compile(matcher)
    except re.error:
        return f"invalid claude-code regex matcher {matcher!r}"
    return None


def matches_matcher(matcher: str | None, query: str) -> bool:
    """Claude literal patterns exact-match pipe alternatives; others are regex."""

    if is_match_all(matcher):
        return True
    assert matcher is not None
    if _CLAUDE_LITERAL.match(matcher):
        return query in matcher.split("|")
    try:
        return re.search(matcher, query) is not None
    except re.error:
        return False


def substitute_command(command: str, *, plugin_root: str | None, project_dir: str | None) -> str:
    if plugin_root is not None:
        command = command.replace("${CLAUDE_PLUGIN_ROOT}", plugin_root)
    if project_dir is not None:
        command = command.replace("${CLAUDE_PROJECT_DIR}", project_dir)
    return command


def parse_claude_code_config(
    raw: Any,
    *,
    plugin_root: str | None = None,
    project_dir: str | None = None,
) -> tuple[dict[str, list[MatcherGroup]], list[tuple[str, str]]]:
    """Parse a settings ``hooks`` value or bare event map.

    Returns the runnable per-event groups plus ``(event, type)`` pairs for
    skipped non-command hooks.  Malformed entries are ignored rather than
    failing boot; a matcher-bearing group with an invalid regex raises
    :class:`HookConfigError`.
    """

    config: dict[str, list[MatcherGroup]] = {}
    skipped: list[tuple[str, str]] = []
    root = raw if isinstance(raw, dict) else None
    hooks_map = root.get("hooks") if root is not None else None
    hooks_map = hooks_map if isinstance(hooks_map, dict) else root
    if hooks_map is None:
        return config, skipped
    for event in CLAUDE_EVENTS:
        raw_groups = hooks_map.get(event)
        if not isinstance(raw_groups, list):
            continue
        groups: list[MatcherGroup] = []
        for raw_group in raw_groups:
            if not isinstance(raw_group, dict) or not isinstance(raw_group.get("hooks"), list):
                continue
            commands: list[CommandHook] = []
            for raw_hook in raw_group["hooks"]:
                if not isinstance(raw_hook, dict):
                    continue
                hook_type = raw_hook.get("type")
                hook_type = hook_type if isinstance(hook_type, str) else "command"
                if hook_type != "command":
                    skipped.append((event, hook_type))
                    continue
                command = raw_hook.get("command")
                if not isinstance(command, str):
                    continue
                timeout = raw_hook.get("timeout")
                commands.append(
                    CommandHook(
                        command=substitute_command(
                            command, plugin_root=plugin_root, project_dir=project_dir
                        ),
                        timeout_sec=timeout if isinstance(timeout, (int, float)) else None,
                    )
                )
            if not commands:
                continue
            matcher = raw_group.get("matcher")
            matcher = matcher if isinstance(matcher, str) else None
            if event in {"UserPromptSubmit", "Stop"}:
                matcher = None
            diagnostic = matcher_diagnostic(matcher)
            if diagnostic is not None:
                raise HookConfigError(f"{diagnostic} on event {event!r}")
            groups.append(MatcherGroup(matcher=matcher, hooks=tuple(commands)))
        if groups:
            config[event] = groups
    return config, skipped


def _str(parsed: dict[str, Any], key: str) -> str | None:
    value = parsed.get(key)
    return value if isinstance(value, str) else None


def parse_hook_output(
    exit_code: int | None,
    stdout: str,
    stderr: str,
    expected_event_name: str | None = None,
) -> HookOutput:
    """Decode one hook run; a total function — malformed JSON stays plain stdout."""

    trimmed_err = stderr.strip()
    trimmed_out = stdout.strip()
    output = HookOutput(exit_code=exit_code, stderr=trimmed_err, stdout=trimmed_out)
    if exit_code == BLOCKING_EXIT_CODE:
        output.decision = "block"
        if trimmed_err:
            output.reason = trimmed_err
    if exit_code == 0 and trimmed_out.startswith("{"):
        try:
            parsed = json.loads(trimmed_out)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            _apply_structured(output, parsed, expected_event_name)
    return output


def _apply_structured(
    output: HookOutput,
    parsed: dict[str, Any],
    expected_event_name: str | None,
) -> None:
    if isinstance(parsed.get("continue"), bool):
        output.stop = parsed["continue"] is False
    stop_reason = _str(parsed, "stopReason")
    if stop_reason is not None:
        output.stop_reason = stop_reason
    system_message = _str(parsed, "systemMessage")
    if system_message is not None:
        output.system_message = system_message
    top_decision = _str(parsed, "decision")
    if top_decision in {"approve", "block"}:
        output.decision = top_decision
    top_reason = _str(parsed, "reason")
    if top_reason is not None:
        output.reason = top_reason
    specific = parsed.get("hookSpecificOutput")
    if not isinstance(specific, dict):
        return
    event_name = _str(specific, "hookEventName")
    if event_name is not None:
        output.hook_event_name = event_name
    if expected_event_name is not None and event_name != expected_event_name:
        return
    permission = _str(specific, "permissionDecision")
    if permission in {"allow", "deny", "ask"}:
        output.decision = permission
    permission_reason = _str(specific, "permissionDecisionReason")
    if permission_reason is not None:
        output.reason = permission_reason
    additional = _str(specific, "additionalContext")
    if additional is not None:
        output.additional_context = additional
    updated = specific.get("updatedInput")
    if isinstance(updated, dict):
        output.updated_input = updated


_DECISION_RANK = {"deny": 3, "block": 3, "ask": 2, "approve": 1, "allow": 1}


def merge_hook_outputs(outputs: Sequence[HookOutput]) -> MergedHookOutcome:
    """Fold matched hook outcomes with ``deny > ask > allow`` precedence."""

    max_rank = 0
    reasons_by_rank: dict[int, list[str]] = {}
    stop = False
    stop_reason: str | None = None
    contexts: list[str] = []
    system_messages: list[str] = []
    for out in outputs:
        rank = _DECISION_RANK.get(out.decision or "", 0)
        if rank > max_rank:
            max_rank = rank
        if rank in {2, 3} and out.reason:
            reasons_by_rank.setdefault(rank, []).append(out.reason)
        if out.stop and not stop:
            stop = True
            if out.stop_reason is not None:
                stop_reason = out.stop_reason
        if out.additional_context:
            contexts.append(out.additional_context)
        if out.system_message:
            system_messages.append(out.system_message)
    reasons = reasons_by_rank.get(max_rank, [])
    decision = {3: "deny", 2: "ask", 1: "allow"}.get(max_rank, "none")
    return MergedHookOutcome(
        decision=decision,
        reason="\n\n".join(reasons) if reasons else None,
        stop=stop,
        stop_reason=stop_reason,
        additional_context=tuple(contexts),
        system_messages=tuple(system_messages),
    )


async def run_hook(
    hook: CommandHook,
    payload: Any,
    *,
    cwd: str,
    env: dict[str, str] | None = None,
    default_timeout_ms: float = DEFAULT_HOOK_TIMEOUT_MS,
    expected_event_name: str | None = None,
) -> tuple[HookOutput, float]:
    """Run one command hook and decode its outcome; never raises."""

    started = time.monotonic()
    timeout_ms = (
        hook.timeout_sec * 1000 if hook.timeout_sec is not None else default_timeout_ms
    )
    stdin = json.dumps(payload, ensure_ascii=False) + "\n"
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    try:
        process = await asyncio.create_subprocess_shell(
            hook.command,
            cwd=cwd,
            env=process_env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        return parse_hook_output(None, "", str(exc), expected_event_name), 0.0

    async def communicate() -> tuple[bytes, bytes]:
        assert process.stdin is not None and process.stdout is not None
        stdout, stderr = await process.communicate(stdin.encode("utf-8"))
        return stdout, stderr

    try:
        stdout, stderr = await asyncio.wait_for(communicate(), timeout_ms / 1000)
    except TimeoutError:
        process.kill()
        await process.wait()
        message = f"hook timed out after {timeout_ms / 1000:g}s"
        return parse_hook_output(None, "", message, expected_event_name), 0.0
    except asyncio.CancelledError:
        process.kill()
        await process.wait()
        raise
    duration_ms = (time.monotonic() - started) * 1000
    returncode = process.returncode
    exit_code = returncode if returncode is not None and returncode >= 0 else None
    return (
        parse_hook_output(
            exit_code,
            stdout.decode("utf-8", "replace"),
            stderr.decode("utf-8", "replace"),
            expected_event_name,
        ),
        duration_ms,
    )


def append_hook_invoked(
    session: Session,
    *,
    turn: int,
    point: str,
    handler_id: str,
    matcher: str | None = None,
) -> SessionEvent:
    data: dict[str, Any] = {
        "turn": turn,
        "point": point,
        "dialect": "claude-code",
        "handlerId": handler_id,
    }
    if matcher is not None:
        data["matcher"] = matcher
    return session.append("hook/invoked", data)


def append_hook_result(
    session: Session,
    *,
    turn: int,
    point: str,
    handler_id: str,
    output: HookOutput,
    duration_ms: float,
    stderr_summary_max_chars: int = DEFAULT_STDERR_SUMMARY_MAX_CHARS,
) -> SessionEvent:
    decision = output.decision or ("stop" if output.stop else "pass")
    data: dict[str, Any] = {
        "turn": turn,
        "point": point,
        "handlerId": handler_id,
        "decision": decision,
        "durationMs": round(duration_ms),
    }
    if output.exit_code is not None:
        data["exitCode"] = output.exit_code
    if output.stderr:
        data["stderrSummary"] = output.stderr[:stderr_summary_max_chars]
    return session.append("hook/result", data)


def blocks_to_text(content: Any) -> str:
    if not isinstance(content, Sequence):
        return ""
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "".join(part for part in parts if isinstance(part, str))


@dataclass(slots=True)
class HookBridge:
    """Loads one Claude Code hook config and runs its points for one service."""

    config_path: str
    plugin_root: str | None = None
    project_dir: str | None = None
    default_timeout_ms: float = DEFAULT_HOOK_TIMEOUT_MS
    stderr_summary_max_chars: int = DEFAULT_STDERR_SUMMARY_MAX_CHARS
    config: dict[str, list[MatcherGroup]] = field(default_factory=dict)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        try:
            with open(self.config_path, encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, ValueError) as exc:
            raise HookConfigError(
                f'could not load hook config "{self.config_path}": {exc}'
            ) from exc
        self.config, self.skipped = parse_claude_code_config(
            raw, plugin_root=self.plugin_root, project_dir=self.project_dir
        )

    @property
    def loaded(self) -> bool:
        return bool(self.config)

    async def run_point(
        self,
        point: str,
        match_query: str,
        payload: dict[str, Any],
        *,
        session: Session | None = None,
        cwd: str,
        turn: int | None = None,
    ) -> MergedHookOutcome:
        groups = self.config.get(point, [])
        outputs: list[HookOutput] = []
        env = {"CLAUDE_PROJECT_DIR": self.project_dir} if self.project_dir else None
        for group in groups:
            if not matches_matcher(group.matcher, match_query):
                continue
            for hook in group.hooks:
                global _plugin_handler_counter
                _plugin_handler_counter += 1
                handler_id = f"claude-code:{point}:{_plugin_handler_counter}"
                record = session is not None and turn is not None
                if record:
                    assert session is not None and turn is not None
                    append_hook_invoked(
                        session,
                        turn=turn,
                        point=point,
                        handler_id=handler_id,
                        matcher=group.matcher,
                    )
                output, duration_ms = await run_hook(
                    hook,
                    payload,
                    cwd=cwd,
                    env=env,
                    default_timeout_ms=self.default_timeout_ms,
                    expected_event_name=point,
                )
                outputs.append(output)
                if record:
                    assert session is not None and turn is not None
                    append_hook_result(
                        session,
                        turn=turn,
                        point=point,
                        handler_id=handler_id,
                        output=output,
                        duration_ms=duration_ms,
                        stderr_summary_max_chars=self.stderr_summary_max_chars,
                    )
        return merge_hook_outputs(outputs)


async def pre_tool_use_hook(
    bridge: HookBridge,
    session: Session,
    name: str,
    arguments: dict[str, Any],
    context: ToolContext,
) -> str | None:
    """The ``ToolRegistry.pre_execute`` seam: return a deny reason or None."""

    if not bridge.loaded:
        return None
    payload = {
        "session_id": session.id,
        "transcript_path": "",
        "cwd": session.header.cwd or os.getcwd(),
        "hook_event_name": "PreToolUse",
        "tool_name": name,
        "tool_input": arguments,
        "tool_use_id": context.call_id or "",
    }
    merged = await bridge.run_point(
        "PreToolUse",
        name,
        payload,
        session=session,
        cwd=session.header.cwd or os.getcwd(),
        turn=_last_turn(session),
    )
    if merged.decision in {"deny", "ask"}:
        return merged.reason or f"blocked by PreToolUse hook ({merged.decision})"
    return None


async def post_tool_use_hook(
    bridge: HookBridge,
    session: Session,
    name: str,
    arguments: dict[str, Any],
    context: ToolContext,
    result: ToolResult,
) -> ToolResult:
    """The ``ToolRegistry.post_execute`` seam: map PostToolUse decisions."""

    if not bridge.loaded:
        return result
    payload = {
        "session_id": session.id,
        "transcript_path": "",
        "cwd": session.header.cwd or os.getcwd(),
        "hook_event_name": "PostToolUse",
        "tool_name": name,
        "tool_input": arguments,
        "tool_use_id": context.call_id or "",
        "tool_response": result.text,
    }
    merged = await bridge.run_point(
        "PostToolUse",
        name,
        payload,
        session=session,
        cwd=session.header.cwd or os.getcwd(),
        turn=_last_turn(session),
    )
    text = result.text
    if merged.additional_context:
        text = "\n\n".join([text, *merged.additional_context])
    if merged.decision == "deny":
        feedback = merged.reason or "blocked by PostToolUse hook"
        if merged.additional_context:
            feedback = "\n\n".join([feedback, *merged.additional_context])
        return ToolResult(feedback, is_error=True)
    if text != result.text:
        return ToolResult(text, is_error=result.is_error)
    return result


def _last_turn(session: Session) -> int:
    for event in reversed(session.events):
        if event.type == "turn/start":
            turn = event.data.get("turn")
            if isinstance(turn, int):
                return turn
    return 0


__all__ = [
    "BLOCKING_EXIT_CODE",
    "CLAUDE_EVENTS",
    "DEFAULT_HOOK_TIMEOUT_MS",
    "DEFAULT_STDERR_SUMMARY_MAX_CHARS",
    "CommandHook",
    "HookBridge",
    "HookConfigError",
    "HookOutput",
    "MergedHookOutcome",
    "MatcherGroup",
    "append_hook_invoked",
    "append_hook_result",
    "blocks_to_text",
    "matches_matcher",
    "merge_hook_outputs",
    "parse_claude_code_config",
    "parse_hook_output",
    "post_tool_use_hook",
    "pre_tool_use_hook",
    "run_hook",
    "substitute_command",
]
