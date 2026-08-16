"""Application services shared by the FastAPI RPC and SSE carriers."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import uuid
import zipfile
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, cast

import httpx

from ..agent import Agent
from ..agent_presets import AgentPresetError, AgentPresetRegistry
from ..attachments import IMAGE_MEDIA_TYPES, AttachmentError, AttachmentStore, ImageAttachment
from ..dynamic_cordis import DynamicCordisService, install_dynamic_tools
from ..errors import HarnessError
from ..goals import GoalError, GoalManager
from ..jobs import JobHandle, JobOutcome, JobRegistry
from ..llm import DeepSeekAdapter, LlmCallConfig
from ..llm.adapter import LlmAdapter
from ..message_feedback import MessageFeedbackManager
from ..models import ImageContent, Message, TextContent
from ..plans import fold as fold_plan_mode
from ..plans import has_open_turn
from ..session import JsonlSessionStore, Session, SessionEvent
from ..settings import (
    CredentialError,
    CredentialStore,
    SettingsConflict,
    SettingsNotFound,
    SettingsRegistry,
)
from ..skills import SkillRegistry
from ..todos import TodoError, TodoManager
from ..tools import PermissionMode, WorkspacePolicy, install_builtin_tools
from ..tools.registry import ToolContext, ToolDefinition, ToolRegistry, ToolResult
from ..workflow import (
    WorkflowChildResult,
    WorkflowError,
    matches_object_schema,
    run_workflow,
)
from ..workspace import (
    WorkspaceInvalidPath,
    WorkspaceMoveInvalid,
    WorkspaceNameConflict,
    WorkspaceNotFound,
    WorkspaceRegistry,
)

JsonObject = dict[str, Any]
Frame = JsonObject
AdapterFactory = Callable[[str], LlmAdapter]

SUBAGENT_DESCRIPTOR_VERSION = 2
MAX_SUBAGENT_DEPTH = 3
DEFAULT_RALPH_MAX_ROUNDS = 256
MAX_RALPH_HANDOFF_CHARS = 16_384
MAX_RALPH_RESULT_CHARS = 16_384


class ApiFault(HarnessError):
    """A business error that is serialized into the DSH RpcResult shape."""

    def __init__(self, code: str, message: str, details: JsonObject | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass(slots=True)
class SessionHandle:
    session: Session
    agent: Agent
    disposers: list[Callable[[], None]]
    task: asyncio.Task[Any] | None = None
    queue: list[QueueItem] = field(default_factory=list)


@dataclass(slots=True)
class QueueItem:
    message: Message
    placement: Literal["queued", "steering", "context"] = "queued"

    def to_dict(self) -> JsonObject:
        return {
            "id": self.message.id,
            "placement": self.placement,
            "message": self.message.to_dict(),
        }


ApprovalOutcome = Literal["allowed-once", "rejected", "cancelled", "unavailable"]


@dataclass(slots=True)
class PendingApproval:
    rpc_id: str
    session_id: str
    approval_id: str
    tool_name: str
    call_id: str | None
    reason: str | None
    future: asyncio.Future[ApprovalOutcome]

    def frame(self) -> Frame:
        return {
            "type": "approval/requested",
            "sessionId": self.session_id,
            "approvalId": self.approval_id,
            "toolName": self.tool_name,
            **({"callId": self.call_id} if self.call_id is not None else {}),
            **({"reason": self.reason} if self.reason is not None else {}),
        }


@dataclass(slots=True)
class PendingQuestion:
    rpc_id: str
    session_id: str
    questions: list[JsonObject]
    future: asyncio.Future[JsonObject]

    def frame(self) -> Frame:
        return {
            "type": "question/requested",
            "sessionId": self.session_id,
            "questions": self.questions,
        }


class HarnessService:
    """Own live agents, persistence, and transport subscribers.

    The service is deliberately independent from FastAPI.  It is therefore usable
    by a future WebSocket carrier, a desktop host, or direct integration tests
    without changing the agent and session seams.
    """

    def __init__(
        self,
        session_root: str | os.PathLike[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        model: str = "deepseek-v4-flash",
        permission_mode: PermissionMode = PermissionMode.WORKSPACE_WRITE,
        adapter_factory: AdapterFactory | None = None,
    ) -> None:
        self.store = JsonlSessionStore(session_root)
        state_root = self.store.root
        self.cwd = Path(cwd or Path.cwd()).expanduser().resolve()
        self.model = model
        self.permission_mode = permission_mode
        self.attachments = AttachmentStore(state_root)
        self.presets = AgentPresetRegistry(state_root)
        self.goals = GoalManager()
        self.todos = TodoManager(allow_parallel_in_progress=True)
        self.message_feedback = MessageFeedbackManager(state_root, max_note_bytes=8192)
        self.settings = SettingsRegistry(state_root)
        self.settings.register(
            "ui-onboarding",
            schema={
                "type": "object",
                "properties": {"welcomeNoticeVersion": {"type": "string"}},
            },
            base={},
        )
        self.settings.register(
            "agent-default-model",
            schema={
                "type": "object",
                "properties": {
                    "provider": {"type": "string"},
                    "model": {"type": "string"},
                },
            },
            base={"provider": "deepseek-official", "model": model},
        )
        self.settings.register(
            "llm-deepseek",
            schema={"type": "object"},
            base={
                "apiKeyEnv": "DEEPSEEK_API_KEY",
                "baseURL": os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
                "thinking": "enabled",
                "reasoningEffort": "high",
                "maxTokens": 256000,
                "defaultContextWindow": 1_000_000,
            },
            secrets=({"path": ["apiKey"]},),
        )
        self.settings.register(
            "agent-presets",
            schema={"type": "object", "properties": {"default": {"type": "string"}}},
            base={},
        )
        self.settings.register(
            "ui-theme",
            schema={
                "type": "object",
                "properties": {"preference": {"enum": ["light", "dark", "system"]}},
            },
            base={"preference": "system"},
        )
        self.settings.register(
            "locale",
            schema={
                "type": "object",
                "properties": {"preference": {"enum": ["zh", "en"]}},
            },
            base={},
        )
        self.settings.register(
            "ui-conversation",
            schema={
                "type": "object",
                "properties": {"busyEnter": {"enum": ["queue", "steer"]}},
            },
            base={"busyEnter": "queue"},
        )
        self.settings.register(
            "permission",
            schema={
                "type": "object",
                "properties": {
                    "defaultPreset": {
                        "enum": ["read-only", "workspace-write", "danger-full-access"]
                    }
                },
            },
            base={"defaultPreset": permission_mode.value},
        )
        self.settings.register("shell", schema={"type": "object"}, base={})
        self.settings.register("agent-loop", schema={"type": "object"}, base={})
        self.settings.register("web-search-deepseek", schema={"type": "object"}, base={})
        self.credentials = CredentialStore(state_root)
        self.workspaces = WorkspaceRegistry(state_root)
        self.skills = SkillRegistry()
        self._adapter_factory = adapter_factory or self._default_adapter
        self._handles: dict[str, SessionHandle] = {}
        self._tool_registries: dict[str, ToolRegistry] = {}
        self._goal_activation: dict[str, Literal["armed", "disarmed"]] = {}
        self._mux_subscribers: set[asyncio.Queue[Frame]] = set()
        self._host_subscribers: set[asyncio.Queue[Frame]] = set()
        self.jobs = JobRegistry(on_changed=self._jobs_changed)
        self.dynamic = DynamicCordisService(
            tool_registries=self._tool_registries,
            remote_event=self._publish_remote_event,
        )
        self._lock = asyncio.Lock()
        self._queue_lock = asyncio.Lock()
        self._pending_approvals: dict[str, PendingApproval] = {}
        self._pending_questions: dict[str, PendingQuestion] = {}
        self._disposed = False

    async def create_session(
        self,
        *,
        session_id: str | None = None,
        cwd: str | None = None,
        parent_session: str | None = None,
        origin: str | None = None,
        agent_preset: str | None = None,
        model_selection: JsonObject | None = None,
    ) -> SessionHandle:
        self._ensure_open()
        resolved_cwd = Path(cwd or self.cwd).expanduser().resolve()
        if not resolved_cwd.is_dir():
            raise ApiFault(
                "workspace-invalid-path", f"workspace is not a directory: {resolved_cwd}"
            )
        if agent_preset is not None:
            await self._require_preset(agent_preset)
        requested_id = session_id or f"session-{uuid.uuid4().hex}"
        async with self._lock:
            existing = self._handles.get(requested_id)
            if existing is not None:
                self._check_session_cwd(existing.session, resolved_cwd)
                self._check_session_preset(existing.session, agent_preset)
                return existing
            try:
                session = await self.store.load(requested_id)
            except FileNotFoundError:
                session = await self.store.create(
                    requested_id,
                    cwd=str(resolved_cwd),
                    parent_session=parent_session,
                    origin=origin,
                    agent_preset=agent_preset,
                    model_selection=model_selection,
                )
            else:
                self._check_session_cwd(session, resolved_cwd)
                self._check_session_preset(session, agent_preset)
            handle = self._attach(session)
        self._publish_host(
            {
                "type": "host/session-added",
                "sessionId": handle.session.id,
                "blank": self._is_blank(handle.session),
                "cwd": handle.session.header.cwd,
                **(
                    {"parentSessionId": handle.session.header.parent_session}
                    if handle.session.header.parent_session
                    else {}
                ),
                **({"origin": "subagent"} if handle.session.header.origin == "subagent" else {}),
                **(
                    {"agentPreset": handle.session.header.agent_preset}
                    if handle.session.header.agent_preset
                    else {}
                ),
            }
        )
        return handle

    async def get_session(self, session_id: str) -> SessionHandle:
        self._ensure_open()
        async with self._lock:
            handle = self._handles.get(session_id)
            if handle is not None:
                return handle
            try:
                session = await self.store.load(session_id)
            except FileNotFoundError as exc:
                raise ApiFault(
                    "session-not-found",
                    f"session does not exist: {session_id}",
                    {"sessionId": session_id},
                ) from exc
            return self._attach(session)

    async def list_sessions(self) -> list[JsonObject]:
        self._ensure_open()
        _, archived = await self.workspaces.list()
        archived_ids = set(archived)
        summaries: list[JsonObject] = []
        for session_id in await self.store.list_ids():
            if session_id in archived_ids:
                continue
            handle = await self.get_session(session_id)
            session = handle.session
            updated_at = max([session.header.created_at, *(event.time for event in session.events)])
            summary: JsonObject = {
                "sessionId": session.id,
                "updatedAt": updated_at,
                "running": handle.agent.status == "running",
                "blank": self._is_blank(session),
            }
            if session.header.parent_session:
                summary["parentSessionId"] = session.header.parent_session
            if session.header.origin == "subagent":
                summary["origin"] = "subagent"
            if session.header.cwd:
                summary["cwd"] = session.header.cwd
            if session.header.agent_preset:
                summary["agentPreset"] = session.header.agent_preset
            summaries.append(summary)
        summaries.sort(key=lambda item: int(item["updatedAt"]), reverse=True)
        return summaries

    async def history(
        self,
        session_id: str,
        *,
        before_seq: int | None = None,
        max_messages: int | None = None,
    ) -> JsonObject:
        handle = await self.get_session(session_id)
        events = list(handle.session.events)
        if before_seq is not None:
            events = [event for event in events if event.seq < before_seq]
        if max_messages is not None and max_messages > 0:
            groups = self._message_groups(events)
            if len(groups) > max_messages:
                events = [event for group in groups[-max_messages:] for event in group]
        result: JsonObject = {
            "events": [{"event": event.to_dict()} for event in events],
            "hasMore": bool(events and events[0].seq > 0),
        }
        if before_seq is None:
            result["projections"] = {
                "asOfSeq": handle.session.seq - 1,
                "values": self._projection_values(handle.session),
            }
        return result

    async def search(self, query: str) -> JsonObject:
        needle = query.casefold().strip()
        if not needle:
            return {"items": [], "hasMore": False}
        matches: list[JsonObject] = []
        for summary in await self.list_sessions():
            handle = await self.get_session(str(summary["sessionId"]))
            for message in handle.session.derive_messages():
                if needle in message.text.casefold():
                    matches.append({"sessionId": handle.session.id, "snippet": message.text[:240]})
                    break
            if len(matches) >= 20:
                break
        return {"items": matches, "hasMore": False}

    async def prompt(
        self,
        session_id: str,
        content: list[JsonObject],
        *,
        mode: Literal["queue", "steer"] = "queue",
        client_time_zone: str | None = None,
        allow_subagent: bool = False,
        include_message_id: bool = False,
    ) -> JsonObject:
        handle = await self.get_session(session_id)
        if handle.session.header.parent_session and not allow_subagent:
            raise ApiFault(
                "agent-busy",
                "session-backed subagents must use subagent.prompt",
                {"reason": "subagent"},
            )
        message = await self._build_message(content, client_time_zone=client_time_zone)
        command_line = self._command_line(message)
        if command_line is not None:
            return await self._execute_command(handle, command_line)
        self._register_message_attachments(handle.session, message)
        item = QueueItem(message, "steering" if mode == "steer" else "queued")
        async with self._queue_lock:
            if mode == "steer":
                handle.queue.insert(0, item)
            else:
                handle.queue.append(item)
            self._publish_queue(handle)
            if handle.task is None or handle.task.done():
                handle.task = asyncio.create_task(self._run_queue(handle))
        return {"accepted": True, **({"messageId": message.id} if include_message_id else {})}

    async def cancel(self, session_id: str) -> JsonObject:
        handle = await self.get_session(session_id)
        task = handle.task
        if task is not None and not task.done():
            task.cancel()
        return {"accepted": True}

    @staticmethod
    def _command_line(message: Message) -> str | None:
        if len(message.content) != 1 or not isinstance(message.content[0], TextContent):
            return None
        text = message.content[0].text
        return text if text.lstrip().startswith("/") else None

    async def _execute_command(self, handle: SessionHandle, line: str) -> JsonObject:
        """Execute the small host command surface needed by the shared UI."""

        body = line.lstrip()[1:]
        name, separator, raw_args = body.partition(" ")
        if name != "plan":
            raise ApiFault("unknown-command", f"unknown command: /{name}")
        args = raw_args if separator else ""
        target = args.strip() != "off"
        command_id = f"cmd-{uuid.uuid4().hex}"
        run = handle.session.append(
            "command/run",
            {
                "commandId": command_id,
                "name": "plan",
                "args": args,
                "source": {"kind": "user"},
            },
        )
        self._publish_event(handle.session.id, run)
        mode_event: SessionEvent | None = None
        if not has_open_turn(handle.session) and fold_plan_mode(handle.session).active != target:
            mode_event = handle.session.append("plan/mode", {"active": target})
            self._publish_event(handle.session.id, mode_event)
        if target:
            text = (
                "Entering plan mode (applies from the next step). Use /plan off to leave."
                if mode_event is None
                else "Plan mode on. Use /plan off to leave."
            )
        else:
            text = (
                "Leaving plan mode (applies from the next step)."
                if mode_event is None
                else "Plan mode off."
            )
        done_data: JsonObject = {
            "commandId": command_id,
            "kind": "success",
            "text": text,
        }
        if mode_event is not None:
            done_data["sourceEventSeq"] = mode_event.seq
        done = handle.session.append("command/done", done_data)
        self._publish_event(handle.session.id, done)
        await self.store.save(handle.session)
        if target and args.strip() and args.strip() != "off":
            await self.prompt(
                handle.session.id,
                [{"type": "text", "text": args.strip()}],
            )
        return {"accepted": True, "command": {"kind": "success", "text": text}}

    async def request_approval(
        self,
        session_id: str,
        tool_name: str,
        *,
        approval_id: str | None = None,
        call_id: str | None = None,
        reason: str | None = None,
    ) -> ApprovalOutcome:
        """Suspend an integration-owned action until the browser responds."""

        handle = await self.get_session(session_id)
        if not tool_name.strip():
            raise ApiFault("bad-request", "approval tool name cannot be empty")
        resolved_id = approval_id or f"approval-{uuid.uuid4().hex}"
        rpc_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ApprovalOutcome] = loop.create_future()
        pending = PendingApproval(
            rpc_id,
            session_id,
            resolved_id,
            tool_name,
            call_id,
            reason,
            future,
        )
        asked: JsonObject = {
            "id": resolved_id,
            "toolName": tool_name,
            **({"callId": call_id} if call_id is not None else {}),
        }
        if reason is not None:
            asked["reason"] = reason
        event = handle.session.append("approval/asked", asked)
        await self.store.save(handle.session)
        self._publish_event(session_id, event)
        self._pending_approvals[rpc_id] = pending
        self._publish_mux(pending.frame())
        try:
            return await future
        except asyncio.CancelledError:
            await self._finish_approval(rpc_id, "cancelled")
            raise
        finally:
            if rpc_id in self._pending_approvals:
                await self._finish_approval(rpc_id, "cancelled")

    async def request_question(
        self,
        session_id: str,
        questions: list[JsonObject],
    ) -> JsonObject:
        """Suspend an integration-owned question batch until it is answered."""

        await self.get_session(session_id)
        if not questions:
            raise ApiFault("bad-request", "question list cannot be empty")
        rpc_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future: asyncio.Future[JsonObject] = loop.create_future()
        pending = PendingQuestion(rpc_id, session_id, questions, future)
        self._pending_questions[rpc_id] = pending
        self._publish_mux(pending.frame())
        try:
            return await future
        except asyncio.CancelledError:
            await self._finish_question(rpc_id, "cancelled")
            raise
        finally:
            if rpc_id in self._pending_questions:
                await self._finish_question(rpc_id, "cancelled")

    async def update_queue(
        self,
        session_id: str,
        item_id: str,
        action: JsonObject,
    ) -> JsonObject:
        handle = await self.get_session(session_id)
        async with self._queue_lock:
            index = next(
                (index for index, item in enumerate(handle.queue) if item.message.id == item_id),
                None,
            )
            if index is None:
                raise ApiFault(
                    "queue-item-not-found",
                    f"queued message does not exist: {item_id}",
                    {"itemId": item_id},
                )
            kind = action.get("kind")
            if kind == "remove":
                handle.queue.pop(index)
            elif kind == "steer":
                item = handle.queue.pop(index)
                item.placement = "steering"
                handle.queue.insert(0, item)
            elif kind == "edit":
                content = action.get("content")
                if not isinstance(content, list):
                    raise ApiFault("bad-request", "queue edit content must be an array")
                replacement = await self._build_message(content)
                old = handle.queue[index]
                self._register_message_attachments(handle.session, replacement)
                handle.queue[index] = QueueItem(
                    Message("user", replacement.content, replacement.source, old.message.id),
                    old.placement,
                )
            else:
                raise ApiFault("bad-request", "queue action kind is invalid")
            self._publish_queue(handle)
        return {"accepted": True}

    async def attachment(self, session_id: str, attachment_id: str) -> JsonObject:
        handle = await self.get_session(session_id)
        ref = self._referenced_attachment(handle.session, attachment_id)
        if ref is None:
            raise ApiFault(
                "attachment-error",
                "image is not referenced by this session",
                {"reason": "ATTACHMENT_NOT_REFERENCED"},
            )
        try:
            stored = self.attachments.read(ref)
        except AttachmentError as exc:
            raise ApiFault("attachment-error", str(exc), {"reason": exc.code}) from exc
        return {
            "attachment": stored.ref.to_dict(),
            "data": base64.b64encode(stored.data).decode("ascii"),
        }

    async def export_zip(self, session_id: str, *, include_descendants: bool = False) -> bytes:
        """Build the host-only session-log download archive.

        The JSONL bytes are read from the durable artifact after flushing any
        currently attached session. This keeps exported logs faithful to the
        persistence format and lets the archive carry the image objects named
        by its own event records.
        """

        root = await self.get_session(session_id)
        artifacts: list[tuple[str, bytes]] = [
            ("session.jsonl", await self._export_raw_artifact(root.session))
        ]
        references: dict[str, ImageAttachment] = {}
        self._collect_export_attachment_refs(artifacts[0][1], references)

        if include_descendants:
            pending = [root.session.id]
            seen = {root.session.id}
            while pending:
                parent_id = pending.pop(0)
                for cold_child in await self._child_sessions(parent_id):
                    if cold_child.id in seen:
                        continue
                    seen.add(cold_child.id)
                    live_child = self._handles.get(cold_child.id)
                    child = live_child.session if live_child is not None else cold_child
                    raw = await self._export_raw_artifact(child)
                    artifacts.append(
                        (
                            f"subagents/{self._safe_archive_segment(child.id)}/session.jsonl",
                            raw,
                        )
                    )
                    self._collect_export_attachment_refs(raw, references)
                    pending.append(child.id)

        archive = io.BytesIO()
        with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as output:
            for path, content in artifacts:
                output.writestr(path, content)
            for reference in references.values():
                try:
                    stored = self.attachments.read(reference)
                except AttachmentError as exc:
                    raise ApiFault("attachment-error", str(exc), {"reason": exc.code}) from exc
                extension = {
                    "image/png": "png",
                    "image/jpeg": "jpg",
                    "image/webp": "webp",
                    "image/gif": "gif",
                }.get(reference.media_type)
                if extension is None:
                    raise ApiFault(
                        "attachment-error",
                        "attachment media type is not exportable",
                        {"reason": "INVALID_IMAGE"},
                    )
                output.writestr(f"media/{reference.attachment_id}.{extension}", stored.data)
        return archive.getvalue()

    async def _export_raw_artifact(self, session: Session) -> bytes:
        live = self._handles.get(session.id)
        if live is not None:
            await self.store.save(live.session)
        path = self.store.path_for(session.id)
        try:
            return await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError as exc:
            raise ApiFault(
                "session-not-found",
                f"session does not exist: {session.id}",
                {"sessionId": session.id},
            ) from exc

    @classmethod
    def _collect_export_attachment_refs(
        cls,
        raw: bytes,
        references: dict[str, ImageAttachment],
    ) -> None:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                if value.get("type") == "image":
                    attachment = value.get("attachment")
                    if isinstance(attachment, dict):
                        attachment_id = attachment.get("attachmentId")
                        media_type = attachment.get("mediaType")
                        if (
                            isinstance(attachment_id, str)
                            and isinstance(media_type, str)
                            and all(
                                isinstance(attachment.get(key), int)
                                for key in ("bytes", "width", "height")
                            )
                        ):
                            references[attachment_id] = ImageAttachment(
                                attachment_id,
                                media_type,
                                attachment["bytes"],
                                attachment["width"],
                                attachment["height"],
                                attachment.get("name")
                                if isinstance(attachment.get("name"), str)
                                else None,
                            )
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        for line in text.splitlines():
            if not line:
                continue
            try:
                visit(json.loads(line))
            except (TypeError, ValueError):
                continue

    @staticmethod
    def _safe_archive_segment(value: str) -> str:
        return "".join(
            char if char.isascii() and (char.isalnum() or char in "_-") else "_" for char in value
        )

    @classmethod
    def export_filename(cls, session_id: str) -> str:
        return f"dsh-session-{cls._safe_archive_segment(session_id)}.zip"

    async def fork(self, session_id: str, at_seq: int | None = None) -> JsonObject:
        source_handle = await self.get_session(session_id)
        source = source_handle.session
        end_seq = self._fork_end_seq(source, at_seq)
        prefix = list(source.events[: end_seq + 1])
        child_id = f"session-{uuid.uuid4().hex}"
        header = replace(
            Session.header_for(
                child_id,
                cwd=source.header.cwd,
                parent_session=source.id,
                model_selection=source.header.model_selection,
                agent_preset=source.header.agent_preset,
            ),
            seed_length=len(prefix),
        )
        child = Session(child_id, header=header, events=prefix)
        await self.store.save(child)
        async with self._lock:
            child_handle = self._attach(child)
        self._publish_host(
            {
                "type": "host/session-added",
                "sessionId": child_id,
                "blank": self._is_blank(child),
                "cwd": child.header.cwd,
                "parentSessionId": source.id,
            }
        )
        owner = await self.workspaces.owns_session(source.id)
        if owner is not None:
            workspace = await self.workspaces.attach_session(owner.workspace_id, child_id)
            self._publish_host({"type": "host/workspace-changed", "workspace": workspace.to_dict()})
        return {"sessionId": child_handle.session.id}

    async def stream(self, kind: str) -> AsyncIterator[Frame]:
        if kind not in {"mux", "host"}:
            raise ValueError(f"unknown event stream: {kind}")
        queue: asyncio.Queue[Frame] = asyncio.Queue()
        subscribers = self._mux_subscribers if kind == "mux" else self._host_subscribers
        subscribers.add(queue)
        try:
            if kind == "mux":
                for summary in await self.list_sessions():
                    session_id = str(summary["sessionId"])
                    handle = self._handles[session_id]
                    yield {
                        "type": "session/subscribed",
                        "sessionId": session_id,
                        "lastSeq": handle.session.seq - 1,
                    }
                    yield {
                        "type": "session/queue",
                        "sessionId": session_id,
                        "items": [item.to_dict() for item in handle.queue],
                    }
                    jobs = self.jobs.list(session_id)
                    if jobs:
                        yield {
                            "type": "session/jobs",
                            "sessionId": session_id,
                            "jobs": [job.to_dict() for job in jobs],
                        }
                    yield {
                        "type": "session/projection",
                        "sessionId": session_id,
                        "key": "goal",
                        "value": self.goals.fold(handle.session).projection(),
                        "seq": max(0, handle.session.seq - 1),
                    }
                    yield {
                        "type": "session/projection",
                        "sessionId": session_id,
                        "key": "plan",
                        "value": fold_plan_mode(handle.session).to_dict(),
                        "seq": max(0, handle.session.seq - 1),
                    }
                    yield {
                        "type": "session/projection",
                        "sessionId": session_id,
                        "key": "todos",
                        "value": self.todos.fold(handle.session),
                        "seq": max(0, handle.session.seq - 1),
                    }
                    yield {
                        "type": "session/projection",
                        "sessionId": session_id,
                        "key": "imageLimits",
                        "value": self._image_limits(),
                        "seq": max(0, handle.session.seq - 1),
                    }
                    for pending in tuple(self._pending_approvals.values()):
                        if pending.session_id == session_id:
                            yield pending.frame()
                    for pending in tuple(self._pending_questions.values()):
                        if pending.session_id == session_id:
                            yield pending.frame()
            while True:
                yield await queue.get()
        finally:
            subscribers.discard(queue)

    async def dispatch(self, method: str, payload: JsonObject) -> Any:
        """Dispatch one wire method while keeping carrier concerns outside."""

        if method == "session.list":
            if "cursor" in payload and not isinstance(payload["cursor"], str):
                raise ApiFault("bad-request", "cursor must be a string")
            return {"items": await self.list_sessions()}
        if method == "session.search":
            query_value = payload.get("query")
            if not isinstance(query_value, str):
                raise ApiFault("bad-request", "query must be a string")
            query = query_value.strip()
            if not query or "\x00" in query or len(query) > 500:
                raise ApiFault(
                    "bad-request",
                    "search query must be non-empty, at most 500 characters, and contain no NUL",
                )
            return await self.search(query)
        if method == "session.create":
            workspace_id = self._optional_payload_string(payload, "workspaceId")
            requested_cwd = self._optional_payload_string(payload, "cwd")
            if workspace_id is not None and requested_cwd is not None:
                raise ApiFault("bad-request", "session.create accepts workspaceId or cwd, not both")
            requested_preset = self._optional_payload_string(payload, "agentPreset")
            if requested_preset is None:
                configured_preset = self.settings.get_value_sync("agent-presets").get("default")
                if isinstance(configured_preset, str) and configured_preset:
                    requested_preset = configured_preset
            workspace_path: str | None = None
            if workspace_id is not None:
                try:
                    workspace_path = (await self.workspaces.get(workspace_id)).path
                except WorkspaceNotFound as exc:
                    raise ApiFault(
                        "workspace-not-found", str(exc), {"workspaceId": workspace_id}
                    ) from exc
            handle = await self.create_session(
                session_id=self._optional_payload_string(payload, "sessionId"),
                cwd=workspace_path or requested_cwd,
                agent_preset=requested_preset,
            )
            if workspace_id is not None:
                workspace = await self.workspaces.attach_session(workspace_id, handle.session.id)
                self._publish_host(
                    {"type": "host/workspace-changed", "workspace": workspace.to_dict()}
                )
            value: JsonObject = {"sessionId": handle.session.id}
            if handle.session.header.agent_preset:
                value["agentPreset"] = handle.session.header.agent_preset
            return value
        if method == "session.history":
            return await self.history(
                self._required_string(payload, "sessionId"),
                before_seq=self._optional_payload_int(payload, "beforeSeq", nonnegative=True),
                max_messages=self._optional_payload_int(payload, "maxMessages", positive=True),
            )
        if method == "session.prompt":
            content = payload.get("content")
            if not isinstance(content, list):
                raise ApiFault("bad-request", "session.prompt content must be an array")
            if "mode" not in payload:
                raise ApiFault("bad-request", "session.prompt mode is required")
            mode = payload.get("mode")
            if mode not in {"queue", "steer"}:
                raise ApiFault("bad-request", "session.prompt mode must be queue or steer")
            return await self.prompt(
                self._required_string(payload, "sessionId"),
                content,
                mode=mode,
                client_time_zone=self._optional_payload_string(payload, "clientTimeZone"),
            )
        if method == "session.updateQueue":
            action = payload.get("action")
            if not isinstance(action, dict):
                raise ApiFault("bad-request", "session.updateQueue action must be an object")
            return await self.update_queue(
                self._required_string(payload, "sessionId"),
                self._required_string(payload, "itemId"),
                action,
            )
        if method == "session.attachment":
            return await self.attachment(
                self._required_string(payload, "sessionId"),
                self._required_string(payload, "attachmentId"),
            )
        if method == "session.fork":
            return await self.fork(
                self._required_string(payload, "sessionId"),
                self._optional_payload_int(payload, "atSeq", nonnegative=True),
            )
        if method == "session.cancel":
            return await self.cancel(self._required_string(payload, "sessionId"))
        if method == "session.models":
            handle = await self.get_session(self._required_string(payload, "sessionId"))
            if handle.session.header.parent_session:
                raise ApiFault(
                    "agent-busy",
                    "subagent model selection is managed by its parent",
                    {"reason": "subagent"},
                )
            return self._models(handle)
        if method == "session.selectModel":
            handle = await self.get_session(self._required_string(payload, "sessionId"))
            if handle.session.header.parent_session:
                raise ApiFault(
                    "agent-busy",
                    "subagent model selection is managed by its parent",
                    {"reason": "subagent"},
                )
            provider = self._required_string(payload, "provider")
            model = self._required_string(payload, "model")
            reasoning_effort = self._optional_payload_string(payload, "reasoningEffort")
            if reasoning_effort is not None and reasoning_effort not in {"off", "high", "max"}:
                raise ApiFault("bad-request", "reasoning effort is not supported")
            self.model = model
            selection: JsonObject = {"provider": provider, "model": model}
            if reasoning_effort:
                selection["reasoningEffort"] = reasoning_effort
            handle.session.header = replace(handle.session.header, model_selection=selection)
            await self.store.save(handle.session)
            return {"selected": selection}
        if method == "session.rename":
            session_id = self._required_string(payload, "sessionId")
            handle = await self.get_session(session_id)
            if handle.session.header.parent_session:
                raise ApiFault(
                    "agent-busy",
                    "subagent sessions cannot be renamed here",
                    {"reason": "subagent"},
                )
            raw_title = payload.get("title")
            if not isinstance(raw_title, str):
                raise ApiFault("bad-request", "title must be a string")
            title = raw_title.strip()
            if not title:
                raise ApiFault(
                    "title-invalid",
                    "session title cannot be empty",
                    {"sessionId": session_id},
                )
            event = handle.session.append(
                "session/title", {"title": title, "source": {"kind": "user"}}
            )
            await self.store.save(handle.session)
            self._publish_event(handle.session.id, event)
            return {"title": title, "seq": event.seq}
        if method == "llm.providers":
            return {"providers": self._providers()}
        if method == "llm.models":
            models = self._models()
            return {"groups": models["groups"], "failures": models["failures"]}
        if method == "llm.discoverModels":
            return await self._discover_models(payload)
        if method == "host.describe":
            return {
                "version": "0.1.0.dev0",
                "cwd": str(self.cwd),
                "provider": "deepseek-official",
                "model": self.model,
                "attachedSessions": len(self._handles),
                "canOpenPath": False,
            }
        if method == "host.listDirectory":
            return self._list_directory(self._optional_payload_string(payload, "path"))
        if method == "host.createDirectory":
            return self._create_directory(
                self._required_string(payload, "path"), self._required_string(payload, "name")
            )
        if method == "host.pickDirectory":
            return {"path": None}
        if method == "host.openPath":
            path = self._required_string(payload, "path")
            if not Path(path).expanduser().exists():
                raise ApiFault("internal", f"path does not exist: {path}")
            return {"opened": True}
        if method == "workspace.list":
            workspaces, archived = await self.workspaces.list()
            return {
                "items": [workspace.to_dict() for workspace in workspaces],
                "archivedSessionIds": list(archived),
            }
        if method == "workspace.create":
            path = self._required_string(payload, "path")
            try:
                workspace, created = await self.workspaces.create(path)
            except WorkspaceInvalidPath as exc:
                raise ApiFault("workspace-invalid-path", str(exc), {"path": exc.path}) from exc
            self._publish_host({"type": "host/workspace-changed", "workspace": workspace.to_dict()})
            return {"workspace": workspace.to_dict(), "created": created}
        if method == "workspace.rename":
            workspace_id = self._required_string(payload, "workspaceId")
            title = self._required_string(payload, "title")
            try:
                workspace = await self.workspaces.rename(workspace_id, title)
            except WorkspaceNotFound as exc:
                raise ApiFault(
                    "workspace-not-found", str(exc), {"workspaceId": workspace_id}
                ) from exc
            except WorkspaceNameConflict as exc:
                raise ApiFault("workspace-name-conflict", str(exc), {"name": exc.title}) from exc
            self._publish_host({"type": "host/workspace-changed", "workspace": workspace.to_dict()})
            return {"workspace": workspace.to_dict()}
        if method == "workspace.delete":
            workspace_id = self._required_string(payload, "workspaceId")
            try:
                await self.workspaces.delete(workspace_id)
            except WorkspaceNotFound as exc:
                raise ApiFault(
                    "workspace-not-found", str(exc), {"workspaceId": workspace_id}
                ) from exc
            self._publish_host({"type": "host/workspace-removed", "workspaceId": workspace_id})
            return {"deleted": True}
        if method == "workspace.insertBefore":
            workspace_id = self._required_string(payload, "workspaceId")
            before = self._optional_payload_string(payload, "beforeWorkspaceId")
            try:
                ids = await self.workspaces.insert_before(workspace_id, before)
            except WorkspaceNotFound as exc:
                raise ApiFault(
                    "workspace-not-found", str(exc), {"workspaceId": exc.workspace_id}
                ) from exc
            self._publish_host({"type": "host/workspace-order-changed", "workspaceIds": list(ids)})
            return {"workspaceIds": list(ids)}
        if method == "workspace.insertSessionBefore":
            workspace_id = self._required_string(payload, "workspaceId")
            session_id = self._required_string(payload, "sessionId")
            before = self._optional_string(payload.get("beforeSessionId"))
            try:
                workspace = await self.workspaces.insert_session_before(
                    workspace_id, session_id, before
                )
            except WorkspaceNotFound as exc:
                raise ApiFault(
                    "workspace-not-found", str(exc), {"workspaceId": exc.workspace_id}
                ) from exc
            except WorkspaceMoveInvalid as exc:
                raise ApiFault(
                    "workspace-move-invalid",
                    str(exc),
                    {
                        "workspaceId": exc.workspace_id,
                        "sessionId": exc.session_id,
                        **(
                            {"beforeSessionId": exc.before_session_id}
                            if exc.before_session_id
                            else {}
                        ),
                    },
                ) from exc
            self._publish_host({"type": "host/workspace-changed", "workspace": workspace.to_dict()})
            return {"workspace": workspace.to_dict()}
        if method == "workspace.archiveSession":
            session_id = self._required_string(payload, "sessionId")
            try:
                await self.get_session(session_id)
            except ApiFault as exc:
                if exc.code == "session-not-found":
                    raise
            archived = await self.workspaces.archive(session_id)
            self._publish_host(
                {"type": "host/archived-sessions-changed", "archivedSessionIds": list(archived)}
            )
            return {"archivedSessionIds": list(archived)}
        if method == "skill.list":
            session = await self.get_session(self._required_string(payload, "sessionId"))
            cwd = session.session.header.cwd or str(self.cwd)
            skills = await self.skills.list(cwd)
            return {"skills": [skill.to_wire() for skill in skills]}
        if method == "settings.describe":
            exposed = {
                "ui-onboarding",
                "agent-presets",
                "agent-default-model",
                "llm-deepseek",
                "ui-theme",
                "locale",
                "ui-conversation",
                "permission",
                "shell",
                "agent-loop",
                "web-search-deepseek",
            }
            return {
                "writable": True,
                "hasDocument": True,
                "namespaces": await self.settings.describe(exposed=exposed),
            }
        if method == "settings.openDocument":
            return await self.settings.open_document()
        if method in {"settings.update", "settings.replace", "settings.mutate"}:
            namespace = self._required_string(payload, "ns")
            expected = self._optional_payload_int(payload, "expectedRevision", nonnegative=True)
            try:
                if method == "settings.update":
                    patch = payload.get("patch")
                    if not isinstance(patch, dict):
                        raise ValueError("settings.update patch must be an object")
                    result = await self.settings.update(namespace, patch, expected)
                    self._publish_settings_changed(namespace, result)
                    return result
                if method == "settings.replace":
                    section = payload.get("section")
                    if not isinstance(section, dict):
                        raise ValueError("settings.replace section must be an object")
                    result = await self.settings.replace(namespace, section, expected)
                    self._publish_settings_changed(namespace, result)
                    return result
                operations = payload.get("ops")
                if not isinstance(operations, list):
                    raise ValueError("settings.mutate ops must be an array")
                result = await self.settings.mutate(namespace, operations, expected)
                self._publish_settings_changed(namespace, result)
                return result
            except SettingsConflict as exc:
                raise ApiFault(
                    "settings-conflict",
                    str(exc),
                    {"ns": exc.namespace, "expected": exc.expected, "actual": exc.actual},
                ) from exc
            except SettingsNotFound as exc:
                raise ApiFault("settings-rejected", str(exc), {"ns": exc.namespace}) from exc
            except (TypeError, ValueError) as exc:
                raise ApiFault("settings-rejected", str(exc), {"ns": namespace}) from exc
        if method == "credentials.describe":
            refs = payload.get("refs")
            if not isinstance(refs, list) or not all(isinstance(ref, str) for ref in refs):
                raise ApiFault("bad-request", "credentials.describe refs must be a string array")
            try:
                return {"credentials": await self.credentials.describe(refs)}
            except (TypeError, ValueError) as exc:
                raise ApiFault("bad-request", str(exc)) from exc
        if method == "credentials.set":
            ref = self._required_string(payload, "ref")
            credential_value = self._required_string(payload, "value")
            try:
                await self.credentials.set(ref, credential_value)
            except (TypeError, ValueError) as exc:
                raise ApiFault("bad-request", str(exc)) from exc
            except CredentialError as exc:
                raise ApiFault("credential-rejected", str(exc), {"ref": exc.ref}) from exc
            return {}
        if method == "credentials.unset":
            ref = self._required_string(payload, "ref")
            try:
                await self.credentials.unset(ref)
            except (TypeError, ValueError) as exc:
                raise ApiFault("bad-request", str(exc)) from exc
            except CredentialError as exc:
                raise ApiFault("credential-rejected", str(exc), {"ref": exc.ref}) from exc
            return {}
        if method == "plan.set":
            session_id = self._required_string(payload, "sessionId")
            active = payload.get("active")
            if not isinstance(active, bool):
                raise ApiFault("bad-request", "plan.set active must be a boolean")
            handle = await self.get_session(session_id)
            await self._execute_command(handle, "/plan" if active else "/plan off")
            return {"plan": fold_plan_mode(handle.session).to_dict()}
        if method.startswith("goal."):
            session_id = self._required_string(payload, "sessionId")
            handle = await self.get_session(session_id)
            if handle.session.header.parent_session:
                raise ApiFault(
                    "agent-busy",
                    "goals are only available on ordinary sessions",
                    {"reason": "subagent"},
                )
            try:
                if method == "goal.create":
                    objective = self._required_string(payload, "objective")
                    ref = self.goals.create(
                        handle.session,
                        objective,
                        self._optional_payload_int(payload, "maxGoalRounds", positive=True),
                    )
                    self._goal_activation[session_id] = "armed"
                else:
                    ref = payload.get("ref")
                    if not isinstance(ref, dict):
                        raise ApiFault("bad-request", "goal ref must be an object")
                    if method == "goal.edit":
                        objective = payload.get("objective")
                        if objective is not None and not isinstance(objective, str):
                            raise ApiFault("bad-request", "goal objective must be a string")
                        if objective is not None and not objective.strip():
                            raise ApiFault("bad-request", "goal objective must be non-empty")
                        if objective is None and "maxGoalRounds" not in payload:
                            raise ApiFault(
                                "bad-request",
                                "goal.edit requires objective or maxGoalRounds",
                            )
                        ref = self.goals.edit(
                            handle.session,
                            ref,
                            objective,
                            self._optional_payload_int(payload, "maxGoalRounds", positive=True),
                        )
                    elif method in {"goal.pause", "goal.resume", "goal.complete"}:
                        operation = method.removeprefix("goal.")
                        if operation not in {"pause", "resume", "complete"}:
                            raise ApiFault("bad-request", f"unsupported goal method: {method}")
                        ref = self.goals.transition(
                            handle.session,
                            cast(Literal["pause", "resume", "complete"], operation),
                            ref,
                        )
                        self._goal_activation[session_id] = (
                            "armed" if operation == "resume" else "disarmed"
                        )
                    elif method == "goal.clear":
                        self.goals.clear(handle.session, ref)
                        self._goal_activation[session_id] = "disarmed"
                        await self.store.save(handle.session)
                        event = handle.session.events[-1]
                        self._publish_event(session_id, event)
                        self._publish_goal_projection(handle)
                        return {"cleared": True}
                    else:
                        raise ApiFault("bad-request", f"unsupported goal method: {method}")
                await self.store.save(handle.session)
                event = handle.session.events[-1]
                self._publish_event(session_id, event)
                self._publish_goal_projection(handle)
                return {"ref": ref}
            except GoalError as exc:
                raise ApiFault("internal", str(exc), {"goalCode": exc.code}) from exc
        if method == "agentPreset.list":
            presets = await self.presets.list()
            return {
                "presets": [preset.entry() for preset in presets],
                "authorable": True,
                "hasDocument": True,
            }
        if method == "agentPreset.select":
            session_id = self._required_string(payload, "sessionId")
            preset_id = self._required_string(payload, "agentPreset")
            handle = await self.get_session(session_id)
            if not self._is_blank(handle.session):
                raise ApiFault(
                    "agent-preset-locked",
                    "a session's agent preset is fixed after its first turn",
                    {"sessionId": session_id, "agentPreset": preset_id},
                )
            await self._require_preset(preset_id)
            handle.session.header = replace(handle.session.header, agent_preset=preset_id)
            event = handle.session.append("agent-preset/selected", {"agentPreset": preset_id})
            await self.store.save(handle.session)
            self._publish_event(session_id, event)
            return {"agentPreset": preset_id}
        if method == "agentPreset.read":
            preset = await self._get_preset(self._required_string(payload, "agentPreset"))
            result: JsonObject = {
                "agentPreset": preset.id,
                "trust": preset.trust,
                "content": preset.content,
            }
            if preset.name is not None:
                result["name"] = preset.name
            if preset.description is not None:
                result["description"] = preset.description
            return result
        if method == "agentPreset.copy":
            source = self._required_string(payload, "from")
            target = self._required_string(payload, "agentPreset")
            try:
                preset = await self.presets.copy(
                    source,
                    target,
                    self._optional_payload_string(payload, "name"),
                )
            except AgentPresetError as exc:
                raise self._preset_fault(exc) from exc
            return {"agentPreset": preset.id}
        if method == "agentPreset.openDocument":
            try:
                return await self.presets.open_document(
                    self._required_string(payload, "agentPreset")
                )
            except AgentPresetError as exc:
                raise self._preset_fault(exc) from exc
        if method == "agentPreset.remove":
            try:
                await self.presets.remove(self._required_string(payload, "agentPreset"))
            except AgentPresetError as exc:
                raise self._preset_fault(exc) from exc
            return {}
        if method == "subagent.list":
            parent = self._required_string(payload, "parentSessionId")
            await self.get_session(parent)
            children: list[JsonObject] = []
            for child in await self._child_sessions(parent):
                descriptor = self._subagent_descriptor(child)
                handle = self._handles.get(child.id)
                if descriptor is None:
                    descriptor = {
                        "mode": "continuable",
                        "label": child.id,
                    }
                children.append(
                    {
                        "kind": "child",
                        "id": child.id,
                        "mode": descriptor.get("mode", "one-shot"),
                        "activity": (
                            "running"
                            if handle is not None and handle.agent.status == "running"
                            else "inactive"
                        ),
                        "hasChildren": bool(await self._child_sessions(child.id)),
                        "label": descriptor.get("label", child.id),
                    }
                )
            return {"entries": children, "parentAvailable": True}
        if method == "subagent.history":
            parent = self._required_string(payload, "parentSessionId")
            child = self._required_string(payload, "childSessionId")
            await self._require_child(parent, child)
            mode = payload.get("mode")
            if mode not in {"one-shot", "continuable"}:
                raise ApiFault("bad-request", "subagent history mode is invalid")
            return await self.history(
                child,
                before_seq=self._optional_payload_int(payload, "beforeSeq", nonnegative=True),
                max_messages=self._optional_payload_int(payload, "maxMessages", positive=True),
            )
        if method == "subagent.prompt":
            parent = self._required_string(payload, "parentSessionId")
            child = self._required_string(payload, "childSessionId")
            if payload.get("mode") != "continuable":
                raise ApiFault("bad-request", "subagent prompt mode must be continuable")
            await self._require_continuable_child(parent, child)
            content = payload.get("content")
            if not isinstance(content, list):
                raise ApiFault("bad-request", "subagent.prompt content must be an array")
            value = await self.prompt(
                child,
                content,
                allow_subagent=True,
                include_message_id=True,
            )
            return {"messageId": value["messageId"]}
        if method == "subagent.interrupt":
            parent = self._required_string(payload, "parentSessionId")
            child = self._required_string(payload, "childSessionId")
            if payload.get("mode") != "continuable":
                raise ApiFault("bad-request", "subagent interrupt mode must be continuable")
            await self._require_continuable_child(parent, child)
            handle = await self.get_session(child)
            if handle.task is not None and not handle.task.done():
                handle.task.cancel()
            return {"accepted": True}
        if method == "messageFeedback/list":
            handle = await self.get_session(self._required_string(payload, "sessionId"))
            return await self.message_feedback.list(handle.session)
        if method == "messageFeedback/put":
            session_id = self._required_string(payload, "sessionId")
            message_id = self._required_string(payload, "messageId")
            rating = payload.get("rating")
            if rating not in {"positive", "negative"}:
                raise ApiFault("bad-request", "message feedback rating is invalid")
            if "ifVersion" not in payload:
                raise ApiFault("bad-request", "message feedback ifVersion is required")
            if_version = payload.get("ifVersion")
            if if_version is not None and not isinstance(if_version, str):
                raise ApiFault("bad-request", "message feedback ifVersion must be a string or null")
            note = payload.get("note")
            if note is not None and not isinstance(note, str):
                raise ApiFault("bad-request", "message feedback note must be a string")
            handle = await self.get_session(session_id)
            return await self.message_feedback.put(
                handle.session,
                message_id,
                rating,
                note,
                if_version,
            )
        if method == "messageFeedback/delete":
            session_id = self._required_string(payload, "sessionId")
            message_id = self._required_string(payload, "messageId")
            if not isinstance(payload.get("ifVersion"), str):
                raise ApiFault("bad-request", "message feedback delete ifVersion must be a string")
            handle = await self.get_session(session_id)
            return await self.message_feedback.delete(
                handle.session,
                message_id,
                payload["ifVersion"],
            )
        if method == "dynamicCordisRunner/syncInspectManifest":
            self.dynamic.sync_inspect_manifest(payload.get("providers", []))
            return None
        if method == "dynamicCordisRunner/inventory":
            return self.dynamic.inventory()
        if method == "dynamicCordisRunner/runHostHalf":
            request_value = payload.get("requestId")
            if request_value is not None and not isinstance(request_value, str):
                raise ApiFault("bad-request", "requestId must be a string or null")
            return await self.dynamic.run_host_half(
                self._required_string(payload, "agentId"),
                self._required_string(payload, "pluginId"),
                self._required_string(payload, "packageId"),
                self._required_string(payload, "mode"),
                request_value,
                payload.get("approveFutureVersions") is True,
            )
        if method == "dynamicCordisRunner/getClientCode":
            return self.dynamic.get_client_code(
                self._required_string(payload, "agentId"),
                self._required_string(payload, "pluginId"),
                self._required_string(payload, "pluginRunId"),
            )
        if method == "dynamicCordisRunner/resolveRequestRun":
            resolution = payload.get("resolution")
            if not isinstance(resolution, dict):
                raise ApiFault("bad-request", "dynamic Cordis resolution must be an object")
            return await self.dynamic.resolve_request_run(
                self._required_string(payload, "requestId"), resolution
            )
        if method == "dynamicCordisRunner/settleUserRun":
            resolution = payload.get("resolution")
            if not isinstance(resolution, dict):
                raise ApiFault("bad-request", "dynamic Cordis resolution must be an object")
            return await self.dynamic.settle_user_run(
                self._required_string(payload, "agentId"),
                self._required_string(payload, "pluginId"),
                resolution,
            )
        if method == "dynamicCordisRunner/stopFromPanel":
            return await self.dynamic.stop(
                self._required_string(payload, "agentId"),
                self._required_string(payload, "pluginId"),
            )
        if method == "dynamicCordisRunner/undefineFromPanel":
            return await self.dynamic.undefine(
                self._required_string(payload, "agentId"),
                self._required_string(payload, "pluginId"),
            )
        if method == "dynamicCordisRunner/reportRenderFailure":
            failure = payload.get("failure")
            if not isinstance(failure, dict):
                raise ApiFault("bad-request", "dynamic Cordis render failure must be an object")
            self.dynamic.report_render_failure(
                self._required_string(payload, "agentId"),
                self._required_string(payload, "pluginId"),
                self._required_string(payload, "pluginRunId"),
                failure,
            )
            return None
        if method == "dynamicCordisRunner/reportClientGuardFailure":
            failure = payload.get("failure")
            if not isinstance(failure, dict):
                raise ApiFault("bad-request", "dynamic Cordis guard failure must be an object")
            self.dynamic.report_client_guard_failure(
                self._required_string(payload, "agentId"),
                self._required_string(payload, "pluginId"),
                self._required_string(payload, "pluginRunId"),
                failure,
            )
            return None
        if method == "dynamicCordisRunner/invoke":
            return await self.dynamic.invoke(
                self._required_string(payload, "pluginId"),
                self._required_string(payload, "pluginRunId"),
                self._required_string(payload, "method"),
                payload.get("args"),
            )
        if method == "dynamicCordisRunner/resolveInspectQuery":
            resolution = payload.get("resolution")
            if not isinstance(resolution, dict):
                raise ApiFault("bad-request", "dynamic Cordis inspect resolution must be an object")
            return self.dynamic.resolve_inspect_query(
                self._required_string(payload, "agentId"),
                self._required_string(payload, "requestId"),
                resolution,
            )
        raise ApiFault("bad-request", f"unsupported RPC method: {method}")

    async def respond(self, message: JsonObject) -> JsonObject:
        """Route a client-response to the pending approval/question registry."""

        rpc_id = message.get("rpcId")
        if not isinstance(rpc_id, str):
            return {"accepted": False, "reason": "bad-response"}
        result = message.get("result")
        if not isinstance(result, dict):
            return {"accepted": False, "reason": "bad-response"}

        approval = self._pending_approvals.get(rpc_id)
        if approval is not None:
            if result.get("ok") is not True:
                return {"accepted": False, "reason": "bad-response"}
            value = result.get("value")
            if not isinstance(value, dict):
                return {"accepted": False, "reason": "bad-response"}
            if (
                value.get("sessionId") != approval.session_id
                or value.get("approvalId") != approval.approval_id
                or value.get("outcome") not in {"allowed-once", "rejected"}
            ):
                return {"accepted": False, "reason": "bad-response"}
            await self._finish_approval(rpc_id, cast(ApprovalOutcome, value["outcome"]))
            return {"accepted": True}

        question = self._pending_questions.get(rpc_id)
        if question is None:
            return {"accepted": False, "reason": "not-pending"}
        if result.get("ok") is not True:
            error = result.get("error")
            if not isinstance(error, dict) or error.get("code") != "cancelled":
                return {"accepted": False, "reason": "bad-response"}
            await self._finish_question(rpc_id, "cancelled")
            return {"accepted": True}
        value = result.get("value")
        if not isinstance(value, dict):
            return {"accepted": False, "reason": "bad-response"}
        session_id = value.get("sessionId")
        answer = value.get("answer")
        if session_id != question.session_id or not isinstance(answer, dict):
            return {"accepted": False, "reason": "bad-response"}
        if not self._matches_question_answer(question.questions, answer):
            return {"accepted": False, "reason": "bad-response"}
        await self._finish_question(rpc_id, "answered", answer)
        return {"accepted": True}

    async def _finish_approval(self, rpc_id: str, outcome: ApprovalOutcome) -> None:
        pending = self._pending_approvals.pop(rpc_id, None)
        if pending is None:
            return
        handle = self._handles.get(pending.session_id)
        if handle is not None:
            event = handle.session.append(
                "approval/decided",
                {"id": pending.approval_id, "outcome": outcome},
            )
            await self.store.save(handle.session)
            self._publish_event(pending.session_id, event)
        self._publish_mux(
            {
                "type": "approval/resolved",
                "sessionId": pending.session_id,
                "approvalId": pending.approval_id,
                "outcome": outcome,
            }
        )
        if not pending.future.done():
            pending.future.set_result(outcome)

    async def _finish_question(
        self,
        rpc_id: str,
        outcome: Literal["answered", "cancelled"],
        answer: JsonObject | None = None,
    ) -> None:
        pending = self._pending_questions.pop(rpc_id, None)
        if pending is None:
            return
        self._publish_mux(
            {
                "type": "question/resolved",
                "sessionId": pending.session_id,
                "questionRpcId": pending.rpc_id,
                "outcome": outcome,
            }
        )
        if not pending.future.done():
            if outcome == "answered" and answer is not None:
                pending.future.set_result(answer)
            else:
                pending.future.set_exception(
                    ApiFault("cancelled", "the user cancelled the pending question")
                )

    @staticmethod
    def _matches_question_answer(questions: list[JsonObject], answer: JsonObject) -> bool:
        answers = answer.get("answers")
        if not isinstance(answers, list) or len(answers) != len(questions):
            return False
        for question, candidate in zip(questions, answers, strict=True):
            if not isinstance(candidate, dict):
                return False
            if candidate.get("id") != question.get("id"):
                return False
            selected = candidate.get("selected")
            if not isinstance(selected, list) or not all(
                isinstance(item, str) for item in selected
            ):
                return False
            if len(set(selected)) != len(selected):
                return False
            custom = candidate.get("custom")
            if custom is not None and (not isinstance(custom, str) or not custom.strip()):
                return False
            if question.get("multiSelect") is not True:
                if len(selected) > 1 or (custom is not None and selected):
                    return False
            options = question.get("options")
            labels = (
                {
                    option.get("label")
                    for option in options
                    if isinstance(option, dict) and isinstance(option.get("label"), str)
                }
                if isinstance(options, list)
                else set()
            )
            if any(item not in labels for item in selected):
                return False
        return True

    async def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        for rpc_id in tuple(self._pending_approvals):
            await self._finish_approval(rpc_id, "cancelled")
        for rpc_id in tuple(self._pending_questions):
            await self._finish_question(rpc_id, "cancelled")
        handles = tuple(self._handles.values())
        for handle in handles:
            if handle.task is not None and not handle.task.done():
                handle.task.cancel()
        for handle in handles:
            if handle.task is not None:
                await asyncio.gather(handle.task, return_exceptions=True)
            await self.store.save(handle.session)
            for dispose in reversed(handle.disposers):
                dispose()
            await handle.agent.dispose()
        self._handles.clear()
        self._tool_registries.clear()
        await self.jobs.close()
        self._mux_subscribers.clear()
        self._host_subscribers.clear()

    async def _run_queue(self, handle: SessionHandle) -> None:
        current_task = asyncio.current_task()
        try:
            while True:
                async with self._queue_lock:
                    if not handle.queue:
                        break
                    item = handle.queue.pop(0)
                    self._publish_queue(handle)
                self._publish_host(
                    {
                        "type": "host/session-status",
                        "sessionId": handle.session.id,
                        "running": True,
                    }
                )
                try:
                    await handle.agent.run(item.message)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._publish_host(
                        {
                            "type": "host/agent-error",
                            "sessionId": handle.session.id,
                            "message": str(exc),
                        }
                    )
                finally:
                    await self.store.save(handle.session)
                    self._publish_host(
                        {
                            "type": "host/session-status",
                            "sessionId": handle.session.id,
                            "running": False,
                        }
                    )
        finally:
            if handle.task is current_task:
                handle.task = None
            self._publish_queue(handle)
            if handle.queue and not self._disposed:
                handle.task = asyncio.create_task(self._run_queue(handle))

    def _install_goal_tools(
        self,
        registry: ToolRegistry,
        session: Session,
    ) -> list[Callable[[], None]]:
        """Install the TS-compatible model-facing goal tool trio."""

        def require_context(context: ToolContext, *, mutation: bool = False) -> None:
            if context.session_id != session.id:
                raise ApiFault(
                    "subagent-unauthorized",
                    "goal tool context does not belong to the registered session",
                    {"sessionId": context.session_id},
                )
            if mutation and session.header.parent_session:
                raise ValueError(
                    "goal mutations require a direct human turn on a top-level session"
                )

        def require_direct_human_turn() -> None:
            if not self._goal_has_direct_human_turn(session):
                raise ValueError(
                    "goal mutations require a direct human turn on a top-level session"
                )

        async def commit() -> None:
            await self.store.save(session)
            event = session.events[-1]
            self._publish_event(session.id, event)
            handle = self._handles.get(session.id)
            if handle is not None:
                self._publish_goal_projection(handle)

        def value() -> JsonObject:
            view = self.goals.view(
                session,
                activation=self._goal_activation.get(session.id, "disarmed"),
            )
            if view is None:
                return {"goal": None}
            goal = {
                key: view[key]
                for key in (
                    "id",
                    "revision",
                    "objective",
                    "phase",
                    "roundsStarted",
                    "maxGoalRounds",
                    "blockedReason",
                )
                if key in view
            }
            return {"goal": goal, "activation": view["activation"]}

        async def get_goal(args: dict[str, Any], context: ToolContext) -> ToolResult:
            require_context(context)
            result = value()
            return ToolResult(
                json.dumps(result, ensure_ascii=False, separators=(",", ":")),
                meta=result,
            )

        async def create_goal(args: dict[str, Any], context: ToolContext) -> ToolResult:
            require_context(context, mutation=True)
            require_direct_human_turn()
            objective = args.get("objective")
            if not isinstance(objective, str):
                raise ValueError("objective must be a string")
            max_rounds = self._goal_round_argument(args, "max_goal_rounds")
            self.goals.create(session, objective, max_rounds)
            self._goal_activation[session.id] = "armed"
            await commit()
            result = value()
            return ToolResult(
                json.dumps(result, ensure_ascii=False, separators=(",", ":")),
                meta=result,
            )

        async def update_goal(args: dict[str, Any], context: ToolContext) -> ToolResult:
            require_context(context, mutation=True)
            require_direct_human_turn()
            goal_id = args.get("goal_id")
            revision = args.get("revision")
            action = args.get("action")
            if (
                not isinstance(goal_id, str)
                or not goal_id.strip()
                or goal_id != goal_id.strip()
            ):
                raise ValueError("goal_id must be a non-empty normalized string")
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
                raise ValueError("revision must be a positive integer")
            if action not in {"edit", "pause", "resume", "complete", "blocked"}:
                raise ValueError("action must be edit, pause, resume, complete, or blocked")
            objective = args.get("objective")
            if objective is not None and not isinstance(objective, str):
                raise ValueError("objective must be a string")
            max_rounds = self._goal_round_argument(args, "max_goal_rounds")
            blocked_reason = args.get("blocked_reason")
            if blocked_reason is not None and not isinstance(blocked_reason, str):
                raise ValueError("blocked_reason must be a string")
            ref = {"id": goal_id, "revision": revision}
            if action == "edit":
                if objective is None and max_rounds is None:
                    raise ValueError("edit requires objective or max_goal_rounds")
                if blocked_reason not in (None, ""):
                    raise ValueError("blocked_reason is valid only with action blocked")
                self.goals.edit(session, ref, objective, max_rounds)
            elif action in {"pause", "resume", "complete"}:
                if (
                    (isinstance(objective, str) and objective.strip())
                    or max_rounds is not None
                    or blocked_reason not in (None, "")
                ):
                    raise ValueError(
                        "objective and max_goal_rounds are valid only with action edit; "
                        "blocked_reason is valid only with action blocked"
                    )
                self.goals.transition(session, action, ref)
                self._goal_activation[session.id] = (
                    "armed" if action == "resume" else "disarmed"
                )
            else:
                if (
                    (isinstance(objective, str) and objective.strip())
                    or max_rounds is not None
                ):
                    raise ValueError(
                        "objective and max_goal_rounds are valid only with action edit"
                    )
                if not isinstance(blocked_reason, str) or not blocked_reason.strip():
                    raise ValueError("blocked_reason is required with action blocked")
                self.goals.block(
                    session,
                    ref,
                    {"code": "model-reported", "message": blocked_reason},
                )
                self._goal_activation[session.id] = "disarmed"
            await commit()
            result = value()
            return ToolResult(
                json.dumps(result, ensure_ascii=False, separators=(",", ":")),
                meta=result,
            )

        definitions = [
            ToolDefinition(
                name="get_goal",
                description=(
                    "Read the current same-session completion goal, including its exact id, "
                    "revision, objective, phase, rounds, and activation. Call this before updating."
                ),
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
                execute=get_goal,
            ),
            ToolDefinition(
                name="create_goal",
                description=(
                    "Create one persisted same-session completion goal for a long-running direct "
                    "human request. Do not use this for routine single-turn work."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "objective": {
                            "type": "string",
                            "description": "The concrete completion objective.",
                        },
                        "max_goal_rounds": {
                            "type": "integer",
                            "description": "Optional positive limit on continuation rounds.",
                        },
                    },
                    "required": ["objective"],
                    "additionalProperties": False,
                },
                execute=create_goal,
            ),
            ToolDefinition(
                name="update_goal",
                description=(
                    "Update the exact current goal revision. Use edit, pause, resume, complete, "
                    "or blocked; copy goal_id and revision from get_goal."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "goal_id": {"type": "string"},
                        "revision": {"type": "integer"},
                        "action": {
                            "type": "string",
                            "enum": ["edit", "pause", "resume", "complete", "blocked"],
                        },
                        "objective": {"type": "string"},
                        "max_goal_rounds": {"type": "integer"},
                        "blocked_reason": {"type": "string"},
                    },
                    "required": ["goal_id", "revision", "action"],
                    "additionalProperties": False,
                },
                execute=update_goal,
            ),
        ]
        return [registry.register(definition) for definition in definitions]

    def _install_plan_tools(
        self,
        registry: ToolRegistry,
        session: Session,
    ) -> list[Callable[[], None]]:
        """Install the stable plan-review exit tool."""

        async def exit_plan_mode(args: dict[str, Any], context: ToolContext) -> ToolResult:
            if context.session_id != session.id:
                raise ApiFault(
                    "subagent-unauthorized",
                    "plan tool context does not belong to the registered session",
                    {"sessionId": context.session_id},
                )
            if not fold_plan_mode(session).active:
                raise ValueError("exit_plan_mode is only available in plan mode")
            plan = args.get("plan")
            if not isinstance(plan, str) or re.match(r"^#\s+\S", plan.strip()) is None:
                raise ValueError(
                    "exit_plan_mode requires a non-empty markdown plan starting with a # heading"
                )
            if session.header.parent_session:
                raise ValueError(
                    "human interaction is unavailable while the calling agent is owned by "
                    "another live agent; include the unresolved question or decision in the "
                    "child agent's final result"
                )
            answer = await self.request_question(
                session.id,
                [
                    {
                        "id": "plan-review",
                        "header": "Plan review",
                        "question": "Approve this plan and leave plan mode?",
                        "detail": plan,
                        "options": [
                            {
                                "label": "Approve",
                                "description": (
                                    "Leave plan mode; carry out the plan from the next step."
                                ),
                            },
                            {
                                "label": "Keep planning",
                                "description": (
                                    "Stay in plan mode; feedback goes back to the model."
                                ),
                            },
                        ],
                        "intent": {"kind": "plan-review", "approve": "Approve"},
                    }
                ],
            )
            answers = answer.get("answers")
            item = next(
                (
                    entry
                    for entry in answers
                    if isinstance(entry, dict) and entry.get("id") == "plan-review"
                ),
                None,
            ) if isinstance(answers, list) else None
            selected = item.get("selected") if isinstance(item, dict) else None
            custom = item.get("custom") if isinstance(item, dict) else None
            if selected != ["Approve"] or custom is not None:
                feedback = custom if isinstance(custom, str) else ""
                raise ValueError(
                    "The user chose to keep planning; revise the plan and present it again."
                    if not feedback
                    else f"The user chose to keep planning; their feedback: {feedback}"
                )
            event = session.append("plan/mode", {"active": False})
            await self.store.save(session)
            self._publish_event(session.id, event)
            return ToolResult(
                "Plan approved — plan mode exited; carry out the plan starting with your "
                "next step.",
                meta={"approved": True},
            )

        return [
            registry.register(
                ToolDefinition(
                    name="exit_plan_mode",
                    description=(
                        "Use only in plan mode. Present the COMPLETE plan as markdown starting "
                        "with a # heading for user review; the user may approve or keep planning."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "plan": {
                                "type": "string",
                                "description": (
                                    "The complete plan as markdown, starting with a # heading."
                                ),
                            }
                        },
                        "required": ["plan"],
                        "additionalProperties": False,
                    },
                    execute=exit_plan_mode,
                )
            )
        ]

    def _install_ralph_tool(
        self,
        registry: ToolRegistry,
        session: Session,
    ) -> list[Callable[[], None]]:
        """Install a foreground fresh-agent Ralph loop.

        The TypeScript implementation expresses the fixed loop as a workflow
        script.  Python keeps the same externally visible contract while
        executing the loop natively: every round gets a blank one-shot child,
        and only a bounded JSON handoff crosses the round boundary.
        """

        async def record(event_type: str, data: JsonObject) -> None:
            event = session.append(event_type, data)
            self._publish_event(session.id, event)
            await self.store.save(session)

        async def execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
            if context.session_id != session.id:
                raise ApiFault(
                    "subagent-unauthorized",
                    "ralph tool context does not belong to the registered session",
                    {"sessionId": context.session_id},
                )
            if session.header.parent_session:
                raise ValueError(
                    "Ralph requires a direct human turn on a top-level session"
                )
            if not self._goal_has_direct_human_turn(session):
                raise ValueError(
                    "Ralph requires a direct human turn on a top-level session"
                )
            objective = args.get("objective")
            if not isinstance(objective, str) or not objective.strip():
                raise ValueError("Ralph objective must be a non-empty string")
            max_rounds = args.get("maxRounds", DEFAULT_RALPH_MAX_ROUNDS)
            if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 1:
                raise ValueError("Ralph maxRounds must be a positive integer")
            if max_rounds > DEFAULT_RALPH_MAX_ROUNDS:
                raise ValueError(
                    f"Ralph maxRounds {max_rounds} exceeds the deployment ceiling "
                    f"{DEFAULT_RALPH_MAX_ROUNDS}"
                )
            run_id = f"ralph-{uuid.uuid4().hex}"
            await record("tool-workflow/run-start", {"runId": run_id, "name": "ralph-loop"})
            await record(
                "tool-workflow/phase",
                {"runId": run_id, "title": "Fresh-agent rounds"},
            )
            previous: JsonObject | None = None
            stop_reason = "completed"
            error: str | None = None
            agents_started = 0
            terminal: JsonObject | None = None
            try:
                for round_number in range(1, max_rounds + 1):
                    prior = (
                        "(none — this is the first round)"
                        if previous is None
                        else json.dumps(previous, ensure_ascii=False, separators=(",", ":"))
                    )
                    prompt = (
                        "You are one fresh worker in a foreground Ralph loop. You receive no "
                        "parent conversation and no prior child session. Do not call the ralph "
                        "tool: this round already is its worker.\n\n"
                        f"Immutable objective:\n{objective.strip()}\n\n"
                        f"Ralph round: {round_number} of {max_rounds}.\n\n"
                        "The shared workspace and its current working tree are long-term memory "
                        "and the source of truth. Inspect it before acting, preserve existing "
                        "work, perform concrete in-scope work, and verify what you change. "
                        "Treat the previous report only as a bounded handoff and confirm it "
                        "against the workspace.\n\n"
                        f"Previous structured handoff:\n{prior}\n\n"
                        "Return ONLY one JSON object with exactly these keys: status, summary, "
                        "evidence, nextSteps, blocker. status is continue, complete, or blocked. "
                        "Use continue with at least one nextSteps entry while useful work "
                        "remains; use complete only with concrete evidence and no nextSteps; "
                        "use blocked only when progress requires human input or external state. "
                        "blocker must be empty unless blocked."
                    )
                    label = f"Ralph round {round_number}"

                    async def announce(
                        child_id: str,
                        sequence: int = round_number,
                        agent_label: str = label,
                    ) -> None:
                        await record(
                            "tool-workflow/agent-start",
                            {
                                "runId": run_id,
                                "seq": sequence,
                                "label": agent_label,
                                "phase": "Fresh-agent rounds",
                                "childId": child_id,
                            },
                        )

                    child_id, finish_reason, output = await self._run_fresh_subagent(
                        await self.get_session(session.id),
                        label,
                        prompt,
                        on_started=announce,
                    )
                    agents_started = round_number
                    await record(
                        "tool-workflow/agent-end",
                        {
                            "runId": run_id,
                            "seq": round_number,
                            "outcome": "completed"
                            if finish_reason in {"completed", "max-tokens"}
                            else "failed",
                        },
                    )
                    if finish_reason not in {"completed", "max-tokens"}:
                        stop_reason = "error"
                        error = (
                            f"Ralph round {round_number} child failed before producing a "
                            "structured report."
                        )
                        rendered = (
                            f"{error}\nNo previous handoff was available."
                            if previous is None
                            else f"{error}\nLast successful handoff:\n"
                            f"{json.dumps(previous, ensure_ascii=False, indent=2)}"
                        )
                        return ToolResult(
                            self._bound_ralph_result(rendered, MAX_RALPH_RESULT_CHARS),
                            is_error=True,
                            meta={
                                "runId": run_id,
                                "agentsStarted": agents_started,
                                "status": "round-failed",
                            },
                        )
                    try:
                        report = self._parse_ralph_report(output)
                    except ValueError as exc:
                        stop_reason = "error"
                        error = str(exc)
                        rendered = (
                            f"Ralph round {round_number} child failed before producing a "
                            f"structured report: {exc}"
                        )
                        return ToolResult(
                            self._bound_ralph_result(rendered, MAX_RALPH_RESULT_CHARS),
                            is_error=True,
                            meta={
                                "runId": run_id,
                                "agentsStarted": agents_started,
                                "status": "round-failed",
                            },
                        )
                    previous = report
                    if report["status"] in {"complete", "blocked"}:
                        terminal = {
                            "status": report["status"],
                            "roundsStarted": round_number,
                            "report": report,
                        }
                        break
                if terminal is None:
                    assert previous is not None
                    terminal = {
                        "status": "budget-limited",
                        "roundsStarted": max_rounds,
                        "report": previous,
                    }
                rendered = self._render_ralph_result(terminal, MAX_RALPH_RESULT_CHARS)
                return ToolResult(
                    rendered,
                    meta={
                        "runId": run_id,
                        "agentsStarted": agents_started,
                        "result": terminal,
                    },
                )
            except asyncio.CancelledError:
                stop_reason = "cancelled"
                error = "parent step aborted"
                raise
            except Exception as exc:
                stop_reason = "error"
                error = str(exc)
                raise
            finally:
                await record(
                    "tool-workflow/run-end",
                    {
                        "runId": run_id,
                        "stopReason": stop_reason,
                        **({"error": error} if error is not None else {}),
                    },
                )

        return [
            registry.register(
                ToolDefinition(
                    name="ralph",
                    description=(
                        "Run a foreground fresh-agent loop toward one immutable objective. "
                        "Use only when the direct human explicitly asks for Ralph or fresh-agent "
                        "iteration; each round starts a blank child and passes only a bounded "
                        "structured handoff."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "objective": {
                                "type": "string",
                                "description": "The immutable completion objective.",
                            },
                            "maxRounds": {
                                "type": "integer",
                                "description": (
                                    "Optional positive round cap, up to the deployment ceiling."
                                ),
                            },
                        },
                        "required": ["objective"],
                        "additionalProperties": False,
                    },
                    execute=execute,
                )
            )
        ]

    def _install_workflow_tool(
        self,
        registry: ToolRegistry,
        session: Session,
    ) -> list[Callable[[], None]]:
        """Install the JavaScript workflow orchestration tool."""

        async def execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
            if context.session_id != session.id:
                raise ApiFault(
                    "subagent-unauthorized",
                    "workflow tool context does not belong to the registered session",
                    {"sessionId": context.session_id},
                )
            if session.header.parent_session:
                raise ValueError(
                    "workflow requires a direct human turn on a top-level session"
                )
            if not self._goal_has_direct_human_turn(session):
                raise ValueError(
                    "workflow requires a direct human turn on a top-level session"
                )
            script = args.get("script")
            meta = args.get("meta")
            workflow_args = args.get("args")
            if not isinstance(script, str):
                raise ValueError("workflow script must be a string")
            if not isinstance(meta, dict):
                raise ValueError("workflow meta must be an object")
            if workflow_args is not None and not isinstance(workflow_args, dict):
                raise ValueError("workflow args must be an object")
            run_id = f"workflow-{uuid.uuid4().hex}"
            name = meta.get("name") if isinstance(meta.get("name"), str) else "workflow"

            async def record(event_type: str, data: JsonObject) -> None:
                event = session.append(event_type, data)
                self._publish_event(session.id, event)
                await self.store.save(session)

            await record("tool-workflow/run-start", {"runId": run_id, "name": name})
            stop_reason = "error"
            error: str | None = None
            try:
                async def event_sink(kind: str, data: dict[str, Any]) -> None:
                    event_name = {
                        "phase": "tool-workflow/phase",
                        "log": "tool-workflow/log",
                        "agent-start": "tool-workflow/agent-start",
                        "agent-end": "tool-workflow/agent-end",
                    }.get(kind)
                    if event_name is None:
                        return
                    await record(event_name, {"runId": run_id, **data})

                async def agent_runner(
                    prompt: str,
                    label: str,
                    options: dict[str, Any],
                    on_started: Callable[[str], Awaitable[None]],
                ) -> WorkflowChildResult:
                    model_selection: JsonObject = {}
                    for key in ("provider", "model"):
                        value = options.get(key)
                        if isinstance(value, str):
                            model_selection[key] = value
                    child_id, finish_reason, output = await self._run_fresh_subagent(
                        await self.get_session(session.id),
                        label,
                        prompt,
                        on_started=on_started,
                        model_selection=model_selection or None,
                    )
                    if finish_reason not in {"completed", "max-tokens"}:
                        return WorkflowChildResult(
                            child_id,
                            False,
                            error=f"child ended with {finish_reason}",
                        )
                    schema = options.get("schema")
                    if schema is None:
                        return WorkflowChildResult(child_id, True, output)
                    value = self._decode_json_text(output)
                    if value is None or not matches_object_schema(value, schema):
                        return WorkflowChildResult(
                            child_id,
                            False,
                            error="child did not return a value matching its output schema",
                        )
                    return WorkflowChildResult(child_id, True, value)

                result = await run_workflow(
                    script=script,
                    meta=meta,
                    args=workflow_args,
                    agent_runner=agent_runner,
                    event_sink=event_sink,
                )
                stop_reason = result.stop_reason
                error = result.error
                if result.stop_reason != "completed":
                    detail = result.error or "unknown workflow failure"
                    return ToolResult(
                        f'workflow "{name}" failed: {detail}',
                        is_error=True,
                        meta={
                            "runId": run_id,
                            "agentsStarted": result.agents_started,
                            "stopReason": result.stop_reason,
                        },
                    )
                rendered = json.dumps(
                    result.value,
                    ensure_ascii=False,
                    indent=2,
                )
                if len(rendered) > 50_000:
                    rendered = (
                        f"{rendered[:50_000]}\n… [truncated: "
                        f"{len(rendered) - 50_000} more characters]"
                    )
                return ToolResult(
                    f'workflow "{name}" completed ({result.agents_started} '
                    f"agent{'s' if result.agents_started != 1 else ''}).\n"
                    f"Return value:\n{rendered}",
                    meta={
                        "runId": run_id,
                        "agentsStarted": result.agents_started,
                        "result": result.value,
                    },
                )
            except WorkflowError as exc:
                stop_reason = "error"
                error = str(exc)
                raise
            except Exception as exc:
                stop_reason = "error"
                error = str(exc)
                raise
            finally:
                await record(
                    "tool-workflow/run-end",
                    {
                        "runId": run_id,
                        "stopReason": stop_reason,
                        **({"error": error} if error is not None else {}),
                    },
                )

        return [
            registry.register(
                ToolDefinition(
                    name="workflow",
                    description=(
                        "Run a JavaScript workflow script that orchestrates subagents at scale. "
                        "Use only when the direct human asks for a workflow or large multi-agent "
                        "fan-out; the script may use agent, pipeline, parallel, phase, and log."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "script": {
                                "type": "string",
                                "description": (
                                    "Plain JavaScript body with top-level await; end with "
                                    "return <json-value>."
                                ),
                            },
                            "meta": {
                                "type": "object",
                                "description": "Workflow identity with name and description.",
                                "properties": {
                                    "name": {"type": "string"},
                                    "description": {"type": "string"},
                                    "whenToUse": {"type": "string"},
                                    "phases": {"type": "array"},
                                },
                                "required": ["name", "description"],
                            },
                            "args": {
                                "type": "object",
                                "description": "Optional JSON input exposed as the args global.",
                            },
                        },
                        "required": ["script", "meta"],
                        "additionalProperties": False,
                    },
                    execute=execute,
                )
            )
        ]

    @staticmethod
    def _decode_json_text(output: str) -> Any | None:
        text = output.strip()
        candidates = [text]
        if text.startswith("```") and text.endswith("```"):
            candidates.append(text.strip("`").removeprefix("json").strip())
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidates.append(text[start : end + 1])
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return None

    @staticmethod
    def _parse_ralph_report(output: str) -> JsonObject:
        """Decode and enforce the fixed structured Ralph handoff."""

        text = output.strip()
        candidates = [text]
        if text.startswith("```") and text.endswith("```"):
            candidates.append(text.strip("`").removeprefix("json").strip())
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidates.append(text[start : end + 1])
        value: Any = None
        for candidate in candidates:
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                break
        if not isinstance(value, dict):
            raise ValueError("Ralph child returned no structured round report")
        if set(value) != {"status", "summary", "evidence", "nextSteps", "blocker"}:
            raise ValueError("Ralph round report has unexpected fields")
        status = value.get("status")
        summary = value.get("summary")
        evidence = value.get("evidence")
        next_steps = value.get("nextSteps")
        blocker = value.get("blocker")
        if status not in {"continue", "complete", "blocked"}:
            raise ValueError("Ralph round report status is invalid")
        if not isinstance(summary, str) or not summary or summary != summary.strip():
            raise ValueError("Ralph round report summary must be non-empty and normalized")
        for name, items in (("evidence", evidence), ("nextSteps", next_steps)):
            if not isinstance(items, list) or any(
                not isinstance(item, str) or not item or item != item.strip() for item in items
            ):
                raise ValueError(
                    f"Ralph round report {name} must contain normalized strings"
                )
        if not isinstance(blocker, str) or blocker != blocker.strip():
            raise ValueError("Ralph round report blocker must be a normalized string")
        if status == "continue" and (not next_steps or blocker != ""):
            raise ValueError("a continuing Ralph report needs nextSteps and an empty blocker")
        if status == "complete" and (not evidence or next_steps or blocker != ""):
            raise ValueError(
                "a complete Ralph report needs evidence, no nextSteps, and an empty blocker"
            )
        if status == "blocked" and not blocker:
            raise ValueError("a blocked Ralph report needs a concrete blocker")
        serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if len(serialized) > MAX_RALPH_HANDOFF_CHARS:
            raise ValueError("Ralph round report exceeds maxHandoffChars")
        return value

    @staticmethod
    def _bound_ralph_result(text: str, maximum: int) -> str:
        notice = "\n… [truncated]"
        if len(text) <= maximum:
            return text
        if maximum <= len(notice):
            return notice[:maximum]
        return f"{text[: maximum - len(notice)]}{notice}"

    @classmethod
    def _render_ralph_result(cls, result: JsonObject, maximum: int) -> str:
        status = result.get("status")
        rounds = result.get("roundsStarted")
        report = result.get("report")
        noun = f"{rounds} round" + ("" if rounds == 1 else "s")
        if status == "complete":
            heading = f"Ralph worker reported completion after {noun}."
        elif status == "blocked":
            heading = f"Ralph worker reported a blocker after {noun}."
        else:
            heading = f"Ralph reached its {noun} limit; the worker reported work remaining."
        return cls._bound_ralph_result(
            f"{heading}\nFinal report:\n{json.dumps(report, ensure_ascii=False, indent=2)}",
            maximum,
        )

    @staticmethod
    def _goal_round_argument(args: dict[str, Any], name: str) -> int | None:
        value = args.get(name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _goal_has_direct_human_turn(session: Session) -> bool:
        if session.header.parent_session:
            return False
        open_turn = False
        for event in session.events:
            if event.type == "turn/start":
                open_turn = True
            elif event.type == "turn/end":
                open_turn = False
            elif open_turn and event.type == "user/message":
                message = event.data.get("message")
                source = message.get("source") if isinstance(message, dict) else None
                if isinstance(source, dict) and source.get("kind") == "user":
                    return True
        return False

    def _install_ask_user_tool(
        self,
        registry: ToolRegistry,
        session: Session,
    ) -> list[Callable[[], None]]:
        """Install the UI-backed ``ask_user_question`` model tool."""

        async def execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
            if context.session_id != session.id:
                raise ApiFault(
                    "subagent-unauthorized",
                    "question tool context does not belong to the registered session",
                    {"sessionId": context.session_id},
                )
            questions = self._normalize_questions(args.get("questions"))
            answer = await self.request_question(session.id, questions)
            return ToolResult(
                json.dumps(answer, ensure_ascii=False, separators=(",", ":")),
                meta=answer,
            )

        return [
            registry.register(
                ToolDefinition(
                    name="ask_user_question",
                    description=(
                        "Ask the user a concise question when you need confirmation, a choice, "
                        "or missing information before proceeding."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "questions": {
                                "type": "array",
                                "description": "Questions to ask the user before continuing.",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": True,
                                    "properties": {
                                        "id": {"type": "string"},
                                        "question": {"type": "string"},
                                        "header": {"type": "string"},
                                        "detail": {"type": "string"},
                                        "intent": {"type": "object"},
                                        "options": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "additionalProperties": True,
                                                "properties": {
                                                    "label": {"type": "string"},
                                                    "description": {"type": "string"},
                                                },
                                                "required": ["label"],
                                            },
                                        },
                                        "multi_select": {"type": "boolean"},
                                    },
                                    "required": ["id", "question"],
                                },
                            }
                        },
                        "required": ["questions"],
                        "additionalProperties": False,
                    },
                    execute=execute,
                )
            )
        ]

    @staticmethod
    def _normalize_questions(raw: Any) -> list[JsonObject]:
        if not isinstance(raw, list) or not raw:
            raise ValueError("questions must be a non-empty array")
        normalized: list[JsonObject] = []
        ids: set[str] = set()
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"question {index} must be an object")
            question_id = item.get("id")
            text = item.get("question")
            if not isinstance(question_id, str) or not question_id.strip():
                raise ValueError(f"question {index} id must be a non-empty string")
            if question_id in ids:
                raise ValueError(f"duplicate question id: {question_id}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"question {question_id} must be a non-empty string")
            ids.add(question_id)
            value: JsonObject = {"id": question_id, "question": text}
            for key in ("header", "detail"):
                candidate = item.get(key)
                if candidate is not None:
                    if not isinstance(candidate, str):
                        raise ValueError(f"question {question_id} {key} must be a string")
                    value[key] = candidate
            intent = item.get("intent")
            if intent is not None:
                if not isinstance(intent, dict):
                    raise ValueError(f"question {question_id} intent must be an object")
                value["intent"] = dict(intent)
            options = item.get("options")
            if options is not None:
                if not isinstance(options, list):
                    raise ValueError(f"question {question_id} options must be an array")
                normalized_options: list[JsonObject] = []
                labels: set[str] = set()
                for option_index, option in enumerate(options):
                    if not isinstance(option, dict):
                        raise ValueError(
                            f"question {question_id} option {option_index} must be an object"
                        )
                    label = option.get("label")
                    if not isinstance(label, str) or not label.strip():
                        raise ValueError(
                            f"question {question_id} option {option_index} label must be non-empty"
                        )
                    if label in labels:
                        raise ValueError(f"duplicate option label: {label}")
                    labels.add(label)
                    normalized_option: JsonObject = {"label": label}
                    description = option.get("description")
                    if description is not None:
                        if not isinstance(description, str):
                            raise ValueError(
                                f"question {question_id} option description must be a string"
                            )
                        normalized_option["description"] = description
                    normalized_options.append(normalized_option)
                value["options"] = normalized_options
            multi_select = item.get("multi_select")
            if multi_select is not None:
                if not isinstance(multi_select, bool):
                    raise ValueError(f"question {question_id} multi_select must be boolean")
                value["multiSelect"] = multi_select
            normalized.append(value)
        return normalized

    def _install_todo_tool(
        self,
        registry: ToolRegistry,
        session: Session,
    ) -> list[Callable[[], None]]:
        """Install the canonical whole-list ``todo_write`` model tool."""

        async def execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
            if context.session_id != session.id:
                raise ApiFault(
                    "subagent-unauthorized",
                    "todo tool context does not belong to the registered session",
                    {"sessionId": context.session_id},
                )
            try:
                write = self.todos.write(session, args.get("todos"))
            except TodoError as exc:
                raise ValueError(str(exc)) from exc
            await self.store.save(session)
            self._publish_event(session.id, write.event)
            value = write.value()
            counts = value["counts"]
            assert isinstance(counts, dict)
            return ToolResult(
                "Updated todo list: "
                f"{counts['pending']} pending, "
                f"{counts['inProgress']} in progress, "
                f"{counts['completed']} completed.",
                meta=value,
            )

        return [
            registry.register(
                ToolDefinition(
                    name="todo_write",
                    description=(
                        "Record and replace the complete structured task list for the current "
                        "work. Use pending, in_progress, or completed; send the entire list "
                        "every call."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "todos": {
                                "type": "array",
                                "description": (
                                    "The complete task list replacing the previous list."
                                ),
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "content": {"type": "string"},
                                        "status": {
                                            "type": "string",
                                            "enum": ["pending", "in_progress", "completed"],
                                        },
                                    },
                                    "required": ["content", "status"],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["todos"],
                        "additionalProperties": False,
                    },
                    execute=execute,
                )
            )
        ]

    def _install_subagent_tools(
        self,
        registry: ToolRegistry,
        session: Session,
    ) -> list[Callable[[], None]]:
        """Install the model-facing subagent and control tools for one session.

        The TS composition keeps these tools in separate Cordis packages.  The
        Python host has one per-session registry, so the equivalent lifecycle is
        expressed as a group of disposers owned by ``SessionHandle``.
        """

        async def run_subagent(
            args: dict[str, Any],
            context: ToolContext,
            *,
            inherit_context: bool,
        ) -> ToolResult:
            parent = await self._tool_parent(context, session)
            label = self._tool_text_argument(args, "description", maximum=160)
            prompt = self._tool_text_argument(args, "prompt", maximum=200_000)
            run_in_background = args.get("run_in_background", True)
            if not isinstance(run_in_background, bool):
                raise ValueError("run_in_background must be a boolean")
            if run_in_background:
                child_id, job_id = await self._start_background_subagent(
                    parent,
                    label,
                    prompt,
                    inherit_context=inherit_context,
                )
                return ToolResult(
                    f"started background subagent {child_id}",
                    meta={
                        "kind": "continuable",
                        "subagentId": child_id,
                        "jobId": job_id,
                    },
                )
            return await self._run_foreground_subagent(
                parent,
                label,
                prompt,
                inherit_context=inherit_context,
            )

        async def send_message(args: dict[str, Any], context: ToolContext) -> ToolResult:
            parent = await self._tool_parent(context, session)
            child_id = self._tool_text_argument(args, "subagent_id", maximum=160)
            message = self._tool_text_argument(args, "message", maximum=200_000)
            await self._require_continuable_child(parent.session.id, child_id)
            result = await self.prompt(
                child_id,
                [{"type": "text", "text": message}],
                allow_subagent=True,
                include_message_id=True,
            )
            message_id = result.get("messageId")
            if not isinstance(message_id, str):
                raise RuntimeError("subagent follow-up did not return a message id")
            return ToolResult(
                f"message queued as the next turn for subagent {child_id}",
                meta={"messageId": message_id, "subagentId": child_id},
            )

        async def interrupt_agent(args: dict[str, Any], context: ToolContext) -> ToolResult:
            parent = await self._tool_parent(context, session)
            target_id = self._tool_text_argument(args, "agent_id", maximum=160)
            if not await self._is_descendant(parent.session.id, target_id):
                raise ApiFault(
                    "subagent-unauthorized",
                    "agent is not a descendant of the calling session",
                    {"agentId": target_id},
                )
            target = await self.get_session(target_id)
            if target.task is not None and not target.task.done():
                target.task.cancel()
            return ToolResult(
                f"interrupt requested for agent {target_id}",
                meta={"accepted": True, "agentId": target_id},
            )

        async def list_agents(args: dict[str, Any], context: ToolContext) -> ToolResult:
            parent = await self._tool_parent(context, session)
            scope = args.get("scope", "children")
            if scope not in {"children", "descendants"}:
                raise ValueError("scope must be children or descendants")
            entries = await self._list_subagent_entries(parent.session.id, scope)
            if not entries:
                rendered = "(no subagents)"
            else:
                rendered_rows: list[str] = []
                for entry in entries:
                    position = ""
                    if scope == "descendants":
                        position = (
                            f" parent={entry.get('parent')} depth={entry.get('depth')}"
                        )
                    if entry.get("kind") == "diagnostic":
                        rendered_rows.append(
                            f"{entry['id']} [diagnostic: {entry.get('reason')}]{position}"
                        )
                    else:
                        rendered_rows.append(
                            f"{entry['id']} [{entry.get('status')}]{position}"
                            f" — {entry.get('label', entry['id'])}"
                        )
                rendered = "\n".join(rendered_rows)
            return ToolResult(rendered, meta={"entries": entries, "scope": scope})

        disposers: list[Callable[[], None]] = []
        for tool_name, description, inherit_context in (
            (
                "subagent",
                "Delegate a self-contained task to a subagent. It runs in the background by "
                "default and returns a durable subagent id; set run_in_background to false "
                "when the result is needed immediately.",
                False,
            ),
            (
                "subagent_fork",
                "Delegate a task to a subagent seeded with this conversation's completed turns. "
                "It runs in the background by default and returns a durable subagent id.",
                True,
            ),
        ):
            disposers.append(
                registry.register(
                    ToolDefinition(
                        name=tool_name,
                        description=description,
                        parameters={
                            "type": "object",
                            "properties": {
                                "description": {
                                    "type": "string",
                                    "description": "A short description of the delegated task.",
                                },
                                "prompt": {
                                    "type": "string",
                                    "description": "The complete task for the subagent.",
                                },
                                "run_in_background": {
                                    "type": "boolean",
                                    "description": (
                                        "Whether to return immediately with a durable subagent id."
                                    ),
                                },
                            },
                            "required": ["description", "prompt"],
                            "additionalProperties": False,
                        },
                        execute=lambda args, context, inherit_context=inherit_context: run_subagent(
                            args,
                            context,
                            inherit_context=inherit_context,
                        ),
                    )
                )
            )

        disposers.extend(
            [
                registry.register(
                    ToolDefinition(
                        name="send_message",
                        description=(
                            "Send the next-turn message to a background subagent by durable id."
                        ),
                        parameters={
                            "type": "object",
                            "properties": {
                                "subagent_id": {"type": "string"},
                                "message": {"type": "string"},
                            },
                            "required": ["subagent_id", "message"],
                            "additionalProperties": False,
                        },
                        execute=send_message,
                    )
                ),
                registry.register(
                    ToolDefinition(
                        name="interrupt_agent",
                        description="Request cancellation of a descendant subagent's current turn.",
                        parameters={
                            "type": "object",
                            "properties": {"agent_id": {"type": "string"}},
                            "required": ["agent_id"],
                            "additionalProperties": False,
                        },
                        execute=interrupt_agent,
                    )
                ),
                registry.register(
                    ToolDefinition(
                        name="list_agents",
                        description=(
                            "List direct or descendant continuable subagents and their status."
                        ),
                        parameters={
                            "type": "object",
                            "properties": {
                                "scope": {"type": "string", "enum": ["children", "descendants"]}
                            },
                            "additionalProperties": False,
                        },
                        execute=list_agents,
                    )
                ),
            ]
        )
        return disposers

    async def _tool_parent(self, context: ToolContext, expected: Session) -> SessionHandle:
        if context.session_id != expected.id:
            raise ApiFault(
                "subagent-unauthorized",
                "tool context does not belong to the registered session",
                {"sessionId": context.session_id},
            )
        return await self.get_session(context.session_id)

    @staticmethod
    def _tool_text_argument(args: dict[str, Any], name: str, *, maximum: int) -> str:
        value = args.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        if len(value) > maximum:
            raise ValueError(f"{name} exceeds the {maximum}-character limit")
        return value

    async def _prepare_subagent(
        self,
        parent: SessionHandle,
        label: str,
        *,
        mode: Literal["one-shot", "continuable"],
        inherit_context: bool,
        model_selection_override: JsonObject | None = None,
    ) -> SessionHandle:
        depth = await self._subagent_depth(parent.session)
        if depth >= MAX_SUBAGENT_DEPTH:
            raise ApiFault(
                "subagent-depth-limit",
                f"subagent maximum depth {MAX_SUBAGENT_DEPTH} has been reached",
                {"sessionId": parent.session.id, "depth": depth},
            )
        model_selection = model_selection_override or parent.session.header.model_selection
        if inherit_context:
            end_seq = self._fork_end_seq(parent.session, None)
            prefix = list(parent.session.events[: end_seq + 1])
            child_id = f"session-{uuid.uuid4().hex}"
            header = replace(
                Session.header_for(
                    child_id,
                    cwd=parent.session.header.cwd,
                    parent_session=parent.session.id,
                    origin="subagent",
                    agent_preset=parent.session.header.agent_preset,
                    model_selection=model_selection,
                ),
                seed_length=len(prefix),
            )
            child = Session(child_id, header=header, events=prefix)
            await self.store.save(child)
            async with self._lock:
                child_handle = self._attach(child)
            self._publish_host(
                {
                    "type": "host/session-added",
                    "sessionId": child.id,
                    "blank": self._is_blank(child),
                    "cwd": child.header.cwd,
                    "parentSessionId": parent.session.id,
                    "origin": "subagent",
                }
            )
        else:
            child_handle = await self.create_session(
                cwd=parent.session.header.cwd,
                parent_session=parent.session.id,
                origin="subagent",
                agent_preset=parent.session.header.agent_preset,
                model_selection=model_selection,
            )
            child = child_handle.session

        descriptor: JsonObject = {
            "version": SUBAGENT_DESCRIPTOR_VERSION,
            "mode": mode,
            "provider": "python-in-process",
            "label": label,
        }
        descriptor_event = child.append("subagent/descriptor", descriptor)
        await self.store.save(child)
        self._publish_event(child.id, descriptor_event)
        started_event = parent.session.append(
            "subagent/started",
            {
                "childSessionId": child.id,
                "label": label,
                "mode": mode,
                "provider": "python-in-process",
            },
        )
        await self.store.save(parent.session)
        self._publish_event(parent.session.id, started_event)
        return child_handle

    async def _run_foreground_subagent(
        self,
        parent: SessionHandle,
        label: str,
        prompt: str,
        *,
        inherit_context: bool,
    ) -> ToolResult:
        child = await self._prepare_subagent(
            parent,
            label,
            mode="one-shot",
            inherit_context=inherit_context,
        )
        task = asyncio.create_task(
            child.agent.run(prompt),
            name=f"dsh-subagent-{child.session.id}",
        )
        child.task = task
        try:
            result = await task
        finally:
            if child.task is task:
                child.task = None
            await self.store.save(child.session)
        output = result.final_response.strip()
        reason = result.finish_reason or "completed"
        if reason == "completed":
            text = f"subagent {child.session.id} completed"
            if output:
                text += f":\n{output}"
            return ToolResult(
                text,
                meta={
                    "kind": "foreground",
                    "subagentId": child.session.id,
                    "finishReason": reason,
                },
            )
        text = f"subagent {child.session.id} ended with {reason}"
        if output:
            text += f"\nPartial output:\n{output}"
        return ToolResult(
            text,
            is_error=True,
            meta={
                "kind": "foreground",
                "subagentId": child.session.id,
                "finishReason": reason,
            },
        )

    async def _run_fresh_subagent(
        self,
        parent: SessionHandle,
        label: str,
        prompt: str,
        *,
        on_started: Callable[[str], Awaitable[None]] | None = None,
        model_selection: JsonObject | None = None,
    ) -> tuple[str, str, str]:
        """Run one blank one-shot child and expose its raw final text."""

        child = await self._prepare_subagent(
            parent,
            label,
            mode="one-shot",
            inherit_context=False,
            model_selection_override=model_selection,
        )
        if on_started is not None:
            await on_started(child.session.id)
        task = asyncio.create_task(
            child.agent.run(prompt),
            name=f"dsh-ralph-{child.session.id}",
        )
        child.task = task
        try:
            result = await task
        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        finally:
            if child.task is task:
                child.task = None
            await self.store.save(child.session)
        return child.session.id, result.finish_reason or "completed", result.final_response.strip()

    async def _start_background_subagent(
        self,
        parent: SessionHandle,
        label: str,
        prompt: str,
        *,
        inherit_context: bool,
    ) -> tuple[str, str]:
        child_ref: dict[str, SessionHandle] = {}

        async def starter() -> JobHandle:
            child = await self._prepare_subagent(
                parent,
                label,
                mode="continuable",
                inherit_context=inherit_context,
            )
            child_ref["handle"] = child
            await self.prompt(
                child.session.id,
                [{"type": "text", "text": prompt}],
                allow_subagent=True,
            )
            task = child.task
            if task is None:
                raise RuntimeError("background subagent task was not scheduled")

            async def settle() -> JobOutcome:
                try:
                    await task
                except asyncio.CancelledError:
                    return JobOutcome("killed", "subagent turn was cancelled")
                except Exception as exc:
                    return JobOutcome("failed", str(exc))
                finally:
                    await self.store.save(child.session)
                reason = self._last_turn_reason(child.session) or "completed"
                output = self._latest_assistant_text(child.session)
                if reason == "aborted":
                    return JobOutcome("killed", "subagent turn was cancelled", output)
                if reason not in {"completed", "max-tokens"}:
                    return JobOutcome("failed", f"subagent ended with {reason}", output)
                return JobOutcome("completed", f"subagent ended with {reason}", output)

            def cancel(_reason: str | None) -> None:
                task.cancel()

            return JobHandle(cancel=cancel, done=settle())

        job_id = await self.jobs.start(
            kind="subagent",
            label=label,
            owner_session=parent.session.id,
            starter=starter,
        )
        child = child_ref.get("handle")
        if child is None:
            raise RuntimeError("background subagent did not publish a child session")
        return child.session.id, job_id

    async def _require_continuable_child(
        self,
        parent_session: str,
        child_session: str,
    ) -> SessionHandle:
        child = await self._require_child(parent_session, child_session)
        descriptor = self._subagent_descriptor(child.session)
        if descriptor is None or descriptor.get("mode") != "continuable":
            raise ApiFault(
                "subagent-not-resumable",
                "subagent does not have a continuable descriptor",
                {"childSessionId": child_session},
            )
        return child

    async def _subagent_depth(self, session: Session) -> int:
        depth = 0
        current = session
        visited: set[str] = set()
        while current.header.parent_session:
            parent_id = current.header.parent_session
            if parent_id in visited:
                break
            visited.add(parent_id)
            depth += 1
            try:
                current = (await self.get_session(parent_id)).session
            except ApiFault:
                break
        return depth

    async def _is_descendant(self, ancestor_id: str, candidate_id: str) -> bool:
        if ancestor_id == candidate_id:
            return False
        try:
            current = (await self.get_session(candidate_id)).session
        except ApiFault:
            return False
        visited: set[str] = set()
        while current.header.parent_session:
            parent_id = current.header.parent_session
            if parent_id == ancestor_id:
                return True
            if parent_id in visited:
                return False
            visited.add(parent_id)
            try:
                current = (await self.get_session(parent_id)).session
            except ApiFault:
                return False
        return False

    async def _list_subagent_entries(
        self,
        parent_session: str,
        scope: Literal["children", "descendants"],
    ) -> list[JsonObject]:
        entries: list[JsonObject] = []
        pending: list[tuple[str, int]] = [(parent_session, 0)]
        while pending:
            current_id, current_depth = pending.pop(0)
            for child in await self._child_sessions(current_id):
                descriptor = self._subagent_descriptor(child)
                child_depth = current_depth + 1
                if descriptor is not None:
                    handle = self._handles.get(child.id)
                    if handle is None:
                        status = "ready"
                    elif handle.agent.status == "running":
                        status = "running"
                    else:
                        status = "idle"
                    entry: JsonObject = {
                        "kind": "child",
                        "id": child.id,
                        "label": descriptor.get("label", child.id),
                        "status": status,
                        "mode": descriptor.get("mode", "one-shot"),
                    }
                    if scope == "descendants":
                        entry["parent"] = current_id
                        entry["depth"] = child_depth
                    entries.append(entry)
                if scope == "descendants":
                    pending.append((child.id, child_depth))
        return entries

    @staticmethod
    def _subagent_descriptor(session: Session) -> JsonObject | None:
        # A forked child replays its ancestor's model-hidden events as a seed.
        # Only the descriptor in the child's own suffix classifies it, matching
        # the TS persistence rule for cold-resumable children.
        own_events = session.events[session.header.seed_length or 0 :]
        for event in own_events:
            if event.type != "subagent/descriptor":
                continue
            data = event.data
            if data.get("version") != SUBAGENT_DESCRIPTOR_VERSION:
                return None
            if data.get("mode") not in {"one-shot", "continuable"}:
                return None
            if not isinstance(data.get("provider"), str):
                return None
            label = data.get("label")
            if label is not None and not isinstance(label, str):
                return None
            if data.get("mode") == "continuable" and not isinstance(label, str):
                return None
            return {
                "version": SUBAGENT_DESCRIPTOR_VERSION,
                "mode": data["mode"],
                "provider": data["provider"],
                **({"label": label} if isinstance(label, str) else {}),
            }
        return None

    @staticmethod
    def _last_turn_reason(session: Session) -> str | None:
        for event in reversed(session.events):
            if event.type != "turn/end":
                continue
            reason = event.data.get("reason")
            if isinstance(reason, dict):
                kind = reason.get("kind")
                if isinstance(kind, str):
                    return kind
        return None

    @staticmethod
    def _latest_assistant_text(session: Session) -> str | None:
        messages = [message for message in session.derive_messages() if message.role == "assistant"]
        if not messages:
            return None
        text = messages[-1].text.strip()
        return text or None

    async def _build_message(
        self,
        content: list[JsonObject],
        *,
        client_time_zone: str | None = None,
    ) -> Message:
        if not content:
            raise ApiFault("bad-request", "prompt content cannot be empty")
        if not all(isinstance(item, dict) for item in content):
            raise ApiFault("bad-request", "prompt content blocks must be objects")
        image_parts = [item for item in content if item.get("type") == "image"]
        if len(image_parts) > self.attachments.max_images_per_message:
            raise ApiFault("attachment-error", "prompt contains too many images")
        blocks: list[Any] = []
        image_bytes = 0
        for item in content:
            kind = item.get("type")
            if kind == "text":
                text = item.get("text")
                if not isinstance(text, str):
                    raise ApiFault("bad-request", "text content must contain a string text")
                blocks.append(TextContent(text))
                continue
            if kind != "image":
                raise ApiFault("bad-request", "prompt content block type is unsupported")
            media_type = item.get("mediaType")
            data = item.get("data")
            if not isinstance(media_type, str) or not isinstance(data, str):
                raise ApiFault(
                    "bad-request", "image content must contain mediaType and base64 data"
                )
            if "name" in item and not isinstance(item["name"], str):
                raise ApiFault("bad-request", "image content name must be a string")
            try:
                raw = AttachmentStore.decode_base64(data)
            except AttachmentError as exc:
                raise ApiFault("attachment-error", str(exc), {"reason": exc.code}) from exc
            image_bytes += len(raw)
            try:
                ref = self.attachments.save(
                    raw,
                    media_type,
                    name=item.get("name") if isinstance(item.get("name"), str) else None,
                )
            except AttachmentError as exc:
                raise ApiFault("attachment-error", str(exc), {"reason": exc.code}) from exc
            block = ImageContent(ref.to_dict(), data)
            blocks.append(block)
        if image_bytes > self.attachments.max_message_image_bytes:
            raise ApiFault("attachment-error", "prompt image bytes exceed the configured limit")
        source: JsonObject = {"kind": "user"}
        if client_time_zone:
            source["clientTimeZone"] = client_time_zone
        message = Message("user", tuple(blocks), source)
        return message

    @staticmethod
    def _register_message_attachments(session: Session, message: Message) -> None:
        # The Session is the authoritative short-lived bridge used by Agent's
        # derive_messages() after it appends the durable event.
        for block in message.content:
            if isinstance(block, ImageContent):
                attachment_id = block.attachment.get("attachmentId")
                if isinstance(attachment_id, str) and block.data is not None:
                    session.register_attachment_data(attachment_id, block.data)

    def _attach(self, session: Session) -> SessionHandle:
        workspace = Path(session.header.cwd or self.cwd).expanduser().resolve()
        registry = ToolRegistry()
        permission_mode = self._effective_permission_mode()
        policy = WorkspacePolicy(workspace, permission_mode)
        disposers = install_builtin_tools(
            registry,
            policy,
            enable_shell=permission_mode is PermissionMode.DANGER_FULL_ACCESS,
            jobs=self.jobs,
        )
        self._tool_registries[session.id] = registry

        def dispose_tool_registry() -> None:
            self._tool_registries.pop(session.id, None)

        disposers.append(dispose_tool_registry)
        self._goal_activation.setdefault(session.id, "disarmed")
        disposers.extend(self._install_goal_tools(registry, session))
        disposers.extend(self._install_plan_tools(registry, session))
        disposers.extend(self._install_workflow_tool(registry, session))
        disposers.extend(self._install_ralph_tool(registry, session))
        disposers.extend(self._install_ask_user_tool(registry, session))
        disposers.extend(self._install_todo_tool(registry, session))
        disposers.extend(self._install_subagent_tools(registry, session))
        disposers.extend(install_dynamic_tools(registry, self.dynamic, session.id))
        selection = session.header.model_selection or self._default_selection()
        provider = selection.get("provider")
        selected_model = selection.get("model")
        provider_name = provider if isinstance(provider, str) else "deepseek-official"
        model_name = selected_model if isinstance(selected_model, str) else self.model
        llm_settings = self.settings.get_value_sync("llm-deepseek")
        configured_max_tokens = llm_settings.get("maxTokens")
        configured_thinking = llm_settings.get("thinking")
        configured_effort = selection.get("reasoningEffort") or llm_settings.get("reasoningEffort")
        max_tokens = configured_max_tokens if isinstance(configured_max_tokens, int) else None
        thinking = configured_thinking if configured_thinking in {"enabled", "disabled"} else None
        reasoning_effort = (
            cast(Literal["off", "high", "max"], configured_effort)
            if configured_effort in {"off", "high", "max"}
            else None
        )
        base_system_prompt = (
            "You are a coding agent powered by the DeepSeek model. "
            f"Your working directory is {workspace}."
            + (
                f" Your agent preset is {session.header.agent_preset}."
                if session.header.agent_preset
                else ""
            )
        )

        def system_prompt() -> str:
            if not fold_plan_mode(session).active:
                return base_system_prompt
            return base_system_prompt + (
                "\n\nPlan mode is active. Explore the workspace and reason about the "
                "implementation "
                "before changing files. Present the complete plan with exit_plan_mode for user "
                "review; if the user keeps planning, revise the plan and present it again."
            )

        agent = Agent(
            session,
            self._adapter_factory(model_name),
            tools=registry,
            config=LlmCallConfig(
                provider=provider_name,
                model=model_name,
                max_tokens=max_tokens,
                thinking=thinking,
                reasoning_effort=reasoning_effort,
            ),
            system_prompt=system_prompt,
        )
        handle = SessionHandle(session, agent, disposers)
        agent.subscribe(lambda event: self._publish_event(session.id, event))
        self._handles[session.id] = handle
        return handle

    def _default_adapter(self, model: str) -> LlmAdapter:
        config = self.settings.get_value_sync("llm-deepseek")
        key_ref = config.get("apiKeyEnv")
        api_key = self.credentials.resolve_sync(
            key_ref if isinstance(key_ref, str) else "DEEPSEEK_API_KEY"
        )
        configured_url = config.get("baseURL")
        base_url = configured_url if isinstance(configured_url, str) else None
        return DeepSeekAdapter(api_key=api_key, base_url=base_url, timeout=120.0)

    def _effective_permission_mode(self) -> PermissionMode:
        value = self.settings.get_value_sync("permission").get("defaultPreset")
        if isinstance(value, str):
            try:
                return PermissionMode(value)
            except ValueError:
                pass
        return self.permission_mode

    def _publish_event(self, session_id: str, event: SessionEvent) -> None:
        self._publish_mux(
            {"type": "session/event", "sessionId": session_id, "event": event.to_dict()}
        )
        handle = self._handles.get(session_id)
        if handle is None:
            return
        if event.type in {"todo/write", "turn/start"}:
            self._publish_todo_projection(handle)
        if event.type in {"plan/mode", "command/run", "turn/start"}:
            self._publish_plan_projection(handle)
        if event.type == "turn/start":
            state = fold_plan_mode(handle.session)
            if state.pending and state.wanted is not None:
                mode_event = handle.session.append("plan/mode", {"active": state.wanted})
                self._publish_event(session_id, mode_event)

    def _publish_mux(self, frame: Frame) -> None:
        for queue in tuple(self._mux_subscribers):
            queue.put_nowait(frame)

    def _publish_host(self, frame: Frame) -> None:
        for queue in tuple(self._host_subscribers):
            queue.put_nowait(frame)

    def _publish_remote_event(self, event: str, args: list[Any]) -> None:
        self._publish_host({"type": "host/remote-event", "event": event, "args": args})

    def _publish_queue(self, handle: SessionHandle) -> None:
        self._publish_mux(
            {
                "type": "session/queue",
                "sessionId": handle.session.id,
                "items": [item.to_dict() for item in handle.queue],
            }
        )

    def _jobs_changed(self, owner_session: str | None) -> None:
        """Mirror the registry's whole visible job set to the mux stream."""

        if owner_session is None:
            session_ids = tuple(self._handles)
        elif owner_session in self._handles:
            session_ids = (owner_session,)
        else:
            return
        for session_id in session_ids:
            self._publish_mux(
                {
                    "type": "session/jobs",
                    "sessionId": session_id,
                    "jobs": [job.to_dict() for job in self.jobs.list(session_id)],
                }
            )

    def _publish_goal_projection(self, handle: SessionHandle) -> None:
        self._publish_mux(
            {
                "type": "session/projection",
                "sessionId": handle.session.id,
                "key": "goal",
                "value": self.goals.fold(handle.session).projection(),
                "seq": handle.session.seq - 1,
            }
        )

    def _publish_plan_projection(self, handle: SessionHandle) -> None:
        self._publish_mux(
            {
                "type": "session/projection",
                "sessionId": handle.session.id,
                "key": "plan",
                "value": fold_plan_mode(handle.session).to_dict(),
                "seq": handle.session.seq - 1,
            }
        )

    def _publish_todo_projection(self, handle: SessionHandle) -> None:
        self._publish_mux(
            {
                "type": "session/projection",
                "sessionId": handle.session.id,
                "key": "todos",
                "value": self.todos.fold(handle.session),
                "seq": handle.session.seq - 1,
            }
        )

    def _projection_values(self, session: Session) -> JsonObject:
        return {
            "goal": self.goals.fold(session).projection(),
            "plan": fold_plan_mode(session).to_dict(),
            "todos": self.todos.fold(session),
            "imageLimits": self._image_limits(),
        }

    def _image_limits(self) -> JsonObject:
        return {
            "maxImageBytes": self.attachments.max_image_bytes,
            "maxImagesPerMessage": self.attachments.max_images_per_message,
            "maxMessageImageBytes": self.attachments.max_message_image_bytes,
            "maxImagePixels": self.attachments.max_image_pixels,
            "mediaTypes": list(IMAGE_MEDIA_TYPES),
        }

    def _publish_settings_changed(self, namespace: str, view: JsonObject) -> None:
        revision = view.get("revision", 0)
        self._publish_host(
            {
                "type": "host/remote-event",
                "event": "settings/document-updated",
                "args": [namespace, revision if isinstance(revision, int) else 0],
            }
        )

    def _child_handles(self, parent_session: str) -> list[tuple[Session, SessionHandle]]:
        return [
            (handle.session, handle)
            for handle in self._handles.values()
            if handle.session.header.parent_session == parent_session
        ]

    async def _child_sessions(self, parent_session: str) -> list[Session]:
        children: list[Session] = []
        for session_id in await self.store.list_ids():
            try:
                session = await self.store.load(session_id)
            except (OSError, ValueError):
                continue
            if session.header.parent_session == parent_session:
                children.append(session)
        return children

    async def _require_child(self, parent_session: str, child_session: str) -> SessionHandle:
        await self.get_session(parent_session)
        try:
            handle = await self.get_session(child_session)
        except ApiFault as exc:
            if exc.code == "session-not-found":
                raise ApiFault(
                    "subagent-not-found",
                    str(exc),
                    {"parentSessionId": parent_session, "childSessionId": child_session},
                ) from exc
            raise
        if handle.session.header.parent_session != parent_session:
            raise ApiFault(
                "subagent-unauthorized",
                "child session does not belong to the requested parent",
                {"childSessionId": child_session},
            )
        return handle

    @staticmethod
    def _check_session_cwd(session: Session, cwd: Path) -> None:
        existing = session.header.cwd
        if existing is not None and Path(existing).expanduser().resolve() != cwd:
            raise ApiFault(
                "session-conflict",
                f"session {session.id} belongs to another workspace",
                {"sessionId": session.id, "requestedCwd": str(cwd), "existingCwd": existing},
            )

    async def _get_preset(self, preset_id: str):
        try:
            return await self.presets.get(preset_id)
        except AgentPresetError as exc:
            raise self._preset_fault(exc) from exc

    async def _require_preset(self, preset_id: str) -> None:
        await self._get_preset(preset_id)

    @staticmethod
    def _preset_fault(exc: AgentPresetError) -> ApiFault:
        details: JsonObject = {}
        if exc.preset is not None:
            details["agentPreset"] = exc.preset
        if exc.code == "agent-preset-not-found":
            details["available"] = []
        if exc.code in {"agent-preset-invalid", "agent-preset-read-only"}:
            details["reason"] = str(exc)
        return ApiFault(exc.code, str(exc), details)

    @staticmethod
    def _check_session_preset(session: Session, requested: str | None) -> None:
        existing = session.header.agent_preset
        if requested is not None and existing is not None and requested != existing:
            raise ApiFault(
                "agent-preset-conflict",
                f"session {session.id} already uses agent preset {existing}",
                {
                    "sessionId": session.id,
                    "requestedPreset": requested,
                    "existingPreset": existing,
                },
            )
        if requested is not None and existing is None:
            raise ApiFault(
                "agent-preset-conflict",
                f"session {session.id} records no agent preset",
                {
                    "sessionId": session.id,
                    "requestedPreset": requested,
                },
            )

    @staticmethod
    def _is_blank(session: Session) -> bool:
        return not any(event.type == "turn/start" for event in session.events)

    @staticmethod
    def _message_groups(events: list[SessionEvent]) -> list[list[SessionEvent]]:
        groups: list[list[SessionEvent]] = []
        for event in events:
            if event.type == "user/message" and groups:
                groups.append([])
            if not groups:
                groups.append([])
            groups[-1].append(event)
        return groups

    @staticmethod
    def _fork_end_seq(session: Session, at_seq: int | None) -> int:
        completed: list[int] = []
        for event in session.events:
            if event.type != "turn/end":
                continue
            reason = event.data.get("reason")
            if isinstance(reason, dict) and reason.get("kind") in {"completed", "max-tokens"}:
                completed.append(event.seq)
        if not completed:
            raise ApiFault("fork-unavailable", "session has no completed turn")
        if at_seq is None or at_seq >= session.seq:
            return completed[-1]
        for end_seq in completed:
            if end_seq >= at_seq:
                return end_seq
        raise ApiFault("fork-unavailable", "the requested fork point is in an open turn")

    @staticmethod
    def _referenced_attachment(
        session: Session,
        attachment_id: str,
    ) -> ImageAttachment | None:
        def visit(value: Any) -> ImageAttachment | None:
            if isinstance(value, dict):
                if value.get("type") == "image" and isinstance(value.get("attachment"), dict):
                    raw = value["attachment"]
                    if raw.get("attachmentId") == attachment_id:
                        required = (
                            "mediaType",
                            "bytes",
                            "width",
                            "height",
                        )
                        if isinstance(raw.get("mediaType"), str) and all(
                            isinstance(raw.get(key), int) for key in required[1:]
                        ):
                            return ImageAttachment(
                                attachment_id,
                                raw["mediaType"],
                                raw["bytes"],
                                raw["width"],
                                raw["height"],
                                raw.get("name") if isinstance(raw.get("name"), str) else None,
                            )
                for child in value.values():
                    found = visit(child)
                    if found is not None:
                        return found
            elif isinstance(value, list):
                for child in value:
                    found = visit(child)
                    if found is not None:
                        return found
            return None

        for event in session.events:
            found = visit(event.data)
            if found is not None:
                return found
        return None

    @staticmethod
    def _prompt_text(content: list[JsonObject]) -> str:
        parts: list[str] = []
        for item in content:
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)

    def _models(self, handle: SessionHandle | None = None) -> JsonObject:
        selection = (
            dict(handle.session.header.model_selection)
            if handle is not None and handle.session.header.model_selection
            else self._default_selection()
        )
        return {
            "current": selection,
            "routable": bool(self.credentials.resolve_sync("DEEPSEEK_API_KEY")),
            "groups": [
                {
                    "id": "deepseek-official",
                    "name": "DeepSeek",
                    "models": [
                        {
                            "id": "deepseek-v4-flash",
                            "name": "DeepSeek-V4-Flash",
                            "contextWindow": 1_000_000,
                            "reasoning": {
                                "efforts": [
                                    {"id": "off", "name": "Off"},
                                    {"id": "high", "name": "High"},
                                    {"id": "max", "name": "Max"},
                                ],
                                "defaultEffort": "high",
                            },
                        },
                        {
                            "id": "deepseek-v4-pro",
                            "name": "DeepSeek-V4-Pro",
                            "contextWindow": 1_000_000,
                            "reasoning": {
                                "efforts": [
                                    {"id": "off", "name": "Off"},
                                    {"id": "high", "name": "High"},
                                    {"id": "max", "name": "Max"},
                                ],
                                "defaultEffort": "high",
                            },
                        },
                        {"id": "deepseek-chat", "name": "DeepSeek Chat"},
                        {"id": "deepseek-reasoner", "name": "DeepSeek Reasoner"},
                    ],
                }
            ],
            "failures": [],
        }

    def _default_selection(self) -> JsonObject:
        value = self.settings.get_value_sync("agent-default-model")
        provider = value.get("provider")
        model = value.get("model")
        return {
            "provider": provider if isinstance(provider, str) else "deepseek-official",
            "model": model if isinstance(model, str) else self.model,
        }

    async def _discover_models(self, payload: JsonObject) -> JsonObject:
        settings_ns = self._required_string(payload, "settingsNs")
        base_url = self._optional_payload_string(payload, "baseURL")
        if not base_url:
            try:
                settings = await self.settings.get(settings_ns)
                value = settings.get("value")
                if isinstance(value, dict) and isinstance(value.get("baseURL"), str):
                    base_url = value["baseURL"]
            except SettingsNotFound as exc:
                raise ApiFault(
                    "model-discovery-failed",
                    str(exc),
                    {"settingsNs": settings_ns},
                ) from exc
        base_url = (base_url or "https://api.deepseek.com").rstrip("/")
        api_key = self._optional_payload_string(payload, "apiKey")
        if not api_key:
            ref = "DEEPSEEK_API_KEY"
            api_key = await self.credentials.resolve(ref)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(f"{base_url}/models", headers=headers)
            if response.status_code >= 400:
                raise RuntimeError(f"provider returned HTTP {response.status_code}")
            raw = response.json()
            rows = raw.get("data", []) if isinstance(raw, dict) else []
            models: list[JsonObject] = []
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                        continue
                    item: JsonObject = {"id": row["id"]}
                    if isinstance(row.get("name"), str):
                        item["name"] = row["name"]
                    models.append(item)
            return {"models": models}
        except (httpx.HTTPError, ValueError, RuntimeError) as exc:
            raise ApiFault(
                "model-discovery-failed",
                str(exc),
                {"settingsNs": settings_ns, "baseURL": base_url},
            ) from exc

    def _providers(self) -> list[JsonObject]:
        return [
            {
                "provider": "deepseek-official",
                "displayName": "DeepSeek",
                "settingsNs": "llm-deepseek",
                "settingsPath": [],
                "active": bool(self.credentials.resolve_sync("DEEPSEEK_API_KEY")),
                "declared": True,
            }
        ]

    def _list_directory(self, raw_path: str | None) -> JsonObject:
        path = Path(raw_path or Path.home()).expanduser().resolve()
        if not path.is_dir():
            raise ApiFault("directory-unreadable", f"directory is not readable: {path}")
        entries = sorted(path.iterdir(), key=lambda item: item.name.casefold())
        return {
            "path": str(path),
            "home": str(Path.home()),
            "crumbs": self._crumbs(path),
            "entries": [
                {"name": item.name, "path": str(item), "hidden": item.name.startswith(".")}
                for item in entries[:500]
            ],
            "truncated": len(entries) > 500,
        }

    @staticmethod
    def _crumbs(path: Path) -> list[JsonObject]:
        crumbs: list[JsonObject] = []
        for part in path.parts:
            current = Path(part) if not crumbs else Path(crumbs[-1]["path"]) / part
            crumbs.append({"name": part, "path": str(current), "hidden": False})
        return crumbs

    @staticmethod
    def _create_directory(parent: str, name: str) -> JsonObject:
        if not name.strip() or name in {".", ".."} or "/" in name or "\\" in name:
            raise ApiFault("bad-request", "directory name must be one path segment")
        target = (Path(parent).expanduser().resolve() / name).resolve()
        if target.exists():
            raise ApiFault("directory-exists", f"directory already exists: {target}")
        try:
            target.mkdir()
        except OSError as exc:
            raise ApiFault("directory-create-failed", str(exc)) from exc
        return {"path": str(target)}

    @staticmethod
    def _required_string(payload: JsonObject, key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise ApiFault("bad-request", f"{key} must be a non-empty string")
        return value

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        return value if isinstance(value, str) else None

    @staticmethod
    def _optional_payload_string(payload: JsonObject, key: str) -> str | None:
        if key not in payload:
            return None
        value = payload[key]
        if not isinstance(value, str):
            raise ApiFault("bad-request", f"{key} must be a string")
        return value

    @staticmethod
    def _optional_payload_int(
        payload: JsonObject,
        key: str,
        *,
        positive: bool = False,
        nonnegative: bool = False,
    ) -> int | None:
        if key not in payload:
            return None
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ApiFault("bad-request", f"{key} must be an integer")
        if positive and value <= 0:
            raise ApiFault("bad-request", f"{key} must be positive")
        if nonnegative and value < 0:
            raise ApiFault("bad-request", f"{key} must be non-negative")
        return value

    def _ensure_open(self) -> None:
        if self._disposed:
            raise RuntimeError("harness service is disposed")
