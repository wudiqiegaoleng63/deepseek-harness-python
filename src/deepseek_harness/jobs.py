"""Process-local background jobs.

The TypeScript host treats background work as a first-class capability rather
than as an untracked ``asyncio`` task.  This module keeps the same useful
boundary for the Python host: producers own their resources, while the
registry owns ids, ownership checks, lifecycle snapshots, output cursors and
observation callbacks.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

JobStatus = Literal["running", "stopping", "completed", "killed", "failed"]
JobOutcomeStatus = Literal["completed", "killed", "failed"]


@dataclass(frozen=True, slots=True)
class JobOutcome:
    """Terminal result supplied by a background producer."""

    status: JobOutcomeStatus
    detail: str | None = None
    output: str | None = None


@dataclass(frozen=True, slots=True)
class JobSnapshot:
    """Wire-safe view of one background job."""

    id: str
    kind: str
    label: str
    status: JobStatus
    detail: str | None
    started_at: int
    finished_at: int | None

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "startedAt": self.started_at,
        }
        if self.detail is not None:
            value["detail"] = self.detail
        if self.finished_at is not None:
            value["finishedAt"] = self.finished_at
        return value


@dataclass(frozen=True, slots=True)
class JobRead:
    text: str
    snapshot: JobSnapshot


@dataclass(slots=True)
class JobHandle:
    """Producer-owned execution handle.

    ``cancel`` is deliberately synchronous, matching the TS producer seam;
    the producer's ``done`` awaitable is the authoritative release boundary.
    """

    cancel: Callable[[str | None], None]
    done: Awaitable[JobOutcome]
    read_output: Callable[[], str] | None = None


@dataclass(slots=True)
class _JobRecord:
    id: str
    kind: str
    label: str
    owner_session: str | None
    handle: JobHandle
    status: JobStatus = "running"
    detail: str | None = None
    output: str | None = None
    started_at: int = 0
    finished_at: int | None = None
    reported: bool = False
    settled: asyncio.Event = field(default_factory=asyncio.Event)


JobChangeListener = Callable[[str | None], None]
JobDoneListener = Callable[[JobSnapshot, str | None], None]


class JobRegistry:
    """In-memory registry with session-scoped access and lifecycle control."""

    def __init__(
        self,
        *,
        on_changed: JobChangeListener | None = None,
        on_done: JobDoneListener | None = None,
        max_concurrent_per_owner: int = 10,
    ) -> None:
        if max_concurrent_per_owner <= 0:
            raise ValueError("max_concurrent_per_owner must be positive")
        self.max_concurrent_per_owner = max_concurrent_per_owner
        self._jobs: dict[str, _JobRecord] = {}
        self._counters: dict[str, int] = {}
        self._watchers: set[asyncio.Task[None]] = set()
        self._on_changed = on_changed
        self._on_done = on_done
        self._closed = False

    async def start(
        self,
        *,
        kind: str,
        label: str,
        owner_session: str | None,
        starter: Callable[[], Awaitable[JobHandle]],
    ) -> str:
        """Start and register a producer-owned job.

        The producer is started before an id is committed.  A spawn failure
        therefore leaves no phantom job in the registry.
        """

        if self._closed:
            raise RuntimeError("background job registry is closed")
        if not kind.strip() or not label.strip():
            raise ValueError("job kind and label must be non-empty")
        if self._active_count(owner_session) >= self.max_concurrent_per_owner:
            raise RuntimeError("background job limit reached for this session")
        handle = await starter()
        count = self._counters.get(kind, 0) + 1
        self._counters[kind] = count
        job_id = f"{kind}-{count}"
        record = _JobRecord(
            id=job_id,
            kind=kind,
            label=label,
            owner_session=owner_session,
            handle=handle,
            started_at=_now_ms(),
        )
        self._jobs[job_id] = record
        watcher = asyncio.create_task(self._watch(record), name=f"dsh-job-{job_id}")
        self._watchers.add(watcher)
        watcher.add_done_callback(self._watchers.discard)
        self._notify_changed(owner_session)
        return job_id

    def list(self, caller_session: str | None = None) -> list[JobSnapshot]:
        return [
            self._snapshot(job)
            for job in self._jobs.values()
            if job.owner_session is None or job.owner_session == caller_session
        ]

    def get(self, job_id: str, caller_session: str | None = None) -> JobSnapshot:
        job = self._expect(job_id)
        self._assert_access(job, caller_session)
        return self._snapshot(job)

    def read(self, job_id: str, caller_session: str | None = None) -> JobRead:
        job = self._expect(job_id)
        self._assert_access(job, caller_session)
        text = job.handle.read_output() if job.handle.read_output is not None else ""
        if job.status != "running" and job.status != "stopping":
            if not text and job.output is not None:
                text = job.output
            job.reported = True
        return JobRead(text, self._snapshot(job))

    def kill(
        self,
        job_id: str,
        caller_session: str | None = None,
        reason: str | None = None,
    ) -> Literal["requested", "already-finished"]:
        job = self._expect(job_id)
        self._assert_access(job, caller_session)
        if job.status not in {"running", "stopping"}:
            job.reported = True
            return "already-finished"
        # Match the TS contract: a producer cancel error does not silently
        # claim that stopping succeeded.
        job.handle.cancel(reason)
        job.status = "stopping"
        job.reported = True
        self._notify_changed(job.owner_session)
        return "requested"

    async def wait(
        self,
        job_id: str,
        timeout_ms: float,
        caller_session: str | None = None,
    ) -> JobSnapshot:
        job = self._expect(job_id)
        self._assert_access(job, caller_session)
        if timeout_ms <= 0 or timeout_ms != timeout_ms or timeout_ms == float("inf"):
            raise ValueError("timeout_ms must be a positive finite number")
        if job.status in {"running", "stopping"}:
            try:
                await asyncio.wait_for(job.settled.wait(), timeout_ms / 1000)
            except TimeoutError:
                return self._snapshot(job)
        job.reported = True
        return self._snapshot(job)

    async def close(self) -> None:
        """Stop live producers and await their release boundary."""

        if self._closed:
            return
        self._closed = True
        live = [job for job in self._jobs.values() if job.status in {"running", "stopping"}]
        for job in live:
            try:
                job.handle.cancel("jobs service disposed")
                if job.status == "running":
                    job.status = "stopping"
                    self._notify_changed(job.owner_session)
            except Exception as exc:  # pragma: no cover - defensive teardown path
                await self._force_fail(job, f"cancel threw during teardown: {exc}")
        await asyncio.gather(*(job.settled.wait() for job in live), return_exceptions=True)
        for owner in {job.owner_session for job in self._jobs.values()}:
            self._notify_changed(owner)
        self._jobs.clear()
        await asyncio.gather(*tuple(self._watchers), return_exceptions=True)

    async def _watch(self, job: _JobRecord) -> None:
        try:
            outcome = await job.handle.done
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            outcome = JobOutcome("failed", str(exc))
        if job.status not in {"running", "stopping"}:
            return
        job.status = outcome.status
        job.detail = outcome.detail
        job.output = outcome.output
        job.finished_at = _now_ms()
        job.settled.set()
        snapshot = self._snapshot(job)
        self._notify_changed(job.owner_session)
        if not job.reported and self._on_done is not None:
            try:
                self._on_done(snapshot, job.owner_session)
            except Exception:
                pass

    async def _force_fail(self, job: _JobRecord, detail: str) -> None:
        if job.status not in {"running", "stopping"}:
            return
        job.status = "failed"
        job.detail = detail
        job.finished_at = _now_ms()
        job.reported = True
        job.settled.set()
        self._notify_changed(job.owner_session)

    def _active_count(self, owner_session: str | None) -> int:
        return sum(
            job.owner_session == owner_session and job.status in {"running", "stopping"}
            for job in self._jobs.values()
        )

    def _expect(self, job_id: str) -> _JobRecord:
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job id must be a non-empty string")
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise KeyError(f"unknown job {job_id}") from exc

    @staticmethod
    def _assert_access(job: _JobRecord, caller_session: str | None) -> None:
        if job.owner_session is not None and job.owner_session != caller_session:
            raise PermissionError(f"job {job.id} belongs to another session")

    @staticmethod
    def _snapshot(job: _JobRecord) -> JobSnapshot:
        return JobSnapshot(
            id=job.id,
            kind=job.kind,
            label=job.label,
            status=job.status,
            detail=job.detail,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )

    def _notify_changed(self, owner_session: str | None) -> None:
        if self._on_changed is not None:
            self._on_changed(owner_session)


class _OutputBuffer:
    """Bounded append-only output with a consuming read cursor."""

    def __init__(self, max_bytes: int = 4 * 1024 * 1024) -> None:
        self.max_bytes = max_bytes
        self._data = bytearray()
        self._cursor = 0
        self._dropped = False

    def append(self, data: bytes) -> None:
        self._data.extend(data)
        if len(self._data) > self.max_bytes:
            drop = len(self._data) - self.max_bytes
            del self._data[:drop]
            self._cursor = max(0, self._cursor - drop)
            self._dropped = True

    def read(self) -> str:
        prefix = "[output truncated]\n" if self._dropped and self._cursor == 0 else ""
        data = bytes(self._data[self._cursor :])
        self._cursor = len(self._data)
        self._dropped = False
        return prefix + data.decode("utf-8", errors="replace")


async def start_bash_process(
    command: str,
    *,
    cwd: Path,
) -> JobHandle:
    """Spawn a detached bash process and expose a JobHandle for it."""

    executable = shutil.which("bash") or shutil.which("sh")
    if executable is None:
        raise RuntimeError("a bash-compatible shell is not installed")
    process = await asyncio.create_subprocess_exec(
        executable,
        "-lc",
        command,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=(os.name != "nt"),
    )
    output = _OutputBuffer()
    cancelled = False

    async def drain(stream: asyncio.StreamReader | None) -> None:
        if stream is None:
            return
        while True:
            chunk = await stream.read(8192)
            if not chunk:
                return
            output.append(chunk)

    stdout_task = asyncio.create_task(drain(process.stdout))
    stderr_task = asyncio.create_task(drain(process.stderr))

    async def wait_for_process() -> JobOutcome:
        nonlocal cancelled
        try:
            return_code = await process.wait()
            await asyncio.gather(stdout_task, stderr_task)
        finally:
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        if cancelled:
            return JobOutcome("killed", "cancelled")
        if return_code == 0:
            return JobOutcome("completed")
        if return_code < 0:
            return JobOutcome("failed", f"signal {-return_code}")
        return JobOutcome("failed", f"exit code: {return_code}")

    def cancel(reason: str | None = None) -> None:
        nonlocal cancelled
        if process.returncode is not None:
            return
        cancelled = True
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGTERM)
                return
            except ProcessLookupError:
                return
        process.terminate()

    return JobHandle(cancel=cancel, done=wait_for_process(), read_output=output.read)


def _now_ms() -> int:
    return int(time.time() * 1000)
