"""Persistent POSIX PTY sessions and owner-scoped terminal registry.

The implementation intentionally uses only the Python standard library.  A
terminal is a real interactive bash attached to a pseudo-terminal, rather than
a collection of one-shot subprocesses, so shell state survives between sends.
"""

from __future__ import annotations

import asyncio
import errno
import os
import shutil
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

try:
    import pty as _pty
except ImportError:  # pragma: no cover - exercised on Windows.
    _pty = None

TerminalSignal = Literal["SIGINT", "SIGTERM", "SIGKILL", "SIGTSTP", "SIGHUP"]
TerminalWaitReason = Literal["stdin_read", "inferred_idle", "timeout", "session_exit"]

ALLOWED_SIGNALS: frozenset[str] = frozenset(
    {"SIGINT", "SIGTERM", "SIGKILL", "SIGTSTP", "SIGHUP"}
)
TRUNCATION_MARKER = "\n[output truncated]"


class TerminalError(RuntimeError):
    """A terminal registry or PTY operation failed."""

    def __init__(self, message: str, code: str = "TERMINAL_ERROR") -> None:
        super().__init__(message)
        self.code = code


class PtyUnsupportedError(TerminalError):
    """The host cannot provide POSIX pseudo-terminals."""

    def __init__(self) -> None:
        super().__init__(
            "persistent PTY terminals require a POSIX platform with the pty module; "
            "the current platform does not support PTYs",
            "PTY_UNSUPPORTED",
        )


@dataclass(frozen=True, slots=True)
class TerminalConfig:
    """Bounds and timing for one terminal service."""

    shell: str = "bash"
    shell_args: tuple[str, ...] = ("--noprofile", "--norc", "-i")
    max_send_bytes: int = 256 * 1024
    max_read_bytes: int = 256 * 1024
    scrollback_bytes: int = 4 * 1024 * 1024
    scrollback_lines: int = 10_000
    idle_seconds: float = 0.75
    send_timeout_seconds: float = 30.0
    startup_timeout_seconds: float = 10.0
    close_grace_seconds: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "max_send_bytes",
            "max_read_bytes",
            "scrollback_bytes",
            "scrollback_lines",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "idle_seconds",
            "send_timeout_seconds",
            "startup_timeout_seconds",
            "close_grace_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be a positive number")


def pty_supported() -> bool:
    """Return whether this process can create POSIX pseudo-terminals."""

    return os.name == "posix" and _pty is not None and hasattr(os, "killpg")


def _utf8_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def _retain_head(text: str, max_bytes: int) -> tuple[str, bool]:
    """Retain the UTF-8-safe head of text and report whether it was cut."""

    if _utf8_bytes(text) <= max_bytes:
        return text, False
    if max_bytes <= 0:
        return "", True
    data = bytearray()
    end = 0
    for index, char in enumerate(text):
        encoded = char.encode("utf-8")
        if len(data) + len(encoded) > max_bytes:
            break
        data.extend(encoded)
        end = index + 1
    return text[:end], True


def _retain_tail(text: str, max_bytes: int) -> tuple[str, bool]:
    """Retain the UTF-8-safe tail of text and report whether it was cut."""

    if _utf8_bytes(text) <= max_bytes:
        return text, False
    if max_bytes <= 0:
        return "", True
    data = bytearray()
    start = len(text)
    for index in range(len(text) - 1, -1, -1):
        encoded = text[index].encode("utf-8")
        if len(data) + len(encoded) > max_bytes:
            break
        data[:0] = encoded
        start = index
    return text[start:], True


def bound_terminal_text(text: str, max_bytes: int) -> str:
    """Bound a complete model-facing response with a visible marker."""

    if max_bytes <= 0:
        return ""
    if _utf8_bytes(text) <= max_bytes:
        return text
    marker = TRUNCATION_MARKER
    marker_bytes = _utf8_bytes(marker)
    if marker_bytes >= max_bytes:
        return _retain_tail(marker, max_bytes)[0]
    return _retain_head(text, max_bytes - marker_bytes)[0] + marker


class _BoundedText:
    """A bounded tail buffer that retains UTF-8 text and line metadata."""

    def __init__(self, max_bytes: int, max_lines: int | None = None) -> None:
        self.max_bytes = max_bytes
        self.max_lines = max_lines
        self.text = ""
        self.truncated = False

    def append(self, text: str) -> None:
        if not text:
            return
        self.text += text
        if self.max_lines is not None:
            lines = self.text.split("\n")
            if len(lines) > self.max_lines:
                self.text = "\n".join(lines[-self.max_lines :])
                self.truncated = True
        self.text, cut = _retain_tail(self.text, self.max_bytes)
        self.truncated = self.truncated or cut

    def consume(self) -> tuple[str, bool]:
        value = self.text
        truncated = self.truncated
        self.text = ""
        self.truncated = False
        return value, truncated


class _TerminalSanitizer:
    """Remove common terminal control sequences while preserving prompt text."""

    def __init__(self) -> None:
        self._pending = ""

    def feed(self, data: bytes) -> tuple[str, bool]:
        text = data.decode("utf-8", errors="replace")
        self._pending += text
        output: list[str] = []
        prompt = False
        cursor = 0
        pending = self._pending
        while cursor < len(pending):
            escape = pending.find("\x1b", cursor)
            if escape < 0:
                output.append(pending[cursor:])
                cursor = len(pending)
                break
            output.append(pending[cursor:escape])
            if escape + 1 >= len(pending):
                cursor = escape
                break
            kind = pending[escape + 1]
            if kind == "]":
                bel = pending.find("\x07", escape + 2)
                st = pending.find("\x1b\\", escape + 2)
                ends = [value for value in (bel, st) if value >= 0]
                if not ends:
                    cursor = escape
                    break
                end = min(ends)
                terminator = 1 if pending[end] == "\x07" else 2
                content = pending[escape + 2 : end]
                if content.startswith("133;D;"):
                    prompt = True
                cursor = end + terminator
                continue
            if kind == "[":
                end = escape + 2
                while end < len(pending):
                    value = ord(pending[end])
                    if 0x40 <= value <= 0x7E:
                        break
                    end += 1
                if end >= len(pending):
                    cursor = escape
                    break
                cursor = end + 1
                continue
            cursor = escape + 2
        self._pending = pending[cursor:]
        rendered = "".join(output).replace("\r\n", "\n").replace("\r", "\n").replace("\x07", "")
        return rendered, prompt

    def flush(self) -> str:
        value = "" if self._pending.startswith("\x1b") else self._pending
        self._pending = ""
        return value.replace("\r\n", "\n").replace("\r", "\n").replace("\x07", "")


@dataclass(slots=True)
class _TerminalStatus:
    kind: Literal["running", "exited"] = "running"
    exit_code: int | None = None
    signal_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "running":
            return {"kind": "running"}
        return {"kind": "exited", "exitCode": self.exit_code, "signal": self.signal_name}


class _SendOperation:
    def __init__(self, session: _PtySession, *, max_bytes: int) -> None:
        self.session = session
        self.output = _BoundedText(max_bytes)
        self.started_at = asyncio.get_running_loop().time()
        self.done: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self.start_prompt_count = session.prompt_count
        self.start_output_count = session.output_count
        self.cancel_requested = False
        self.cancel_finished = asyncio.Event()
        self._cancel_task: asyncio.Task[None] | None = None

    def append(self, text: str) -> None:
        if not self.done.done():
            self.output.append(text)

    def read_output(self) -> tuple[str, bool]:
        return self.output.consume()

    def cancel(self) -> bool:
        if self.done.done() or self.cancel_requested:
            return False
        self.cancel_requested = True
        self._cancel_task = asyncio.create_task(
            self._interrupt(), name=f"dsh-terminal-interrupt-{self.session.pid}"
        )
        return True

    async def _interrupt(self) -> None:
        try:
            await self.session._signal_foreground("SIGINT")
        except ProcessLookupError:
            pass
        except Exception as exc:
            self.session._fail_active(self, exc)
        finally:
            self.cancel_finished.set()
            self.session._output_event.set()

    def settle(self, wait_reason: TerminalWaitReason) -> None:
        if self.done.done():
            return
        self.done.set_result(
            {
                "viewport": self.output.text,
                "waitReason": wait_reason,
                "sessionStatus": self.session.status.to_dict(),
                "truncated": self.output.truncated or self.session.scrollback.truncated,
            }
        )

    def fail(self, error: BaseException) -> None:
        if not self.done.done():
            self.done.set_exception(error)


class _PtySession:
    """One bash process attached to one PTY master."""

    def __init__(self, pid: int, master_fd: int, config: TerminalConfig) -> None:
        self.pid = pid
        self.master_fd = master_fd
        self.config = config
        self.loop = asyncio.get_running_loop()
        self.status = _TerminalStatus()
        self.scrollback = _BoundedText(config.scrollback_bytes, config.scrollback_lines)
        self.sanitizer = _TerminalSanitizer()
        self.active: _SendOperation | None = None
        self.prompt_count = 0
        self.output_count = 0
        self.last_output_at = self.loop.time()
        self._output_event = asyncio.Event()
        self._exit_event = asyncio.Event()
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None
        self._reader_registered = True
        self.loop.add_reader(master_fd, self._on_readable)
        self._wait_task = asyncio.create_task(
            self._wait_for_child(), name=f"dsh-terminal-wait-{pid}"
        )

    @classmethod
    async def spawn(
        cls,
        cwd: Path,
        config: TerminalConfig,
        session_id: str,
        owner: str,
    ) -> _PtySession:
        if not pty_supported():
            raise PtyUnsupportedError()
        shell = shutil.which(config.shell)
        if shell is None:
            raise TerminalError(
                f"persistent PTY shell is unavailable: {config.shell}", "SHELL_UNAVAILABLE"
            )
        try:
            if _pty is None:
                raise PtyUnsupportedError()
            pid, master_fd = _pty.fork()
        except (AttributeError, OSError) as exc:
            raise PtyUnsupportedError() from exc
        if pid == 0:  # pragma: no cover - executed by the forked child.
            environment = os.environ.copy()
            environment.update(
                {
                    "TERM": "dumb",
                    "PAGER": "cat",
                    "GIT_PAGER": "cat",
                    "PS1": "dsh> ",
                    "PROMPT_COMMAND": "printf \\\"\\033]133;D;%s\\007\\\" \\\"$?\\\"",
                    "HISTFILE": "/dev/null",
                    "BASH_SILENCE_DEPRECATION_WARNING": "1",
                    "DSH_SHELL": "1",
                    "DSH_SESSION_ID": owner,
                    "DSH_PTY_SESSION_ID": session_id,
                }
            )
            try:
                os.chdir(cwd)
                os.execvpe(shell, [shell, *config.shell_args], environment)
            except BaseException as exc:
                try:
                    os.write(2, f"PTY startup failed: {exc}\n".encode("utf-8", "replace"))
                finally:
                    os._exit(127)
        os.set_blocking(master_fd, False)
        return cls(pid, master_fd, config)

    async def initialize(self) -> str:
        operation = self.start_send(text="", submit=False)
        try:
            result = await asyncio.wait_for(operation.done, self.config.startup_timeout_seconds)
        except TimeoutError as exc:
            raise TerminalError(
                "PTY shell did not reach readiness before startup timeout", "PTY_STARTUP_TIMEOUT"
            ) from exc
        if result["waitReason"] == "session_exit":
            raise TerminalError("PTY shell exited during startup", "PTY_STARTUP_EXIT")
        return str(result["viewport"])

    def start_send(self, *, text: str, submit: bool) -> _SendOperation:
        if self._closing:
            raise TerminalError("PTY session is closing", "SESSION_CLOSING")
        if self.status.kind == "exited":
            raise TerminalError("PTY session has exited", "SESSION_EXITED")
        if self.active is not None and not self.active.done.done():
            raise TerminalError("PTY session already has an active send", "SEND_ACTIVE")
        if not isinstance(text, str):
            raise TypeError("terminal text must be a string")
        if not isinstance(submit, bool):
            raise TypeError("terminal submit must be a boolean")
        if _utf8_bytes(text) > self.config.max_send_bytes:
            raise ValueError(f"terminal input exceeds {self.config.max_send_bytes} UTF-8 bytes")
        operation = _SendOperation(self, max_bytes=self.config.max_read_bytes)
        self.active = operation
        asyncio.create_task(
            self._run_send(operation, text, submit),
            name=f"dsh-terminal-send-{self.pid}",
        )
        return operation

    async def _run_send(self, operation: _SendOperation, text: str, submit: bool) -> None:
        try:
            if text:
                await self._write(text + ("\r" if submit else ""))
            await self._wait_for_send(operation)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail_active(operation, exc)
        finally:
            if self.active is operation and operation.done.done():
                self.active = None

    async def _wait_for_send(self, operation: _SendOperation) -> None:
        deadline = self.loop.time() + self.config.send_timeout_seconds
        while not operation.done.done():
            if self.status.kind == "exited":
                operation.settle("session_exit")
                break
            now = self.loop.time()
            if (
                self.prompt_count > operation.start_prompt_count
                and self._foreground_pgid() == self.pid
            ):
                operation.settle("stdin_read")
                break
            if (
                self.output_count > operation.start_output_count
                and now - self.last_output_at >= self.config.idle_seconds
            ):
                operation.settle("inferred_idle")
                break
            if now >= deadline:
                operation.settle("timeout")
                break
            timeout = min(
                0.1,
                max(0.01, deadline - now),
                max(0.01, self.config.idle_seconds - (now - self.last_output_at)),
            )
            self._output_event.clear()
            self._exit_event.clear()
            try:
                await asyncio.wait_for(self._output_event.wait(), timeout=timeout)
            except TimeoutError:
                continue

    async def _write(self, text: str) -> None:
        data = text.encode("utf-8")
        offset = 0
        while offset < len(data):
            if self.status.kind == "exited" or self._closing:
                raise TerminalError("PTY session exited while writing", "SESSION_EXITED")
            try:
                written = os.write(self.master_fd, data[offset:])
            except BlockingIOError:
                await asyncio.sleep(0.01)
                continue
            except OSError as exc:
                raise TerminalError(f"PTY write failed: {exc}", "PTY_WRITE_FAILED") from exc
            if written <= 0:
                raise TerminalError("PTY write returned no progress", "PTY_WRITE_FAILED")
            offset += written

    def read(self, *, offset: int = 0, count: int = 500) -> dict[str, Any]:
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("PTY read offset must be a non-negative integer")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("PTY read count must be a positive integer")
        snapshot = self.scrollback.text
        lines = snapshot.split("\n") if snapshot else []
        total_lines = len(lines)
        if offset >= total_lines:
            return {
                "text": "",
                "totalLines": total_lines,
                "lineBegin": offset,
                "lineEnd": offset,
                "truncated": self.scrollback.truncated,
            }
        end = total_lines - offset
        start = max(0, end - count)
        selected = "\n".join(lines[start:end])
        bounded, cut = _retain_tail(selected, self.config.max_read_bytes)
        returned_lines = len(bounded.split("\n")) if bounded else 0
        return {
            "text": bounded,
            "totalLines": total_lines,
            "lineBegin": offset,
            "lineEnd": offset + returned_lines,
            "truncated": self.scrollback.truncated or cut,
        }

    async def signal(self, signal_name: str) -> dict[str, Any]:
        if signal_name not in ALLOWED_SIGNALS:
            raise ValueError(
                "signal must be one of SIGINT, SIGTERM, SIGKILL, SIGTSTP, or SIGHUP"
            )
        if self._closing:
            raise TerminalError("PTY session is closing", "SESSION_CLOSING")
        target = await self._signal_foreground(cast(TerminalSignal, signal_name))
        return {"delivered": True, "targetPgid": target}

    async def _signal_foreground(self, signal_name: TerminalSignal) -> int:
        target = self._foreground_pgid()
        if target is None:
            raise TerminalError("cannot resolve foreground process group", "NO_FOREGROUND_GROUP")
        if signal_name == "SIGKILL" and target == self.pid:
            raise TerminalError(
                "refusing to SIGKILL the terminal shell; use terminal_close instead",
                "SHELL_SIGKILL_REFUSED",
            )
        try:
            os.killpg(target, getattr(signal, signal_name))
        except ProcessLookupError as exc:
            raise TerminalError(
                "foreground process group no longer exists", "NO_FOREGROUND_GROUP"
            ) from exc
        except OSError as exc:
            raise TerminalError(f"could not deliver {signal_name}: {exc}", "SIGNAL_FAILED") from exc
        return target

    def _foreground_pgid(self) -> int | None:
        try:
            value = os.tcgetpgrp(self.master_fd)
        except OSError:
            value = self.pid
        if value <= 0:
            return None
        return value

    async def close(self, reason: str = "model request") -> None:
        self._closing = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close_once(reason), name=f"dsh-terminal-close-{self.pid}"
            )
        await self._close_task

    async def _close_once(self, reason: str) -> None:
        del reason  # Kept in the signature for diagnostic parity with the TS backend.
        groups = self._signalable_groups()
        self._send_groups(groups, signal.SIGTERM)
        deadline = self.loop.time() + self.config.close_grace_seconds
        while self.status.kind != "exited" and self.loop.time() < deadline:
            await asyncio.sleep(0.05)
        if self.status.kind != "exited":
            self._send_groups(self._signalable_groups() | groups, signal.SIGKILL)
            deadline = self.loop.time() + self.config.close_grace_seconds
            while self.status.kind != "exited" and self.loop.time() < deadline:
                await asyncio.sleep(0.05)
        if self.status.kind != "exited":
            try:
                os.kill(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await self._wait_task
        if self.active is not None and not self.active.done.done():
            self.active.settle("session_exit")
        if self._reader_registered:
            self.loop.remove_reader(self.master_fd)
            self._reader_registered = False
        try:
            os.close(self.master_fd)
        except OSError:
            pass

    def _signalable_groups(self) -> set[int]:
        groups = {self.pid}
        foreground = self._foreground_pgid()
        if foreground is not None:
            groups.add(foreground)
        for child_pid in _descendant_pids(self.pid):
            try:
                groups.add(os.getpgid(child_pid))
            except ProcessLookupError:
                pass
        own_group = os.getpgrp()
        return {group for group in groups if group > 0 and group != own_group}

    @staticmethod
    def _send_groups(groups: set[int], sig: signal.Signals) -> None:
        for group in groups:
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                continue
            except PermissionError:
                continue

    def _on_readable(self) -> None:
        try:
            data = os.read(self.master_fd, 8192)
        except BlockingIOError:
            return
        except OSError as exc:
            if exc.errno in {errno.EIO, errno.EBADF}:
                return
            self._fail_active(None, TerminalError(f"PTY read failed: {exc}", "PTY_READ_FAILED"))
            return
        if data:
            self._append_data(data)

    def _append_data(self, data: bytes) -> None:
        text, prompt = self.sanitizer.feed(data)
        if not text:
            if prompt:
                self.prompt_count += 1
                self._output_event.set()
            return
        self.output_count += 1
        self.last_output_at = self.loop.time()
        self.scrollback.append(text)
        if self.active is not None:
            self.active.append(text)
        if prompt:
            self.prompt_count += 1
        self._output_event.set()

    async def _wait_for_child(self) -> None:
        while self.status.kind != "exited":
            try:
                waited_pid, wait_status = os.waitpid(self.pid, os.WNOHANG)
            except ChildProcessError:
                waited_pid, wait_status = self.pid, 0
            if waited_pid == self.pid:
                if os.WIFEXITED(wait_status):
                    self.status = _TerminalStatus("exited", os.WEXITSTATUS(wait_status), None)
                elif os.WIFSIGNALED(wait_status):
                    number = os.WTERMSIG(wait_status)
                    try:
                        name = signal.Signals(number).name
                    except ValueError:
                        name = None
                    self.status = _TerminalStatus("exited", None, name)
                else:
                    self.status = _TerminalStatus("exited", None, None)
                if self._reader_registered:
                    self.loop.remove_reader(self.master_fd)
                    self._reader_registered = False
                if self.active is not None:
                    self.active.settle("session_exit")
                self._exit_event.set()
                self._output_event.set()
                return
            await asyncio.sleep(0.05)

    def _fail_active(self, operation: _SendOperation | None, error: BaseException) -> None:
        target = operation or self.active
        if target is not None:
            target.fail(error)
        if operation is None and self.status.kind != "exited":
            self.status = _TerminalStatus("exited", None, None)
            self._exit_event.set()


@dataclass(slots=True)
class _TerminalRecord:
    session_id: str
    owner: str
    name: str | None
    type: str
    session: _PtySession
    closing: asyncio.Task[None] | None = None


class TerminalSessionService:
    """Owner/session-isolated registry for persistent PTY sessions."""

    def __init__(self, config: TerminalConfig | None = None) -> None:
        self.config = config or TerminalConfig()
        self._sessions: dict[str, _TerminalRecord] = {}
        self._reserved_names: dict[str, set[str]] = {}
        self._next_id = 0
        self._lock = asyncio.Lock()
        self._disposing = False
        self._closing_owners: set[str] = set()
        self._pending_spawns: dict[str, set[asyncio.Future[None]]] = {}

    async def spawn(
        self,
        owner: str,
        *,
        type: str,
        name: str | None = None,
        cwd: str | os.PathLike[str] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(owner, str) or not owner:
            raise TerminalError("terminal owner must be a non-empty session id", "OWNER_NOT_LIVE")
        if self._disposing:
            raise TerminalError("PTY service is disposing", "SERVICE_DISPOSING")
        if not pty_supported():
            raise PtyUnsupportedError()
        if not isinstance(type, str) or not type:
            raise ValueError("type must be a non-empty string")
        if type != "shell":
            raise TerminalError(f'no PTY backend registered for "{type}"', "NO_BACKEND")
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError("terminal name must be a non-empty string")
        if cwd is None:
            path = Path.cwd().resolve()
        else:
            path = Path(cwd).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"terminal cwd is not a directory: {path}")
        async with self._lock:
            if self._disposing:
                raise TerminalError("PTY service is disposing", "SERVICE_DISPOSING")
            if owner in self._closing_owners:
                raise TerminalError("PTY owner is being disposed", "OWNER_NOT_LIVE")
            if name is not None:
                duplicate = any(
                    record.owner == owner and record.name == name
                    for record in self._sessions.values()
                )
                if duplicate:
                    raise TerminalError(
                        f'terminal session name "{name}" already exists for this owner',
                        "DUPLICATE_NAME",
                    )
                reserved = self._reserved_names.setdefault(owner, set())
                if name in reserved:
                    raise TerminalError(
                        f'terminal session name "{name}" is already being created',
                        "DUPLICATE_NAME",
                    )
                reserved.add(name)
            self._next_id += 1
            session_id = f"pty-{self._next_id}"
            pending = asyncio.get_running_loop().create_future()
            self._pending_spawns.setdefault(owner, set()).add(pending)
        session: _PtySession | None = None
        try:
            session = await _PtySession.spawn(path, self.config, session_id, owner)
            motd = await session.initialize()
            async with self._lock:
                if self._disposing:
                    raise TerminalError("PTY service is disposing", "SERVICE_DISPOSING")
                record = _TerminalRecord(session_id, owner, name, type, session)
                self._sessions[session_id] = record
            return self._snapshot(record, motd=motd)
        except Exception:
            if session is not None:
                try:
                    await session.close("PTY spawn rolled back")
                except Exception:
                    pass
            raise
        finally:
            async with self._lock:
                owned_pending = self._pending_spawns.get(owner)
                if owned_pending is not None:
                    owned_pending.discard(pending)
                    if not owned_pending:
                        self._pending_spawns.pop(owner, None)
                if not pending.done():
                    pending.set_result(None)
                if name is not None:
                    reserved = self._reserved_names.get(owner)
                    if reserved is not None:
                        reserved.discard(name)
                        if not reserved:
                            self._reserved_names.pop(owner, None)

    def start_send(self, owner: str, session_id: str, *, text: str, submit: bool) -> _SendOperation:
        return self._expect_owned(owner, session_id).session.start_send(text=text, submit=submit)

    def read(
        self,
        owner: str,
        session_id: str,
        *,
        offset: int = 0,
        count: int = 500,
    ) -> dict[str, Any]:
        return self._expect_owned(owner, session_id).session.read(offset=offset, count=count)

    async def signal(self, owner: str, session_id: str, signal_name: str) -> dict[str, Any]:
        return await self._expect_owned(owner, session_id).session.signal(signal_name)

    async def kill(self, owner: str, session_id: str, reason: str = "model request") -> bool:
        record = self._expect_owned(owner, session_id)
        if record.closing is not None:
            await record.closing
            return False
        record.closing = asyncio.create_task(
            record.session.close(reason), name=f"dsh-terminal-kill-{session_id}"
        )
        try:
            await record.closing
        except Exception:
            record.closing = None
            raise
        self._sessions.pop(session_id, None)
        return True

    def list(self, owner: str) -> list[dict[str, Any]]:
        return [
            self._snapshot(record)
            for record in self._sessions.values()
            if record.owner == owner
        ]

    def has_owner_activity(self, owner: str) -> bool:
        return (
            any(record.owner == owner for record in self._sessions.values())
            or owner in self._reserved_names
            or owner in self._pending_spawns
        )

    async def close_owner(self, owner: str, reason: str = "PTY owner disposed") -> None:
        async with self._lock:
            self._closing_owners.add(owner)
            pending = tuple(self._pending_spawns.get(owner, ()))
            records = [record for record in self._sessions.values() if record.owner == owner]
        try:
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            await self._close_records(records, reason)
        finally:
            async with self._lock:
                self._closing_owners.discard(owner)
                self._reserved_names.pop(owner, None)

    async def close_all(self, reason: str = "PTY service disposed") -> None:
        async with self._lock:
            if self._disposing:
                return
            self._disposing = True
            records = list(self._sessions.values())
            pending = tuple(
                future
                for futures in self._pending_spawns.values()
                for future in futures
            )
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        errors = await self._close_records(records, reason, clear=True)
        async with self._lock:
            self._reserved_names.clear()
            self._closing_owners.clear()
            self._pending_spawns.clear()
        if errors:
            raise RuntimeError(f"failed to clean up {len(errors)} PTY session(s): {errors[0]}")

    def _expect_owned(self, owner: str, session_id: str) -> _TerminalRecord:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("sessionId must be a non-empty string")
        record = self._sessions.get(session_id)
        if record is None:
            raise TerminalError(f"unknown PTY session {session_id}", "NO_SESSION")
        if record.owner != owner:
            raise TerminalError(
                f"PTY session {session_id} belongs to another session", "FOREIGN_SESSION"
            )
        return record

    @staticmethod
    def _snapshot(record: _TerminalRecord, *, motd: str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sessionId": record.session_id,
            "type": record.type,
            "status": record.session.status.to_dict(),
        }
        if record.name is not None:
            result["name"] = record.name
        if record.session.pid is not None:
            result["pid"] = record.session.pid
        if motd is not None:
            result["motd"] = motd
        return result

    async def _close_records(
        self,
        records: list[_TerminalRecord],
        reason: str,
        *,
        clear: bool = False,
    ) -> list[BaseException]:
        async def close_one(record: _TerminalRecord) -> None:
            if record.closing is None:
                record.closing = asyncio.create_task(
                    record.session.close(reason), name=f"dsh-terminal-close-{record.session_id}"
                )
            await record.closing
            self._sessions.pop(record.session_id, None)

        results = await asyncio.gather(
            *(close_one(record) for record in records), return_exceptions=True
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if clear:
            self._sessions.clear()
        return errors


def _descendant_pids(root_pid: int) -> set[int]:
    """Best-effort Linux process-tree inspection used only during PTY teardown."""

    proc = Path("/proc")
    if not proc.is_dir():
        return set()
    parents: dict[int, int] = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="ascii")
            end = stat.rfind(")")
            fields = stat[end + 2 :].split()
            parents[int(entry.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    descendants: set[int] = set()
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if (parent == root_pid or parent in descendants) and pid not in descendants:
                descendants.add(pid)
                changed = True
    return descendants


__all__ = [
    "ALLOWED_SIGNALS",
    "PtyUnsupportedError",
    "TerminalConfig",
    "TerminalError",
    "TerminalSessionService",
    "TerminalSignal",
    "TerminalWaitReason",
    "bound_terminal_text",
    "pty_supported",
]
