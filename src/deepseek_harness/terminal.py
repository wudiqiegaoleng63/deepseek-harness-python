"""Persistent POSIX PTY sessions and owner-scoped terminal registry.

The implementation intentionally uses only the Python standard library.  A
terminal is a real interactive bash attached to a pseudo-terminal, rather than
a collection of one-shot subprocesses, so shell state survives between sends.
"""

from __future__ import annotations

import asyncio
import codecs
import os
import platform
import shutil
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

try:
    import fcntl as _fcntl
    import pty as _pty
    import struct as _struct
    import termios as _termios
except ImportError:  # pragma: no cover - exercised on Windows.
    _fcntl = None
    _pty = None
    _struct = None
    _termios = None

TerminalSignal = Literal["SIGINT", "SIGTERM", "SIGKILL", "SIGTSTP", "SIGHUP"]
TerminalWaitReason = Literal["stdin_read", "inferred_idle", "timeout", "session_exit"]

ALLOWED_SIGNALS: frozenset[str] = frozenset(
    {"SIGINT", "SIGTERM", "SIGKILL", "SIGTSTP", "SIGHUP"}
)
TRUNCATION_MARKER = "\n[output truncated]"
_MAX_PENDING_BYTES = 65_536


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
    rows: int = 40
    cols: int = 160
    max_send_bytes: int = 256 * 1024
    max_read_bytes: int = 256 * 1024
    scrollback_bytes: int = 4 * 1024 * 1024
    scrollback_lines: int = 10_000
    poll_seconds: float = 0.05
    probe_after_seconds: float = 0.15
    idle_seconds: float = 3.0
    handoff_grace_seconds: float = 0.5
    send_timeout_seconds: float = 30.0
    startup_timeout_seconds: float = 10.0
    close_grace_seconds: float = 1.0
    cancel_grace_seconds: float = 2.0
    max_sessions_per_owner: int = 8

    def __post_init__(self) -> None:
        for name in (
            "rows",
            "cols",
            "max_send_bytes",
            "max_read_bytes",
            "scrollback_bytes",
            "scrollback_lines",
            "max_sessions_per_owner",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "poll_seconds",
            "probe_after_seconds",
            "idle_seconds",
            "handoff_grace_seconds",
            "send_timeout_seconds",
            "startup_timeout_seconds",
            "close_grace_seconds",
            "cancel_grace_seconds",
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
    """Remove common terminal control sequences while preserving prompt text.

    Decoding is incremental, so a UTF-8 character split across PTY reads is
    joined instead of being replaced.  An unterminated escape sequence is
    retained only up to a hard byte bound; past the bound the sequence is
    discarded (and its terminator skipped later) so a hostile stream cannot
    grow the parser state without limit.
    """

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._pending = ""
        self._discard: Literal["osc", "csi"] | None = None

    def feed(self, data: bytes) -> tuple[str, bool]:
        text = self._decoder.decode(data)
        if self._discard is not None:
            text = self._strip_discarded(text)
        self._pending += text
        self._enforce_pending_bound()
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
        try:
            value = self._decoder.decode(b"", True)
        except ValueError:
            value = ""
        if not self._pending.startswith("\x1b"):
            value = self._pending + value
        self._pending = ""
        self._discard = None
        return value.replace("\r\n", "\n").replace("\r", "\n").replace("\x07", "")

    def _enforce_pending_bound(self) -> None:
        if len(self._pending) <= _MAX_PENDING_BYTES:
            return
        # Drop the incomplete escape sequence; remember its family so the
        # terminator arriving in a later chunk is skipped instead of leaking.
        self._discard = "osc" if self._pending[1:2] == "]" else "csi"
        self._pending = ""

    def _strip_discarded(self, chunk: str) -> str:
        if self._discard == "csi":
            for index, char in enumerate(chunk):
                if 0x40 <= ord(char) <= 0x7E:
                    self._discard = None
                    return chunk[index + 1 :]
            return ""
        index = 0
        while index < len(chunk):
            char = chunk[index]
            if char == "\x07":
                self._discard = None
                return chunk[index + 1 :]
            if char == "\x1b" and chunk[index + 1 : index + 2] == "\\":
                self._discard = None
                return chunk[index + 2 :]
            index += 1
        return ""


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

    async def join(self) -> dict[str, Any]:
        """Await settlement without letting caller cancellation cancel the send.

        A cancelled waiter still owns the foreground command, so cancellation
        requests the usual foreground SIGINT before propagating.
        """

        try:
            return await asyncio.shield(self.done)
        except asyncio.CancelledError:
            self.cancel()
            raise

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
        # A command that traps or ignores SIGINT would otherwise hold the send
        # slot until the full timeout; escalate after a bounded grace period.
        await asyncio.sleep(self.session.config.cancel_grace_seconds)
        if not self.done.done():
            try:
                await self.session._signal_foreground("SIGKILL")
            except Exception:
                pass
            finally:
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
        self._master_closed = False
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
                    # Single quotes survive into the format string so printf
                    # itself expands \033/\007 into the OSC prompt marker.
                    "PROMPT_COMMAND": "printf '\\033]133;D;%s\\007' \"$?\"",
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
        cls._apply_window_size(master_fd, config)
        try:
            return cls(pid, master_fd, config)
        except BaseException as exc:
            # Post-fork setup failed; never leak the forked child or its PTY.
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                os.waitpid(pid, 0)
            except OSError:
                pass
            try:
                os.close(master_fd)
            except OSError:
                pass
            raise exc

    @staticmethod
    def _apply_window_size(master_fd: int, config: TerminalConfig) -> None:
        if _fcntl is None or _termios is None or _struct is None:
            return
        try:
            winsize = _struct.pack("HHHH", config.rows, config.cols, 0, 0)
            _fcntl.ioctl(master_fd, _termios.TIOCSWINSZ, winsize)
        except OSError:
            pass

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
            if text and not operation.cancel_requested:
                await self._write(text + ("\r" if submit else ""))
            await self._wait_for_send(operation)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail_active(operation, exc)
        finally:
            if self.active is operation and operation.done.done():
                if operation.done.cancelled():
                    # Defensive: an externally cancelled future must not leave
                    # the foreground command running with the slot silently
                    # released.  The shielded join() above makes this rare.
                    asyncio.create_task(
                        self._signal_foreground("SIGINT"), name=f"dsh-terminal-orphan-{self.pid}"
                    )
                self.active = None

    async def _wait_for_send(self, operation: _SendOperation) -> None:
        deadline = operation.started_at + self.config.send_timeout_seconds
        while not operation.done.done():
            if self.status.kind == "exited":
                operation.settle("session_exit")
                break
            now = self.loop.time()
            prompt_seen = self.prompt_count > operation.start_prompt_count
            foreground = self._foreground_pgid()
            if prompt_seen and foreground == self.pid:
                operation.settle("stdin_read")
                break
            # Exact probe: a foreground process genuinely blocked in a tty
            # read is waiting for input, marker or not.  Mirrors the TS
            # exactProbeAfterMs behaviour.
            if (
                now - operation.started_at >= self.config.probe_after_seconds
                and foreground is not None
                and _stdin_waiting(foreground)
            ):
                operation.settle("stdin_read")
                break
            idle_bound = self.config.idle_seconds
            if prompt_seen:
                # A prompt marker can race the kernel's foreground handoff;
                # hold the idle fallback for the grace window (TS handoffGraceMs).
                idle_bound += self.config.handoff_grace_seconds
            if (
                self.output_count > operation.start_output_count
                and now - self.last_output_at >= idle_bound
            ):
                operation.settle("inferred_idle")
                break
            if now >= deadline:
                operation.settle("timeout")
                break
            timeout = min(
                self.config.poll_seconds,
                max(0.005, deadline - now),
                max(0.005, idle_bound - (now - self.last_output_at)),
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
        if self.status.kind == "exited":
            raise TerminalError("PTY session has exited", "SESSION_EXITED")
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
        # After the child is reaped the pid may be reused by an unrelated
        # process; never fall back to it as a foreground group.
        if self.status.kind == "exited" or self._master_closed:
            return None
        try:
            value = os.tcgetpgrp(self.master_fd)
        except OSError:
            return None
        return value if value > 0 else None

    async def close(self, reason: str = "model request") -> None:
        self._closing = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close_once(reason), name=f"dsh-terminal-close-{self.pid}"
            )
        # A cancelled caller must not cancel the shared teardown task.
        await asyncio.shield(self._close_task)

    async def _close_once(self, reason: str) -> None:
        del reason  # Kept in the signature for diagnostic parity with the TS backend.
        if self.status.kind != "exited":
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
        if self.status.kind != "exited":
            await self._wait_task
        if self.active is not None and not self.active.done.done():
            self.active.settle("session_exit")
        self._close_master()

    def _close_master(self) -> None:
        if self._reader_registered:
            self.loop.remove_reader(self.master_fd)
            self._reader_registered = False
        if not self._master_closed:
            self._master_closed = True
            try:
                os.close(self.master_fd)
            except OSError:
                pass

    def _signalable_groups(self) -> set[int]:
        groups: set[int] = set()
        # The shell's own group is only a valid target while it is still the
        # unreaped child; afterwards the pid may belong to another process.
        if self.status.kind != "exited":
            groups.add(self.pid)
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
        except OSError:
            # EIO/EBADF mean the slave side is gone; stop the reader so the
            # loop cannot spin.  The wait task reaps the child and closes fd.
            self._close_master()
            return
        if data:
            self._append_data(data)

    def _append_data(self, data: bytes) -> None:
        text, prompt = self.sanitizer.feed(data)
        self._append_text(text, prompt)

    def _append_text(self, text: str, prompt: bool = False) -> None:
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
                # Drain whatever the PTY still buffers, then release the fd so
                # a naturally exited session leaks neither descriptor nor
                # reader callback.
                while True:
                    try:
                        data = os.read(self.master_fd, 8192)
                    except (BlockingIOError, OSError):
                        break
                    if not data:
                        break
                    self._append_data(data)
                tail = self.sanitizer.flush()
                self._append_text(tail)
                self._close_master()
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
            active = sum(
                1 for record in self._sessions.values() if record.owner == owner
            ) + len(self._pending_spawns.get(owner, ()))
            if active >= self.config.max_sessions_per_owner:
                raise TerminalError(
                    f"terminal session limit reached for this owner "
                    f"({self.config.max_sessions_per_owner})",
                    "OWNER_LIMIT",
                )
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
                if owner in self._closing_owners:
                    # The owner began disposing while this spawn was in flight;
                    # publish nothing and roll the PTY back below.
                    raise TerminalError("PTY owner is being disposed", "OWNER_NOT_LIVE")
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
            await asyncio.shield(record.closing)
            return False
        record.closing = asyncio.create_task(
            record.session.close(reason), name=f"dsh-terminal-kill-{session_id}"
        )
        try:
            await asyncio.shield(record.closing)
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
            await asyncio.shield(record.closing)
            self._sessions.pop(record.session_id, None)

        results = await asyncio.gather(
            *(close_one(record) for record in records), return_exceptions=True
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if clear:
            self._sessions.clear()
        return errors


_SYSCALL_TABLES: dict[str, dict[str, int]] = {
    "x86_64": {
        "read": 0,
        "select": 23,
        "pselect": 270,
        "poll": 7,
        "ppoll": 271,
        "epoll_wait": 232,
        "epoll_pwait": 281,
    },
    "aarch64": {"read": 63, "pselect": 72, "ppoll": 73, "epoll_pwait": 22},
}


def _read_process_memory(pid: int, address: int, length: int) -> bytes | None:
    try:
        fd = os.open(f"/proc/{pid}/mem", os.O_RDONLY)
    except OSError:
        return None
    try:
        return os.pread(fd, length, address)
    except OSError:
        return None
    finally:
        os.close(fd)


def _fd_set_has_stdin(pid: int, address: int) -> bool:
    if address == 0:
        return False
    memory = _read_process_memory(pid, address, 8)
    return memory is not None and len(memory) >= 1 and memory[0] % 2 == 1


def _poll_has_stdin(pid: int, address: int, count: int) -> bool:
    if address == 0 or count <= 0:
        return False
    memory = _read_process_memory(pid, address, min(count, 1024) * 8)
    if memory is None:
        return False
    for offset in range(0, len(memory) - 7, 8):
        fd = int.from_bytes(memory[offset : offset + 4], "little", signed=True)
        events = int.from_bytes(memory[offset + 4 : offset + 6], "little")
        if fd == 0 and events & 0x001:
            return True
    return False


def _epoll_has_stdin(pid: int, epfd: int) -> bool:
    try:
        info = Path(f"/proc/{pid}/fdinfo/{epfd}").read_text(encoding="ascii", errors="replace")
    except OSError:
        return False
    for line in info.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0] == "tfd:" and parts[1] == "0":
            return True
    return False


def _syscall_waits_on_stdin(pid: int, fields: list[str], table: dict[str, int]) -> bool:
    if fields[0] == "running" or fields[0] == "-1":
        return False
    try:
        number = int(fields[0])
        args = [int(value, 16) for value in fields[1:7]]
    except (ValueError, IndexError):
        return False
    if number == table["read"]:
        return args[0] == 0
    if number in {table.get("select"), table.get("pselect")}:
        return args[0] >= 1 and _fd_set_has_stdin(pid, args[1])
    if number in {table.get("poll"), table.get("ppoll")}:
        return args[1] >= 1 and _poll_has_stdin(pid, args[0], args[1])
    if number in {table.get("epoll_wait"), table.get("epoll_pwait")}:
        return args[2] >= 1 and _epoll_has_stdin(pid, args[0])
    return False


def _stdin_waiting(pgid: int) -> bool:
    """Report whether any process in ``pgid`` waits on stdin.

    A faithful port of the TS process inspector: the per-architecture syscall
    table classifies the blocked syscall, and select/poll sets plus epoll
    interest are verified through ``/proc/<pid>/mem`` and ``fdinfo``.
    """

    table = _SYSCALL_TABLES.get(platform.machine())
    if table is None:
        return False
    proc = Path("/proc")
    try:
        entries = list(proc.iterdir())
    except OSError:
        return False
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="ascii", errors="replace")
            end = stat.rfind(")")
            fields = stat[end + 2 :].split()
            if int(fields[2]) != pgid:
                continue
            pid = int(entry.name)
            tasks = entry / "task"
            task_ids = [item for item in tasks.iterdir() if item.name.isdigit()]
            if not task_ids:
                task_ids = [entry]
            for task in task_ids:
                try:
                    syscall = (task / "syscall").read_text(encoding="ascii").split()
                except (OSError, ValueError):
                    continue
                if _syscall_waits_on_stdin(pid, syscall, table):
                    return True
        except (OSError, ValueError, IndexError):
            continue
    return False


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
