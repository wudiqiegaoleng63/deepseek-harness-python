"""Dynamic Cordis packages for the native Python host.

The browser UI speaks the same ``dynamicCordisRunner`` remote namespace as the
TypeScript host.  Python cannot mount a TypeScript Cordis Fiber directly, so
the host half is evaluated in a short-lived Node ``vm`` bridge.  The bridge
only exposes the registration façade (handlers, tools and basic lifecycle
context); Node globals such as ``require``, ``process`` and network APIs are
not placed in the dynamic package realm.

The registry, immutable package versions, approval round trip and stale-run
checks remain Python-owned.  This makes host-only packages useful immediately
and lets the existing frontend load client halves through the exact same
``getClientCode``/``invoke`` protocol.
"""

# The embedded Node bridge is intentionally kept as one readable JavaScript
# program; its long source lines are not Python formatting concerns.
# ruff: noqa: E501

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from .tools.registry import ToolDefinition, ToolRegistry, ToolResult

JsonObject = dict[str, Any]
JsonValue = Any
DynamicRunMode = Literal["run", "update"]


class DynamicCordisError(RuntimeError):
    """A dynamic package definition or activation failed."""


@dataclass(slots=True)
class _Package:
    package_id: str
    name: str
    purpose: str
    host_code: str | None
    client_code: str | None

    def view(self) -> JsonObject:
        return {
            "packageId": self.package_id,
            "name": self.name,
            "purpose": self.purpose,
            "hasHostHalf": self.host_code is not None,
            "hasClientHalf": self.client_code is not None,
        }


@dataclass(slots=True)
class _Attempt:
    run_id: str
    package_id: str
    mode: DynamicRunMode
    status: str
    approval_id: str | None = None
    requires_approval: bool | None = None
    host_status: str = "absent"
    host_waiting: list[str] = field(default_factory=list)
    host_error: str | None = None
    client_status: str = "absent"
    client_waiting: list[str] = field(default_factory=list)
    client_error: str | None = None
    error: JsonObject | None = None

    def to_dict(self) -> JsonObject:
        value: JsonObject = {
            "pluginRunId": self.run_id,
            "packageId": self.package_id,
            "mode": self.mode,
            "status": self.status,
            "host": {
                "status": self.host_status,
                "waitingFor": list(self.host_waiting),
            },
            "client": {
                "status": self.client_status,
                "waitingFor": list(self.client_waiting),
            },
        }
        if self.approval_id is not None:
            value["approvalRequestId"] = self.approval_id
        if self.requires_approval is not None:
            value["requiresApproval"] = self.requires_approval
        if self.host_error is not None:
            value["host"]["error"] = self.host_error
        if self.client_error is not None:
            value["client"]["error"] = self.client_error
        if self.error is not None:
            value["error"] = self.error
        return value


@dataclass(slots=True)
class _Run:
    run_id: str
    package_id: str
    tool_disposers: list[Callable[[], None]] = field(default_factory=list)
    handlers: set[str] = field(default_factory=set)


@dataclass(slots=True)
class _Plugin:
    plugin_id: str
    session_id: str
    packages: dict[str, _Package] = field(default_factory=dict)
    approved_packages: set[str] = field(default_factory=set)
    approve_future_versions: bool = False
    current_package_id: str | None = None
    next_package_id: str | None = None
    run: _Run | None = None
    latest: _Attempt | None = None


@dataclass(frozen=True, slots=True)
class _PendingRun:
    request_id: str
    session_id: str
    plugin_id: str
    package_id: str
    run_id: str
    mode: DynamicRunMode
    requires_approval: bool


_PREFIX = re.compile(r"^[a-z]{3,6}$")


class DynamicCordisService:
    """Process-local dynamic package registry and remote lifecycle service."""

    def __init__(
        self,
        *,
        tool_registries: dict[str, ToolRegistry] | None = None,
        remote_event: Callable[[str, list[JsonValue]], None] | None = None,
    ) -> None:
        self._plugins: dict[str, _Plugin] = {}
        self._pending: dict[str, _PendingRun] = {}
        self._tool_registries = tool_registries if tool_registries is not None else {}
        self._remote_event = remote_event
        self._client_manifest: list[JsonObject] = []
        self._plugin_number = 0
        self._package_number = 0
        self._run_number = 0
        self._approval_number = 0

    def define(
        self,
        session_id: str,
        plugin: JsonObject,
        name: Any,
        purpose: Any,
        code: JsonObject,
    ) -> JsonObject:
        clean_name = self._required_text(name, "name")
        clean_purpose = self._required_text(purpose, "purpose")
        host = self._optional_code(code, "host")
        client = self._optional_code(code, "client")
        if host is None and client is None:
            raise DynamicCordisError("cordis_define needs code.host, code.client, or both")
        if host is not None:
            _node_call_sync("check", host, ("host",))
        if client is not None:
            _node_call_sync("check", client, ("client",))

        kind = plugin.get("kind")
        if kind == "new":
            prefix = plugin.get("idPrefix")
            if not isinstance(prefix, str) or not _PREFIX.fullmatch(prefix.strip()):
                raise DynamicCordisError(
                    "cordis_define plugin.idPrefix must contain 3–6 lowercase English letters"
                )
            self._plugin_number += 1
            plugin_id = f"{prefix.strip()}-{self._plugin_number}"
            record = _Plugin(plugin_id, session_id)
            self._plugins[plugin_id] = record
        elif kind == "existing":
            plugin_id = plugin.get("pluginId")
            if not isinstance(plugin_id, str):
                raise DynamicCordisError("cordis_define existing plugin needs pluginId")
            record = self._owned(session_id, plugin_id)
        else:
            raise DynamicCordisError('cordis_define plugin.kind must be "new" or "existing"')

        self._package_number += 1
        package_id = f"pkg-{self._package_number}"
        record.packages[package_id] = _Package(package_id, clean_name, clean_purpose, host, client)
        return {
            "pluginId": record.plugin_id,
            "packageId": package_id,
            "name": clean_name,
            "purpose": clean_purpose,
            "hasHostHalf": host is not None,
            "hasClientHalf": client is not None,
        }

    async def run(
        self,
        session_id: str,
        plugin_id: str,
        package_id: str,
        mode: str,
    ) -> JsonObject:
        record = self._owned(session_id, plugin_id)
        package = record.packages.get(package_id)
        if package is None:
            return self._failure(
                "package-missing", f'plugin "{plugin_id}" has no package "{package_id}"'
            )
        if mode not in {"run", "update"}:
            return self._failure("invalid-mode", 'mode must be "run" or "update"')
        run_mode = cast(DynamicRunMode, mode)
        if run_mode == "update" and (
            record.current_package_id is None or record.current_package_id == package_id
        ):
            return self._failure("invalid-mode", "update requires a different current package")
        if (
            run_mode == "run"
            and record.current_package_id is not None
            and record.current_package_id != package_id
        ):
            return self._failure(
                "invalid-mode",
                f'package "{package_id}" differs from current "{record.current_package_id}"; use mode "update"',
            )
        if any(item.plugin_id == plugin_id for item in self._pending.values()):
            return self._failure(
                "transition-in-flight",
                f'dynamic plugin "{plugin_id}" already has a pending run request',
            )
        self._run_number += 1
        attempt = _Attempt(f"run-{self._run_number}", package_id, run_mode, "starting-host")
        record.next_package_id = package_id
        record.latest = attempt

        if package.client_code is None:
            try:
                await self._activate_host(record, package, attempt)
            except Exception as exc:
                self._fail_attempt(record, attempt, "host-load", str(exc))
                return self._failure("host-half-failed", str(exc))
            self._finish_host_only(record, attempt)
            return self._run_response(record, package, attempt, "running")

        self._approval_number += 1
        request_id = f"approval-{self._approval_number}"
        requires = not record.approve_future_versions and package_id not in record.approved_packages
        attempt.approval_id = request_id
        attempt.requires_approval = requires
        attempt.status = "awaiting-approval" if requires else "starting-host"
        self._pending[request_id] = _PendingRun(
            request_id,
            session_id,
            plugin_id,
            package_id,
            attempt.run_id,
            run_mode,
            requires,
        )
        self._emit(
            "cordis/request-run",
            {
                "requestId": request_id,
                "agentId": session_id,
                "pluginId": plugin_id,
                "packageId": package_id,
                "mode": run_mode,
                "name": package.name,
                "purpose": package.purpose,
                "requiresApproval": requires,
            },
        )
        return self._run_response(
            record,
            package,
            attempt,
            "awaiting-approval" if requires else "starting",
        )

    async def run_host_half(
        self,
        session_id: str,
        plugin_id: str,
        package_id: str,
        mode: str,
        request_id: str | None,
        approve_future_versions: bool,
    ) -> JsonObject:
        record = self._owned(session_id, plugin_id)
        package = record.packages.get(package_id)
        if package is None:
            return {"ok": False, "message": f'plugin "{plugin_id}" has no package "{package_id}"'}
        pending = self._pending.get(request_id) if request_id is not None else None
        if request_id is not None and (
            pending is None
            or pending.session_id != session_id
            or pending.plugin_id != plugin_id
            or pending.package_id != package_id
        ):
            return {
                "ok": False,
                "message": f'run request "{request_id}" no longer authorizes this package',
            }
        if mode not in {"run", "update"}:
            return {"ok": False, "message": 'mode must be "run" or "update"'}
        run_mode = cast(DynamicRunMode, mode)
        attempt = record.latest
        if attempt is None or attempt.package_id != package_id:
            self._run_number += 1
            attempt = _Attempt(f"run-{self._run_number}", package_id, run_mode, "starting-host")
            record.next_package_id = package_id
            record.latest = attempt
        if pending is not None and pending.requires_approval:
            record.approved_packages.add(package_id)
            if approve_future_versions:
                record.approve_future_versions = True
        try:
            started_here = record.run is None or record.run.run_id != attempt.run_id
            await self._activate_host(record, package, attempt)
        except Exception as exc:
            self._fail_attempt(record, attempt, "host-load", str(exc))
            return {"ok": False, "message": str(exc)}
        if package.client_code is not None:
            attempt.status = "client-pending"
            attempt.client_status = "pending"
        return {
            "ok": True,
            "pluginId": plugin_id,
            "packageId": package_id,
            "pluginRunId": attempt.run_id,
            "waitingFor": list(attempt.host_waiting),
            "startedHere": started_here,
        }

    def get_client_code(self, session_id: str, plugin_id: str, run_id: str) -> JsonObject:
        record = self._owned(session_id, plugin_id)
        if record.run is None or record.run.run_id != run_id:
            raise DynamicCordisError(
                f'dynamic plugin "{plugin_id}" is not running activation "{run_id}"'
            )
        package = record.packages[record.run.package_id]
        if package.client_code is None:
            raise DynamicCordisError(f'package "{package.package_id}" has no Client half')
        return {
            "code": package.client_code,
            "name": package.name,
            "pluginId": plugin_id,
            "packageId": package.package_id,
            "pluginRunId": run_id,
        }

    async def resolve_request_run(self, request_id: str, resolution: JsonObject) -> JsonObject:
        pending = self._pending.pop(request_id, None)
        if pending is None:
            return {"accepted": False}
        record = self._plugins.get(pending.plugin_id)
        if record is None or record.run is None or record.run.run_id != pending.run_id:
            return {"accepted": False}
        attempt = record.latest
        if attempt is None or attempt.run_id != pending.run_id:
            return {"accepted": False}
        if resolution.get("ok") is True and resolution.get("pluginRunId") == pending.run_id:
            attempt.status = "running"
            attempt.client_status = "running"
            waiting = resolution.get("waitingFor")
            attempt.client_waiting = (
                [item for item in waiting if isinstance(item, str)]
                if isinstance(waiting, list)
                else []
            )
            record.current_package_id = pending.package_id
            record.next_package_id = None
            self._emit(
                "cordis/request-run-resolved",
                {
                    "requestId": request_id,
                    "outcome": "approved" if pending.requires_approval else "completed",
                },
            )
            return {"accepted": True}
        reason = resolution.get("reason")
        attempt.status = "rejected" if reason == "rejected" else "failed" if reason else "cancelled"
        attempt.client_status = "failed" if reason not in {"rejected", "cancelled"} else "stopped"
        if isinstance(resolution.get("message"), str):
            attempt.client_error = resolution["message"]
        await self._retract(record)
        self._emit(
            "cordis/request-run-resolved",
            {
                "requestId": request_id,
                "outcome": reason if reason in {"rejected", "cancelled"} else "failed",
            },
        )
        return {"accepted": True}

    async def settle_user_run(
        self,
        session_id: str,
        plugin_id: str,
        resolution: JsonObject,
    ) -> JsonObject:
        record = self._owned(session_id, plugin_id)
        attempt = record.latest
        if attempt is None:
            return self._failure(
                "not-running", f'dynamic plugin "{plugin_id}" has no activation attempt'
            )
        if resolution.get("ok") is True and resolution.get("pluginRunId") == attempt.run_id:
            attempt.status = "running"
            attempt.client_status = "running"
            record.current_package_id = attempt.package_id
            record.next_package_id = None
            package = record.packages[attempt.package_id]
            return self._run_response(record, package, attempt, "running")
        attempt.status = "failed"
        attempt.client_status = "failed"
        attempt.client_error = str(resolution.get("message") or "client half failed")
        await self._retract(record)
        return self._failure("client-half-failed", attempt.client_error)

    async def stop(self, session_id: str, plugin_id: str) -> JsonObject:
        record = self._owned(session_id, plugin_id)
        pending_ids = [key for key, item in self._pending.items() if item.plugin_id == plugin_id]
        for request_id in pending_ids:
            self._pending.pop(request_id, None)
            self._emit(
                "cordis/request-run-resolved", {"requestId": request_id, "outcome": "cancelled"}
            )
        if record.run is None and not pending_ids:
            return {
                "ok": False,
                "reason": "not-running",
                "message": f'dynamic plugin "{plugin_id}" is not running',
            }
        await self._retract(record)
        if record.latest is not None:
            record.latest.status = "stopped"
            record.latest.host_status = (
                "stopped" if record.latest.host_status != "absent" else "absent"
            )
            record.latest.client_status = (
                "stopped" if record.latest.client_status != "absent" else "absent"
            )
        return {"ok": True}

    async def undefine(self, session_id: str, plugin_id: str) -> JsonObject:
        record = self._plugins.get(plugin_id)
        if record is None or record.session_id != session_id:
            return {
                "ok": False,
                "reason": "plugin-missing",
                "message": f'dynamic plugin "{plugin_id}" does not exist',
            }
        was_running = record.run is not None or any(
            item.plugin_id == plugin_id for item in self._pending.values()
        )
        if was_running:
            await self.stop(session_id, plugin_id)
        self._plugins.pop(plugin_id, None)
        return {"ok": True, "wasRunning": was_running}

    async def invoke(
        self,
        plugin_id: str,
        run_id: str,
        method: str,
        args: JsonValue,
    ) -> JsonObject:
        record = self._plugins.get(plugin_id)
        if record is None or record.run is None:
            return {
                "ok": False,
                "code": "plugin-not-running",
                "message": f'dynamic plugin "{plugin_id}" is not running',
            }
        if record.run.run_id != run_id:
            return {
                "ok": False,
                "code": "stale-run",
                "message": f'activation "{run_id}" is no longer active',
            }
        if method not in record.run.handlers:
            return {
                "ok": False,
                "code": "method-not-found",
                "message": f'dynamic plugin "{plugin_id}" registered no Host method "{method}"',
            }
        package = record.packages[record.run.package_id]
        if package.host_code is None:
            return {
                "ok": False,
                "code": "method-not-found",
                "message": "the active package has no Host half",
            }
        try:
            value = await _node_call("invoke", package.host_code, method, args)
        except Exception as exc:
            return {"ok": False, "code": "handler-error", "message": str(exc)}
        return {"ok": True, "value": value}

    def sync_inspect_manifest(self, providers: Any) -> None:
        if not isinstance(providers, list):
            raise DynamicCordisError("inspect providers must be an array")
        self._client_manifest = [dict(item) for item in providers if isinstance(item, dict)]

    def inspect_list(self) -> list[JsonObject]:
        host = {
            "id": "python-host",
            "description": "Native Python host runtime capabilities.",
            "platform": "host",
            "methods": [
                {
                    "name": "runtime",
                    "description": "Return the host implementation and dynamic runner state.",
                    "inputSchema": {"type": "object"},
                    "outputSchema": {"type": "object"},
                }
            ],
        }
        return [host, *[{**item, "platform": "client"} for item in self._client_manifest]]

    def resolve_inspect_query(
        self, session_id: str, request_id: str, resolution: JsonObject
    ) -> JsonObject:
        # Client inspect queries are optional until a browser sends one.  The
        # remote method is still real and rejects stale answers safely.
        del session_id, request_id, resolution
        return {"accepted": False}

    def report_render_failure(
        self, session_id: str, plugin_id: str, run_id: str, failure: JsonObject
    ) -> None:
        record = self._owned(session_id, plugin_id)
        if record.latest is not None and record.latest.run_id == run_id:
            record.latest.status = "failed"
            record.latest.client_status = "failed"
            record.latest.client_error = str(failure.get("message") or "client render failed")
            record.latest.error = {
                "phase": "client-render",
                "message": record.latest.client_error,
                "pluginId": plugin_id,
                "packageId": record.latest.package_id,
                "pluginRunId": run_id,
            }

    def report_client_guard_failure(
        self, session_id: str, plugin_id: str, run_id: str, failure: JsonObject
    ) -> None:
        self.report_render_failure(
            session_id,
            plugin_id,
            run_id,
            {"message": failure.get("message", "client guard failed")},
        )

    def inventory(self) -> list[JsonObject]:
        rows: list[JsonObject] = []
        for record in self._plugins.values():
            row: JsonObject = {
                "pluginId": record.plugin_id,
                "agentId": record.session_id,
                "packages": [package.view() for package in record.packages.values()],
            }
            if record.current_package_id is not None:
                row["currentPackageId"] = record.current_package_id
            if record.next_package_id is not None:
                row["nextPackageId"] = record.next_package_id
            if record.run is not None:
                row["activeRun"] = {
                    "pluginRunId": record.run.run_id,
                    "packageId": record.run.package_id,
                }
            if record.latest is not None:
                row["latestRun"] = record.latest.to_dict()
            rows.append(row)
        return rows

    def list_plugins(self, session_id: str) -> list[JsonObject]:
        return [
            self._inspect_plugin(record)
            for record in self._plugins.values()
            if record.session_id == session_id
        ]

    def inspect_plugin(self, session_id: str, plugin_id: str) -> JsonObject:
        return self._inspect_plugin(self._owned(session_id, plugin_id))

    def inspect_package(self, session_id: str, plugin_id: str, package_id: str) -> JsonObject:
        record = self._owned(session_id, plugin_id)
        package = record.packages.get(package_id)
        if package is None:
            raise DynamicCordisError(f'plugin "{plugin_id}" has no package "{package_id}"')
        result = self._inspect_plugin(record)
        result.update(
            {"packageId": package_id, "name": package.name, "purpose": package.purpose, "code": {}}
        )
        if package.host_code is not None:
            result["code"]["host"] = package.host_code
        if package.client_code is not None:
            result["code"]["client"] = package.client_code
        return result

    def _inspect_plugin(self, record: _Plugin) -> JsonObject:
        result: JsonObject = {
            "pluginId": record.plugin_id,
            "packages": [package.view() for package in record.packages.values()],
        }
        package_id = (
            record.next_package_id or record.current_package_id or next(iter(record.packages), None)
        )
        if package_id is not None:
            package = record.packages[package_id]
            result.update(
                {"packageId": package_id, "name": package.name, "purpose": package.purpose}
            )
        if record.current_package_id is not None:
            result["currentPackageId"] = record.current_package_id
        if record.next_package_id is not None:
            result["nextPackageId"] = record.next_package_id
        if record.run is not None:
            result["activeRun"] = {
                "pluginRunId": record.run.run_id,
                "packageId": record.run.package_id,
            }
        if record.latest is not None:
            result["latestRun"] = record.latest.to_dict()
        return result

    async def _activate_host(self, record: _Plugin, package: _Package, attempt: _Attempt) -> None:
        if record.run is not None and record.run.run_id == attempt.run_id:
            return
        if record.run is not None:
            await self._retract(record)
        run = _Run(attempt.run_id, package.package_id)
        attempt.host_status = "pending" if package.host_code is not None else "absent"
        if package.host_code is not None:
            descriptor = await _node_call("activate", package.host_code, package.package_id)
            handlers = descriptor.get("handlers", [])
            run.handlers = {item for item in handlers if isinstance(item, str)}
            try:
                self._register_dynamic_tools(
                    record.session_id, record.plugin_id, run, package, descriptor.get("tools", [])
                )
            except Exception:
                for dispose in reversed(run.tool_disposers):
                    dispose()
                raise
            attempt.host_status = "running"
        record.run = run
        self._emit(
            "cordis/dynamic-package",
            {
                "pluginId": record.plugin_id,
                "packageId": package.package_id,
                "pluginRunId": attempt.run_id,
                "name": package.name,
            },
        )

    async def _retract(self, record: _Plugin) -> None:
        run = record.run
        if run is None:
            return
        for dispose in reversed(run.tool_disposers):
            dispose()
        package_id = run.package_id
        record.run = None
        self._emit(
            "cordis/dynamic-retract",
            {"pluginId": record.plugin_id, "packageId": package_id, "pluginRunId": run.run_id},
        )

    def _register_dynamic_tools(
        self,
        session_id: str,
        plugin_id: str,
        run: _Run,
        package: _Package,
        tools: Any,
    ) -> None:
        registry = self._tool_registries.get(session_id)
        if registry is None or not isinstance(tools, list):
            return
        for item in tools:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                continue
            name = item["name"]
            if not name.strip():
                continue
            raw_description = item.get("description")
            description = (
                raw_description if isinstance(raw_description, str) else "Dynamic Cordis tool"
            )
            raw_parameters = item.get("parameters")
            parameters = raw_parameters if isinstance(raw_parameters, dict) else {"type": "object"}

            async def execute(args: dict[str, Any], _ctx: Any, tool_name: str = name) -> ToolResult:
                try:
                    value = await _node_call("tool", package.host_code or "", tool_name, args)
                except Exception as exc:
                    return ToolResult(str(exc), is_error=True)
                return ToolResult(json.dumps(value, ensure_ascii=False, separators=(",", ":")))

            run.tool_disposers.append(
                registry.register(
                    ToolDefinition(
                        name=name, description=description, parameters=parameters, execute=execute
                    )
                )
            )

    def _finish_host_only(self, record: _Plugin, attempt: _Attempt) -> None:
        attempt.status = "running"
        record.current_package_id = attempt.package_id
        record.next_package_id = None

    def _fail_attempt(self, record: _Plugin, attempt: _Attempt, phase: str, message: str) -> None:
        attempt.status = "failed"
        attempt.host_status = "failed" if phase.startswith("host") else attempt.host_status
        attempt.host_error = message if phase.startswith("host") else attempt.host_error
        attempt.error = {
            "phase": phase,
            "message": message,
            "pluginId": record.plugin_id,
            "packageId": attempt.package_id,
            "pluginRunId": attempt.run_id,
        }

    @staticmethod
    def _failure(reason: str, message: str) -> JsonObject:
        return {"ok": False, "reason": reason, "message": message}

    @staticmethod
    def _run_response(
        record: _Plugin, package: _Package, attempt: _Attempt, status: str
    ) -> JsonObject:
        result: JsonObject = {
            "ok": True,
            "status": status,
            "pluginId": record.plugin_id,
            "packageId": package.package_id,
            "pluginRunId": attempt.run_id,
            "waitingFor": list(attempt.host_waiting),
            "mode": attempt.mode,
            "nextPackageId": package.package_id,
        }
        if record.current_package_id is not None:
            result["currentPackageId"] = record.current_package_id
        if attempt.client_waiting:
            result["clientWaitingFor"] = list(attempt.client_waiting)
        return result

    def _owned(self, session_id: str, plugin_id: str) -> _Plugin:
        record = self._plugins.get(plugin_id)
        if record is None or record.session_id != session_id:
            raise DynamicCordisError(
                f'dynamic plugin "{plugin_id}" does not exist for this session'
            )
        return record

    @staticmethod
    def _required_text(value: Any, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise DynamicCordisError(f"cordis_define needs a non-empty `{field_name}`")
        return value.strip()

    @staticmethod
    def _optional_code(code: JsonObject, key: str) -> str | None:
        value = code.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise DynamicCordisError(f"code.{key} must be a string")
        if not value.strip():
            raise DynamicCordisError(f"code.{key} must not be empty")
        return value

    def _emit(self, event: str, value: JsonObject) -> None:
        if self._remote_event is not None:
            self._remote_event(event, [value])


def install_dynamic_tools(
    registry: ToolRegistry,
    runner: DynamicCordisService,
    session_id: str,
) -> list[Callable[[], None]]:
    """Install the model-facing Cordis tool family for one session."""

    async def inspect_list(_args: dict[str, Any], _ctx: Any) -> ToolResult:
        return _json_result({"providers": runner.inspect_list()})

    async def inspect_query(args: dict[str, Any], _ctx: Any) -> ToolResult:
        platform = args.get("platform")
        provider = args.get("provider")
        method = args.get("method")
        if not all(isinstance(item, str) and item for item in (platform, provider, method)):
            return ToolResult(
                "platform, provider and method must be non-empty strings", is_error=True
            )
        if platform == "host" and provider == "python-host" and method == "runtime":
            return _json_result(
                {
                    "implementation": "deepseek-harness-python",
                    "dynamicPlugins": len(runner.inventory()),
                    "nodeBridge": shutil.which("node") is not None,
                }
            )
        if platform == "client":
            return ToolResult(
                "client Cordis inspect queries require a connected browser provider",
                is_error=True,
            )
        return ToolResult(f"inspect method {provider}.{method} is not available", is_error=True)

    async def inspect_self(args: dict[str, Any], _ctx: Any) -> ToolResult:
        try:
            plugin_id = args.get("pluginId")
            package_id = args.get("packageId")
            if package_id is not None and plugin_id is None:
                raise DynamicCordisError("packageId requires pluginId")
            if plugin_id is None:
                return _json_result({"mode": "plugins", "plugins": runner.list_plugins(session_id)})
            if not isinstance(plugin_id, str):
                raise DynamicCordisError("pluginId must be a string")
            if package_id is None:
                return _json_result(
                    {"mode": "plugin", **runner.inspect_plugin(session_id, plugin_id)}
                )
            if not isinstance(package_id, str):
                raise DynamicCordisError("packageId must be a string")
            return _json_result(
                {"mode": "package", **runner.inspect_package(session_id, plugin_id, package_id)}
            )
        except Exception as exc:
            return ToolResult(str(exc), is_error=True)

    async def define(args: dict[str, Any], _ctx: Any) -> ToolResult:
        try:
            plugin = args.get("plugin")
            code = args.get("code")
            if not isinstance(plugin, dict) or not isinstance(code, dict):
                raise DynamicCordisError("plugin and code must be objects")
            return _json_result(
                runner.define(
                    session_id,
                    plugin,
                    args.get("name"),
                    args.get("purpose"),
                    code,
                )
            )
        except Exception as exc:
            return ToolResult(str(exc), is_error=True)

    async def run(args: dict[str, Any], _ctx: Any) -> ToolResult:
        try:
            result = await runner.run(
                session_id,
                str(args.get("pluginId", "")),
                str(args.get("packageId", "")),
                str(args.get("mode", "")),
            )
            return _json_result(result, is_error=not result.get("ok", False))
        except Exception as exc:
            return ToolResult(str(exc), is_error=True)

    async def stop(args: dict[str, Any], _ctx: Any) -> ToolResult:
        try:
            return _json_result(await runner.stop(session_id, str(args.get("pluginId", ""))))
        except Exception as exc:
            return ToolResult(str(exc), is_error=True)

    async def undefine(args: dict[str, Any], _ctx: Any) -> ToolResult:
        try:
            result = await runner.undefine(session_id, str(args.get("pluginId", "")))
            return _json_result(result, is_error=not result.get("ok", False))
        except Exception as exc:
            return ToolResult(str(exc), is_error=True)

    definitions = [
        ToolDefinition(
            "cordis_inspect_list",
            "List read-only dynamic Cordis inspect providers before writing a package.",
            {"type": "object", "additionalProperties": False},
            inspect_list,
        ),
        ToolDefinition(
            "cordis_inspect_self",
            "Inspect dynamic Cordis packages owned by this session.",
            {
                "type": "object",
                "properties": {"pluginId": {"type": "string"}, "packageId": {"type": "string"}},
                "additionalProperties": False,
            },
            inspect_self,
        ),
        ToolDefinition(
            "cordis_inspect_query",
            "Run one explicitly declared read-only Cordis inspect query.",
            {
                "type": "object",
                "properties": {
                    "platform": {"type": "string", "enum": ["host", "client"]},
                    "provider": {"type": "string"},
                    "method": {"type": "string"},
                    "input": {"type": "json"},
                },
                "required": ["platform", "provider", "method"],
                "additionalProperties": False,
            },
            inspect_query,
        ),
        ToolDefinition(
            "cordis_define",
            "Define an immutable dynamic Cordis package; it is not running until cordis_run.",
            {
                "type": "object",
                "properties": {
                    "plugin": {"type": "object"},
                    "name": {"type": "string"},
                    "purpose": {"type": "string"},
                    "code": {"type": "object"},
                },
                "required": ["plugin", "name", "purpose", "code"],
                "additionalProperties": False,
            },
            define,
        ),
        ToolDefinition(
            "cordis_run",
            "Activate an exact dynamic Cordis package with mode run or update.",
            {
                "type": "object",
                "properties": {
                    "pluginId": {"type": "string"},
                    "packageId": {"type": "string"},
                    "mode": {"type": "string", "enum": ["run", "update"]},
                },
                "required": ["pluginId", "packageId", "mode"],
                "additionalProperties": False,
            },
            run,
        ),
        ToolDefinition(
            "cordis_stop",
            "Stop a dynamic Cordis package while retaining its versions.",
            {
                "type": "object",
                "properties": {"pluginId": {"type": "string"}},
                "required": ["pluginId"],
                "additionalProperties": False,
            },
            stop,
        ),
        ToolDefinition(
            "cordis_undefine",
            "Permanently remove a dynamic Cordis package and all versions.",
            {
                "type": "object",
                "properties": {"pluginId": {"type": "string"}},
                "required": ["pluginId"],
                "additionalProperties": False,
            },
            undefine,
        ),
    ]
    return [registry.register(definition) for definition in definitions]


def _json_result(value: JsonValue, *, is_error: bool = False) -> ToolResult:
    return ToolResult(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")), is_error=is_error
    )


async def _node_call(operation: str, code: str, *args: JsonValue) -> JsonValue:
    return await asyncio.to_thread(_node_call_sync, operation, code, args)


def _node_call_sync(operation: str, code: str, args: tuple[JsonValue, ...]) -> JsonValue:
    executable = shutil.which("node")
    if executable is None:
        raise DynamicCordisError("dynamic Cordis host execution requires Node.js")
    payload = json.dumps(
        {"operation": operation, "code": code, "args": list(args)},
        ensure_ascii=False,
        allow_nan=False,
    )
    try:
        completed = subprocess.run(
            [executable, "-e", _NODE_BRIDGE],
            input=payload,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise DynamicCordisError("dynamic Cordis host evaluation timed out") from exc
    output = completed.stdout.strip()
    if not output:
        raise DynamicCordisError(
            completed.stderr.strip() or "dynamic Cordis host evaluation returned no result"
        )
    try:
        result = json.loads(output)
    except json.JSONDecodeError as exc:
        raise DynamicCordisError(output[:1000]) from exc
    if not isinstance(result, dict) or result.get("ok") is not True:
        message = result.get("error") if isinstance(result, dict) else output
        raise DynamicCordisError(str(message))
    return result.get("value")


_NODE_BRIDGE = r"""'use strict';
const vm = require('node:vm')
const fs = require('node:fs')

function jsonOut(value) {
  process.stdout.write(JSON.stringify({ ok: true, value: value === undefined ? null : value }))
}
function fail(error) {
  const message = error && error.stack ? error.stack : String(error)
  process.stdout.write(JSON.stringify({ ok: false, error: message }))
}
function run() {
  const input = JSON.parse(fs.readFileSync(0, 'utf8'))
  const handlers = new Map()
  const tools = new Map()
  const provided = Object.create(null)
  const disposable = []
  const addTool = (definition) => {
    if (!definition || typeof definition !== 'object' || typeof definition.name !== 'string' || !definition.name.trim()) {
      throw new Error('dynamic tool registration needs a non-empty name')
    }
    tools.set(definition.name, definition)
  }
  const harness = {
    handle(name, fn) {
      if (typeof name !== 'string' || !name.trim() || typeof fn !== 'function') throw new Error('harness.handle needs a name and function')
      handlers.set(name, fn)
      return () => handlers.delete(name)
    },
    defineTool(definition) { return definition },
    registerTool(ctx, definition) {
      if (!ctx || !ctx.tools || typeof ctx.tools.register !== 'function') throw new Error('dynamic context has no tools.register')
      ctx.tools.register(definition)
      return () => {}
    },
  }
  const fakeCtx = {
    provide(name, value) { provided[name] = value },
    get(name) { return provided[name] },
    on() { return () => {} },
    effect(fn) { disposable.push(typeof fn === 'function' ? fn : () => {}); return () => {} },
    tools: {
      register(definition) { addTool(definition); return () => tools.delete(definition.name) },
      get(name) { const def = tools.get(name); return def ? { name: def.name, description: def.description, parameters: def.parameters } : undefined },
      schemas() { return Array.from(tools.values()).map(def => ({ name: def.name, description: def.description, parameters: def.parameters })) },
    },
  }
  const sandbox = {
    harness,
    console: { log() {}, warn() {}, error() {} },
    TextEncoder,
    TextDecoder,
  }
  vm.createContext(sandbox, { name: 'deepseek-harness-dynamic' })
  const source = `(async () => {\n${input.code}\n})()`
  if (input.operation === 'check') {
    new vm.Script(source, { filename: `cordis-dynamic-${input.args[0] || 'package'}.js` })
    return jsonOut(true)
  }
  const plugin = vm.runInContext(source, sandbox, { timeout: 5000, filename: 'cordis-dynamic-package.js' })
  Promise.resolve(plugin).then(async (value) => {
    if (!value || typeof value !== 'object') throw new Error('dynamic host code must return a Cordis Plugin object')
    if (typeof value.apply === 'function') await value.apply(fakeCtx)
    if (input.operation === 'activate') {
      return {
        handlers: Array.from(handlers.keys()),
        tools: Array.from(tools.values()).map(def => ({
          name: def.name,
          description: typeof def.description === 'string' ? def.description : 'Dynamic Cordis tool',
          parameters: def.parameters && typeof def.parameters === 'object' ? def.parameters : { type: 'object' },
        })),
      }
    }
    if (input.operation === 'invoke') {
      const fn = handlers.get(input.args[0])
      if (!fn) throw new Error(`dynamic host handler not found: ${input.args[0]}`)
      return await fn(input.args[1])
    }
    if (input.operation === 'tool') {
      const def = tools.get(input.args[0])
      if (!def || typeof def.execute !== 'function') throw new Error(`dynamic host tool not found: ${input.args[0]}`)
      return await def.execute(input.args[1], { signal: new AbortController().signal })
    }
    throw new Error(`unknown dynamic operation: ${input.operation}`)
  }).then(jsonOut, fail)
}
try { run() } catch (error) { fail(error) }
"""
