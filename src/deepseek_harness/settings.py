"""Host-owned settings and credential registries.

The wire contract intentionally exposes redacted settings descriptors while
credentials have a separate value-free describe API.  This module keeps those
rules in one place so the FastAPI service does not accidentally serialize a
secret while adding a new RPC method.
"""

from __future__ import annotations

import asyncio
import copy
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .persistence import JsonStateStore

_CREDENTIAL_REF = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SettingsError(Exception):
    """Base class for settings-domain failures."""


class SettingsNotFound(SettingsError):
    def __init__(self, namespace: str) -> None:
        super().__init__(f'settings namespace "{namespace}" not found')
        self.namespace = namespace


class SettingsConflict(SettingsError):
    def __init__(self, namespace: str, expected: int, actual: int) -> None:
        super().__init__(
            f'settings namespace "{namespace}" changed: expected revision {expected}, '
            f"actual revision {actual}"
        )
        self.namespace = namespace
        self.expected = expected
        self.actual = actual


class CredentialError(Exception):
    def __init__(self, ref: str, message: str) -> None:
        super().__init__(message)
        self.ref = ref


@dataclass(frozen=True, slots=True)
class SettingsNamespace:
    namespace: str
    schema: Any
    base: dict[str, Any]
    user: dict[str, Any]
    applies: str
    secrets: tuple[dict[str, Any], ...]
    revision: int

    def view(self) -> dict[str, Any]:
        value = _deep_merge(self.base, self.user)
        secret_views: list[dict[str, Any]] = []
        for secret in self.secrets:
            path = secret.get("path")
            secret_view = copy.deepcopy(secret)
            secret_view["set"] = _has_path(value, path)
            secret_views.append(secret_view)
        result: dict[str, Any] = {
            "ns": self.namespace,
            "schema": copy.deepcopy(self.schema),
            "value": _redact(value, self.secrets),
            "applies": self.applies,
            "secrets": secret_views,
            "revision": self.revision,
        }
        if self.base:
            result["base"] = _redact(self.base, self.secrets)
        if self.user:
            result["user"] = _redact(self.user, self.secrets)
        return result


class SettingsRegistry:
    """Versioned user-layer settings with atomic snapshot persistence."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._store = JsonStateStore(
            Path(root).expanduser().resolve() / "settings.json",
            default={"version": 1, "namespaces": {}},
        )
        self._definitions: dict[str, SettingsNamespace] = {}
        self._user: dict[str, dict[str, Any]] = {}
        self._revisions: dict[str, int] = {}
        self._loaded = False
        self._lock = asyncio.Lock()

    def register(
        self,
        namespace: str,
        *,
        schema: Any = None,
        base: dict[str, Any] | None = None,
        applies: str = "live",
        secrets: tuple[dict[str, Any], ...] = (),
    ) -> None:
        if not namespace.strip():
            raise ValueError("settings namespace cannot be empty")
        if applies not in {"live", "restart"}:
            raise ValueError("settings applies must be live or restart")
        self._definitions[namespace] = SettingsNamespace(
            namespace=namespace,
            schema={} if schema is None else schema,
            base=copy.deepcopy(base or {}),
            user={},
            applies=applies,
            secrets=tuple(copy.deepcopy(secrets)),
            revision=0,
        )

    async def describe(self, *, exposed: set[str] | None = None) -> list[dict[str, Any]]:
        async with self._lock:
            await self._ensure_loaded()
            names = list(self._definitions)
            if exposed is not None:
                names = [name for name in names if name in exposed]
            return [self._view_unlocked(name) for name in names]

    async def get(self, namespace: str) -> dict[str, Any]:
        async with self._lock:
            await self._ensure_loaded()
            return self._view_unlocked(namespace)

    def get_value_sync(self, namespace: str) -> dict[str, Any]:
        """Read the effective value for synchronous adapter construction."""

        definition = self._definitions.get(namespace)
        if definition is None:
            raise SettingsNotFound(namespace)
        user: dict[str, Any] = copy.deepcopy(self._user.get(namespace, {}))
        if not self._loaded:
            try:
                import json

                raw = json.loads(self._store.path.read_text(encoding="utf-8"))
                persisted = raw.get("namespaces", {}).get(namespace, {})
                candidate = persisted.get("user", {}) if isinstance(persisted, dict) else {}
                if isinstance(candidate, dict):
                    user = copy.deepcopy(candidate)
            except (FileNotFoundError, OSError, TypeError, ValueError):
                pass
        return _deep_merge(definition.base, user)

    async def update(
        self,
        namespace: str,
        patch: dict[str, Any],
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            await self._ensure_loaded()
            current = self._namespace_unlocked(namespace)
            self._check_revision(current, expected_revision)
            self._user[namespace] = _deep_merge(self._user.get(namespace, {}), patch)
            self._revisions[namespace] = current.revision + 1
            await self._save()
            return self._view_unlocked(namespace)

    async def replace(
        self,
        namespace: str,
        section: dict[str, Any],
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            await self._ensure_loaded()
            current = self._namespace_unlocked(namespace)
            self._check_revision(current, expected_revision)
            self._user[namespace] = copy.deepcopy(section)
            self._revisions[namespace] = current.revision + 1
            await self._save()
            return self._view_unlocked(namespace)

    async def mutate(
        self,
        namespace: str,
        operations: list[dict[str, Any]],
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            await self._ensure_loaded()
            current = self._namespace_unlocked(namespace)
            self._check_revision(current, expected_revision)
            section = copy.deepcopy(self._user.get(namespace, {}))
            if not isinstance(section, dict):
                section = {}
            for operation in operations:
                if not isinstance(operation, dict):
                    raise ValueError("settings mutation must be an object")
                path = operation.get("path")
                if not isinstance(path, list) or not all(isinstance(item, str) for item in path):
                    raise ValueError("settings mutation path must be a string array")
                if operation.get("op") == "set":
                    _set_path(section, path, copy.deepcopy(operation.get("value")))
                elif operation.get("op") == "unset":
                    _unset_path(section, path)
                else:
                    raise ValueError("settings mutation op must be set or unset")
            self._user[namespace] = section
            self._revisions[namespace] = current.revision + 1
            await self._save()
            return self._view_unlocked(namespace)

    async def open_document(self) -> dict[str, Any]:
        # The native editor hand-off is a carrier capability.  Returning an
        # acknowledgement keeps the browser settings surface usable in the
        # server deployment; the CLI can expose the JSON file directly.
        return {"opened": True}

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        value = await self._store.load()
        namespaces = value.get("namespaces", {})
        if isinstance(namespaces, dict):
            for name, raw in namespaces.items():
                if isinstance(raw, dict):
                    user = raw.get("user", {})
                    revision = raw.get("revision", 0)
                    if isinstance(user, dict):
                        self._user[str(name)] = copy.deepcopy(user)
                    if isinstance(revision, int) and revision >= 0:
                        self._revisions[str(name)] = revision
        self._loaded = True

    async def _save(self) -> None:
        await self._store.save(
            {
                "version": 1,
                "namespaces": {
                    name: {
                        "user": copy.deepcopy(self._user.get(name, {})),
                        "revision": self._revisions.get(name, 0),
                    }
                    for name in self._definitions
                    if self._user.get(name) or self._revisions.get(name, 0) > 0
                },
            }
        )

    def _namespace_unlocked(self, namespace: str) -> SettingsNamespace:
        definition = self._definitions.get(namespace)
        if definition is None:
            raise SettingsNotFound(namespace)
        return SettingsNamespace(
            namespace=namespace,
            schema=definition.schema,
            base=definition.base,
            user=copy.deepcopy(self._user.get(namespace, {})),
            applies=definition.applies,
            secrets=definition.secrets,
            revision=self._revisions.get(namespace, 0),
        )

    def _view_unlocked(self, namespace: str) -> dict[str, Any]:
        return self._namespace_unlocked(namespace).view()

    @staticmethod
    def _check_revision(current: SettingsNamespace, expected_revision: int | None) -> None:
        if expected_revision is not None and expected_revision != current.revision:
            raise SettingsConflict(current.namespace, expected_revision, current.revision)


class CredentialStore:
    """Local writable credential layer with environment shadowing."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._store = JsonStateStore(
            Path(root).expanduser().resolve() / "credentials.json",
            default={"version": 1, "credentials": {}},
        )
        self._values: dict[str, str] = {}
        self._loaded = False
        self._lock = asyncio.Lock()

    async def describe(self, refs: list[str]) -> dict[str, dict[str, Any]]:
        async with self._lock:
            await self._ensure_loaded()
            return {ref: self._describe_unlocked(ref) for ref in refs}

    async def set(self, ref: str, value: str) -> None:
        _validate_ref(ref)
        if not value:
            raise CredentialError(ref, "credential value cannot be empty")
        async with self._lock:
            await self._ensure_loaded()
            if os.getenv(ref):
                raise CredentialError(ref, f'credential "{ref}" is shadowed by the environment')
            self._values[ref] = value
            await self._save()

    async def unset(self, ref: str) -> None:
        _validate_ref(ref)
        async with self._lock:
            await self._ensure_loaded()
            if os.getenv(ref):
                raise CredentialError(ref, f'credential "{ref}" is shadowed by the environment')
            self._values.pop(ref, None)
            await self._save()

    async def resolve(self, ref: str) -> str | None:
        _validate_ref(ref)
        async with self._lock:
            await self._ensure_loaded()
            return os.getenv(ref) or self._values.get(ref)

    def resolve_sync(self, ref: str) -> str | None:
        """Resolve credentials while constructing a synchronous adapter."""

        _validate_ref(ref)
        environment = os.getenv(ref)
        if environment:
            return environment
        if self._loaded:
            return self._values.get(ref)
        try:
            import json

            raw = json.loads(self._store.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, TypeError, ValueError, OSError):
            return None
        values = raw.get("credentials", {}) if isinstance(raw, dict) else {}
        value = values.get(ref) if isinstance(values, dict) else None
        return value if isinstance(value, str) and value else None

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        value = await self._store.load()
        credentials = value.get("credentials", {})
        if isinstance(credentials, dict):
            self._values = {
                str(key): str(item)
                for key, item in credentials.items()
                if isinstance(key, str) and isinstance(item, str) and item
            }
        self._loaded = True

    async def _save(self) -> None:
        await self._store.save({"version": 1, "credentials": dict(self._values)})

    def _describe_unlocked(self, ref: str) -> dict[str, Any]:
        _validate_ref(ref)
        environment = os.getenv(ref)
        if environment:
            return {"configured": True, "source": "env", "writable": False}
        if self._values.get(ref):
            return {"configured": True, "source": "file", "writable": True}
        return {"configured": False, "writable": True}


def _validate_ref(ref: str) -> None:
    if not _CREDENTIAL_REF.fullmatch(ref):
        raise ValueError(f"invalid credential reference: {ref!r}")


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(left)
    for key, value in right.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _redact(value: Any, secrets: tuple[dict[str, Any], ...]) -> Any:
    result = copy.deepcopy(value)
    for secret in secrets:
        path = secret.get("path")
        if isinstance(path, list) and all(isinstance(item, str) for item in path):
            _unset_path(result, path)
    return result


def _set_path(root: dict[str, Any], path: list[str], value: Any) -> None:
    if not path:
        if not isinstance(value, dict):
            raise ValueError("root settings value must be an object")
        root.clear()
        root.update(value)
        return
    target: dict[str, Any] = root
    for key in path[:-1]:
        child = target.get(key)
        if not isinstance(child, dict):
            child = {}
            target[key] = child
        target = child
    target[path[-1]] = value


def _unset_path(root: Any, path: list[str]) -> None:
    if not path:
        if isinstance(root, dict):
            root.clear()
        return
    target = root
    for key in path[:-1]:
        if not isinstance(target, dict):
            return
        target = target.get(key)
    if isinstance(target, dict):
        target.pop(path[-1], None)


def _has_path(root: Any, path: Any) -> bool:
    if not isinstance(path, list) or not all(isinstance(item, str) for item in path):
        return False
    current = root
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


__all__ = [
    "CredentialError",
    "CredentialStore",
    "SettingsConflict",
    "SettingsError",
    "SettingsNamespace",
    "SettingsNotFound",
    "SettingsRegistry",
]
