"""Stable exception classes used across runtime seams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class HarnessError(Exception):
    """Base class for expected DeepSeek Harness failures."""


class ConfigurationError(HarnessError):
    """Raised when a runtime configuration cannot be used."""


class PluginError(HarnessError):
    """Raised when a plugin cannot be mounted or disposed."""


class SessionError(HarnessError):
    """Raised when a session log is malformed or cannot be persisted."""


class LlmError(HarnessError):
    """Raised when an LLM adapter cannot complete a request.

    ``code`` and the optional transport facts are intentionally separate from
    the human-readable message so the Agent can make retry decisions without
    parsing provider text.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "UNKNOWN",
        status: int | None = None,
        retry_after_ms: int | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.retry_after_ms = retry_after_ms
        self.request_id = request_id

    def failure(self) -> LlmFailure:
        return LlmFailure(
            message=str(self),
            code=self.code,
            status=self.status,
            retry_after_ms=self.retry_after_ms,
            request_id=self.request_id,
        )


class ToolError(HarnessError):
    """Raised when a model-facing tool cannot be executed."""


class WebError(HarnessError):
    """Raised when a web provider cannot safely complete a search or fetch."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "WEB_PROVIDER_ERROR",
        status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass(frozen=True, slots=True)
class LlmFailure:
    """Provider-neutral facts used by retry and terminal event policy."""

    message: str
    code: str
    status: int | None = None
    retry_after_ms: int | None = None
    request_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"message": self.message, "code": self.code}
        for key, value in (
            ("status", self.status),
            ("providerRetryAfterMs", self.retry_after_ms),
            ("requestId", self.request_id),
        ):
            if value is not None:
                result[key] = value
        return result
