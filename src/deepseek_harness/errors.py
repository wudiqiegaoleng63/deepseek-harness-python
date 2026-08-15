"""Stable exception classes used across runtime seams."""


class HarnessError(Exception):
    """Base class for expected DeepSeek Harness failures."""


class ConfigurationError(HarnessError):
    """Raised when a runtime configuration cannot be used."""


class PluginError(HarnessError):
    """Raised when a plugin cannot be mounted or disposed."""


class SessionError(HarnessError):
    """Raised when a session log is malformed or cannot be persisted."""


class LlmError(HarnessError):
    """Raised when an LLM adapter cannot complete a request."""


class ToolError(HarnessError):
    """Raised when a model-facing tool cannot be executed."""
