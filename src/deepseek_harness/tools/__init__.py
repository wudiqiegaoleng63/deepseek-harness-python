"""Model-facing tool registry and local capability providers."""

from .builtin import install_builtin_tools
from .policy import PermissionMode, WorkspacePolicy
from .registry import ToolContext, ToolDefinition, ToolRegistry, ToolResult

__all__ = [
    "PermissionMode",
    "ToolContext",
    "ToolDefinition",
    "ToolRegistry",
    "ToolResult",
    "WorkspacePolicy",
    "install_builtin_tools",
]
