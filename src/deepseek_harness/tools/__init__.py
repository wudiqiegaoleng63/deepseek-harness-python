"""Model-facing tool registry and local capability providers."""

from .builtin import install_builtin_tools, install_shell_tools
from .policy import PermissionMode, WorkspacePolicy
from .registry import ToolContext, ToolDefinition, ToolRegistry, ToolResult
from .terminal import install_terminal_tools

__all__ = [
    "PermissionMode",
    "ToolContext",
    "ToolDefinition",
    "ToolRegistry",
    "ToolResult",
    "WorkspacePolicy",
    "install_builtin_tools",
    "install_shell_tools",
    "install_terminal_tools",
]
