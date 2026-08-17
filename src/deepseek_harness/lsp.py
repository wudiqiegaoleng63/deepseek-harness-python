"""Provider-neutral LSP capability and the model-facing ``lsp`` tool.

The TypeScript implementation keeps the LSP boundary deliberately small: providers own
language-server lifecycle and protocol details, while the harness selects a provider by file
extension and exposes only four semantic queries.  This module mirrors that boundary without
requiring an LSP process or a third-party server in the default installation.
"""

from __future__ import annotations

import math
import ntpath
import posixpath
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast
from urllib.parse import unquote, urlsplit

from .errors import LspError
from .tools.registry import ToolContext, ToolDefinition, ToolRegistry, ToolResult

LspOperation = Literal[
    "goToDefinition",
    "findReferences",
    "goToImplementation",
    "hover",
]

LSP_OPERATIONS: tuple[LspOperation, ...] = (
    "goToDefinition",
    "findReferences",
    "goToImplementation",
    "hover",
)
DEFAULT_MAX_LOCATIONS = 100
DEFAULT_MAX_RESULT_CHARS = 16_000
DEFAULT_LSP_TOOL_TIMEOUT_SECONDS = 60.0
LSP_PROMPT_TEXT = (
    "Use search/read for ordinary navigation. Use lsp when textual matches are ambiguous or "
    "before a change requires precise definitions, implementations, or references. Positions "
    "are one-based line and character (UTF-16) at the cursor; an off-symbol position may return "
    "no results. findReferences always includes the declaration."
)
_EXTENSION_PATTERN = re.compile(r"\.[^./\\]+$")
_WINDOWS_DRIVE_PATH = re.compile(r"^/[a-z](?::|%3a)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class LspPosition:
    """A zero-based UTF-16 cursor coordinate."""

    line: int
    character: int


@dataclass(frozen=True, slots=True)
class LspRange:
    """A zero-based UTF-16 half-open range."""

    start: LspPosition
    end: LspPosition


@dataclass(frozen=True, slots=True)
class LspLocation:
    """A document URI and the range returned by a language server."""

    uri: str
    range: LspRange


@dataclass(frozen=True, slots=True)
class LspHover:
    """Normalized hover text and its optional source range."""

    contents: str
    range: LspRange | None = None


@dataclass(frozen=True, slots=True)
class LspQueryRequest:
    """A normalized query received by the capability seam."""

    operation: LspOperation
    file_path: str
    position: LspPosition
    workspace_root: str


@dataclass(frozen=True, slots=True)
class LspProviderQuery:
    """A query after extension routing has supplied the provider language id."""

    operation: LspOperation
    file_path: str
    position: LspPosition
    workspace_root: str
    language_id: str


@dataclass(frozen=True, slots=True)
class LspLocationsResult:
    kind: Literal["locations"]
    locations: tuple[LspLocation, ...]
    resolved_workspace_uri: str


@dataclass(frozen=True, slots=True)
class LspHoverResult:
    kind: Literal["hover"]
    hover: LspHover | None


LspQueryResult = LspLocationsResult | LspHoverResult


class LspProvider(Protocol):
    """The narrow provider contract; protocol lifecycle stays outside the harness seam."""

    id: str
    extension_to_language: Mapping[str, str]

    async def query(self, request: LspProviderQuery) -> LspQueryResult: ...


@dataclass(frozen=True, slots=True)
class LspToolInput:
    operation: LspOperation
    file_path: str
    position: LspPosition


@dataclass(frozen=True, slots=True)
class _LspRoute:
    provider: LspProvider
    language_id: str


def final_extension(file_path: str) -> str:
    """Return the lowercase final extension, treating dotfiles as extensionless."""

    last_separator = max(file_path.rfind("/"), file_path.rfind("\\"))
    basename = file_path[last_separator + 1 :]
    dot = basename.rfind(".")
    if dot <= 0:
        return ""
    return basename[dot:].lower()


def normalize_extension(extension: str) -> str:
    """Normalize and validate a provider extension as a leading-dot lowercase key."""

    if not isinstance(extension, str):
        raise LspError("LSP provider extensions must be strings", code="LSP_INVALID_PROVIDER")
    normalized = extension.lower()
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    if _EXTENSION_PATTERN.fullmatch(normalized) is None:
        raise LspError(
            f'invalid LSP provider extension "{extension}"',
            code="LSP_INVALID_PROVIDER",
        )
    return normalized


class LspRuntime:
    """Register providers and route queries by the requested file's final extension."""

    def __init__(self) -> None:
        self._providers: dict[str, LspProvider] = {}
        self._routes: dict[str, _LspRoute] = {}

    def register_provider(self, provider: LspProvider) -> Callable[[], None]:
        """Atomically reserve a provider id and all of its normalized extensions."""

        provider_id = getattr(provider, "id", None)
        if not isinstance(provider_id, str) or not provider_id.strip():
            raise LspError(
                "an LSP provider id must be a non-empty string",
                code="LSP_INVALID_PROVIDER",
            )
        if provider_id in self._providers:
            raise LspError(
                f'an LSP provider with id "{provider_id}" is already registered',
                code="LSP_CONFLICT",
            )

        extensions = getattr(provider, "extension_to_language", None)
        if not isinstance(extensions, Mapping) or not extensions:
            raise LspError(
                f'LSP provider "{provider_id}" registers no file extensions',
                code="LSP_INVALID_PROVIDER",
            )

        pending: dict[str, _LspRoute] = {}
        for raw_extension, language_id in extensions.items():
            if not isinstance(raw_extension, str):
                raise LspError(
                    f'LSP provider "{provider_id}" maps a non-string extension',
                    code="LSP_INVALID_PROVIDER",
                )
            normalized = normalize_extension(raw_extension)
            if not isinstance(language_id, str) or not language_id.strip():
                raise LspError(
                    f'LSP provider "{provider_id}" maps extension "{normalized}" '
                    "to an empty language id",
                    code="LSP_INVALID_PROVIDER",
                )
            if normalized in pending:
                raise LspError(
                    f'LSP provider "{provider_id}" maps extension "{normalized}" more than once',
                    code="LSP_INVALID_PROVIDER",
                )
            pending[normalized] = _LspRoute(provider, language_id)

        for extension in pending:
            if extension in self._routes:
                raise LspError(
                    f'extension "{extension}" is already handled by another LSP provider',
                    code="LSP_CONFLICT",
                )

        self._providers[provider_id] = provider
        self._routes.update(pending)
        active = True

        def dispose() -> None:
            nonlocal active
            if not active:
                return
            active = False
            if self._providers.get(provider_id) is provider:
                self._providers.pop(provider_id, None)
            for extension in pending:
                route = self._routes.get(extension)
                if route is not None and route.provider is provider:
                    self._routes.pop(extension, None)

        return dispose

    async def query(self, request: LspQueryRequest) -> LspQueryResult:
        """Select a provider and forward the normalized request plus language id."""

        if request.operation not in LSP_OPERATIONS:
            raise LspError(
                f"unsupported LSP operation: {request.operation}",
                code="LSP_UNSUPPORTED_OPERATION",
            )
        route = self._routes.get(final_extension(request.file_path))
        if route is None:
            raise LspError(
                f'no LSP provider handles "{request.file_path}"',
                code="LSP_UNAVAILABLE",
            )
        return await route.provider.query(
            LspProviderQuery(
                operation=request.operation,
                file_path=request.file_path,
                position=request.position,
                workspace_root=request.workspace_root,
                language_id=route.language_id,
            )
        )


def parse_lsp_args(args: Mapping[str, Any]) -> LspToolInput:
    """Validate model-facing arguments and convert coordinates to zero-based positions."""

    operation = args.get("operation")
    if not isinstance(operation, str) or operation not in LSP_OPERATIONS:
        choices = ", ".join(LSP_OPERATIONS)
        raise ValueError(f"operation must be one of {choices}")
    file_path = args.get("file_path")
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError("file_path must be a non-empty string")
    line = _one_based(args.get("line"), "line")
    character = _one_based(args.get("character"), "character")
    return LspToolInput(
        operation=cast(LspOperation, operation),
        file_path=file_path,
        position=LspPosition(line - 1, character - 1),
    )


def _one_based(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive integer (one-based)")
    if not math.isfinite(float(value)) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer (one-based)")
    return int(value)


def format_locations(
    locations: Sequence[LspLocation],
    workspace_uri: str,
    max_locations: int = DEFAULT_MAX_LOCATIONS,
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
) -> str:
    """Render bounded locations as workspace-relative ``path:line:character`` entries."""

    _positive_integer(max_locations, "max_locations")
    _positive_integer(max_result_chars, "max_result_chars")
    if not locations:
        return _bound_result("No results.", max_result_chars, "locations")

    shown = locations[:max_locations]
    omitted = len(locations) - len(shown)
    grouped: dict[str, list[str]] = {}
    for location in shown:
        path = render_uri(location.uri, workspace_uri)
        entry = (
            f"{path}:{location.range.start.line + 1}:{location.range.start.character + 1}"
        )
        grouped.setdefault(path, []).append(entry)
    lines = [entry for entries in grouped.values() for entry in entries]
    if omitted > 0:
        plural = "s" if omitted != 1 else ""
        lines.append(f"… {omitted} more location{plural} omitted (limit {max_locations}).")
    return _bound_result("\n".join(lines), max_result_chars, "locations")


def format_hover(
    hover: LspHover | None,
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
) -> str:
    """Render hover content with a complete-result character cap."""

    _positive_integer(max_result_chars, "max_result_chars")
    return _bound_result(
        "No hover information." if hover is None else hover.contents,
        max_result_chars,
        "hover",
    )


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _utf16_length(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def _take_utf16(value: str, units: int) -> str:
    if _utf16_length(value) <= units:
        return value
    return value.encode("utf-16-le")[: units * 2].decode("utf-16-le", errors="ignore")


def _bound_result(text: str, max_chars: int, label: str) -> str:
    if _utf16_length(text) <= max_chars:
        return text
    notice = f"\n… {label} truncated (limit {max_chars} characters)."
    if _utf16_length(notice) >= max_chars:
        return _take_utf16(notice, max_chars)
    return _take_utf16(text, max_chars - _utf16_length(notice)) + notice


def render_uri(uri: str, workspace_uri: str) -> str:
    """Render a file URI relative to the provider's canonical workspace URI."""

    if not uri.startswith("file:"):
        return uri
    try:
        target = urlsplit(uri)
        workspace = urlsplit(workspace_uri)
    except ValueError:
        return uri
    if target.scheme != "file" or workspace.scheme != "file":
        return uri

    windows_world = bool(workspace.netloc) or _WINDOWS_DRIVE_PATH.match(workspace.path) is not None
    target_windows_world = windows_world and (
        bool(target.netloc) or _WINDOWS_DRIVE_PATH.match(target.path) is not None
    )
    workspace_path = _file_uri_path(workspace, windows_world)
    target_path = _file_uri_path(target, target_windows_world)
    if workspace_path is None or target_path is None:
        return uri
    if windows_world != target_windows_world:
        return target_path

    path_module = ntpath if windows_world else posixpath
    try:
        relative = path_module.relpath(target_path, workspace_path)
    except (ValueError, TypeError):
        return uri
    outside = relative == ".." or relative.startswith(f"..{path_module.sep}")
    rendered = "." if relative == "." else target_path if outside else relative
    return rendered.replace("\\", "/") if windows_world else rendered


def _file_uri_path(parts: Any, windows: bool) -> str | None:
    # Node's fileURLToPath rejects encoded separators.  Keep malformed locations verbatim rather
    # than turning a provider response into an unsafe or misleading display path.
    if re.search(r"%(?:2f|2F|5c|5C)", parts.path):
        return None
    try:
        path = unquote(parts.path)
    except Exception:
        return None
    if "\x00" in path:
        return None
    if windows:
        if parts.netloc:
            return f"//{parts.netloc}{path}"
        return path[1:] if re.match(r"^/[a-zA-Z]:", path) else path
    if parts.netloc and parts.netloc.lower() != "localhost":
        return f"//{parts.netloc}{path}"
    return path


def install_lsp_tools(
    registry: ToolRegistry,
    runtime: LspRuntime,
    *,
    max_locations: int = DEFAULT_MAX_LOCATIONS,
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
    timeout_seconds: float = DEFAULT_LSP_TOOL_TIMEOUT_SECONDS,
) -> list[Callable[[], None]]:
    """Install the one bounded, read-only model-facing LSP tool."""

    _positive_integer(max_locations, "max_locations")
    _positive_integer(max_result_chars, "max_result_chars")
    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise ValueError("timeout_seconds must be finite and positive")

    async def execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
        parsed = parse_lsp_args(args)
        if not context.cwd.strip():
            raise LspError(
                "the lsp tool requires a session workspace cwd",
                code="LSP_WORKSPACE_REQUIRED",
            )
        result = await runtime.query(
            LspQueryRequest(
                operation=parsed.operation,
                file_path=parsed.file_path,
                position=parsed.position,
                workspace_root=context.cwd,
            )
        )
        if result.kind == "locations":
            return ToolResult(
                format_locations(
                    result.locations,
                    result.resolved_workspace_uri,
                    max_locations,
                    max_result_chars,
                )
            )
        return ToolResult(format_hover(result.hover, max_result_chars))

    definition = ToolDefinition(
        name="lsp",
        description=(
            "Query a language server for precise code navigation. operation is one of "
            "goToDefinition, findReferences, goToImplementation, hover. line and character "
            "are one-based UTF-16 cursor coordinates. findReferences includes the declaration."
        ),
        parameters={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": list(LSP_OPERATIONS),
                    "description": ", ".join(LSP_OPERATIONS),
                },
                "file_path": {
                    "type": "string",
                    "description": "The source file, relative to the workspace or absolute.",
                },
                "line": {
                    "type": "integer",
                    "description": "One-based line of the cursor.",
                },
                "character": {
                    "type": "integer",
                    "description": "One-based UTF-16 character of the cursor.",
                },
            },
            "required": ["operation", "file_path", "line", "character"],
        },
        execute=execute,
        timeout_seconds=timeout_seconds,
    )
    return [registry.register(definition)]


__all__ = [
    "DEFAULT_LSP_TOOL_TIMEOUT_SECONDS",
    "DEFAULT_MAX_LOCATIONS",
    "DEFAULT_MAX_RESULT_CHARS",
    "LSP_OPERATIONS",
    "LSP_PROMPT_TEXT",
    "LspError",
    "LspHover",
    "LspHoverResult",
    "LspLocation",
    "LspLocationsResult",
    "LspOperation",
    "LspPosition",
    "LspProvider",
    "LspProviderQuery",
    "LspQueryRequest",
    "LspQueryResult",
    "LspRange",
    "LspRuntime",
    "LspToolInput",
    "final_extension",
    "format_hover",
    "format_locations",
    "install_lsp_tools",
    "normalize_extension",
    "parse_lsp_args",
    "render_uri",
]
