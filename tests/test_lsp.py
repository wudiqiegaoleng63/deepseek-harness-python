from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from deepseek_harness.errors import LspError
from deepseek_harness.lsp import (
    LspHover,
    LspHoverResult,
    LspLocation,
    LspLocationsResult,
    LspPosition,
    LspProviderQuery,
    LspQueryRequest,
    LspRange,
    LspRuntime,
    final_extension,
    format_hover,
    format_locations,
    install_lsp_tools,
    parse_lsp_args,
    render_uri,
)
from deepseek_harness.tools.registry import ToolContext, ToolRegistry


class RecordingProvider:
    def __init__(
        self,
        provider_id: str,
        extensions: dict[str, str],
        result_factory: Callable[[LspProviderQuery], LspLocationsResult | LspHoverResult],
    ) -> None:
        self.id = provider_id
        self.extension_to_language: Mapping[str, str] = extensions
        self.result_factory = result_factory
        self.requests: list[LspProviderQuery] = []

    async def query(self, request: LspProviderQuery) -> LspLocationsResult | LspHoverResult:
        self.requests.append(request)
        return self.result_factory(request)


def test_final_extension_is_case_insensitive_and_separator_stable() -> None:
    assert final_extension("src/Foo.d.TS") == ".ts"
    assert final_extension(r"src\Foo.PY") == ".py"
    assert final_extension(".bashrc") == ""
    assert final_extension("src/Makefile") == ""


def test_lsp_registration_is_atomic_and_routes_independently_of_order(tmp_path: Path) -> None:
    runtime = LspRuntime()
    workspace_uri = tmp_path.resolve().as_uri()
    python_provider = RecordingProvider(
        "python",
        {"PY": "python"},
        lambda _request: LspLocationsResult("locations", (), workspace_uri),
    )
    typescript_provider = RecordingProvider(
        "typescript",
        {".TS": "typescript"},
        lambda _request: LspLocationsResult("locations", (), workspace_uri),
    )

    dispose_python = runtime.register_provider(python_provider)
    dispose_typescript = runtime.register_provider(typescript_provider)

    async def scenario() -> None:
        await runtime.query(
            LspQueryRequest(
                "goToDefinition",
                "src/App.TS",
                LspPosition(2, 4),
                str(tmp_path),
            )
        )
        assert typescript_provider.requests[0].language_id == "typescript"
        assert typescript_provider.requests[0].position == LspPosition(2, 4)

    asyncio.run(scenario())

    conflicting = RecordingProvider(
        "conflicting",
        {".py": "python", ".rs": "rust"},
        lambda _request: LspLocationsResult("locations", (), workspace_uri),
    )
    with pytest.raises(LspError) as error:
        runtime.register_provider(conflicting)
    assert error.value.code == "LSP_CONFLICT"

    # The failed registration did not reserve the non-conflicting .rs route.
    rust = RecordingProvider(
        "rust",
        {".rs": "rust"},
        lambda _request: LspLocationsResult("locations", (), workspace_uri),
    )
    dispose_rust = runtime.register_provider(rust)
    dispose_typescript()
    dispose_typescript()
    dispose_rust()
    dispose_python()

    async def unavailable() -> None:
        with pytest.raises(LspError) as error:
            await runtime.query(
                LspQueryRequest("hover", "src/App.TS", LspPosition(0, 0), str(tmp_path))
            )
        assert error.value.code == "LSP_UNAVAILABLE"

    asyncio.run(unavailable())


def test_lsp_tool_converts_coordinates_and_formats_locations(tmp_path: Path) -> None:
    source = tmp_path / "src" / "app.py"
    source.parent.mkdir()
    source.write_text("value\n", encoding="utf-8")
    workspace_uri = tmp_path.resolve().as_uri()
    provider = RecordingProvider(
        "python",
        {".py": "python"},
        lambda _request: LspLocationsResult(
            "locations",
            (
                LspLocation(
                    source.as_uri(),
                    LspRange(LspPosition(3, 2), LspPosition(3, 7)),
                ),
            ),
            workspace_uri,
        ),
    )
    runtime = LspRuntime()
    runtime.register_provider(provider)
    registry = ToolRegistry()
    disposers = install_lsp_tools(registry, runtime)

    async def scenario() -> None:
        result = await registry.execute(
            "lsp",
            json.dumps(
                {
                    "operation": "goToDefinition",
                    "file_path": "src/app.py",
                    "line": 2,
                    "character": 3,
                }
            ),
            ToolContext("session-lsp", str(tmp_path)),
        )
        assert not result.is_error
        assert result.text == "src/app.py:4:3"
        assert provider.requests[0].position == LspPosition(1, 2)
        assert provider.requests[0].workspace_root == str(tmp_path)

    try:
        asyncio.run(scenario())
    finally:
        for dispose in reversed(disposers):
            dispose()


def test_lsp_tool_returns_structured_unavailable_and_workspace_errors() -> None:
    async def scenario() -> None:
        registry = ToolRegistry()
        disposers = install_lsp_tools(registry, LspRuntime())
        try:
            unavailable = await registry.execute(
                "lsp",
                '{"operation":"hover","file_path":"main.py","line":1,"character":1}',
                ToolContext("session-lsp", "/tmp"),
            )
            assert unavailable.is_error
            assert unavailable.meta == {"code": "LSP_UNAVAILABLE"}

            missing_workspace = await registry.execute(
                "lsp",
                '{"operation":"hover","file_path":"main.py","line":1,"character":1}',
                ToolContext("session-lsp", ""),
            )
            assert missing_workspace.is_error
            assert missing_workspace.meta == {"code": "LSP_WORKSPACE_REQUIRED"}
        finally:
            for dispose in reversed(disposers):
                dispose()

    asyncio.run(scenario())


def test_lsp_hover_and_location_rendering_are_bounded(tmp_path: Path) -> None:
    workspace_uri = tmp_path.resolve().as_uri()
    inside = (tmp_path / "inside.py").as_uri()
    outside = (tmp_path.parent / "outside.py").as_uri()
    locations = [
        LspLocation(inside, LspRange(LspPosition(0, 0), LspPosition(0, 1))),
        LspLocation(outside, LspRange(LspPosition(1, 2), LspPosition(1, 3))),
    ]

    rendered = format_locations(locations, workspace_uri, max_locations=1)
    assert rendered == "inside.py:1:1\n… 1 more location omitted (limit 1)."
    assert render_uri(inside, workspace_uri) == "inside.py"
    assert render_uri(outside, workspace_uri).endswith("outside.py")
    assert render_uri("untitled:buffer", workspace_uri) == "untitled:buffer"

    bounded = format_hover(LspHover("😀" * 100), max_result_chars=32)
    assert len(bounded.encode("utf-16-le")) // 2 <= 32
    assert "hover truncated" in bounded


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"operation": "symbols", "file_path": "a.py", "line": 1, "character": 1}, "operation"),
        ({"operation": "hover", "file_path": "a.py", "line": 0, "character": 1}, "line"),
        ({"operation": "hover", "file_path": "a.py", "line": 1, "character": 0}, "character"),
        ({"operation": "hover", "file_path": "", "line": 1, "character": 1}, "file_path"),
    ],
)
def test_parse_lsp_args_rejects_invalid_model_coordinates(
    args: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_lsp_args(args)


def test_harness_service_exposes_lsp_without_requiring_a_provider(tmp_path: Path) -> None:
    async def scenario() -> None:
        from deepseek_harness.web import HarnessService

        service = HarnessService(tmp_path / "state", cwd=tmp_path)
        handle = await service.create_session(session_id="lsp-session", cwd=str(tmp_path))
        assert "lsp" in handle.agent.tools.names()
        result = await handle.agent.tools.execute(
            "lsp",
            '{"operation":"hover","file_path":"main.py","line":1,"character":1}',
            ToolContext("lsp-session", str(tmp_path)),
        )
        assert result.meta == {"code": "LSP_UNAVAILABLE"}
        await service.dispose()

    asyncio.run(scenario())
