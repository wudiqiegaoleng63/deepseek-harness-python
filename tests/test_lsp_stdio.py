from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from deepseek_harness.errors import LspError
from deepseek_harness.lsp import (
    LspHoverResult,
    LspLocationsResult,
    LspPosition,
    LspProviderQuery,
    LspQueryRequest,
    LspRuntime,
)
from deepseek_harness.lsp_stdio import (
    LspStdioProvider,
    LspStdioServerConfig,
    normalize_hover,
    normalize_locations,
)

FAKE_SERVER = r'''
import json
import sys


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line == b"\r\n":
            break
        key, value = line.decode("ascii").split(":", 1)
        headers[key.lower()] = value.strip()
    length = int(headers["content-length"])
    return json.loads(sys.stdin.buffer.read(length).decode("utf-8"))


def send(message):
    body = json.dumps(message, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
    sys.stdout.buffer.flush()


while True:
    message = read_message()
    if message is None:
        break
    method = message.get("method")
    request_id = message.get("id")
    if method == "initialize":
        send({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "capabilities": {
                    "definitionProvider": True,
                    "referencesProvider": True,
                    "implementationProvider": True,
                    "hoverProvider": True,
                    "textDocumentSync": {"openClose": True},
                }
            },
        })
    elif method == "shutdown":
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
    elif method == "textDocument/definition":
        uri = message["params"]["textDocument"]["uri"]
        send({"jsonrpc": "2.0", "id": request_id, "result": {
            "uri": uri,
            "range": {"start": {"line": 4, "character": 2}, "end": {"line": 4, "character": 7}},
        }})
    elif method == "textDocument/references":
        uri = message["params"]["textDocument"]["uri"]
        assert message["params"]["context"]["includeDeclaration"] is True
        send({"jsonrpc": "2.0", "id": request_id, "result": [{
            "targetUri": uri,
            "targetSelectionRange": {
                "start": {"line": 1, "character": 0},
                "end": {"line": 1, "character": 4},
            },
        }]})
    elif method == "textDocument/implementation":
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
    elif method == "textDocument/hover":
        send({"jsonrpc": "2.0", "id": request_id, "result": {
            "contents": {"kind": "markdown", "value": "**value**"},
            "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 5}},
        }})
    elif method == "exit":
        break
'''


def test_stdio_provider_runs_four_read_only_queries_and_reuses_workspace_process(
    tmp_path: Path,
) -> None:
    source = tmp_path / "main.py"
    source.write_text("value = 1\n", encoding="utf-8")
    script = tmp_path / "fake_lsp.py"
    script.write_text(FAKE_SERVER, encoding="utf-8")
    provider = LspStdioProvider(
        "python",
        LspStdioServerConfig(
            command=sys.executable,
            args=(str(script),),
            extension_to_language={".py": "python"},
            shutdown_timeout_seconds=1,
            kill_grace_seconds=0.5,
        ),
    )
    runtime = LspRuntime()
    runtime.register_provider(provider)

    async def scenario() -> None:
        definition = await runtime.query(
            LspQueryRequest("goToDefinition", "main.py", LspPosition(0, 0), str(tmp_path))
        )
        assert isinstance(definition, LspLocationsResult)
        assert definition.locations[0].range.start == LspPosition(4, 2)

        references = await runtime.query(
            LspQueryRequest("findReferences", "main.py", LspPosition(0, 0), str(tmp_path))
        )
        assert isinstance(references, LspLocationsResult)
        assert len(references.locations) == 1

        implementation = await runtime.query(
            LspQueryRequest("goToImplementation", "main.py", LspPosition(0, 0), str(tmp_path))
        )
        assert isinstance(implementation, LspLocationsResult)
        assert implementation.locations == ()

        hover = await runtime.query(
            LspQueryRequest("hover", "main.py", LspPosition(0, 0), str(tmp_path))
        )
        assert isinstance(hover, LspHoverResult)
        assert hover.hover is not None
        assert hover.hover.contents == "**value**"
        await provider.dispose_all()

    asyncio.run(scenario())


def test_stdio_provider_rejects_sources_outside_workspace(tmp_path: Path) -> None:
    provider = LspStdioProvider(
        "python",
        LspStdioServerConfig(command=sys.executable, extension_to_language={".py": "python"}),
    )

    async def scenario() -> None:
        with pytest.raises(LspError) as error:
            await provider.query(
                LspProviderQuery(
                    "hover",
                    "../outside.py",
                    LspPosition(0, 0),
                    str(tmp_path),
                    "python",
                )
            )
        assert error.value.code == "LSP_SOURCE_OUTSIDE_WORKSPACE"

    asyncio.run(scenario())


def test_stdio_normalizers_handle_location_links_hover_arrays_and_malformed_values() -> None:
    location = normalize_locations(
        {
            "targetUri": "file:///workspace/main.py",
            "targetSelectionRange": {
                "start": {"line": 1, "character": 2},
                "end": {"line": 1, "character": 5},
            },
        }
    )
    assert location[0].uri.endswith("main.py")
    assert normalize_hover(
        {"contents": [{"language": "python", "value": "x = 1"}, "plain"]}
    ) == normalize_hover(
        {"contents": "```python\nx = 1\n```\n\nplain"}
    )
    with pytest.raises(LspError) as error:
        normalize_locations([{"uri": "file:///bad", "range": {}}])
    assert error.value.code == "LSP_MALFORMED_RESPONSE"
