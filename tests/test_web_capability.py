from __future__ import annotations

import asyncio
import json

import httpx

from deepseek_harness import (
    DeepSeekSearchOptions,
    DeepSeekSearchProvider,
    HttpFetchLimits,
    HttpFetchProvider,
    WebError,
    WebRuntime,
    format_fetch_output,
)
from deepseek_harness.tools.registry import ToolContext, ToolRegistry
from deepseek_harness.web import HarnessService
from deepseek_harness.web_capability import (
    WebFetchBody,
    WebFetchResult,
    WebSearchResult,
    WebSource,
    format_search_output,
    install_web_tools,
)


def test_http_fetch_provider_follows_same_origin_redirect_and_bounds_html() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/page"}, request=request)
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            content=(
                b"<html><head><style>hidden</style><script>secret()</script></head>"
                b"<body><h1>Hello</h1><p>Read <a href='https://example.test/docs'>docs</a>.</p></body></html>"
            ),
            request=request,
        )

    async def scenario() -> None:
        provider = HttpFetchProvider(transport=httpx.MockTransport(handler))
        result = await provider.fetch("https://example.test/start")
        assert result.url == "https://example.test/page"
        assert result.status_code == 200
        assert result.body.kind == "html"
        assert "Hello" in result.body.content
        rendered, truncated = format_fetch_output(result, 500)
        assert not truncated
        assert "Fetched https://example.test/page (HTTP 200)" in rendered
        assert "[docs](https://example.test/docs)" in rendered
        assert "secret" not in rendered

    asyncio.run(scenario())


def test_http_fetch_provider_rejects_cross_origin_redirect_and_large_declared_body() -> None:
    async def redirect_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"location": "https://other.test/page"},
            request=request,
        )

    async def large_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/plain", "content-length": "100"},
            content=b"small",
            request=request,
        )

    async def scenario() -> None:
        redirecting = HttpFetchProvider(transport=httpx.MockTransport(redirect_handler))
        try:
            await redirecting.fetch("https://example.test/start")
        except WebError as exc:
            assert exc.code == "WEB_REDIRECT_BLOCKED"
        else:
            raise AssertionError("cross-origin redirects must be blocked")

        oversized = HttpFetchProvider(
            HttpFetchLimits(max_response_bytes=10),
            transport=httpx.MockTransport(large_handler),
        )
        try:
            await oversized.fetch("https://example.test/large")
        except WebError as exc:
            assert exc.code == "WEB_FETCH_TOO_LARGE"
        else:
            raise AssertionError("declared oversized responses must be rejected")

    asyncio.run(scenario())


def test_deepseek_search_provider_maps_native_result_blocks_and_citations() -> None:
    observed: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "content": [
                    {
                        "type": "text",
                        "text": "answer",
                        "citations": [
                            {"url": "https://example.test/a", "cited_text": "snippet A"}
                        ],
                    },
                    {
                        "type": "web_search_tool_result",
                        "content": [
                            {
                                "type": "web_search_result",
                                "url": "https://example.test/a",
                                "title": "A",
                                "page_age": "2026-08-17",
                            },
                            {
                                "type": "web_search_result",
                                "url": "https://example.test/b",
                            },
                        ],
                    },
                ]
            },
            request=request,
        )

    async def scenario() -> None:
        provider = DeepSeekSearchProvider(
            lambda: DeepSeekSearchOptions(api_key="secret"),
            transport=httpx.MockTransport(handler),
        )
        result = await provider.search("python dsh", 8)
        assert [source.url for source in result.sources] == [
            "https://example.test/a",
            "https://example.test/b",
        ]
        assert result.sources[0].snippet == "snippet A"
        assert result.sources[0].published_at == "2026-08-17"
        assert observed[0]["tools"] == [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
        ]

    asyncio.run(scenario())


def test_web_runtime_selection_and_model_tools_keep_structured_provider_errors() -> None:
    class FakeFetch:
        id = "fake"

        def available(self) -> bool:
            return True

        async def fetch(self, url: str) -> WebFetchResult:
            return WebFetchResult(url, 200, WebFetchBody("text", "hello"))

    async def scenario() -> None:
        runtime = WebRuntime()
        runtime.register_fetch_provider(FakeFetch())
        assert (await runtime.fetch("https://example.test")).body.content == "hello"
        registry = ToolRegistry()
        disposers = install_web_tools(registry, runtime, max_output_chars=1000)
        try:
            context = ToolContext("session-web", ".")
            fetched = await registry.execute(
                "web_fetch", '{"url":"https://example.test"}', context
            )
            assert fetched.text.endswith("hello")
            assert fetched.meta == {
                "url": "https://example.test",
                "statusCode": 200,
                "truncated": False,
            }
            unavailable = await registry.execute("web_search", '{"query":"missing"}', context)
            assert unavailable.is_error
            assert unavailable.meta == {"code": "WEB_PROVIDER_UNAVAILABLE"}
        finally:
            for dispose in reversed(disposers):
                dispose()

    asyncio.run(scenario())


def test_web_format_search_output_is_citable() -> None:
    result = WebSearchResult(
        sources=(WebSource("https://example.test", title="Example", snippet="useful"),),
        content="summary",
        truncated=True,
    )
    rendered = format_search_output(result)
    assert "summary" in rendered
    assert "[Example](https://example.test)" in rendered
    assert "Cite the relevant URLs" in rendered


def test_harness_service_exposes_web_tools_and_search_settings(tmp_path) -> None:
    async def scenario() -> None:
        service = HarnessService(tmp_path / "state", cwd=tmp_path)
        handle = await service.create_session(session_id="web-session", cwd=str(tmp_path))
        assert {"web_search", "web_fetch"}.issubset(set(handle.agent.tools.names()))
        described = await service.dispatch("settings.describe", {})
        search = next(
            item for item in described["namespaces"] if item["ns"] == "web-search-deepseek"
        )
        assert search["value"]["model"] == "deepseek-v4-flash"
        await service.dispose()

    asyncio.run(scenario())
