"""Provider-neutral web access and model-facing web tools.

This is the Python counterpart of DSH's ``ctx.web`` plus ``tool-web`` seams.
Providers own retrieval; the Agent-facing tools own validation, bounded
rendering, and replayable result metadata.
"""

from __future__ import annotations

import asyncio
import codecs
import re
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Protocol, TypeVar
from urllib.parse import urljoin, urlsplit

import httpx

from .errors import WebError
from .tools.registry import ToolContext, ToolDefinition, ToolRegistry, ToolResult

DEFAULT_FETCH_MAX_OUTPUT_CHARS = 200_000
DEFAULT_FETCH_TIMEOUT_SECONDS = 30.0
DEFAULT_SEARCH_MAX_RESULTS = 8
DEFAULT_SEARCH_MAX_USES = 5
DEFAULT_SEARCH_MAX_TOKENS = 4096
DEFAULT_SEARCH_BASE_URL = "https://api.deepseek.com/anthropic/v1"
DEFAULT_SEARCH_MODEL = "deepseek-v4-flash"
DEFAULT_USER_AGENT = "deepseek-harness-python/0.1.0"
TRUNCATION_FOOTER = (
    "\n\n(Content truncated. Fetch a more specific URL or section for the full text.)"
)


@dataclass(frozen=True, slots=True)
class WebSource:
    url: str
    title: str | None = None
    snippet: str | None = None
    published_at: str | None = None

    def to_dict(self) -> dict[str, str]:
        result = {"url": self.url}
        if self.title:
            result["title"] = self.title
        if self.snippet:
            result["snippet"] = self.snippet
        if self.published_at:
            result["publishedAt"] = self.published_at
        return result


@dataclass(frozen=True, slots=True)
class WebSearchResult:
    sources: tuple[WebSource, ...]
    content: str | None = None
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class WebFetchBody:
    kind: str
    content: str


@dataclass(frozen=True, slots=True)
class WebFetchResult:
    url: str
    status_code: int
    body: WebFetchBody
    truncated: bool = False


class ProviderCommon(Protocol):
    id: str

    def available(self) -> bool: ...


class SearchProvider(ProviderCommon, Protocol):

    async def search(self, query: str, max_results: int) -> WebSearchResult: ...


class FetchProvider(ProviderCommon, Protocol):
    async def fetch(self, url: str) -> WebFetchResult: ...


ProviderT = TypeVar("ProviderT", bound=ProviderCommon)


class WebRuntime:
    """Provider registry with deterministic selection semantics."""

    def __init__(
        self,
        *,
        search_provider_id: str | None = None,
        fetch_provider_id: str | None = None,
    ) -> None:
        self.search_provider_id = search_provider_id
        self.fetch_provider_id = fetch_provider_id
        self._search: dict[str, SearchProvider] = {}
        self._fetch: dict[str, FetchProvider] = {}

    def register_search_provider(self, provider: SearchProvider) -> Callable[[], None]:
        if provider.id in self._search:
            raise WebError(
                f'a web search provider with id "{provider.id}" is already registered',
                code="WEB_DUPLICATE_PROVIDER",
            )
        self._search[provider.id] = provider
        active = True

        def dispose() -> None:
            nonlocal active
            if active:
                active = False
                if self._search.get(provider.id) is provider:
                    self._search.pop(provider.id, None)

        return dispose

    def register_fetch_provider(self, provider: FetchProvider) -> Callable[[], None]:
        if provider.id in self._fetch:
            raise WebError(
                f'a web fetch provider with id "{provider.id}" is already registered',
                code="WEB_DUPLICATE_PROVIDER",
            )
        self._fetch[provider.id] = provider
        active = True

        def dispose() -> None:
            nonlocal active
            if active:
                active = False
                if self._fetch.get(provider.id) is provider:
                    self._fetch.pop(provider.id, None)

        return dispose

    async def search(
        self, query: str, *, max_results: int = DEFAULT_SEARCH_MAX_RESULTS
    ) -> WebSearchResult:
        if not query.strip():
            raise WebError("query must be a non-empty string", code="WEB_INVALID_REQUEST")
        if max_results <= 0:
            raise WebError("max_results must be positive", code="WEB_INVALID_REQUEST")
        provider = self._resolve(self._search, self.search_provider_id, "search")
        result = await provider.search(query, max_results)
        if len(result.sources) <= max_results:
            return result
        return WebSearchResult(
            sources=result.sources[:max_results],
            content=result.content,
            truncated=True,
        )

    async def fetch(self, url: str) -> WebFetchResult:
        if not url.strip():
            raise WebError("url must be a non-empty string", code="WEB_INVALID_REQUEST")
        provider = self._resolve(self._fetch, self.fetch_provider_id, "fetch")
        return await provider.fetch(url)

    @staticmethod
    def _resolve(
        providers: dict[str, ProviderT],
        configured_id: str | None,
        kind: str,
    ) -> ProviderT:
        if configured_id is not None:
            provider = providers.get(configured_id)
            if provider is None:
                raise WebError(
                    f'configured web {kind} provider "{configured_id}" is not registered',
                    code="WEB_PROVIDER_CONFIGURED_MISSING",
                )
            if not provider.available():
                raise WebError(
                    f'configured web {kind} provider "{configured_id}" is unavailable',
                    code="WEB_PROVIDER_CONFIGURED_UNAVAILABLE",
                )
            return provider
        usable = [provider for provider in providers.values() if provider.available()]
        if not usable:
            raise WebError("no usable web provider is registered", code="WEB_PROVIDER_UNAVAILABLE")
        if len(usable) > 1:
            ids = ", ".join(provider.id for provider in usable)
            raise WebError(
                f"multiple usable web providers are registered ({ids}); configure one explicitly",
                code="WEB_PROVIDER_AMBIGUOUS",
            )
        return usable[0]


@dataclass(frozen=True, slots=True)
class HttpFetchLimits:
    max_url_length: int = 2048
    max_response_bytes: int = 5_000_000
    max_body_chars: int = 100_000
    timeout_seconds: float = DEFAULT_FETCH_TIMEOUT_SECONDS
    max_redirects: int = 5
    user_agent: str = DEFAULT_USER_AGENT

    def __post_init__(self) -> None:
        if self.max_url_length <= 0 or self.max_response_bytes <= 0 or self.max_body_chars <= 0:
            raise ValueError("web fetch size limits must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("web fetch timeout must be positive")
        if self.max_redirects < 0:
            raise ValueError("web fetch max_redirects must be non-negative")


class HttpFetchProvider:
    """Bounded anonymous HTTP(S) provider with same-origin redirects only."""

    id = "http"
    _redirect_statuses = {301, 302, 303, 307, 308}

    def __init__(
        self,
        limits: HttpFetchLimits | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.limits = limits or HttpFetchLimits()
        self._transport = transport

    def available(self) -> bool:
        return True

    async def fetch(self, url: str) -> WebFetchResult:
        current = _validate_fetch_url(url, self.limits.max_url_length)
        timeout = httpx.Timeout(self.limits.timeout_seconds)
        try:
            async with httpx.AsyncClient(
                follow_redirects=False,
                timeout=timeout,
                transport=self._transport,
                headers={
                    "user-agent": self.limits.user_agent,
                    "accept": "text/html,application/xhtml+xml,text/*;q=0.9,application/json;q=0.8",
                },
            ) as client:
                redirects = 0
                while True:
                    try:
                        async with client.stream("GET", current) as response:
                            if response.status_code in self._redirect_statuses:
                                if redirects >= self.limits.max_redirects:
                                    raise WebError(
                                        "exceeded the maximum of "
                                        f"{self.limits.max_redirects} redirects",
                                        code="WEB_REDIRECT_BLOCKED",
                                    )
                                location = response.headers.get("location")
                                if location is None:
                                    raise WebError(
                                        "redirect response "
                                        f"(HTTP {response.status_code}) without a Location header",
                                        code="WEB_PROVIDER_ERROR",
                                    )
                                target = _validate_fetch_url(
                                    urljoin(current, location), self.limits.max_url_length
                                )
                                if not _same_origin(current, target):
                                    raise WebError(
                                        "cross-origin redirect to "
                                        f"{target} is not followed automatically; "
                                        "retry against that URL directly",
                                        code="WEB_REDIRECT_BLOCKED",
                                    )
                                current = target
                                redirects += 1
                                continue
                            return await self._read_response(response, current)
                    except WebError:
                        raise
                    except httpx.TimeoutException as exc:
                        raise WebError("web fetch timed out", code="WEB_FETCH_TIMEOUT") from exc
                    except httpx.RequestError as exc:
                        raise WebError(
                            f"web fetch failed: {exc}", code="WEB_PROVIDER_ERROR"
                        ) from exc
        except asyncio.CancelledError:
            raise

    async def _read_response(self, response: httpx.Response, current: str) -> WebFetchResult:
        declared = response.headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > self.limits.max_response_bytes:
                    raise WebError(
                        f"response exceeds the maximum of {self.limits.max_response_bytes} bytes",
                        code="WEB_FETCH_TOO_LARGE",
                    )
            except ValueError:
                pass
        kind = _classify_content_type(response.headers.get("content-type"))
        if kind is None:
            raise WebError(
                f'unsupported content type "{response.headers.get("content-type") or "unknown"}"',
                code="WEB_UNSUPPORTED_CONTENT_TYPE",
            )
        chunks: list[bytes] = []
        total = 0
        truncated_by_bytes = False
        async for chunk in response.aiter_bytes():
            remaining = self.limits.max_response_bytes - total
            if len(chunk) > remaining:
                chunks.append(chunk[:remaining])
                total += remaining
                truncated_by_bytes = True
                break
            chunks.append(chunk)
            total += len(chunk)
        raw = b"".join(chunks)
        charset = _parse_charset(response.headers.get("content-type")) or "utf-8"
        try:
            codecs.lookup(charset)
        except LookupError as exc:
            raise WebError(
                f'unsupported charset "{charset}"', code="WEB_UNSUPPORTED_CONTENT_TYPE"
            ) from exc
        decoded = raw.decode(charset, errors="replace")
        truncated_by_chars = len(decoded) > self.limits.max_body_chars
        content = decoded[: self.limits.max_body_chars]
        return WebFetchResult(
            url=current,
            status_code=response.status_code,
            body=WebFetchBody(kind, content),
            truncated=truncated_by_bytes or truncated_by_chars,
        )


@dataclass(frozen=True, slots=True)
class DeepSeekSearchOptions:
    api_key: str | None
    base_url: str = DEFAULT_SEARCH_BASE_URL
    model: str = DEFAULT_SEARCH_MODEL
    api_version: str = "2023-06-01"
    max_tokens: int = DEFAULT_SEARCH_MAX_TOKENS
    max_uses: int = DEFAULT_SEARCH_MAX_USES


class DeepSeekSearchProvider:
    """DeepSeek Anthropic-compatible native web-search provider."""

    id = "deepseek-official"

    def __init__(
        self,
        options: Callable[[], DeepSeekSearchOptions],
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._options = options
        self._transport = transport

    def available(self) -> bool:
        options = self._options()
        return (
            bool(options.api_key)
            and bool(options.base_url)
            and options.max_tokens > 0
            and options.max_uses > 0
        )

    async def search(self, query: str, max_results: int) -> WebSearchResult:
        del max_results
        options = self._options()
        if not options.api_key:
            raise WebError(
                'DeepSeek search has no API key for "DEEPSEEK_API_KEY"',
                code="WEB_PROVIDER_CREDENTIAL_MISSING",
            )
        endpoint = options.base_url.rstrip("/") + "/messages"
        body = {
            "model": options.model,
            "max_tokens": options.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Perform a web search for the query: {query}"}
                    ],
                }
            ],
            "tools": [
                {"type": "web_search_20250305", "name": "web_search", "max_uses": options.max_uses}
            ],
        }
        try:
            async with httpx.AsyncClient(
                timeout=120.0,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    endpoint,
                    headers={
                        "x-api-key": options.api_key,
                        "authorization": f"Bearer {options.api_key}",
                        "anthropic-version": options.api_version,
                        "content-type": "application/json",
                        "accept": "application/json",
                        "user-agent": DEFAULT_USER_AGENT,
                    },
                    json=body,
                )
        except httpx.TimeoutException as exc:
            raise WebError("DeepSeek search timed out", code="WEB_PROVIDER_ERROR") from exc
        except httpx.RequestError as exc:
            raise WebError(
                f"DeepSeek search request failed: {exc}", code="WEB_PROVIDER_ERROR"
            ) from exc
        if response.status_code < 200 or response.status_code >= 300:
            detail = _response_error_detail(response)
            raise WebError(
                detail or f"DeepSeek API error (HTTP {response.status_code})",
                code="WEB_PROVIDER_ERROR",
                status=response.status_code,
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise WebError(
                "DeepSeek returned an unprocessable response body", code="WEB_PROVIDER_ERROR"
            ) from exc
        return _map_deepseek_response(payload)


def install_web_tools(
    registry: ToolRegistry,
    runtime: WebRuntime,
    *,
    max_results: int = DEFAULT_SEARCH_MAX_RESULTS,
    max_output_chars: int = DEFAULT_FETCH_MAX_OUTPUT_CHARS,
    search_timeout_seconds: float = 60.0,
    fetch_timeout_seconds: float = DEFAULT_FETCH_TIMEOUT_SECONDS,
) -> list[Callable[[], None]]:
    """Install the DSH ``web_search`` and ``web_fetch`` model tools."""

    if (
        max_results <= 0
        or max_output_chars <= 0
        or search_timeout_seconds <= 0
        or fetch_timeout_seconds <= 0
    ):
        raise ValueError("web tool limits must be positive")

    async def search(args: dict[str, Any], _context: ToolContext) -> ToolResult:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult("query must be a non-empty string", is_error=True)
        result = await runtime.search(query, max_results=max_results)
        meta = {
            "sources": [source.to_dict() for source in result.sources],
            "truncated": result.truncated,
        }
        if result.content:
            meta["answer"] = result.content
        return ToolResult(format_search_output(result), meta=meta)

    async def fetch(args: dict[str, Any], _context: ToolContext) -> ToolResult:
        url = args.get("url")
        if not isinstance(url, str) or not url.strip():
            return ToolResult("url must be a non-empty string", is_error=True)
        result = await runtime.fetch(url)
        text, truncated = format_fetch_output(result, max_output_chars)
        return ToolResult(
            text,
            meta={
                "url": result.url,
                "statusCode": result.status_code,
                "truncated": truncated,
            },
        )

    return [
        registry.register(
            ToolDefinition(
                name="web_search",
                description="Search the web for current information and return cited source URLs.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                execute=search,
                timeout_seconds=search_timeout_seconds,
            )
        ),
        registry.register(
            ToolDefinition(
                name="web_fetch",
                description="Fetch a specific HTTP(S) URL and return bounded text or markdown.",
                parameters={
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                    "additionalProperties": False,
                },
                execute=fetch,
                timeout_seconds=fetch_timeout_seconds,
            )
        ),
    ]


def format_search_output(result: WebSearchResult) -> str:
    parts: list[str] = []
    if result.content:
        parts.append(result.content)
    if result.sources:
        lines = []
        for source in result.sources:
            label = source.title or _hostname(source.url)
            suffix = ""
            metadata = [value for value in (source.snippet, source.published_at) if value]
            if metadata:
                suffix = " — " + " ".join(metadata)
            lines.append(f"- [{label}]({source.url}){suffix}")
        parts.append("Sources:\n" + "\n".join(lines))
    elif not result.content:
        parts.append("No results found.")
    if result.truncated:
        parts.append(
            f"(Showing the first {len(result.sources)} sources. Refine the query for more.)"
        )
    parts.append("Cite the relevant URLs above as markdown links in your answer.")
    return "\n\n".join(parts)


def format_fetch_output(result: WebFetchResult, max_output_chars: int) -> tuple[str, bool]:
    body = result.body.content
    if result.body.kind == "html":
        body = html_to_markdown(body)
    prefix = f"Fetched {result.url} (HTTP {result.status_code})\n\n{body}"
    truncated = result.truncated or len(prefix) > max_output_chars
    full = prefix + (TRUNCATION_FOOTER if truncated else "")
    if len(full) <= max_output_chars:
        return full, truncated
    if max_output_chars <= len(TRUNCATION_FOOTER):
        return full[:max_output_chars], True
    return full[: max_output_chars - len(TRUNCATION_FOOTER)] + TRUNCATION_FOOTER, True


class _HtmlMarkdownParser(HTMLParser):
    _skip_tags = {"script", "style", "noscript"}
    _block_tags = {
        "article",
        "div",
        "header",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "p",
        "section",
        "table",
        "tr",
        "ul",
        "ol",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0
        self.links: list[str | None] = []
        self.preformatted = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if self.skip_depth:
            if tag in self._skip_tags:
                self.skip_depth += 1
            return
        if tag in self._skip_tags:
            self.skip_depth = 1
            return
        if tag in self._block_tags:
            self.parts.append("\n\n")
        if tag == "br":
            self.parts.append("\n")
        if tag == "li":
            self.parts.append("\n- ")
        if tag == "pre":
            self.preformatted += 1
            self.parts.append("\n\n````\n")
        if tag == "a":
            href = next((value for key, value in attrs if key.lower() == "href"), None)
            self.links.append(href)
            self.parts.append("[")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self.skip_depth:
            if tag in self._skip_tags:
                self.skip_depth -= 1
            return
        if tag == "a" and self.links:
            href = self.links.pop()
            if href:
                self.parts.append(f"]({href})")
        if tag == "pre" and self.preformatted:
            self.preformatted -= 1
            self.parts.append("\n````\n")
        if tag in self._block_tags:
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data if self.preformatted else re.sub(r"\s+", " ", data))


def html_to_markdown(html: str) -> str:
    """Convert ordinary HTML to bounded readable markdown without script text."""

    if _html_depth_exceeded(html, 512):
        return html
    parser = _HtmlMarkdownParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        return html
    return re.sub(r"\n{3,}", "\n\n", "".join(parser.parts)).strip()


def _validate_fetch_url(value: str, max_length: int) -> str:
    if len(value) > max_length:
        raise WebError(f"URL exceeds the maximum length of {max_length}", code="WEB_INVALID_URL")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise WebError(
            "invalid URL: only absolute HTTP(S) URLs are allowed", code="WEB_INVALID_URL"
        )
    if parsed.username is not None or parsed.password is not None:
        raise WebError("credentials in URLs are not allowed", code="WEB_BLOCKED_URL")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise WebError("invalid URL port", code="WEB_INVALID_URL") from exc
    return value


def _same_origin(left: str, right: str) -> bool:
    a, b = urlsplit(left), urlsplit(right)
    return (
        a.scheme == b.scheme
        and a.hostname == b.hostname
        and (a.port or _default_port(a.scheme)) == (b.port or _default_port(b.scheme))
    )


def _default_port(scheme: str) -> int:
    return 443 if scheme == "https" else 80


def _classify_content_type(value: str | None) -> str | None:
    mime = (value or "").split(";", 1)[0].strip().lower()
    if mime in {"text/html", "application/xhtml+xml"}:
        return "html"
    if (
        mime.startswith("text/")
        or mime in {"application/json", "application/xml"}
        or mime.endswith(("+json", "+xml"))
    ):
        return "text"
    return None


def _parse_charset(value: str | None) -> str | None:
    match = re.search(r";\s*charset\s*=\s*\"?([^;\"]+)", value or "", re.IGNORECASE)
    return match.group(1).strip() if match else None


def _html_depth_exceeded(value: str, limit: int) -> bool:
    depth = 0
    for tag in re.finditer(r"<\s*(/?)\s*([A-Za-z][A-Za-z0-9-]*)\b[^>]*>", value):
        name = tag.group(2).lower()
        if name in {
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
        }:
            continue
        if tag.group(1):
            depth = max(0, depth - 1)
        elif not tag.group(0).rstrip().endswith("/>") and name not in {
            "script",
            "style",
            "noscript",
        }:
            depth += 1
            if depth > limit:
                return True
    return False


def _hostname(url: str) -> str:
    return urlsplit(url).hostname or url


def _response_error_detail(response: httpx.Response) -> str | None:
    try:
        value = response.json()
    except ValueError:
        return None
    if not isinstance(value, dict):
        return None
    error = value.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    if isinstance(error, str):
        return error
    message = value.get("message")
    return message if isinstance(message, str) else None


def _map_deepseek_response(payload: Any) -> WebSearchResult:
    if not isinstance(payload, dict) or not isinstance(payload.get("content"), list):
        raise WebError(
            "DeepSeek returned an unprocessable response body", code="WEB_PROVIDER_ERROR"
        )
    blocks = payload["content"]
    result_blocks = [
        block
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "web_search_tool_result"
    ]
    if not result_blocks:
        raise WebError(
            "DeepSeek returned no web_search_tool_result blocks; the request may not have "
            "triggered native web search",
            code="WEB_PROVIDER_ERROR",
        )
    snippets: dict[str, str] = {}
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        citations = block.get("citations")
        if not isinstance(citations, list):
            continue
        for citation in citations:
            if not isinstance(citation, dict):
                continue
            url, cited = citation.get("url"), citation.get("cited_text")
            if (
                isinstance(url, str)
                and isinstance(cited, str)
                and url
                and cited
                and url not in snippets
            ):
                snippets[url] = cited
    seen: set[str] = set()
    sources: list[WebSource] = []
    for block in result_blocks:
        items = block.get("content")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict) or item.get("type") != "web_search_result":
                continue
            url = item.get("url")
            if not isinstance(url, str) or not url or url in seen:
                continue
            seen.add(url)
            sources.append(
                WebSource(
                    url=url,
                    title=item.get("title") if isinstance(item.get("title"), str) else None,
                    snippet=snippets.get(url),
                    published_at=item.get("page_age")
                    if isinstance(item.get("page_age"), str)
                    else None,
                )
            )
    return WebSearchResult(tuple(sources))


__all__ = [
    "DeepSeekSearchOptions",
    "DeepSeekSearchProvider",
    "DEFAULT_FETCH_MAX_OUTPUT_CHARS",
    "DEFAULT_FETCH_TIMEOUT_SECONDS",
    "DEFAULT_SEARCH_BASE_URL",
    "DEFAULT_SEARCH_MAX_RESULTS",
    "DEFAULT_SEARCH_MAX_USES",
    "DEFAULT_SEARCH_MAX_TOKENS",
    "HttpFetchLimits",
    "HttpFetchProvider",
    "WebFetchBody",
    "WebFetchResult",
    "WebError",
    "WebRuntime",
    "WebSearchResult",
    "WebSource",
    "format_fetch_output",
    "format_search_output",
    "html_to_markdown",
    "install_web_tools",
]
