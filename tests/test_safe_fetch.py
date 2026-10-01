import asyncio
import ipaddress
from collections.abc import Callable, Coroutine
from typing import Any

import httpx
import pytest

from core.errors import AppError, ErrorCode
from tools import safe_fetch, ssrf
from tools.safe_fetch import MAX_BYTES, MAX_REDIRECTS, fetch_page
from tools.ssrf import IpAddress

PUBLIC_IP = "93.184.216.34"
HTML = {"content-type": "text/html; charset=utf-8"}

Handler = (
    Callable[[httpx.Request], httpx.Response] | Callable[[httpx.Request], Coroutine[Any, Any, httpx.Response]]
)


@pytest.fixture(autouse=True)
def dns(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """Hostname -> addresses. Anything not listed resolves to a public address."""
    table: dict[str, list[str]] = {"intranet.example.com": ["10.1.2.3"]}

    async def lookup(host: str, port: int) -> list[IpAddress]:
        return [ipaddress.ip_address(a) for a in table.get(host, [PUBLIC_IP])]

    monkeypatch.setattr(ssrf, "_lookup", lookup)
    return table


def transport(handler: Handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


async def code_of(url: str, handler: Handler) -> ErrorCode:
    with pytest.raises(AppError) as exc:
        await fetch_page(url, transport=transport(handler))
    return exc.value.code


async def test_connects_to_the_resolved_ip_and_keeps_the_hostname() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, headers=HTML, content=b"<html>hello</html>")

    page = await fetch_page("https://user:pw@example.com:8443/a/b?q=1", transport=transport(handler))

    request = seen[0]
    assert page.content == b"<html>hello</html>"
    assert page.url == "https://user:pw@example.com:8443/a/b?q=1"
    assert request.url.host == PUBLIC_IP
    assert request.url.port == 8443
    assert request.url.raw_path == b"/a/b?q=1"
    assert request.url.userinfo == b""
    assert request.headers["host"] == "example.com:8443"
    assert request.extensions["sni_hostname"] == "example.com"


async def test_ipv6_literal_urls_work() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, headers=HTML, content=b"ok")

    await fetch_page("http://[2606:4700:4700::1111]/", transport=transport(handler))

    assert seen[0].url.host == "2606:4700:4700::1111"
    assert seen[0].headers["host"] == "[2606:4700:4700::1111]"


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost.internal/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "http://10.0.0.1:8080/admin",
        "http://intranet.example.com/",
        "ftp://example.com/file",
        "file:///etc/passwd",
    ],
)
async def test_blocked_targets_never_receive_a_request(url: str, dns: dict[str, list[str]]) -> None:
    dns["localhost.internal"] = ["127.0.0.1"]
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, headers=HTML, content=b"secret")

    assert await code_of(url, handler) is ErrorCode.URL_BLOCKED
    assert calls == []


async def test_redirect_to_a_private_ip_is_blocked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})

    assert await code_of("https://example.com/", handler) is ErrorCode.URL_BLOCKED


async def test_redirect_to_a_hostname_that_resolves_privately_is_blocked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(301, headers={"location": "https://intranet.example.com/"})

    assert await code_of("https://example.com/", handler) is ErrorCode.URL_BLOCKED


async def test_relative_redirects_are_followed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(302, headers={"location": "/new"})
        return httpx.Response(200, headers=HTML, content=b"final")

    page = await fetch_page("https://example.com/old", transport=transport(handler))

    assert page.content == b"final"
    assert page.url == "https://example.com/new"


async def test_redirect_loops_stop() -> None:
    hops = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal hops
        hops += 1
        return httpx.Response(302, headers={"location": "/again"})

    assert await code_of("https://example.com/", handler) is ErrorCode.URL_FETCH_FAILED
    assert hops == MAX_REDIRECTS + 1


async def test_http_errors_are_fetch_failures() -> None:
    assert await code_of("https://example.com/", lambda r: httpx.Response(404)) is ErrorCode.URL_FETCH_FAILED
    assert await code_of("https://example.com/", lambda r: httpx.Response(503)) is ErrorCode.URL_FETCH_FAILED


async def test_non_html_content_is_not_a_web_page() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF")

    assert await code_of("https://example.com/a.pdf", handler) is ErrorCode.PARSE_FAILED


async def test_oversized_pages_are_refused_while_streaming() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=HTML, content=b"x" * (MAX_BYTES + 1))

    assert await code_of("https://example.com/", handler) is ErrorCode.URL_FETCH_FAILED


async def test_oversized_pages_are_refused_from_content_length() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={**HTML, "content-length": str(MAX_BYTES + 1)}, content=b"x")

    assert await code_of("https://example.com/", handler) is ErrorCode.URL_FETCH_FAILED


async def test_page_just_under_the_limit_is_accepted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=HTML, content=b"x" * MAX_BYTES)

    assert len((await fetch_page("https://example.com/", transport=transport(handler))).content) == MAX_BYTES


async def test_connection_errors_are_fetch_failures() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    assert await code_of("https://example.com/", handler) is ErrorCode.URL_FETCH_FAILED


async def test_next_address_is_tried_after_a_connect_error(dns: dict[str, list[str]]) -> None:
    dns["two.example.com"] = ["93.184.216.34", "93.184.216.35"]
    tried: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        tried.append(request.url.host)
        if request.url.host == "93.184.216.34":
            raise httpx.ConnectError("refused")
        return httpx.Response(200, headers=HTML, content=b"ok")

    await fetch_page("https://two.example.com/", transport=transport(handler))

    assert tried == ["93.184.216.34", "93.184.216.35"]


async def test_overall_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(safe_fetch, "TIMEOUT_SECONDS", 0.05)

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1)
        return httpx.Response(200, headers=HTML, content=b"late")

    assert await code_of("https://example.com/", handler) is ErrorCode.URL_FETCH_FAILED
