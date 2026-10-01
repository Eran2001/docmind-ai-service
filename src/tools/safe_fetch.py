import asyncio

import httpx
from pydantic import BaseModel

from core.errors import AppError, ErrorCode
from tools.ssrf import resolve_public

TIMEOUT_SECONDS = 15.0
MAX_BYTES = 5 * 1024 * 1024
MAX_REDIRECTS = 5
_HTML_TYPES = {"", "text/html", "application/xhtml+xml"}
_HEADERS = {
    "User-Agent": "DocMindBot/1.0 (+document indexing)",
    "Accept": "text/html,application/xhtml+xml;q=0.9",
}


class FetchedPage(BaseModel):
    url: str
    content: bytes


async def fetch_page(url: str, *, transport: httpx.AsyncBaseTransport | None = None) -> FetchedPage:
    """GETs a web page, refusing non-public addresses on the first request and on every redirect.

    Each hop resolves DNS once, checks every address, then connects to that exact IP (Host header and TLS
    server name stay the original hostname), so the check and the connection can't disagree.
    """
    try:
        async with (
            asyncio.timeout(TIMEOUT_SECONDS),
            httpx.AsyncClient(
                transport=transport,
                headers=_HEADERS,
                follow_redirects=False,
                trust_env=False,  # proxy variables must not route around the address check
            ) as client,
        ):
            current = url
            for _ in range(MAX_REDIRECTS + 1):
                redirect, content = await _fetch_once(client, current)
                if redirect is None:
                    return FetchedPage(url=current, content=content)
                current = redirect
    except TimeoutError:
        raise AppError(ErrorCode.URL_FETCH_FAILED, "The page took too long to respond.") from None
    except httpx.HTTPError:
        raise AppError(ErrorCode.URL_FETCH_FAILED, "The page could not be loaded.") from None
    raise AppError(ErrorCode.URL_FETCH_FAILED, "The page redirected too many times.")


async def _fetch_once(client: httpx.AsyncClient, url: str) -> tuple[str | None, bytes]:
    """One hop: returns (redirect target, b"") or (None, body)."""
    target = httpx.URL(url)
    if target.scheme not in ("http", "https") or not target.host:
        raise AppError(ErrorCode.URL_BLOCKED, "Only http and https links are allowed.")
    port = target.port or (443 if target.scheme == "https" else 80)
    host_header = f"[{target.host}]" if ":" in target.host else target.host
    if target.port:
        host_header += f":{target.port}"

    last_error: httpx.HTTPError | None = None
    for address in await resolve_public(target.host, port):
        request = client.build_request(
            "GET",
            target.copy_with(host=address, userinfo=b""),
            headers={"Host": host_header},
            extensions={"sni_hostname": target.host},
        )
        try:
            response = await client.send(request, stream=True)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            last_error = exc  # try the next address
            continue
        try:
            return await _read(response, target)
        finally:
            await response.aclose()
    assert last_error is not None
    raise last_error


async def _read(response: httpx.Response, target: httpx.URL) -> tuple[str | None, bytes]:
    if response.is_redirect:
        location = response.headers.get("location")
        if not location:
            raise AppError(ErrorCode.URL_FETCH_FAILED, "The page redirected without a destination.")
        return str(target.join(location)), b""
    if response.status_code >= 400:
        raise AppError(
            ErrorCode.URL_FETCH_FAILED, f"The website answered with an error ({response.status_code})."
        )

    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type not in _HTML_TYPES:
        raise AppError(ErrorCode.PARSE_FAILED, "That link is not a web page.")
    declared = response.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_BYTES:
        raise _too_large()

    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk)
        if len(body) > MAX_BYTES:
            raise _too_large()
    return None, bytes(body)


def _too_large() -> AppError:
    return AppError(ErrorCode.URL_FETCH_FAILED, "The page is larger than 5 MB.")
