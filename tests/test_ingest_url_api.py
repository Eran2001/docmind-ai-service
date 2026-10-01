import pytest
from fastapi.testclient import TestClient

from core.errors import AppError, ErrorCode
from services import ingest
from tests.fakes import FakeEmbedder
from tests.test_web import PAGE
from tools.safe_fetch import FetchedPage


@pytest.fixture
def served_page(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    requested: list[str] = []

    async def fake_fetch(url: str) -> FetchedPage:
        requested.append(url)
        return FetchedPage(url=url, content=PAGE)

    monkeypatch.setattr(ingest, "fetch_page", fake_fetch)
    return requested


def test_url_ingest_returns_title_and_embedded_chunks(
    client: TestClient, served_page: list[str], embedder: FakeEmbedder
) -> None:
    res = client.post("/ingest/url", json={"url": "https://acme.test/benefits"})

    assert res.status_code == 200
    body = res.json()
    assert served_page == ["https://acme.test/benefits"]
    assert body["title"]
    assert body["page_count"] is None
    assert body["chunks"][0]["page_number"] is None
    assert body["chunks"][0]["heading"] == "Employee Benefits"
    assert body["chunks"][0]["embedding"] == [0.5] * 4
    assert body["usage"]["model"] == "fake-embedding"
    assert len(embedder.calls[0]) == len(body["chunks"])


def test_blocked_url_is_reported_with_its_code(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def blocked(url: str) -> FetchedPage:
        raise AppError(ErrorCode.URL_BLOCKED, "That address is not allowed.")

    monkeypatch.setattr(ingest, "fetch_page", blocked)

    res = client.post("/ingest/url", json={"url": "http://169.254.169.254/"})

    assert res.status_code == 400
    assert res.json() == {"error": {"code": "URL_BLOCKED", "message": "That address is not allowed."}}


@pytest.mark.parametrize("url", ["ftp://acme.test/file", "file:///etc/passwd", "not a url", ""])
def test_only_http_urls_are_accepted(client: TestClient, url: str) -> None:
    res = client.post("/ingest/url", json={"url": url})

    assert res.status_code == 422
    assert res.json()["error"]["code"] == "INVALID_REQUEST"


def test_url_ingest_requires_the_internal_key(client: TestClient) -> None:
    res = client.post("/ingest/url", json={"url": "https://acme.test/"}, headers={"X-Internal-Key": "wrong"})

    assert res.status_code == 401
