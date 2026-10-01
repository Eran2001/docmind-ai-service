from typing import Any

import httpx2
from fastapi.testclient import TestClient

from core.errors import AppError, ErrorCode
from tests.builders import make_docx, make_pdf
from tests.fakes import FakeEmbedder

SENTENCE = "The refund policy allows returns within thirty days of purchase for any reason at all."
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def upload(
    client: TestClient, data: bytes, mime_type: str, *, headers: dict[str, str] | None = None
) -> httpx2.Response:
    return client.post(
        "/ingest/file",
        files={"file": ("doc", data, mime_type)},
        data={"mime_type": mime_type},
        headers=headers,
    )


def test_text_file_returns_embedded_chunks(client: TestClient, embedder: FakeEmbedder) -> None:
    res = upload(client, " ".join([SENTENCE] * 6).encode(), "text/plain")

    assert res.status_code == 200
    body = res.json()
    assert body["page_count"] is None
    assert body["usage"]["model"] == "fake-embedding"
    chunk = body["chunks"][0]
    assert set(chunk) == {"index", "content", "page_number", "heading", "token_count", "embedding"}
    assert chunk["index"] == 0 and chunk["page_number"] is None
    assert chunk["embedding"] == [0.5] * 4
    assert len(embedder.calls[0]) == len(body["chunks"])


def test_pdf_reports_page_count_and_start_pages(client: TestClient) -> None:
    pdf = make_pdf([[(SENTENCE, 11)], [(SENTENCE.replace("refund", "exchange"), 11)]])

    body = upload(client, pdf, "application/pdf").json()

    assert body["page_count"] == 2
    assert {c["page_number"] for c in body["chunks"]} <= {1, 2}


def test_docx_is_supported(client: TestClient) -> None:
    res = upload(client, make_docx(), DOCX_MIME)

    assert res.status_code == 200
    assert res.json()["chunks"][0]["heading"] == "Leave policy"


def test_mime_type_parameters_and_case_are_ignored(client: TestClient) -> None:
    assert upload(client, SENTENCE.encode() * 2, "Text/Plain; charset=utf-8").status_code == 200


def test_unsupported_type_is_parse_failed(client: TestClient) -> None:
    res = upload(client, b"x", "image/png")

    assert res.status_code == 422
    assert res.json()["error"]["code"] == "PARSE_FAILED"


def test_empty_file_is_empty_document(client: TestClient) -> None:
    res = upload(client, b"", "text/plain")

    assert res.status_code == 422
    assert res.json()["error"]["code"] == "EMPTY_DOCUMENT"


def test_too_little_text_is_empty_document(client: TestClient, embedder: FakeEmbedder) -> None:
    res = upload(client, b"Just a few words.", "text/plain")

    assert res.json()["error"]["code"] == "EMPTY_DOCUMENT"
    assert embedder.calls == []


def test_embedding_failure_is_llm_error(client: TestClient, embedder: FakeEmbedder) -> None:
    embedder.error = AppError(ErrorCode.LLM_ERROR, "The embedding service is unavailable.")

    res = upload(client, SENTENCE.encode() * 2, "text/plain")

    assert res.status_code == 502
    assert res.json()["error"]["code"] == "LLM_ERROR"


def test_ingest_requires_the_internal_key(client: TestClient) -> None:
    res = upload(client, SENTENCE.encode(), "text/plain", headers={"X-Internal-Key": "wrong"})

    assert res.status_code == 401


def test_missing_fields_are_invalid_request(client: TestClient) -> None:
    res = client.post("/ingest/file", data={"mime_type": "text/plain"})

    assert res.status_code == 422
    assert res.json()["error"]["code"] == "INVALID_REQUEST"
    assert "file" in res.json()["error"]["message"]


def test_embed_returns_vectors_and_usage(client: TestClient, embedder: FakeEmbedder) -> None:
    res = client.post("/embed", json={"texts": ["  what is the refund policy?  ", "second"]})

    assert res.status_code == 200
    assert res.json()["embeddings"] == [[0.5] * 4, [0.5] * 4]
    assert res.json()["usage"]["input_tokens"] == 6
    assert embedder.calls == [["what is the refund policy?", "second"]]


def test_embed_accepts_up_to_100_texts(client: TestClient) -> None:
    assert client.post("/embed", json={"texts": ["a"] * 100}).status_code == 200
    assert client.post("/embed", json={"texts": ["a"] * 101}).status_code == 422


def test_embed_rejects_empty_input(client: TestClient) -> None:
    bodies: list[dict[str, Any]] = [{"texts": []}, {"texts": [""]}, {"texts": ["   "]}, {}]
    for body in bodies:
        res = client.post("/embed", json=body)
        assert res.status_code == 422
        assert res.json()["error"]["code"] == "INVALID_REQUEST"


def test_embed_requires_the_internal_key(client: TestClient) -> None:
    res = client.post("/embed", json={"texts": ["a"]}, headers={"X-Internal-Key": "wrong"})

    assert res.status_code == 401
