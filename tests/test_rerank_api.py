import pytest
from fastapi.testclient import TestClient

from core.config import get_settings
from core.errors import AppError, ErrorCode
from prompts.rerank import RERANK_SYSTEM_PROMPT
from tests.fakes import FakeLlm

PASSAGES = [
    {"id": "a", "text": "Chunks are embedded in batches of 100."},
    {"id": "b", "text": "Chunks are inserted in batches of 200."},
    {"id": "c", "text": "Rate limits: chat 20 per minute."},
    {"id": "d", "text": "The citation snippet is 240 characters."},
]
BODY = {"question": "How many chunks are inserted per batch?", "passages": PASSAGES, "top_n": 3}


def test_returns_the_models_picks_first_then_the_rest_in_original_order(
    client: TestClient, llm: FakeLlm
) -> None:
    llm.texts = ['{"ranking": [2]}']

    res = client.post("/rerank", json=BODY)

    assert res.status_code == 200
    data = res.json()
    assert data["ids"] == ["b", "a", "c"]
    assert data["usage"] == {
        "model": "test-fast-model",
        "input_tokens": 11,
        "output_tokens": 5,
        "latency_ms": 7,
    }
    call = llm.calls[0]
    assert call["messages"][0].content == RERANK_SYSTEM_PROMPT and call["temperature"] == 0
    user = call["messages"][1].content
    assert (
        "Return the best 3 passage numbers." in user and "[2] Chunks are inserted in batches of 200." in user
    )


def test_ignores_invalid_repeated_and_out_of_range_numbers(client: TestClient, llm: FakeLlm) -> None:
    llm.texts = ['```json\n{"ranking": [4, 4, "1", 9, 0, true, "x"]}\n```']

    data = client.post("/rerank", json={**BODY, "top_n": 4}).json()

    assert data["ids"] == ["d", "a", "b", "c"]


def test_top_n_is_capped_by_the_number_of_passages(client: TestClient, llm: FakeLlm) -> None:
    llm.texts = ['{"ranking": [1, 2, 3, 4]}']

    data = client.post("/rerank", json={**BODY, "top_n": 30}).json()

    assert data["ids"] == ["a", "b", "c", "d"]


def test_very_long_passages_are_cut(client: TestClient, llm: FakeLlm) -> None:
    llm.texts = ['{"ranking": [1]}']
    long = {"id": "x", "text": "word " * 1000}

    client.post("/rerank", json={"question": "q", "passages": [long], "top_n": 1})

    user = llm.calls[0]["messages"][1].content
    assert len(user) < 3000 and "…" in user


def test_an_unreadable_reply_is_an_llm_error(client: TestClient, llm: FakeLlm) -> None:
    llm.texts = ["the second one is best"]

    res = client.post("/rerank", json=BODY)

    assert res.status_code == 502 and res.json()["error"]["code"] == "LLM_ERROR"


def test_a_model_failure_is_passed_on(client: TestClient, llm: FakeLlm) -> None:
    llm.error = AppError(ErrorCode.LLM_ERROR, "down")

    assert client.post("/rerank", json=BODY).status_code == 502


def test_a_configured_rerank_model_is_the_one_used(
    client: TestClient, llm: FakeLlm, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_RERANK_MODEL", "rerank-model")
    get_settings.cache_clear()
    llm.texts = ['{"ranking": [1]}']

    client.post("/rerank", json=BODY)

    assert llm.calls[0]["model"] == "rerank-model"


def test_validates_the_request_and_needs_the_internal_key(client: TestClient) -> None:
    assert client.post("/rerank", json={"question": "q", "passages": []}).status_code == 422
    assert client.post("/rerank", json=BODY, headers={"X-Internal-Key": "wrong"}).status_code == 401
