from fastapi.testclient import TestClient

from core.errors import AppError, ErrorCode
from prompts.judge import JUDGE_SYSTEM_PROMPT
from tests.fakes import FakeLlm

CHUNK = {"id": "c1", "document_title": "Refunds.pdf", "page_number": 4, "content": "Returns within 30 days."}
BODY = {
    "question": "What is the refund window?",
    "expected": "30 days",
    "generated": "You can return items within 30 days [1].",
    "chunks": [CHUNK],
}


def test_judge_returns_scores_with_usage(client: TestClient, llm: FakeLlm) -> None:
    llm.texts = ['{"correctness": 1, "faithfulness": 0.9, "reasoning": "Matches the source."}']

    res = client.post("/evals/judge", json=BODY)

    assert res.status_code == 200
    data = res.json()
    assert (data["correctness"], data["faithfulness"], data["reasoning"]) == (1.0, 0.9, "Matches the source.")
    assert data["usage"] == {"model": "test-model", "input_tokens": 11, "output_tokens": 5, "latency_ms": 7}
    call = llm.calls[0]
    assert call["temperature"] == 0 and call["messages"][0].content == JUDGE_SYSTEM_PROMPT
    user = call["messages"][1].content
    assert "Expected answer: 30 days" in user and "[1] (Refunds.pdf, page 4)" in user


def test_judge_accepts_a_fenced_reply_and_clamps_scores(client: TestClient, llm: FakeLlm) -> None:
    llm.texts = ['```json\n{"correctness": 1.4, "faithfulness": "-0.2", "reasoning": "ok"}\n```']

    data = client.post("/evals/judge", json=BODY).json()

    assert (data["correctness"], data["faithfulness"]) == (1.0, 0.0)


def test_unreadable_reply_is_an_llm_error(client: TestClient, llm: FakeLlm) -> None:
    llm.texts = ["I think it is mostly right."]

    res = client.post("/evals/judge", json=BODY)

    assert res.status_code == 502 and res.json()["error"]["code"] == "LLM_ERROR"


def test_llm_failure_is_passed_on(client: TestClient, llm: FakeLlm) -> None:
    llm.error = AppError(ErrorCode.LLM_ERROR, "down")

    assert client.post("/evals/judge", json=BODY).status_code == 502


def test_empty_body_is_422_which_the_api_uses_as_a_liveness_probe(client: TestClient) -> None:
    assert client.post("/evals/judge", json={}).status_code == 422


def test_requires_the_internal_key(client: TestClient) -> None:
    res = client.post("/evals/judge", json=BODY, headers={"X-Internal-Key": "wrong"})

    assert res.status_code == 401
