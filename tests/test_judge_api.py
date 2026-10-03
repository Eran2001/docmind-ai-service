import pytest
from fastapi.testclient import TestClient

from core.config import get_settings
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


def test_both_saying_not_found_is_full_marks_without_a_model_call(client: TestClient, llm: FakeLlm) -> None:
    body = {
        "question": "Who is the CEO?",
        "expected": "I couldn't find that in your documents. The answer is not in them.",
        "generated": "I couldn't find that in your documents.",
        "chunks": [CHUNK],
    }

    data = client.post("/evals/judge", json=body).json()

    assert (data["correctness"], data["faithfulness"]) == (1.0, 1.0)
    assert data["usage"]["input_tokens"] == 0 and llm.calls == []


def test_not_found_when_the_answer_exists_scores_zero_without_a_model_call(
    client: TestClient, llm: FakeLlm
) -> None:
    body = {**BODY, "generated": "I couldn't find that in your documents."}

    data = client.post("/evals/judge", json=body).json()

    assert (data["correctness"], data["faithfulness"]) == (0.0, 1.0)
    assert llm.calls == []


def test_an_answer_where_none_exists_is_wrong_whatever_the_model_says(
    client: TestClient, llm: FakeLlm
) -> None:
    llm.texts = ['{"correctness": 0.9, "faithfulness": 0.2, "reasoning": "Invented."}']
    body = {
        "question": "Who is the CEO?",
        "expected": "I couldn't find that in your documents.",
        "generated": "The CEO is Jane Smith.",
        "chunks": [CHUNK],
    }

    data = client.post("/evals/judge", json=body).json()

    assert data["correctness"] == 0.0 and data["faithfulness"] == 0.2


def test_the_prompt_tells_the_judge_extra_detail_and_citation_markers_are_fine(
    client: TestClient, llm: FakeLlm
) -> None:
    llm.texts = ['{"correctness": 1, "faithfulness": 1, "reasoning": "ok"}']

    client.post("/evals/judge", json=BODY)

    system = llm.calls[0]["messages"][0].content
    assert "Extra correct detail" in system and "Citation markers" in system


def test_a_configured_judge_model_is_the_one_reported(
    client: TestClient, llm: FakeLlm, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_JUDGE_MODEL", "judge-model")
    get_settings.cache_clear()
    llm.texts = ['{"correctness": 1, "faithfulness": 1, "reasoning": "ok"}']

    data = client.post("/evals/judge", json=BODY).json()

    assert data["usage"]["model"] == "judge-model"
    assert llm.calls[0]["model"] == "judge-model"


def test_an_answer_that_says_not_found_and_then_answers_goes_to_the_model(
    client: TestClient, llm: FakeLlm
) -> None:
    llm.texts = ['{"correctness": 1, "faithfulness": 1, "reasoning": "Gives the right answer."}']
    body = {
        **BODY,
        "generated": (
            "I couldn't find that in your documents. According to the handbook, returns take 30 days [1]."
        ),
    }

    data = client.post("/evals/judge", json=body).json()

    assert data["correctness"] == 1.0
    assert len(llm.calls) == 1


def test_not_found_with_a_suggestion_still_counts_as_not_found(client: TestClient, llm: FakeLlm) -> None:
    body = {
        "question": "Who is the CEO?",
        "expected": "I couldn't find that in your documents. The answer is not in them.",
        "generated": (
            "I couldn't find that in your documents. Try asking about what they contain: **a.md**, **b.md**."
        ),
        "chunks": [CHUNK],
    }

    data = client.post("/evals/judge", json=body).json()

    assert (data["correctness"], data["faithfulness"]) == (1.0, 1.0)
    assert llm.calls == []
