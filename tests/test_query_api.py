from fastapi.testclient import TestClient

from core.errors import AppError, ErrorCode
from prompts.rewrite import REWRITE_QUERY_PROMPT
from prompts.title import TITLE_PROMPT
from tests.fakes import FakeLlm

HISTORY = [
    {"role": "user", "content": "What is the Model X plan?"},
    {"role": "assistant", "content": "The Model X plan costs $20 per month [1]."},
]


def test_rewrite_uses_the_fast_model_with_spec_settings(client: TestClient, llm: FakeLlm) -> None:
    llm.texts = ["What is the price of the Model X plan?"]

    res = client.post("/rewrite-query", json={"history": HISTORY, "question": "what about its price?"})

    assert res.status_code == 200
    assert res.json()["query"] == "What is the price of the Model X plan?"
    usage = {"model": "test-fast-model", "input_tokens": 11, "output_tokens": 5, "latency_ms": 7}
    assert res.json()["usage"] == usage
    call = llm.calls[0]
    assert (call["model"], call["max_tokens"], call["temperature"]) == ("test-fast-model", 100, 0)
    system, user = call["messages"]
    assert system.role == "system" and system.content == REWRITE_QUERY_PROMPT
    assert "User: What is the Model X plan?" in user.content
    assert "Assistant: The Model X plan costs $20 per month." in user.content
    assert "[1]" not in user.content
    assert "Latest message: what about its price?" in user.content


def test_rewrite_looks_at_the_last_six_messages_only(client: TestClient, llm: FakeLlm) -> None:
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"} for i in range(10)]

    client.post("/rewrite-query", json={"history": history, "question": "and then?"})

    prompt = llm.calls[0]["messages"][1].content
    assert "turn 3" not in prompt and "turn 4" in prompt and "turn 9" in prompt


def test_first_message_needs_no_model_call(client: TestClient, llm: FakeLlm) -> None:
    res = client.post("/rewrite-query", json={"history": [], "question": "  What is the refund policy?  "})

    assert res.json()["query"] == "What is the refund policy?"
    usage = {"model": "test-fast-model", "input_tokens": 0, "output_tokens": 0, "latency_ms": 0}
    assert res.json()["usage"] == usage
    assert llm.calls == []


def test_history_is_optional(client: TestClient, llm: FakeLlm) -> None:
    assert client.post("/rewrite-query", json={"question": "hello"}).json()["query"] == "hello"
    assert llm.calls == []


def test_rewrite_output_is_cleaned(client: TestClient, llm: FakeLlm) -> None:
    llm.texts = ['\n  Standalone search query: "What is the price of Model X?"\nExtra text nobody asked for']

    res = client.post("/rewrite-query", json={"history": HISTORY, "question": "price?"})

    assert res.json()["query"] == "What is the price of Model X?"


def test_empty_rewrite_falls_back_to_the_question(client: TestClient, llm: FakeLlm) -> None:
    llm.texts = ["  \n "]

    res = client.post("/rewrite-query", json={"history": HISTORY, "question": "price?"})

    assert res.json()["query"] == "price?"
    assert res.json()["usage"]["input_tokens"] == 11


def test_rewrite_model_failure_is_llm_error(client: TestClient, llm: FakeLlm) -> None:
    llm.error = AppError(ErrorCode.LLM_ERROR, "The language model is unavailable. Try again shortly.")

    res = client.post("/rewrite-query", json={"history": HISTORY, "question": "price?"})

    assert res.status_code == 502
    assert res.json()["error"]["code"] == "LLM_ERROR"


def test_rewrite_validates_input(client: TestClient) -> None:
    bad_bodies = [
        {"history": HISTORY},
        {"question": "   "},
        {"question": "x" * 4001},
        {"question": "ok", "history": [{"role": "system", "content": "nope"}]},
    ]
    for body in bad_bodies:
        res = client.post("/rewrite-query", json=body)
        assert res.status_code == 422
        assert res.json()["error"]["code"] == "INVALID_REQUEST"


def test_title_is_short_and_clean(client: TestClient, llm: FakeLlm) -> None:
    llm.texts = ['"Refund policy for annual plans and more."']

    res = client.post("/title", json={"question": "Can I get a refund on my annual plan?"})

    assert res.status_code == 200
    assert res.json()["title"] == "Refund policy for annual plans and"
    call = llm.calls[0]
    assert (call["model"], call["max_tokens"], call["temperature"]) == ("test-fast-model", 100, 0)
    assert call["messages"][0].content == TITLE_PROMPT
    assert call["messages"][1].content == "Can I get a refund on my annual plan?"


def test_title_label_and_trailing_punctuation_are_removed(client: TestClient, llm: FakeLlm) -> None:
    llm.texts = ["Title: Vacation days policy."]

    res = client.post("/title", json={"question": "How many vacation days?"})

    assert res.json()["title"] == "Vacation days policy"


def test_empty_title_falls_back_to_the_question(client: TestClient, llm: FakeLlm) -> None:
    llm.texts = [""]

    question = "How many vacation days do new hires get in their first year?"
    res = client.post("/title", json={"question": question})

    assert res.json()["title"] == "How many vacation days do new"


def test_title_requires_a_question(client: TestClient) -> None:
    res = client.post("/title", json={})

    assert res.status_code == 422
    assert res.json()["error"]["code"] == "INVALID_REQUEST"


def test_rewrite_and_title_require_the_internal_key(client: TestClient) -> None:
    wrong = {"X-Internal-Key": "wrong"}

    assert client.post("/rewrite-query", json={"question": "a"}, headers=wrong).status_code == 401
    assert client.post("/title", json={"question": "a"}, headers=wrong).status_code == 401
