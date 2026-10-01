from typing import Any

from fastapi.testclient import TestClient

from core.errors import AppError, ErrorCode
from core.llm import StreamDone, TextDelta
from core.usage import Usage
from prompts.answer import ANSWER_SYSTEM_PROMPT
from tests.fakes import FakeLlm

HISTORY = [
    {"role": "user", "content": "What is the refund policy?"},
    {"role": "assistant", "content": "Returns are accepted for 30 days [1]."},
]
CHUNKS = [
    {
        "id": "chunk-1",
        "document_title": "Refunds.pdf",
        "page_number": 4,
        "content": "Returns are accepted within thirty days of purchase.",
    },
    {
        "id": "chunk-2",
        "document_title": "Returns.md",
        "page_number": None,
        "content": "Contact support to start a return.",
    },
]


def test_answer_json_uses_model_and_formats_grounded_context(client: TestClient, llm: FakeLlm) -> None:
    llm.texts = ["Returns are accepted for 30 days [1]."]

    res = client.post(
        "/answer",
        json={"question": "Can I return this?", "history": HISTORY, "chunks": CHUNKS, "stream": False},
    )

    assert res.status_code == 200
    assert res.json()["answer"] == "Returns are accepted for 30 days [1]."
    assert res.json()["usage"] == {
        "model": "test-model",
        "input_tokens": 11,
        "output_tokens": 5,
        "latency_ms": 7,
    }
    call = llm.calls[0]
    assert (call["model"], call["max_tokens"], call["temperature"]) == ("test-model", 1024, 0.2)
    system, previous_user, previous_assistant, current = call["messages"]
    assert system.content == ANSWER_SYSTEM_PROMPT
    assert previous_user.content == "What is the refund policy?"
    assert previous_assistant.content == "Returns are accepted for 30 days."
    assert "[1] (Refunds.pdf, page 4)" in current.content
    assert "[2] (Returns.md)" in current.content
    assert "[1]" not in previous_assistant.content
    assert "Question: Can I return this?" in current.content


def test_answer_limits_history_to_last_six(client: TestClient, llm: FakeLlm) -> None:
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i} [1]"} for i in range(10)
    ]

    client.post("/answer", json={"question": "latest?", "history": history, "chunks": [], "stream": False})

    messages = llm.calls[0]["messages"]
    assert len(messages) == 8
    assert "message 3" not in " ".join(m.content for m in messages)
    assert "message 4" in messages[1].content
    assert all("[1]" not in m.content for m in messages[1:])


def test_answer_stream_emits_tokens_then_usage(client: TestClient, llm: FakeLlm) -> None:
    llm.stream_events = [
        TextDelta(text="Returns are accepted "),
        TextDelta(text="for thirty days [1]."),
        StreamDone(usage=Usage(model="test-model", input_tokens=10, output_tokens=6, latency_ms=8)),
    ]

    with client.stream(
        "POST",
        "/answer",
        json={"question": "Can I return this?", "history": [], "chunks": CHUNKS, "stream": True},
    ) as res:
        body = res.read().decode()

    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    frames = body.strip().split("\n\n")
    done_frame = (
        'event: done\ndata: {"usage":{"model":"test-model","input_tokens":10,'
        '"output_tokens":6,"latency_ms":8}}'
    )
    assert frames == [
        'event: token\ndata: {"text":"Returns are accepted "}',
        'event: token\ndata: {"text":"for thirty days [1]."}',
        done_frame,
    ]
    assert res.headers["cache-control"] == "no-cache"


def test_answer_stream_reports_errors_as_sse_event(client: TestClient, llm: FakeLlm) -> None:
    llm.stream_events = [TextDelta(text="partial")]
    llm.stream_error = AppError(ErrorCode.LLM_ERROR, "The language model is unavailable.")

    with client.stream(
        "POST",
        "/answer",
        json={"question": "Question?", "history": [], "chunks": [], "stream": True},
    ) as res:
        body = res.read().decode()

    assert res.status_code == 200
    error_frame = 'event: error\ndata: {"message":"The language model is unavailable."}\n\n'
    assert body.endswith(error_frame)


def test_answer_model_failure_uses_standard_error_json(client: TestClient, llm: FakeLlm) -> None:
    llm.error = AppError(ErrorCode.LLM_ERROR, "The language model is unavailable.")

    res = client.post("/answer", json={"question": "Question?", "history": [], "chunks": [], "stream": False})

    assert res.status_code == 502
    assert res.json()["error"]["code"] == "LLM_ERROR"


def test_answer_validates_chunks_question_and_stream(client: TestClient) -> None:
    invalid: list[dict[str, Any]] = [
        {"question": " ", "history": [], "chunks": [], "stream": False},
        {"question": "ok", "history": [], "chunks": CHUNKS * 5, "stream": False},
        {
            "question": "ok",
            "history": [],
            "chunks": [
                {
                    "id": "bad-page",
                    "document_title": "Doc",
                    "page_number": 0,
                    "content": "Some source content.",
                }
            ],
            "stream": False,
        },
        {"question": "ok", "history": [], "chunks": CHUNKS},
    ]
    for body in invalid:
        res = client.post("/answer", json=body)
        assert res.status_code == 422
        assert res.json()["error"]["code"] == "INVALID_REQUEST"


def test_answer_requires_internal_key(client: TestClient) -> None:
    res = client.post(
        "/answer",
        json={"question": "Question?", "history": [], "chunks": [], "stream": False},
        headers={"X-Internal-Key": "wrong"},
    )

    assert res.status_code == 401
