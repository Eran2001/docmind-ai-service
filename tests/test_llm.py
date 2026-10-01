import openai
import pytest

from core import retry
from core.errors import AppError, ErrorCode
from core.llm import ChatMessage, LlmClient, StreamDone, StreamEvent, TextDelta
from tests.fakes import Script, completion, fake_openai, status_error, stream_chunk

MESSAGES = [ChatMessage(role="user", content="hi")]


@pytest.fixture(autouse=True)
def _no_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retry, "_BASE_DELAY_SECONDS", 0.0)


async def test_complete_returns_text_and_usage() -> None:
    chat = Script(completion("hello", prompt_tokens=12, completion_tokens=4))
    llm = LlmClient(fake_openai(chat=chat))

    result = await llm.complete(MESSAGES, model="m", max_tokens=100, temperature=0.2)

    assert result.text == "hello"
    assert (result.usage.model, result.usage.input_tokens, result.usage.output_tokens) == ("m", 12, 4)
    assert result.usage.latency_ms >= 0
    assert chat.calls[0]["max_completion_tokens"] == 100
    assert chat.calls[0]["temperature"] == 0.2
    assert chat.calls[0]["messages"] == [{"role": "user", "content": "hi"}]


async def test_complete_retries_then_succeeds() -> None:
    chat = Script(status_error(openai.RateLimitError, 429), completion("ok"))
    llm = LlmClient(fake_openai(chat=chat))

    assert (await llm.complete(MESSAGES, model="m", max_tokens=10, temperature=0)).text == "ok"
    assert len(chat.calls) == 2


async def test_complete_failure_becomes_llm_error() -> None:
    chat = Script(status_error(openai.BadRequestError, 400))
    llm = LlmClient(fake_openai(chat=chat))

    with pytest.raises(AppError) as exc:
        await llm.complete(MESSAGES, model="m", max_tokens=10, temperature=0)

    assert exc.value.code is ErrorCode.LLM_ERROR


async def test_stream_yields_text_then_usage() -> None:
    chat = Script([stream_chunk("Hel"), stream_chunk("lo"), stream_chunk(usage=(9, 2))])
    llm = LlmClient(fake_openai(chat=chat))

    events: list[StreamEvent] = [
        e async for e in llm.stream(MESSAGES, model="m", max_tokens=10, temperature=0)
    ]

    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Hel", "lo"]
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert (done.usage.input_tokens, done.usage.output_tokens) == (9, 2)
    assert chat.calls[0]["stream"] is True


async def test_stream_retries_before_first_token() -> None:
    chat = Script(
        status_error(openai.InternalServerError, 502), [stream_chunk("a"), stream_chunk(usage=(1, 1))]
    )
    llm = LlmClient(fake_openai(chat=chat))

    events = [e async for e in llm.stream(MESSAGES, model="m", max_tokens=10, temperature=0)]

    assert isinstance(events[0], TextDelta)
    assert len(chat.calls) == 2
