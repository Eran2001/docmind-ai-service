from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import httpx2
import openai
from openai import AsyncOpenAI

from core.embeddings import EmbeddingResult
from core.llm import ChatMessage, LlmResult, StreamDone, StreamEvent, TextDelta
from core.usage import Usage


class Script:
    """A fake async `create`: each call returns the next scripted item (raised if it is an exception)."""

    def __init__(self, *items: Any) -> None:
        self.items = list(items)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, list):
            return _chunks(item)
        return item


async def _chunks(items: list[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


def fake_openai(*, chat: Script | None = None, embeddings: Script | None = None) -> AsyncOpenAI:
    fake = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=chat)),
        embeddings=SimpleNamespace(create=embeddings),
    )
    return cast(AsyncOpenAI, fake)


def status_error(
    cls: type[openai.APIStatusError], status_code: int, *, code: str | None = None
) -> openai.APIStatusError:
    request = httpx2.Request("POST", "http://test")
    body = {"code": code} if code else None
    return cls("boom", response=httpx2.Response(status_code, request=request), body=body)


def connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx2.Request("POST", "http://test"))


def completion(text: str, prompt_tokens: int = 7, completion_tokens: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def stream_chunk(text: str | None = None, *, usage: tuple[int, int] | None = None) -> SimpleNamespace:
    choices = [SimpleNamespace(delta=SimpleNamespace(content=text))] if text is not None else []
    u = SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1]) if usage else None
    return SimpleNamespace(choices=choices, usage=u)


class FakeEmbedder:
    """Stands in for EmbeddingClient: fixed-size vectors, or an error when `error` is set."""

    def __init__(self, dimensions: int = 4, error: Exception | None = None) -> None:
        self.dimensions = dimensions
        self.error = error
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        self.calls.append(texts)
        if self.error:
            raise self.error
        usage = Usage(model="fake-embedding", input_tokens=len(texts) * 3, output_tokens=0, latency_ms=1)
        return EmbeddingResult(embeddings=[[0.5] * self.dimensions for _ in texts], usage=usage)


class FakeLlm:
    """Stands in for LlmClient.complete: returns scripted texts in order (then "ok"), or raises `error`."""

    def __init__(self, *texts: str, error: Exception | None = None) -> None:
        self.texts = list(texts)
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.stream_events: list[StreamEvent] | None = None
        self.stream_error: Exception | None = None
        self.stream_calls: list[dict[str, Any]] = []

    async def complete(
        self, messages: list[ChatMessage], *, model: str, max_tokens: int, temperature: float
    ) -> LlmResult:
        self.calls.append(
            {"messages": messages, "model": model, "max_tokens": max_tokens, "temperature": temperature}
        )
        if self.error:
            raise self.error
        text = self.texts.pop(0) if self.texts else "ok"
        return LlmResult(text=text, usage=Usage(model=model, input_tokens=11, output_tokens=5, latency_ms=7))

    async def stream(
        self, messages: list[ChatMessage], *, model: str, max_tokens: int, temperature: float
    ) -> AsyncIterator[StreamEvent]:
        self.stream_calls.append(
            {"messages": messages, "model": model, "max_tokens": max_tokens, "temperature": temperature}
        )
        if self.stream_error:
            raise self.stream_error
        events = self.stream_events or [
            TextDelta(text="ok"),
            StreamDone(usage=Usage(model=model, input_tokens=11, output_tokens=5, latency_ms=7)),
        ]
        for event in events:
            yield event
