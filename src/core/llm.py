import time
from collections.abc import AsyncIterator
from typing import Literal, cast

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from core.config import Settings
from core.errors import AppError, ErrorCode
from core.logging import get_logger
from core.retry import with_retries
from core.usage import Usage

log = get_logger(__name__)

TIMEOUT_SECONDS = 60.0
_LLM_UNAVAILABLE = "The language model is unavailable. Try again shortly."


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LlmResult(BaseModel):
    text: str
    usage: Usage


class TextDelta(BaseModel):
    text: str


class StreamDone(BaseModel):
    usage: Usage


type StreamEvent = TextDelta | StreamDone


class LlmClient:
    """Chat completions against any OpenAI-compatible endpoint (Ollama today, OpenAI later)."""

    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> "LlmClient":
        client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value(),
            timeout=TIMEOUT_SECONDS,
            max_retries=0,  # retries are ours (core/retry.py) so they follow the spec's policy
        )
        return cls(client)

    async def complete(
        self, messages: list[ChatMessage], *, model: str, max_tokens: int, temperature: float
    ) -> LlmResult:
        started = time.perf_counter()
        try:
            response = await with_retries(
                lambda: self._client.chat.completions.create(
                    model=model,
                    messages=_params(messages),
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                )
            )
        except openai.OpenAIError as exc:
            raise _llm_error(exc) from exc
        usage = response.usage
        return LlmResult(
            text=response.choices[0].message.content or "" if response.choices else "",
            usage=Usage(
                model=model,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                latency_ms=_elapsed_ms(started),
            ),
        )

    async def stream(
        self, messages: list[ChatMessage], *, model: str, max_tokens: int, temperature: float
    ) -> AsyncIterator[StreamEvent]:
        """Yields a `TextDelta` per piece of text, then one `StreamDone` with usage.

        Retries only before the first token.
        """
        started = time.perf_counter()
        try:
            chunks = await with_retries(
                lambda: self._client.chat.completions.create(
                    model=model,
                    messages=_params(messages),
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                    stream_options={"include_usage": True},
                )
            )
            input_tokens = output_tokens = 0
            async for chunk in chunks:
                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens
                if chunk.choices and chunk.choices[0].delta.content:
                    yield TextDelta(text=chunk.choices[0].delta.content)
        except openai.OpenAIError as exc:
            raise _llm_error(exc) from exc
        yield StreamDone(
            usage=Usage(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=_elapsed_ms(started),
            )
        )


def _params(messages: list[ChatMessage]) -> list[ChatCompletionMessageParam]:
    return cast(list[ChatCompletionMessageParam], [m.model_dump() for m in messages])


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def _llm_error(exc: openai.OpenAIError) -> AppError:
    status = exc.status_code if isinstance(exc, openai.APIStatusError) else None
    log.error("llm_call_failed", error_type=type(exc).__name__, status=status)
    return AppError(ErrorCode.LLM_ERROR, _LLM_UNAVAILABLE)
