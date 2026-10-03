import time
from collections.abc import AsyncIterator
from typing import Any, Literal, cast

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


REASONING_HEADROOM = 3000  # tokens a reasoning model may spend thinking, on top of the visible answer
DEFAULT_REASONING_EFFORT = "low"


def is_reasoning_model(model: str) -> bool:
    """OpenAI's GPT-5 and o-series models think before answering and take different parameters."""
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


class LlmClient:
    """Chat completions against any OpenAI-compatible endpoint (Ollama today, OpenAI later)."""

    def __init__(self, client: AsyncOpenAI, reasoning_effort: str | None = None) -> None:
        self._client = client
        self._reasoning_effort = reasoning_effort

    @classmethod
    def from_settings(cls, settings: Settings) -> "LlmClient":
        return cls.from_parts(
            settings.llm_base_url,
            (settings.llm_api_key or settings.openai_api_key).get_secret_value(),
            settings.llm_reasoning_effort,
        )

    @classmethod
    def from_parts(
        cls, base_url: str | None, api_key: str, reasoning_effort: str | None = None
    ) -> "LlmClient":
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=TIMEOUT_SECONDS,
            max_retries=0,  # retries are ours (core/retry.py) so they follow the spec's policy
        )
        return cls(client, reasoning_effort)

    def _params(self, model: str, max_tokens: int, temperature: float) -> dict[str, Any]:
        """Per-call sampling parameters.

        Reasoning models reject `temperature`, and their hidden thinking counts against the token limit,
        so they get extra headroom and a (low by default) reasoning effort instead.
        """
        if is_reasoning_model(model):
            return {
                "max_completion_tokens": max_tokens + REASONING_HEADROOM,
                "reasoning_effort": self._reasoning_effort or DEFAULT_REASONING_EFFORT,
            }
        return {"max_completion_tokens": max_tokens, "temperature": temperature}

    async def complete(
        self, messages: list[ChatMessage], *, model: str, max_tokens: int, temperature: float
    ) -> LlmResult:
        started = time.perf_counter()
        try:
            response = await with_retries(
                lambda: self._client.chat.completions.create(
                    model=model,
                    messages=_messages(messages),
                    **self._params(model, max_tokens, temperature),
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
                    messages=_messages(messages),
                    **self._params(model, max_tokens, temperature),
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


def _messages(messages: list[ChatMessage]) -> list[ChatCompletionMessageParam]:
    return cast(list[ChatCompletionMessageParam], [m.model_dump() for m in messages])


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def _llm_error(exc: openai.OpenAIError) -> AppError:
    status = exc.status_code if isinstance(exc, openai.APIStatusError) else None
    log.error("llm_call_failed", error_type=type(exc).__name__, status=status)
    return AppError(ErrorCode.LLM_ERROR, _LLM_UNAVAILABLE)
