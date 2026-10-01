from collections.abc import AsyncIterator
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from core.llm import ChatMessage, LlmClient, StreamDone, TextDelta
from core.usage import Usage
from prompts.answer import ANSWER_SYSTEM_PROMPT, build_answer_input
from services.history import HistoryMessage, prepare_history

ANSWER_MAX_TOKENS = 1024
ANSWER_TEMPERATURE = 0.2
MAX_CHUNKS = 8
MAX_CHUNK_CHARS = 20_000


class AnswerChunk(BaseModel):
    id: str = Field(min_length=1, max_length=100)
    document_title: str = Field(min_length=1, max_length=500)
    page_number: int | None = Field(default=None, ge=1)
    content: str = Field(min_length=1, max_length=MAX_CHUNK_CHARS)


class AnswerInput(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
    history: list[HistoryMessage] = Field(default_factory=list)
    chunks: list[AnswerChunk] = Field(max_length=MAX_CHUNKS)


class AnswerResult(BaseModel):
    answer: str
    usage: Usage


type AnswerEvent = TextDelta | StreamDone


def build_messages(request: AnswerInput) -> list[ChatMessage]:
    history = prepare_history(request.history)
    messages = [ChatMessage(role="system", content=ANSWER_SYSTEM_PROMPT)]
    messages.extend(ChatMessage(role=m.role, content=m.content) for m in history)
    messages.append(ChatMessage(role="user", content=build_answer_input(request.question, request.chunks)))
    return messages


async def answer_once(llm: LlmClient, *, model: str, request: AnswerInput) -> AnswerResult:
    result = await llm.complete(
        build_messages(request),
        model=model,
        max_tokens=ANSWER_MAX_TOKENS,
        temperature=ANSWER_TEMPERATURE,
    )
    return AnswerResult(answer=result.text, usage=result.usage)


async def answer_stream(llm: LlmClient, *, model: str, request: AnswerInput) -> AsyncIterator[AnswerEvent]:
    async for event in llm.stream(
        build_messages(request),
        model=model,
        max_tokens=ANSWER_MAX_TOKENS,
        temperature=ANSWER_TEMPERATURE,
    ):
        yield event
