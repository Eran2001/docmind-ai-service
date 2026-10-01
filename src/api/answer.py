import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.deps import get_llm
from core.config import Settings, get_settings
from core.errors import AppError
from core.llm import LlmClient, StreamDone, TextDelta
from core.usage import Usage
from services.answer import AnswerInput, answer_once, answer_stream

router = APIRouter(tags=["answer"])


class AnswerRequest(AnswerInput):
    stream: bool


class AnswerResponse(BaseModel):
    answer: str
    usage: Usage


def _sse(event: str, data: dict[str, object]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n".encode()


async def _event_stream(llm: LlmClient, model: str, request: AnswerInput) -> AsyncIterator[bytes]:
    try:
        async for event in answer_stream(llm, model=model, request=request):
            if isinstance(event, TextDelta):
                yield _sse("token", {"text": event.text})
            elif isinstance(event, StreamDone):
                yield _sse("done", {"usage": event.usage.model_dump()})
    except AppError as exc:
        yield _sse("error", {"message": exc.message})


@router.post("/answer", response_model=None)
async def answer(
    body: AnswerRequest,
    llm: Annotated[LlmClient, Depends(get_llm)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AnswerResponse | StreamingResponse:
    request = AnswerInput(question=body.question, history=body.history, chunks=body.chunks)
    if body.stream:
        return StreamingResponse(
            _event_stream(llm, settings.llm_model, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    result = await answer_once(llm, model=settings.llm_model, request=request)
    return AnswerResponse(answer=result.answer, usage=result.usage)
