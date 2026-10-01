from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.deps import get_llm
from api.schemas import Question
from core.config import Settings, get_settings
from core.llm import LlmClient
from core.usage import Usage
from services.history import HistoryMessage
from services.query import rewrite_query

router = APIRouter(tags=["rewrite"])


class RewriteRequest(BaseModel):
    history: list[HistoryMessage] = []
    question: Question


class RewriteResponse(BaseModel):
    query: str
    usage: Usage


@router.post("/rewrite-query")
async def rewrite(
    body: RewriteRequest,
    llm: Annotated[LlmClient, Depends(get_llm)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RewriteResponse:
    result = await rewrite_query(
        llm, model=settings.llm_fast_model, history=body.history, question=body.question
    )
    return RewriteResponse(query=result.query, usage=result.usage)
