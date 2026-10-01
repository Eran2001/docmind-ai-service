from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.deps import get_llm
from api.schemas import Question
from core.config import Settings, get_settings
from core.llm import LlmClient
from core.usage import Usage
from services.query import make_title

router = APIRouter(tags=["title"])


class TitleRequest(BaseModel):
    question: Question


class TitleResponse(BaseModel):
    title: str
    usage: Usage


@router.post("/title")
async def title(
    body: TitleRequest,
    llm: Annotated[LlmClient, Depends(get_llm)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> TitleResponse:
    result = await make_title(llm, model=settings.llm_fast_model, question=body.question)
    return TitleResponse(title=result.title, usage=result.usage)
