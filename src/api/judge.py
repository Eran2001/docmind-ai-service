from typing import Annotated

from fastapi import APIRouter, Depends

from api.deps import get_judge_llm
from core.config import Settings, get_settings
from core.llm import LlmClient
from services.judge import JudgeInput, JudgeResult, judge_answer

router = APIRouter(tags=["evals"])


@router.post("/evals/judge")
async def judge(
    body: JudgeInput,
    llm: Annotated[LlmClient, Depends(get_judge_llm)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> JudgeResult:
    # The run reports whichever model graded it as the judge model.
    return await judge_answer(llm, model=settings.llm_judge_model or settings.llm_model, request=body)
