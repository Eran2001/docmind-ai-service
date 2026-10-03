from typing import Annotated

from fastapi import APIRouter, Depends

from api.deps import get_rerank_llm
from core.config import Settings, get_settings
from core.llm import LlmClient
from services.rerank import RerankInput, RerankResult, rerank

router = APIRouter(tags=["rerank"])


@router.post("/rerank")
async def rerank_passages(
    body: RerankInput,
    llm: Annotated[LlmClient, Depends(get_rerank_llm)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RerankResult:
    return await rerank(llm, model=settings.llm_rerank_model or settings.llm_fast_model, request=body)
