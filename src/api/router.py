from fastapi import APIRouter, Depends

from api import answer, embed, health, ingest, judge, rerank, rewrite, title
from api.deps import verify_internal_key

# Every route except /health requires X-Internal-Key; feature routers are included on `protected_router`.
protected_router = APIRouter(dependencies=[Depends(verify_internal_key)])
protected_router.include_router(ingest.router)
protected_router.include_router(embed.router)
protected_router.include_router(rewrite.router)
protected_router.include_router(title.router)
protected_router.include_router(answer.router)
protected_router.include_router(judge.router)
protected_router.include_router(rerank.router)

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(protected_router)
