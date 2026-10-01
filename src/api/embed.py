from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, StringConstraints

from api.deps import get_embedder
from core.embeddings import EmbeddingClient
from core.usage import Usage

router = APIRouter(tags=["embed"])

MAX_TEXTS = 100

type Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class EmbedRequest(BaseModel):
    texts: Annotated[list[Text], Field(min_length=1, max_length=MAX_TEXTS)]


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    usage: Usage


@router.post("/embed")
async def embed(
    body: EmbedRequest, embedder: Annotated[EmbeddingClient, Depends(get_embedder)]
) -> EmbedResponse:
    result = await embedder.embed(body.texts)
    return EmbedResponse(embeddings=result.embeddings, usage=result.usage)
