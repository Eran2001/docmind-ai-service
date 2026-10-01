from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import AnyHttpUrl, BaseModel

from api.deps import get_embedder
from core.embeddings import EmbeddingClient
from core.usage import Usage
from services.ingest import EmbeddedChunk, ingest_file, ingest_url

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestFileResponse(BaseModel):
    page_count: int | None
    chunks: list[EmbeddedChunk]
    usage: Usage


@router.post("/file")
async def ingest_file_route(
    file: Annotated[UploadFile, File()],
    mime_type: Annotated[str, Form()],
    embedder: Annotated[EmbeddingClient, Depends(get_embedder)],
) -> IngestFileResponse:
    # "text/plain; charset=utf-8" and "Text/Plain" are the same type.
    normalized = mime_type.split(";")[0].strip().lower()
    result = await ingest_file(await file.read(), normalized, embedder)
    return IngestFileResponse(**result.model_dump())


class IngestUrlRequest(BaseModel):
    url: AnyHttpUrl


class IngestUrlResponse(BaseModel):
    title: str | None
    page_count: None = None
    chunks: list[EmbeddedChunk]
    usage: Usage


@router.post("/url")
async def ingest_url_route(
    body: IngestUrlRequest, embedder: Annotated[EmbeddingClient, Depends(get_embedder)]
) -> IngestUrlResponse:
    result = await ingest_url(str(body.url), embedder)
    return IngestUrlResponse(title=result.title, chunks=result.chunks, usage=result.usage)
