import asyncio

from pydantic import BaseModel

from core.chunking import Chunk, chunk_document
from core.cleaning import ParsedDocument, remove_repeated_lines
from core.config import get_settings
from core.embeddings import EmbeddingClient
from core.errors import AppError, ErrorCode
from core.parsing import parse_file
from core.usage import Usage
from core.web import extract_page
from tools.safe_fetch import fetch_page


class EmbeddedChunk(Chunk):
    embedding: list[float]


class IngestResult(BaseModel):
    title: str | None = None
    page_count: int | None
    chunks: list[EmbeddedChunk]
    usage: Usage


async def ingest_file(data: bytes, mime_type: str, embedder: EmbeddingClient) -> IngestResult:
    """Parse, clean, chunk and embed one file."""
    # Parsing and chunking are CPU-bound; keep them off the event loop.
    document = await asyncio.to_thread(
        parse_file,
        data,
        mime_type,
        libreoffice_path=get_settings().libreoffice_path,
    )
    return await _chunk_and_embed(document, embedder)


async def ingest_url(url: str, embedder: EmbeddingClient) -> IngestResult:
    """Fetch a public web page (SSRF-guarded), extract its text, then chunk and embed it."""
    page = await fetch_page(url)
    document = await asyncio.to_thread(extract_page, page.content, page.url)
    return await _chunk_and_embed(document, embedder)


async def _chunk_and_embed(document: ParsedDocument, embedder: EmbeddingClient) -> IngestResult:
    chunks = await asyncio.to_thread(_chunk, document)
    embedded = await embedder.embed([chunk.content for chunk in chunks])
    return IngestResult(
        title=document.title,
        page_count=document.page_count,
        chunks=[
            EmbeddedChunk(**chunk.model_dump(), embedding=vector)
            for chunk, vector in zip(chunks, embedded.embeddings, strict=True)
        ],
        usage=embedded.usage,
    )


def _chunk(document: ParsedDocument) -> list[Chunk]:
    cleaned = document.model_copy(update={"pages": remove_repeated_lines(document.pages)})
    settings = get_settings()
    chunks = chunk_document(cleaned, settings.chunk_tokens, settings.chunk_overlap_tokens)
    if not chunks:
        raise AppError(ErrorCode.EMPTY_DOCUMENT, "This document has too little text to index.")
    return chunks
