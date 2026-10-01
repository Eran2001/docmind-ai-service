import time

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel

from core.config import Settings
from core.errors import AppError, ErrorCode
from core.logging import get_logger
from core.retry import with_retries
from core.usage import Usage

log = get_logger(__name__)

BATCH_SIZE = 100
TIMEOUT_SECONDS = 60.0


class EmbeddingResult(BaseModel):
    embeddings: list[list[float]]
    usage: Usage


class EmbeddingClient:
    """OpenAI embeddings. The size must match the DB column, so `dimensions` is always sent and checked."""

    def __init__(self, client: AsyncOpenAI, *, model: str, dimensions: int) -> None:
        self._client = client
        self._model = model
        self._dimensions = dimensions

    @classmethod
    def from_settings(cls, settings: Settings) -> "EmbeddingClient":
        client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(), timeout=TIMEOUT_SECONDS, max_retries=0
        )
        return cls(client, model=settings.embedding_model, dimensions=settings.embedding_dimensions)

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        started = time.perf_counter()
        vectors: list[list[float]] = []
        input_tokens = 0
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            try:
                response = await with_retries(
                    lambda batch=batch: self._client.embeddings.create(  # type: ignore[misc]
                        model=self._model, input=batch, dimensions=self._dimensions
                    )
                )
            except openai.OpenAIError as exc:
                log.error(
                    "embedding_call_failed",
                    error_type=type(exc).__name__,
                    status=exc.status_code if isinstance(exc, openai.APIStatusError) else None,
                )
                raise AppError(
                    ErrorCode.LLM_ERROR, "The embedding service is unavailable. Try again shortly."
                ) from exc
            batch_vectors = [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
            if len(batch_vectors) != len(batch) or any(len(v) != self._dimensions for v in batch_vectors):
                log.error("embedding_shape_mismatch", expected_dimensions=self._dimensions)
                raise AppError(ErrorCode.LLM_ERROR, "The embedding service returned an unexpected result.")
            vectors.extend(batch_vectors)
            input_tokens += response.usage.prompt_tokens
        return EmbeddingResult(
            embeddings=vectors,
            usage=Usage(
                model=self._model,
                input_tokens=input_tokens,
                output_tokens=0,
                latency_ms=round((time.perf_counter() - started) * 1000),
            ),
        )
