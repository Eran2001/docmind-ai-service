import hmac
from functools import cache
from typing import Annotated

from fastapi import Depends, Header

from core.config import Settings, get_settings
from core.embeddings import EmbeddingClient
from core.errors import AppError, ErrorCode
from core.llm import LlmClient


@cache
def get_llm() -> LlmClient:
    return LlmClient.from_settings(get_settings())


@cache
def get_judge_llm() -> LlmClient:
    """The client for grading evals: the separate judge model if one is configured, else the chat client."""
    settings = get_settings()
    if not settings.llm_judge_model:
        return get_llm()
    key = settings.llm_judge_api_key or settings.openai_api_key
    return LlmClient.from_parts(
        settings.llm_judge_base_url, key.get_secret_value(), settings.llm_judge_reasoning_effort
    )


@cache
def get_rerank_llm() -> LlmClient:
    """The client for reranking: a separate model if LLM_RERANK_MODEL is set, otherwise the chat client."""
    settings = get_settings()
    if not settings.llm_rerank_model:
        return get_llm()
    key = settings.llm_rerank_api_key or settings.openai_api_key
    return LlmClient.from_parts(
        settings.llm_rerank_base_url,
        key.get_secret_value(),
        settings.llm_rerank_reasoning_effort or "minimal",
    )


@cache
def get_embedder() -> EmbeddingClient:
    return EmbeddingClient.from_settings(get_settings())


def verify_internal_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_internal_key: Annotated[str | None, Header()] = None,
) -> None:
    expected = settings.internal_api_key.get_secret_value().encode()
    if x_internal_key is None or not hmac.compare_digest(x_internal_key.encode(), expected):
        raise AppError(ErrorCode.UNAUTHORIZED, "Missing or invalid internal key.")
