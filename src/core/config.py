from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parents[2] / "config" / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    internal_api_key: SecretStr = Field(min_length=1)
    log_level: str = "INFO"
    ai_host: str = "127.0.0.1"
    ai_port: int = 8000
    libreoffice_path: str | None = None

    # Chunking (spec 7.1: 500 tokens, 80 overlap). Changing these only affects documents processed afterwards.
    chunk_tokens: int = Field(default=500, ge=100, le=2000)
    chunk_overlap_tokens: int = Field(default=80, ge=0)

    # Chat models. Ollama (LLM_BASE_URL=http://localhost:11434/v1) or, with the base URL blank, OpenAI.
    llm_base_url: str | None = None
    # Blank = use OPENAI_API_KEY (the normal case when chat goes to OpenAI). Ollama ignores the key.
    llm_api_key: SecretStr | None = None
    llm_model: str = Field(min_length=1)
    llm_fast_model: str = Field(min_length=1)
    # How hard OpenAI reasoning models (gpt-5, o-series) think: minimal, low, medium or high. Unset = low.
    llm_reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None

    # Optional separate model for grading evals (a stronger judge makes scores more trustworthy).
    # Unset = use LLM_MODEL. With LLM_JUDGE_MODEL set and LLM_JUDGE_BASE_URL unset, the judge goes to
    # OpenAI using LLM_JUDGE_API_KEY (or OPENAI_API_KEY).
    llm_judge_model: str | None = None
    llm_judge_base_url: str | None = None
    llm_judge_api_key: SecretStr | None = None
    # Optional model for reranking retrieved passages (POST /rerank). Unset = LLM_FAST_MODEL on the chat
    # client. With LLM_RERANK_MODEL set and LLM_RERANK_BASE_URL unset it goes to OpenAI
    # (LLM_RERANK_API_KEY or OPENAI_API_KEY).
    llm_rerank_model: str | None = None
    llm_rerank_base_url: str | None = None
    llm_rerank_api_key: SecretStr | None = None
    llm_rerank_reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    llm_judge_reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None

    # Embeddings always go to OpenAI; the DB column is vector(1536).
    openai_api_key: SecretStr = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    embedding_dimensions: int = Field(gt=0)

    @model_validator(mode="after")
    def _overlap_is_smaller_than_a_chunk(self) -> "Settings":
        if self.chunk_overlap_tokens >= self.chunk_tokens // 2:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be less than half of CHUNK_TOKENS")
        return self

    @field_validator(
        "llm_base_url",
        "llm_api_key",
        "llm_reasoning_effort",
        "llm_judge_model",
        "llm_judge_base_url",
        "llm_judge_api_key",
        "llm_judge_reasoning_effort",
        "llm_rerank_model",
        "llm_rerank_base_url",
        "llm_rerank_api_key",
        "llm_rerank_reasoning_effort",
        mode="before",
    )
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        return None if value == "" else value


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']).upper()}: {e['msg']}" for e in exc.errors()
        )
        raise SystemExit(f"Invalid or missing environment variables ({ENV_FILE}): {problems}") from None
