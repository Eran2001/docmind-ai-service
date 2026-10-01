from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parents[2] / "config" / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    internal_api_key: SecretStr = Field(min_length=1)
    log_level: str = "INFO"
    ai_host: str = "127.0.0.1"
    ai_port: int = 8000

    # Chat models. Ollama today (LLM_BASE_URL=http://localhost:11434/v1); unset the base URL to use OpenAI.
    llm_base_url: str | None = None
    llm_api_key: SecretStr = Field(min_length=1)
    llm_model: str = Field(min_length=1)
    llm_fast_model: str = Field(min_length=1)

    # Embeddings always go to OpenAI; the DB column is vector(1536).
    openai_api_key: SecretStr = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    embedding_dimensions: int = Field(gt=0)

    @field_validator("llm_base_url", mode="before")
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
