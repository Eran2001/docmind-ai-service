import pytest
from fastapi.testclient import TestClient

from api.deps import get_embedder, get_llm
from core.config import get_settings
from docmind_ai.app import create_app
from tests.fakes import FakeEmbedder, FakeLlm

KEY = "test-internal-key"


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "INTERNAL_API_KEY": KEY,
        "LLM_API_KEY": "ollama",
        "LLM_MODEL": "test-model",
        "LLM_FAST_MODEL": "test-fast-model",
        "OPENAI_API_KEY": "sk-test",
        "EMBEDDING_MODEL": "test-embedding",
        "EMBEDDING_DIMENSIONS": "1536",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def llm() -> FakeLlm:
    return FakeLlm()


@pytest.fixture
def client(embedder: FakeEmbedder, llm: FakeLlm) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_llm] = lambda: llm
    return TestClient(app, headers={"X-Internal-Key": KEY})
