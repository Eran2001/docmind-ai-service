from types import SimpleNamespace

import openai
import pytest

from core.embeddings import EmbeddingClient
from core.errors import AppError, ErrorCode
from tests.fakes import Script, fake_openai, status_error

DIMS = 4


def response(count: int, *, dims: int = DIMS, tokens: int = 5) -> SimpleNamespace:
    data = [SimpleNamespace(index=i, embedding=[0.1] * dims) for i in range(count)]
    return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=tokens))


def client(script: Script) -> EmbeddingClient:
    return EmbeddingClient(fake_openai(embeddings=script), model="emb", dimensions=DIMS)


async def test_embeds_in_batches_of_100_and_sums_usage() -> None:
    script = Script(response(100), response(100), response(50))

    result = await client(script).embed(["t"] * 250)

    assert len(result.embeddings) == 250
    assert [len(c["input"]) for c in script.calls] == [100, 100, 50]
    assert script.calls[0]["dimensions"] == DIMS
    assert (result.usage.model, result.usage.input_tokens, result.usage.output_tokens) == ("emb", 15, 0)


async def test_empty_input_makes_no_call() -> None:
    script = Script()
    result = await client(script).embed([])
    assert result.embeddings == []
    assert script.calls == []


async def test_wrong_dimensions_is_an_error() -> None:
    with pytest.raises(AppError) as exc:
        await client(Script(response(1, dims=DIMS + 1))).embed(["t"])
    assert exc.value.code is ErrorCode.LLM_ERROR


async def test_api_failure_becomes_llm_error() -> None:
    with pytest.raises(AppError) as exc:
        await client(Script(status_error(openai.AuthenticationError, 401))).embed(["t"])
    assert exc.value.code is ErrorCode.LLM_ERROR
