import openai
import pytest

from core import retry
from core.retry import MAX_RETRIES, with_retries
from tests.fakes import connection_error, status_error


@pytest.fixture(autouse=True)
def _no_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retry, "_BASE_DELAY_SECONDS", 0.0)


class Flaky:
    def __init__(self, *failures: Exception) -> None:
        self.failures = list(failures)
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return "ok"


async def test_retries_429_then_succeeds() -> None:
    call = Flaky(status_error(openai.RateLimitError, 429), status_error(openai.InternalServerError, 503))
    assert await with_retries(call) == "ok"
    assert call.calls == 3


async def test_retries_connection_errors() -> None:
    call = Flaky(connection_error())
    assert await with_retries(call) == "ok"


async def test_gives_up_after_max_retries() -> None:
    call = Flaky(*[status_error(openai.InternalServerError, 500) for _ in range(10)])
    with pytest.raises(openai.InternalServerError):
        await with_retries(call)
    assert call.calls == MAX_RETRIES + 1


async def test_does_not_retry_exhausted_quota() -> None:
    call = Flaky(status_error(openai.RateLimitError, 429, code="insufficient_quota"))
    with pytest.raises(openai.RateLimitError):
        await with_retries(call)
    assert call.calls == 1


async def test_does_not_retry_client_errors() -> None:
    call = Flaky(status_error(openai.BadRequestError, 400))
    with pytest.raises(openai.BadRequestError):
        await with_retries(call)
    assert call.calls == 1
