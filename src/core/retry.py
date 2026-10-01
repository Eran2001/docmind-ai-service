import asyncio
import random
from collections.abc import Awaitable, Callable

import openai

from core.logging import get_logger

log = get_logger(__name__)

MAX_RETRIES = 3
_BASE_DELAY_SECONDS = 1.0


def is_retryable(exc: Exception) -> bool:
    """429, 5xx, network failures and timeouts. Other 4xx and an exhausted quota won't fix themselves."""
    if isinstance(exc, openai.RateLimitError):
        return exc.code != "insufficient_quota"
    if isinstance(exc, openai.APIConnectionError):
        return True
    return isinstance(exc, openai.APIStatusError) and exc.status_code >= 500


async def with_retries[T](
    call: Callable[[], Awaitable[T]],
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Runs `call`, retrying retryable failures up to MAX_RETRIES times (exponential backoff, jitter)."""
    attempt = 0
    while True:
        try:
            return await call()
        except openai.OpenAIError as exc:
            if attempt >= MAX_RETRIES or not is_retryable(exc):
                raise
            delay = _BASE_DELAY_SECONDS * 2**attempt * (1 + random.random() * 0.5)  # noqa: S311
            attempt += 1
            log.warning(
                "model_call_retry", attempt=attempt, error_type=type(exc).__name__, delay_s=round(delay, 2)
            )
            await sleep(delay)
