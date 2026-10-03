import json
import re
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from core.errors import AppError, ErrorCode
from core.llm import ChatMessage, LlmClient
from core.logging import get_logger
from core.usage import Usage
from prompts.rerank import RERANK_SYSTEM_PROMPT, build_rerank_input

log = get_logger(__name__)

MAX_PASSAGES = 30
# About a whole 500-token chunk. Cutting passages shorter hid the answers deeper inside them
# (docs benchmark, top-8 hit: 79% with 800 characters, 97% with the whole passage).
PASSAGE_CHARS = 2600
RERANK_MAX_TOKENS = 200

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class Passage(BaseModel):
    id: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    text: Annotated[str, StringConstraints(min_length=1, max_length=20_000)]


class RerankInput(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
    passages: list[Passage] = Field(min_length=1, max_length=MAX_PASSAGES)
    top_n: int = Field(default=8, ge=1, le=MAX_PASSAGES)


class RerankResult(BaseModel):
    """Passage ids, best first: the model's picks, then the rest in their original order."""

    ids: list[str]
    usage: Usage


async def rerank(llm: LlmClient, *, model: str, request: RerankInput) -> RerankResult:
    passages = request.passages
    top_n = min(request.top_n, len(passages))
    result = await llm.complete(
        [
            ChatMessage(role="system", content=RERANK_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=build_rerank_input(request.question, [_shorten(p.text) for p in passages], top_n),
            ),
        ],
        model=model,
        max_tokens=RERANK_MAX_TOKENS,
        temperature=0.0,
    )
    picked = _parse(result.text, len(passages))
    # Whatever the model left out keeps its original (search) order after the picks.
    order = [*picked, *(i for i in range(len(passages)) if i not in set(picked))]
    return RerankResult(ids=[passages[i].id for i in order[:top_n]], usage=result.usage)


def _shorten(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= PASSAGE_CHARS else flat[:PASSAGE_CHARS] + "…"


def _parse(text: str, count: int) -> list[int]:
    """Zero-based indexes from the model's `{"ranking": [3, 1, ...]}`, valid and without repeats."""
    match = _JSON_OBJECT.search(text)
    try:
        ranking = json.loads(match.group(0))["ranking"] if match else None
    except (ValueError, KeyError, TypeError):
        ranking = None
    if not isinstance(ranking, list):
        log.error("rerank_reply_unparseable")
        raise AppError(ErrorCode.LLM_ERROR, "The rerank model returned an unreadable ranking.")
    seen: list[int] = []
    for item in ranking:
        if isinstance(item, bool) or not isinstance(item, int | str):
            continue
        try:
            number = int(item)
        except ValueError:
            continue
        if 1 <= number <= count and number - 1 not in seen:
            seen.append(number - 1)
    return seen
