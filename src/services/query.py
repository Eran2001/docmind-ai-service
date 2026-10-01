import re

from pydantic import BaseModel

from core.llm import ChatMessage, LlmClient
from core.usage import Usage
from prompts.rewrite import REWRITE_QUERY_PROMPT, build_rewrite_input
from prompts.title import TITLE_PROMPT
from services.history import HistoryMessage, prepare_history

# Rewrite and title use the fast model (spec 7.5).
FAST_MAX_TOKENS = 100
FAST_TEMPERATURE = 0.0
MAX_TITLE_WORDS = 6

_LABEL = re.compile(
    r"^(standalone search query|search query|rewritten query|query|title)\s*:\s*", re.IGNORECASE
)
_QUOTES = " \t\"'`\u201c\u201d\u2018\u2019"


class RewriteResult(BaseModel):
    query: str
    usage: Usage


class TitleResult(BaseModel):
    title: str
    usage: Usage


async def rewrite_query(
    llm: LlmClient, *, model: str, history: list[HistoryMessage], question: str
) -> RewriteResult:
    recent = prepare_history(history)
    if not recent:
        # Nothing to resolve: a first message is already standalone.
        usage = Usage(model=model, input_tokens=0, output_tokens=0, latency_ms=0)
        return RewriteResult(query=question, usage=usage)

    result = await llm.complete(
        [
            ChatMessage(role="system", content=REWRITE_QUERY_PROMPT),
            ChatMessage(role="user", content=build_rewrite_input(recent, question)),
        ],
        model=model,
        max_tokens=FAST_MAX_TOKENS,
        temperature=FAST_TEMPERATURE,
    )
    return RewriteResult(query=_first_line(result.text) or question, usage=result.usage)


async def make_title(llm: LlmClient, *, model: str, question: str) -> TitleResult:
    result = await llm.complete(
        [
            ChatMessage(role="system", content=TITLE_PROMPT),
            ChatMessage(role="user", content=question),
        ],
        model=model,
        max_tokens=FAST_MAX_TOKENS,
        temperature=FAST_TEMPERATURE,
    )
    words = (_first_line(result.text) or _first_line(question)).rstrip(".,:;!").split()
    return TitleResult(title=" ".join(words[:MAX_TITLE_WORDS]), usage=result.usage)


def _first_line(text: str) -> str:
    """The first non-empty line, without a leading label ("Title:") or surrounding quotes."""
    for line in text.splitlines():
        cleaned = _LABEL.sub("", line.strip()).strip(_QUOTES)
        if cleaned:
            return cleaned
    return ""
