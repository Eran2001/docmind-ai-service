import re
from typing import Literal

from pydantic import BaseModel, Field

MAX_HISTORY_MESSAGES = 6

_CITATION = re.compile(r"\s*\[\d+\]")


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20_000)


def strip_citations(text: str) -> str:
    return _CITATION.sub("", text).strip()


def prepare_history(history: list[HistoryMessage]) -> list[HistoryMessage]:
    """The last 6 messages with citation markers like [1] removed, so the model won't copy them."""
    recent = history[-MAX_HISTORY_MESSAGES:]
    return [m.model_copy(update={"content": strip_citations(m.content)}) for m in recent]


def transcript(history: list[HistoryMessage]) -> str:
    return "\n".join(f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}" for m in history)
