import json
import re
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from core.errors import AppError, ErrorCode
from core.llm import ChatMessage, LlmClient
from core.logging import get_logger
from core.usage import Usage
from prompts.judge import JUDGE_SYSTEM_PROMPT, build_judge_input
from services.answer import AnswerChunk

log = get_logger(__name__)

JUDGE_MAX_TOKENS = 300
JUDGE_TEMPERATURE = 0.0
MAX_JUDGE_CHUNKS = 20  # the API's top-k can be up to 20

type Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class JudgeInput(BaseModel):
    question: Text
    expected: Text
    generated: Annotated[str, StringConstraints(strip_whitespace=True, max_length=8000)]
    chunks: list[AnswerChunk] = Field(max_length=MAX_JUDGE_CHUNKS)


class JudgeResult(BaseModel):
    correctness: float
    faithfulness: float
    reasoning: str
    usage: Usage


async def judge_answer(llm: LlmClient, *, model: str, request: JudgeInput) -> JudgeResult:
    result = await llm.complete(
        [
            ChatMessage(role="system", content=JUDGE_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=build_judge_input(
                    request.question, request.expected, request.generated, request.chunks
                ),
            ),
        ],
        model=model,
        max_tokens=JUDGE_MAX_TOKENS,
        temperature=JUDGE_TEMPERATURE,
    )
    scores = _parse(result.text)
    return JudgeResult(**scores, usage=result.usage)


def _parse(text: str) -> dict[str, float | str]:
    """Pulls the JSON object out of the reply (models sometimes wrap it in a code fence)."""
    match = _JSON_OBJECT.search(text)
    try:
        data = json.loads(match.group(0)) if match else {}
        return {
            "correctness": _score(data["correctness"]),
            "faithfulness": _score(data["faithfulness"]),
            "reasoning": str(data.get("reasoning", "")).strip(),
        }
    except (ValueError, KeyError, TypeError, AttributeError):
        log.error("judge_reply_unparseable")
        raise AppError(ErrorCode.LLM_ERROR, "The judge model returned an unreadable score.") from None


def _score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError("score is not a number")
    return min(1.0, max(0.0, float(value)))
