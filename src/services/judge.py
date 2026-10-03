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

# "I couldn't find that in your documents", "not in the provided excerpts", "no information about ..."
_NOT_FOUND = re.compile(
    r"(?:couldn'?t|could not|can'?t|cannot|can not|didn'?t|did not|unable to|not able to) (?:find|locate)"
    r"|not (?:in|present in|contained in|found in|mentioned in|covered in)"
    r" (?:the |your )?(?:provided |given )?(?:documents?|excerpts?|sources?)"
    r"|no (?:relevant )?information (?:about|on|in)",
    re.IGNORECASE,
)


def says_not_found(text: str) -> bool:
    """Any sentence says the information isn't there. Used on the expected answer (we write those)."""
    return bool(_NOT_FOUND.search(text))


_SUGGESTION = re.compile(r"^try asking", re.IGNORECASE)
_SENTENCES = re.compile(r"(?<=[.!?])\s+")


def is_only_not_found(text: str) -> bool:
    """The answer is nothing but "I couldn't find that" (optionally followed by "Try asking about ...").

    An answer that says it couldn't find something and then gives one anyway is not "only not found".
    """
    cleaned = re.sub(r"\[\d+\]", "", text).replace("*", "").replace("_", "").strip()
    sentences = [s.strip() for s in _SENTENCES.split(cleaned) if s.strip()]
    if not sentences:
        return False
    return any(_NOT_FOUND.search(s) for s in sentences) and all(
        _NOT_FOUND.search(s) or _SUGGESTION.match(s) for s in sentences
    )


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
    expected_not_found = says_not_found(request.expected)
    generated_not_found = is_only_not_found(request.generated)
    free = Usage(model=model, input_tokens=0, output_tokens=0, latency_ms=0)
    # "Not in the documents" cases are decided by rule: small judge models are unreliable on them.
    if expected_not_found and generated_not_found:
        return JudgeResult(
            correctness=1.0,
            faithfulness=1.0,
            reasoning="Both say the answer is not in the documents.",
            usage=free,
        )
    if generated_not_found and not expected_not_found:
        return JudgeResult(
            correctness=0.0,
            faithfulness=1.0,
            reasoning="The answer says it could not find the information, but the expected answer has it.",
            usage=free,
        )

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
    if expected_not_found:
        # The expected answer is "it isn't there", yet the answer states something: wrong however it reads.
        scores["correctness"] = 0.0
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
