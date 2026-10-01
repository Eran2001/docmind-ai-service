import re
from dataclasses import dataclass, replace
from functools import cache

import tiktoken
from pydantic import BaseModel

from core.cleaning import ParsedDocument

TARGET_TOKENS = 500
OVERLAP_TOKENS = 80
MIN_TOKENS = 20

_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
# A unit must leave room for the 1-token join estimate, so it never exceeds a chunk on its own.
_MAX_UNIT_TOKENS = TARGET_TOKENS - 1


class Chunk(BaseModel):
    index: int
    content: str
    page_number: int | None
    heading: str | None
    token_count: int


@dataclass(frozen=True)
class _Unit:
    text: str
    tokens: int
    page: int | None
    heading: str | None
    new_paragraph: bool


@cache
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    # encode_ordinary: document text may contain strings like "<|endoftext|>" that encode() would reject.
    return len(_encoder().encode_ordinary(text))


def chunk_document(document: ParsedDocument) -> list[Chunk]:
    """Splits a document into ~500-token chunks (80-token overlap), keeping the start page and heading."""
    units = _units(document)
    chunks: list[Chunk] = []
    current: list[_Unit] = []
    fresh = 0  # units in `current` that are not overlap carried from the previous chunk

    for unit in units:
        if current and _estimate(current) + unit.tokens + 1 > TARGET_TOKENS:
            chunks.append(_build(current))
            current, fresh = _overlap(current), 0
            while current and _estimate(current) + unit.tokens + 1 > TARGET_TOKENS:
                current.pop(0)
        current.append(unit)
        fresh += 1
    if fresh:
        chunks.append(_build(current))

    kept = [c for c in chunks if c.token_count >= MIN_TOKENS]
    return [c.model_copy(update={"index": i}) for i, c in enumerate(kept)]


def _estimate(units: list[_Unit]) -> int:
    # One extra token per unit covers the paragraph breaks between them.
    return sum(u.tokens + 1 for u in units)


def _build(units: list[_Unit]) -> Chunk:
    content = units[0].text
    for unit in units[1:]:
        content += ("\n\n" if unit.new_paragraph else " ") + unit.text
    return Chunk(
        index=0,
        content=content,
        page_number=units[0].page,
        heading=units[0].heading,
        token_count=count_tokens(content),
    )


def _overlap(units: list[_Unit]) -> list[_Unit]:
    """The last sentences of a chunk, up to OVERLAP_TOKENS, that open the next chunk."""
    tail: list[_Unit] = []
    total = 0
    for unit in reversed(units):
        for sentence in reversed(_sentences(unit)):
            if total + sentence.tokens + 1 > OVERLAP_TOKENS:
                return tail[::-1]
            tail.append(sentence)
            total += sentence.tokens + 1
    return tail[::-1]


def _sentences(unit: _Unit) -> list[_Unit]:
    parts = _SENTENCE_END.split(unit.text)
    if len(parts) == 1:
        return [unit]
    return [
        replace(unit, text=part, tokens=count_tokens(part), new_paragraph=unit.new_paragraph and i == 0)
        for i, part in enumerate(parts)
    ]


def _units(document: ParsedDocument) -> list[_Unit]:
    units: list[_Unit] = []
    heading: str | None = None
    for page in document.pages:
        for text, is_heading in _paragraphs(page.text):
            if is_heading:
                heading = text
            units.extend(_split(text, page.page_number, heading))
    return units


def _paragraphs(page_text: str) -> list[tuple[str, bool]]:
    """(text, is_heading) per paragraph: blank lines split paragraphs; a heading line is its own paragraph."""
    result: list[tuple[str, bool]] = []
    lines: list[str] = []

    def flush() -> None:
        text = " ".join(" ".join(lines).split())
        if text:
            result.append((text, False))
        lines.clear()

    for line in page_text.splitlines():
        match = _HEADING.match(line.strip())
        if match:
            flush()
            result.append((" ".join(match.group(1).split()), True))
        elif line.strip():
            lines.append(line)
        else:
            flush()
    flush()
    return result


def _split(text: str, page: int | None, heading: str | None) -> list[_Unit]:
    """A paragraph, or its sentences when it is too long, or its words when a sentence is too long."""
    tokens = count_tokens(text)
    if tokens <= _MAX_UNIT_TOKENS:
        return [_Unit(text, tokens, page, heading, new_paragraph=True)]

    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        if count_tokens(sentence) <= _MAX_UNIT_TOKENS:
            pieces.append(sentence)
        else:
            pieces.extend(_split_words(sentence))
    return [
        _Unit(piece, count_tokens(piece), page, heading, new_paragraph=i == 0)
        for i, piece in enumerate(pieces)
    ]


def _split_words(sentence: str) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    for word in sentence.split():
        if count_tokens(word) > _MAX_UNIT_TOKENS:  # no spaces to cut at (e.g. a long unbroken string)
            if current:
                pieces.append(" ".join(current))
                current = []
            pieces.extend(_hard_split(word))
        elif current and count_tokens(" ".join([*current, word])) > _MAX_UNIT_TOKENS:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _hard_split(word: str) -> list[str]:
    ids = _encoder().encode_ordinary(word)
    return [_encoder().decode(ids[i : i + _MAX_UNIT_TOKENS]) for i in range(0, len(ids), _MAX_UNIT_TOKENS)]
