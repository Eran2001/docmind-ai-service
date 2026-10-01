import re

from pydantic import BaseModel


class ParsedPage(BaseModel):
    """`page_number` is None for formats without pages (DOCX, TXT, MD, web pages)."""

    page_number: int | None
    text: str


class ParsedDocument(BaseModel):
    pages: list[ParsedPage]
    page_count: int | None = None
    title: str | None = None


# Headers and footers sit at the top or bottom of a page; looking only there keeps repeated body lines safe.
_EDGE_LINES = 3
_MIN_PAGES = 3
_REPEAT_SHARE = 0.5


def _line_key(line: str) -> str:
    # "Page 3 of 10" and "Page 4 of 10" must count as the same line.
    return re.sub(r"\d+", "#", " ".join(line.lower().split()))


def _edge_indices(lines: list[str]) -> list[int]:
    filled = [i for i, line in enumerate(lines) if line.strip()]
    return filled[:_EDGE_LINES] + [i for i in filled[-_EDGE_LINES:] if i not in filled[:_EDGE_LINES]]


def remove_repeated_lines(pages: list[ParsedPage]) -> list[ParsedPage]:
    """Drops lines that open or close more than half of the pages (running headers, footers, page numbers)."""
    if len(pages) < _MIN_PAGES:
        return pages

    all_lines = [page.text.splitlines() for page in pages]
    all_edges = [_edge_indices(lines) for lines in all_lines]
    counts: dict[str, int] = {}
    for lines, edges in zip(all_lines, all_edges, strict=True):
        for key in {_line_key(lines[i]) for i in edges}:
            counts[key] = counts.get(key, 0) + 1

    repeated = {key for key, n in counts.items() if n / len(pages) > _REPEAT_SHARE}
    if not repeated:
        return pages

    cleaned: list[ParsedPage] = []
    for page, lines, edges in zip(pages, all_lines, all_edges, strict=True):
        drop = {i for i in edges if _line_key(lines[i]) in repeated}
        kept = "\n".join(line for i, line in enumerate(lines) if i not in drop)
        cleaned.append(ParsedPage(page_number=page.page_number, text=kept))
    return cleaned
