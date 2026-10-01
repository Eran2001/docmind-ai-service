import trafilatura

from core.cleaning import ParsedDocument, ParsedPage
from core.errors import AppError, ErrorCode

_MAX_TITLE_CHARS = 200


def extract_page(content: bytes, url: str) -> ParsedDocument:
    """Main text of a web page as Markdown (headings kept), without menus, footers and ads."""
    text = trafilatura.extract(
        content,
        url=url,
        output_format="markdown",
        include_tables=True,
        include_links=False,
        favor_recall=True,
    )
    if not text or not text.strip():
        raise AppError(ErrorCode.EMPTY_DOCUMENT, "No readable text was found on that page.")

    metadata = trafilatura.extract_metadata(content, default_url=url)
    title = metadata.title if metadata and metadata.title else url
    return ParsedDocument(
        pages=[ParsedPage(page_number=None, text=text)],
        page_count=None,
        title=" ".join(title.split())[:_MAX_TITLE_CHARS],
    )
