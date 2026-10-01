import pytest

from core.errors import AppError, ErrorCode
from core.web import extract_page


def _paragraphs() -> str:
    # Very short pages make trafilatura fall back to a plainer extractor that drops headings.
    return " ".join(
        f"Acme offers benefit number {i}, which covers every employee from day one." for i in range(8)
    )


PAGE = f"""<html><head><title>Benefits Guide | Acme</title></head><body>
<nav><a href="/">Home</a><a href="/jobs">Jobs</a></nav>
<article><h1>Employee Benefits</h1>
<p>{_paragraphs()}</p>
<h2>Health</h2>
<p>{_paragraphs()}</p>
</article><footer>Copyright Acme 2026 all rights reserved</footer></body></html>
""".encode()


def test_extracts_main_text_with_markdown_headings() -> None:
    doc = extract_page(PAGE, "https://acme.test/benefits")

    text = doc.pages[0].text
    assert "# Employee Benefits" in text
    assert "## Health" in text
    assert "Acme offers benefit number 3" in text
    assert "Copyright" not in text
    assert doc.pages[0].page_number is None and doc.page_count is None
    assert doc.title


def test_page_without_text_is_empty_document() -> None:
    page = b"<html><head><title>x</title><script>var a = 1;</script></head><body></body></html>"

    with pytest.raises(AppError) as exc:
        extract_page(page, "https://acme.test/")

    assert exc.value.code is ErrorCode.EMPTY_DOCUMENT


def test_title_is_trimmed_to_200_characters() -> None:
    long_title = "word " * 100
    body = "Some body text here. " * 20
    page = f"<html><head><title>{long_title}</title></head><body><p>{body}</p></body></html>"

    assert len(extract_page(page.encode(), "https://acme.test/").title or "") <= 200
