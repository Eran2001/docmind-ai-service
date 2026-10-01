import pytest

from core.errors import AppError, ErrorCode
from core.parsing import DOCX, MARKDOWN, PDF, TEXT, parse_file
from tests.builders import make_docx, make_pdf


def test_pdf_is_parsed_page_by_page_with_headings() -> None:
    data = make_pdf(
        [
            [("Annual Report", 22), ("Revenue grew steadily during the year.", 11)],
            [("Second page text goes here.", 11)],
        ]
    )

    doc = parse_file(data, PDF)

    assert doc.page_count == 2
    assert [p.page_number for p in doc.pages] == [1, 2]
    assert doc.pages[0].text.startswith("# Annual Report")
    assert "Revenue grew steadily" in doc.pages[0].text
    assert "# " not in doc.pages[1].text


def test_blank_pdf_is_empty_document() -> None:
    with pytest.raises(AppError) as exc:
        parse_file(make_pdf([[]]), PDF)
    assert exc.value.code is ErrorCode.EMPTY_DOCUMENT


def test_corrupt_pdf_fails_to_parse() -> None:
    with pytest.raises(AppError) as exc:
        parse_file(b"this is not a pdf", PDF)
    assert exc.value.code is ErrorCode.PARSE_FAILED


def test_password_protected_pdf_fails_to_parse() -> None:
    with pytest.raises(AppError) as exc:
        parse_file(make_pdf([[("secret", 11)]], password="pw"), PDF)
    assert exc.value.code is ErrorCode.PARSE_FAILED
    assert "password" in exc.value.message


def test_docx_keeps_headings_paragraphs_and_tables_in_order() -> None:
    doc = parse_file(make_docx(), DOCX)

    assert doc.page_count is None
    assert doc.pages[0].page_number is None
    assert doc.pages[0].text == (
        "# Leave policy\n\n"
        "Employees get 25 days of paid leave.\n\n"
        "Tier | Days\n\nSenior | 30\n\n"
        "Ask HR for details."
    )


def test_corrupt_docx_fails_to_parse() -> None:
    with pytest.raises(AppError) as exc:
        parse_file(b"not a zip", DOCX)
    assert exc.value.code is ErrorCode.PARSE_FAILED


@pytest.mark.parametrize("mime", [TEXT, MARKDOWN])
def test_text_files_are_decoded_and_bom_is_dropped(mime: str) -> None:
    doc = parse_file("\ufeff# Title\n\nHéllo wörld".encode(), mime)

    assert doc.pages[0].text == "# Title\n\nHéllo wörld"
    assert doc.pages[0].page_number is None


def test_invalid_utf8_text_fails_to_parse() -> None:
    with pytest.raises(AppError) as exc:
        parse_file(b"\xff\xfe\x00bad", TEXT)
    assert exc.value.code is ErrorCode.PARSE_FAILED


def test_whitespace_only_text_is_empty_document() -> None:
    with pytest.raises(AppError) as exc:
        parse_file(b"  \n\n \t ", TEXT)
    assert exc.value.code is ErrorCode.EMPTY_DOCUMENT


def test_unsupported_type() -> None:
    with pytest.raises(AppError) as exc:
        parse_file(b"x", "image/png")
    assert exc.value.code is ErrorCode.PARSE_FAILED
