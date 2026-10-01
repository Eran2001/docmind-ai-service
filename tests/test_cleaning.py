from core.cleaning import ParsedPage, remove_repeated_lines

BODIES = ["Revenue grew steadily.", "Costs fell sharply.", "Margins improved again.", "Hiring was paused."]


def page(number: int, *lines: str) -> ParsedPage:
    return ParsedPage(page_number=number, text="\n".join(lines))


def test_removes_running_header_and_numbered_footer() -> None:
    pages = [page(n, "ACME Handbook", BODIES[n - 1], f"Page {n} of 4") for n in range(1, 5)]

    cleaned = remove_repeated_lines(pages)

    assert [p.text for p in cleaned] == BODIES
    assert [p.page_number for p in cleaned] == [1, 2, 3, 4]


def test_keeps_blank_lines_between_paragraphs() -> None:
    pages = [page(n, "Header", BODIES[n - 1], "", f"Second {BODIES[n - 1]}") for n in range(1, 4)]

    assert remove_repeated_lines(pages)[0].text == f"{BODIES[0]}\n\nSecond {BODIES[0]}"


def test_short_documents_are_left_alone() -> None:
    pages = [page(1, "Title", "Body one."), page(2, "Title", "Body two.")]

    assert remove_repeated_lines(pages) == pages


def test_repeated_body_lines_in_the_middle_are_kept() -> None:
    body = [f"line {chr(97 + i)} is different" for i in range(10)]
    pages = [page(n, *body[:5], "Total", *body[5:]) for n in range(1, 5)]

    assert all("Total" in p.text for p in remove_repeated_lines(pages))


def test_line_on_fewer_than_half_of_pages_is_kept() -> None:
    pages = [
        page(1, "Unique alpha", "first body"),
        page(2, "Other beta", "second body"),
        page(3, "Another gamma", "third body"),
        page(4, "Last delta", "fourth body"),
    ]

    assert remove_repeated_lines(pages) == pages
