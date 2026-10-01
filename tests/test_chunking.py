from core.chunking import MIN_TOKENS, OVERLAP_TOKENS, TARGET_TOKENS, Chunk, chunk_document, count_tokens
from core.cleaning import ParsedDocument, ParsedPage


def doc(*texts: str) -> ParsedDocument:
    pages = [ParsedPage(page_number=i, text=t) for i, t in enumerate(texts, start=1)]
    return ParsedDocument(pages=pages, page_count=len(pages))


def text_doc(text: str) -> ParsedDocument:
    return ParsedDocument(pages=[ParsedPage(page_number=None, text=text)])


def one_sentence(tokens: int, tag: str) -> str:
    """A single sentence of at least `tokens` tokens (no sentence ends before the final period)."""
    words = [tag]
    while count_tokens(" ".join(words)) < tokens:
        words.append("lorem")
    return " ".join(words) + "."


def sentences(count: int) -> str:
    return " ".join(f"Sentence number {i} covers topic {i} in some detail." for i in range(count))


def overlap_between(previous: Chunk, following: Chunk) -> str:
    first_sentence = following.content.split(". ")[0]
    start = previous.content.rfind(first_sentence)
    assert start != -1, "next chunk does not start with text from the previous chunk"
    return previous.content[start:]


def test_chunks_respect_token_limits_and_cover_the_text() -> None:
    text = sentences(300)

    chunks = chunk_document(text_doc(text))

    assert len(chunks) > 3
    assert all(c.token_count <= TARGET_TOKENS for c in chunks)
    assert all(c.token_count == count_tokens(c.content) for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    for i in range(300):
        assert any(f"Sentence number {i} covers" in c.content for c in chunks)


def test_consecutive_chunks_overlap_by_at_most_80_tokens() -> None:
    chunks = chunk_document(text_doc(sentences(300)))

    for previous, following in zip(chunks, chunks[1:], strict=False):
        shared = overlap_between(previous, following)
        assert 0 < count_tokens(shared) <= OVERLAP_TOKENS
        assert following.content.startswith(shared.split(". ")[0])


def test_paragraphs_are_kept_whole_when_they_fit() -> None:
    first = sentences(36)
    second = " ".join(f"Another point {i} about something else entirely." for i in range(30))
    assert 400 < count_tokens(first) < TARGET_TOKENS and 250 < count_tokens(second) < TARGET_TOKENS

    chunks = chunk_document(text_doc(f"{first}\n\n{second}"))

    assert len(chunks) == 2
    assert "Another point" not in chunks[0].content
    assert chunks[1].content.endswith(second)


def test_page_number_is_the_page_where_the_chunk_starts() -> None:
    chunks = chunk_document(doc(one_sentence(300, "first"), one_sentence(300, "second")))

    assert [c.page_number for c in chunks] == [1, 2]


def test_non_paged_documents_have_no_page_number() -> None:
    assert chunk_document(text_doc(sentences(40)))[0].page_number is None


def test_heading_is_tracked_and_hashes_removed_from_content() -> None:
    text = f"# Intro\n\n{one_sentence(300, 'alpha')}\n\n## Details\n\n{one_sentence(300, 'beta')}"

    chunks = chunk_document(text_doc(text))

    assert [c.heading for c in chunks] == ["Intro", "Details"]
    assert all("#" not in c.content for c in chunks)
    assert chunks[0].content.startswith("Intro\n\nalpha")


def test_chunks_under_20_tokens_are_dropped() -> None:
    assert chunk_document(text_doc("Too short to keep.")) == []
    assert MIN_TOKENS == 20


def test_long_sentences_are_split_on_word_boundaries() -> None:
    words = [f"word{i}" for i in range(2000)]

    chunks = chunk_document(text_doc(" ".join(words) + "."))

    assert len(chunks) > 2
    assert all(c.token_count <= TARGET_TOKENS for c in chunks)
    known = {*words, "word1999."}
    assert all(w in known for c in chunks for w in c.content.split())
    assert {w for c in chunks for w in c.content.split()} >= set(words[:1999])


def test_a_word_with_no_spaces_is_still_split() -> None:
    chunks = chunk_document(text_doc("x" * 6000))

    assert len(chunks) > 1
    assert all(c.token_count <= TARGET_TOKENS for c in chunks)


def test_special_token_text_in_a_document_does_not_break_counting() -> None:
    chunks = chunk_document(text_doc("<|endoftext|> " + sentences(10)))

    assert chunks
    assert "<|endoftext|>" in chunks[0].content


def test_whitespace_is_normalized() -> None:
    text = (
        "Line one of the paragraph\nwraps   onto the next line "
        "and keeps going for a while longer than twenty tokens."
    )

    chunk = chunk_document(text_doc(text))[0]

    assert chunk.content == " ".join(text.split())
