from services.history import HistoryMessage, prepare_history, strip_citations, transcript


def msg(role: str, content: str) -> HistoryMessage:
    return HistoryMessage.model_validate({"role": role, "content": content})


def test_citation_markers_are_removed() -> None:
    assert strip_citations("Employees get 25 days [1]. Part-timers get less [2][3].") == (
        "Employees get 25 days. Part-timers get less."
    )


def test_numbers_in_brackets_that_are_not_citations_survive_only_as_digits_markers() -> None:
    assert strip_citations("See [a] and [12] here") == "See [a] and here"


def test_only_the_last_six_messages_are_kept() -> None:
    history = [msg("user" if i % 2 == 0 else "assistant", f"message {i}") for i in range(10)]

    kept = prepare_history(history)

    assert [m.content for m in kept] == [f"message {i}" for i in range(4, 10)]


def test_history_is_not_mutated() -> None:
    history = [msg("assistant", "Answer [1]")]

    prepare_history(history)

    assert history[0].content == "Answer [1]"


def test_transcript_labels_speakers() -> None:
    assert transcript([msg("user", "Hi"), msg("assistant", "Hello")]) == "User: Hi\nAssistant: Hello"
