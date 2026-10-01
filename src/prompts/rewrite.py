from services.history import HistoryMessage, transcript

REWRITE_QUERY_PROMPT = (
    "Rewrite the user's latest message into a standalone search query that makes sense "
    "without the conversation. "
    "Keep names, numbers, and technical terms. Output ONLY the rewritten query, nothing else."
)


def build_rewrite_input(history: list[HistoryMessage], question: str) -> str:
    # A transcript, not chat turns: small models tend to answer the question instead of rewriting it.
    return (
        f"<conversation>\n{transcript(history)}\n</conversation>\n\n"
        f"Latest message: {question}\n\n"
        "Standalone search query:"
    )
