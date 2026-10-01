from pydantic import BaseModel


class Usage(BaseModel):
    """What one model call cost, so the API can record it."""

    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
