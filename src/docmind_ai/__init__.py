import uvicorn

from core.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "docmind_ai.app:create_app",
        factory=True,
        host=settings.ai_host,
        port=settings.ai_port,
        log_config=None,
    )
