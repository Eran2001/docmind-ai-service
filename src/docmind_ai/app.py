import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import FastAPI, Request, Response

from api.router import api_router
from core.config import get_settings
from core.errors import register_error_handlers
from core.logging import configure_logging, get_logger

log = get_logger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()  # exits with a clear message when the environment is incomplete
    configure_logging(settings.log_level)

    app = FastAPI(title="DocMind AI Service", docs_url=None, redoc_url=None, openapi_url=None)
    register_error_handlers(app)
    app.include_router(api_router)

    @app.middleware("http")
    async def _log_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        log.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round((time.perf_counter() - started) * 1000),
        )
        return response

    return app
