import logging
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

_SENSITIVE = ("key", "secret", "token", "password", "authorization", "cookie")


def _redact(_: Any, __: str, event_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    for name in event_dict:
        if any(part in name.lower() for part in _SENSITIVE):
            event_dict[name] = "[redacted]"
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    shared: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _redact,
    ]
    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # Our middleware logs each request; uvicorn's own access log would duplicate it.
    logging.getLogger("uvicorn.access").disabled = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.stdlib.get_logger(name)
