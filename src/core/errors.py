from enum import StrEnum

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from core.logging import get_logger

log = get_logger(__name__)


class ErrorCode(StrEnum):
    PARSE_FAILED = "PARSE_FAILED"
    EMPTY_DOCUMENT = "EMPTY_DOCUMENT"
    URL_FETCH_FAILED = "URL_FETCH_FAILED"
    URL_BLOCKED = "URL_BLOCKED"
    LLM_ERROR = "LLM_ERROR"
    # Not in the spec's list, but every route needs a code for these cases.
    UNAUTHORIZED = "UNAUTHORIZED"
    INVALID_REQUEST = "INVALID_REQUEST"
    NOT_FOUND = "NOT_FOUND"
    INTERNAL_ERROR = "INTERNAL_ERROR"


_STATUS: dict[ErrorCode, int] = {
    ErrorCode.PARSE_FAILED: status.HTTP_422_UNPROCESSABLE_CONTENT,
    ErrorCode.EMPTY_DOCUMENT: status.HTTP_422_UNPROCESSABLE_CONTENT,
    ErrorCode.URL_FETCH_FAILED: status.HTTP_502_BAD_GATEWAY,
    ErrorCode.URL_BLOCKED: status.HTTP_400_BAD_REQUEST,
    ErrorCode.LLM_ERROR: status.HTTP_502_BAD_GATEWAY,
    ErrorCode.UNAUTHORIZED: status.HTTP_401_UNAUTHORIZED,
    ErrorCode.INVALID_REQUEST: status.HTTP_422_UNPROCESSABLE_CONTENT,
    ErrorCode.NOT_FOUND: status.HTTP_404_NOT_FOUND,
    ErrorCode.INTERNAL_ERROR: status.HTTP_500_INTERNAL_SERVER_ERROR,
}


class ErrorBody(BaseModel):
    code: ErrorCode
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class AppError(Exception):
    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _response(code: ErrorCode, message: str) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message))
    return JSONResponse(status_code=_STATUS[code], content=body.model_dump(mode="json"))


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return _response(exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Field paths only: the offending input may be document text and must not be echoed or logged.
        fields = ", ".join(".".join(str(p) for p in e["loc"]) for e in exc.errors())
        return _response(ErrorCode.INVALID_REQUEST, f"Invalid request: {fields}.")

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = (
            ErrorCode.NOT_FOUND if exc.status_code == status.HTTP_404_NOT_FOUND else ErrorCode.INVALID_REQUEST
        )
        return _response(code, "Not found." if code is ErrorCode.NOT_FOUND else "Request not allowed.")

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled_error", error_type=type(exc).__name__, exc_info=True)
        return _response(ErrorCode.INTERNAL_ERROR, "Unexpected error in the AI service.")
