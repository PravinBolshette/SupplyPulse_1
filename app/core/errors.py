"""
Global Exception Hierarchy and Centralized HTTP Error Handlers.

Enforces a standard unified error response envelope:
{
    "success": false,
    "error": {
        "code": "ERROR_CODE",
        "message": "Human readable description",
        "details": [...]
    }
}
"""

import traceback
from typing import Any, Optional
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger

logger = get_logger("errors")


# ---------------------------------------------------------------------------
# Custom Domain Exception Hierarchy
# ---------------------------------------------------------------------------

class AppCustomException(Exception):
    """
    Base domain exception for all business logic failures.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "BUSINESS_LOGIC_ERROR",
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: Optional[list[dict[str, Any]]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or []


class EntityNotFoundException(AppCustomException):
    """Raised when a requested resource does not exist."""

    def __init__(
        self,
        message: str = "Requested resource not found",
        entity_name: Optional[str] = None,
        entity_id: Optional[str] = None,
    ):
        details = []
        if entity_name and entity_id:
            details.append({"entity": entity_name, "id": str(entity_id)})
        super().__init__(
            message=message,
            error_code="ENTITY_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            details=details,
        )


class EntityConflictException(AppCustomException):
    """Raised when creating an entity with a duplicate unique key."""

    def __init__(self, message: str = "Resource conflict detected"):
        super().__init__(
            message=message,
            error_code="ENTITY_CONFLICT",
            status_code=status.HTTP_409_CONFLICT,
        )


class LeadTimeBreachException(AppCustomException):
    """Raised when an operation would violate supplier replenishment lead time constraints."""

    def __init__(
        self,
        message: str = "Lead time constraint breached: TTI is insufficient for replenishment",
        sku: Optional[str] = None,
        tti_days: Optional[float] = None,
        lead_time_days: Optional[float] = None,
    ):
        details = []
        if sku:
            details.append({
                "sku": sku,
                "predicted_tti_days": tti_days,
                "lead_time_days": lead_time_days,
            })
        super().__init__(
            message=message,
            error_code="LEAD_TIME_BREACH",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details,
        )


class InvalidStateTransitionException(AppCustomException):
    """Raised when an entity attempts an illegal state lifecycle change."""

    def __init__(self, message: str = "Invalid state transition requested"):
        super().__init__(
            message=message,
            error_code="INVALID_STATE_TRANSITION",
            status_code=status.HTTP_400_BAD_REQUEST,
        )


# ---------------------------------------------------------------------------
# Centralized HTTP & Validation Exception Handlers
# ---------------------------------------------------------------------------

async def app_custom_exception_handler(
    request: Request, exc: AppCustomException
) -> JSONResponse:
    """
    Handles all application-specific domain exceptions.
    """
    logger.warning(
        f"Domain exception on {request.method} {request.url.path}: "
        f"[{exc.error_code}] {exc.message}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "details": exc.details if exc.details else None,
            },
        },
    )


async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Sanitizes Pydantic v2 validation errors into clean field-level details.
    """
    cleaned_errors: list[dict[str, Any]] = []
    for err in exc.errors():
        loc = err.get("loc", ())
        # Strip internal 'body' or query markers for readable field paths
        field_path = ".".join(str(item) for item in loc if item not in ("body",))
        cleaned_errors.append({
            "field": field_path or "payload",
            "message": err.get("msg", "Validation error"),
            "type": err.get("type", "value_error"),
        })

    logger.warning(
        f"Validation error on {request.method} {request.url.path}: {len(cleaned_errors)} invalid fields"
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Input validation failed. Please check the request parameters.",
                "details": cleaned_errors,
            },
        },
    )


async def starlette_http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """
    Standard Starlette/FastAPI HTTP status code handler (404, 405, 401, 403, etc.).
    """
    status_code_map = {
        404: ("RESOURCE_NOT_FOUND", "The requested endpoint or resource was not found."),
        405: ("METHOD_NOT_ALLOWED", f"Method '{request.method}' not allowed on this route."),
        401: ("UNAUTHORIZED", "Authentication credentials missing or invalid."),
        403: ("FORBIDDEN", "Insufficient permissions to access this resource."),
    }

    code, default_msg = status_code_map.get(
        exc.status_code, (f"HTTP_{exc.status_code}", exc.detail or "HTTP error occurred.")
    )
    message = str(exc.detail) if exc.detail else default_msg

    logger.warning(f"HTTP {exc.status_code} on {request.method} {request.url.path}: {message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": code,
                "message": message,
                "details": None,
            },
        },
    )


async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """
    Catch-all exception handler that masks sensitive internal tracebacks
    while logging complete stack trace for troubleshooting.
    """
    tb = traceback.format_exc()
    logger.error(
        f"Unhandled 500 error on {request.method} {request.url.path}: {str(exc)}\n{tb}"
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected internal server error occurred. Please contact system administrator.",
                "details": None,
            },
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """
    Registers all centralized exception handlers to the FastAPI application instance.
    """
    app.add_exception_handler(AppCustomException, app_custom_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, request_validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, starlette_http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)
