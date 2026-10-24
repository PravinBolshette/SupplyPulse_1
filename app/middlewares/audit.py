"""
Request Audit Logging Middleware.

Calculates request processing latency, captures route metadata, client IP,
and response status codes, logging structured audit records for every interaction.
"""

import time
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import get_logger

logger = get_logger("audit")


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to record structured audit trails for all HTTP traffic.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start_time = time.perf_counter()

        # Extract client IP, prioritizing proxy headers if present
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        elif request.client:
            client_ip = request.client.host
        else:
            client_ip = "unknown"

        method = request.method
        path = request.url.path
        query = request.url.query
        full_path = f"{path}?{query}" if query else path

        logger.info(
            f"--> Incoming request: {method} {full_path} from {client_ip}"
        )

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            # Attach performance header
            response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"

            log_msg = (
                f"<-- Completed request: {method} {full_path} - "
                f"Status: {response.status_code} - Latency: {duration_ms:.2f}ms"
            )

            if response.status_code >= 500:
                logger.error(log_msg)
            elif response.status_code >= 400:
                logger.warning(log_msg)
            else:
                logger.info(log_msg)

            return response

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                f"<-- Failed request: {method} {full_path} - "
                f"Exception: {type(exc).__name__} - Latency: {duration_ms:.2f}ms"
            )
            raise exc
