"""
Correlation ID Propagation Middleware.

Ensures every incoming HTTP request is assigned a unique correlation ID (UUIDv4)
or preserves an existing incoming 'X-Request-ID' header. Propagates the ID to:
1. Starlette request.state.correlation_id
2. Async contextvar for logging context injection
3. Outgoing HTTP response header 'X-Request-ID'
"""

import uuid
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import correlation_id_ctx

CORRELATION_HEADER_NAME = "X-Request-ID"


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware that attaches and tracks a correlation ID across the request lifecycle.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Extract existing header or generate a new UUIDv4
        header_val = request.headers.get(CORRELATION_HEADER_NAME)
        correlation_id = header_val.strip() if header_val and header_val.strip() else str(uuid.uuid4())

        # Bind to request state
        request.state.correlation_id = correlation_id

        # Bind to async task contextvar
        token = correlation_id_ctx.set(correlation_id)

        try:
            response = await call_next(request)
            # Propagate back to client in response header
            response.headers[CORRELATION_HEADER_NAME] = correlation_id
            return response
        finally:
            correlation_id_ctx.reset(token)
