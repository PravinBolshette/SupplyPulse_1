"""
SupplyPulse FastAPI Application Factory & Main Entrypoint.

Initializes the FastAPI application with lifespan management, CORS configuration,
custom audit and correlation ID middlewares, centralized exception handling,
and modular RESTful routers.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.middlewares.audit import AuditLoggingMiddleware
from app.middlewares.correlation import CorrelationIdMiddleware
from app.routers import alerts, anomalies, auth, events, mitigation, model, suppliers


logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager for startup and graceful shutdown.
    """
    settings = get_settings()
    setup_logging(settings)
    logger.info(
        f"Booting {settings.PROJECT_NAME} v{settings.VERSION} "
        f"[env={settings.ENVIRONMENT}, debug={settings.DEBUG}]"
    )

    yield

    logger.info(f"Shutting down {settings.PROJECT_NAME}... cleaning up resources.")


def create_application() -> FastAPI:
    """
    Application factory building and configuring the FastAPI instance.
    """
    settings: Settings = get_settings()

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.DESCRIPTION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url=f"{settings.API_V1_STR}/docs",
        redoc_url=f"{settings.API_V1_STR}/redoc",
        lifespan=lifespan,
    )

    # 1. Register Global Exception Handlers
    register_exception_handlers(app)

    # 2. Register Middleware Stack (Order: Correlation -> Audit -> CORS)
    # In Starlette, last added executes first on incoming request.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
    )
    app.add_middleware(AuditLoggingMiddleware)
    app.add_middleware(CorrelationIdMiddleware)

    # 3. Mount Domain Routers
    app.include_router(
        auth.router,
        prefix=f"{settings.API_V1_STR}/auth",
        tags=["Authentication & Roles"],
    )
    app.include_router(
        suppliers.router,
        prefix=f"{settings.API_V1_STR}/suppliers",
        tags=["Suppliers"],
    )
    app.include_router(
        anomalies.router,
        prefix=f"{settings.API_V1_STR}/anomalies",
        tags=["Anomalies"],
    )
    app.include_router(
        alerts.router,
        prefix=f"{settings.API_V1_STR}/alerts",
        tags=["Disruption Alerts"],
    )
    app.include_router(
        mitigation.router,
        prefix=f"{settings.API_V1_STR}/mitigation",
        tags=["Responsible AI & Mitigation"],
    )
    app.include_router(
        model.router,
        prefix=f"{settings.API_V1_STR}/model",
        tags=["Model Lineage & MLOps"],
    )
    app.include_router(
        events.router,
        prefix=f"{settings.API_V1_STR}/events",
        tags=["External Disruption Events"],
    )



    # 4. Top-Level Health & Root Endpoints
    @app.get(
        "/health",
        tags=["System"],
        summary="Service Health Check",
        status_code=status.HTTP_200_OK,
    )
    async def health_check() -> JSONResponse:
        """
        Returns live operational readiness and version status.
        """
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "healthy",
                "service": settings.PROJECT_NAME,
                "version": settings.VERSION,
                "environment": settings.ENVIRONMENT,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        """
        Redirects root index directly to the interactive OpenAPI docs.
        """
        return RedirectResponse(url=f"{settings.API_V1_STR}/docs")

    return app


app = create_application()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
