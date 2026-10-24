"""
Application Configuration Module.

Leverages Pydantic v2 BaseSettings for type-safe environment variable parsing,
validation, and hierarchical settings management.
"""

from functools import lru_cache
from typing import Annotated, List, Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Global application settings loaded from environment variables and .env files.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Project Metadata
    PROJECT_NAME: Annotated[str, Field(default="SupplyPulse Risk Engine")]
    VERSION: Annotated[str, Field(default="1.0.0")]
    DESCRIPTION: Annotated[
        str,
        Field(default="Automated Supply Chain Disruption & Early-Warning Risk Monitoring API"),
    ]
    API_V1_STR: Annotated[str, Field(default="/api/v1")]

    # Runtime Environment
    ENVIRONMENT: Annotated[
        Literal["development", "staging", "production", "testing"],
        Field(default="development"),
    ]
    DEBUG: Annotated[bool, Field(default=False)]

    # Server Binding
    HOST: Annotated[str, Field(default="0.0.0.0")]
    PORT: Annotated[int, Field(default=8000)]

    # Observability & Logging
    LOG_LEVEL: Annotated[
        Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        Field(default="INFO"),
    ]
    LOG_FORMAT: Annotated[Literal["json", "text"], Field(default="json")]

    # Security & CORS
    CORS_ORIGINS: Annotated[
        List[str],
        Field(
            default_factory=lambda: [
                "http://localhost:3000",
                "http://localhost:8000",
                "http://localhost:8501",  # Streamlit default port
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8000",
                "http://127.0.0.1:8501",
            ]
        ),
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Supports comma-separated strings from environment variables."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, (list, str)):
            return v  # type: ignore[return-value]
        raise ValueError(f"Invalid CORS_ORIGINS format: {v}")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached singleton instance of the application settings.
    """
    return Settings()
