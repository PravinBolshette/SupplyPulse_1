"""
Authentication & Authorization Schemas for SupplyPulse.

Defines Pydantic v2 data models for user registration, login, JWT token issuance,
and role-based access control.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator


class UserRole(str, Enum):
    """Permitted user roles in the SupplyPulse platform."""
    ANALYST = "analyst"
    MANAGER = "manager"
    ADMIN = "admin"


class UserRegister(BaseModel):
    """Payload for user registration."""
    username: str = Field(..., min_length=3, max_length=50, description="Unique username identifier")
    password: str = Field(..., min_length=6, max_length=72, description="User password (min 6 chars)")
    role: UserRole = Field(default=UserRole.ANALYST, description="Operational role")

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        clean = v.strip().lower()
        if not clean.replace("_", "").replace(".", "").isalnum():
            raise ValueError("Username may only contain alphanumeric characters, underscores, and dots.")
        return clean


class UserLogin(BaseModel):
    """Payload for user authentication."""
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    """JWT Token response returned upon successful authentication."""
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str
    expires_in_minutes: int = 1440


class UserResponse(BaseModel):
    """Public user profile response."""
    user_id: str
    username: str
    role: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
