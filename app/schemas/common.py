"""
Common Pydantic v2 Data Models & Generic Response Envelopes.

Defines the standard API response structure, pagination containers,
and error detail schemas used across all endpoints.
"""

from typing import Any, Generic, Optional, Sequence, TypeVar
from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ErrorDetail(BaseModel):
    """Specific field or contextual error detail."""
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    field: Optional[str] = Field(default=None, description="Path of the offending field")
    message: str = Field(..., description="Validation or business rule violation description")
    type: Optional[str] = Field(default=None, description="Machine-readable error type identifier")


class ErrorEnvelope(BaseModel):
    """Structured error payload nested within failed API responses."""
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    code: str = Field(..., description="High-level machine-readable error code")
    message: str = Field(..., description="Human-readable summary of the error")
    details: Optional[list[dict[str, Any]]] = Field(
        default=None, description="Granular error breakdowns"
    )


class APIResponse(BaseModel, Generic[T]):
    """
    Unified API response wrapper enforcing consistent contract structure.
    """
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    success: bool = Field(default=True, description="Indicates whether the request was successful")
    data: Optional[T] = Field(default=None, description="Payload returned by the endpoint")
    error: Optional[ErrorEnvelope] = Field(default=None, description="Error detail if success is false")
    metadata: Optional[dict[str, Any]] = Field(
        default=None, description="Optional telemetry, correlation, or timing metadata"
    )


class PaginationParams(BaseModel):
    """Standard offset-based pagination query parameters."""
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of items to return")
    offset: int = Field(default=0, ge=0, description="Offset index for pagination")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic envelope for paginated collection responses."""
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    items: Sequence[T] = Field(..., description="Collection of items for the requested page")
    total: int = Field(..., ge=0, description="Total count of items matching the query")
    limit: int = Field(..., ge=1, description="Page limit utilized")
    offset: int = Field(..., ge=0, description="Offset index utilized")
    has_more: bool = Field(..., description="Flag indicating if more records exist beyond this page")
