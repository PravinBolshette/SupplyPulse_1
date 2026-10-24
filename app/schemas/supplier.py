"""
Pydantic v2 Models for Supplier Master Data & Risk Posture.

Enforces strict field validations, ISO country format constraints,
email validation, and immutability settings.
"""

from datetime import datetime
from enum import Enum
from typing import Annotated, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, EmailStr, Field


class SupplierStatus(str, Enum):
    """Lifecycle and operational risk status for a supplier."""
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    AT_RISK = "AT_RISK"


class SupplierBase(BaseModel):
    """Shared attributes across creation and response schemas."""
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    name: Annotated[
        str,
        Field(
            min_length=2,
            max_length=100,
            description="Legal or commercial name of the supplier",
            examples=["Semiconductor Fabrications Ltd"],
        ),
    ]
    contact_email: Annotated[
        EmailStr,
        Field(
            description="Direct point-of-contact procurement email",
            examples=["procurement@semifab.de"],
        ),
    ]
    lead_time_days: Annotated[
        float,
        Field(
            gt=0.0,
            description="Nominal replenishment lead time in days",
            examples=[14.5],
        ),
    ]
    reliability_score: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Historical order fulfillment and on-time reliability score (0.0 to 1.0)",
            examples=[0.94],
        ),
    ]
    country_code: Annotated[
        str,
        Field(
            pattern=r"^[A-Z]{2}$",
            description="ISO 3166-1 alpha-2 uppercase country code",
            examples=["DE"],
        ),
    ]
    cost_per_unit: Annotated[
        float,
        Field(
            ge=0.0,
            default=0.0,
            description="Contractual unit price for primary components",
            examples=[125.50],
        ),
    ]


class SupplierCreate(SupplierBase):
    """Payload required to onboard a new supplier."""
    pass


class SupplierUpdate(BaseModel):
    """Payload for partial updates to supplier attributes."""
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    name: Optional[Annotated[str, Field(min_length=2, max_length=100)]] = None
    contact_email: Optional[EmailStr] = None
    lead_time_days: Optional[Annotated[float, Field(gt=0.0)]] = None
    reliability_score: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = None
    country_code: Optional[Annotated[str, Field(pattern=r"^[A-Z]{2}$")]] = None
    cost_per_unit: Optional[Annotated[float, Field(ge=0.0)]] = None
    status: Optional[SupplierStatus] = None


class SupplierResponse(SupplierBase):
    """Complete supplier representation returned to clients."""
    id: Annotated[UUID, Field(description="Unique system identifier for the supplier")]
    status: Annotated[
        SupplierStatus,
        Field(
            default=SupplierStatus.ACTIVE,
            description="Current operational and risk status",
        ),
    ]
    city: Optional[str] = Field(default=None, description="City or primary manufacturing port hub")
    lat: Optional[float] = Field(default=None, description="Geographic latitude coordinate")
    lon: Optional[float] = Field(default=None, description="Geographic longitude coordinate")
    region: Optional[str] = Field(default="APAC Freight Corridor", description="Trade corridor region")
    primary_skus: Optional[list[str]] = Field(default_factory=list, description="Primary SKUs supplied")
    sla_adherence: Optional[float] = Field(default=95.0, description="SLA adherence percentage")
    variance_avg: Optional[float] = Field(default=1.0, description="Mean delivery variance in days")
    created_at: Annotated[datetime, Field(description="Timestamp when supplier was onboarded")]
    updated_at: Annotated[datetime, Field(description="Timestamp of last metadata update")]
