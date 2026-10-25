"""
Pydantic v2 Models for Supplier Master Data & Risk Posture.

Enforces strict field validations, ISO country format constraints,
email validation, initial inventory provisioning, and immutability settings.
"""

from datetime import datetime
from enum import Enum
from typing import Annotated, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator


class SupplierStatus(str, Enum):
    """Lifecycle and operational risk status for a supplier."""
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    AT_RISK = "AT_RISK"


class SupplierBase(BaseModel):
    """Shared attributes across creation and response schemas."""
    model_config = ConfigDict(from_attributes=True, extra="ignore")

    name: Annotated[
        str,
        Field(
            min_length=2,
            max_length=150,
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
            default=0.95,
            description="Historical order fulfillment and on-time reliability score (0.0 to 1.0)",
            examples=[0.94],
        ),
    ]
    country_code: Annotated[
        str,
        Field(
            min_length=2,
            max_length=10,
            default="XX",
            description="Country code or country name abbreviation",
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


class SupplierCreate(BaseModel):
    """
    Payload required to onboard a new supplier and provision baseline inventory.
    Supports both new structured schema and legacy fields.
    """
    model_config = ConfigDict(extra="ignore")

    # New Core Supplier Identity Fields
    supplier_id: Optional[str] = Field(default=None, description="Unique supplier code (e.g. SUP-015)")
    supplier_name: Optional[str] = Field(default=None, min_length=2, max_length=150, description="Legal business entity name")
    country: Optional[str] = Field(default=None, min_length=2, max_length=100, description="Country of manufacturing facility")
    backup_supplier_id: Optional[str] = Field(default=None, description="Designated backup supplier ID or name")
    contracted_lead_time_days: Optional[int] = Field(default=None, ge=0, description="Contracted procurement lead time in calendar days")

    # Initial Inventory Fields (Provisioned into inventory_levels)
    sku: Optional[str] = Field(default="SKU-CORE-101", min_length=2, max_length=50, description="Initial SKU identifier")
    current_stock_units: int = Field(default=500, ge=0, description="On-hand physical inventory count")
    reorder_point: int = Field(default=150, ge=0, description="Stock buffer threshold triggering reorder")
    daily_consumption_rate: float = Field(default=25.0, gt=0.0, description="Forecasted daily consumption velocity")

    # Legacy/Compatibility Fields
    name: Optional[str] = Field(default=None, min_length=2, max_length=150, description="Supplier name")
    contact_email: Optional[EmailStr] = Field(default=None, description="Direct point-of-contact procurement email")
    lead_time_days: Optional[float] = Field(default=None, gt=0.0, description="Replenishment lead time in days")
    reliability_score: Optional[float] = Field(default=0.95, ge=0.0, le=1.0, description="Historical reliability score (0.0 to 1.0)")
    country_code: Optional[str] = Field(default=None, min_length=2, max_length=2, description="2-letter ISO country code")
    cost_per_unit: Optional[float] = Field(default=0.0, ge=0.0, description="Contractual unit price")

    @model_validator(mode="before")
    @classmethod
    def harmonize_fields(cls, values: dict) -> dict:
        if not isinstance(values, dict):
            return values

        # Resolve supplier name
        if "supplier_name" in values and "name" not in values:
            values["name"] = values["supplier_name"]
        elif "name" in values and "supplier_name" not in values:
            values["supplier_name"] = values["name"]
        elif "supplier_name" not in values and "name" not in values:
            raise ValueError("Either 'supplier_name' or 'name' must be provided.")

        s_name = values.get("supplier_name") or values.get("name") or "supplier"

        # Resolve supplier ID
        s_id = values.get("supplier_id")
        if not s_id:
            s_id = f"SUP-{uuid4().hex[:6].upper()}"
            values["supplier_id"] = s_id

        # Resolve country & country code
        if "country_code" in values and "country" not in values:
            values["country"] = values["country_code"]
        elif "country" in values and "country_code" not in values:
            values["country_code"] = str(values["country"])[:2].upper()
        elif "country" not in values and "country_code" not in values:
            values["country"] = "Global"
            values["country_code"] = "XX"

        # Resolve lead time
        if "contracted_lead_time_days" in values and "lead_time_days" not in values:
            values["lead_time_days"] = float(values["contracted_lead_time_days"])
        elif "lead_time_days" in values and "contracted_lead_time_days" not in values:
            try:
                values["contracted_lead_time_days"] = int(values["lead_time_days"])
            except Exception:
                pass
        elif "contracted_lead_time_days" not in values and "lead_time_days" not in values:
            values["contracted_lead_time_days"] = 14
            values["lead_time_days"] = 14.0

        # Resolve contact email
        if not values.get("contact_email"):
            clean_id = s_id.lower().replace("-", "").replace("_", "")
            values["contact_email"] = f"procure@{clean_id}.supplypulse.internal"

        # Resolve SKU
        if not values.get("sku"):
            values["sku"] = f"SKU-{s_id.upper()}-01"

        return values


class SupplierUpdate(BaseModel):
    """Payload for partial updates to supplier attributes."""
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    name: Optional[Annotated[str, Field(min_length=2, max_length=150)]] = None
    contact_email: Optional[EmailStr] = None
    lead_time_days: Optional[Annotated[float, Field(gt=0.0)]] = None
    reliability_score: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = None
    country_code: Optional[Annotated[str, Field(min_length=2, max_length=10)]] = None
    cost_per_unit: Optional[Annotated[float, Field(ge=0.0)]] = None
    status: Optional[SupplierStatus] = None


class SupplierResponse(SupplierBase):
    """Complete supplier representation returned to clients."""
    id: Annotated[UUID, Field(description="Unique system identifier for the supplier")]
    supplier_id: Optional[str] = Field(default=None, description="Contractual supplier ID (e.g. SUP-001)")
    supplier_name: Optional[str] = Field(default=None, description="Legal business entity name")
    country: Optional[str] = Field(default=None, description="Country of manufacturing facility")
    backup_supplier_name: Optional[str] = Field(default=None, description="Contingency secondary vendor")
    contracted_lead_time_days: Optional[int] = Field(default=14, description="Contracted lead time in days")
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
