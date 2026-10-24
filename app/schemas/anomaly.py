"""
Pydantic v2 Models for Telemetry Anomalies and Pipeline Detection.

Captures multi-modal anomaly classifications, severity scores, and resolution status.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class AnomalyType(str, Enum):
    """Categorical classification of supply chain disruption drivers."""
    DELAY_SURGE = "DELAY_SURGE"
    ROUTE_CONGESTION = "ROUTE_CONGESTION"
    DEFECT_SPIKE = "DEFECT_SPIKE"
    WEATHER_SHOCK = "WEATHER_SHOCK"
    DEMAND_SPIKE = "DEMAND_SPIKE"


class AnomalySeverity(str, Enum):
    """Severity classification calibrated against operational impact."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ResolutionStatus(str, Enum):
    """Lifecycle workflow status of a flagged anomaly."""
    OPEN = "OPEN"
    INVESTIGATING = "INVESTIGATING"
    RESOLVED = "RESOLVED"


class TelemetryInput(BaseModel):
    """
    Real-time multi-dimensional sensor and operational telemetry vector
    ingested for automated anomaly scoring.
    """
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    supplier_id: Annotated[UUID, Field(description="Supplier identifier being monitored")]
    sku: Annotated[str, Field(min_length=1, max_length=50, description="Affected Stock Keeping Unit")]
    delivery_variance_days: Annotated[
        float,
        Field(description="Variance between actual delivery time and contract lead time in days"),
    ]
    port_delay_days: Annotated[
        float,
        Field(ge=0.0, default=0.0, description="Observed maritime or border port delay in days"),
    ]
    congestion_ratio: Annotated[
        float,
        Field(ge=0.0, le=1.0, default=0.0, description="Port or transit route congestion factor (0.0 - 1.0)"),
    ]
    external_severity_score: Annotated[
        float,
        Field(ge=0.0, le=10.0, default=0.0, description="NLP-extracted weather/disaster severity (0.0 - 10.0)"),
    ]
    order_fulfillment_ratio: Annotated[
        float,
        Field(ge=0.0, le=1.0, default=1.0, description="Delivered quantity vs ordered quantity ratio"),
    ]
    lead_time_days: Annotated[
        float,
        Field(gt=0.0, description="Baseline replenishment lead time in days"),
    ]


class AnomalyCreate(BaseModel):
    """Payload to record or flag a confirmed anomaly."""
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    supplier_id: Annotated[UUID, Field(description="Reference ID of the anomalous supplier")]
    sku: Annotated[
        Optional[str],
        Field(default=None, max_length=50, description="Associated component or product SKU"),
    ] = None
    anomaly_type: Annotated[
        AnomalyType,
        Field(description="Type classification of the operational anomaly"),
    ]
    severity: Annotated[
        AnomalySeverity,
        Field(description="Impact severity tier"),
    ]
    metric_deviation: Annotated[
        float,
        Field(description="Magnitude of metric deviation from standard baseline (in units or standard deviations)"),
    ]
    description: Annotated[
        Optional[str],
        Field(default=None, max_length=500, description="Contextual narrative or sensor trigger explanation"),
    ] = None
    detected_at: Annotated[
        datetime,
        Field(
            default_factory=lambda: datetime.now(timezone.utc),
            description="Timestamp when anomaly condition was identified",
        ),
    ]


class AnomalyResponse(AnomalyCreate):
    """Complete anomaly record with resolution tracking and confidence score."""
    id: Annotated[UUID, Field(description="Unique identifier for the detected anomaly")]
    resolution_status: Annotated[
        ResolutionStatus,
        Field(
            default=ResolutionStatus.OPEN,
            description="Operational handling status of the anomaly",
        ),
    ]
    model_confidence: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            default=1.0,
            description="Statistical or machine learning detection confidence",
        ),
    ]
    created_at: Annotated[
        datetime,
        Field(description="System persistence timestamp"),
    ]
