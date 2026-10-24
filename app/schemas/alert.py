"""
Pydantic v2 Models for Disruption Risk Notifications and Human-in-the-Loop Acknowledgment.

Encapsulates Time-To-Impact (TTI) predictions, actionable recommendations,
and operator mitigation audit records.
"""

from datetime import datetime
from enum import Enum
from typing import Annotated, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class RiskLevel(str, Enum):
    """Categorical risk tier based on Time-to-Impact vs replenishment lead time."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    STABLE = "STABLE"


class AlertResponse(BaseModel):
    """Disruption early-warning alert payload returned to operators."""
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    alert_id: Annotated[
        UUID,
        Field(description="Unique identifier for the disruption alert notification"),
    ]
    sku: Annotated[
        str,
        Field(min_length=1, max_length=50, description="Impacted inventory stock keeping unit"),
    ]
    predicted_tti_days: Annotated[
        float,
        Field(
            ge=0.0,
            description="Predicted Time-To-Impact countdown until complete stockout",
        ),
    ]
    risk_level: Annotated[
        RiskLevel,
        Field(description="Evaluated supply depletion risk tier"),
    ]
    triggering_anomaly_id: Annotated[
        UUID,
        Field(description="Foreign identifier of the anomaly driving this disruption alert"),
    ]
    suggested_action: Annotated[
        str,
        Field(
            min_length=1,
            max_length=500,
            description="Algorithmic mitigation recommendation (e.g. backup supplier reroute)",
        ),
    ]
    reason: Optional[str] = Field(
        default="Port congestion combined with warehouse inventory buffer deficit",
        description="Underlying causal reason for disruption alert",
    )
    tti_range: Optional[str] = Field(
        default="3 - 6 days",
        description="Estimated confidence interval for Time-to-Impact",
    )
    expected_delay_days: Optional[float] = Field(
        default=5.0,
        description="Expected shipment transit delay in days",
    )
    shortfall_days: Optional[float] = Field(
        default=2.5,
        description="Projected inventory buffer shortfall duration in days",
    )
    acknowledged: Annotated[
        bool,
        Field(default=False, description="Whether an operator has reviewed and acknowledged the alert"),
    ]
    acknowledged_by: Annotated[
        Optional[str],
        Field(default=None, description="Operator user identifier who acknowledged the alert"),
    ]
    acknowledgment_note: Annotated[
        Optional[str],
        Field(default=None, max_length=500, description="Notes recorded during human-in-the-loop review"),
    ]
    acknowledged_at: Annotated[
        Optional[datetime],
        Field(default=None, description="Timestamp when acknowledgment occurred"),
    ]
    created_at: Annotated[
        datetime,
        Field(description="Timestamp when early-warning alert was triggered"),
    ]



class AlertAcknowledgeRequest(BaseModel):
    """Payload submitted by a human operator to acknowledge a disruption alert."""
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    user_id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=100,
            description="Identifier or email of the acknowledging operator",
            examples=["operator.john@fujitsu.com"],
        ),
    ]
    note: Annotated[
        Optional[str],
        Field(
            default=None,
            max_length=500,
            description="Optional comment documenting mitigation or action taken",
            examples=["Initiated secondary procurement order from Backup Supplier Alpha."],
        ),
    ] = None
