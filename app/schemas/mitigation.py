"""
Responsible AI & Human-in-the-Loop (HITL) Mitigation Schemas.

Enforces strict human oversight validation, advisory-only constraints,
override justification requirements, and immutable audit logging contracts.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Dict, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ActionTaken(str, Enum):
    """Permitted human operator mitigation decisions."""
    APPROVED = "APPROVED"
    OVERRIDDEN = "OVERRIDDEN"
    REJECTED = "REJECTED"


class AIRecommendation(BaseModel):
    """
    Structured representation of the AI model's advisory recommendation.
    """
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    summary: Annotated[str, Field(min_length=1, max_length=500, description="High-level advisory synopsis")]
    suggested_action: Annotated[str, Field(min_length=1, max_length=500, description="Specific mitigation protocol proposed by AI")]
    predicted_tti_days: Annotated[float, Field(ge=0.0, description="Computed Time-To-Impact in days")]
    risk_level: Annotated[str, Field(min_length=1, max_length=50, description="Assessed risk tier (e.g. CRITICAL, HIGH)")]
    target_sku: Optional[Annotated[str, Field(max_length=50)]] = None
    backup_supplier_name: Optional[Annotated[str, Field(max_length=150)]] = None


class MitigationActionRequest(BaseModel):
    """
    Operator payload authorizing or overriding an AI disruption mitigation.
    Enforces Responsible AI guardrails at schema level.
    """
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    operator_id: Optional[str] = Field(
        default=None,
        description="Optional client hint. Note: Server strictly enforces operator_id from the authenticated JWT session.",
        examples=["operator.lead@supplypulse.ai"],
    )
    alert_id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=100,
            description="ID of the disruption alert being addressed",
            examples=["aaaaaaaa-1111-1111-1111-aaaaaaaaaaaa"],
        ),
    ]
    ai_recommendation: AIRecommendation = Field(
        ..., description="Snapshot of the AI advisory recommendation at decision time"
    )
    action_taken: ActionTaken = Field(
        ..., description="Human operator decision: APPROVED, OVERRIDDEN, or REJECTED"
    )
    operator_notes: Optional[Annotated[str, Field(max_length=1000)]] = Field(
        default=None,
        description="Operator justification or audit notes (mandatory for overrides and rejections)",
    )
    confirmed: Annotated[
        bool,
        Field(
            description="Explicit human confirmation flag. Must be strictly True to authorize mitigation.",
        ),
    ]
    mitigation_target: Optional[str] = Field(
        default=None,
        description="Target supplier, freight corridor, or buffer being modified",
    )
    cost_exposure_usd: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Financial exposure evaluated during mitigation review",
    )

    @model_validator(mode="after")
    def enforce_responsible_ai_guardrails(self) -> "MitigationActionRequest":
        """
        Enforces Responsible AI principles:
        1. Mandatory human confirmation.
        2. Prohibits automated / system / bot execution of state-changing decisions.
        3. Requires explicit justification note when overriding or rejecting AI advice.
        """
        # Guardrail 1: Human Confirmation Enforcement
        if not self.confirmed:
            raise ValueError(
                "Human confirmation required: The 'confirmed' flag must be True to execute any mitigation action."
            )

        # Guardrail 2: Advisory-Only AI Constraint (Optional extra check on operator hint if provided)
        if self.operator_id:
            disallowed_prefixes = ("system", "ai_", "bot_", "automated_", "auto_", "cron_")
            normalized_operator = self.operator_id.strip().lower()
            if (
                normalized_operator in ("system", "ai_agent", "bot", "automated", "auto_pilot", "cron")
                or normalized_operator.startswith(disallowed_prefixes)
            ):
                raise ValueError(
                    "AUTONOMOUS_EXECUTION_BLOCKED: AI models and automated background services are strictly "
                    "restricted to advisory output. Mitigation actions require authorization by a verified human operator."
                )


        # Guardrail 3: Accountability & Override Justification Enforcer
        if self.action_taken in (ActionTaken.OVERRIDDEN, ActionTaken.REJECTED):
            if not self.operator_notes or len(self.operator_notes.strip()) < 5:
                raise ValueError(
                    f"Accountability policy requires a detailed operator justification note "
                    f"(minimum 5 characters) when action is '{self.action_taken.value}'."
                )

        return self


class AuditLogResponse(BaseModel):
    """
    Representation of an immutable decision entry in the audit ledger.
    """
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    log_id: UUID = Field(default_factory=uuid4, description="Unique immutable log ID")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Immutable UTC timestamp of operator decision",
    )
    operator_id: str
    alert_id: str
    ai_recommendation: Dict[str, Any]
    action_taken: ActionTaken
    operator_notes: Optional[str] = None
    confirmed: bool = True
    mitigation_target: Optional[str] = None
    cost_exposure_usd: Optional[float] = None
    audit_immutable: bool = Field(default=True, description="Signifies record cannot be updated or deleted")
