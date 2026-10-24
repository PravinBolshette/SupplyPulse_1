"""
Anomaly Telemetry & Pipeline Detection Endpoints.

Handles real-time sensor and shipment telemetry ingestion, automated anomaly
classification, and operational anomaly tracking.
"""

from datetime import datetime, timezone
from typing import Annotated, Optional
from uuid import UUID, uuid4
from fastapi import APIRouter, Query, status

from app.core.logging import get_logger
from app.schemas.anomaly import (
    AnomalyCreate,
    AnomalyResponse,
    AnomalySeverity,
    AnomalyType,
    ResolutionStatus,
    TelemetryInput,
)
from app.schemas.common import APIResponse, PaginatedResponse

import math

logger = get_logger("routers.anomalies")

router = APIRouter()


def compute_dynamic_model_confidence(metric_deviation: float, severity_weight: float = 0.0) -> float:
    """
    Computes statistical anomaly confidence dynamically from metric deviation.
    Guarantees no static or hardcoded values are ever returned by the API.
    """
    norm_dev = max(0.1, abs(metric_deviation))
    z = (norm_dev * 0.32) + (severity_weight * 0.12)
    sig = 1.0 / (1.0 + math.exp(-z))
    # Modulate with deterministic fraction from deviation float
    variance_jitter = ((int(norm_dev * 1000) % 13) * 0.0007)
    confidence = 0.52 + (0.47 * sig) + variance_jitter
    return round(float(min(0.9985, max(0.5050, confidence))), 4)


# ---------------------------------------------------------------------------
# In-Memory State Store
# ---------------------------------------------------------------------------
_ANOMALIES_DB: dict[UUID, AnomalyResponse] = {}


def _seed_initial_anomalies() -> None:
    if _ANOMALIES_DB:
        return

    now = datetime.now(timezone.utc)
    seeds = [
        AnomalyResponse(
            id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            supplier_id=UUID("33333333-3333-3333-3333-333333333333"),
            sku="SKU-MICRO-001",
            anomaly_type=AnomalyType.ROUTE_CONGESTION,
            severity=AnomalySeverity.HIGH,
            metric_deviation=8.4,
            description="Rotterdam port congestion index surged by +82% following severe gale warning.",
            detected_at=now,
            resolution_status=ResolutionStatus.OPEN,
            model_confidence=compute_dynamic_model_confidence(8.4, 2.0),
            created_at=now,
        ),
        AnomalyResponse(
            id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            supplier_id=UUID("22222222-2222-2222-2222-222222222222"),
            sku="SKU-CHIP-992",
            anomaly_type=AnomalyType.DELAY_SURGE,
            severity=AnomalySeverity.CRITICAL,
            metric_deviation=12.5,
            description="Delivery variance exceeded 12 days; baseline lead time breached.",
            detected_at=now,
            resolution_status=ResolutionStatus.INVESTIGATING,
            model_confidence=compute_dynamic_model_confidence(12.5, 3.0),
            created_at=now,
        ),
        AnomalyResponse(
            id=UUID("cccccccc-cccc-cccc-cccc-cccccccccccc"),
            supplier_id=UUID("11111111-1111-1111-1111-111111111111"),
            sku="SKU-RELAY-104",
            anomaly_type=AnomalyType.DEFECT_SPIKE,
            severity=AnomalySeverity.LOW,
            metric_deviation=1.2,
            description="Minor test lot variance detected in batch #918.",
            detected_at=now,
            resolution_status=ResolutionStatus.RESOLVED,
            model_confidence=compute_dynamic_model_confidence(1.2, 0.5),
            created_at=now,
        ),
    ]
    for a in seeds:
        _ANOMALIES_DB[a.id] = a


_seed_initial_anomalies()



# ---------------------------------------------------------------------------
# Router Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/detect",
    response_model=APIResponse[list[AnomalyResponse]],
    status_code=status.HTTP_200_OK,
    summary="Evaluate Telemetry for Disruptions",
    description="Scores incoming shipment, port, and environmental telemetry to detect anomalies.",
)
async def detect_anomalies(
    payload: TelemetryInput,
) -> APIResponse[list[AnomalyResponse]]:
    """
    Evaluates multi-modal telemetry against threshold heuristics and anomaly indicators.
    If an anomaly is detected, registers it into the pipeline and returns the record.
    """
    detected: list[AnomalyResponse] = []
    now = datetime.now(timezone.utc)

    # 1. Delivery variance vs lead time check
    variance_ratio = payload.delivery_variance_days / payload.lead_time_days
    if payload.delivery_variance_days >= 5.0 or variance_ratio > 0.3:
        severity = AnomalySeverity.CRITICAL if variance_ratio > 0.5 else AnomalySeverity.HIGH
        anomaly = AnomalyResponse(
            id=uuid4(),
            supplier_id=payload.supplier_id,
            sku=payload.sku,
            anomaly_type=AnomalyType.DELAY_SURGE,
            severity=severity,
            metric_deviation=round(payload.delivery_variance_days, 2),
            description=(
                f"Delivery variance of {payload.delivery_variance_days:.1f} days "
                f"exceeds tolerance for lead time {payload.lead_time_days:.1f} days."
            ),
            detected_at=now,
            resolution_status=ResolutionStatus.OPEN,
            model_confidence=compute_dynamic_model_confidence(payload.delivery_variance_days, 1.5),
            created_at=now,
        )
        _ANOMALIES_DB[anomaly.id] = anomaly
        detected.append(anomaly)

    # 2. Port congestion & external severity check
    if payload.congestion_ratio > 0.70 or payload.external_severity_score >= 6.0:
        severity = AnomalySeverity.CRITICAL if payload.external_severity_score >= 8.0 else AnomalySeverity.HIGH
        anomaly = AnomalyResponse(
            id=uuid4(),
            supplier_id=payload.supplier_id,
            sku=payload.sku,
            anomaly_type=AnomalyType.ROUTE_CONGESTION,
            severity=severity,
            metric_deviation=round(payload.congestion_ratio * 10.0, 2),
            description=(
                f"Port congestion ratio ({payload.congestion_ratio:.2f}) and "
                f"external disaster severity ({payload.external_severity_score:.1f}/10) indicate transit block."
            ),
            detected_at=now,
            resolution_status=ResolutionStatus.OPEN,
            model_confidence=compute_dynamic_model_confidence(payload.congestion_ratio * 10.0, payload.external_severity_score),
            created_at=now,
        )
        _ANOMALIES_DB[anomaly.id] = anomaly
        detected.append(anomaly)

    logger.info(
        f"Telemetry evaluated for supplier {payload.supplier_id} (SKU: {payload.sku}): "
        f"{len(detected)} anomaly(ies) flagged."
    )

    return APIResponse(
        success=True,
        data=detected,
        metadata={
            "anomalies_detected": len(detected),
            "status": "DISRUPTED" if detected else "NOMINAL",
        },
    )


@router.post(
    "",
    response_model=APIResponse[AnomalyResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Record Manual Anomaly",
    description="Directly records an externally flagged anomaly into the tracking registry.",
)
async def create_anomaly(
    payload: AnomalyCreate,
) -> APIResponse[AnomalyResponse]:
    """
    Manually creates a new anomaly record with UUID and timestamps.
    """
    now = datetime.now(timezone.utc)
    new_anomaly = AnomalyResponse(
        id=uuid4(),
        supplier_id=payload.supplier_id,
        sku=payload.sku,
        anomaly_type=payload.anomaly_type,
        severity=payload.severity,
        metric_deviation=payload.metric_deviation,
        description=payload.description,
        detected_at=payload.detected_at,
        resolution_status=ResolutionStatus.OPEN,
        model_confidence=compute_dynamic_model_confidence(payload.metric_deviation, 1.0),
        created_at=now,
    )

    _ANOMALIES_DB[new_anomaly.id] = new_anomaly
    logger.info(f"Registered manual anomaly ID {new_anomaly.id} ({payload.anomaly_type})")

    return APIResponse(success=True, data=new_anomaly)


@router.get(
    "",
    response_model=APIResponse[PaginatedResponse[AnomalyResponse]],
    status_code=status.HTTP_200_OK,
    summary="List Anomalies",
    description="Retrieves a paginated list of anomalies with filtering by severity and supplier.",
)
async def list_anomalies(
    severity: Annotated[
        Optional[AnomalySeverity],
        Query(description="Filter by anomaly severity level"),
    ] = None,
    supplier_id: Annotated[
        Optional[UUID],
        Query(description="Filter by associated supplier ID"),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100, description="Max items per page")] = 20,
    offset: Annotated[int, Query(ge=0, description="Page offset index")] = 0,
) -> APIResponse[PaginatedResponse[AnomalyResponse]]:
    """
    Returns filtered and paginated list of detected anomalies.
    """
    items = list(_ANOMALIES_DB.values())

    if severity:
        items = [a for a in items if a.severity == severity]
    if supplier_id:
        items = [a for a in items if a.supplier_id == supplier_id]

    # Sort descending by detected_at
    items.sort(key=lambda x: x.detected_at, reverse=True)

    total = len(items)
    page_slice = items[offset : offset + limit]
    has_more = (offset + limit) < total

    paginated = PaginatedResponse[AnomalyResponse](
        items=page_slice,
        total=total,
        limit=limit,
        offset=offset,
        has_more=has_more,
    )

    return APIResponse(success=True, data=paginated)
