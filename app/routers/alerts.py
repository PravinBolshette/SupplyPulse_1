"""
Disruption Risk Alerts & Human-in-the-Loop Acknowledgment Endpoints.

Provides capabilities to inspect active early-warning alerts, evaluate
predicted Time-To-Impact countdowns, and record human operator mitigation actions.
"""

from datetime import datetime, timezone
import os
from typing import Annotated, Optional
from uuid import UUID
from fastapi import APIRouter, Path, Query, status

from app.core.errors import EntityNotFoundException, InvalidStateTransitionException
from app.core.logging import get_logger
from app.schemas.alert import (
    AlertAcknowledgeRequest,
    AlertResponse,
    RiskLevel,
)
from app.schemas.common import APIResponse, PaginatedResponse

logger = get_logger("routers.alerts")

router = APIRouter()

# ---------------------------------------------------------------------------
# In-Memory State Store
# ---------------------------------------------------------------------------
_ALERTS_DB: dict[UUID, AlertResponse] = {}


def _seed_initial_alerts() -> None:
    if _ALERTS_DB:
        return

    now = datetime.now(timezone.utc)
    seeds = [
        AlertResponse(
            alert_id=UUID("aaaaaaaa-1111-1111-1111-aaaaaaaaaaaa"),
            sku="SKU-CHIP-992",
            predicted_tti_days=4.5,
            risk_level=RiskLevel.CRITICAL,
            triggering_anomaly_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            suggested_action="Activate Backup Supplier Alpha (lead time: 3.5 days) immediately to prevent line halt.",
            reason="Typhoon warning causing severe congestion at Port of Shanghai + 45% daily consumption spike",
            tti_range="3.5 - 5.5 days",
            expected_delay_days=6.5,
            shortfall_days=2.0,
            acknowledged=False,
            acknowledged_by=None,
            acknowledgment_note=None,
            acknowledged_at=None,
            created_at=now,
        ),
        AlertResponse(
            alert_id=UUID("bbbbbbbb-2222-2222-2222-bbbbbbbbbbbb"),
            sku="SKU-MICRO-001",
            predicted_tti_days=9.8,
            risk_level=RiskLevel.HIGH,
            triggering_anomaly_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            suggested_action="Reroute incoming shipment from Rotterdam to Hamburg rail corridor.",
            reason="Port of Rotterdam labor strike causing 8-day berthing queue and container backlog",
            tti_range="8.0 - 11.5 days",
            expected_delay_days=8.0,
            shortfall_days=3.2,
            acknowledged=False,
            acknowledged_by=None,
            acknowledgment_note=None,
            acknowledged_at=None,
            created_at=now,
        ),
        AlertResponse(
            alert_id=UUID("cccccccc-3333-3333-3333-cccccccccccc"),
            sku="SKU-STEEL-401",
            predicted_tti_days=28.0,
            risk_level=RiskLevel.MODERATE,
            triggering_anomaly_id=UUID("cccccccc-cccc-cccc-cccc-cccccccccccc"),
            suggested_action="Monitor rolling buffer inventory; standard replenishment sufficient.",
            reason="Mild canal lock maintenance delay; warehouse buffer remains healthy",
            tti_range="25.0 - 31.0 days",
            expected_delay_days=2.5,
            shortfall_days=0.0,
            acknowledged=True,
            acknowledged_by="operator.sarah@fujitsu.com",
            acknowledgment_note="Confirmed secondary warehouse has 15-day safety cushion.",
            acknowledged_at=now,
            created_at=now,
        ),
    ]
    for a in seeds:
        _ALERTS_DB[a.alert_id] = a


_seed_initial_alerts()


def _get_alerts_from_db_if_available() -> Optional[list[AlertResponse]]:
    """Attempts to query disruption alerts from PostgreSQL."""
    db_uri = os.getenv("DATABASE_URL")
    if not db_uri or "postgresql" not in db_uri:
        return None
    try:
        import psycopg2
        import hashlib
        conn = psycopg2.connect(db_uri, connect_timeout=1)
        cur = conn.cursor()
        query = """
            SELECT da.alert_id, da.supplier_id, s.supplier_name, s.backup_supplier_name,
                   da.anomaly_score, da.is_flagged, da.predicted_tti_days,
                   da.recommended_action, da.created_at
            FROM disruption_alerts da
            LEFT JOIN suppliers s ON da.supplier_id = s.supplier_id
            ORDER BY da.predicted_tti_days ASC;
        """
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return None

        db_alerts = []
        for row in rows:
            aid, sid, sname, bname, score, is_flagged, tti, action, cat = row
            h = hashlib.md5(f"db-alert-{aid}-{sid}".encode("utf-8")).hexdigest()
            auuid = UUID(f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}")
            tti_float = float(tti)
            risk_tier = (
                RiskLevel.CRITICAL if tti_float <= 5.0
                else (RiskLevel.HIGH if tti_float <= 14.0 else RiskLevel.MODERATE)
            )
            tti_lower = max(0.5, round(tti_float * 0.8, 1))
            tti_upper = round(tti_float * 1.25, 1)
            db_alerts.append(
                AlertResponse(
                    alert_id=auuid,
                    sku=f"SKU-{sid}",
                    predicted_tti_days=tti_float,
                    risk_level=risk_tier,
                    triggering_anomaly_id=auuid,
                    suggested_action=action or f"Activate backup supplier {bname}",
                    reason=f"Anomaly score {score:.3f} on corridor for {sname}",
                    tti_range=f"{tti_lower} - {tti_upper} days",
                    expected_delay_days=round(tti_float * 0.7, 1),
                    shortfall_days=round(max(0.0, 7.0 - tti_float), 1),
                    acknowledged=False,
                    created_at=cat or datetime.now(timezone.utc),
                )
            )
        return db_alerts
    except Exception as exc:
        logger.debug(f"PostgreSQL alerts query skipped: {exc}")
        return None


# ---------------------------------------------------------------------------
# Router Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/active",
    response_model=APIResponse[list[AlertResponse]],
    status_code=status.HTTP_200_OK,
    summary="Get Active High-Risk Alerts",
    description="Returns all unacknowledged CRITICAL and HIGH disruption alerts sorted by urgency.",
)
async def get_active_alerts() -> APIResponse[list[AlertResponse]]:
    """
    Returns unacknowledged high-severity alerts prioritized by ascending Time-To-Impact.
    """
    db_alerts = _get_alerts_from_db_if_available()
    pool = db_alerts if db_alerts is not None else list(_ALERTS_DB.values())

    active = [
        alert
        for alert in pool
        if not alert.acknowledged and alert.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH)
    ]
    # Prioritize by shortest predicted TTI first
    active.sort(key=lambda a: a.predicted_tti_days)

    return APIResponse(
        success=True,
        data=active,
        metadata={"total_active_alerts": len(active)},
    )


@router.get(
    "",
    response_model=APIResponse[PaginatedResponse[AlertResponse]],
    status_code=status.HTTP_200_OK,
    summary="List All Alerts with Filtering",
    description="Returns a paginated list of all alerts with filtering by risk level and acknowledgment status.",
)
async def list_alerts(
    risk_level: Annotated[
        Optional[RiskLevel],
        Query(description="Filter alerts by assessed risk tier"),
    ] = None,
    acknowledged: Annotated[
        Optional[bool],
        Query(description="Filter by operator acknowledgment status"),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100, description="Max alerts per page")] = 20,
    offset: Annotated[int, Query(ge=0, description="Page offset index")] = 0,
) -> APIResponse[PaginatedResponse[AlertResponse]]:
    """
    Returns paginated list of alerts.
    """
    db_alerts = _get_alerts_from_db_if_available()
    items = db_alerts if db_alerts is not None else list(_ALERTS_DB.values())


    if risk_level:
        items = [a for a in items if a.risk_level == risk_level]
    if acknowledged is not None:
        items = [a for a in items if a.acknowledged == acknowledged]

    # Sort by urgency: shortest TTI first
    items.sort(key=lambda a: a.predicted_tti_days)

    total = len(items)
    page_slice = items[offset : offset + limit]
    has_more = (offset + limit) < total

    paginated = PaginatedResponse[AlertResponse](
        items=page_slice,
        total=total,
        limit=limit,
        offset=offset,
        has_more=has_more,
    )

    return APIResponse(success=True, data=paginated)


@router.patch(
    "/{alert_id}/acknowledge",
    response_model=APIResponse[AlertResponse],
    status_code=status.HTTP_200_OK,
    summary="Acknowledge Disruption Alert",
    description="Marks a disruption alert as reviewed and acknowledged by an authorized operator.",
)
async def acknowledge_alert(
    alert_id: Annotated[UUID, Path(description="UUID of the alert to acknowledge")],
    payload: AlertAcknowledgeRequest,
) -> APIResponse[AlertResponse]:
    """
    Acknowledge an open alert, storing operator ID, note, and acknowledgment timestamp.
    """
    alert = _ALERTS_DB.get(alert_id)
    if not alert:
        raise EntityNotFoundException(
            message=f"Alert with ID '{alert_id}' does not exist.",
            entity_name="Alert",
            entity_id=str(alert_id),
        )

    if alert.acknowledged:
        raise InvalidStateTransitionException(
            f"Alert '{alert_id}' has already been acknowledged by '{alert.acknowledged_by}'."
        )

    now = datetime.now(timezone.utc)
    updated_alert = AlertResponse(
        alert_id=alert.alert_id,
        sku=alert.sku,
        predicted_tti_days=alert.predicted_tti_days,
        risk_level=alert.risk_level,
        triggering_anomaly_id=alert.triggering_anomaly_id,
        suggested_action=alert.suggested_action,
        acknowledged=True,
        acknowledged_by=payload.user_id,
        acknowledgment_note=payload.note,
        acknowledged_at=now,
        created_at=alert.created_at,
    )

    _ALERTS_DB[alert_id] = updated_alert
    logger.info(
        f"Alert '{alert_id}' successfully acknowledged by '{payload.user_id}' "
        f"(SKU: {alert.sku}, TTI: {alert.predicted_tti_days}d)"
    )

    return APIResponse(
        success=True,
        data=updated_alert,
        metadata={"action": "alert_acknowledged", "acknowledged_by": payload.user_id},
    )
