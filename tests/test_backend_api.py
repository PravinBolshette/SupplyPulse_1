"""
Automated Test Suite for SupplyPulse FastAPI Backend API.

Validates:
1. Health check and root redirects
2. Correlation ID injection & propagation (X-Request-ID)
3. Audit execution latency headers (X-Process-Time-Ms)
4. Centralized Pydantic v2 validation error formatting (422)
5. Custom domain exceptions (404 EntityNotFoundException, 409 Conflict, 400 State Transition)
6. Supplier onboarding and paginated querying
7. Telemetry anomaly detection scoring
8. Disruption alert prioritization and operator acknowledgment lifecycle
"""

import uuid
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Health & Middleware Observability Tests
# ---------------------------------------------------------------------------

def test_health_check():
    """Validates health check returns 200 OK and valid JSON envelope."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data


def test_correlation_id_auto_generation():
    """Validates X-Request-ID is generated if not provided."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "x-request-id" in response.headers
    corr_id = response.headers["x-request-id"]
    # Ensure it is a valid UUID
    parsed_uuid = uuid.UUID(corr_id)
    assert str(parsed_uuid) == corr_id


def test_correlation_id_propagation():
    """Validates incoming X-Request-ID is preserved and echoed back."""
    custom_corr_id = "test-custom-correlation-uuid-9999"
    response = client.get("/health", headers={"X-Request-ID": custom_corr_id})
    assert response.status_code == 200
    assert response.headers.get("x-request-id") == custom_corr_id


def test_process_time_header():
    """Validates X-Process-Time-Ms latency header is attached to responses."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "x-process-time-ms" in response.headers
    latency = float(response.headers["x-process-time-ms"])
    assert latency >= 0.0


def get_auth_headers(role="manager", username="test_mgr"):
    from app.core.security import create_access_token, _USERS_CACHE, hash_password
    clean = username.strip().lower()
    if clean not in _USERS_CACHE:
        _USERS_CACHE[clean] = {
            "user_id": f"test-uuid-{clean}",
            "username": clean,
            "password_hash": hash_password("pass123"),
            "role": role,
        }
    token = create_access_token(data={"sub": clean, "role": role})
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# 2. Global Error Handling & Validation Tests
# ---------------------------------------------------------------------------

def test_validation_error_envelope_sanitization():
    """
    Validates invalid Pydantic input produces standard 422 JSON envelope
    with clean field paths.
    """
    invalid_supplier = {
        "name": "A",  # Too short (min_length=2)
        "contact_email": "invalid-not-an-email",
        "lead_time_days": -5.0,  # Must be > 0
        "reliability_score": 1.5,  # Must be <= 1.0
        "country_code": "deu",  # Must be 2 uppercase chars
    }
    mgr_headers = get_auth_headers(role="manager", username="validator_mgr")
    response = client.post("/api/v1/suppliers", json=invalid_supplier, headers=mgr_headers)
    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert "error" in body
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert isinstance(body["error"]["details"], list)
    assert len(body["error"]["details"]) >= 4

    fields = [d["field"] for d in body["error"]["details"]]
    assert "name" in fields
    assert "contact_email" in fields
    assert "lead_time_days" in fields
    assert "reliability_score" in fields


def test_entity_not_found_exception():
    """Validates 404 EntityNotFoundException returns standard error structure."""
    fake_uuid = str(uuid.uuid4())
    response = client.get(f"/api/v1/suppliers/{fake_uuid}")
    assert response.status_code == 404
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "ENTITY_NOT_FOUND"
    assert fake_uuid in body["error"]["message"]


# ---------------------------------------------------------------------------
# 3. Suppliers Endpoints Tests
# ---------------------------------------------------------------------------

def test_list_suppliers_paginated():
    """Validates listing pre-seeded suppliers with pagination metadata."""
    response = client.get("/api/v1/suppliers?limit=2&offset=0")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    data = body["data"]
    assert len(data["items"]) == 2
    assert data["limit"] == 2
    assert data["offset"] == 0
    assert data["total"] >= 4
    assert data["has_more"] is True


def test_create_supplier_success_and_conflict():
    """Validates role-gated supplier onboarding, initial inventory provisioning, and conflict handling."""
    unique_email = f"procure-{uuid.uuid4().hex[:6]}@kyoto-semicon.jp"
    unique_sup_id = f"SUP-TEST-{uuid.uuid4().hex[:4].upper()}"
    payload = {
        "supplier_id": unique_sup_id,
        "supplier_name": "Kyoto Advanced Semiconductor Corp",
        "country": "Japan",
        "backup_supplier_id": "SUP-001",
        "contracted_lead_time_days": 16,
        "sku": "SKU-KYOTO-01",
        "current_stock_units": 650,
        "reorder_point": 200,
        "daily_consumption_rate": 30.0,
        "contact_email": unique_email,
        "cost_per_unit": 210.0,
    }

    # 1. Unauthenticated request rejected (401)
    unauth_res = client.post("/api/v1/suppliers", json=payload)
    assert unauth_res.status_code == 401

    # 2. Analyst request rejected by RBAC (403)
    analyst_headers = get_auth_headers(role="analyst", username="analyst_test")
    analyst_res = client.post("/api/v1/suppliers", json=payload, headers=analyst_headers)
    assert analyst_res.status_code == 403

    # 3. Manager successfully provisions supplier & inventory (201)
    mgr_headers = get_auth_headers(role="manager", username="mgr_test")
    response = client.post("/api/v1/suppliers", json=payload, headers=mgr_headers)
    assert response.status_code == 201
    created = response.json()["data"]
    assert created["name"] == payload["supplier_name"]
    assert created["contact_email"] == payload["contact_email"]
    assert created["status"] == "ACTIVE"
    supplier_id = created["id"]

    # 4. Retrieve created supplier
    get_res = client.get(f"/api/v1/suppliers/{supplier_id}")
    assert get_res.status_code == 200
    assert get_res.json()["data"]["id"] == supplier_id

    # 5. Duplicate ID rejected (409)
    dup_res = client.post("/api/v1/suppliers", json=payload, headers=mgr_headers)
    assert dup_res.status_code == 409
    assert dup_res.json()["error"]["code"] == "ENTITY_CONFLICT"


# ---------------------------------------------------------------------------
# 4. Anomalies & Telemetry Tests
# ---------------------------------------------------------------------------

def test_telemetry_detect_anomaly():
    """Validates detection scoring triggers anomaly when lead time variance is high."""
    telemetry = {
        "supplier_id": str(uuid.uuid4()),
        "sku": "SKU-SENSOR-707",
        "delivery_variance_days": 8.5,
        "port_delay_days": 4.0,
        "congestion_ratio": 0.78,
        "external_severity_score": 7.5,
        "order_fulfillment_ratio": 0.65,
        "lead_time_days": 14.0,
    }
    response = client.post("/api/v1/anomalies/detect", json=telemetry)
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    anomalies = body["data"]
    assert len(anomalies) >= 1
    assert any(a["anomaly_type"] in ("DELAY_SURGE", "ROUTE_CONGESTION") for a in anomalies)


def test_list_anomalies_filtered():
    """Validates listing anomalies with severity filtering."""
    response = client.get("/api/v1/anomalies?severity=CRITICAL")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    for item in body["data"]["items"]:
        assert item["severity"] == "CRITICAL"


# ---------------------------------------------------------------------------
# 5. Disruption Alerts & Human-in-the-Loop Lifecycle Tests
# ---------------------------------------------------------------------------

def test_get_active_alerts():
    """Validates retrieving unacknowledged critical/high alerts sorted by TTI."""
    response = client.get("/api/v1/alerts/active")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    alerts = body["data"]
    assert len(alerts) >= 1
    # Check all are unacknowledged and sorted by predicted_tti_days ascending
    for a in alerts:
        assert a["acknowledged"] is False
        assert a["risk_level"] in ("CRITICAL", "HIGH")
    ttis = [a["predicted_tti_days"] for a in alerts]
    assert ttis == sorted(ttis)


def test_acknowledge_alert_workflow():
    """
    Validates acknowledging an active alert and verifies subsequent duplicate
    acknowledgment raises InvalidStateTransitionException.
    """
    active_res = client.get("/api/v1/alerts/active")
    target_alert = active_res.json()["data"][0]
    alert_id = target_alert["alert_id"]

    ack_payload = {
        "user_id": "operator.david@fujitsu.com",
        "note": "Re-routed 500 units to Hamburg air freight corridor.",
    }
    patch_res = client.patch(f"/api/v1/alerts/{alert_id}/acknowledge", json=ack_payload)
    assert patch_res.status_code == 200
    updated = patch_res.json()["data"]
    assert updated["acknowledged"] is True
    assert updated["acknowledged_by"] == ack_payload["user_id"]
    assert updated["acknowledgment_note"] == ack_payload["note"]
    assert updated["acknowledged_at"] is not None

    # Duplicate acknowledgment attempt should raise 400
    dup_patch = client.patch(f"/api/v1/alerts/{alert_id}/acknowledge", json=ack_payload)
    assert dup_patch.status_code == 400
    assert dup_patch.json()["error"]["code"] == "INVALID_STATE_TRANSITION"
