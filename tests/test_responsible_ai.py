"""
Automated Test Suite for Responsible AI Guardrails & Human-in-the-Loop (HITL) Framework.

Validates:
1. Advisory-Only AI Constraint: Rejects autonomous execution attempts by AI/system bots.
2. Mandatory Human Confirmation: Blocks actions without explicit confirmation.
3. Override Accountability: Requires detailed operator justification for overrides and rejections.
4. Immutable Audit Ledger: Records every authorized mitigation with UUID, operator ID, and timestamp.
5. Audit Query & Policy Posture Endpoints.
"""

from fastapi.testclient import TestClient
from app.core.security import create_access_token
from app.main import app

client = TestClient(app)

SAMPLE_AI_RECOMMENDATION = {
    "summary": "Critical port congestion threat at Rotterdam.",
    "suggested_action": "Switch primary polymer transit to Antwerp rail corridor.",
    "predicted_tti_days": 4.5,
    "risk_level": "CRITICAL",
    "target_sku": "SKU-POLYMER-303",
    "backup_supplier_name": "Antwerp Chemical Specialities",
}


def get_auth_headers(username="manager", role="manager"):
    from datetime import datetime, timezone
    from app.core.security import create_access_token, _USERS_CACHE, hash_password
    clean = username.strip().lower()
    if clean not in _USERS_CACHE:
        _USERS_CACHE[clean] = {
            "user_id": f"test-uuid-{clean}",
            "username": clean,
            "password_hash": hash_password("password123"),
            "role": role,
            "created_at": datetime.now(timezone.utc),
        }
    token = create_access_token(data={"sub": clean, "role": role})
    return {"Authorization": f"Bearer {token}"}



def test_advisory_constraint_blocks_autonomous_ai_execution():
    """
    Validates that AI models and system bots cannot execute write actions autonomously.
    """
    for bot_id in ["system", "ai_agent_v1", "bot_autopilot", "automated_service", "cron_job"]:
        payload = {
            "operator_id": bot_id,
            "alert_id": "aaaaaaaa-1111-1111-1111-aaaaaaaaaaaa",
            "ai_recommendation": SAMPLE_AI_RECOMMENDATION,
            "action_taken": "APPROVED",
            "operator_notes": "Automated pipeline trigger",
            "confirmed": True,
        }
        response = client.post("/api/v1/mitigation/action", json=payload, headers=get_auth_headers())
        assert response.status_code == 422
        body = response.json()
        assert body["success"] is False
        assert "AUTONOMOUS_EXECUTION_BLOCKED" in str(body["error"])



def test_human_confirmation_strictly_required():
    """
    Validates that missing explicit human confirmation raises 422.
    """
    payload = {
        "operator_id": "operator.vanita@fujitsu.com",
        "alert_id": "aaaaaaaa-1111-1111-1111-aaaaaaaaaaaa",
        "ai_recommendation": SAMPLE_AI_RECOMMENDATION,
        "action_taken": "APPROVED",
        "operator_notes": "Reviewed and agreed.",
        "confirmed": False,  # Missing confirmation
    }
    response = client.post("/api/v1/mitigation/action", json=payload, headers=get_auth_headers())
    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert "Human confirmation required" in str(body["error"])


def test_override_justification_enforced():
    """
    Validates that overriding or rejecting AI advice requires detailed operator notes.
    """
    for action in ["OVERRIDDEN", "REJECTED"]:
        payload = {
            "operator_id": "operator.vanita@fujitsu.com",
            "alert_id": "aaaaaaaa-1111-1111-1111-aaaaaaaaaaaa",
            "ai_recommendation": SAMPLE_AI_RECOMMENDATION,
            "action_taken": action,
            "operator_notes": None,  # Missing required justification
            "confirmed": True,
        }
        response = client.post("/api/v1/mitigation/action", json=payload, headers=get_auth_headers())
        assert response.status_code == 422
        body = response.json()
        assert body["success"] is False
        assert "Accountability policy requires a detailed operator justification" in str(body["error"])


def test_successful_human_approved_mitigation():
    """
    Validates a valid human-authorized mitigation approval creates an audit record.
    """
    payload = {
        "operator_id": "operator.lead@supplypulse.ai",
        "alert_id": "aaaaaaaa-1111-1111-1111-aaaaaaaaaaaa",
        "ai_recommendation": SAMPLE_AI_RECOMMENDATION,
        "action_taken": "APPROVED",
        "operator_notes": "Reviewed TTI model. Authorizing Tokyo Semiconductor expedited shipment.",
        "confirmed": True,
        "mitigation_target": "Tokyo Semiconductor",
        "cost_exposure_usd": 420000.0,
    }
    response = client.post(
        "/api/v1/mitigation/action",
        json=payload,
        headers=get_auth_headers("operator.lead@supplypulse.ai", "manager"),
    )
    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True
    data = body["data"]
    assert "log_id" in data
    assert data["operator_id"] == "operator.lead@supplypulse.ai"
    assert data["action_taken"] == "APPROVED"
    assert data["confirmed"] is True
    assert data["audit_immutable"] is True


def test_successful_human_override_with_justification():
    """
    Validates an operator overriding the AI recommendation with valid notes succeeds.
    """
    payload = {
        "operator_id": "operator.senior@supplypulse.ai",
        "alert_id": "bbbbbbbb-2222-2222-2222-bbbbbbbbbbbb",
        "ai_recommendation": SAMPLE_AI_RECOMMENDATION,
        "action_taken": "OVERRIDDEN",
        "operator_notes": "Local inventory warehouse has 10-day buffer; delaying PO until Monday cycle.",
        "confirmed": True,
        "mitigation_target": "Internal Safety Stock",
    }
    response = client.post(
        "/api/v1/mitigation/action",
        json=payload,
        headers=get_auth_headers("operator.senior@supplypulse.ai", "admin"),
    )
    assert response.status_code == 201
    data = response.json()["data"]
    assert data["action_taken"] == "OVERRIDDEN"
    assert data["operator_notes"] == payload["operator_notes"]



def test_audit_logs_paginated_query():
    """
    Validates querying the immutable audit logs trail.
    """
    response = client.get("/api/v1/mitigation/audit-logs?limit=5")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    items = body["data"]["items"]
    assert len(items) >= 2
    # Ensure newest logs are returned first
    timestamps = [item["timestamp"] for item in items]
    assert timestamps == sorted(timestamps, reverse=True)


def test_guardrails_status_endpoint():
    """
    Validates the policy posture reporting endpoint.
    """
    response = client.get("/api/v1/mitigation/guardrails/status")
    assert response.status_code == 200
    data = response.json()
    assert data["advisory_mode_enforced"] is True
    assert data["autonomous_execution_allowed"] is False
    assert data["human_in_the_loop_required"] is True
    assert data["override_justification_enforced"] is True
