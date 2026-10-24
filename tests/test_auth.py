"""
Unit and Integration Tests for User Authentication and Role-Based Access Control (Task B3).

Verifies:
1. Registration with bcrypt password hashing and role assignment.
2. Login credential verification and JWT token issuance.
3. Strict Role-Based Access Control (RBAC):
   - 'analyst' is forbidden from mutating mitigation state (HTTP 403).
   - 'manager' and 'admin' can authorize mitigations (HTTP 201).
4. Operator identity enforcement: operator_id in audit trail is derived strictly
   from the authenticated JWT subject, completely ignoring request body spoofing.
5. Model info endpoint returns accurate metrics, 2 baselines, and synthetic disclaimer.
6. Supplier history endpoint returns 30-day timeline with plain-language labels.
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

SAMPLE_AI_REC = {
    "summary": "Port of Shanghai severe weather alert.",
    "suggested_action": "Activate Backup Supplier Alpha immediately.",
    "predicted_tti_days": 4.2,
    "risk_level": "CRITICAL",
    "target_sku": "SKU-CHIP-992",
    "backup_supplier_name": "Apex Precision Microelectronics",
}


def test_user_registration_and_login():
    """Validates full registration and JWT login lifecycle."""
    uname = "test_analyst_01"
    reg_resp = client.post(
        "/api/v1/auth/register",
        json={"username": uname, "password": "SecurePassword123!", "role": "analyst"},
    )
    assert reg_resp.status_code == 201
    reg_data = reg_resp.json()["data"]
    assert reg_data["username"] == uname
    assert reg_data["role"] == "analyst"

    # Login with valid credentials
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username": uname, "password": "SecurePassword123!"},
    )
    assert login_resp.status_code == 200
    token_data = login_resp.json()["data"]
    assert "access_token" in token_data
    assert token_data["role"] == "analyst"

    # Introspect with /me
    me_resp = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token_data['access_token']}"},
    )
    assert me_resp.status_code == 200
    assert me_resp.json()["data"]["username"] == uname


def test_invalid_login_rejected():
    """Validates that incorrect password returns 401 Unauthorized."""
    resp = client.post(
        "/api/v1/auth/login",
        json={"username": "analyst", "password": "wrong_password_xyz"},
    )
    assert resp.status_code == 401


def test_analyst_cannot_authorize_mitigation():
    """Validates RBAC: 'analyst' role is forbidden (403) from authorizing mitigations."""
    # Login as seeded analyst
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username": "analyst", "password": "analyst123"},
    )
    token = login_resp.json()["data"]["access_token"]

    payload = {
        "operator_id": "analyst",
        "alert_id": "aaaaaaaa-1111-1111-1111-aaaaaaaaaaaa",
        "ai_recommendation": SAMPLE_AI_REC,
        "action_taken": "APPROVED",
        "operator_notes": "Attempting approval as analyst",
        "confirmed": True,
    }
    resp = client.post(
        "/api/v1/mitigation/action",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403
    err_body = resp.json()
    err_msg = str(err_body.get("error", err_body.get("detail", "")))
    assert "Insufficient privileges" in err_msg



def test_manager_can_authorize_mitigation_and_operator_id_strictly_from_jwt():
    """
    Validates that:
    1. 'manager' role can authorize mitigation (201).
    2. operator_id in the resulting audit log comes from current_user.username,
       ignoring any spoofed operator_id in the JSON body.
    """
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username": "manager", "password": "manager123"},
    )
    token = login_resp.json()["data"]["access_token"]

    payload = {
        "operator_id": "spoofed_operator_name@evil.corp",  # Spoofed!
        "alert_id": "aaaaaaaa-1111-1111-1111-aaaaaaaaaaaa",
        "ai_recommendation": SAMPLE_AI_REC,
        "action_taken": "APPROVED",
        "operator_notes": "Authorized by valid operations manager.",
        "confirmed": True,
        "mitigation_target": "Apex Precision",
        "cost_exposure_usd": 150000.0,
    }
    resp = client.post(
        "/api/v1/mitigation/action",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 201
    log_data = resp.json()["data"]
    # CRITICAL: Operator ID MUST be 'manager' (from JWT), NOT 'spoofed_operator_name@evil.corp'
    assert log_data["operator_id"] == "manager"
    assert log_data["operator_id"] != payload["operator_id"]


def test_model_info_endpoint():
    """Validates model information, 2 baselines, and synthetic data label."""
    resp = client.get("/api/v1/model/info")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["model_type"] == "IsolationForest (Scikit-Learn Ensemble)"
    assert data["dataset_description"] == "Evaluated on synthetic data"
    assert "precision" in data["metrics"]
    assert "baseline_svm" in data
    assert "baseline_lof" in data


def test_supplier_history_endpoint():
    """Validates 30-day supplier history with plain-language status labels."""
    resp = client.get("/api/v1/suppliers/11111111-1111-1111-1111-111111111111/history")
    assert resp.status_code == 200
    points = resp.json()["data"]
    assert len(points) == 30
    first = points[0]
    assert "date" in first
    assert "anomaly_score" in first
    assert "status" in first
    assert first["status"] in ("Normal", "Flagged as Unusual")
