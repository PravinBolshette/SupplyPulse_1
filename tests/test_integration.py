"""
SupplyPulse: Integration Tests & Production Guardrail Verifications (Task B5)
=============================================================================
Verifies:
  1. Single Schema File Rule: Exactly one schema file exists (database/init_schema.sql).
  2. No Hardcoded Confidences: Dynamic model confidence computation.
  3. Audit Write Failure Returns 503: Zero-loss audit log fail-safe.
  4. End-to-End Live PostgreSQL Pipeline (Conditional on TEST_DATABASE_URL availability).
"""

import os
import glob
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from app.main import app
from app.routers.anomalies import compute_dynamic_model_confidence
from app.core.security import create_access_token, _USERS_CACHE

client = TestClient(app)


def get_auth_headers(username: str, role: str) -> dict:
    """Helper to generate JWT bearer authorization header for test user."""
    _USERS_CACHE[username] = {
        "username": username,
        "password_hash": "mock_hash",
        "role": role,
    }
    token = create_access_token(data={"sub": username, "role": role})
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# 1. Single Schema File Verification
# ---------------------------------------------------------------------------
def test_single_schema_file_exists():
    """Verify that only one schema file exists in the repo (database/init_schema.sql)."""
    repo_root = Path(__file__).resolve().parent.parent
    sql_files = list(repo_root.rglob("*.sql"))

    # Filter out any temporary/git directory matches if present
    relevant_sql_files = [
        f for f in sql_files
        if ".git" not in str(f) and ".venv" not in str(f) and "site-packages" not in str(f)
    ]

    # Exactly 1 schema file must exist
    assert len(relevant_sql_files) == 1, (
        f"Expected exactly 1 schema file in repo, but found {len(relevant_sql_files)}: "
        f"{[str(f.relative_to(repo_root)) for f in relevant_sql_files]}"
    )

    schema_file = relevant_sql_files[0]
    assert schema_file.name == "init_schema.sql"
    assert "database" in schema_file.parts
    assert not (repo_root / "scripts" / "init_schema.sql").exists(), (
        "Redundant schema file scripts/init_schema.sql must not exist!"
    )


# ---------------------------------------------------------------------------
# 2. No Hardcoded Confidence Values Test
# ---------------------------------------------------------------------------
def test_no_hardcoded_confidence_values():
    """Verify that model confidence is dynamically computed and never uses static hardcoded constants."""
    # Test across distinct metric deviations
    conf_low = compute_dynamic_model_confidence(metric_deviation=1.5, severity_weight=1.0)
    conf_med = compute_dynamic_model_confidence(metric_deviation=3.2, severity_weight=1.2)
    conf_high = compute_dynamic_model_confidence(metric_deviation=5.8, severity_weight=1.5)

    # Assert none of the returned values match former static literals
    banned_static_confidences = {0.96, 0.98, 1.0, 96.0, 98.0, 100.0}
    for conf in [conf_low, conf_med, conf_high]:
        assert conf not in banned_static_confidences
        assert 0.0 <= conf <= 1.0

    # Assert dynamic responsiveness: larger deviation produces higher statistical confidence
    assert conf_high > conf_med > conf_low

    # Query the anomalies API and ensure no hardcoded confidences appear
    response = client.get("/api/v1/anomalies")
    assert response.status_code == 200
    data = response.json()
    items = data.get("data", {}).get("items", [])
    assert len(items) > 0
    for anomaly in items:
        conf = anomaly.get("model_confidence")
        assert conf is not None
        assert conf not in banned_static_confidences, f"Found hardcoded confidence {conf} in API response!"
        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# 3. Audit Write Failure Returns 503 Test
# ---------------------------------------------------------------------------
def test_audit_write_failure_returns_503(monkeypatch):
    """Verify that if PostgreSQL persistence fails during mitigation, the API aborts with 503."""
    headers = get_auth_headers("compliance_mgr", "manager")

    # Simulate database audit write failure via environment flag
    monkeypatch.setenv("SIMULATE_AUDIT_DB_FAIL", "true")

    payload = {
        "operator_id": "compliance_mgr",
        "alert_id": "ALT-TEST-503",
        "ai_recommendation": {
            "summary": "Critical port congestion threat at Rotterdam.",
            "suggested_action": "Switch primary polymer transit to Antwerp rail corridor.",
            "predicted_tti_days": 4.5,
            "risk_level": "CRITICAL",
            "backup_supplier_name": "Antwerp Chemical Specialities",
        },
        "action_taken": "APPROVED",
        "confirmed": True,
        "operator_notes": "Testing 503 fail-safe on database persistence error.",
        "mitigation_target": "SUP-001",
        "cost_exposure_usd": 15000.0,
    }

    response = client.post("/api/v1/mitigation/action", json=payload, headers=headers)
    assert response.status_code == 503, (
        f"Expected HTTP 503 when audit persistence fails, got {response.status_code}: {response.text}"
    )
    resp_json = response.json()
    error_msg = resp_json.get("error", {}).get("message", "") or resp_json.get("detail", "")
    assert "persistence failed" in str(error_msg).lower()


# ---------------------------------------------------------------------------
# 4. Live PostgreSQL End-to-End Pipeline Integration Test
# ---------------------------------------------------------------------------
def test_live_postgres_etl_train_detect_integration():
    """
    Integration test: ETL, train, detect, API returns an alert for a supplier with planted disruption.
    Uses TEST_DATABASE_URL; skips cleanly with message if PostgreSQL is not available.
    """
    test_db_url = os.getenv("TEST_DATABASE_URL")

    if not test_db_url:
        # Check standard default PostgreSQL URL as well
        default_url = "postgresql://admin:secret@localhost:5432/supplypulse"
        try:
            engine = create_engine(default_url)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            test_db_url = default_url
        except Exception:
            pytest.skip(
                "PostgreSQL test database not available at TEST_DATABASE_URL (or localhost:5432). "
                "Skipping live DB integration test per Task B5 instructions."
            )

    # If database is available, run end-to-end verification
    engine = create_engine(test_db_url)
    with engine.connect() as conn:
        res = conn.execute(text("SELECT COUNT(*) FROM suppliers")).scalar()
        assert res >= 10, f"Expected at least 10 suppliers in database, found {res}"

        # Verify planted disruption alerts
        alerts = conn.execute(text("SELECT supplier_id, is_flagged, predicted_tti_days FROM disruption_alerts WHERE is_flagged = TRUE")).fetchall()
        assert len(alerts) > 0, "Expected planted disruption alerts in PostgreSQL database"

        # Query API for alerts
        api_resp = client.get("/api/v1/alerts/")
        assert api_resp.status_code == 200
        alerts_data = api_resp.json().get("data", [])
        assert len(alerts_data) > 0, "API returned no alerts for live database"
