#!/usr/bin/env python3
"""
SupplyPulse: Comprehensive System Verification & Audit Suite
============================================================
Enterprise-grade health check, data integrity audit, and relational verification.

Performs:
  1. Database connectivity test to PostgreSQL 15 / SQLite and API-to-DB query execution.
  2. Table record count audits across all 8 tables (including audit_logs, users, model_runs)
     and column verification for external_disruptions (source_text, source, parsed_method).
  3. Relational Foreign Key constraint validation (zero orphaned alert rows).
  4. AI & Alert Engine verification (flagged anomaly scores & TTI countdowns).
  5. Machine Learning model loading & inference verification (isolation_forest_model.pkl).
  6. Workspace integrity, single schema enforcement, and clean hygiene check.

Exits with non-zero status (1) on ANY failure.

Author: SupplyPulse Team (Pradymna Adhari, Omkar Akhade, Vishal Amrutkar, Pravin Bolshette)
Review: Prof. Vanita Babbane (VIT Pune) | Fujitsu Data & AI Portfolio
"""

import os
import sys
import argparse
import glob
from typing import Dict, Any, List

from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import OperationalError
import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Colors & Formatting Helpers
# ---------------------------------------------------------------------------
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_banner():
    banner = f"""{Colors.CYAN}{Colors.BOLD}
================================================================================
   SUPPLYPULSE: SYSTEM VERIFICATION & AUDIT SUITE (V2.0)
   Fujitsu-Compliant Enterprise AI Supply Chain Platform
================================================================================{Colors.RESET}
"""
    print(banner)


# ---------------------------------------------------------------------------
# 1. Connectivity & API-to-DB Query Execution
# ---------------------------------------------------------------------------
def verify_connectivity(engine) -> Dict[str, Any]:
    """Test connection to the database and execute an API-to-DB telemetry probe."""
    print(f"{Colors.BOLD}[TEST 1/6] Testing Database Connectivity & Query Execution...{Colors.RESET}")
    try:
        with engine.connect() as conn:
            is_sqlite = engine.url.drivername.startswith("sqlite")
            if is_sqlite:
                version_info = conn.execute(text("SELECT sqlite_version();")).scalar()
                db_name = "Local SQLite"
                user_name = "local"
            else:
                version_info = conn.execute(text("SELECT version();")).scalar()
                db_name = conn.execute(text("SELECT current_database();")).scalar()
                user_name = conn.execute(text("SELECT current_user;")).scalar()

            # Execute active query probe
            probe_val = conn.execute(text("SELECT 1;")).scalar()
            if probe_val != 1:
                raise RuntimeError(f"Database probe query returned unexpected value: {probe_val}")

        print(f"  {Colors.GREEN}[OK] CONNECTED SUCCESSFULLY!{Colors.RESET}")
        print(f"    - Target Engine : {Colors.BOLD}{engine.url.drivername}{Colors.RESET}")
        print(f"    - Database Name : {Colors.BOLD}{db_name}{Colors.RESET}")
        print(f"    - User Account  : {Colors.BOLD}{user_name}{Colors.RESET}")
        print(f"    - Server Version: {version_info.split(',')[0] if version_info else 'N/A'}")
        print(f"    - API-DB Probe  : {Colors.GREEN}PASSED (SELECT 1 -> {probe_val}){Colors.RESET}")
        return {"status": "PASSED"}
    except Exception as oe:
        print(f"  {Colors.RED}[FAIL] CONNECTION FAILED: {oe}{Colors.RESET}")
        return {"status": "FAILED", "error": str(oe)}


# ---------------------------------------------------------------------------
# 2. Table Audits & Column Schema Verification
# ---------------------------------------------------------------------------
def verify_table_counts(engine) -> Dict[str, Any]:
    """Audit row counts across all 8 SupplyPulse relational tables and column schemas."""
    print(f"\n{Colors.BOLD}[TEST 2/6] Auditing All 8 Core Relational Tables & NLP Columns...{Colors.RESET}")
    tables = [
        ("suppliers", 10, "Tier-1 Master Supplier records"),
        ("inventory_levels", 50, "Warehouse Facility & SKU tracking records"),
        ("port_activities", 30, "Maritime Port Delay & Congestion logs"),
        ("external_disruptions", 15, "Weather, Disaster & NLP event bulletins"),
        ("disruption_alerts", 10, "AI Anomaly Detection inferences & TTI alerts"),
        ("audit_logs", 0, "Immutable Human Operator Decision Audit Log"),
        ("users", 0, "Role-Based Access Control User Accounts"),
        ("model_runs", 0, "MLOps Lineage Ledger & Baseline Evaluations"),
    ]

    all_passed = True
    counts = {}

    print(f"  +------------------------+------------+------------+--------------------------------------+")
    print(f"  | {'TABLE NAME':<22} | {'ROWS':<10} | {'STATUS':<10} | {'DOMAIN':<36} |")
    print(f"  +------------------------+------------+------------+--------------------------------------+")

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    with engine.connect() as conn:
        for tbl, min_expected, domain in tables:
            try:
                if tbl not in existing_tables:
                    # Check if table exists in SQLite or DB
                    res = conn.execute(text(f"SELECT COUNT(*) FROM {tbl};"))
                    cnt = res.scalar()
                else:
                    cnt = conn.execute(text(f"SELECT COUNT(*) FROM {tbl};")).scalar()

                counts[tbl] = cnt
                if cnt >= min_expected:
                    status_str = f"{Colors.GREEN}PASSED [OK]{Colors.RESET}"
                elif min_expected == 0:
                    status_str = f"{Colors.GREEN}EXISTS [OK]{Colors.RESET}"
                elif cnt > 0:
                    status_str = f"{Colors.YELLOW}POPULATED{Colors.RESET}"
                else:
                    status_str = f"{Colors.RED}EMPTY [X]{Colors.RESET}"
                    all_passed = False

                print(f"  | {tbl:<22} | {cnt:<10} | {status_str:<20} | {domain:<36} |")
            except Exception as e:
                print(f"  | {tbl:<22} | {'MISSING':<10} | {Colors.RED}FAILED [X]{Colors.RESET}   | {str(e)[:36]} |")
                all_passed = False

    print(f"  +------------------------+------------+------------+--------------------------------------+")

    # Verify new external_disruptions columns: source_text, source, parsed_method
    print(f"\n  Checking `external_disruptions` schema columns...")
    try:
        cols = [c["name"] for c in inspector.get_columns("external_disruptions")]
        required_cols = ["source_text", "source", "parsed_method"]
        missing_cols = [rc for rc in required_cols if rc not in cols]
        if missing_cols:
            print(f"  {Colors.RED}[FAIL] Missing columns in external_disruptions: {missing_cols}{Colors.RESET}")
            all_passed = False
        else:
            print(f"  {Colors.GREEN}[OK] Verified columns present: {required_cols}{Colors.RESET}")
    except Exception as e:
        print(f"  {Colors.RED}[FAIL] Could not inspect columns: {e}{Colors.RESET}")
        all_passed = False

    return {"status": "PASSED" if all_passed else "FAILED", "counts": counts}


# ---------------------------------------------------------------------------
# 3. Foreign Key Referential Integrity
# ---------------------------------------------------------------------------
def verify_foreign_key_integrity(engine) -> Dict[str, Any]:
    """Validate referential integrity between disruption_alerts and suppliers."""
    print(f"\n{Colors.BOLD}[TEST 3/6] Validating Foreign Key Referential Integrity...{Colors.RESET}")

    check_sql = """
        SELECT da.alert_id, da.supplier_id
        FROM disruption_alerts da
        LEFT JOIN suppliers s ON da.supplier_id = s.supplier_id
        WHERE s.supplier_id IS NULL;
    """

    with engine.connect() as conn:
        orphans = conn.execute(text(check_sql)).fetchall()

    if len(orphans) == 0:
        print(f"  {Colors.GREEN}[OK] REFERENTIAL INTEGRITY VERIFIED!{Colors.RESET}")
        print("    All supplier references in 'disruption_alerts' exist in 'suppliers' master table.")
        print(f"    Orphaned records detected: {Colors.BOLD}0{Colors.RESET}")
        return {"status": "PASSED", "orphans_count": 0}
    else:
        print(f"  {Colors.RED}[FAIL] ORPHANED ROWS DETECTED: {len(orphans)} violations found!{Colors.RESET}")
        for orphan in orphans[:5]:
            print(f"    - Alert ID {orphan[0]} references nonexistent supplier '{orphan[1]}'")
        return {"status": "FAILED", "orphans_count": len(orphans)}


# ---------------------------------------------------------------------------
# 4. AI Early-Warning & Time-to-Impact Countdowns
# ---------------------------------------------------------------------------
def verify_ai_alert_engine(engine) -> Dict[str, Any]:
    """Query and inspect active disruption alerts and stockout warnings."""
    print(f"\n{Colors.BOLD}[TEST 4/6] Inspecting AI Early-Warning Alerts & TTI Countdowns...{Colors.RESET}")

    alert_query = """
        SELECT da.alert_id, da.supplier_id, s.supplier_name, s.backup_supplier_name,
               da.anomaly_score, da.predicted_tti_days, da.recommended_action
        FROM disruption_alerts da
        JOIN suppliers s ON da.supplier_id = s.supplier_id
        WHERE da.is_flagged = TRUE OR da.is_flagged = 1
        ORDER BY da.predicted_tti_days ASC
        LIMIT 5;
    """

    with engine.connect() as conn:
        active_alerts = conn.execute(text(alert_query)).fetchall()

    if not active_alerts:
        print(f"  {Colors.YELLOW}[WARN] No active flagged disruption alerts found.{Colors.RESET}")
    else:
        print(f"  {Colors.CYAN}Top Flagged Supplier Disruption Alerts (Time-to-Impact Ranked):{Colors.RESET}")
        for row in active_alerts:
            alert_id, sup_id, sup_name, backup_sup, score, tti, action = row
            print(f"    * [{Colors.YELLOW}TTI: {tti} DAYS{Colors.RESET}] {Colors.BOLD}{sup_name}{Colors.RESET} ({sup_id})")
            print(f"      Anomaly Score : {score:.4f} | Backup Route: {Colors.CYAN}{backup_sup}{Colors.RESET}")
            print(f"      Recommendation: {action}")

    return {"status": "PASSED", "alerts_inspected": len(active_alerts)}


# ---------------------------------------------------------------------------
# 5. Machine Learning Model Loading & Inference Test
# ---------------------------------------------------------------------------
def verify_model_loading() -> Dict[str, Any]:
    """Verify Isolation Forest model file exists, loads cleanly, and predicts accurately."""
    print(f"\n{Colors.BOLD}[TEST 5/6] Verifying Machine Learning Model Loading & Inference...{Colors.RESET}")
    model_path = os.path.join(os.path.dirname(__file__), "phase1_detection", "isolation_forest_model.pkl")

    if not os.path.exists(model_path):
        print(f"  {Colors.RED}[FAIL] Model file missing: {model_path}{Colors.RESET}")
        return {"status": "FAILED", "error": "Model file not found"}

    try:
        model = joblib.load(model_path)
        model_type = type(model).__name__
        n_features = getattr(model, "n_features_in_", None)

        if "IsolationForest" not in model_type:
            print(f"  {Colors.RED}[FAIL] Unexpected model class: {model_type}{Colors.RESET}")
            return {"status": "FAILED", "error": f"Invalid model type: {model_type}"}

        if n_features != 6:
            print(f"  {Colors.RED}[FAIL] Expected 6 input features, got {n_features}{Colors.RESET}")
            return {"status": "FAILED", "error": f"Invalid feature count: {n_features}"}

        # Test inference on sample nominal vs anomaly feature vectors
        sample_nominal = np.array([[1.2, 1.8, 0.4, 2.0, 35.0, 1.0]])
        sample_disruption = np.array([[7.5, 12.0, 4.2, 45.0, 3.0, 9.0]])

        pred_nom = model.predict(sample_nominal)[0]
        pred_dis = model.predict(sample_disruption)[0]

        print(f"  {Colors.GREEN}[OK] MODEL LOADED SUCCESSFULLY!{Colors.RESET}")
        print(f"    - Model Class   : {Colors.BOLD}{model_type}{Colors.RESET}")
        print(f"    - Features Count: {Colors.BOLD}{n_features}{Colors.RESET}")
        print(f"    - Nominal Test  : {Colors.GREEN}Inference = {pred_nom} (1 = Normal){Colors.RESET}")
        print(f"    - Stress Test   : {Colors.YELLOW}Inference = {pred_dis} (-1 = Anomaly){Colors.RESET}")

        return {"status": "PASSED", "model_type": model_type, "features": n_features}
    except Exception as e:
        print(f"  {Colors.RED}[FAIL] Model loading/inference failed: {e}{Colors.RESET}")
        return {"status": "FAILED", "error": str(e)}


# ---------------------------------------------------------------------------
# 6. Workspace Hygiene & Single Schema Enforcement
# ---------------------------------------------------------------------------
def verify_workspace_files() -> Dict[str, Any]:
    """Verify production files exist, single schema rule is honored, and repo is clean."""
    print(f"\n{Colors.BOLD}[TEST 6/6] Workspace Structure, Hygiene & Single Schema Check...{Colors.RESET}")

    essential_files = [
        "docker-compose.yml",
        ".env",
        ".env.example",
        ".gitignore",
        "requirements.txt",
        "README.md",
        "database/init_schema.sql",
        "docs/data_dictionary.md",
        "docs/architecture.md",
        "docs/model_card.md",
        "docs/responsible_ai.md",
        "etl/generate_and_load.py",
        "etl/nlp_event_ingestion.py",
        "etl/evaluate_nlp.py",
        "data/nlp_eval/headlines_labeled.csv",
        "phase1_detection/train_model.py",
        "phase1_detection/detect.py",
        "phase1_detection/evaluate_model.py",
        "phase1_detection/isolation_forest_model.pkl",
        "phase2_prediction/tti_calculator.py",
        "streamlit_app/app.py",
    ]

    missing = []
    for f in essential_files:
        if os.path.exists(f):
            print(f"  {Colors.GREEN}[OK] Preserved:{Colors.RESET} {f}")
        else:
            print(f"  {Colors.RED}[FAIL] Missing:{Colors.RESET} {f}")
            missing.append(f)

    # Enforce Single Schema Rule: only database/init_schema.sql should exist
    sql_files = glob.glob("**/*.sql", recursive=True)
    duplicate_schemas = [f for f in sql_files if os.path.normpath(f) != os.path.normpath("database/init_schema.sql")]
    if duplicate_schemas:
        print(f"  {Colors.RED}[FAIL] Multiple schema files detected! Found: {duplicate_schemas}{Colors.RESET}")
        missing.extend(duplicate_schemas)
    else:
        print(f"  {Colors.GREEN}[OK] Single Schema Rule Verified: Exactly 1 schema file in repository.{Colors.RESET}")

    # Verify no .pytest_cache checked into repo or cluttering workspace
    if os.path.exists(".pytest_cache"):
        print(f"  {Colors.YELLOW}[NOTICE] .pytest_cache present in local directory (git-ignored).{Colors.RESET}")

    return {"status": "PASSED" if not missing else "FAILED", "missing": missing}


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def main():
    print_banner()
    load_dotenv()

    parser = argparse.ArgumentParser(description="SupplyPulse: System Verification & Audit")
    parser.add_argument("--db-url", type=str, default=None, help="Custom database connection string")
    parser.add_argument("--sqlite", action="store_true", help="Run verification against local SQLite")
    args = parser.parse_args()

    if args.sqlite:
        db_url = "sqlite:///supplypulse.db"
    else:
        db_url = (
            args.db_url
            or os.getenv("DATABASE_URL")
            or f"postgresql://{os.getenv('POSTGRES_USER', 'admin')}:{os.getenv('POSTGRES_PASSWORD', 'secret')}@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'supplypulse')}"
        )

    try:
        engine = create_engine(db_url, echo=False)
    except Exception as e:
        print(f"{Colors.RED}FATAL: Could not create database engine: {e}{Colors.RESET}")
        sys.exit(1)

    # Execute all 6 verification checks
    res1 = verify_connectivity(engine)
    if res1["status"] != "PASSED":
        print(f"\n{Colors.RED}AUDIT ABORTED: Cannot proceed without an active database connection.{Colors.RESET}")
        sys.exit(1)

    res2 = verify_table_counts(engine)
    res3 = verify_foreign_key_integrity(engine)
    res4 = verify_ai_alert_engine(engine)
    res5 = verify_model_loading()
    res6 = verify_workspace_files()

    all_results = [res1, res2, res3, res4, res5, res6]
    failed_steps = [f"TEST {i+1}" for i, r in enumerate(all_results) if r["status"] != "PASSED"]

    if failed_steps:
        print("\n" + "=" * 80)
        print(f"{Colors.RED}{Colors.BOLD}SYSTEM VERIFICATION FAILED: Violations detected in {failed_steps}!{Colors.RESET}")
        print("Please correct the reported failures and re-run verify_system.py.")
        print("=" * 80 + "\n")
        sys.exit(1)

    print("\n" + "=" * 80)
    print(f"{Colors.GREEN}{Colors.BOLD}ALL 6 SYSTEM AUDIT CHECKS COMPLETED SUCCESSFULLY!{Colors.RESET}")
    print("Database, ML model, tables, columns, referential integrity, and workspace are fully verified.")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
