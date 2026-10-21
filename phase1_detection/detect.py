#!/usr/bin/env python3
"""
SupplyPulse: Phase 1 Inference & Alert Detection Engine
======================================================
Real-time anomaly inference and Time-to-Impact (TTI) alert generation.

Loads the pre-trained `isolation_forest_model.pkl`, queries live multi-source
telemetry across suppliers, port congestion, warehouse inventory, and NLP-parsed
external disruptions, computes anomaly scores, calculates deterministic TTI
countdowns, and updates the `disruption_alerts` table in PostgreSQL.

Author: SupplyPulse Team (Pradymna Adhari, Omkar Akhade, Vishal Amrutkar, Pravin Bolshette)
Review: Prof. Vanita Babbane (VIT Pune) | Fujitsu Data & AI Portfolio
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SupplyPulse.Detect")

# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "isolation_forest_model.pkl")

FEATURE_COLUMNS = [
    "avg_port_congestion_delay",
    "max_port_congestion_delay",
    "delivery_delay_variance",
    "order_volume_drop_pct",
    "stock_buffer_days",
    "external_disruption_severity",
]

# Regional Mapping: Link Suppliers to Ports, Facilities & Disruption Regions
SUPPLIER_REGIONAL_MAP = {
    "China": {
        "ports": ["Port of Shanghai", "Port of Ningbo-Zhoushan"],
        "facility": "WH-SHG-01",
        "regions": ["East China Sea / Yangtze Delta", "East China Sea / Ningbo Hub", "South China Sea"],
    },
    "Germany": {
        "ports": ["Port of Hamburg"],
        "facility": "WH-HAM-01",
        "regions": ["Elbe Estuary & Northern Germany"],
    },
    "Netherlands": {
        "ports": ["Port of Rotterdam", "Port of Antwerp"],
        "facility": "WH-ROT-01",
        "regions": ["Rhine-Meuse-Scheldt Delta", "Scheldt Estuary / Antwerp Hub"],
    },
    "India": {
        "ports": ["Port of Chennai"],
        "facility": "WH-CHE-01",
        "regions": ["Coromandel Coast / Bay of Bengal", "Southern India Tech Corridor"],
    },
    "Singapore": {
        "ports": ["Port of Singapore"],
        "facility": "WH-SIN-01",
        "regions": ["Strait of Malacca / Jurong", "Strait of Malacca"],
    },
    "Japan": {
        "ports": ["Port of Busan", "Port of Shanghai"],
        "facility": "WH-SIN-01",
        "regions": ["Tokyo Bay & Kanto Coastal Plain", "Korea Strait / Busan Maritime Hub"],
    },
    "Taiwan": {
        "ports": ["Port of Ningbo-Zhoushan", "Port of Shanghai"],
        "facility": "WH-SHG-01",
        "regions": ["Taiwan Strait / Hsinchu Corridor", "Taiwan Strait"],
    },
}


# ---------------------------------------------------------------------------
# Feature Extraction from Relational Database
# ---------------------------------------------------------------------------
def extract_live_supplier_features(engine) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Queries live relational tables in PostgreSQL, aggregates multi-source telemetry
    for every Tier-1 supplier, and constructs the current operational feature matrix.
    """
    logger.info("Extracting live multi-source features from PostgreSQL tables...")

    with engine.connect() as conn:
        suppliers_df = pd.read_sql("SELECT * FROM suppliers ORDER BY supplier_id;", conn)
        port_df = pd.read_sql("SELECT * FROM port_activities;", conn)
        inv_df = pd.read_sql("SELECT * FROM inventory_levels;", conn)
        dis_df = pd.read_sql("SELECT * FROM external_disruptions;", conn)

    logger.info(
        f"Retrieved {len(suppliers_df)} suppliers, {len(port_df)} port logs, "
        f"{len(inv_df)} inventory records, and {len(dis_df)} disruption events."
    )

    feature_rows = []
    metadata_rows = []

    # Supplier-to-SKU mapping to evaluate supplier-specific component stockouts
    sku_catalog = [
        "SKU-SEMI-101", "SKU-OPT-520", "SKU-HYDR-412", "SKU-SENS-204",
        "SKU-POWR-309", "SKU-ACT-631", "SKU-RELAY-740", "SKU-MCU-855",
        "SKU-DISP-962", "SKU-CAP-078", "SKU-CONN-189", "SKU-MEM-290"
    ]

    for idx, sup in suppliers_df.iterrows():
        sup_id = sup["supplier_id"]
        sup_name = sup["supplier_name"]
        country = sup["country"]
        backup_sup = sup["backup_supplier_name"] or "Global Alternate Logistics"
        primary_sku = sku_catalog[idx % len(sku_catalog)]

        reg_info = SUPPLIER_REGIONAL_MAP.get(country, SUPPLIER_REGIONAL_MAP["China"])

        # 1. Port Congestion Metrics
        matched_ports = port_df[port_df["port_name"].isin(reg_info["ports"])]
        if not matched_ports.empty:
            avg_port_delay = float(matched_ports["congestion_delay_days"].mean())
            max_port_delay = float(matched_ports["congestion_delay_days"].max())
            std_port_delay = float(matched_ports["congestion_delay_days"].std() or 0.5)
        else:
            avg_port_delay = 1.5
            max_port_delay = 2.5
            std_port_delay = 0.5

        # Delivery delay variance correlates with maritime volatility
        deliv_variance = float(round(std_port_delay * 1.2, 2))

        # 2. Facility Inventory Buffer & TTI Calculation for Supplier's Component SKU
        facility_inv = inv_df[inv_df["facility_id"] == reg_info["facility"]]
        sku_inv = facility_inv[facility_inv["sku_id"] == primary_sku]

        if not sku_inv.empty:
            current_stock = float(sku_inv["current_stock_units"].iloc[0])
            daily_burn = float(sku_inv["daily_consumption_rate"].iloc[0])
            buffer_days = current_stock / max(daily_burn, 1.0)
        elif not facility_inv.empty:
            buffer_days = float((facility_inv["current_stock_units"] / facility_inv["daily_consumption_rate"]).median())
        else:
            buffer_days = 25.0

        # Deterministic Time-to-Impact (TTI) formula: Days = Current Stock / Daily Consumption Rate
        predicted_tti_days = int(max(1, min(30, round(buffer_days))))

        # Order volume drop percentage
        if max_port_delay >= 5.0 or buffer_days <= 3.0:
            order_volume_drop_pct = float(round(min(65.0, max_port_delay * 4.8 + 12.0), 1))
        else:
            order_volume_drop_pct = float(round(max(1.0, np.random.uniform(2.0, 9.0)), 1))

        # 3. External NLP / Disruption Severity
        matched_dis = dis_df[dis_df["affected_region"].isin(reg_info["regions"])]
        if not matched_dis.empty:
            max_ext_sev = float(matched_dis["severity_score"].max())
        else:
            sub_matches = dis_df[dis_df["affected_region"].str.contains(country, case=False, na=False)]
            if not sub_matches.empty:
                max_ext_sev = float(sub_matches["severity_score"].max())
            else:
                max_ext_sev = 1.5

        # Assemble feature vector matching FEATURE_COLUMNS
        feature_rows.append({
            "avg_port_congestion_delay": round(avg_port_delay, 2),
            "max_port_congestion_delay": round(max_port_delay, 2),
            "delivery_delay_variance": round(deliv_variance, 2),
            "order_volume_drop_pct": round(order_volume_drop_pct, 2),
            "stock_buffer_days": round(buffer_days, 2),
            "external_disruption_severity": round(max_ext_sev, 2),
        })

        metadata_rows.append({
            "supplier_id": sup_id,
            "supplier_name": sup_name,
            "country": country,
            "backup_supplier_name": backup_sup,
            "primary_sku": primary_sku,
            "predicted_tti_days": predicted_tti_days,
            "buffer_days": buffer_days,
            "max_port_delay": max_port_delay,
            "max_ext_sev": max_ext_sev,
        })

    df_features = pd.DataFrame(feature_rows)[FEATURE_COLUMNS]
    df_meta = pd.DataFrame(metadata_rows)
    return df_features, df_meta


# ---------------------------------------------------------------------------
# Anomaly Scoring & Recommendation Logic
# ---------------------------------------------------------------------------
def generate_recommendation(
    is_flagged: bool,
    tti_days: int,
    max_port_delay: float,
    ext_severity: float,
    backup_supplier: str,
) -> str:
    """
    Dynamically generates actionable mitigation advice tailored to the specific
    root cause of the disruption.
    """
    if not is_flagged:
        return "Standard operational cadence maintained; continuous telemetry active."

    # Cause 1: Imminent Inventory Depletion
    if tti_days <= 3:
        return (
            f"CRITICAL STOCKOUT RISK (TTI: {tti_days}d). Expedite air freight safety buffer "
            f"and activate primary purchase orders with {backup_supplier}."
        )
    # Cause 2: Severe Maritime Port Bottleneck
    elif max_port_delay >= 5.0:
        return (
            f"Severe port congestion ({max_port_delay:.1f}d delay). Reroute container shipments "
            f"to secondary terminal and engage backup supplier {backup_supplier}."
        )
    # Cause 3: High-Severity Natural Disaster or Weather
    elif ext_severity >= 7.0:
        return (
            f"High-severity regional event (Score: {ext_severity:.1f}). Shift 50% production quota "
            f"to alternative facility and notify {backup_supplier}."
        )
    # Cause 4: General Vulnerability
    else:
        return (
            f"Anomaly threshold breached (TTI: {tti_days}d). Escalate to procurement manager "
            f"and verify secondary contract readiness at {backup_supplier}."
        )


def run_anomaly_inference(
    model: IsolationForest,
    df_features: pd.DataFrame,
    df_meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Executes model.predict() and model.decision_function(), assigns anomaly flags,
    and constructs the final disruption alerts DataFrame.
    """
    logger.info("Running Isolation Forest inference on feature vectors...")
    X = df_features[FEATURE_COLUMNS].to_numpy()

    # Predictions: 1 = normal, -1 = anomaly
    predictions = model.predict(X)
    # Decision function: continuous scores; negative values indicate anomalies
    raw_scores = model.decision_function(X)

    alerts = []
    for i in range(len(df_meta)):
        meta = df_meta.iloc[i]
        pred = predictions[i]
        score = float(round(raw_scores[i], 4))
        tti_days = int(meta["predicted_tti_days"])
        max_port = float(meta["max_port_delay"])
        ext_sev = float(meta["max_ext_sev"])
        backup_sup = meta["backup_supplier_name"]

        # Flagged if Isolation Forest signals anomaly (-1 or score <= 0.0)
        # OR if physical buffer indicates critical stockout (< 3 days)
        is_flagged = bool(pred == -1 or score <= 0.005 or tti_days <= 3)

        action = generate_recommendation(
            is_flagged=is_flagged,
            tti_days=tti_days,
            max_port_delay=max_port,
            ext_severity=ext_sev,
            backup_supplier=backup_sup,
        )

        alerts.append({
            "supplier_id": meta["supplier_id"],
            "supplier_name": meta["supplier_name"],
            "country": meta["country"],
            "backup_supplier_name": backup_sup,
            "anomaly_score": score,
            "is_flagged": is_flagged,
            "predicted_tti_days": tti_days,
            "recommended_action": action,
            "created_at": datetime.now(),
        })

    df_alerts = pd.DataFrame(alerts)
    return df_alerts


# ---------------------------------------------------------------------------
# Database Update & Persistence
# ---------------------------------------------------------------------------
def update_disruption_alerts_in_db(
    engine,
    df_alerts: pd.DataFrame,
    clean_previous: bool = True,
):
    """
    Writes or refreshes the `disruption_alerts` table in PostgreSQL.
    """
    logger.info("Writing updated model inferences to 'disruption_alerts' table...")

    insert_cols = [
        "supplier_id",
        "anomaly_score",
        "is_flagged",
        "predicted_tti_days",
        "recommended_action",
        "created_at",
    ]
    df_to_insert = df_alerts[insert_cols].copy()

    with engine.begin() as conn:
        if clean_previous:
            conn.execute(text("DELETE FROM disruption_alerts;"))
            logger.info("Refreshed previous alert records in 'disruption_alerts'.")

        df_to_insert.to_sql(
            name="disruption_alerts",
            con=conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=500,
        )

    logger.info(f"Successfully committed {len(df_to_insert)} records to 'disruption_alerts'.")


def display_inference_results(df_alerts: pd.DataFrame):
    """
    Renders a formatted ASCII summary of all scored suppliers to the console.
    """
    print("\n" + "=" * 110)
    print("  SUPPLYPULSE PHASE 1: REAL-TIME ANOMALY INFERENCE & ALERT ENGINE")
    print("=" * 110)
    print(f"  {'SUPPLIER ID':<12} | {'SUPPLIER NAME':<34} | {'SCORE':<9} | {'STATUS':<11} | {'TTI':<8} | {'BACKUP SUPPLIER':<24}")
    print("  " + "-" * 106)

    for _, row in df_alerts.iterrows():
        status_label = "[ALERT !]" if row["is_flagged"] else "[NOMINAL]"
        tti_str = f"{row['predicted_tti_days']}d"
        print(f"  {row['supplier_id']:<12} | {row['supplier_name'][:34]:<34} | {row['anomaly_score']:>8.4f} | {status_label:<11} | {tti_str:<8} | {row['backup_supplier_name'][:24]:<24}")

    print("=" * 110)
    flagged_total = df_alerts["is_flagged"].sum()
    print(f"  Total Suppliers Scored: {len(df_alerts)} | Flagged Critical Alerts: {flagged_total}")
    print("=" * 110 + "\n")


# ---------------------------------------------------------------------------
# CLI Argument Parser & Orchestration
# ---------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="SupplyPulse: Phase 1 Real-Time Anomaly Detection & Alert Generation"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to serialized isolation_forest_model.pkl",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="PostgreSQL connection string (defaults to .env DATABASE_URL)",
    )
    parser.add_argument(
        "--sqlite",
        action="store_true",
        help="Connect to local SQLite database (supplypulse.db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute inference and display alerts without updating database",
    )
    return parser.parse_args()


def main():
    print("""
    =================================================================
       SUPPLYPULSE: AI SUPPLY CHAIN DISRUPTION PLATFORM
       Phase 1: Real-Time Anomaly Detection & Alert Engine
    =================================================================
    """)

    load_dotenv()
    args = parse_arguments()

    # 1. Load Pre-trained Isolation Forest Model
    if not os.path.exists(args.model_path):
        logger.error(
            f"Model artifact not found at: {args.model_path}\n"
            "Please train the model first by executing: python phase1_detection/train_model.py"
        )
        sys.exit(1)

    logger.info(f"Loading pre-trained Isolation Forest model from: {args.model_path}")
    model = joblib.load(args.model_path)
    logger.info("Model artifact loaded successfully.")

    # 2. Database Connection
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
        with engine.connect() as conn:
            conn.execute(text("SELECT 1;"))
        logger.info(f"Connected to database: {engine.url.drivername}")
    except OperationalError as oe:
        logger.error(f"Cannot connect to database: {oe}")
        sys.exit(1)

    # 3. Extract Features for Active Suppliers
    df_features, df_meta = extract_live_supplier_features(engine)

    # 4. Run Anomaly Inference & TTI Calculation
    df_alerts = run_anomaly_inference(model, df_features, df_meta)

    # 5. Display Formatted Results
    display_inference_results(df_alerts)

    # 6. Database Update
    if args.dry_run:
        logger.info("Dry-run mode active: Skipped database update.")
    else:
        update_disruption_alerts_in_db(engine, df_alerts, clean_previous=True)

    logger.info("SUCCESS: Phase 1 Anomaly Detection & Alert Engine Completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
