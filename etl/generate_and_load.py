#!/usr/bin/env python3
"""
SupplyPulse: ETL Pipeline & Synthetic Data Generation (Step 2)
==============================================================
Enterprise-grade, Fujitsu-compliant data generation and ingestion pipeline.

Generates realistic tabular datasets with embedded supply chain anomalies across
the 5 core SupplyPulse relational tables:
  1. suppliers: Tier-1 suppliers across global manufacturing hubs & backup mappings.
  2. inventory_levels: Facilities & SKUs with daily consumption & stockout anomalies.
  3. port_activities: Daily vessel delays & congestion spikes at major maritime hubs.
  4. external_disruptions: Geospatial & weather disruption events with severity scores.
  5. disruption_alerts: AI anomaly detection inferences & Time-to-Impact (TTI) countdowns.

Author: SupplyPulse Team (Pradymna Adhari, Omkar Akhade, Vishal Amrutkar, Pravin Bolshette)
Review: Prof. Vanita Babbane (VIT Pune) | Fujitsu Data & AI Portfolio
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import OperationalError, IntegrityError

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SupplyPulse.ETL")

# ---------------------------------------------------------------------------
# Global Constants & Master Domain Seeds
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
DEFAULT_DB_URL = "postgresql://admin:secret@localhost:5432/supplypulse"

# Manufacturing Hubs and Tier-1 Suppliers with pre-mapped Backup Suppliers
SUPPLIER_MASTER_DATA = [
    {
        "supplier_id": "SUP-001",
        "supplier_name": "Shanghai Advanced Silicon Fab",
        "country": "China",
        "backup_supplier_name": "Hsinchu Wafer Packaging Corp",
    },
    {
        "supplier_id": "SUP-002",
        "supplier_name": "Shenzhen Opto-Precision Tech",
        "country": "China",
        "backup_supplier_name": "Nippon Sensor Mechatronics",
    },
    {
        "supplier_id": "SUP-003",
        "supplier_name": "Hanseatic Precision Hydraulics GmbH",
        "country": "Germany",
        "backup_supplier_name": "Rotterdam Marine Assemblies",
    },
    {
        "supplier_id": "SUP-004",
        "supplier_name": "Bavaria Automotive Sensor Systems",
        "country": "Germany",
        "backup_supplier_name": "Stuttgart Micro Actuators",
    },
    {
        "supplier_id": "SUP-005",
        "supplier_name": "Rotterdam Maritime Logistics Hub",
        "country": "Netherlands",
        "backup_supplier_name": "Antwerp Terminal Solutions",
    },
    {
        "supplier_id": "SUP-006",
        "supplier_name": "Eindhoven Semiconductor Photolithography",
        "country": "Netherlands",
        "backup_supplier_name": "Dresden Advanced Silicon",
    },
    {
        "supplier_id": "SUP-007",
        "supplier_name": "Chennai Fasteners & Industrial Alloys",
        "country": "India",
        "backup_supplier_name": "Pune Industrial Castings Ltd",
    },
    {
        "supplier_id": "SUP-008",
        "supplier_name": "Bengaluru Embedded Core Systems",
        "country": "India",
        "backup_supplier_name": "Hyderabad VLSI Foundry",
    },
    {
        "supplier_id": "SUP-009",
        "supplier_name": "SingaTech Precision Assembly Ltd",
        "country": "Singapore",
        "backup_supplier_name": "Penang Mechatronics SDN BHD",
    },
    {
        "supplier_id": "SUP-010",
        "supplier_name": "Jurong Intermodal Transshipment",
        "country": "Singapore",
        "backup_supplier_name": "Tanjung Pelepas Logistics",
    },
    {
        "supplier_id": "SUP-011",
        "supplier_name": "Yokohama Precision Dynamics",
        "country": "Japan",
        "backup_supplier_name": "Nagoya Industrial Automation",
    },
    {
        "supplier_id": "SUP-012",
        "supplier_name": "Taipei High-Density Interconnects",
        "country": "Taiwan",
        "backup_supplier_name": "SingaTech Precision Assembly Ltd",
    },
]

WAREHOUSE_FACILITIES = [
    "WH-SHG-01",  # Shanghai Hub
    "WH-HAM-01",  # Hamburg Hub
    "WH-ROT-01",  # Rotterdam Hub
    "WH-CHE-01",  # Chennai Hub
    "WH-SIN-01",  # Singapore Hub
]

CORE_SKUS = [
    "SKU-SEMI-101",
    "SKU-SENS-204",
    "SKU-POWR-309",
    "SKU-HYDR-412",
    "SKU-OPT-520",
    "SKU-ACT-631",
    "SKU-RELAY-740",
    "SKU-MCU-855",
    "SKU-DISP-962",
    "SKU-CAP-078",
    "SKU-CONN-189",
    "SKU-MEM-290",
]

MAJOR_PORTS = [
    "Port of Shanghai",
    "Port of Hamburg",
    "Port of Rotterdam",
    "Port of Singapore",
    "Port of Chennai",
    "Port of Ningbo-Zhoushan",
    "Port of Busan",
    "Port of Antwerp",
]

DISRUPTION_TEMPLATES = [
    ("Typhoon", "East China Sea / Yangtze Delta", 8.2),
    ("Severe Flooding", "Rhine-Meuse-Scheldt Delta", 7.4),
    ("Earthquake", "Taiwan Strait / Hsinchu Coast", 8.9),
    ("Port Worker Strike", "Elbe Estuary & Hamburg Area", 7.8),
    ("Monsoon Surge & Flash Flood", "Coromandel Coast / Bay of Bengal", 7.1),
    ("Strait Navigational Blockage", "Strait of Malacca", 8.5),
    ("Winter Blizzard & Freezing Ice", "Baltic Sea Shipping Approaches", 6.8),
    ("Severe Typhoon Warning", "South China Sea / Pearl River", 7.6),
    ("Coastal Storm Surge", "Tokyo Bay & Kanto Region", 6.2),
    ("Customs Clearance Freeze", "Rotterdam Logistics Terminal", 5.9),
    ("Geopolitical Export Review", "East Asian Semiconductor Corridor", 8.7),
    ("Inland Rail Freight Stoppage", "Northern European Freight Corridor", 6.5),
    ("Tropical Depression", "Bay of Bengal Coastal Zone", 5.4),
    ("Container Berth Fire", "Jurong Island Terminal", 7.3),
    ("Dense Fog Navigational Ban", "Shanghai Outer Anchorage", 4.8),
    ("Labor Negotiation Stoppage", "Antwerp Outer Docklands", 6.1),
]

# ---------------------------------------------------------------------------
# Schema DDL Definitions (PostgreSQL & SQLite Compatibility)
# ---------------------------------------------------------------------------
POSTGRES_SCHEMA_SQL = """
-- Core Table 1: Suppliers
CREATE TABLE IF NOT EXISTS suppliers (
    supplier_id TEXT PRIMARY KEY,
    supplier_name TEXT NOT NULL,
    country TEXT NOT NULL,
    backup_supplier_name TEXT
);

-- Core Table 2: Inventory Levels
CREATE TABLE IF NOT EXISTS inventory_levels (
    record_id SERIAL PRIMARY KEY,
    facility_id TEXT NOT NULL,
    sku_id TEXT NOT NULL,
    current_stock_units INT NOT NULL,
    daily_consumption_rate FLOAT NOT NULL,
    stockout_flag BOOLEAN DEFAULT FALSE
);

-- Core Table 3: Port Activities
CREATE TABLE IF NOT EXISTS port_activities (
    activity_id SERIAL PRIMARY KEY,
    port_name TEXT NOT NULL,
    date DATE NOT NULL,
    congestion_delay_days FLOAT NOT NULL
);

-- Core Table 4: External Disruptions
CREATE TABLE IF NOT EXISTS external_disruptions (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    affected_region TEXT NOT NULL,
    severity_score FLOAT CHECK (severity_score >= 0.0 AND severity_score <= 10.0),
    event_date DATE DEFAULT CURRENT_DATE,
    source_text TEXT,
    source TEXT,
    parsed_method TEXT
);

-- Core Table 5: Disruption Alerts
CREATE TABLE IF NOT EXISTS disruption_alerts (
    alert_id SERIAL PRIMARY KEY,
    supplier_id TEXT REFERENCES suppliers(supplier_id) ON DELETE CASCADE,
    anomaly_score FLOAT NOT NULL,
    is_flagged BOOLEAN DEFAULT FALSE,
    predicted_tti_days INT NOT NULL,
    recommended_action TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Core Table 6: Audit Logs
CREATE TABLE IF NOT EXISTS audit_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
    operator_id VARCHAR(100) NOT NULL,
    alert_id VARCHAR(100) NOT NULL,
    ai_recommendation JSONB NOT NULL,
    action_taken VARCHAR(50) NOT NULL CHECK (action_taken IN ('APPROVED', 'OVERRIDDEN', 'REJECTED')),
    operator_notes TEXT,
    confirmed BOOLEAN NOT NULL DEFAULT TRUE,
    mitigation_target TEXT,
    cost_exposure_usd NUMERIC(12, 2)
);

-- Core Table 7: Users
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('analyst', 'manager', 'admin')),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Core Table 8: Model Runs
CREATE TABLE IF NOT EXISTS model_runs (
    run_id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    trained_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    baseline_precision FLOAT,
    baseline_recall FLOAT,
    baseline_f1 FLOAT,
    dataset_description TEXT DEFAULT 'Evaluated on synthetic data',
    is_active BOOLEAN DEFAULT TRUE
);

-- Performance Indexes
CREATE INDEX IF NOT EXISTS idx_inventory_facility_sku ON inventory_levels (facility_id, sku_id);
CREATE INDEX IF NOT EXISTS idx_port_activities_port_date ON port_activities (port_name, date);
CREATE INDEX IF NOT EXISTS idx_disruption_alerts_supplier ON disruption_alerts (supplier_id);
"""

SQLITE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS suppliers (
    supplier_id TEXT PRIMARY KEY,
    supplier_name TEXT NOT NULL,
    country TEXT NOT NULL,
    backup_supplier_name TEXT
);

CREATE TABLE IF NOT EXISTS inventory_levels (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    facility_id TEXT NOT NULL,
    sku_id TEXT NOT NULL,
    current_stock_units INTEGER NOT NULL,
    daily_consumption_rate REAL NOT NULL,
    stockout_flag BOOLEAN DEFAULT 0
);

CREATE TABLE IF NOT EXISTS port_activities (
    activity_id INTEGER PRIMARY KEY AUTOINCREMENT,
    port_name TEXT NOT NULL,
    date TEXT NOT NULL,
    congestion_delay_days REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS external_disruptions (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    affected_region TEXT NOT NULL,
    severity_score REAL CHECK (severity_score >= 0.0 AND severity_score <= 10.0),
    event_date TEXT,
    source_text TEXT,
    source TEXT,
    parsed_method TEXT
);

CREATE TABLE IF NOT EXISTS disruption_alerts (
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    supplier_id TEXT REFERENCES suppliers(supplier_id) ON DELETE CASCADE,
    anomaly_score REAL NOT NULL,
    is_flagged BOOLEAN DEFAULT 0,
    predicted_tti_days INTEGER NOT NULL,
    recommended_action TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_logs (
    log_id TEXT PRIMARY KEY,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    operator_id TEXT NOT NULL,
    alert_id TEXT NOT NULL,
    ai_recommendation TEXT NOT NULL,
    action_taken TEXT NOT NULL,
    operator_notes TEXT,
    confirmed INTEGER NOT NULL DEFAULT 1,
    mitigation_target TEXT,
    cost_exposure_usd REAL
);

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    model_type TEXT NOT NULL,
    trained_at TEXT DEFAULT CURRENT_TIMESTAMP,
    precision REAL,
    recall REAL,
    f1_score REAL,
    baseline_precision REAL,
    baseline_recall REAL,
    baseline_f1 REAL,
    dataset_description TEXT DEFAULT 'Evaluated on synthetic data',
    is_active INTEGER DEFAULT 1
);
"""


# ---------------------------------------------------------------------------
# Synthetic Data Generators (NumPy & Pandas)
# ---------------------------------------------------------------------------
def generate_suppliers_data(count: int = 12) -> pd.DataFrame:
    """
    Generate Tier-1 suppliers spanning major global manufacturing hubs
    with pre-mapped backup suppliers for resilience bypass.
    """
    logger.info(f"Generating suppliers dataset ({count} rows)...")
    np.random.seed(RANDOM_SEED)

    records = list(SUPPLIER_MASTER_DATA)
    if count > len(records):
        # Dynamically append additional vendors if requested
        for i in range(len(records) + 1, count + 1):
            records.append({
                "supplier_id": f"SUP-{i:03d}",
                "supplier_name": f"Global Precision Hub-{i} Corp",
                "country": np.random.choice(["Germany", "Japan", "Singapore", "India", "Netherlands"]),
                "backup_supplier_name": "Apex Alternate Logistics Corp",
            })
    elif count < len(records):
        records = records[:count]

    df = pd.DataFrame(records)
    logger.info(f"Successfully generated {len(df)} suppliers.")
    return df


def generate_inventory_levels_data(target_rows: int = 60) -> pd.DataFrame:
    """
    Generate facility and SKU inventory levels with embedded stockout anomalies.
    Intentionally assigns low current_stock_units for a subset of records to test
    automated stockout early warnings.
    """
    logger.info(f"Generating inventory_levels dataset ({target_rows} rows)...")
    np.random.seed(RANDOM_SEED)

    records = []
    # Cross-product combinations across warehouse facilities and SKUs
    combos = [(fac, sku) for fac in WAREHOUSE_FACILITIES for sku in CORE_SKUS]
    np.random.shuffle(combos)

    actual_count = min(target_rows, len(combos))

    for i in range(actual_count):
        facility_id, sku_id = combos[i]

        # Normal daily consumption rate: 10.0 to 75.0 units/day
        daily_rate = float(np.round(np.random.uniform(12.0, 75.0), 2))

        # Inject intentional stockout anomalies in ~20% of the rows
        is_anomaly = (i % 5 == 0)

        if is_anomaly:
            # Low stock buffer: 1.0 to 2.5 days of consumption
            current_stock = int(np.random.randint(10, int(daily_rate * 2.5) + 1))
            stockout_flag = True
        else:
            # Healthy stock buffer: 10 to 60 days of consumption
            current_stock = int(np.random.randint(int(daily_rate * 10), int(daily_rate * 60) + 500))
            stockout_flag = False

        records.append({
            "facility_id": facility_id,
            "sku_id": sku_id,
            "current_stock_units": current_stock,
            "daily_consumption_rate": daily_rate,
            "stockout_flag": stockout_flag,
        })

    df = pd.DataFrame(records)
    stockouts_count = df["stockout_flag"].sum()
    logger.info(
        f"Generated {len(df)} inventory records (with {stockouts_count} embedded stockout anomalies)."
    )
    return df


def generate_port_activities_data(target_rows: int = 40) -> pd.DataFrame:
    """
    Generate daily vessel delay logs across international container ports.
    Injects occasional severe congestion spikes (> 5.0 to 11.5 days) to simulate
    critical shipping bottlenecks and maritime disruptions.
    """
    logger.info(f"Generating port_activities dataset ({target_rows} rows)...")
    np.random.seed(RANDOM_SEED)

    base_date = datetime.now().date()
    records = []

    for i in range(target_rows):
        port_name = MAJOR_PORTS[i % len(MAJOR_PORTS)]
        # Dates spanning the past 30 days
        log_date = base_date - timedelta(days=int(i // len(MAJOR_PORTS)) * 3)

        # Baseline delay: 0.5 to 2.8 days
        base_delay = np.random.gamma(shape=2.0, scale=0.7)

        # Inject severe congestion spikes in ~18% of rows
        is_spike = (i % 6 == 0)
        if is_spike:
            # Severe congestion delay: 5.5 to 11.2 days
            delay_days = float(np.round(np.random.uniform(5.5, 11.2), 2))
        else:
            delay_days = float(np.round(min(base_delay, 3.8), 2))

        records.append({
            "port_name": port_name,
            "date": log_date,
            "congestion_delay_days": delay_days,
        })

    df = pd.DataFrame(records)
    spikes_count = (df["congestion_delay_days"] >= 5.0).sum()
    logger.info(
        f"Generated {len(df)} port activity logs (with {spikes_count} severe bottleneck spikes > 5.0 days)."
    )
    return df


def generate_external_disruptions_data(target_rows: int = 16) -> pd.DataFrame:
    """
    Generate natural disaster, weather alerts, and labor strikes with severity scores
    ranging strictly between 0.0 and 10.0 mapped to key supply chain regions.
    """
    logger.info(f"Generating external_disruptions dataset ({target_rows} rows)...")
    np.random.seed(RANDOM_SEED)

    records = []
    for i in range(target_rows):
        event_id = f"EVT-2026-{i+1:03d}"
        template_idx = i % len(DISRUPTION_TEMPLATES)
        event_type, affected_region, base_severity = DISRUPTION_TEMPLATES[template_idx]

        # Add slight natural variation to severity while strictly enforcing [0.0, 10.0]
        noise = np.random.uniform(-0.5, 0.6)
        severity = float(np.round(np.clip(base_severity + noise, 0.0, 10.0), 1))

        records.append({
            "event_id": event_id,
            "event_type": event_type,
            "affected_region": affected_region,
            "severity_score": severity,
        })

    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} external disruption events (Severity: 0.0 - 10.0).")
    return df


def generate_disruption_alerts_data(
    suppliers_df: pd.DataFrame, target_rows: int = 15
) -> pd.DataFrame:
    """
    Generate synthetic AI disruption alerts linking Isolation Forest anomaly scores,
    flagged indicators, predicted Time-to-Impact (TTI) days, and mitigation actions.
    Ensures foreign key integrity by referencing existing supplier_ids.
    """
    logger.info(f"Generating disruption_alerts dataset ({target_rows} rows)...")
    np.random.seed(RANDOM_SEED)

    supplier_ids = suppliers_df["supplier_id"].tolist()
    records = []

    recommended_actions_catalog = [
        "Reroute critical purchase orders to pre-mapped backup supplier.",
        "Expedite priority safety stock replenishment buffer via air freight.",
        "Reallocate production quota to alternative regional assembly lines.",
        "Activate contingency stock buffer and escalate to procurement director.",
        "Pre-book alternate container slot via secondary maritime carrier.",
        "Standard operational cadence maintained; continuous monitoring active.",
    ]

    for i in range(target_rows):
        supplier_id = supplier_ids[i % len(supplier_ids)]

        # Anomaly score between -0.85 and +0.65 (Isolation Forest format: negative = anomalous)
        is_alert = (i % 3 == 0) or (i % 5 == 0)
        if is_alert:
            anomaly_score = float(np.round(np.random.uniform(-0.85, -0.15), 4))
            is_flagged = True
            predicted_tti_days = int(np.random.randint(2, 9))
            action = recommended_actions_catalog[i % 5]
        else:
            anomaly_score = float(np.round(np.random.uniform(0.05, 0.65), 4))
            is_flagged = False
            predicted_tti_days = int(np.random.randint(14, 30))
            action = recommended_actions_catalog[5]

        created_at = datetime.now() - timedelta(hours=int(i * 4))

        records.append({
            "supplier_id": supplier_id,
            "anomaly_score": anomaly_score,
            "is_flagged": is_flagged,
            "predicted_tti_days": predicted_tti_days,
            "recommended_action": action,
            "created_at": created_at,
        })

    df = pd.DataFrame(records)
    flagged_count = df["is_flagged"].sum()
    logger.info(
        f"Generated {len(df)} AI disruption alerts ({flagged_count} critical alerts flagged for intervention)."
    )
    return df


# ---------------------------------------------------------------------------
# Database Management & Schema Migration
# ---------------------------------------------------------------------------
def initialize_database_schema(engine, is_sqlite: bool = False):
    """
    Execute DDL statements to ensure all 5 core tables exist with proper constraints,
    foreign keys, check constraints, and performance indexes.
    """
    logger.info("Initializing relational database schema...")
    schema_sql = SQLITE_SCHEMA_SQL if is_sqlite else POSTGRES_SCHEMA_SQL

    with engine.begin() as conn:
        # Split statements by semicolon and execute non-empty statements
        for statement in schema_sql.strip().split(";"):
            clean_stmt = statement.strip()
            if clean_stmt:
                conn.execute(text(clean_stmt))

    logger.info("Database schema initialized successfully.")


def clean_existing_tables(engine):
    """
    Safely truncate or delete all rows from the 5 core tables in reverse
    dependency order to respect foreign key constraints.
    """
    logger.info("Cleaning existing data from tables (clean mode active)...")
    tables_in_order = [
        "disruption_alerts",
        "external_disruptions",
        "port_activities",
        "inventory_levels",
        "suppliers",
    ]

    with engine.begin() as conn:
        for tbl in tables_in_order:
            try:
                conn.execute(text(f"DELETE FROM {tbl};"))
                logger.info(f"  Truncated table '{tbl}'.")
            except Exception as e:
                logger.warning(f"  Could not truncate '{tbl}': {e}")


# ---------------------------------------------------------------------------
# Ingestion Pipeline & Verification
# ---------------------------------------------------------------------------
def load_datasets_to_database(
    engine, datasets: Dict[str, pd.DataFrame]
) -> Dict[str, int]:
    """
    Batch-load synthetic DataFrames into relational tables in dependency order.
    Returns a dictionary of inserted row counts verified against the database.
    """
    # Required dependency insertion order:
    # 1. suppliers (Parent table for foreign keys)
    # 2. inventory_levels
    # 3. port_activities
    # 4. external_disruptions
    # 5. disruption_alerts (References suppliers.supplier_id)
    load_sequence = [
        ("suppliers", datasets["suppliers"]),
        ("inventory_levels", datasets["inventory_levels"]),
        ("port_activities", datasets["port_activities"]),
        ("external_disruptions", datasets["external_disruptions"]),
        ("disruption_alerts", datasets["disruption_alerts"]),
    ]

    insertion_stats = {}

    for table_name, df in load_sequence:
        logger.info(f"Loading {len(df)} records into table '{table_name}'...")
        try:
            # Use pandas to_sql with append mode and chunked multi-row inserts
            df.to_sql(
                name=table_name,
                con=engine,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=500,
            )
            logger.info(f"Successfully inserted rows into '{table_name}'.")
        except IntegrityError as ie:
            logger.error(
                f"IntegrityError loading '{table_name}'! Constraint violation: {ie}"
            )
            raise
        except Exception as ex:
            logger.error(f"Unexpected error loading table '{table_name}': {ex}")
            raise

    # Verification: Query the database directly to confirm row counts
    logger.info("\n" + "=" * 60)
    logger.info("DATABASE INGESTION VERIFICATION & ROW COUNT AUDIT")
    logger.info("=" * 60)

    with engine.connect() as conn:
        for table_name, _ in load_sequence:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            count = result.scalar()
            insertion_stats[table_name] = count
            logger.info(f" Table: {table_name:<22} | Row Count in DB: {count}")

    logger.info("=" * 60 + "\n")
    return insertion_stats


def export_datasets_to_csv(datasets: Dict[str, pd.DataFrame], output_dir: str):
    """
    Export synthetic datasets to CSV files in the data directory for inspection
    or offline analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Exporting synthetic datasets to CSV in '{output_dir}'...")

    for name, df in datasets.items():
        csv_path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"  Wrote: {csv_path} ({len(df)} rows)")


# ---------------------------------------------------------------------------
# CLI Argument Parser & Execution Flow
# ---------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="SupplyPulse: Step 2 Synthetic Data Generation & Database Ingestion"
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="PostgreSQL connection string (defaults to DATABASE_URL env var or localhost:5432)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Truncate/delete existing table data before inserting new synthetic records",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate synthetic data and export CSVs without connecting to the database",
    )
    parser.add_argument(
        "--sqlite",
        action="store_true",
        help="Ingest into a local SQLite database (supplypulse.db) for testing without Docker",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        default=True,
        help="Export generated datasets to data/synthetic/*.csv (enabled by default)",
    )
    parser.add_argument(
        "--suppliers-count",
        type=int,
        default=12,
        help="Number of Tier-1 suppliers to generate (default: 12)",
    )
    parser.add_argument(
        "--inventory-count",
        type=int,
        default=60,
        help="Number of inventory level records to generate (default: 60)",
    )
    parser.add_argument(
        "--ports-count",
        type=int,
        default=40,
        help="Number of port activity logs to generate (default: 40)",
    )
    parser.add_argument(
        "--disruptions-count",
        type=int,
        default=16,
        help="Number of external disruption events to generate (default: 16)",
    )
    parser.add_argument(
        "--alerts-count",
        type=int,
        default=15,
        help="Number of AI disruption alerts to generate (default: 15)",
    )
    return parser.parse_args()


def main():
    print("""
    =================================================================
       SUPPLYPULSE: AI SUPPLY CHAIN DISRUPTION PLATFORM
       Step 2: Synthetic Data Generation & PostgreSQL ETL Pipeline
    =================================================================
    """)

    # 1. Load environment variables from .env
    load_dotenv()
    args = parse_arguments()

    # 2. Generate Synthetic Tabular Datasets (Pandas & NumPy)
    logger.info("--- Phase A: Synthetic Data Generation ---")
    suppliers_df = generate_suppliers_data(count=args.suppliers_count)
    inventory_df = generate_inventory_levels_data(target_rows=args.inventory_count)
    ports_df = generate_port_activities_data(target_rows=args.ports_count)
    disruptions_df = generate_external_disruptions_data(target_rows=args.disruptions_count)
    alerts_df = generate_disruption_alerts_data(suppliers_df, target_rows=args.alerts_count)

    datasets = {
        "suppliers": suppliers_df,
        "inventory_levels": inventory_df,
        "port_activities": ports_df,
        "external_disruptions": disruptions_df,
        "disruption_alerts": alerts_df,
    }

    # 3. Optional CSV Export
    if args.export_csv or args.dry_run:
        csv_dir = os.path.join(os.getcwd(), "data", "synthetic")
        export_datasets_to_csv(datasets, csv_dir)

    if args.dry_run:
        logger.info("Dry run completed successfully. Skipped database ingestion.")
        return 0

    # 4. Database Connection Setup
    logger.info("--- Phase B: Database Ingestion Pipeline ---")

    if args.sqlite:
        db_url = "sqlite:///supplypulse.db"
        is_sqlite = True
        logger.info(f"Target Database: Local SQLite ({db_url})")
    else:
        is_sqlite = False
        db_url = (
            args.db_url
            or os.getenv("DATABASE_URL")
            or f"postgresql://{os.getenv('POSTGRES_USER', 'admin')}:{os.getenv('POSTGRES_PASSWORD', 'secret')}@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'supplypulse')}"
        )
        logger.info(f"Target Database: PostgreSQL ({db_url})")

    try:
        engine = create_engine(db_url, echo=False)
        # Test connection
        with engine.connect() as test_conn:
            test_conn.execute(text("SELECT 1"))
        logger.info("Successfully connected to the database engine.")
    except OperationalError as oe:
        logger.error(
            "\n" + "!" * 70 + "\n"
            "DATABASE CONNECTION FAILED!\n"
            f"Could not connect to PostgreSQL at: {db_url}\n\n"
            "Troubleshooting Steps:\n"
            "  1. Ensure the PostgreSQL container is started:\n"
            "       docker compose up -d\n"
            "  2. Check container status:\n"
            "       docker compose ps\n"
            "  3. Verify port 5432 is not blocked by another service.\n"
            "  4. Or test offline with SQLite:\n"
            "       python etl/generate_and_load.py --sqlite\n"
            "  5. Or dry-run data generation only:\n"
            "       python etl/generate_and_load.py --dry-run\n"
            + "!" * 70
        )
        sys.exit(1)

    # 5. Schema Provisioning & Clean
    try:
        initialize_database_schema(engine, is_sqlite=is_sqlite)

        if args.clean:
            clean_existing_tables(engine)

        # 6. Load Datasets & Verify
        stats = load_datasets_to_database(engine, datasets)

        logger.info("SUCCESS: SupplyPulse ETL Pipeline Step 2 Completed!")
        logger.info(f"Total Tables Processed: {len(stats)}")
        return 0

    except Exception as e:
        logger.error(f"FATAL: Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
