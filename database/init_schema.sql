-- SupplyPulse PostgreSQL 15 Database Schema Initialization
-- Defines the 5 core tables for early disruption warning and resilient supply chain analytics.

-- 1. Suppliers Table (Tier-1 Suppliers & Backup Mappings)
CREATE TABLE IF NOT EXISTS suppliers (
    supplier_id TEXT PRIMARY KEY,
    supplier_name TEXT NOT NULL,
    country TEXT NOT NULL,
    backup_supplier_name TEXT
);

-- 2. Inventory Levels Table (Facility & SKU Stock Depletion Tracking)
CREATE TABLE IF NOT EXISTS inventory_levels (
    record_id SERIAL PRIMARY KEY,
    facility_id TEXT NOT NULL,
    sku_id TEXT NOT NULL,
    current_stock_units INT NOT NULL,
    daily_consumption_rate FLOAT NOT NULL,
    stockout_flag BOOLEAN DEFAULT FALSE
);

-- 3. Port Activity Table (Vessel Congestion & Delay Telemetry)
CREATE TABLE IF NOT EXISTS port_activities (
    activity_id SERIAL PRIMARY KEY,
    port_name TEXT NOT NULL,
    date DATE NOT NULL,
    congestion_delay_days FLOAT NOT NULL
);

-- 4. External Disruption Table (NLP Parsed Weather & Geospatial Alerts)
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

-- 5. AI Disruption Alerts Table (Output of Anomaly Detection & Time-to-Impact Engine)
CREATE TABLE IF NOT EXISTS disruption_alerts (
    alert_id SERIAL PRIMARY KEY,
    supplier_id TEXT REFERENCES suppliers(supplier_id) ON DELETE CASCADE,
    anomaly_score FLOAT NOT NULL,
    is_flagged BOOLEAN DEFAULT FALSE,
    predicted_tti_days INT NOT NULL,
    recommended_action TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance Indexes
CREATE INDEX IF NOT EXISTS idx_inventory_facility_sku ON inventory_levels (facility_id, sku_id);
CREATE INDEX IF NOT EXISTS idx_port_activities_port_date ON port_activities (port_name, date);
CREATE INDEX IF NOT EXISTS idx_disruption_alerts_supplier ON disruption_alerts (supplier_id);
CREATE INDEX IF NOT EXISTS idx_external_disruptions_date ON external_disruptions (event_date);

-- 6. Immutable Audit Logs Table (Responsible AI & Human-in-the-Loop Framework)
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

CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_logs_operator ON audit_logs (operator_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_alert ON audit_logs (alert_id);

-- Database-Level Immutability Trigger
CREATE OR REPLACE FUNCTION prevent_audit_log_modification()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'SECURITY VIOLATION: audit_logs records are strictly immutable. UPDATE and DELETE operations are forbidden by Responsible AI compliance policy.';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_audit_logs_immutable ON audit_logs;
CREATE TRIGGER trg_audit_logs_immutable
BEFORE UPDATE OR DELETE ON audit_logs
FOR EACH ROW
EXECUTE FUNCTION prevent_audit_log_modification();

-- 7. Users Table (Authentication & Role-Based Access Control)
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('analyst', 'manager', 'admin')),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);

-- 8. Model Runs Table (MLOps Model Lineage & Validation Metrics)
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

