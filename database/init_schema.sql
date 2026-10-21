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
    event_date DATE DEFAULT CURRENT_DATE
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
