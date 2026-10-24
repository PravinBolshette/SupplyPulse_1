# SupplyPulse Data Dictionary

## Overview
This document specifies the complete relational database schema for the SupplyPulse Supply Chain Disruption & Risk Monitoring platform. All tables are defined in [`database/init_schema.sql`](file:///c:/Projects/SupplyPulse/database/init_schema.sql) and provisioned in PostgreSQL 15.

---

### Table: `suppliers`
Master registry of Tier-1 suppliers and their operational metrics.

| Column Name | SQL Type | Nullable | Default | Constraints | Description | Data Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `supplier_id` | `VARCHAR(50)` | No | None | PRIMARY KEY | Unique supplier identifier (e.g., `SUP-001`) | Synthetic / ERP Master |
| `supplier_name` | `VARCHAR(255)` | No | None | None | Legal business entity name | Synthetic / ERP Master |
| `tier_level` | `INT` | No | `1` | `CHECK (tier_level IN (1, 2, 3))` | Supply chain tier hierarchy level | Synthetic / ERP Master |
| `country` | `VARCHAR(100)` | No | None | None | Country where supplier headquarters/facility is located | Synthetic / ERP Master |
| `latitude` | `DECIMAL(9, 6)` | Yes | None | None | Geographical latitude of manufacturing site | Synthetic / Geocoding |
| `longitude` | `DECIMAL(9, 6)` | Yes | None | None | Geographical longitude of manufacturing site | Synthetic / Geocoding |
| `lead_time_days` | `INT` | No | None | `CHECK (lead_time_days > 0)` | Contracted standard lead time in calendar days | Synthetic / ERP Master |
| `historical_reliability_score` | `DECIMAL(5, 4)` | No | None | `CHECK (score BETWEEN 0.0 AND 1.0)` | Past delivery performance reliability index | Synthetic / ERP Historical |
| `primary_material` | `VARCHAR(100)` | No | None | None | Critical component or raw material supplied | Synthetic / ERP Master |
| `backup_supplier_name` | `VARCHAR(255)` | Yes | None | None | Designated secondary vendor for contingency routing | Synthetic / Procurement |
| `contract_value_usd` | `DECIMAL(12, 2)` | Yes | `0.00` | None | Total annual contract value in USD | Synthetic / Procurement |
| `created_at` | `TIMESTAMPTZ` | Yes | `CURRENT_TIMESTAMP` | None | Record creation timestamp | System Generated |

---

### Table: `inventory_levels`
Current stock, consumption velocity, and safety thresholds across storage facilities.

| Column Name | SQL Type | Nullable | Default | Constraints | Description | Data Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `inventory_id` | `SERIAL` | No | Autoincrement | PRIMARY KEY | Unique inventory record identifier | System Generated |
| `facility_id` | `VARCHAR(50)` | No | None | None | Warehouse or factory identifier (e.g., `WH-EAST-1`) | Synthetic / WMS |
| `sku_id` | `VARCHAR(50)` | No | None | None | Stock Keeping Unit identifier | Synthetic / ERP |
| `current_stock_units` | `INT` | No | None | `CHECK (current_stock_units >= 0)` | Physical on-hand inventory count | Synthetic / WMS |
| `safety_stock_units` | `INT` | No | None | `CHECK (safety_stock_units >= 0)` | Minimum buffer stock threshold | Synthetic / MRP |
| `daily_consumption_rate` | `DECIMAL(10, 2)` | No | None | `CHECK (rate > 0)` | Average units consumed per calendar day | Synthetic / MRP |
| `reorder_point` | `INT` | No | None | `CHECK (reorder_point >= 0)` | Inventory level triggering replenishment order | Synthetic / MRP |
| `lead_time_days` | `INT` | No | None | `CHECK (lead_time_days > 0)` | Replenishment procurement cycle time | Synthetic / MRP |
| `stockout_flag` | `BOOLEAN` | Yes | `FALSE` | None | Flagged when current stock is below safety threshold | System Computed |
| `last_updated` | `TIMESTAMPTZ` | Yes | `CURRENT_TIMESTAMP` | None | Timestamp of last stock balance telemetry update | System Generated |

---

### Table: `port_activities`
Maritime port transit telemetry, congestion indexes, and transit delay indicators.

| Column Name | SQL Type | Nullable | Default | Constraints | Description | Data Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `activity_id` | `SERIAL` | No | Autoincrement | PRIMARY KEY | Unique maritime activity record identifier | System Generated |
| `port_code` | `VARCHAR(20)` | No | None | None | UN/LOCODE port identifier (e.g., `CNSHA`, `SGSIN`) | Synthetic / AIS Marine |
| `port_name` | `VARCHAR(100)` | No | None | None | Port common name (e.g., Shanghai, Singapore) | Synthetic / AIS Marine |
| `country` | `VARCHAR(100)` | No | None | None | Nation governing port terminal operations | Synthetic / AIS Marine |
| `latitude` | `DECIMAL(9, 6)` | Yes | None | None | Maritime terminal latitude | Synthetic / AIS Marine |
| `longitude` | `DECIMAL(9, 6)` | Yes | None | None | Maritime terminal longitude | Synthetic / AIS Marine |
| `vessel_count` | `INT` | Yes | `0` | `CHECK (vessel_count >= 0)` | Number of container vessels in anchorage/berth | Synthetic / AIS Marine |
| `average_delay_days` | `DECIMAL(6, 2)` | Yes | `0.00` | None | Average vessel waiting and turnaround delay in days | Synthetic / AIS Marine |
| `congestion_level` | `VARCHAR(20)` | Yes | `'LOW'` | `CHECK (level IN ('LOW','MEDIUM','HIGH','SEVERE'))` | Categorical terminal congestion status | Synthetic / AIS Marine |
| `disruption_flag` | `BOOLEAN` | Yes | `FALSE` | None | Indicates abnormal maritime congestion (> 3.5 days delay) | System Computed |
| `timestamp` | `TIMESTAMPTZ` | Yes | `CURRENT_TIMESTAMP` | None | Timestamp of maritime telemetry observation | System Generated |
| `data_source` | `VARCHAR(50)` | Yes | `'synthetic'` | None | Lineage origin (`synthetic` or `real` PortWatch) | ETL Lineage |

---

### Table: `external_disruptions`
Geopolitical, natural disaster, and supply chain news events ingested via NLP.

| Column Name | SQL Type | Nullable | Default | Constraints | Description | Data Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `disruption_id` | `SERIAL` | No | Autoincrement | PRIMARY KEY | Unique disruption bulletin identifier | System Generated |
| `event_type` | `VARCHAR(100)` | No | None | None | Classification (e.g., Typhoon, Port Strike, Earthquake) | NLP Rule/Gazetteer |
| `region` | `VARCHAR(100)` | No | None | None | Impacted maritime corridor or geographical region | Gazetteer / spaCy NER |
| `severity` | `VARCHAR(20)` | Yes | `'MEDIUM'` | `CHECK (severity IN ('LOW','MEDIUM','HIGH','CRITICAL'))` | Assessed disruption severity level | NLP Keyword Rule |
| `description` | `TEXT` | No | None | None | Cleaned description of the disruption event | News Feed / Synthetic |
| `entities_affected` | `TEXT` | Yes | None | None | Comma-delimited list of detected entities (ports, orgs) | spaCy NER / Gazetteer |
| `timestamp` | `TIMESTAMPTZ` | Yes | `CURRENT_TIMESTAMP` | None | Event timestamp | RSS Bulletin / Feed |
| `source_text` | `TEXT` | Yes | None | None | Raw headline or verbatim feed content | Raw Ingestion Source |
| `source` | `VARCHAR(100)` | Yes | `'news_feed'` | None | Publisher or origin feed identifier (e.g., Reuters, RSS) | Ingestion Feed |
| `parsed_method` | `VARCHAR(50)` | Yes | `'rule_gazetteer'` | None | Parsing pipeline method used (`keyword_gazetteer`, `spacy_ner`, `default_corridor`) | ETL Lineage |

---

### Table: `disruption_alerts`
AI anomaly detection inferences, depletion risk scores, and actionable mitigation recommendations.

| Column Name | SQL Type | Nullable | Default | Constraints | Description | Data Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `alert_id` | `SERIAL` | No | Autoincrement | PRIMARY KEY | Unique disruption alert identifier | System Generated |
| `supplier_id` | `VARCHAR(50)` | No | None | REFERENCES `suppliers(supplier_id)` | Target supplier referenced by early-warning alert | AI Model Inference |
| `anomaly_score` | `DECIMAL(7, 4)` | No | None | None | Raw or normalized Isolation Forest anomaly decision score | Phase 1 ML Engine |
| `is_flagged` | `BOOLEAN` | Yes | `FALSE` | None | True when anomaly score exceeds decision threshold | Phase 1 ML Engine |
| `predicted_tti_days` | `DECIMAL(6, 2)` | Yes | None | None | Dynamic Time-To-Impact countdown in calendar days | Phase 2 TTI Engine |
| `recommended_action` | `VARCHAR(255)` | Yes | None | None | Prescriptive mitigation strategy proposed by system | AI Advisory Engine |
| `created_at` | `TIMESTAMPTZ` | Yes | `CURRENT_TIMESTAMP` | None | Timestamp when inference was generated | System Generated |

---

### Table: `audit_logs`
Immutable record of all human operator actions, approvals, overrides, and rationale.

| Column Name | SQL Type | Nullable | Default | Constraints | Description | Data Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `log_id` | `UUID` | No | `gen_random_uuid()` | PRIMARY KEY | Cryptographically secure unique log entry UUID | System Generated |
| `timestamp` | `TIMESTAMPTZ` | Yes | `CURRENT_TIMESTAMP` | None | Immutable UTC timestamp of operator decision | System Generated |
| `operator_id` | `VARCHAR(100)` | No | None | None | Authenticated user identity (extracted strictly from JWT) | Auth Session |
| `alert_id` | `INT` | Yes | None | REFERENCES `disruption_alerts(alert_id)` | Associated disruption alert being addressed | Human Operator |
| `ai_recommendation`| `JSONB` | Yes | None | None | Snapshot of AI recommendation shown to operator | UI / API State |
| `human_action` | `VARCHAR(50)` | No | None | None | Action taken: `APPROVED`, `REJECTED`, `MODIFIED` | Human Operator |
| `rationale` | `TEXT` | No | None | None | Mandatory operator justification (min 10 chars) | Human Operator |
| `risk_override` | `BOOLEAN` | Yes | `FALSE` | None | Flagged True if human overrides high-risk recommendation | Human Operator |
| `financial_exposure_usd` | `DECIMAL(12, 2)` | Yes | `0.00` | None | Operator-entered estimated financial impact in USD | Human Operator |
| `compliance_flag` | `BOOLEAN` | Yes | `TRUE` | None | Internal governance compliance indicator | System Enforced |

---

### Table: `users`
Authenticated operator accounts and Role-Based Access Control (RBAC) tiers.

| Column Name | SQL Type | Nullable | Default | Constraints | Description | Data Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `user_id` | `UUID` | No | `gen_random_uuid()` | PRIMARY KEY | Unique user account identifier | System Generated |
| `username` | `VARCHAR(50)` | No | None | UNIQUE | Unique login handle | User Registration |
| `password_hash` | `VARCHAR(255)` | No | None | None | Direct bcrypt-hashed salted password | Bcrypt Hashing |
| `role` | `VARCHAR(20)` | No | None | `CHECK (role IN ('analyst', 'manager', 'admin'))` | RBAC role governing system capabilities | Auth System |
| `created_at` | `TIMESTAMPTZ` | Yes | `CURRENT_TIMESTAMP` | None | Account creation timestamp | System Generated |

---

### Table: `model_runs`
MLOps lineage ledger documenting model versions, training timestamps, and performance vs baselines.

| Column Name | SQL Type | Nullable | Default | Constraints | Description | Data Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `run_id` | `SERIAL` | No | Autoincrement | PRIMARY KEY | Unique training run identifier | System Generated |
| `model_name` | `VARCHAR(100)` | No | None | None | Name of the evaluated algorithm (e.g. `Isolation Forest`) | MLOps Lineage |
| `model_version` | `VARCHAR(50)` | No | None | None | Semantic version string (e.g., `v1.2.0`) | Model Pipeline |
| `trained_at` | `TIMESTAMPTZ` | Yes | `CURRENT_TIMESTAMP` | None | Timestamp when model was trained | Model Pipeline |
| `precision_score` | `DECIMAL(5, 4)` | No | None | None | Primary model test precision on held-out evaluation set | Model Evaluation |
| `recall_score` | `DECIMAL(5, 4)` | No | None | None | Primary model test recall on held-out evaluation set | Model Evaluation |
| `f1_score` | `DECIMAL(5, 4)` | No | None | None | Primary model test F1 score on held-out evaluation set | Model Evaluation |
| `baseline_precision` | `DECIMAL(5, 4)` | No | None | None | One-Class SVM baseline test precision | Model Evaluation |
| `baseline_recall` | `DECIMAL(5, 4)` | No | None | None | One-Class SVM baseline test recall | Model Evaluation |
| `baseline_f1` | `DECIMAL(5, 4)` | No | None | None | One-Class SVM baseline test F1 score | Model Evaluation |
| `dataset_split` | `VARCHAR(50)` | Yes | `'temporal_80_20'` | None | Train/test split methodology | Model Pipeline |
| `metrics_metadata` | `JSONB` | Yes | None | None | Detailed breakdown including LOF baseline and confusion matrices | Model Evaluation |
