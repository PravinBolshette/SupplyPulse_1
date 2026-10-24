# SupplyPulse Architecture Documentation

## 1. System Overview
SupplyPulse is an enterprise early-warning AI supply chain disruption monitoring and resilience platform. It ingests multi-modal data streams (ERP supplier records, warehouse inventory balances, maritime port activity AIS logs, and external geopolitical/weather RSS headlines), runs machine learning anomaly detection and physical inventory depletion forecasting, and serves interactive decision support dashboards governed by strict Human-in-the-Loop (HITL) compliance guardrails.

---

## 2. End-to-End System Architecture Diagram

```
+---------------------------------------------------------------------------------------+
|                                     DATA SOURCES                                      |
|   +-------------------+  +------------------+  +-----------------+  +---------------+ |
|   | ERP Master Data   |  | Warehouse (WMS)  |  | AIS Port Delays |  | RSS / News    | |
|   | (Suppliers)       |  | (Inventory)      |  | (Port Transit)  |  | (Headlines)   | |
|   +---------+---------+  +--------+---------+  +--------+--------+  +-------+-------+ |
+-------------|---------------------|---------------------|-------------------|---------+
              |                     |                     |                   |
              v                     v                     v                   v
+---------------------------------------------------------------------------------------+
|                                  ETL & INGESTION                                      |
|   +---------------------------------------+  +------------------------------------+   |
|   | etl/generate_and_load.py              |  | etl/nlp_event_ingestion.py         |   |
|   | - Seeds relational synthetic baselines|  | - Keyword matching & Gazetteer     |   |
|   | - Calculates depletion & lead metrics |  | - spaCy NER fallback entity parse  |   |
|   +-------------------+-------------------+  +-----------------+------------------+   |
+-----------------------|----------------------------------------|----------------------+
                        |                                        |
                        +-------------------+--------------------+
                                            v
+---------------------------------------------------------------------------------------+
|                             DATABASE STORAGE (PostgreSQL 15)                          |
|                       Provisioned via database/init_schema.sql                        |
|                                                                                       |
|   * suppliers            * inventory_levels    * port_activities                      |
|   * external_disruptions * disruption_alerts   * audit_logs (Immutable)              |
|   * users (RBAC)         * model_runs (MLOps)                                         |
+-------------------------------------------+-------------------------------------------+
                                            |
                        +-------------------+--------------------+
                        |                                        |
                        v                                        v
+-----------------------------------+  +------------------------------------------------+
|       AI / ML PIPELINES           |  |             FASTAPI REST BACKEND               |
|                                   |  |                    (app/)                      |
| * Phase 1 Anomaly Detection:      |  |                                                |
|   - train_model.py                |  | Middlewares:                                   |
|   - evaluate_model.py             |  | - Request audit logging (latency, IP, status)  |
|     (Evaluates IF vs OC-SVM & LOF)|  | - Correlation ID (X-Request-ID propagation)   |
|   - detect.py                     |  | - Global Exception Sanitization & Envelopes   |
|   - Artifact:                     |  |                                                |
|     isolation_forest_model.pkl    |  | Routers:                                       |
|                                   |  | - /api/v1/auth (Register, Login, JWT tokens)   |
| * Phase 2 Time-to-Impact:         |  | - /api/v1/anomalies (Dynamic model confidence) |
|   - tti_calculator.py             |  | - /api/v1/alerts (Ranked active alerts)        |
|   - Multi-tier dynamic depletion: |  | - /api/v1/suppliers (30-day telemetry history) |
|     TTI = Stock / Dynamic Burn    |  | - /api/v1/events/parsed (NLP disruption feed)  |
|     shortfall, buffer, delay      |  | - /api/v1/model (Model lineage & test metrics) |
|                                   |  | - /api/v1/mitigation/action (RBAC & 503 Guard) |
+-----------------+-----------------+  +-----------------------+------------------------+
                  |                                            |
                  +--------------------+-----------------------+
                                       v
+---------------------------------------------------------------------------------------+
|                           STREAMLIT OPERATOR INTERFACE                                |
|                              (streamlit_app/app.py)                                   |
|                                                                                       |
|  * Strict Demo Mode Gate: DEMO_MODE=true shows red banner; otherwise halts with       |
|    clean command-line startup instructions if FastAPI is offline                      |
|  * RBAC Login / Registration Screen (JWT session token storage)                       |
|  * Operational Tabs:                                                                  |
|    1. Overview: Platform metrics, risk summaries, high-level KPIs                     |
|    2. Alerts: Real-time alerts with TTI ranges, delay days, operator cost entry,      |
|       and Human-in-the-Loop mitigation workflow                                       |
|    3. Suppliers: Real 30-day historical time-series from database endpoint            |
|       with plain-language labels ("Normal", "Flagged as Unusual")                     |
|    4. News events: Disruption bulletins parsed by NLP with source attribution        |
|    5. Audit log: Historical trace of signed operator decisions and risk overrides     |
|    6. Model information: Production version, trained date, test P/R/F1 vs 2 baselines |
|       with explicit line "Evaluated on synthetic data"                                |
+---------------------------------------------------------------------------------------+
```

---

## 3. Data Flow Specification

### Step 1: Ingestion & Storage
1. Baseline tabular telemetry is generated and loaded into PostgreSQL using `etl/generate_and_load.py`.
2. News headlines are processed by `etl/nlp_event_ingestion.py`:
   - Headlines first match against a deterministic keyword rule set and an explicit geographical gazetteer.
   - If geographical entities are not resolved in the gazetteer, spaCy's `en_core_web_sm` Named Entity Recognizer (NER) is invoked as a fallback.
   - Parsed bulletins are stored in `external_disruptions` with `source_text`, `source`, and `parsed_method`.

### Step 2: Machine Learning Anomaly Detection & Prediction
1. `phase1_detection/train_model.py` loads supplier and port activity telemetry, engineers normalized delay, reliability, and variance features, and fits an `IsolationForest` model.
2. `phase1_detection/evaluate_model.py` runs a temporal split evaluation comparing the Isolation Forest model against two benchmark baselines:
   - Baseline 1: `OneClassSVM`
   - Baseline 2: `LocalOutlierFactor`
   All metrics (Precision, Recall, F1) are recorded to the `model_runs` table.
3. `phase1_detection/detect.py` performs live inference on new observation windows, computes anomaly scores, and triggers `phase2_prediction/tti_calculator.py`.
4. `phase2_prediction/tti_calculator.py` computes the exact Time-To-Impact (TTI):
   $$\text{Predicted TTI} = \frac{\text{Current Stock Units}}{\text{Dynamic Daily Consumption Rate}}$$
   taking into account disruption severities, expected transit delays, and inventory buffer shortfalls.
5. Inferences and alerts are stored in `disruption_alerts`.

### Step 3: Backend REST Services & Security
1. The FastAPI application initializes with global middleware for correlation tracking (`X-Request-ID`) and request audit logging.
2. Authentication (`/api/v1/auth`):
   - Users authenticate with username and password.
   - Passwords are verified using direct `bcrypt.checkpw`.
   - Successful authentication issues a signed HS256 JWT access token encoding user identity and RBAC role (`analyst`, `manager`, `admin`).
3. Anomaly and Alert APIs:
   - `/api/v1/anomalies/detect` dynamically computes statistical confidence based on standard deviation deviations rather than returning static numbers.
   - `/api/v1/suppliers/{supplier_id}/history` returns the true 30-day trend of reliability, delivery delays, and inventory levels directly from database records.
4. Mitigation & Governance (`/api/v1/mitigation/action`):
   - Protected endpoint requiring `manager` or `admin` role.
   - `operator_id` is extracted strictly from the verified JWT payload—never trusted from the HTTP request body.
   - If PostgreSQL persistence to `audit_logs` fails, the backend immediately returns `503 Service Unavailable` to guarantee zero unlogged operator actions.

### Step 4: Streamlit Visual Operations Center
1. On startup, Streamlit checks the FastAPI backend `/health` endpoint.
2. If unreachable and `DEMO_MODE` is not explicitly set to `true`, the dashboard halts execution and displays the exact terminal commands required to boot the backend and database.
3. If `DEMO_MODE=true`, a prominent red banner warns operators: `"Demo data, not from the database"`.
4. Operators log in via the interactive authentication gate.
5. Tabs provide clear, plain-language insights:
   - "Flagged as Unusual" / "Normal" statuses.
   - Editable financial exposure inputs.
   - Model lineage panel displaying the model version, trained timestamp, test metrics vs baselines, and synthetic evaluation disclaimer.
