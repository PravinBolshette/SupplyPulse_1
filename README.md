# SupplyPulse: Enterprise AI Supply Chain Disruption & Risk Monitoring Platform

SupplyPulse is a production-grade supply chain early-warning platform. It combines machine learning anomaly detection (Isolation Forest with baseline comparisons against One-Class SVM and LOF), physical multi-tier inventory Time-To-Impact (TTI) depletion models, NLP-driven geopolitical and weather event ingestion, and a FastAPI backend with a Streamlit operations center governed by strict Human-in-the-Loop (HITL) and Role-Based Access Control (RBAC).

---

## Complete Setup Guide (Exact Order)

Execute the following commands in sequence to install, configure, provision, train, evaluate, and launch the platform.

### Step 1: Create Virtual Environment
- **Windows PowerShell:**
  ```powershell
  python -m venv .venv
  ```
- **Linux / macOS:**
  ```bash
  python3 -m venv .venv
  ```

### Step 2: Activate Virtual Environment
- **Windows PowerShell:**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

### Step 3: Install Dependencies
- **Windows PowerShell & Linux:**
  ```bash
  pip install -r requirements.txt
  ```

### Step 4: Download spaCy Language Model
- **Windows PowerShell & Linux:**
  ```bash
  python -m spacy download en_core_web_sm
  ```

### Step 5: Copy Environment Configuration
- **Windows PowerShell:**
  ```powershell
  Copy-Item .env.example .env
  ```
- **Linux / macOS:**
  ```bash
  cp .env.example .env
  ```

### Step 6: Teardown Any Stale Containers and Volumes
- **Windows PowerShell & Linux:**
  ```bash
  docker compose down -v
  ```

### Step 7: Provision PostgreSQL Database Container
- **Windows PowerShell & Linux:**
  ```bash
  docker compose up -d
  ```
  *(Verifies that PostgreSQL 15 initializes and mounts `database/init_schema.sql` automatically).*

### Step 8: Run ETL Pipelines
- **Windows PowerShell:**
  ```powershell
  python etl/generate_and_load.py
  python etl/nlp_event_ingestion.py
  ```
- **Linux / macOS:**
  ```bash
  python etl/generate_and_load.py && python etl/nlp_event_ingestion.py
  ```

### Step 9: Train Anomaly Detection Model
- **Windows PowerShell & Linux:**
  ```bash
  python phase1_detection/train_model.py
  ```

### Step 10: Evaluate Model and NLP Pipelines
- **Windows PowerShell:**
  ```powershell
  python phase1_detection/evaluate_model.py
  python etl/evaluate_nlp.py
  ```
- **Linux / macOS:**
  ```bash
  python phase1_detection/evaluate_model.py && python etl/evaluate_nlp.py
  ```

### Step 11: Execute Disruption Anomaly Detection
- **Windows PowerShell & Linux:**
  ```bash
  python phase1_detection/detect.py
  ```

### Step 12: Launch FastAPI Backend Service
- **Windows PowerShell & Linux:**
  ```bash
  uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
  ```
  *(Interactive Swagger API docs available at: `http://localhost:8000/docs`)*

### Step 13: Launch Streamlit Operations Center Dashboard
- **Windows PowerShell & Linux (in a separate terminal):**
  ```bash
  streamlit run streamlit_app/app.py
  ```
  *(Dashboard UI available at: `http://localhost:8501`)*

---

## System Verification & Automated Testing

### Comprehensive Verification Suite
To verify database connectivity, schema counts, foreign-key referential integrity, AI alert countdowns, and workspace artifacts:
```bash
python verify_system.py
```
*(Use `python verify_system.py --sqlite` if running against a local SQLite instance).*

### Automated Pytest Suite
Run all unit, integration, and security tests:
```bash
python -m pytest tests/ -v
```

---

## Architectural & Governance Documentation
Detailed documentation is maintained in the `docs/` directory:
- [Data Dictionary](docs/data_dictionary.md): Comprehensive reference of all relational tables, types, constraints, and data sources.
- [System Architecture](docs/architecture.md): High-level topology, component diagrams, request lifecycle, and data flow.
- [Model Card](docs/model_card.md): Model architecture, hyperparameter specs, temporal evaluation metrics vs baselines, and synthetic data disclaimer.
- [Responsible AI & HITL Framework](docs/responsible_ai.md): Human-in-the-loop mitigation workflows, false alarm vs stockout cost models, and 503 fail-safe guarantees.
