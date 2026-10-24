# SupplyPulse Responsible AI & Human-in-the-Loop (HITL) Framework

## 1. Executive Summary
The SupplyPulse platform integrates machine learning anomaly detection and physical inventory depletion forecasting into critical supply chain operations. Because autonomous supply chain decisions can lead to millions of dollars in unneeded expedited freight or catastrophic factory stockouts, SupplyPulse enforces strict **Responsible AI Guardrails** and **Human-in-the-Loop (HITL)** governance.

The AI system is strictly **advisory**. No automated process is permitted to commit financial capital, re-route vessels, or alter supplier contracts without authenticated human manager authorization.

---

## 2. Core Governance Principles

### Principle 1: Advisory-Only Constraint
- The AI model functions solely as an intelligent early-warning radar.
- Predictions (e.g., Anomaly Scores, Predicted Time-To-Impact) and suggested actions (e.g., "Trigger emergency procurement from backup supplier") are presented to operators as actionable recommendations.
- Under no circumstances does the backend execute mitigation actions autonomously.

### Principle 2: Authenticated Human Approval & Role Permissions
Mitigation workflows are guarded by strict Role-Based Access Control (RBAC):
- **`analyst`**: May review alerts, inspect supplier telemetry history, view NLP news events, and examine model performance metrics. Analysts are restricted from executing or modifying mitigation decisions (HTTP 403 Forbidden).
- **`manager`**: Authorized to approve, modify, or reject mitigation actions, provide override justifications, and commit financial mitigation budgets.
- **`admin`**: Full operational privileges, including model retraining triggers (`POST /api/v1/model/train`) and user management.

### Principle 3: Non-Spoofable Operator Attribution
- The `operator_id` logged in the audit trail is **never** extracted from the HTTP request body.
- The backend middleware verifies the cryptographically signed JWT Bearer token and extracts `current_user["username"]` directly from the token claims. This prevents malicious or mistaken identity spoofing.

---

## 3. Financial Cost Trade-Offs: False Alarms vs. Stockouts

Supply chain risk mitigation balances two competing financial hazards:

$$\text{Total Risk Cost} = (\text{False Positive Count} \times \text{Cost}_{\text{False Alarm}}) + (\text{False Negative Count} \times \text{Cost}_{\text{Stockout}})$$

### Cost of a False Alarm (False Positive)
- Occurs when normal transit or supplier noise is flagged as a critical disruption.
- Mitigation impact: Unnecessary expedited air freight, inventory holding surcharges, or premature cancellation of standard purchase orders.
- Estimated Cost: **$5,000 – $25,000 per incident**.
- Mitigation Guardrail: The operator interface requires user-entered financial exposure estimation and an explicit text rationale (minimum 10 characters) before any action is approved.

### Cost of a Missed Disruption (False Negative)
- Occurs when an imminent supplier stockout is not flagged in time.
- Mitigation impact: Assembly line shutdowns, breach of customer SLAs, emergency spot-market part procurement, brand reputation loss.
- Estimated Cost: **$100,000 – $1,500,000+ per shutdown**.
- Mitigation Guardrail: The Isolation Forest model is tuned with a high sensitivity target (96.00% Recall) and low contamination parameter (0.10) to minimize false negatives while maintaining a 97.78% specificity.

---

## 4. Immutable Audit Logging & 503 Fail-Safe

All human decisions responding to AI alerts are recorded in the `audit_logs` PostgreSQL table:
- **`log_id`**: Cryptographic UUID (`gen_random_uuid()`).
- **`timestamp`**: Non-modifiable UTC timestamp.
- **`operator_id`**: Authenticated username extracted from JWT.
- **`alert_id`**: Foreign key to the alert being actioned.
- **`ai_recommendation`**: Full JSON snapshot of the original AI advice.
- **`human_action`**: Final decision (`APPROVED`, `REJECTED`, `MODIFIED`).
- **`rationale`**: Mandatory justification text explaining why the human agreed with or overrode the model.
- **`financial_exposure_usd`**: Explicit user-entered financial impact.

### Strict 503 Fail-Safe Rule
To guarantee an unbroken chain of custody, the backend implements a zero-loss audit guarantee:
```python
# In app/routers/mitigation.py
if not pg_saved:
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Audit log persistence to PostgreSQL failed. Action aborted to maintain compliance.",
    )
```
If the database write fails, the mitigation action is aborted and returns HTTP 503. No unrecorded decision can ever be finalized.

---

## 5. Synthetic Data Limitations & Continuous Re-evaluation

> [!IMPORTANT]
> **Synthetic Data Disclaimer**:
> Current anomaly detection and TTI prediction models are trained and benchmarked on synthetic baseline distributions generated via `etl/generate_and_load.py` and evaluated via `phase1_detection/evaluate_model.py`.
>
> 1. Synthetic data ensures privacy and reproducibility during development.
> 2. Real-world distribution shifts (e.g., labor strikes, unexpected canal blockages, extreme climate events) require continuous validation.
> 3. Ground-truth NLP headlines (`data/nlp_eval/headlines_labeled.csv`) must be reviewed and labeled by human domain experts before quoting final operational accuracy.
