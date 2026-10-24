#!/usr/bin/env python3
"""
SupplyPulse: Phase 1 AI Model Evaluation & Visualization Suite
==============================================================
Rigorous accuracy benchmarking and graphical performance diagnostics for
the Isolation Forest anomaly detection model.

Generates:
  1. Comprehensive classification metrics:
     - Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, ROC-AUC, PR-AUC.
  2. Publication-quality graphical evaluation figures (Saved in `phase1_detection/evaluation_plots/`):
     - Figure 1: 4-Panel Master Performance Dashboard (Confusion Matrix, ROC Curve, Score Distribution, PR Curve).
     - Figure 2: Operational Decision Boundary Plot (Port Delay vs. Inventory Stock Buffer).
     - Figure 3: Feature Divergence & Anomaly Sensitivity Analysis.
  3. Interactive HTML Evaluation Report (`phase1_detection/evaluation_report.html`).

Author: SupplyPulse Team (Pradymna Adhari, Omkar Akhade, Vishal Amrutkar, Pravin Bolshette)
Review: Prof. Vanita Babbane (VIT Pune) | Fujitsu Data & AI Portfolio
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for automated generation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SupplyPulse.EvaluateModel")

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "isolation_forest_model.pkl")
OUTPUT_PLOTS_DIR = os.path.join(os.path.dirname(__file__), "evaluation_plots")
OUTPUT_HTML_REPORT = os.path.join(os.path.dirname(__file__), "evaluation_report.html")

FEATURE_COLUMNS = [
    "avg_port_congestion_delay",
    "max_port_congestion_delay",
    "delivery_delay_variance",
    "order_volume_drop_pct",
    "stock_buffer_days",
    "external_disruption_severity",
]


# ---------------------------------------------------------------------------
# Evaluation Benchmark Dataset Generation
# ---------------------------------------------------------------------------
def generate_labeled_benchmark_dataset(
    n_samples: int = 300,
    anomaly_ratio: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Constructs a ground-truth labeled benchmark test set (y_true: 0 = Nominal, 1 = Disruption Anomaly)
    simulating realistic supply chain variance across both normal and stressed scenarios.
    """
    logger.info(f"Generating ground-truth evaluation benchmark ({n_samples} samples, {anomaly_ratio*100:.0f}% anomalies)...")
    np.random.seed(RANDOM_SEED + 99)  # Distinct seed from training to evaluate generalization

    n_anomalies = int(n_samples * anomaly_ratio)
    n_nominal = n_samples - n_anomalies

    # Class 0: Nominal Operational Scenarios (y = 0)
    nom_port_avg = np.random.uniform(0.6, 2.4, n_nominal)
    nom_port_max = nom_port_avg + np.random.uniform(0.4, 1.8, n_nominal)
    nom_deliv_var = np.random.gamma(shape=1.4, scale=0.7, size=n_nominal)
    nom_vol_drop = np.random.uniform(0.0, 14.0, n_nominal)
    nom_stock_buf = np.random.uniform(16.0, 58.0, n_nominal)
    nom_ext_sev = np.random.uniform(0.0, 4.2, n_nominal)

    df_nominal = pd.DataFrame({
        "avg_port_congestion_delay": nom_port_avg,
        "max_port_congestion_delay": nom_port_max,
        "delivery_delay_variance": nom_deliv_var,
        "order_volume_drop_pct": nom_vol_drop,
        "stock_buffer_days": nom_stock_buf,
        "external_disruption_severity": nom_ext_sev,
        "ground_truth": 0,
    })

    # Class 1: Disruption Anomaly Scenarios (y = 1)
    # Sub-scenario A: Maritime port bottleneck spikes
    # Sub-scenario B: Critical warehouse stockout depletion
    # Sub-scenario C: High-severity natural disaster / geopolitical shock
    anom_port_avg = np.random.uniform(4.8, 11.2, n_anomalies)
    anom_port_max = anom_port_avg + np.random.uniform(2.0, 5.5, n_anomalies)
    anom_deliv_var = np.random.uniform(3.2, 9.0, n_anomalies)
    anom_vol_drop = np.random.uniform(28.0, 75.0, n_anomalies)
    anom_stock_buf = np.random.uniform(0.3, 3.2, n_anomalies)
    anom_ext_sev = np.random.uniform(6.8, 9.9, n_anomalies)

    df_anomaly = pd.DataFrame({
        "avg_port_congestion_delay": anom_port_avg,
        "max_port_congestion_delay": anom_port_max,
        "delivery_delay_variance": anom_deliv_var,
        "order_volume_drop_pct": anom_vol_drop,
        "stock_buffer_days": anom_stock_buf,
        "external_disruption_severity": anom_ext_sev,
        "ground_truth": 1,
    })

    df_bench = pd.concat([df_nominal, df_anomaly], ignore_index=True)
    df_bench = df_bench.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    df_bench = df_bench.round(2)

    X_test = df_bench[FEATURE_COLUMNS].to_numpy()
    y_true = df_bench["ground_truth"].to_numpy()

    logger.info(f"Benchmark test dataset assembled: X={X_test.shape}, y={y_true.shape} (Anomalies: {n_anomalies}, Nominal: {n_nominal})")
    return X_test, y_true, df_bench


# ---------------------------------------------------------------------------
# Metric Calculation
# ---------------------------------------------------------------------------
def compute_evaluation_metrics(
    model,
    X_test: np.ndarray,
    y_true: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes all standard machine learning classification metrics.
    In Isolation Forest:
      model.predict() returns 1 (nominal) and -1 (outlier/anomaly).
      model.decision_function() returns raw scores (lower/more negative = more anomalous).
    """
    logger.info("Computing evaluation metrics on benchmark dataset...")

    # Isolation Forest predictions: 1 = Inlier (Nominal), -1 = Outlier (Anomaly)
    raw_preds = model.predict(X_test)
    y_pred = (raw_preds == -1).astype(int)  # 1 = Anomaly, 0 = Nominal

    # Raw decision scores: higher = normal, lower = anomalous
    # Invert scores so that higher score represents higher anomaly probability for ROC-AUC
    raw_scores = model.decision_function(X_test)
    anomaly_scores = -raw_scores

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, anomaly_scores)
    pr_auc = average_precision_score(y_true, anomaly_scores)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": specificity,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "y_pred": y_pred,
        "raw_scores": raw_scores,
        "anomaly_scores": anomaly_scores,
    }

    return metrics


# ---------------------------------------------------------------------------
# Visualization Generation (3 High-Resolution Figures)
# ---------------------------------------------------------------------------
def generate_visualizations(
    metrics: Dict[str, Any],
    X_test: np.ndarray,
    y_true: np.ndarray,
    df_bench: pd.DataFrame,
    output_dir: str = OUTPUT_PLOTS_DIR,
) -> Dict[str, str]:
    """
    Renders and saves 3 publication-quality diagnostic figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Generating graphical diagnostic plots in '{output_dir}'...")

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    saved_plots = {}

    # -----------------------------------------------------------------------
    # Figure 1: 4-Panel Master Performance Dashboard
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("SupplyPulse: Phase 1 Isolation Forest Anomaly Detection Dashboard", fontsize=18, fontweight="bold", y=0.98)

    # Panel A: Confusion Matrix Heatmap
    ax_cm = axes[0, 0]
    cm = metrics["confusion_matrix"]
    cm_labels = np.array([
        [f"True Neg (TN)\n{cm[0, 0]}", f"False Pos (FP)\n{cm[0, 1]}"],
        [f"False Neg (FN)\n{cm[1, 0]}", f"True Pos (TP)\n{cm[1, 1]}"],
    ])
    sns.heatmap(cm, annot=cm_labels, fmt="", cmap="Blues", cbar=False, ax=ax_cm, annot_kws={"fontsize": 13, "fontweight": "bold"})
    ax_cm.set_title("A. Confusion Matrix (Operational Validation)", fontsize=13, fontweight="bold", pad=10)
    ax_cm.set_xlabel("Predicted Label (0 = Nominal, 1 = Anomaly)", fontsize=11)
    ax_cm.set_ylabel("True Ground Truth (0 = Nominal, 1 = Anomaly)", fontsize=11)
    ax_cm.set_xticklabels(["Nominal (0)", "Disruption (1)"], fontsize=10)
    ax_cm.set_yticklabels(["Nominal (0)", "Disruption (1)"], fontsize=10, rotation=0)

    # Panel B: Receiver Operating Characteristic (ROC) Curve
    ax_roc = axes[0, 1]
    fpr, tpr, _ = roc_curve(y_true, metrics["anomaly_scores"])
    ax_roc.plot(fpr, tpr, color="#0052cc", lw=2.5, label=f"Isolation Forest (ROC-AUC = {metrics['roc_auc']:.4f})")
    ax_roc.plot([0, 1], [0, 1], color="#888888", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
    ax_roc.fill_between(fpr, tpr, alpha=0.15, color="#0052cc")
    ax_roc.set_title("B. Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold", pad=10)
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    ax_roc.legend(loc="lower right", frameon=True, fontsize=10)
    ax_roc.grid(True, linestyle="--", alpha=0.6)

    # Panel C: Anomaly Score Distribution (Nominal vs Disrupted)
    ax_dist = axes[1, 0]
    nominal_scores = metrics["raw_scores"][y_true == 0]
    anomaly_scores_dist = metrics["raw_scores"][y_true == 1]
    sns.kdeplot(nominal_scores, ax=ax_dist, color="#2e7d32", fill=True, alpha=0.35, lw=2, label="Nominal Operations (y=0)")
    sns.kdeplot(anomaly_scores_dist, ax=ax_dist, color="#d32f2f", fill=True, alpha=0.35, lw=2, label="Disruption Anomalies (y=1)")
    ax_dist.axvline(0.0, color="#212121", linestyle="--", lw=1.8, label="Decision Threshold (Score = 0.0)")
    ax_dist.set_title("C. Decision Score Separation (KDE Distribution)", fontsize=13, fontweight="bold", pad=10)
    ax_dist.set_xlabel("Isolation Forest Decision Score (Negative = Anomaly)", fontsize=11)
    ax_dist.set_ylabel("Density", fontsize=11)
    ax_dist.legend(loc="upper left", frameon=True, fontsize=10)
    ax_dist.grid(True, linestyle="--", alpha=0.6)

    # Panel D: Precision-Recall Curve
    ax_pr = axes[1, 1]
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, metrics["anomaly_scores"])
    ax_pr.plot(recall_curve, precision_curve, color="#7b1fa2", lw=2.5, label=f"PR Curve (Avg Precision = {metrics['pr_auc']:.4f})")
    baseline_pr = y_true.sum() / len(y_true)
    ax_pr.axhline(baseline_pr, color="#888888", linestyle="--", lw=1.5, label=f"No-Skill Baseline ({baseline_pr:.2f})")
    ax_pr.fill_between(recall_curve, precision_curve, alpha=0.15, color="#7b1fa2")
    ax_pr.set_title("D. Precision-Recall Curve (Disruption Detection)", fontsize=13, fontweight="bold", pad=10)
    ax_pr.set_xlabel("Recall (Coverage of Disruptions)", fontsize=11)
    ax_pr.set_ylabel("Precision (Accuracy of Raised Alerts)", fontsize=11)
    ax_pr.legend(loc="lower left", frameon=True, fontsize=10)
    ax_pr.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    dash_path = os.path.join(output_dir, "model_performance_dashboard.png")
    plt.savefig(dash_path, dpi=300)
    plt.close()
    saved_plots["dashboard"] = dash_path
    logger.info(f"Saved: {dash_path}")

    # -----------------------------------------------------------------------
    # Figure 2: Operational Decision Boundary in 2D Space
    # (Port Congestion Delay vs Inventory Stock Buffer Days)
    # -----------------------------------------------------------------------
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        df_bench["avg_port_congestion_delay"],
        df_bench["stock_buffer_days"],
        c=metrics["raw_scores"],
        cmap="coolwarm_r",
        s=65,
        alpha=0.85,
        edgecolors="k",
        linewidth=0.5,
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Anomaly Decision Score (More Negative = High Risk)", fontsize=11)

    # Annotate critical operational vulnerability quadrant
    plt.axvline(x=4.0, color="#d32f2f", linestyle=":", lw=1.5, label="High Port Delay Threshold (4.0d)")
    plt.axhline(y=5.0, color="#d32f2f", linestyle="--", lw=1.5, label="Critical Stockout Threshold (5.0d)")
    plt.fill_between([4.0, 14.0], [0.0, 0.0], [5.0, 5.0], color="#d32f2f", alpha=0.08, label="High Vulnerability Zone")

    plt.title("SupplyPulse: Operational Anomaly Boundary\n(Port Congestion vs. Warehouse Stock Buffer)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Average Port Congestion Delay (Days)", fontsize=12)
    plt.ylabel("Warehouse Stock Buffer Days (Current Stock / Daily Consumption)", fontsize=12)
    plt.xlim(0.0, 13.0)
    plt.ylim(0.0, 65.0)
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()

    boundary_path = os.path.join(output_dir, "feature_decision_boundary.png")
    plt.savefig(boundary_path, dpi=300)
    plt.close()
    saved_plots["boundary"] = boundary_path
    logger.info(f"Saved: {boundary_path}")

    # -----------------------------------------------------------------------
    # Figure 3: Feature Divergence & Sensitivity Analysis
    # -----------------------------------------------------------------------
    plt.figure(figsize=(11, 6))
    nom_means = df_bench[df_bench["ground_truth"] == 0][FEATURE_COLUMNS].mean()
    anom_means = df_bench[df_bench["ground_truth"] == 1][FEATURE_COLUMNS].mean()

    # Normalize differences into standardized z-score separation
    overall_stds = df_bench[FEATURE_COLUMNS].std()
    feature_shifts = ((anom_means - nom_means) / overall_stds).sort_values(ascending=True)

    readable_names = {
        "avg_port_congestion_delay": "Avg Port Congestion (Days)",
        "max_port_congestion_delay": "Max Port Congestion Spike (Days)",
        "delivery_delay_variance": "Delivery Transit Variance",
        "order_volume_drop_pct": "Order Volume Drop (%)",
        "stock_buffer_days": "Stock Buffer Days (Inverted)",
        "external_disruption_severity": "External Disaster Severity (0-10)",
    }
    plot_labels = [readable_names.get(f, f) for f in feature_shifts.index]

    colors = ["#d32f2f" if v > 0 else "#0052cc" for v in feature_shifts.values]
    bars = plt.barh(plot_labels, feature_shifts.values, color=colors, alpha=0.85, edgecolor="k", height=0.6)

    # Annotate bar values
    for bar in bars:
        val = bar.get_width()
        offset = 0.08 if val >= 0 else -0.35
        plt.text(val + offset, bar.get_y() + bar.get_height()/2, f"{val:+.2f}σ", va="center", fontsize=10, fontweight="bold")

    plt.axvline(0, color="k", lw=1.2)
    plt.title("SupplyPulse: Feature Divergence in Anomaly Detection\n(Normalized Shift in Disruption Conditions vs. Nominal Baseline)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Standardized Separation Shift (Z-Score Difference: Disruption vs. Nominal)", fontsize=11)
    plt.tight_layout()

    shift_path = os.path.join(output_dir, "feature_importance_analysis.png")
    plt.savefig(shift_path, dpi=300)
    plt.close()
    saved_plots["feature_shift"] = shift_path
    logger.info(f"Saved: {shift_path}")

    return saved_plots


# ---------------------------------------------------------------------------
# HTML Evaluation Dashboard Generator
# ---------------------------------------------------------------------------
def generate_html_evaluation_report(
    metrics: Dict[str, Any],
    plots: Dict[str, str],
    output_path: str = OUTPUT_HTML_REPORT,
):
    """
    Compiles an interactive, modern HTML report embedding metrics and graphs.
    """
    logger.info(f"Generating HTML evaluation report: '{output_path}'...")

    rel_dash = os.path.relpath(plots["dashboard"], os.path.dirname(output_path)).replace("\\", "/")
    rel_boundary = os.path.relpath(plots["boundary"], os.path.dirname(output_path)).replace("\\", "/")
    rel_shift = os.path.relpath(plots["feature_shift"], os.path.dirname(output_path)).replace("\\", "/")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SupplyPulse: Isolation Forest AI Evaluation Report</title>
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: #111827;
            --accent-blue: #2563eb;
            --accent-cyan: #06b6d4;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --text-main: #f9fafb;
            --text-muted: #9ca3af;
            --border-color: #1f2937;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 2.5rem;
            line-height: 1.6;
        }}
        .header {{
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 1.5rem;
            margin-bottom: 2rem;
        }}
        .header h1 {{
            font-size: 2.2rem;
            margin: 0 0 0.5rem 0;
            background: linear-gradient(90deg, #60a5fa, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .header p {{
            color: var(--text-muted);
            margin: 0;
            font-size: 1.05rem;
        }}
        .grid-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.2rem;
            margin-bottom: 2.5rem;
        }}
        .metric-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.4rem;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }}
        .metric-label {{
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.4rem;
        }}
        .metric-value {{
            font-size: 2.1rem;
            font-weight: 700;
            color: var(--text-main);
        }}
        .metric-badge {{
            display: inline-block;
            margin-top: 0.5rem;
            padding: 0.2rem 0.6rem;
            font-size: 0.75rem;
            border-radius: 9999px;
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-green);
        }}
        .section-title {{
            font-size: 1.4rem;
            font-weight: 700;
            margin: 2.5rem 0 1.2rem 0;
            border-left: 4px solid var(--accent-blue);
            padding-left: 0.8rem;
        }}
        .image-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4);
        }}
        .image-card img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            display: block;
        }}
        .image-caption {{
            margin-top: 1rem;
            font-size: 0.95rem;
            color: var(--text-muted);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            background: var(--card-bg);
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 0.9rem 1.2rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            background: #1e293b;
            font-size: 0.85rem;
            text-transform: uppercase;
            color: var(--text-muted);
        }}
        .footer {{
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border-color);
            text-align: center;
            font-size: 0.85rem;
            color: var(--text-muted);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>SupplyPulse AI Engine: Model Accuracy & Diagnostic Audit</h1>
        <p>Phase 1 Unsupervised Anomaly Detection (Isolation Forest) Benchmark Report | Fujitsu Data & AI Portfolio</p>
    </div>

    <div class="grid-metrics">
        <div class="metric-card">
            <div class="metric-label">Model Accuracy</div>
            <div class="metric-value">{metrics['accuracy']*100:.1f}%</div>
            <span class="metric-badge">Overall Accuracy</span>
        </div>
        <div class="metric-card">
            <div class="metric-label">ROC-AUC Score</div>
            <div class="metric-value">{metrics['roc_auc']:.4f}</div>
            <span class="metric-badge">Discriminative Power</span>
        </div>
        <div class="metric-card">
            <div class="metric-label">Precision (Anomalies)</div>
            <div class="metric-value">{metrics['precision']*100:.1f}%</div>
            <span class="metric-badge">Low False Alarms</span>
        </div>
        <div class="metric-card">
            <div class="metric-label">Recall / Sensitivity</div>
            <div class="metric-value">{metrics['recall']*100:.1f}%</div>
            <span class="metric-badge">Disruption Capture Rate</span>
        </div>
        <div class="metric-card">
            <div class="metric-label">F1-Score</div>
            <div class="metric-value">{metrics['f1_score']:.4f}</div>
            <span class="metric-badge">Harmonic Mean</span>
        </div>
        <div class="metric-card">
            <div class="metric-label">PR-AUC Score</div>
            <div class="metric-value">{metrics['pr_auc']:.4f}</div>
            <span class="metric-badge">Avg Precision</span>
        </div>
    </div>

    <div class="section-title">Confusion Matrix Breakdown</div>
    <table>
        <thead>
            <tr>
                <th>Metric Category</th>
                <th>Count (Samples)</th>
                <th>Operational Supply Chain Interpretation</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>True Positives (TP)</strong></td>
                <td><strong style="color: #10b981;">{metrics['tp']}</strong></td>
                <td>Severe disruptions correctly flagged for proactive procurement intervention.</td>
            </tr>
            <tr>
                <td><strong>True Negatives (TN)</strong></td>
                <td><strong style="color: #60a5fa;">{metrics['tn']}</strong></td>
                <td>Nominal supplier operations correctly identified without false alarms.</td>
            </tr>
            <tr>
                <td><strong>False Positives (FP)</strong></td>
                <td><strong style="color: #fbbf24;">{metrics['fp']}</strong></td>
                <td>Nominal operations flagged as suspicious (Type I Error / False Warning).</td>
            </tr>
            <tr>
                <td><strong>False Negatives (FN)</strong></td>
                <td><strong style="color: #ef4444;">{metrics['fn']}</strong></td>
                <td>Disruptions missed by the model (Type II Error / Blind Spot).</td>
            </tr>
        </tbody>
    </table>

    <div class="section-title">Figure 1: Master Performance Diagnostics Dashboard</div>
    <div class="image-card">
        <img src="{rel_dash}" alt="Model Performance Dashboard">
        <div class="image-caption">
            <strong>Figure 1 Description:</strong> (A) Annotated Confusion Matrix; (B) Receiver Operating Characteristic (ROC) Curve displaying {metrics['roc_auc']:.4f} AUC; (C) Kernel Density Estimation (KDE) demonstrating separation between nominal operations and disruption anomalies; (D) Precision-Recall curve with average precision of {metrics['pr_auc']:.4f}.
        </div>
    </div>

    <div class="section-title">Figure 2: Operational Decision Boundary (Port Delay vs Stock Buffer)</div>
    <div class="image-card">
        <img src="{rel_boundary}" alt="Feature Decision Boundary">
        <div class="image-caption">
            <strong>Figure 2 Description:</strong> Continuous Isolation Forest decision score mapping across maritime transit port delays (X-axis) and warehouse inventory stock buffer days (Y-axis). Identifies the critical vulnerability corridor where delays > 4.0 days intersect buffers < 5.0 days.
        </div>
    </div>

    <div class="section-title">Figure 3: Feature Divergence & Sensitivity Analysis</div>
    <div class="image-card">
        <img src="{rel_shift}" alt="Feature Divergence">
        <div class="image-caption">
            <strong>Figure 3 Description:</strong> Normalized standardized separation shift (&sigma;) across all 6 engineered supply chain features. Illustrates how port congestion spikes, external disaster severity, and stock buffer depletion drive anomaly classification.
        </div>
    </div>

    <div class="footer">
        Generated automatically by <strong>SupplyPulse Verification Engine</strong> &bull; {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &bull; Fujitsu-Compliant Academic Review
    </div>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"HTML evaluation report successfully written to: {os.path.abspath(output_path)}")


# ---------------------------------------------------------------------------
# CLI Argument Parser & Execution Flow
# ---------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="SupplyPulse: Phase 1 Isolation Forest Accuracy & Graphical Evaluation Suite"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to pre-trained isolation_forest_model.pkl",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=300,
        help="Number of labeled test samples for benchmark evaluation (default: 300)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_PLOTS_DIR,
        help="Directory to save generated evaluation plots",
    )
    return parser.parse_args()


def print_terminal_report(metrics: Dict[str, Any]):
    """
    Renders an ASCII-safe terminal classification report.
    """
    print("\n" + "=" * 80)
    print("  SUPPLYPULSE PHASE 1: MODEL ACCURACY & PERFORMANCE AUDIT")
    print("=" * 80)
    print(f"  * Overall Classification Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"  * ROC-AUC Score (Discrimination)  : {metrics['roc_auc']:.4f}")
    print(f"  * Precision (Anomaly Detection)   : {metrics['precision']*100:.2f}%")
    print(f"  * Recall / Sensitivity (Coverage) : {metrics['recall']*100:.2f}%")
    print(f"  * Specificity (Nominal Coverage)  : {metrics['specificity']*100:.2f}%")
    print(f"  * F1-Score (Harmonic Mean)        : {metrics['f1_score']:.4f}")
    print(f"  * PR-AUC (Average Precision)      : {metrics['pr_auc']:.4f}")
    print("-" * 80)
    print("  CONFUSION MATRIX BREAKDOWN:")
    print(f"    - True Negatives  (TN - Correct Nominal)     : {metrics['tn']:>4}")
    print(f"    - False Positives (FP - False Alarm)         : {metrics['fp']:>4}")
    print(f"    - False Negatives (FN - Missed Disruption)   : {metrics['fn']:>4}")
    print(f"    - True Positives  (TP - Detected Disruption) : {metrics['tp']:>4}")
    print("=" * 80 + "\n")


def main():
    print("""
    =================================================================
       SUPPLYPULSE: AI SUPPLY CHAIN DISRUPTION PLATFORM
       Phase 1 AI Model Evaluation & Graphical Visualization Suite
    =================================================================
    """)

    args = parse_arguments()

    # 1. Load Pre-trained Model
    if not os.path.exists(args.model_path):
        logger.error(
            f"Model artifact not found at: {args.model_path}\n"
            "Please train the model first by executing: python phase1_detection/train_model.py"
        )
        sys.exit(1)

    logger.info(f"Loading trained Isolation Forest model from: {args.model_path}")
    model = joblib.load(args.model_path)

    # 2. Generate Benchmark Dataset
    X_test, y_true, df_bench = generate_labeled_benchmark_dataset(
        n_samples=args.samples,
        anomaly_ratio=0.25,
    )

    # 3. Compute Metrics
    metrics = compute_evaluation_metrics(model, X_test, y_true)

    # 4. Display Terminal Audit
    print_terminal_report(metrics)

    # 5. Generate Visual Graphs
    plots = generate_visualizations(metrics, X_test, y_true, df_bench, output_dir=args.output_dir)

    # 6. Generate HTML Report
    generate_html_evaluation_report(metrics, plots, output_path=OUTPUT_HTML_REPORT)

    print("\n" + "=" * 80)
    print("  EVALUATION ASSETS GENERATED SUCCESSFULLY:")
    print(f"  1. Master Performance Dashboard : {plots['dashboard']}")
    print(f"  2. Decision Boundary Plot       : {plots['boundary']}")
    print(f"  3. Feature Importance Analysis  : {plots['feature_shift']}")
    print(f"  4. Interactive HTML Report      : {OUTPUT_HTML_REPORT}")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
