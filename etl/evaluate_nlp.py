#!/usr/bin/env python3
"""
SupplyPulse: NLP Event Classification & Entity Extraction Evaluator
===================================================================
Evaluates the rule-and-gazetteer NLP pipeline against the curated labeled dataset
`data/nlp_eval/headlines_labeled.csv`.

Computes:
1. Event-Type Classification Accuracy
2. Detailed Confusion Matrix Table (Actual vs. Predicted)
3. Region Extraction Accuracy & Method Breakdown
4. Per-Class Precision, Recall, and F1 Metrics

HONEST DISCLOSURE:
This pipeline is built on deterministic keyword heuristics combined with a maritime
place-name gazetteer, falling back to spaCy NER (en_core_web_sm) when the gazetteer misses.
THIS IS NOT DEEP LEARNING.

IMPORTANT NOTICE:
The labels in `headlines_labeled.csv` were drafted as an engineering evaluation baseline.
Your team MUST review and verify every ground-truth label before quoting any official accuracy
or production metrics.
"""

import os
import sys
from typing import Any, Dict, List, Tuple
from collections import Counter, defaultdict

import pandas as pd

# Add repo root to sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from etl.nlp_event_ingestion import parse_headline


DEFAULT_DATASET_PATH = os.path.join(REPO_ROOT, "data", "nlp_eval", "headlines_labeled.csv")


def evaluate_nlp_pipeline(csv_path: str = DEFAULT_DATASET_PATH) -> Dict[str, Any]:
    """
    Evaluates parse_headline over the labeled CSV and prints a complete performance audit.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Evaluation dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    total_samples = len(df)
    if total_samples < 100:
        print(f"WARNING: Dataset contains {total_samples} samples; at least 100 recommended.")

    predictions = []
    classes = sorted(list(df["expected_event_type"].unique()))
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    method_counter = Counter()

    correct_event_type = 0
    correct_region = 0

    for _, row in df.iterrows():
        headline = str(row["headline"])
        expected_type = str(row["expected_event_type"]).strip()
        expected_region = str(row["expected_region"]).strip()

        parsed = parse_headline(headline)
        pred_type = parsed["event_type"]
        pred_region = parsed["region"]
        method = parsed["method"]

        method_counter[method] += 1
        confusion_matrix[expected_type][pred_type] += 1

        if pred_type == expected_type:
            correct_event_type += 1

        # Region match: exact or mutual substring overlap
        if pred_region == expected_region or expected_region.lower() in pred_region.lower() or pred_region.lower() in expected_region.lower():
            correct_region += 1

        predictions.append({
            "headline": headline,
            "expected_type": expected_type,
            "pred_type": pred_type,
            "expected_region": expected_region,
            "pred_region": pred_region,
            "method": method,
            "severity": parsed["severity"],
        })

    accuracy = (correct_event_type / total_samples) * 100.0
    region_accuracy = (correct_region / total_samples) * 100.0

    # Print Report
    print("\n" + "=" * 90)
    print("   SUPPLYPULSE NLP EVALUATION REPORT: KEYWORD RULES + GAZETTEER + SPACY NER")
    print("=" * 90)
    print(f"  Total Labeled Headlines Evaluated : {total_samples}")
    print(f"  Event-Type Classification Accuracy: {accuracy:.2f}% ({correct_event_type}/{total_samples})")
    print(f"  Region Resolution Accuracy        : {region_accuracy:.2f}% ({correct_region}/{total_samples})")
    print("-" * 90)
    print("  Extraction Method Breakdown:")
    for m, cnt in method_counter.most_common():
        pct = (cnt / total_samples) * 100.0
        print(f"    - {m:<35}: {cnt:>3} ({pct:>5.1f}%)")
    print("=" * 90)

    # Print Confusion Matrix Table
    print("\n  CONFUSION MATRIX (Rows: Ground Truth Actual, Columns: Pipeline Prediction):")
    all_pred_classes = sorted(list(set(classes) | {p["pred_type"] for p in predictions}))
    
    # Header
    col_width = 14
    header_str = f"  {'ACTUAL / PREDICTED':<26} | " + " | ".join(f"{c[:col_width]:<{col_width}}" for c in all_pred_classes)
    print("  " + "-" * len(header_str))
    print(header_str)
    print("  " + "-" * len(header_str))

    for actual in classes:
        row_counts = [str(confusion_matrix[actual][pred_col]) for pred_col in all_pred_classes]
        row_str = f"  {actual:<26} | " + " | ".join(f"{cnt:^{col_width}}" for cnt in row_counts)
        print(row_str)

    print("  " + "-" * len(header_str) + "\n")

    # Per-Class Precision, Recall, F1
    print("  PER-CLASS METRICS SUMMARY:")
    print(f"  {'CLASS':<26} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10} | {'SUPPORT':<8}")
    print("  " + "-" * 74)

    per_class = {}
    for c in classes:
        tp = confusion_matrix[c][c]
        fp = sum(confusion_matrix[other][c] for other in classes if other != c)
        fn = sum(confusion_matrix[c][other] for other in all_pred_classes if other != c)
        support = sum(confusion_matrix[c].values())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "support": support}
        print(f"  {c:<26} | {prec:>9.2%} | {rec:>9.2%} | {f1:>9.4f} | {support:>8}")

    print("  " + "-" * 74)
    print("""
  ==============================================================================
  IMPORTANT NOTICE ON HUMAN VERIFICATION OF DRAFTED LABELS:
  The evaluation labels above were drafted as an engineering evaluation baseline.
  The supply chain operations team MUST review, cross-examine, and verify every
  single label in `data/nlp_eval/headlines_labeled.csv` before quoting any official
  production accuracy or compliance metrics.
  ==============================================================================
""")

    return {
        "total_samples": total_samples,
        "event_accuracy": accuracy,
        "region_accuracy": region_accuracy,
        "per_class": per_class,
        "confusion_matrix": dict(confusion_matrix),
    }


if __name__ == "__main__":
    evaluate_nlp_pipeline()
