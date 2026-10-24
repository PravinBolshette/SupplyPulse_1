"""
Automated Test for NLP Headline Parsing and Evaluation (Task B2).

Verifies:
1. `parse_headline(text)` returns expected dictionary keys (event_type, region, severity, entities, method).
2. Honest extraction method attribution ('keyword_rules+gazetteer', 'keyword_rules+spacy_ner', etc.).
3. `evaluate_nlp_pipeline()` executes over `data/nlp_eval/headlines_labeled.csv` with >=100 samples
   and produces valid accuracy and confusion matrix metrics.
"""

import os
import pytest
from etl.nlp_event_ingestion import parse_headline
from etl.evaluate_nlp import evaluate_nlp_pipeline, DEFAULT_DATASET_PATH


def test_parse_headline_structure_and_types():
    """Validates returned keys, data types, and method description from parse_headline."""
    text = "Super Typhoon Saola approaching Shanghai port terminals with Category 4 storm surge"
    result = parse_headline(text)

    assert isinstance(result, dict)
    assert "event_type" in result
    assert "region" in result
    assert "severity" in result
    assert "entities" in result
    assert "method" in result

    assert result["event_type"] == "Typhoon"
    assert "Yangtze" in result["region"] or "Shanghai" in result["region"]
    assert result["severity"] >= 7.0
    assert isinstance(result["entities"], list)
    assert "keyword_rules" in result["method"]


def test_parse_headline_spacy_fallback():
    """Validates that unseen geographical location uses spaCy NER fallback when gazetteer misses."""
    # Location not in gazetteer: Reykjavik, Iceland
    text = "Volcanic eruption and toxic ash cloud halts air cargo near Reykjavik"
    result = parse_headline(text)

    assert result["event_type"] == "Severe Environmental Alert"
    assert "entities" in result
    assert "Reykjavik" in result["entities"] or "Reykjavik" in result["region"]
    assert result["method"] in ("keyword_rules+spacy_ner", "keyword_rules+heuristic_fallback")


def test_nlp_evaluation_dataset_and_accuracy():
    """Validates that labeled dataset exists with >=100 samples and accuracy >= 80%."""
    assert os.path.exists(DEFAULT_DATASET_PATH), f"Missing dataset: {DEFAULT_DATASET_PATH}"

    metrics = evaluate_nlp_pipeline(DEFAULT_DATASET_PATH)
    assert metrics["total_samples"] >= 100
    assert metrics["event_accuracy"] >= 80.0
    assert "confusion_matrix" in metrics
    assert len(metrics["per_class"]) >= 5
