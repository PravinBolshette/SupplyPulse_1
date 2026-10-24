"""
SupplyPulse: Unit Test Suite for TTI Calculator & Dynamic Depletion Engine
==========================================================================
Tests dynamic demand adjustment, depletion math, edge cases, and automated
backup supplier allocation.

Author: SupplyPulse Team
Review: Prof. Vanita Babbane (VIT Pune) | Fujitsu Data & AI Portfolio
"""

import math
import pytest
from phase2_prediction.tti_calculator import (
    Supplier,
    InventoryItem,
    DemandForecastTrend,
    TTIResult,
    TTICalculator,
    evaluate_inventory_fleet,
)


@pytest.fixture
def sample_suppliers() -> list[Supplier]:
    return [
        Supplier("SUP-01", "Fast Reliable Supplier", lead_time_days=3.0, cost_per_unit=50.0, reliability_score=0.95),
        Supplier("SUP-02", "Slow Cheap Supplier", lead_time_days=8.0, cost_per_unit=30.0, reliability_score=0.80),
        Supplier("SUP-03", "Ultra-Fast Supplier", lead_time_days=1.5, cost_per_unit=70.0, reliability_score=0.90),
    ]


@pytest.fixture
def calculator() -> TTICalculator:
    return TTICalculator()


# ---------------------------------------------------------------------------
# Test Case 1: Standard Depletion Calculation
# ---------------------------------------------------------------------------
def test_standard_depletion_calculation(calculator, sample_suppliers):
    """Verify nominal TTI computation without demand adjustments."""
    item = InventoryItem(
        sku="SKU-TEST-001",
        name="Standard Bearing",
        current_stock=500.0,
        base_daily_consumption=50.0,
        lead_time_days=5.0,
        backup_suppliers=sample_suppliers,
    )

    result = calculator.compute_tti(item, trend=None)

    assert result.sku == "SKU-TEST-001"
    assert result.adjusted_daily_rate == 50.0
    assert result.predicted_tti_days == 10.0
    assert result.lead_time_breached is False
    # TTI (10.0d) > lead_time (5.0d) * 1.5 (7.5d) -> STABLE
    assert result.risk_level == "STABLE"
    assert result.allocated_backup_supplier is None


# ---------------------------------------------------------------------------
# Test Case 2: Surge Demand Spike Adjustments
# ---------------------------------------------------------------------------
def test_surge_demand_spike_adjustment(calculator, sample_suppliers):
    """Verify that a forward-looking demand surge increases daily burn and reduces TTI."""
    item = InventoryItem(
        sku="SKU-TEST-002",
        name="Power Chip",
        current_stock=300.0,
        base_daily_consumption=40.0,
        lead_time_days=6.0,
        backup_suppliers=sample_suppliers,
    )
    # +25% surge forecast -> dynamic burn = 40.0 * 1.25 = 50.0
    surge_trend = DemandForecastTrend(sku="SKU-TEST-002", trend_multiplier=0.25)

    result = calculator.compute_tti(item, trend=surge_trend)

    assert result.adjusted_daily_rate == 50.0
    # TTI = 300.0 / 50.0 = 6.0 days
    assert result.predicted_tti_days == 6.0
    # Lead time = 6.0d, TTI = 6.0d -> TTI <= lead_time -> HIGH risk
    assert result.lead_time_breached is False  # exactly equal to lead time, not strictly less
    assert result.risk_level == "HIGH"
    # High risk should trigger backup allocation
    assert result.allocated_backup_supplier is not None


# ---------------------------------------------------------------------------
# Test Case 3: Zero Consumption / Infinite TTI Edge Case
# ---------------------------------------------------------------------------
def test_zero_or_negative_consumption_rate(calculator, sample_suppliers):
    """Verify division-by-zero protection returning float('inf') with an explanatory flag."""
    item_zero_base = InventoryItem(
        sku="SKU-TEST-003A",
        name="Obsolete Hardware",
        current_stock=200.0,
        base_daily_consumption=0.0,
        lead_time_days=4.0,
        backup_suppliers=sample_suppliers,
    )
    result_zero = calculator.compute_tti(item_zero_base, trend=None)
    assert math.isinf(result_zero.predicted_tti_days)
    assert result_zero.risk_level == "STABLE"
    assert result_zero.lead_time_breached is False
    assert "Infinite TTI" in result_zero.explanation

    # Negative dynamic consumption from a -100% contraction trend
    item_contract = InventoryItem(
        sku="SKU-TEST-003B",
        name="Idle Raw Material",
        current_stock=100.0,
        base_daily_consumption=20.0,
        lead_time_days=5.0,
    )
    trend_freeze = DemandForecastTrend(sku="SKU-TEST-003B", trend_multiplier=-1.0)
    result_contract = calculator.compute_tti(item_contract, trend=trend_freeze)
    assert math.isinf(result_contract.predicted_tti_days)
    assert result_contract.risk_level == "STABLE"


# ---------------------------------------------------------------------------
# Test Case 4: Immediate Stockout (0 or Negative Current Units)
# ---------------------------------------------------------------------------
def test_immediate_stockout_escalation(calculator, sample_suppliers):
    """Verify that current stock <= 0 immediately escalates to CRITICAL_STOCKOUT."""
    item_zero_stock = InventoryItem(
        sku="SKU-TEST-004",
        name="Exhausted Capacitor",
        current_stock=0.0,
        base_daily_consumption=30.0,
        lead_time_days=7.0,
        backup_suppliers=sample_suppliers,
    )

    result = calculator.compute_tti(item_zero_stock, trend=None)

    assert result.predicted_tti_days == 0.0
    assert result.risk_level == "CRITICAL_STOCKOUT"
    assert result.lead_time_breached is True
    assert result.allocated_backup_supplier is not None
    assert "Immediate stockout" in result.explanation


# ---------------------------------------------------------------------------
# Test Case 5: Lead Time Breach & Optimal Backup Supplier Allocation
# ---------------------------------------------------------------------------
def test_lead_time_breach_and_backup_mapping(calculator, sample_suppliers):
    """
    Verify that TTI < lead_time marks breach and selects the best backup
    preferring feasible candidates (lead_time <= TTI).
    """
    # Current stock = 50, Burn = 25 -> TTI = 2.0 days. Lead time = 10.0 days.
    # TTI (2.0d) <= lead_time * 0.5 (5.0d) -> CRITICAL risk.
    item = InventoryItem(
        sku="SKU-TEST-005",
        name="Critical Sensor",
        current_stock=50.0,
        base_daily_consumption=25.0,
        lead_time_days=10.0,
        backup_suppliers=sample_suppliers,  # Candidate 3 has lead_time=1.5d, Candidate 1 has 3.0d, Candidate 2 has 8.0d
    )

    result = calculator.compute_tti(item, trend=None)

    assert result.predicted_tti_days == 2.0
    assert result.lead_time_breached is True
    assert result.risk_level == "CRITICAL"
    # Feasible supplier: SUP-03 has lead_time_days=1.5 <= TTI (2.0d). SUP-01 (3.0d) is not feasible.
    assert result.allocated_backup_supplier is not None
    assert result.allocated_backup_supplier.supplier_id == "SUP-03"


# ---------------------------------------------------------------------------
# Test Case 6: Batch Fleet Evaluation & Ascending Urgency Prioritization
# ---------------------------------------------------------------------------
def test_batch_fleet_evaluation_prioritization(calculator, sample_suppliers):
    """Verify that batch evaluation sorts ascending by TTI (0 days first, infinite last)."""
    items = [
        InventoryItem("SKU-STABLE", "Stable Item", current_stock=1000.0, base_daily_consumption=10.0, lead_time_days=5.0), # TTI = 100d
        InventoryItem("SKU-OUT", "Stockout Item", current_stock=0.0, base_daily_consumption=20.0, lead_time_days=5.0),      # TTI = 0d
        InventoryItem("SKU-URGENT", "Urgent Item", current_stock=20.0, base_daily_consumption=10.0, lead_time_days=5.0),   # TTI = 2d
        InventoryItem("SKU-INF", "Zero Burn Item", current_stock=50.0, base_daily_consumption=0.0, lead_time_days=5.0),     # TTI = inf
    ]

    fleet_results = evaluate_inventory_fleet(items)

    assert len(fleet_results) == 4
    assert fleet_results[0].sku == "SKU-OUT"
    assert fleet_results[0].predicted_tti_days == 0.0
    assert fleet_results[1].sku == "SKU-URGENT"
    assert fleet_results[1].predicted_tti_days == 2.0
    assert fleet_results[2].sku == "SKU-STABLE"
    assert fleet_results[2].predicted_tti_days == 100.0
    assert fleet_results[3].sku == "SKU-INF"
    assert math.isinf(fleet_results[3].predicted_tti_days)


# ---------------------------------------------------------------------------
# Test Case 7: Data Model Validation
# ---------------------------------------------------------------------------
def test_supplier_validation():
    """Verify data model constraints on reliability and lead time."""
    with pytest.raises(ValueError, match="reliability_score"):
        Supplier("S-BAD", "Invalid Supplier", lead_time_days=2.0, cost_per_unit=10.0, reliability_score=1.5)

    with pytest.raises(ValueError, match="lead_time_days"):
        Supplier("S-BAD2", "Invalid Supplier", lead_time_days=-3.0, cost_per_unit=10.0, reliability_score=0.8)
