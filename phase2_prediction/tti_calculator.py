#!/usr/bin/env python3
"""
SupplyPulse: Dynamic Demand Forecasting & Time-To-Impact (TTI) Engine
=====================================================================
Enterprise-grade inventory depletion modeling and automated mitigation allocation.

Mathematical Formulation:
    Predicted TTI (Days) = Current Stock Units / Dynamic Daily Consumption Rate
    Dynamic Daily Consumption Rate = Base Daily Consumption Rate * (1 + Demand Adjustment Factor)

Risk Classification:
    - CRITICAL_STOCKOUT: Current Stock <= 0 (Immediate stockout)
    - CRITICAL: TTI <= Lead Time * 0.5 (Imminent stockout before replenishment)
    - HIGH:     TTI <= Lead Time (Replenishment lead time breached)
    - MODERATE: TTI <= Lead Time * 1.5 (Buffer depletion warning)
    - STABLE:   TTI > Lead Time * 1.5 (Nominal inventory buffer)

Author: SupplyPulse Team (Pradymna Adhari, Omkar Akhade, Vishal Amrutkar, Pravin Bolshette)
Review: Prof. Vanita Babbane (VIT Pune) | Fujitsu Data & AI Portfolio
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple


# ---------------------------------------------------------------------------
# Data Classes & Domain Entities
# ---------------------------------------------------------------------------
@dataclass
class Supplier:
    """Represents a Tier-1 or pre-mapped alternative backup supplier."""
    supplier_id: str
    name: str
    lead_time_days: float
    cost_per_unit: float
    reliability_score: float  # Bounded strictly [0.0, 1.0]

    def __post_init__(self):
        if not (0.0 <= self.reliability_score <= 1.0):
            raise ValueError(f"reliability_score must be in [0.0, 1.0], got {self.reliability_score}")
        if self.lead_time_days < 0:
            raise ValueError(f"lead_time_days cannot be negative, got {self.lead_time_days}")


@dataclass
class InventoryItem:
    """Represents a warehouse SKU with inventory levels, burn rates, and backup routes."""
    sku: str
    name: str
    current_stock: float
    base_daily_consumption: float
    lead_time_days: float
    backup_suppliers: list[Supplier] = field(default_factory=list)


@dataclass
class DemandForecastTrend:
    """Captures forward-looking dynamic demand surge or contraction forecasts."""
    sku: str
    trend_multiplier: float  # e.g., +0.15 for +15% demand surge, -0.05 for -5% drop
    confidence_interval: tuple[float, float] = (0.0, 0.0)


@dataclass
class TTIResult:
    """Detailed evaluation result of the inventory depletion & TTI calculation."""
    sku: str
    predicted_tti_days: float
    adjusted_daily_rate: float
    risk_level: str  # "CRITICAL_STOCKOUT", "CRITICAL", "HIGH", "MODERATE", "STABLE"
    lead_time_breached: bool
    allocated_backup_supplier: Optional[Supplier] = None
    explanation: str = ""


# ---------------------------------------------------------------------------
# Core Time-To-Impact (TTI) Calculator Engine
# ---------------------------------------------------------------------------
class TTICalculator:
    """
    Dynamic Demand Forecasting & Time-To-Impact (TTI) Engine.
    Evaluates warehouse inventory depletion under variable demand conditions
    and allocates optimal backup suppliers for high-risk SKUs.
    """

    def calculate_dynamic_consumption(
        self,
        item: InventoryItem,
        trend: Optional[DemandForecastTrend] = None,
    ) -> float:
        """
        Calculates the forward-looking dynamic consumption rate:
            Dynamic Rate = Base Rate * (1 + Demand Adjustment Factor)
        """
        if trend is not None:
            adjustment = 1.0 + trend.trend_multiplier
            dynamic_rate = item.base_daily_consumption * adjustment
        else:
            dynamic_rate = item.base_daily_consumption

        return dynamic_rate

    def map_backup_supplier(
        self,
        item: InventoryItem,
        predicted_tti_days: float,
        risk_level: str,
    ) -> Optional[Supplier]:
        """
        Dynamically allocates the best backup supplier from `item.backup_suppliers`.

        Ranking Formula:
            Score = (Reliability * 0.6) - (Normalized Lead Time * 0.4)

        Constraint:
            Prioritizes candidates where `lead_time_days <= predicted_tti_days`
            whenever feasible to prevent stockouts before the backup arrives.
        """
        if risk_level not in ("CRITICAL_STOCKOUT", "CRITICAL", "HIGH"):
            return None

        if not item.backup_suppliers:
            return None

        # Filter feasible suppliers whose lead time arrives before stock runs out
        feasible_candidates = [
            s for s in item.backup_suppliers if s.lead_time_days <= predicted_tti_days
        ]

        # If feasible candidates exist, select from them; otherwise fallback to fastest available
        candidates = feasible_candidates if feasible_candidates else item.backup_suppliers

        # Normalize lead times across candidates to [0.0, 1.0]
        lead_times = [s.lead_time_days for s in candidates]
        min_lt = min(lead_times)
        max_lt = max(lead_times)

        def score_supplier(s: Supplier) -> float:
            if max_lt > min_lt:
                # Lower lead time -> lower normalized value -> smaller penalty -> higher score
                norm_lead_time = (s.lead_time_days - min_lt) / (max_lt - min_lt)
            else:
                norm_lead_time = 0.5 if max_lt > 0 else 0.0

            composite_score = (s.reliability_score * 0.6) - (norm_lead_time * 0.4)
            return composite_score

        # Select candidate with highest composite score (tie-breaking with lower cost)
        best_supplier = max(candidates, key=lambda s: (score_supplier(s), -s.cost_per_unit))
        return best_supplier

    def compute_tti(
        self,
        item: InventoryItem,
        trend: Optional[DemandForecastTrend] = None,
    ) -> TTIResult:
        """
        Computes the predicted TTI, checks for lead time breaches, classifies risk,
        and assigns backup suppliers if necessary.
        """
        adjusted_rate = self.calculate_dynamic_consumption(item, trend)

        # Edge Case 1: Immediate Stockout (Current stock <= 0)
        if item.current_stock <= 0.0:
            predicted_tti = 0.0
            risk_level = "CRITICAL_STOCKOUT"
            lead_time_breached = True
            explanation = (
                f"Immediate stockout: current stock is depleted ({item.current_stock} units). "
                f"Production halted."
            )
            backup = self.map_backup_supplier(item, predicted_tti, risk_level)
            return TTIResult(
                sku=item.sku,
                predicted_tti_days=0.0,
                adjusted_daily_rate=adjusted_rate,
                risk_level=risk_level,
                lead_time_breached=lead_time_breached,
                allocated_backup_supplier=backup,
                explanation=explanation,
            )

        # Edge Case 2: Zero or Negative Consumption Rate (Infinite TTI)
        if adjusted_rate <= 0.0:
            predicted_tti = float("inf")
            risk_level = "STABLE"
            lead_time_breached = False
            explanation = (
                f"Non-positive dynamic daily consumption rate ({adjusted_rate:.2f} units/day). "
                f"No stock depletion forecasted (Infinite TTI)."
            )
            return TTIResult(
                sku=item.sku,
                predicted_tti_days=float("inf"),
                adjusted_daily_rate=adjusted_rate,
                risk_level=risk_level,
                lead_time_breached=lead_time_breached,
                allocated_backup_supplier=None,
                explanation=explanation,
            )

        # Standard Depletion Model: TTI = Current Stock / Dynamic Daily Rate
        predicted_tti = item.current_stock / adjusted_rate
        lead_time_breached = bool(predicted_tti < item.lead_time_days)

        # Risk Level Classification Hierarchy
        if predicted_tti <= item.lead_time_days * 0.5:
            risk_level = "CRITICAL"
            explanation = (
                f"CRITICAL risk: Predicted TTI ({predicted_tti:.1f}d) is <= 50% of replenishment "
                f"lead time ({item.lead_time_days:.1f}d). Stockout imminent."
            )
        elif predicted_tti <= item.lead_time_days:
            risk_level = "HIGH"
            explanation = (
                f"HIGH risk: Predicted TTI ({predicted_tti:.1f}d) breaches replenishment "
                f"lead time ({item.lead_time_days:.1f}d). Orders placed today will arrive late."
            )
        elif predicted_tti <= item.lead_time_days * 1.5:
            risk_level = "MODERATE"
            explanation = (
                f"MODERATE risk: TTI ({predicted_tti:.1f}d) provides buffer between 1.0x and 1.5x "
                f"of lead time ({item.lead_time_days:.1f}d). Active monitoring recommended."
            )
        else:
            risk_level = "STABLE"
            explanation = (
                f"STABLE: TTI ({predicted_tti:.1f}d) comfortably exceeds 1.5x lead time "
                f"({item.lead_time_days:.1f}d)."
            )

        # Allocate backup supplier if risk is elevated
        backup = self.map_backup_supplier(item, predicted_tti, risk_level)

        return TTIResult(
            sku=item.sku,
            predicted_tti_days=round(predicted_tti, 2),
            adjusted_daily_rate=round(adjusted_rate, 2),
            risk_level=risk_level,
            lead_time_breached=lead_time_breached,
            allocated_backup_supplier=backup,
            explanation=explanation,
        )


# ---------------------------------------------------------------------------
# Batch Fleet Evaluation
# ---------------------------------------------------------------------------
def evaluate_inventory_fleet(
    inventory: list[InventoryItem],
    forecasts: Optional[dict[str, DemandForecastTrend]] = None,
) -> list[TTIResult]:
    """
    Evaluates a fleet of warehouse inventory items, calculates forward-looking TTI,
    and returns a prioritized list sorted ascending by predicted TTI days (most urgent first).
    """
    if forecasts is None:
        forecasts = {}

    calculator = TTICalculator()
    results: list[TTIResult] = []

    for item in inventory:
        trend = forecasts.get(item.sku)
        res = calculator.compute_tti(item, trend)
        results.append(res)

    # Sort ascending by predicted TTI (0.0 stockouts first, infinite TTI at the end)
    results.sort(key=lambda r: (math.isinf(r.predicted_tti_days), r.predicted_tti_days))
    return results


# ---------------------------------------------------------------------------
# Runnable Demonstration
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("""
================================================================================
   SUPPLYPULSE: DYNAMIC TIME-TO-IMPACT (TTI) ENGINE DEMONSTRATION
   Phase 2 Operational Stockout Forecasting & Automated Backup Allocation
================================================================================
    """)

    # 1. Define Candidate Backup Suppliers
    sup_alpha = Supplier("SUP-ALT-01", "Hsinchu Wafer Packaging Corp", lead_time_days=3.0, cost_per_unit=45.0, reliability_score=0.95)
    sup_beta = Supplier("SUP-ALT-02", "Penang Precision Mechatronics", lead_time_days=6.0, cost_per_unit=38.0, reliability_score=0.88)
    sup_gamma = Supplier("SUP-ALT-03", "Stuttgart Micro Actuators GmbH", lead_time_days=2.0, cost_per_unit=55.0, reliability_score=0.92)

    # 2. Define Sample Warehouse Inventory Fleet (3 Key SKUs)
    sample_inventory = [
        InventoryItem(
            sku="SKU-SEMI-101",
            name="Automotive Microcontroller Unit",
            current_stock=180.0,
            base_daily_consumption=50.0,
            lead_time_days=7.0,
            backup_suppliers=[sup_alpha, sup_beta],
        ),
        InventoryItem(
            sku="SKU-SENS-204",
            name="Optical LiDAR Proximity Sensor",
            current_stock=30.0,
            base_daily_consumption=20.0,
            lead_time_days=5.0,
            backup_suppliers=[sup_gamma, sup_beta],
        ),
        InventoryItem(
            sku="SKU-POWR-309",
            name="High-Voltage Power Regulator",
            current_stock=1200.0,
            base_daily_consumption=40.0,
            lead_time_days=10.0,
            backup_suppliers=[sup_alpha, sup_gamma],
        ),
    ]

    # 3. Simulate Forward-Looking Demand Trends (Surge on Microcontrollers and LiDAR)
    simulated_trends = {
        "SKU-SEMI-101": DemandForecastTrend(sku="SKU-SEMI-101", trend_multiplier=0.20, confidence_interval=(0.15, 0.25)),  # +20% surge
        "SKU-SENS-204": DemandForecastTrend(sku="SKU-SENS-204", trend_multiplier=0.50, confidence_interval=(0.40, 0.60)),  # +50% surge
        "SKU-POWR-309": DemandForecastTrend(sku="SKU-POWR-309", trend_multiplier=-0.10, confidence_interval=(-0.15, -0.05)), # -10% drop
    }

    # 4. Execute Batch Fleet Evaluation
    eval_results = evaluate_inventory_fleet(sample_inventory, simulated_trends)

    # 5. Render Formatted Output Table
    print(f"+----------------+------------------+------------+-------------+--------------------+--------------------------------+")
    print(f"| {'SKU':<14} | {'PREDICTED TTI':<16} | {'DAILY BURN':<10} | {'RISK LEVEL':<11} | {'LEAD TIME BREACH':<18} | {'ALLOCATED BACKUP SUPPLIER':<30} |")
    print(f"+----------------+------------------+------------+-------------+--------------------+--------------------------------+")

    for res in eval_results:
        tti_str = f"{res.predicted_tti_days:.1f} days" if not math.isinf(res.predicted_tti_days) else "INFINITE"
        rate_str = f"{res.adjusted_daily_rate:.1f}/day"
        breach_str = "YES [BREACH]" if res.lead_time_breached else "NO [SAFE]"
        backup_str = res.allocated_backup_supplier.name[:30] if res.allocated_backup_supplier else "None Required"

        print(f"| {res.sku:<14} | {tti_str:<16} | {rate_str:<10} | {res.risk_level:<11} | {breach_str:<18} | {backup_str:<30} |")

    print(f"+----------------+------------------+------------+-------------+--------------------+--------------------------------+")
    print("\nDetailed Scenario Explanations:")
    for res in eval_results:
        print(f"  * [{res.sku}] Risk: {res.risk_level} -> {res.explanation}")
        if res.allocated_backup_supplier:
            print(f"    -> Mitigation Action: Route emergency PO to {res.allocated_backup_supplier.name} (Lead Time: {res.allocated_backup_supplier.lead_time_days}d)")
    print()
