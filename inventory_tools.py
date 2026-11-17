"""Supply Chain Analytics Inventory Tools.

This module provides verified numerical calculation engines for the 5 fundamental
inventory management models:
1. Economic Order Quantity (EOQ)
2. Quantity Discount Model
3. Safety Stock and Reorder Point (Continuous Review)
4. Fixed-Period Review Model (Periodic Review / Order-up-to Level)
5. Newsvendor Model (Single-Period Stochastic Demand)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy import stats


# -----------------------------------------------------------------------------
# Common Utility Functions
# -----------------------------------------------------------------------------

def unit_normal_loss(z: float) -> float:
    """Calculate the standard normal loss function L(z) = phi(z) - z * (1 - Phi(z)).

    Used for expected lost sales and expected shortage calculations.
    """
    phi = stats.norm.pdf(z)
    Phi = stats.norm.cdf(z)
    return float(phi - z * (1.0 - Phi))


def normalize_service_level(service_level: float) -> float:
    """Ensure service level is a float between 0.0 and 1.0."""
    if service_level > 1.0:
        service_level /= 100.0
    if not (0.0 < service_level < 1.0):
        raise ValueError(
            f"Service level must be between 0 and 1 (or 0% and 100%), got {service_level}"
        )
    return service_level


# -----------------------------------------------------------------------------
# 1. Economic Order Quantity (EOQ)
# -----------------------------------------------------------------------------

def calculate_eoq(
    annual_demand: float,
    order_cost: float,
    holding_cost: Optional[float] = None,
    holding_cost_rate: Optional[float] = None,
    unit_cost: Optional[float] = None,
    working_days_per_year: int = 250,
) -> Dict[str, Any]:
    """Calculate the Economic Order Quantity (EOQ) and associated cost metrics.

    Parameters
    ----------
    annual_demand : float
        Annual demand (D) in units/year. Must be > 0.
    order_cost : float
        Fixed ordering/setup cost (S) in dollars/order. Must be > 0.
    holding_cost : float, optional
        Annual inventory holding/carrying cost per unit (H) in dollars/unit/year.
    holding_cost_rate : float, optional
        Annual carrying cost rate as a fraction or percentage (i), e.g. 0.20 or 20%.
    unit_cost : float, optional
        Purchase cost per unit (C). Required if holding_cost_rate is given.
    working_days_per_year : int, default=250
        Operating days in a year to calculate cycle time in days.

    Returns
    -------
    dict
        Dictionary containing optimal order quantity, order frequency, cycle time,
        annual holding cost, annual ordering cost, and total annual inventory cost.
    """
    if annual_demand <= 0:
        raise ValueError(f"Annual demand must be positive, got {annual_demand}")
    if order_cost <= 0:
        raise ValueError(f"Order cost must be positive, got {order_cost}")

    # Determine unit holding cost H
    if holding_cost is not None and holding_cost > 0:
        h = float(holding_cost)
    elif holding_cost_rate is not None and unit_cost is not None:
        rate = holding_cost_rate / 100.0 if holding_cost_rate > 1.0 else holding_cost_rate
        if rate <= 0:
            raise ValueError(f"Holding cost rate must be positive, got {holding_cost_rate}")
        if unit_cost <= 0:
            raise ValueError(f"Unit cost must be positive, got {unit_cost}")
        h = float(rate * unit_cost)
    else:
        raise ValueError("Must provide either 'holding_cost' or both 'holding_cost_rate' and 'unit_cost'.")

    # Wilson EOQ Formula: Q* = sqrt(2 * D * S / H)
    eoq = math.sqrt((2.0 * annual_demand * order_cost) / h)
    eoq_rounded = round(eoq)

    # Performance metrics at optimal Q*
    orders_per_year = annual_demand / eoq
    cycle_time_years = eoq / annual_demand
    cycle_time_days = cycle_time_years * working_days_per_year

    annual_ordering_cost = (annual_demand / eoq) * order_cost
    annual_holding_cost = (eoq / 2.0) * h
    total_annual_holding_and_ordering = annual_ordering_cost + annual_holding_cost

    total_annual_cost = total_annual_holding_and_ordering
    annual_purchase_cost = None
    if unit_cost is not None and unit_cost > 0:
        annual_purchase_cost = annual_demand * unit_cost
        total_annual_cost += annual_purchase_cost

    return {
        "model": "Economic Order Quantity (EOQ)",
        "inputs": {
            "annual_demand": float(annual_demand),
            "order_cost": float(order_cost),
            "holding_cost_per_unit_year": float(h),
            "unit_cost": float(unit_cost) if unit_cost is not None else None,
            "working_days_per_year": int(working_days_per_year),
        },
        "results": {
            "optimal_order_quantity_exact": float(eoq),
            "optimal_order_quantity": int(eoq_rounded),
            "orders_per_year": float(round(orders_per_year, 2)),
            "cycle_time_days": float(round(cycle_time_days, 1)),
            "annual_ordering_cost": float(round(annual_ordering_cost, 2)),
            "annual_holding_cost": float(round(annual_holding_cost, 2)),
            "total_inventory_cost": float(round(total_annual_holding_and_ordering, 2)),
            "annual_purchase_cost": float(round(annual_purchase_cost, 2)) if annual_purchase_cost else None,
            "total_annual_cost": float(round(total_annual_cost, 2)),
        },
        "formulas": {
            "eoq": "Q* = sqrt((2 * D * S) / H)",
            "orders_per_year": "N = D / Q*",
            "cycle_time": "T = (Q* / D) * working_days",
            "annual_ordering_cost": "AOC = (D / Q*) * S",
            "annual_holding_cost": "AHC = (Q* / 2) * H",
            "total_inventory_cost": "TIC = AOC + AHC",
        },
    }


# -----------------------------------------------------------------------------
# 2. Quantity Discount Model
# -----------------------------------------------------------------------------

def calculate_quantity_discount(
    annual_demand: float,
    order_cost: float,
    tiers: List[Dict[str, Any]],
    holding_cost_rate: Optional[float] = None,
    fixed_holding_cost: Optional[float] = None,
) -> Dict[str, Any]:
    """Calculate the optimal order quantity under an all-units quantity discount schedule.

    Parameters
    ----------
    annual_demand : float
        Annual demand (D) in units/year.
    order_cost : float
        Fixed ordering cost (S) in dollars/order.
    tiers : list of dict
        Discount tiers. Each tier must have:
        - 'min_qty' (float or int)
        - 'unit_cost' (float)
        - 'max_qty' (optional float or int, defaults to infinity)
    holding_cost_rate : float, optional
        Carrying cost rate as fraction of unit cost (e.g. 0.20 for 20%).
    fixed_holding_cost : float, optional
        Constant holding cost per unit per year, independent of purchase price.

    Returns
    -------
    dict
        Evaluation of each tier and the overall cost-minimizing order quantity.
    """
    if annual_demand <= 0:
        raise ValueError(f"Annual demand must be positive, got {annual_demand}")
    if order_cost <= 0:
        raise ValueError(f"Order cost must be positive, got {order_cost}")
    if not tiers:
        raise ValueError("Must provide at least one discount tier")
    if holding_cost_rate is None and fixed_holding_cost is None:
        raise ValueError("Must provide either 'holding_cost_rate' or 'fixed_holding_cost'.")

    rate = None
    if holding_cost_rate is not None:
        rate = holding_cost_rate / 100.0 if holding_cost_rate > 1.0 else holding_cost_rate
        if rate <= 0:
            raise ValueError(f"Holding cost rate must be positive, got {holding_cost_rate}")

    sorted_tiers = sorted(tiers, key=lambda x: x.get("min_qty", 0))
    tier_evaluations = []
    best_cost = float("inf")
    best_tier = None
    best_q = 0.0

    for idx, tier in enumerate(sorted_tiers):
        min_q = float(tier.get("min_qty", 0))
        max_q = float(tier.get("max_qty", float("inf")))
        c = float(tier["unit_cost"])

        if fixed_holding_cost is not None and fixed_holding_cost > 0:
            h = float(fixed_holding_cost)
        else:
            h = float(rate * c)

        # Unconstrained EOQ for this tier
        q_unconstrained = math.sqrt((2.0 * annual_demand * order_cost) / h)

        # Feasibility check: adjust to price break if below min_q
        if q_unconstrained < min_q:
            candidate_q = min_q
            status = f"Adjusted to price break threshold ({min_q:,.0f})"
        elif q_unconstrained > max_q:
            # Beyond tier upper bound, will be evaluated at next tier
            candidate_q = max_q
            status = f"Exceeds tier maximum ({max_q:,.0f}); dominated by higher discount tier"
        else:
            candidate_q = q_unconstrained
            status = "Feasible unconstrained EOQ in tier range"

        # Calculate costs for candidate quantity
        purchase_cost = annual_demand * c
        ordering_cost = (annual_demand / candidate_q) * order_cost
        holding_cost = (candidate_q / 2.0) * h
        total_cost = purchase_cost + ordering_cost + holding_cost

        eval_data = {
            "tier_index": idx + 1,
            "min_qty": min_q,
            "max_qty": max_q if max_q != float("inf") else "inf",
            "unit_cost": c,
            "holding_cost_unit": round(h, 4),
            "unconstrained_eoq": round(q_unconstrained, 2),
            "candidate_order_quantity": round(candidate_q, 2),
            "status": status,
            "annual_purchase_cost": round(purchase_cost, 2),
            "annual_ordering_cost": round(ordering_cost, 2),
            "annual_holding_cost": round(holding_cost, 2),
            "total_annual_cost": round(total_cost, 2),
        }
        tier_evaluations.append(eval_data)

        if total_cost < best_cost:
            best_cost = total_cost
            best_tier = idx + 1
            best_q = candidate_q

    base_cost = tier_evaluations[0]["total_annual_cost"]
    savings_vs_base = base_cost - best_cost

    return {
        "model": "Quantity Discount Model",
        "inputs": {
            "annual_demand": float(annual_demand),
            "order_cost": float(order_cost),
            "holding_cost_rate": rate,
            "fixed_holding_cost": fixed_holding_cost,
        },
        "tier_evaluations": tier_evaluations,
        "results": {
            "optimal_tier": best_tier,
            "optimal_order_quantity": round(best_q, 2),
            "optimal_order_quantity_rounded": int(round(best_q)),
            "minimum_total_annual_cost": round(best_cost, 2),
            "annual_savings_vs_base_tier": round(savings_vs_base, 2),
            "percent_savings": round((savings_vs_base / base_cost) * 100.0, 2) if base_cost > 0 else 0.0,
        },
        "decision_rule": "Calculate EOQ for each tier; if below price break, evaluate at minimum quantity breakpoint. Select quantity that minimizes Total Cost = Purchase + Ordering + Holding.",
    }


# -----------------------------------------------------------------------------
# 3. Safety Stock & Reorder Point (Continuous Review)
# -----------------------------------------------------------------------------

def calculate_safety_stock_rop(
    lead_time: float,
    mean_demand_per_period: Optional[float] = None,
    std_demand_per_period: Optional[float] = None,
    lead_time_demand_mean: Optional[float] = None,
    lead_time_demand_std: Optional[float] = None,
    service_level: float = 0.95,
    lead_time_std: Optional[float] = None,
) -> Dict[str, Any]:
    """Calculate Safety Stock and Reorder Point (ROP) under normal stochastic demand.

    Parameters
    ----------
    lead_time : float
        Replenishment lead time (L) in consistent time units (e.g. days or weeks).
    mean_demand_per_period : float, optional
        Mean demand per period (d).
    std_demand_per_period : float, optional
        Standard deviation of demand per period (sigma_d).
    lead_time_demand_mean : float, optional
        Direct lead-time demand mean (mu_L).
    lead_time_demand_std : float, optional
        Direct lead-time demand standard deviation (sigma_L).
    service_level : float, default=0.95
        Target Cycle Service Level (CSL) as decimal (0.95) or percent (95).
    lead_time_std : float, optional
        Standard deviation of lead time (sigma_L_time), if lead time itself is stochastic.

    Returns
    -------
    dict
        Safety stock, reorder point, z-score, expected shortage per replenishment cycle.
    """
    if lead_time < 0:
        raise ValueError(f"Lead time cannot be negative, got {lead_time}")

    csl = normalize_service_level(service_level)

    # Case 1: Stochastic lead time and stochastic demand
    if (
        lead_time_std is not None
        and lead_time_std > 0
        and mean_demand_per_period is not None
        and std_demand_per_period is not None
    ):
        mu_l = mean_demand_per_period * lead_time
        sigma_l = math.sqrt(
            lead_time * (std_demand_per_period ** 2)
            + (mean_demand_per_period ** 2) * (lead_time_std ** 2)
        )
        case_desc = "Stochastic demand and stochastic lead time"

    # Case 2: Constant lead time with stochastic period demand
    elif mean_demand_per_period is not None and std_demand_per_period is not None:
        if mean_demand_per_period < 0:
            raise ValueError("Mean demand per period cannot be negative")
        if std_demand_per_period < 0:
            raise ValueError("Standard deviation of demand cannot be negative")
        mu_l = mean_demand_per_period * lead_time
        sigma_l = math.sqrt(lead_time) * std_demand_per_period
        case_desc = "Stochastic demand with deterministic lead time"

    # Case 3: Direct lead time demand distribution given
    elif lead_time_demand_mean is not None and lead_time_demand_std is not None:
        if lead_time_demand_mean < 0:
            raise ValueError("Lead time demand mean cannot be negative")
        if lead_time_demand_std < 0:
            raise ValueError("Lead time demand std cannot be negative")
        mu_l = float(lead_time_demand_mean)
        sigma_l = float(lead_time_demand_std)
        case_desc = "Direct lead time demand distribution"
    else:
        raise ValueError(
            "Must provide either (mean_demand_per_period and std_demand_per_period) or "
            "(lead_time_demand_mean and lead_time_demand_std)."
        )

    z = float(stats.norm.ppf(csl))
    safety_stock = z * sigma_l
    rop = mu_l + safety_stock

    expected_shortage_per_cycle = sigma_l * unit_normal_loss(z)
    stockout_risk = 1.0 - csl

    return {
        "model": "Safety Stock & Reorder Point (Continuous Review)",
        "scenario": case_desc,
        "inputs": {
            "lead_time": float(lead_time),
            "service_level": float(csl),
            "mean_demand_per_period": float(mean_demand_per_period) if mean_demand_per_period is not None else None,
            "std_demand_per_period": float(std_demand_per_period) if std_demand_per_period is not None else None,
            "lead_time_std": float(lead_time_std) if lead_time_std is not None else None,
        },
        "results": {
            "expected_lead_time_demand": round(mu_l, 2),
            "lead_time_demand_std": round(sigma_l, 2),
            "z_score": round(z, 4),
            "safety_stock_exact": round(safety_stock, 2),
            "safety_stock": int(math.ceil(safety_stock)),
            "reorder_point_exact": round(rop, 2),
            "reorder_point": int(math.ceil(rop)),
            "cycle_service_level_percent": round(csl * 100.0, 2),
            "stockout_probability_percent": round(stockout_risk * 100.0, 2),
            "expected_shortage_per_cycle": round(expected_shortage_per_cycle, 4),
        },
        "formulas": {
            "lead_time_demand_std": "sigma_L = sqrt(L) * sigma_d",
            "safety_stock": "SS = z * sigma_L",
            "reorder_point": "ROP = mu_L + SS = (d * L) + (z * sigma_L)",
            "expected_shortage": "ESC = sigma_L * L(z)",
        },
    }


# -----------------------------------------------------------------------------
# 4. Fixed-Period Review Model (Periodic Review / Order-up-to Level)
# -----------------------------------------------------------------------------

def calculate_fixed_period_review(
    review_period: float,
    lead_time: float,
    mean_demand_per_period: float,
    std_demand_per_period: float,
    service_level: float = 0.95,
    on_hand_inventory: Optional[float] = None,
    on_order_inventory: float = 0.0,
) -> Dict[str, Any]:
    """Calculate Order-Up-To Level (S / M) and Recommended Order for Fixed-Period Review.

    Under periodic review (R, S), orders are placed every P periods to bring inventory
    position up to target level S. The protection interval is (P + L) periods.

    Parameters
    ----------
    review_period : float
        Review cycle interval (P or R) in periods (e.g. 7 days or 2 weeks).
    lead_time : float
        Supplier delivery lead time (L) in the same period units.
    mean_demand_per_period : float
        Average demand per period (d).
    std_demand_per_period : float
        Standard deviation of demand per period (sigma_d).
    service_level : float, default=0.95
        Cycle service level (CSL).
    on_hand_inventory : float, optional
        Current inventory on hand (I).
    on_order_inventory : float, default=0.0
        Existing pipeline orders (open orders placed but not yet received).

    Returns
    -------
    dict
        Protection interval, safety stock, target order-up-to level (S), and order quantity.
    """
    if review_period <= 0:
        raise ValueError(f"Review period must be positive, got {review_period}")
    if lead_time < 0:
        raise ValueError(f"Lead time cannot be negative, got {lead_time}")
    if mean_demand_per_period < 0:
        raise ValueError("Mean demand cannot be negative")
    if std_demand_per_period < 0:
        raise ValueError("Standard deviation cannot be negative")

    csl = normalize_service_level(service_level)
    protection_interval = review_period + lead_time

    # Demand over protection interval (P + L)
    mu_protection = mean_demand_per_period * protection_interval
    sigma_protection = math.sqrt(protection_interval) * std_demand_per_period

    z = float(stats.norm.ppf(csl))
    safety_stock = z * sigma_protection
    order_up_to_level = mu_protection + safety_stock

    recommended_order = None
    inventory_position = None
    if on_hand_inventory is not None:
        inventory_position = float(on_hand_inventory + on_order_inventory)
        recommended_order = max(0.0, order_up_to_level - inventory_position)

    return {
        "model": "Fixed-Period Review Model (Periodic Review)",
        "inputs": {
            "review_period": float(review_period),
            "lead_time": float(lead_time),
            "protection_interval": float(protection_interval),
            "mean_demand_per_period": float(mean_demand_per_period),
            "std_demand_per_period": float(std_demand_per_period),
            "service_level": float(csl),
            "on_hand_inventory": float(on_hand_inventory) if on_hand_inventory is not None else None,
            "on_order_inventory": float(on_order_inventory),
        },
        "results": {
            "protection_interval": float(round(protection_interval, 2)),
            "protection_interval_mean_demand": float(round(mu_protection, 2)),
            "protection_interval_std_demand": float(round(sigma_protection, 2)),
            "z_score": float(round(z, 4)),
            "safety_stock_exact": float(round(safety_stock, 2)),
            "safety_stock": int(math.ceil(safety_stock)),
            "order_up_to_level_exact": float(round(order_up_to_level, 2)),
            "order_up_to_level": int(math.ceil(order_up_to_level)),
            "current_inventory_position": float(round(inventory_position, 2)) if inventory_position is not None else None,
            "recommended_order_quantity_exact": float(round(recommended_order, 2)) if recommended_order is not None else None,
            "recommended_order_quantity": int(math.ceil(recommended_order)) if recommended_order is not None else None,
        },
        "formulas": {
            "protection_interval": "T_protect = P + L",
            "safety_stock": "SS = z * sigma_d * sqrt(P + L)",
            "order_up_to_level": "S = d * (P + L) + SS",
            "order_quantity": "Q = max(0, S - (On_Hand + On_Order))",
        },
    }


# -----------------------------------------------------------------------------
# 5. Newsvendor Model (Single-Period Stochastic Demand)
# -----------------------------------------------------------------------------

def calculate_newsvendor(
    unit_cost: float,
    selling_price: float,
    salvage_value: float = 0.0,
    mean_demand: float = 0.0,
    std_demand: float = 0.0,
    underage_penalty: float = 0.0,
) -> Dict[str, Any]:
    """Calculate the optimal order quantity and profit metrics for the Newsvendor model.

    Applicable to seasonal, perishable, or single-lifecycle inventory with stochastic demand.

    Parameters
    ----------
    unit_cost : float
        Purchase / manufacturing cost per unit (C).
    selling_price : float
        Regular retail sales price per unit (P). Must be > unit_cost.
    salvage_value : float, default=0.0
        Salvage value or clearance price for unsold inventory (V). Must be < unit_cost.
    mean_demand : float
        Forecasted average demand (mu). Must be > 0.
    std_demand : float
        Forecasted demand standard deviation (sigma). Must be >= 0.
    underage_penalty : float, default=0.0
        Additional penalty or customer goodwill loss per unit of lost sale.

    Returns
    -------
    dict
        Critical ratio, optimal order quantity Q*, expected sales, expected leftovers,
        expected lost sales, and expected total profit.
    """
    if selling_price <= unit_cost:
        raise ValueError(
            f"Selling price ({selling_price}) must exceed unit cost ({unit_cost}) for positive margin."
        )
    if salvage_value > unit_cost:
        raise ValueError(
            f"Salvage value ({salvage_value}) cannot exceed unit cost ({unit_cost})."
        )
    if mean_demand <= 0:
        raise ValueError(f"Mean demand must be positive, got {mean_demand}")
    if std_demand < 0:
        raise ValueError(f"Standard deviation cannot be negative, got {std_demand}")

    # Cost of Underage (Cu) and Cost of Overage (Co)
    cost_underage = (selling_price - unit_cost) + underage_penalty
    cost_overage = unit_cost - salvage_value

    # Critical Fractile (Critical Ratio CR)
    critical_ratio = cost_underage / (cost_underage + cost_overage)

    if std_demand == 0.0:
        q_star = mean_demand
        z_star = 0.0
        expected_lost_sales = 0.0
        expected_leftovers = 0.0
        expected_sales = mean_demand
    else:
        z_star = float(stats.norm.ppf(critical_ratio))
        q_star = mean_demand + z_star * std_demand

        # Expected lost sales via standard normal loss function: ELS = sigma * L(z)
        expected_lost_sales = std_demand * unit_normal_loss(z_star)

        # Expected leftovers: ELI = Q - mu + ELS
        expected_leftovers = q_star - mean_demand + expected_lost_sales

        # Expected sales: ES = mu - ELS
        expected_sales = mean_demand - expected_lost_sales

    # Profit calculations
    expected_revenue = selling_price * expected_sales
    expected_salvage = salvage_value * expected_leftovers
    expected_total_cost = unit_cost * q_star
    expected_profit = expected_revenue + expected_salvage - expected_total_cost

    # Maximum potential profit (perfect information benchmark)
    max_potential_profit = (selling_price - unit_cost) * mean_demand
    expected_value_of_perfect_information = max_potential_profit - expected_profit

    fill_rate = (expected_sales / mean_demand) * 100.0 if mean_demand > 0 else 100.0

    return {
        "model": "Newsvendor (Single-Period Stochastic Demand)",
        "inputs": {
            "unit_cost": float(unit_cost),
            "selling_price": float(selling_price),
            "salvage_value": float(salvage_value),
            "mean_demand": float(mean_demand),
            "std_demand": float(std_demand),
            "underage_penalty": float(underage_penalty),
        },
        "results": {
            "cost_of_underage_Cu": round(cost_underage, 2),
            "cost_of_overage_Co": round(cost_overage, 2),
            "critical_ratio": round(critical_ratio, 4),
            "critical_ratio_percent": round(critical_ratio * 100.0, 2),
            "z_score": round(z_star, 4),
            "optimal_order_quantity_exact": round(q_star, 2),
            "optimal_order_quantity": int(round(q_star)),
            "expected_sales": round(expected_sales, 2),
            "expected_lost_sales": round(expected_lost_sales, 2),
            "expected_leftover_inventory": round(expected_leftovers, 2),
            "expected_profit": round(expected_profit, 2),
            "fill_rate_percent": round(fill_rate, 2),
            "expected_value_of_perfect_information": round(max_potential_profit - expected_profit, 2),
        },
        "formulas": {
            "cost_of_underage": "Cu = Price - Cost + Underage_Penalty",
            "cost_of_overage": "Co = Cost - Salvage_Value",
            "critical_ratio": "CR = Cu / (Cu + Co)",
            "optimal_quantity": "Q* = mu + z* * sigma where Phi(z*) = CR",
            "expected_profit": "E[Profit] = P * E[Sales] + V * E[Leftovers] - C * Q*",
        },
    }
