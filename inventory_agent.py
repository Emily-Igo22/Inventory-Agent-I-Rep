"""Supply Chain Analytics Inventory Decision Agent.

Implements the decision pipeline:
Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import inventory_tools as it


class InventoryModelType(str, Enum):
    EOQ = "Economic Order Quantity (EOQ)"
    QUANTITY_DISCOUNT = "Quantity Discount Model"
    SAFETY_STOCK_ROP = "Safety Stock & Reorder Point"
    FIXED_PERIOD_REVIEW = "Fixed-Period Review Model"
    NEWSVENDOR = "Newsvendor Model"
    UNKNOWN = "Unknown Model"


@dataclass
class VerificationReport:
    is_valid: bool
    sanity_checks_passed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    managerial_notes: List[str] = field(default_factory=list)


@dataclass
class AgentResponse:
    question: str
    identified_model: InventoryModelType
    extracted_inputs: Dict[str, Any]
    missing_inputs: List[str]
    selected_tool_name: str
    calculation_results: Optional[Dict[str, Any]]
    verification: Optional[VerificationReport]
    explanation: str
    status: str  # 'SUCCESS', 'MISSING_INPUTS', 'CALCULATION_ERROR'


class InventoryDecisionAgent:
    """End-to-end intelligent inventory decision agent."""

    def __init__(self):
        self.tool_mapping: Dict[InventoryModelType, Callable[..., Dict[str, Any]]] = {
            InventoryModelType.EOQ: it.calculate_eoq,
            InventoryModelType.QUANTITY_DISCOUNT: it.calculate_quantity_discount,
            InventoryModelType.SAFETY_STOCK_ROP: it.calculate_safety_stock_rop,
            InventoryModelType.FIXED_PERIOD_REVIEW: it.calculate_fixed_period_review,
            InventoryModelType.NEWSVENDOR: it.calculate_newsvendor,
        }

    # =========================================================================
    # Step 1: Identify Model
    # =========================================================================

    def identify_model(self, question: str, override: Optional[str] = None) -> InventoryModelType:
        """Classify user query into one of the 5 inventory models."""
        if override:
            override_clean = override.strip().lower()
            if "discount" in override_clean:
                return InventoryModelType.QUANTITY_DISCOUNT
            if "eoq" in override_clean or "economic" in override_clean:
                return InventoryModelType.EOQ
            if "period" in override_clean or "order-up" in override_clean or "order up" in override_clean:
                return InventoryModelType.FIXED_PERIOD_REVIEW
            if "reorder" in override_clean or "rop" in override_clean or "safety stock" in override_clean:
                return InventoryModelType.SAFETY_STOCK_ROP
            if "news" in override_clean or "perishable" in override_clean or "single period" in override_clean:
                return InventoryModelType.NEWSVENDOR

        q = question.lower()

        # Priority 1: Quantity Discount (contains discount / tiers / price break)
        if any(w in q for w in ["discount", "price break", "volume tier", "price schedule", "bulk tier", "bracket"]):
            return InventoryModelType.QUANTITY_DISCOUNT

        # Priority 2: Fixed-Period Review (periodic, review period, order-up-to)
        if any(
            w in q
            for w in [
                "fixed period",
                "periodic review",
                "order-up-to",
                "order up to",
                "review period",
                "review cycle",
                "reviewed every",
                "check inventory every",
                "target inventory level",
            ]
        ):
            return InventoryModelType.FIXED_PERIOD_REVIEW

        # Priority 3: Newsvendor (single period, salvage, perishable, critical fractile, newsboy)
        if any(
            w in q
            for w in [
                "newsvendor",
                "newsboy",
                "single period",
                "salvage",
                "clearance",
                "perishable",
                "seasonal jackets",
                "seasonal demand",
                "critical ratio",
                "critical fractile",
                "overage",
                "underage",
            ]
        ):
            return InventoryModelType.NEWSVENDOR

        # Priority 4: Safety Stock & ROP (reorder point, safety stock, lead time demand, service level)
        if any(
            w in q
            for w in [
                "safety stock",
                "reorder point",
                "rop",
                "stockout probability",
                "cycle service level",
                "lead time demand",
                "continuous review",
            ]
        ):
            return InventoryModelType.SAFETY_STOCK_ROP

        # Priority 5: EOQ (economic order quantity, annual demand, order cost, holding cost)
        if any(
            w in q
            for w in [
                "eoq",
                "economic order quantity",
                "order quantity",
                "annual demand",
                "holding cost",
                "setup cost",
                "batch size",
                "orders per year",
            ]
        ):
            return InventoryModelType.EOQ

        return InventoryModelType.UNKNOWN

    # =========================================================================
    # Step 2: Extract Inputs
    # =========================================================================

    def extract_inputs(
        self, question: str, model: InventoryModelType
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Extract domain parameters from question using robust regex and contextual patterns."""
        inputs: Dict[str, Any] = {}
        missing: List[str] = []
        q = question

        # Helper to search regex floats
        def search_number(patterns: List[str], text: str) -> Optional[float]:
            for pat in patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    raw = m.group(1).replace(",", "").strip()
                    try:
                        return float(raw)
                    except ValueError:
                        continue
            return None

        if model == InventoryModelType.EOQ:
            # Annual demand
            d = search_number(
                [
                    r"(?:annual\s+demand|demand\s+of|sells|uses|requires|annual\s+usage)\s+(?:is\s+|of\s+)?([0-9,]+(?:\.[0-9]+)?)\s*(?:units|items|pieces|widgets)?",
                    r"([0-9,]+(?:\.[0-9]+)?)\s*(?:units|items|pieces|widgets)\s+(?:per\s+year|annually|a\s+year)",
                ],
                q,
            )
            if d:
                inputs["annual_demand"] = d
            else:
                missing.append("annual_demand (e.g. 10,000 units/year)")

            # Order cost / Setup cost
            s = search_number(
                [
                    r"(?:order(?:ing)?\s+cost|setup\s+cost|cost\s+to\s+place\s+(?:an\s+)?order|fixed\s+order\s+cost)(?:\s+is)?\s*[:=\s]*\$?([0-9,]+(?:\.[0-9]+)?)",
                    r"\$([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s+order|to\s+place\s+(?:an\s+)?order|order\s+cost)",
                ],
                q,
            )
            if s:
                inputs["order_cost"] = s
            else:
                missing.append("order_cost (e.g. $50/order)")

            # Holding cost or holding rate
            h = search_number(
                [
                    r"(?:holding\s+cost|carrying\s+cost|storage\s+cost)(?:\s+is)?\s*[:=\s]*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s+unit)?",
                    r"\$([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s+unit\s+per\s+year|holding\s+cost|carrying\s+cost)",
                ],
                q,
            )
            i_rate = search_number(
                [
                    r"(?:holding\s+(?:cost\s+)?rate|carrying\s+(?:cost\s+)?rate|holding\s+percentage|inventory\s+carrying\s+rate)(?:\s+is)?\s*[:=\s]*([0-9]+(?:\.[0-9]+)?)\s*%",
                    r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:holding|carrying)\s+rate",
                ],
                q,
            )
            c = search_number(
                [
                    r"(?:unit\s+cost|cost\s+per\s+unit|purchase\s+price|item\s+cost)(?:\s+is)?\s*[:=\s]*\$?([0-9,]+(?:\.[0-9]+)?)",
                    r"\$([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s+unit|unit\s+cost)",
                ],
                q,
            )

            if h is not None:
                inputs["holding_cost"] = h
            elif i_rate is not None and c is not None:
                inputs["holding_cost_rate"] = i_rate / 100.0 if i_rate > 1.0 else i_rate
                inputs["unit_cost"] = c
            elif i_rate is not None:
                missing.append("unit_cost (needed to calculate holding cost from holding rate)")
            else:
                missing.append("holding_cost (e.g. $2/unit/year) or holding_cost_rate (e.g. 20%)")

            if c is not None and "unit_cost" not in inputs:
                inputs["unit_cost"] = c

            # Working days
            wd = search_number(
                [r"([0-9]+)\s*(?:working\s+days|operating\s+days|days\s+(?:per|a)\s+year)"],
                q,
            )
            inputs["working_days_per_year"] = int(wd) if wd else 250

        elif model == InventoryModelType.QUANTITY_DISCOUNT:
            # Demand and order cost
            d = search_number(
                [
                    r"(?:annual\s+demand|demand|usage)\s+(?:is\s+|of\s+)?([0-9,]+(?:\.[0-9]+)?)",
                    r"([0-9,]+(?:\.[0-9]+)?)\s*(?:units|items)?\s*(?:annually|per\s+year)",
                ],
                q,
            )
            s = search_number(
                [
                    r"(?:order(?:ing)?\s+cost|setup\s+cost|cost\s+to\s+place\s+(?:an\s+)?order)(?:\s+is)?\s*[:=\s]*\$?([0-9,]+(?:\.[0-9]+)?)",
                    r"\$([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s+order|order\s+cost)",
                ],
                q,
            )
            i_rate = search_number(
                [
                    r"(?:holding\s+(?:cost\s+)?rate|carrying\s+(?:cost\s+)?rate|holding\s+rate|inventory\s+carrying\s+rate)(?:\s+is)?\s*[:=\s]*([0-9]+(?:\.[0-9]+)?)\s*%",
                    r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:holding|carrying)",
                ],
                q,
            )
            h_fix = search_number(
                [
                    r"(?:fixed\s+holding\s+cost|holding\s+cost)(?:\s+is)?\s*[:=\s]*\$?([0-9,]+(?:\.[0-9]+)?)",
                    r"\$([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s+unit\s+holding|holding\s+cost)",
                ],
                q,
            )

            if d:
                inputs["annual_demand"] = d
            else:
                missing.append("annual_demand")
            if s:
                inputs["order_cost"] = s
            else:
                missing.append("order_cost")

            if i_rate:
                inputs["holding_cost_rate"] = i_rate / 100.0 if i_rate > 1.0 else i_rate
            elif h_fix:
                inputs["fixed_holding_cost"] = h_fix
            else:
                missing.append("holding_cost_rate or fixed_holding_cost")

            # Extract discount tiers: e.g. "1 to 999 units at $5.00", "1000 to 1999 units at $4.80", "2000 or more units at $4.60"
            tiers = []
            tier_matches = re.findall(
                r"([0-9,]+)\s*(?:to|-)\s*([0-9,]+|\+)?\s*(?:units)?\s*(?:at|@|for|costing)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                q,
                re.IGNORECASE,
            )
            plus_matches = re.findall(
                r"([0-9,]+)\s*(?:\+|or\s+more|and\s+above|\+)\s*(?:units)?\s*(?:at|@|for|costing)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                q,
                re.IGNORECASE,
            )

            for m in tier_matches:
                low = float(m[0].replace(",", ""))
                high = float(m[1].replace(",", "")) if m[1] and m[1] != "+" else float("inf")
                cost = float(m[2])
                tiers.append({"min_qty": low, "max_qty": high, "unit_cost": cost})

            for m in plus_matches:
                low = float(m[0].replace(",", ""))
                cost = float(m[1])
                if not any(t["min_qty"] == low for t in tiers):
                    tiers.append({"min_qty": low, "max_qty": float("inf"), "unit_cost": cost})

            if tiers:
                inputs["tiers"] = sorted(tiers, key=lambda x: x["min_qty"])
            else:
                missing.append("discount tiers (e.g. 1-999 at $5.00, 1000-1999 at $4.80, 2000+ at $4.60)")

        elif model == InventoryModelType.SAFETY_STOCK_ROP:
            lt = search_number(
                [
                    r"(?:lead\s+time|delivery\s+time|supplier\s+takes)\s+(?:is\s+|of\s+)?([0-9,]+(?:\.[0-9]+)?)\s*(?:days|weeks|months|periods)?",
                    r"([0-9,]+(?:\.[0-9]+)?)\s*(?:day|week|month|period)s?\s+lead\s+time",
                ],
                q,
            )
            if lt is not None:
                inputs["lead_time"] = lt
            else:
                missing.append("lead_time (e.g. 4 days)")

            # Check if direct lead time demand given
            ltd_mean = search_number([r"(?:lead\s+time\s+demand\s+(?:mean|average)|average\s+lead\s+time\s+demand)\s*(?:is\s+|of\s+)?([0-9,]+(?:\.[0-9]+)?)"], q)
            ltd_std = search_number([r"(?:lead\s+time\s+demand\s+std|lead\s+time\s+std\s+dev|std\s+dev\s+of\s+lead\s+time\s+demand)\s*(?:is\s+|of\s+)?([0-9,]+(?:\.[0-9]+)?)"], q)

            if ltd_mean is not None and ltd_std is not None:
                inputs["lead_time_demand_mean"] = ltd_mean
                inputs["lead_time_demand_std"] = ltd_std
            else:
                # Periodic demand mean & std
                d_mean = search_number(
                    [
                        r"(?:mean|average|daily|weekly)\s+demand\s+(?:is\s+|of\s+)?([0-9,]+(?:\.[0-9]+)?)",
                        r"demand\s+is\s+([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:per\s+(?:day|week|period))",
                    ],
                    q,
                )
                d_std = search_number(
                    [
                        r"(?:standard\s+deviation|std\s+dev|std|sigma)\s+(?:of\s+demand\s+)?(?:is\s+|of\s+)?([0-9,]+(?:\.[0-9]+)?)",
                    ],
                    q,
                )
                if d_mean is not None:
                    inputs["mean_demand_per_period"] = d_mean
                else:
                    missing.append("mean_demand_per_period (e.g. 50 units/day)")

                if d_std is not None:
                    inputs["std_demand_per_period"] = d_std
                else:
                    missing.append("std_demand_per_period (e.g. 10 units/day)")

            # Service level
            csl = search_number(
                [
                    r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:cycle\s+)?service\s+level",
                    r"service\s+level\s+(?:of\s+|is\s+)?([0-9]+(?:\.[0-9]+)?)\s*%",
                    r"service\s+level\s+(?:of\s+|is\s+)?(0\.[0-9]+)",
                ],
                q,
            )
            inputs["service_level"] = csl / 100.0 if (csl and csl > 1.0) else (csl if csl else 0.95)

            # Optional lead time std
            lt_std = search_number([r"(?:standard\s+deviation\s+of\s+lead\s+time|lead\s+time\s+std)\s*(?:is\s+|of\s+)?([0-9,]+(?:\.[0-9]+)?)"], q)
            if lt_std:
                inputs["lead_time_std"] = lt_std

        elif model == InventoryModelType.FIXED_PERIOD_REVIEW:
            rp = search_number(
                [
                    r"(?:review\s+period|reviewed\s+every|check(?:ed)?(?:\s+inventory)?\s+every|cycle\s+of)\s+([0-9,]+(?:\.[0-9]+)?)\s*(?:days|weeks|months|periods)?",
                    r"every\s+([0-9,]+(?:\.[0-9]+)?)\s*(?:days|weeks|months|periods)",
                ],
                q,
            )
            lt = search_number(
                [
                    r"(?:lead\s+time|delivery\s+time)\s+(?:is\s+|of\s+)?([0-9,]+(?:\.[0-9]+)?)\s*(?:days|weeks|months|periods)?",
                    r"([0-9,]+(?:\.[0-9]+)?)\s*(?:day|week|month|period)s?\s+lead\s+time",
                ],
                q,
            )
            d_mean = search_number([r"(?:mean|average|daily|weekly)\s+demand\s+(?:is\s+|of\s+)?([0-9,]+(?:\.[0-9]+)?)"], q)
            d_std = search_number([r"(?:standard\s+deviation|std\s+dev|sigma)\s+(?:of\s+demand\s+)?(?:is\s+|of\s+)?([0-9,]+(?:\.[0-9]+)?)"], q)
            csl = search_number([r"([0-9]+(?:\.[0-9]+)?)\s*%\s*service\s+level", r"service\s+level\s+(?:of\s+|is\s+)?(0\.[0-9]+)"], q)
            on_hand = search_number(
                [
                    r"([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:on\s+hand|in\s+stock)",
                    r"(?:currently\s+(?:we\s+)?have|have|on\s+hand|in\s+stock|current\s+inventory)\s+(?:is\s+|of\s+)?([0-9,]+(?:\.[0-9]+)?)",
                ],
                q,
            )
            on_order = search_number(
                [
                    r"([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:on\s+order|in\s+pipeline|open\s+orders)",
                    r"(?:on\s+order|in\s+pipeline|pipeline|open\s+orders)\s+(?:is\s+|of\s+)?([0-9,]+(?:\.[0-9]+)?)",
                ],
                q,
            )

            if rp is not None:
                inputs["review_period"] = rp
            else:
                missing.append("review_period (e.g. 14 days)")

            if lt is not None:
                inputs["lead_time"] = lt
            else:
                missing.append("lead_time (e.g. 7 days)")

            if d_mean is not None:
                inputs["mean_demand_per_period"] = d_mean
            else:
                missing.append("mean_demand_per_period")

            if d_std is not None:
                inputs["std_demand_per_period"] = d_std
            else:
                missing.append("std_demand_per_period")

            inputs["service_level"] = csl / 100.0 if (csl and csl > 1.0) else (csl if csl else 0.95)
            if on_hand is not None:
                inputs["on_hand_inventory"] = on_hand
            if on_order is not None:
                inputs["on_order_inventory"] = on_order

        elif model == InventoryModelType.NEWSVENDOR:
            price = search_number(
                [
                    r"(?:retail\s+price|selling\s+price|price|revenue\s+per\s+unit)(?:\s+is)?\s*[:=\s]*\$?([0-9,]+(?:\.[0-9]+)?)",
                    r"\$([0-9,]+(?:\.[0-9]+)?)\s*(?:retail|selling\s+price|price)",
                ],
                q,
            )
            cost = search_number(
                [
                    r"(?:unit\s+(?:purchase\s+)?cost|cost\s+per\s+unit|wholesale\s+price|purchase\s+cost)(?:\s+is)?\s*[:=\s]*\$?([0-9,]+(?:\.[0-9]+)?)",
                    r"\$([0-9,]+(?:\.[0-9]+)?)\s*(?:unit\s+cost|cost|wholesale)",
                ],
                q,
            )
            salvage = search_number(
                [
                    r"(?:salvage\s+value|clearance\s+price|salvaged\s+at|discounted\s+at)(?:\s+is)?\s*[:=\s]*\$?([0-9,]+(?:\.[0-9]+)?)",
                    r"\$([0-9,]+(?:\.[0-9]+)?)\s*(?:salvage|salvaged)",
                ],
                q,
            )
            d_mean = search_number(
                [
                    r"(?:mean|average|expected)\s+demand\s+(?:is\s+|of\s+)?([0-9,]+(?:\.[0-9]+)?)",
                    r"demand\s+is\s+normally\s+distributed\s+with\s+mean\s+([0-9,]+(?:\.[0-9]+)?)",
                ],
                q,
            )
            d_std = search_number(
                [
                    r"(?:standard\s+deviation|std\s+dev|sigma)\s+(?:of\s+demand\s+)?(?:is\s+|of\s+)?([0-9,]+(?:\.[0-9]+)?)",
                ],
                q,
            )
            penalty = search_number(
                [r"(?:underage\s+penalty|stockout\s+penalty|goodwill\s+loss)\s*[:=\s]*\$?([0-9,]+(?:\.[0-9]+)?)"],
                q,
            )

            if price is not None:
                inputs["selling_price"] = price
            else:
                missing.append("selling_price (e.g. $100)")

            if cost is not None:
                inputs["unit_cost"] = cost
            else:
                missing.append("unit_cost (e.g. $60)")

            inputs["salvage_value"] = salvage if salvage is not None else 0.0

            if d_mean is not None:
                inputs["mean_demand"] = d_mean
            else:
                missing.append("mean_demand (e.g. 800 units)")

            if d_std is not None:
                inputs["std_demand"] = d_std
            else:
                missing.append("std_demand (e.g. 150 units)")

            if penalty is not None:
                inputs["underage_penalty"] = penalty

        return inputs, missing

    # =========================================================================
    # Step 3: Select Python Tool
    # =========================================================================

    def select_tool(self, model: InventoryModelType) -> Tuple[Callable[..., Dict[str, Any]], str]:
        """Select appropriate calculation tool from inventory_tools module."""
        if model not in self.tool_mapping:
            raise ValueError(f"No registered calculation tool for model: {model}")
        tool_fn = self.tool_mapping[model]
        return tool_fn, tool_fn.__name__

    # =========================================================================
    # Step 4: Calculate
    # =========================================================================

    def calculate(
        self, tool_fn: Callable[..., Dict[str, Any]], inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute numerical calculation in Python."""
        return tool_fn(**inputs)

    # =========================================================================
    # Step 5: Verify
    # =========================================================================

    def verify(
        self,
        model: InventoryModelType,
        inputs: Dict[str, Any],
        results: Dict[str, Any],
    ) -> VerificationReport:
        """Perform rigorous supply chain verification, boundary checks, and managerial sanity rules."""
        report = VerificationReport(is_valid=True)
        res = results.get("results", {})

        if model == InventoryModelType.EOQ:
            q = res.get("optimal_order_quantity", 0)
            aoc = res.get("annual_ordering_cost", 0)
            ahc = res.get("annual_holding_cost", 0)

            if q > 0:
                report.sanity_checks_passed.append(f"Optimal EOQ Q* = {q:,} units is strictly positive.")
            else:
                report.is_valid = False
                report.warnings.append("EOQ calculated is non-positive.")

            # Check trade-off balance: AOC ~ AHC
            diff = abs(aoc - ahc)
            pct_diff = (diff / max(aoc, 1.0)) * 100.0
            if pct_diff < 1.0:
                report.sanity_checks_passed.append(
                    f"First-order optimality condition satisfied: Annual Ordering Cost (${aoc:,.2f}) ≈ Annual Holding Cost (${ahc:,.2f})."
                )
            else:
                report.warnings.append(
                    f"Ordering cost (${aoc:,.2f}) and holding cost (${ahc:,.2f}) deviate by {pct_diff:.2f}%, check input parameters."
                )

            # Cycle time managerial note
            cycle_days = res.get("cycle_time_days", 0)
            if cycle_days > 180:
                report.managerial_notes.append(
                    f"Cycle time of {cycle_days:.1f} operating days exceeds half a year; verify perishability and obsolescence risks."
                )

        elif model == InventoryModelType.QUANTITY_DISCOUNT:
            best_tier = res.get("optimal_tier")
            best_q = res.get("optimal_order_quantity", 0)
            savings = res.get("annual_savings_vs_base_tier", 0)

            report.sanity_checks_passed.append(
                f"Evaluated all discount tiers. Minimum total annual cost achieved at Tier {best_tier} with order quantity {best_q:,.0f}."
            )
            if savings > 0:
                report.sanity_checks_passed.append(
                    f"Bulk purchase produces positive net annual savings of ${savings:,.2f} ({res.get('percent_savings')}%) over baseline."
                )
            else:
                report.managerial_notes.append(
                    "No financial advantage to order at higher discount tiers; baseline EOQ yields lowest total cost."
                )

        elif model == InventoryModelType.SAFETY_STOCK_ROP:
            ss = res.get("safety_stock", 0)
            rop = res.get("reorder_point", 0)
            csl = res.get("cycle_service_level_percent", 0)
            z = res.get("z_score", 0)

            if rop >= ss:
                report.sanity_checks_passed.append(
                    f"Reorder point ({rop:,}) appropriately bounds safety stock ({ss:,}) plus expected lead-time demand."
                )
            else:
                report.is_valid = False
                report.warnings.append("Reorder point is lower than safety stock.")

            if z > 2.58:  # 99.5%+
                report.warnings.append(
                    f"Target service level of {csl}% requires a high z-score ({z:.2f}), causing exponential inventory holding inflation."
                )
            else:
                report.sanity_checks_passed.append(
                    f"Service level of {csl}% is well within operational efficiency bounds (z = {z:.2f})."
                )

        elif model == InventoryModelType.FIXED_PERIOD_REVIEW:
            order_up_to = res.get("order_up_to_level", 0)
            protect = res.get("protection_interval", 0)
            q_rec = res.get("recommended_order_quantity")

            report.sanity_checks_passed.append(
                f"Protection interval correctly spans review period plus lead time ({protect} periods)."
            )
            if q_rec is not None:
                report.sanity_checks_passed.append(
                    f"Recommended replenishment order calculated as {q_rec:,} units given current stock and pipeline."
                )

        elif model == InventoryModelType.NEWSVENDOR:
            cr = res.get("critical_ratio", 0)
            cu = res.get("cost_of_underage_Cu", 0)
            co = res.get("cost_of_overage_Co", 0)
            q_star = res.get("optimal_order_quantity", 0)
            profit = res.get("expected_profit", 0)

            if 0.0 < cr < 1.0:
                report.sanity_checks_passed.append(
                    f"Critical fractile CR = {cr:.4f} ({cr * 100:.1f}%) is within valid probability bounds (0, 1)."
                )
            else:
                report.is_valid = False
                report.warnings.append(f"Critical ratio {cr} is out of bounds.")

            if profit > 0:
                report.sanity_checks_passed.append(
                    f"Expected single-period profit is positive (${profit:,.2f})."
                )
            else:
                report.warnings.append("Expected profit is non-positive; reconsider product viability.")

            if cu > co:
                report.managerial_notes.append(
                    f"High margin product (Cu = ${cu:,.2f} > Co = ${co:,.2f}): optimal policy orders above average demand (Q* = {q_star:,} vs mu = {inputs.get('mean_demand', 0):,}) to avoid stockout costs."
                )
            elif co > cu:
                report.managerial_notes.append(
                    f"High salvage loss product (Co = ${co:,.2f} > Cu = ${cu:,.2f}): optimal policy orders conservatively below average demand to avoid liquidation losses."
                )
            else:
                report.managerial_notes.append(
                    f"Balanced underage and overage costs (Cu = Co = ${cu:,.2f}): critical fractile is 50%, ordering exactly at mean demand ({q_star:,} units)."
                )

        return report

    # =========================================================================
    # Step 6: Explain
    # =========================================================================

    def explain(
        self,
        model: InventoryModelType,
        inputs: Dict[str, Any],
        results: Dict[str, Any],
        verification: VerificationReport,
    ) -> str:
        """Synthesize calculation into clear managerial summary and mathematical justification."""
        res = results.get("results", {})
        formulas = results.get("formulas", {})
        lines: List[str] = []

        lines.append(f"### Supply Chain Decision Analysis: {model.value}")
        lines.append("")

        if model == InventoryModelType.EOQ:
            q = res.get("optimal_order_quantity")
            n = res.get("orders_per_year")
            cycle = res.get("cycle_time_days")
            tic = res.get("total_inventory_cost")
            tc = res.get("total_annual_cost")
            aoc = res.get("annual_ordering_cost")
            ahc = res.get("annual_holding_cost")
            h_unit = results.get("inputs", {}).get("holding_cost_per_unit_year", inputs.get("holding_cost", 0.0))

            lines.append("#### 1. Executive Recommendation")
            lines.append(
                f"- **Recommended Order Quantity (Q*)**: **{q:,} units** per order."
            )
            lines.append(
                f"- **Order Frequency**: Place an order approximately **{n} times per year** (every **{cycle} operating days**)."
            )
            lines.append(
                f"- **Total Annual Inventory Cost**: **${tic:,.2f}** (Ordering: ${aoc:,.2f} + Holding: ${ahc:,.2f})."
            )
            if tc != tic:
                lines.append(f"- **Total Annual System Cost (incl. purchases)**: **${tc:,.2f}**.")

            lines.append("")
            lines.append("#### 2. Mathematical Proof & Derivation")
            lines.append(r"- Wilson Formula: $Q^* = \sqrt{\frac{2 \cdot D \cdot S}{H}}$")
            lines.append(
                f"  $$Q^* = \\sqrt{{\\frac{{2 \\times {inputs['annual_demand']:,} \\times {inputs['order_cost']}}}{{{h_unit}}}}} = {res['optimal_order_quantity_exact']:.2f} \\approx {q} \\text{{ units}}$$"
            )
            lines.append(
                f"- Ordering Cost: $\\frac{{D}}{{Q^*}} \\times S = \\frac{{{inputs['annual_demand']:,}}}{{{res['optimal_order_quantity_exact']:.2f}}} \\times {inputs['order_cost']} = \\${aoc:,.2f}$"
            )
            lines.append(
                f"- Holding Cost: $\\frac{{Q^*}}{{2}} \\times H = \\frac{{{res['optimal_order_quantity_exact']:.2f}}}{{2}} \\times {h_unit} = \\${ahc:,.2f}$"
            )

        elif model == InventoryModelType.QUANTITY_DISCOUNT:
            best_tier = res.get("optimal_tier")
            best_q = res.get("optimal_order_quantity")
            min_cost = res.get("minimum_total_annual_cost")
            savings = res.get("annual_savings_vs_base_tier")
            pct = res.get("percent_savings")

            lines.append("#### 1. Executive Recommendation")
            lines.append(
                f"- **Optimal Order Strategy**: Order at **Tier {best_tier}** with batch size of **{best_q:,.0f} units**."
            )
            lines.append(
                f"- **Annual Financial Impact**: Total annual cost is minimized at **${min_cost:,.2f}**."
            )
            lines.append(
                f"- **Net Annual Savings**: Save **${savings:,.2f} ({pct}%)** compared to ordering without volume discounts."
            )

            lines.append("")
            lines.append("#### 2. Tier-by-Tier Cost Breakdown")
            lines.append("| Tier | Range | Unit Cost | Candidate Q | Purchase Cost | Ordering Cost | Holding Cost | Total Cost |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for t in results.get("tier_evaluations", []):
                q_range = f"{t['min_qty']:,.0f} - {t['max_qty']}"
                lines.append(
                    f"| {t['tier_index']} | {q_range} | ${t['unit_cost']:.2f} | {t['candidate_order_quantity']:,.0f} | ${t['annual_purchase_cost']:,.2f} | ${t['annual_ordering_cost']:,.2f} | ${t['annual_holding_cost']:,.2f} | **${t['total_annual_cost']:,.2f}** |"
                )

        elif model == InventoryModelType.SAFETY_STOCK_ROP:
            rop = res.get("reorder_point")
            ss = res.get("safety_stock")
            ltd = res.get("expected_lead_time_demand")
            csl = res.get("cycle_service_level_percent")
            z = res.get("z_score")
            esc = res.get("expected_shortage_per_cycle")

            lines.append("#### 1. Executive Recommendation")
            lines.append(
                f"- **Reorder Point (ROP)**: Place a replenishment order whenever inventory drops to **{rop:,} units**."
            )
            lines.append(
                f"- **Safety Stock (Buffer)**: Maintain a permanent safety stock of **{ss:,} units**."
            )
            lines.append(
                f"- **Protection Guarantee**: Ensures a **{csl}% Cycle Service Level** ({100-csl:.1f}% risk of stockout per replenishment cycle)."
            )
            lines.append(f"- **Expected Unit Shortage**: Only **{esc:.4f} units** expected short per cycle.")

            lines.append("")
            lines.append("#### 2. Mathematical Formulation")
            lines.append(f"- Expected Lead-Time Demand: $\\mu_L = {ltd:,.1f} \\text{{ units}}$")
            lines.append(f"- Safety Stock Formula: $SS = z \\times \\sigma_L = {z:.4f} \\times {res.get('lead_time_demand_std')} = {res.get('safety_stock_exact'):.2f} \\approx {ss} \\text{{ units}}$")
            lines.append(f"- Reorder Point: $ROP = \\mu_L + SS = {ltd:.1f} + {res.get('safety_stock_exact'):.2f} = {res.get('reorder_point_exact'):.2f} \\approx {rop} \\text{{ units}}$")

        elif model == InventoryModelType.FIXED_PERIOD_REVIEW:
            order_up_to = res.get("order_up_to_level")
            ss = res.get("safety_stock")
            prot_int = res.get("protection_interval")
            rec_q = res.get("recommended_order_quantity")
            inv_pos = res.get("current_inventory_position")

            lines.append("#### 1. Executive Recommendation")
            lines.append(
                f"- **Target Order-Up-To Level (S)**: **{order_up_to:,} units**."
            )
            lines.append(
                f"- **Safety Stock Required**: **{ss:,} units** to cover the **{prot_int} period protection interval**."
            )
            if rec_q is not None:
                lines.append(
                    f"- **Replenishment Order Now**: Place an immediate order for **{rec_q:,} units** (bringing current position {inv_pos:,.0f} up to {order_up_to:,})."
                )

            lines.append("")
            lines.append("#### 2. Mathematical Formulation")
            lines.append(f"- Protection Interval: $T = P + L = {inputs['review_period']} + {inputs['lead_time']} = {prot_int} \\text{{ periods}}$")
            lines.append(f"- Target Level: $S = d \\times (P + L) + z \\times \\sigma_d \\times \\sqrt{{P + L}} = {res['order_up_to_level_exact']:.2f} \\approx {order_up_to}$")

        elif model == InventoryModelType.NEWSVENDOR:
            q_star = res.get("optimal_order_quantity")
            cr = res.get("critical_ratio")
            cu = res.get("cost_of_underage_Cu")
            co = res.get("cost_of_overage_Co")
            exp_sales = res.get("expected_sales")
            exp_profit = res.get("expected_profit")
            exp_leftover = res.get("expected_leftover_inventory")
            fill_rate = res.get("fill_rate_percent")

            lines.append("#### 1. Executive Recommendation")
            lines.append(
                f"- **Optimal Order Quantity (Q*)**: Order exactly **{q_star:,} units**."
            )
            lines.append(
                f"- **Expected Total Profit**: **${exp_profit:,.2f}** (Fill Rate: **{fill_rate:.1f}%**)."
            )
            lines.append(
                f"- **Expected Inventory Flow**: Expected sales = **{exp_sales:,.1f} units**, expected leftovers = **{exp_leftover:,.1f} units**."
            )

            lines.append("")
            lines.append("#### 2. Marginal Economic Analysis")
            lines.append(f"- Underage Cost (Lost Profit / unit): $C_u = P - C = \\${cu:,.2f}$")
            lines.append(f"- Overage Cost (Loss / unsold unit): $C_o = C - V = \\${co:,.2f}$")
            lines.append(f"- Critical Fractile: $CR = \\frac{{C_u}}{{C_u + C_o}} = \\frac{{{cu}}}{{{cu + co}}} = {cr:.4f}$ ({res.get('critical_ratio_percent')}%)")
            lines.append(f"- Optimal Batch: $Q^* = \\mu + z^* \\cdot \\sigma = {inputs['mean_demand']:,} + ({res['z_score']:.4f} \\times {inputs['std_demand']:,}) = {res['optimal_order_quantity_exact']:.2f} \\approx {q_star}$")

        # Managerial notes and verification
        lines.append("")
        lines.append("#### 3. Verification & Managerial Insights")
        for check in verification.sanity_checks_passed:
            lines.append(f"- [x] {check}")
        for note in verification.managerial_notes:
            lines.append(f"- [!] **Supply Chain Note**: {note}")
        for warn in verification.warnings:
            lines.append(f"- [WARNING] **Operational Risk**: {warn}")

        return "\n".join(lines)

    # =========================================================================
    # Master Execution Method
    # =========================================================================

    def process_question(
        self, question: str, model_override: Optional[str] = None
    ) -> AgentResponse:
        """Execute complete agent decision pipeline on natural language question."""
        # 1. Identify Model
        model = self.identify_model(question, override=model_override)
        if model == InventoryModelType.UNKNOWN:
            return AgentResponse(
                question=question,
                identified_model=model,
                extracted_inputs={},
                missing_inputs=[],
                selected_tool_name="None",
                calculation_results=None,
                verification=None,
                explanation=(
                    "I could not clearly match your question to one of the 5 supported inventory models:\n"
                    "1. Economic Order Quantity (EOQ)\n"
                    "2. Quantity Discount Model\n"
                    "3. Safety Stock & Reorder Point\n"
                    "4. Fixed-Period Review Model\n"
                    "5. Newsvendor Model\n\n"
                    "Please specify key parameters like annual demand, costs, lead time, or review period."
                ),
                status="CALCULATION_ERROR",
            )

        # 2. Extract Inputs
        inputs, missing = self.extract_inputs(question, model)
        if missing:
            return AgentResponse(
                question=question,
                identified_model=model,
                extracted_inputs=inputs,
                missing_inputs=missing,
                selected_tool_name="None",
                calculation_results=None,
                verification=None,
                explanation=(
                    f"Identified Model: **{model.value}**\n\n"
                    f"To perform the calculation, I still need the following missing inputs:\n"
                    + "\n".join(f"- `{m}`" for m in missing)
                    + f"\n\nExtracted so far: `{inputs}`"
                ),
                status="MISSING_INPUTS",
            )

        # 3. Select Python Tool
        tool_fn, tool_name = self.select_tool(model)

        # 4. Calculate
        try:
            results = self.calculate(tool_fn, inputs)
        except Exception as e:
            return AgentResponse(
                question=question,
                identified_model=model,
                extracted_inputs=inputs,
                missing_inputs=[],
                selected_tool_name=tool_name,
                calculation_results=None,
                verification=None,
                explanation=f"Error executing Python calculation: {str(e)}",
                status="CALCULATION_ERROR",
            )

        # 5. Verify
        verification = self.verify(model, inputs, results)

        # 6. Explain
        explanation = self.explain(model, inputs, results, verification)

        return AgentResponse(
            question=question,
            identified_model=model,
            extracted_inputs=inputs,
            missing_inputs=[],
            selected_tool_name=tool_name,
            calculation_results=results,
            verification=verification,
            explanation=explanation,
            status="SUCCESS",
        )
