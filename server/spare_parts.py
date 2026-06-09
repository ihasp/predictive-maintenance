from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


FailureMode = Literal["TWF", "HDF", "PWF", "OSF", "RNF"]
InventoryAction = Literal[
    "NO_ACTION",
    "MONITOR",
    "RESERVE_FROM_STOCK",
    "REORDER",
    "URGENT_ORDER",
]


@dataclass(frozen=True)
class SparePart:
    failure_mode: FailureMode
    part_id: str
    name: str
    category: str
    supplier: str
    stock_level: int
    reorder_point: int
    reorder_quantity: int
    lead_time_days: int
    unit_cost_pln: float
    downtime_cost_per_hour_pln: float
    estimated_downtime_hours: float


SPARE_PART_CATALOG: dict[FailureMode, SparePart] = {
    "TWF": SparePart(
        failure_mode="TWF",
        part_id="TOOL-INSERT-KIT",
        name="Zestaw narzedzi roboczych",
        category="Narzedzia eksploatacyjne",
        supplier="Dostawca narzedzi CNC",
        stock_level=3,
        reorder_point=4,
        reorder_quantity=8,
        lead_time_days=2,
        unit_cost_pln=420.0,
        downtime_cost_per_hour_pln=1200.0,
        estimated_downtime_hours=2.5,
    ),
    "HDF": SparePart(
        failure_mode="HDF",
        part_id="COOLING-PUMP-KIT",
        name="Zestaw ukladu chlodzenia",
        category="Chlodzenie i wymiana ciepla",
        supplier="Serwis automatyki",
        stock_level=1,
        reorder_point=2,
        reorder_quantity=3,
        lead_time_days=5,
        unit_cost_pln=1850.0,
        downtime_cost_per_hour_pln=1600.0,
        estimated_downtime_hours=4.0,
    ),
    "PWF": SparePart(
        failure_mode="PWF",
        part_id="DRIVE-BELT-POWER",
        name="Elementy ukladu napedu i zasilania",
        category="Naped i zasilanie",
        supplier="Hurtownia techniczna",
        stock_level=0,
        reorder_point=1,
        reorder_quantity=2,
        lead_time_days=7,
        unit_cost_pln=2600.0,
        downtime_cost_per_hour_pln=2200.0,
        estimated_downtime_hours=6.0,
    ),
    "OSF": SparePart(
        failure_mode="OSF",
        part_id="BEARING-STRUCT-KIT",
        name="Lozyska i elementy konstrukcyjne",
        category="Mechanika",
        supplier="Dostawca lozysk",
        stock_level=2,
        reorder_point=2,
        reorder_quantity=4,
        lead_time_days=4,
        unit_cost_pln=980.0,
        downtime_cost_per_hour_pln=1800.0,
        estimated_downtime_hours=5.0,
    ),
    "RNF": SparePart(
        failure_mode="RNF",
        part_id="INSPECTION-KIT",
        name="Pakiet diagnostyczny awarii losowych",
        category="Diagnostyka",
        supplier="Magazyn UR",
        stock_level=5,
        reorder_point=2,
        reorder_quantity=5,
        lead_time_days=1,
        unit_cost_pln=250.0,
        downtime_cost_per_hour_pln=1000.0,
        estimated_downtime_hours=1.5,
    ),
}


def optimize_inventory(
    risks: dict[FailureMode, float],
    high_risk_threshold: float = 0.50,
    medium_risk_threshold: float = 0.25,
) -> dict:
    decisions = [
        build_part_decision(
            failure_mode=failure_mode,
            risk=float(risk),
            high_risk_threshold=high_risk_threshold,
            medium_risk_threshold=medium_risk_threshold,
        )
        for failure_mode, risk in risks.items()
    ]

    decisions.sort(
        key=lambda decision: (
            decision["priority_score"],
            decision["expected_downtime_cost_pln"],
        ),
        reverse=True,
    )

    return {
        "overall_action": overall_action(decisions),
        "total_expected_downtime_cost_pln": round(
            sum(decision["expected_downtime_cost_pln"] for decision in decisions), 2
        ),
        "total_recommended_order_value_pln": round(
            sum(decision["recommended_order_value_pln"] for decision in decisions), 2
        ),
        "decisions": decisions,
    }


def build_part_decision(
    failure_mode: FailureMode,
    risk: float,
    high_risk_threshold: float,
    medium_risk_threshold: float,
) -> dict:
    part = SPARE_PART_CATALOG[failure_mode]
    expected_downtime_cost = (
        risk * part.estimated_downtime_hours * part.downtime_cost_per_hour_pln
    )

    action: InventoryAction = "NO_ACTION"
    reason = "Ryzyko ponizej progu, brak potrzeby rezerwacji czesci."
    priority_score = risk
    recommended_order_quantity = 0

    if risk >= high_risk_threshold and part.stock_level > 0:
        action = "RESERVE_FROM_STOCK"
        reason = "Wysokie ryzyko awarii, czesc dostepna w magazynie."
        priority_score += 2.0
    elif risk >= high_risk_threshold and part.stock_level == 0:
        action = "URGENT_ORDER"
        reason = "Wysokie ryzyko awarii i brak czesci na stanie."
        priority_score += 3.0
        recommended_order_quantity = part.reorder_quantity
    elif risk >= medium_risk_threshold or part.stock_level <= part.reorder_point:
        action = "REORDER"
        reason = "Ryzyko lub stan magazynu uzasadnia uzupelnienie zapasu."
        priority_score += 1.0
        recommended_order_quantity = max(
            part.reorder_quantity, part.reorder_point + 1 - part.stock_level
        )
    elif risk > 0.10:
        action = "MONITOR"
        reason = "Ryzyko umiarkowane, obserwowac trend parametrow."
        priority_score += 0.25

    return {
        "failure_mode": failure_mode,
        "risk": round(risk, 3),
        "action": action,
        "reason": reason,
        "priority": priority_label(priority_score),
        "priority_score": round(priority_score, 3),
        "part": asdict(part),
        "recommended_order_quantity": recommended_order_quantity,
        "recommended_order_value_pln": round(
            recommended_order_quantity * part.unit_cost_pln, 2
        ),
        "expected_downtime_cost_pln": round(expected_downtime_cost, 2),
    }


def overall_action(decisions: list[dict]) -> InventoryAction:
    actions = {decision["action"] for decision in decisions}
    if "URGENT_ORDER" in actions:
        return "URGENT_ORDER"
    if "RESERVE_FROM_STOCK" in actions:
        return "RESERVE_FROM_STOCK"
    if "REORDER" in actions:
        return "REORDER"
    if "MONITOR" in actions:
        return "MONITOR"
    return "NO_ACTION"


def priority_label(priority_score: float) -> str:
    if priority_score >= 3.0:
        return "KRYTYCZNY"
    if priority_score >= 2.0:
        return "WYSOKI"
    if priority_score >= 1.0:
        return "SREDNI"
    return "NISKI"
