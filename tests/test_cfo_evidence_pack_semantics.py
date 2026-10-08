from __future__ import annotations

import pandas as pd

from src.cfo_evidence_pack_engine import CFOEvidencePackEngine


def test_cfo_pack_attaches_financial_semantics():
    variance = pd.DataFrame(
        [
            {
                "Business_Unit": "Consumer",
                "Department": "Marketing",
                "Category": "Digital Advertising",
                "Budget_Amount": 1_000_000,
                "Actual_Amount": 1_200_000,
                "Variance": 200_000,
            },
            {
                "Business_Unit": "Enterprise",
                "Department": "Procurement",
                "Category": "Vendor Services",
                "Budget_Amount": 500_000,
                "Actual_Amount": 450_000,
                "Variance": -50_000,
            },
        ]
    )
    materiality = pd.DataFrame({"Is_Material": [True, False]})
    forecast = pd.DataFrame(
        [
            {
                "Business_Unit": "Consumer",
                "Department": "Marketing",
                "Category": "Digital Advertising",
                "Projected_Variance": 250_000,
                "Projected_Variance_Pct": 12.5,
                "Early_Warning": "RED",
            }
        ]
    )
    risk = pd.DataFrame(
        [
            {
                "Business_Unit": "Consumer",
                "Department": "Marketing",
                "Category": "Digital Advertising",
                "Risk_Score": 90,
                "Risk_Band": "HIGH",
            }
        ]
    )
    recommendations = pd.DataFrame(
        [
            {
                "Business_Unit": "Consumer",
                "Department": "Marketing",
                "Category": "Digital Advertising",
                "Management_Recommendation": "Investigate spend",
            }
        ]
    )
    action_sizing = pd.DataFrame(
        [
            {
                "Business_Unit": "Consumer",
                "Department": "Marketing",
                "Category": "Digital Advertising",
                "Required_Cost_Control": 100_000,
            }
        ]
    )

    pack = CFOEvidencePackEngine(
        outputs={
            "variance": variance,
            "materiality": materiality,
            "forecast": forecast,
            "warning": forecast,
            "risk": risk,
            "recommendations": recommendations,
            "action_sizing": action_sizing,
            "planning_summary": {
                "Exception Transactions": 2,
                "Exception Amount": 60_569.56,
                "Exception Types": {
                    "MISSING_DIMENSION": 1,
                    "UNMAPPED_BUDGET_LINE": 1,
                },
            },
        }
    ).build()

    by_id = {item["id"]: item for item in pack["evidence_ledger"]}

    assert by_id["EXEC-003"]["direction"] == "UNFAVORABLE"
    assert by_id["DRV-001-B"]["semantic_type"] == "UNFAVORABLE_OVERSPEND"
    assert by_id["ISS-001-B"]["direction"] == "UNFAVORABLE"
    assert by_id["PLAN-002"]["semantic_type"] == "PLANNING_EXCEPTION_AMOUNT"
    assert by_id["PLAN-002"]["direction"] == "NOT_APPLICABLE"
    assert pack["pack_version"] == "2.0"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_projected_record_counts_carry_direction_semantics():
    # This regression protects the semantic contract used by the LLM validator:
    # overspend/underspend record counts are directional facts, not neutral counts.
    outputs = {
        "variance": pd.DataFrame({"Category": ["A"], "Variance": [100.0]}),
        "scenario_ranking": pd.DataFrame({
            "Scenario_Name": ["Base Scenario"],
            "Base_Projected_Variance": [100.0],
            "Scenario_Projected_Variance": [100.0],
            "Variance_Improvement": [0.0],
            "Forecast_Savings": [0.0],
            "Budget_Headroom_Change": [0.0],
            "Projected_Overspend_Records": [21],
            "Projected_Underspend_Records": [54],
            "On_Budget_Records": [0],
            "Outcome_Rank": [1],
        }),
        "planning_summary": {},
    }
    pack = CFOEvidencePackEngine(outputs=outputs).build()
    ledger = {x["id"]: x for x in pack["evidence_ledger"]}
    assert ledger["SCN-001-Projected_Overspend_Records"]["direction"] == "UNFAVORABLE"
    assert ledger["SCN-001-Projected_Underspend_Records"]["direction"] == "FAVORABLE"
