
import pandas as pd

from src.decision_intelligence_pipeline import (
    DecisionIntelligencePipeline,
)


def build_test_data():

    months = pd.date_range(
        "2025-04-01",
        "2026-03-01",
        freq="MS",
    )

    budget_rows = []
    actual_rows = []

    for i, month in enumerate(
        months,
        start=1,
    ):

        fy = "FY25-26"

        for bu in [
            "Consumer",
            "Enterprise",
        ]:

            for dept, category, base in [
                (
                    "Marketing",
                    "Digital Advertising",
                    100000,
                ),
                (
                    "Operations",
                    "Utilities",
                    80000,
                ),
            ]:

                budget_amount = (
                    base + i * 1000
                )

                actual_multiplier = (
                    1.15
                    if category
                    == "Digital Advertising"
                    else 0.96
                )

                actual_amount = (
                    budget_amount
                    * actual_multiplier
                )

                budget_rows.append(
                    {
                        "Month": month,
                        "Year": month.year,
                        "Month_Number": month.month,
                        "Fiscal_Year": fy,
                        "Business_Unit": bu,
                        "Department": dept,
                        "Category": category,
                        "Budget_Amount": budget_amount,
                    }
                )

                actual_rows.append(
                    {
                        "Transaction_ID":
                            f"T-{len(actual_rows)+1:05d}",
                        "Transaction_Date": month,
                        "Month": month,
                        "Business_Unit": bu,
                        "Department": dept,
                        "Category": category,
                        "Region": "North",
                        "Cost_Centre": "CC-1",
                        "Vendor": "Test Vendor",
                        "Actual_Amount":
                            actual_amount,
                        "Status": "Approved",
                    }
                )

    return (
        pd.DataFrame(budget_rows),
        pd.DataFrame(actual_rows),
    )


def main():

    print("\n" + "=" * 80)
    print("VARIA FULL PIPELINE TEST")
    print("=" * 80)

    budget, actuals = build_test_data()

    pipeline = DecisionIntelligencePipeline(
        budget=budget,
        actuals=actuals,
    )

    outputs = pipeline.run()

    required_outputs = [
        "actuals_cleaned",
        "planning_reconciliation",
        "matched_actuals",
        "planning_exceptions",
        "planning_summary",
        "variance",
        "materiality",
        "root_cause",
        "forecast",
        "warning",
        "management",
        "cfo_queue",
        "ai_output",
        "anomaly",
        "advanced_forecast",
        "risk",
        "recommendations",
        "base_scenario",
        "scenario_summary",
        "cfo_report",
        "audit",
    ]

    for output in required_outputs:
        assert output in outputs

    assert len(outputs["planning_reconciliation"]) == len(actuals)
    assert len(outputs["planning_exceptions"]) == 0
    assert outputs["planning_summary"]["Exception Transactions"] == 0
    assert len(outputs["matched_actuals"]) == len(actuals)

    assert len(outputs["variance"]) > 0
    assert len(outputs["forecast"]) > 0
    assert len(outputs["risk"]) > 0
    assert len(outputs["recommendations"]) > 0
    assert len(outputs["audit"]) >= 3

    assert (
        "VARIA CFO Brief"
        in outputs["cfo_report"]
    )

    print("\nPIPELINE OUTPUTS")
    for output in required_outputs:
        obj = outputs[output]
        shape = (
            obj.shape
            if hasattr(obj, "shape")
            else type(obj).__name__
        )
        print(
            f"{output:25s} {shape}"
        )

    print("\nAUDIT SUMMARY")
    print(
        outputs["audit_summary"]
    )

    print("\n" + "=" * 80)
    print("FULL PIPELINE TEST: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
