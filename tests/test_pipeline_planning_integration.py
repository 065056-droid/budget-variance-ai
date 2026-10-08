import pandas as pd

from src.data_loader import load_workbook

from src.decision_intelligence_pipeline import (
    DecisionIntelligencePipeline,
)


def main() -> None:

    print("\n" + "=" * 80)
    print("VARIA PIPELINE + PLANNING RECONCILIATION INTEGRATION TEST")
    print("=" * 80)

    data = load_workbook()

    budget = data["Budget"]
    actuals = data["Actuals_Raw"]

    pipeline = DecisionIntelligencePipeline(
        budget=budget,
        actuals=actuals,
    )

    outputs = pipeline.run()

    planning_summary = outputs[
        "planning_summary"
    ]

    exceptions = outputs[
        "planning_exceptions"
    ]

    variance = outputs["variance"]

    assert planning_summary[
        "Total Transactions"
    ] == 6685

    assert planning_summary[
        "Matched Transactions"
    ] == 6683

    assert planning_summary[
        "Exception Transactions"
    ] == 2

    assert abs(
        planning_summary["Matched Amount"]
        - 787365053.22
    ) < 0.01

    assert abs(
        planning_summary["Exception Amount"]
        - 60569.56
    ) < 0.01

    exception_map = exceptions.set_index(
        "Transaction_ID"
    )[
        "Planning_Exception_Type"
    ].to_dict()

    assert exception_map == {
        "TXN-0000011": "MISSING_DIMENSION",
        "TXN-0000051": "UNMAPPED_BUDGET_LINE",
    }

    variance_actual = float(
        pd.to_numeric(
            variance["Actual_Amount"],
            errors="coerce",
        ).sum()
    )

    assert abs(
        variance_actual
        - planning_summary["Matched Amount"]
    ) < 0.01

    # The two planning exceptions must not disappear silently into the
    # variance layer. Their signed amount remains outside variance totals
    # until management resolves them.
    assert abs(
        variance_actual
        - planning_summary["Total Actual Amount"]
    ) > 1000.0

    audit = outputs["audit"]
    planning_audit = audit.loc[
        audit["Stage"] == "Planning Reconciliation"
    ]

    assert len(planning_audit) == 1
    assert planning_audit.iloc[0]["Status"] == "REVIEW"

    manifest = outputs[
        "governance_manifest"
    ]

    assert manifest[
        "Final_Status"
    ] == "REVIEW"

    governance_planning = [
        stage
        for stage in manifest["Stages"]
        if stage["Stage"] == "Planning Reconciliation"
    ]

    assert len(governance_planning) == 1
    assert governance_planning[0][
        "Status"
    ] == "REVIEW"

    print("\nPLANNING SUMMARY")
    for key, value in planning_summary.items():
        print(f"{key:28s} {value}")

    print("\nVARIANCE CONTROL")
    print(
        f"Variance actual total: ₹{variance_actual:,.2f}"
    )
    print(
        "Planning exception exposure: "
        f"₹{planning_summary['Gross Exception Exposure']:,.2f}"
    )
    print(
        "Governance final status: "
        f"{manifest['Final_Status']}"
    )

    print("\n" + "=" * 80)
    print("PIPELINE PLANNING INTEGRATION TEST: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
