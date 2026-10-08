from src.data_loader import load_workbook
from src.data_cleaner import DataCleaner
from src.planning_reconciliation import PlanningReconciliation
from src.variance_engine import VarianceEngine
from src.variance_interpretation import VarianceInterpretation


def main():

    # =========================================================
    # 1. LOAD DATA
    # =========================================================

    data = load_workbook()

    budget = data["Budget"]
    raw_actuals = data["Actuals_Raw"]

    # =========================================================
    # 2. CLEAN ACTUALS
    # =========================================================

    cleaner = DataCleaner(
        actuals=raw_actuals,
        category_mapping={
            "Digital Ads": "Digital Advertising"
        }
    )

    cleaned_actuals = cleaner.run()

    # =========================================================
    # 3. PLANNING RECONCILIATION
    # =========================================================

    reconciliation = PlanningReconciliation(
        budget=budget,
        actuals=cleaned_actuals
    )

    reconciliation.run()

    matched_actuals = (
        reconciliation.matched_actuals()
    )

    # =========================================================
    # 4. VARIANCE ENGINE
    # =========================================================

    variance_engine = VarianceEngine(
        budget=budget,
        actuals=matched_actuals
    )

    variance = variance_engine.run()

    # =========================================================
    # 5. VARIANCE INTERPRETATION
    # =========================================================

    interpretation_engine = (
        VarianceInterpretation(
            variance=variance
        )
    )

    interpreted = interpretation_engine.run()

    # =========================================================
    # 6. HEADER
    # =========================================================

    print("\n")
    print("=" * 80)
    print("VARIA VARIANCE INTERPRETATION")
    print("=" * 80)

    # =========================================================
    # 7. SUMMARY
    # =========================================================

    print("\nINTERPRETATION SUMMARY")
    print("-" * 80)

    summary = interpretation_engine.summary()

    for key, value in summary.items():

        print(
            f"{key}: {value:,}"
        )

    # =========================================================
    # 8. TYPE SUMMARY
    # =========================================================

    print("\nVARIANCE TYPE SUMMARY")
    print("-" * 80)

    type_summary = (
        interpretation_engine
        .type_summary()
    )

    print(
        type_summary.to_string(
            index=False
        )
    )

    # =========================================================
    # 9. REVERSALS / CREDITS
    # =========================================================

    print("\nREVERSAL / CREDIT INVESTIGATION QUEUE")
    print("-" * 80)

    reversals = (
        interpretation_engine
        .investigation_queue()
        .sort_values(
            "Variance",
            key=lambda x: x.abs(),
            ascending=False,
        )
    )

    columns = [
        "Month",
        "Year",
        "Business_Unit",
        "Department",
        "Category",
        "Budget_Amount",
        "Actual_Amount",
        "Variance",
        "Variance_Pct",
        "Variance_Type",
        "Business_Interpretation",
        "Requires_Investigation",
    ]

    print(
        reversals[
            columns
        ].to_string(
            index=False
        )
    )

    # =========================================================
    # 10. TOP OVERSPENDS
    # =========================================================

    print("\nTOP OVERSPENDS")
    print("-" * 80)

    overspends = (
        interpretation_engine
        .by_type("Overspend")
        .sort_values(
            "Variance",
            ascending=False,
        )
    )

    print(
        overspends[
            columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    # =========================================================
    # 11. TOP UNDERSPENDS
    # =========================================================

    print("\nTOP UNDERSPENDS")
    print("-" * 80)

    underspends = (
        interpretation_engine
        .by_type("Underspend")
        .sort_values(
            "Variance",
            ascending=True,
        )
    )

    print(
        underspends[
            columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()