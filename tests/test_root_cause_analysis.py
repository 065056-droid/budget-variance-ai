from src.data_loader import load_workbook
from src.data_cleaner import DataCleaner
from src.planning_reconciliation import PlanningReconciliation
from src.variance_engine import VarianceEngine
from src.root_cause_analysis import RootCauseAnalysis


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
    # 5. ROOT CAUSE ANALYSIS
    # =========================================================

    root_cause = RootCauseAnalysis(
        variance=variance
    )

    # =========================================================
    # 6. HEADER
    # =========================================================

    print("\n")
    print("=" * 80)
    print("VARIA ROOT CAUSE & DRIVER DRILL-DOWN")
    print("=" * 80)

    # =========================================================
    # 7. SUMMARY
    # =========================================================

    print("\nDRIVER SUMMARY")
    print("-" * 80)

    summary = root_cause.summary()

    for key, value in summary.items():

        if "VARIANCE" in key:

            print(
                f"{key}: ₹{value:,.2f}"
            )

        else:

            print(
                f"{key}: {value:,}"
            )

    # =========================================================
    # 8. DIGITAL ADVERTISING → DEPARTMENT
    # =========================================================

    print(
        "\nDIGITAL ADVERTISING → DEPARTMENT"
    )
    print("-" * 80)

    category_department = (
        root_cause
        .category_department(
            "Digital Advertising"
        )
    )

    print(
        category_department.to_string(
            index=False
        )
    )

    # =========================================================
    # 9. DIGITAL ADVERTISING → BUSINESS UNIT
    # =========================================================

    print(
        "\nDIGITAL ADVERTISING → BUSINESS UNIT"
    )
    print("-" * 80)

    category_business_unit = (
        root_cause
        .category_business_unit(
            "Digital Advertising"
        )
    )

    print(
        category_business_unit.to_string(
            index=False
        )
    )

    # =========================================================
    # 10. DIGITAL ADVERTISING → MONTH
    # =========================================================

    print(
        "\nDIGITAL ADVERTISING → MONTH"
    )
    print("-" * 80)

    category_month = (
        root_cause
        .category_month(
            "Digital Advertising"
        )
    )

    print(
        category_month.to_string(
            index=False
        )
    )

    # =========================================================
    # 11. RECURRING DRIVERS
    # =========================================================

    print("\nRECURRING DRIVERS")
    print("-" * 80)

    recurring = (
        root_cause
        .recurring_drivers(
            min_occurrences=3
        )
    )

    print(
        recurring
        .head(20)
        .to_string(index=False)
    )

    # =========================================================
    # 12. DRIVER MATRIX
    # =========================================================

    print("\nDEPARTMENT × CATEGORY DRIVER MATRIX")
    print("-" * 80)

    matrix = root_cause.driver_matrix()

    print(
        matrix.to_string()
    )

    # =========================================================
    # 13. TOP DRIVER
    # =========================================================

    print("\nTOP RECURRING DRIVER")
    print("-" * 80)

    top_driver = root_cause.top_driver()

    if top_driver is not None:

        print(
            top_driver.to_string()
        )

    else:

        print("No unfavorable drivers found.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()