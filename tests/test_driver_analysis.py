from src.data_loader import load_workbook
from src.data_cleaner import DataCleaner
from src.planning_reconciliation import PlanningReconciliation
from src.variance_engine import VarianceEngine
from src.driver_analysis import DriverAnalysis


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
    # 5. DRIVER ANALYSIS
    # =========================================================

    driver_engine = DriverAnalysis(
        variance=variance
    )

    # =========================================================
    # 6. HEADER
    # =========================================================

    print("\n")
    print("=" * 80)
    print("VARIA DRIVER & PARETO ANALYSIS")
    print("=" * 80)

    # =========================================================
    # 7. OVERALL SUMMARY
    # =========================================================

    print("\nOVERALL VARIANCE SUMMARY")
    print("-" * 80)

    summary = driver_engine.summary()

    for key, value in summary.items():

        if isinstance(value, float):

            print(
                f"{key}: ₹{value:,.2f}"
            )

        else:

            print(
                f"{key}: {value:,}"
            )

    # =========================================================
    # 8. BUSINESS UNIT
    # =========================================================

    print("\nBUSINESS UNIT DRIVERS")
    print("-" * 80)

    business_units = (
        driver_engine
        .business_unit_analysis()
    )

    print(
        business_units.to_string(
            index=False
        )
    )

    # =========================================================
    # 9. DEPARTMENT
    # =========================================================

    print("\nDEPARTMENT DRIVERS")
    print("-" * 80)

    departments = (
        driver_engine
        .department_analysis()
    )

    print(
        departments.to_string(
            index=False
        )
    )

    # =========================================================
    # 10. CATEGORY
    # =========================================================

    print("\nCATEGORY DRIVERS")
    print("-" * 80)

    categories = (
        driver_engine
        .category_analysis()
    )

    print(
        categories.to_string(
            index=False
        )
    )

    # =========================================================
    # 11. MONTH
    # =========================================================

    print("\nMONTHLY DRIVERS")
    print("-" * 80)

    months = (
        driver_engine
        .month_analysis()
    )

    print(
        months.to_string(
            index=False
        )
    )

    # =========================================================
    # 12. TOP 10 CATEGORY DRIVERS
    # =========================================================

    print("\nTOP 10 CATEGORY DRIVERS")
    print("-" * 80)

    top_categories = (
        driver_engine
        .top_drivers(
            "Category",
            n=10,
        )
    )

    print(
        top_categories.to_string(
            index=False
        )
    )

    # =========================================================
    # 13. PARETO CATEGORY DRIVERS
    # =========================================================

    print("\nPARETO CATEGORY DRIVERS")
    print("-" * 80)

    pareto_categories = (
        driver_engine
        .pareto_drivers(
            "Category"
        )
    )

    print(
        pareto_categories.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()