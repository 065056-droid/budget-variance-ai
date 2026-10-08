from src.data_loader import load_workbook
from src.data_cleaner import DataCleaner
from src.planning_reconciliation import PlanningReconciliation
from src.variance_engine import VarianceEngine


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
    # 3. RECONCILE ACTUALS AGAINST BUDGET
    # =========================================================

    reconciliation = PlanningReconciliation(
        budget=budget,
        actuals=cleaned_actuals
    )

    reconciliation.run()

    matched_actuals = reconciliation.matched_actuals()
    exceptions = reconciliation.exception_queue()

    reconciliation_summary = reconciliation.summary()


    # =========================================================
    # 4. RUN VARIANCE ENGINE
    # =========================================================

    engine = VarianceEngine(
        budget=budget,
        actuals=matched_actuals,
    )

    variance = engine.run()

    validation = engine.validation_report()


    # =========================================================
    # 5. HEADER
    # =========================================================

    print("\n")
    print("=" * 80)
    print("VARIA BUDGET VARIANCE ENGINE")
    print("=" * 80)


    # =========================================================
    # 6. DATA VOLUME
    # =========================================================

    print("\nDATA VOLUME")
    print("-" * 80)

    print(
        f"Budget records:             {len(budget):,}"
    )

    print(
        f"Raw actual transactions:    {len(raw_actuals):,}"
    )

    print(
        f"Cleaned actual transactions:{len(cleaned_actuals):,}"
    )

    print(
        f"Matched actual transactions:{len(matched_actuals):,}"
    )

    print(
        f"Exception transactions:     {len(exceptions):,}"
    )

    print(
        f"Variance records:            {len(variance):,}"
    )


    # =========================================================
    # 7. RECONCILIATION
    # =========================================================

    print("\nPLANNING RECONCILIATION")
    print("-" * 80)

    print(
        f"Total actual amount: "
        f"₹{reconciliation_summary['Total Actual Amount']:,.2f}"
    )

    print(
        f"Matched actual amount: "
        f"₹{reconciliation_summary['Matched Amount']:,.2f}"
    )

    print(
        f"Exception amount: "
        f"₹{reconciliation_summary['Exception Amount']:,.2f}"
    )

    print(
        f"Exception transactions: "
        f"{reconciliation_summary['Exception Transactions']:,}"
    )


    # =========================================================
    # 8. DATA GRAIN VALIDATION
    # =========================================================

    print("\nDATA GRAIN VALIDATION")
    print("-" * 80)

    print(
        f"Budget duplicate keys: "
        f"{validation['Budget_Duplicate_Keys']}"
    )

    print(
        f"Actual duplicate keys after aggregation: "
        f"{validation['Actual_Duplicate_Keys_After_Aggregation']}"
    )

    print(
        f"Matched budget keys: "
        f"{validation['Matched_Budget_Keys']:,}"
    )

    print(
        f"Unmatched budget keys: "
        f"{validation['Unmatched_Budget_Keys']:,}"
    )


    # =========================================================
    # 9. FINANCIAL RESULTS
    # =========================================================

    print("\nFINANCIAL RESULTS")
    print("-" * 80)

    total_budget = variance[
        "Budget_Amount"
    ].sum()

    total_actual = variance[
        "Actual_Amount"
    ].sum()

    total_variance = variance[
        "Variance"
    ].sum()

    print(
        f"Total Budget:   "
        f"₹{total_budget:,.2f}"
    )

    print(
        f"Total Actual:   "
        f"₹{total_actual:,.2f}"
    )

    print(
        f"Total Variance: "
        f"₹{total_variance:,.2f}"
    )

    if total_budget != 0:

        total_variance_pct = (
            total_variance
            / total_budget
        ) * 100

        print(
            f"Variance %:     "
            f"{total_variance_pct:.2f}%"
        )


    # =========================================================
    # 10. VARIANCE CLASSIFICATION
    # =========================================================

    print("\nVARIANCE CLASSIFICATION")
    print("-" * 80)

    print(
        variance[
            "Variance_Direction"
        ].value_counts()
    )


    # =========================================================
    # 11. TOP 10 UNFAVORABLE VARIANCES
    # =========================================================

    print("\nTOP 10 UNFAVORABLE VARIANCES")
    print("-" * 80)

    top_unfavorable = (
        variance[
            variance["Variance"] > 0
        ]
        .sort_values(
            "Variance",
            ascending=False,
        )
        .head(10)
    )

    columns_to_display = [
        "Month",
        "Year",
        "Business_Unit",
        "Department",
        "Category",
        "Budget_Amount",
        "Actual_Amount",
        "Variance",
        "Variance_Pct",
    ]

    print(
        top_unfavorable[
            columns_to_display
        ].to_string(index=False)
    )


    # =========================================================
    # 12. EXCEPTION QUEUE
    # =========================================================

    print("\nEXCEPTION QUEUE")
    print("-" * 80)

    exception_columns = [
        "Transaction_ID",
        "Business_Unit",
        "Department",
        "Category",
        "Actual_Amount",
        "Planning_Exception_Type",
        "Planning_Exception_Detail",
    ]

    print(
        exceptions[
            exception_columns
        ].to_string(index=False)
    )


    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()