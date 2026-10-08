from src.data_loader import load_workbook
from src.data_cleaner import DataCleaner
from src.reconciliation import ReconciliationEngine


def main():

    # Load workbook
    data = load_workbook()

    # Raw actual transactions
    raw_actuals = data["Actuals_Raw"]

    # Clean the data using our deterministic rules
    cleaner = DataCleaner(
        actuals=raw_actuals,
        category_mapping={
            "Digital Ads": "Digital Advertising"
        },
    )

    cleaned_actuals = cleaner.run()

    # Reconcile raw vs cleaned financial data
    reconciliation = ReconciliationEngine(
        raw_data=raw_actuals,
        cleaned_data=cleaned_actuals,
    )

    report = reconciliation.generate_report()

    # Display report
    print("\n")
    print("=" * 70)
    print("VARIA FINANCIAL RECONCILIATION REPORT")
    print("=" * 70)

    print(
        f"Raw transactions:       "
        f"{report['Raw_Transaction_Count']:,}"
    )

    print(
        f"Cleaned transactions:   "
        f"{report['Cleaned_Transaction_Count']:,}"
    )

    print(
        f"Raw actual total:       "
        f"₹{report['Raw_Actual_Total']:,.2f}"
    )

    print(
        f"Cleaned actual total:   "
        f"₹{report['Cleaned_Actual_Total']:,.2f}"
    )

    print(
        f"Total difference:       "
        f"₹{report['Total_Difference']:,.2f}"
    )

    print(
        f"Amount requiring review:"
        f" ₹{report['Amount_Requiring_Review']:,.2f}"
    )

    print("-" * 70)

    print(
        f"RECONCILIATION STATUS: "
        f"{report['Reconciliation_Status']}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()