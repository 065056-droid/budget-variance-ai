from src.data_loader import load_workbook
from src.data_cleaner import DataCleaner


def main():

    data = load_workbook()

    actuals = data["Actuals_Raw"]

    category_mapping = {
        "Digital Ads": "Digital Advertising"
    }

    cleaner = DataCleaner(
        actuals=actuals,
        category_mapping=category_mapping,
    )

    cleaned = cleaner.run()

    print("\n")
    print("=" * 70)
    print("VARIA DATA CLEANING REPORT")
    print("=" * 70)

    print(
        f"Original records: {len(actuals):,}"
    )

    print(
        f"Processed records: {len(cleaned):,}"
    )

    print(
        f"Records requiring review: "
        f"{cleaned['Requires_Review'].sum():,}"
    )

    print("-" * 70)

    print("Cleaning actions:")

    log = cleaner.cleaning_log_dataframe()

    if len(log) > 0:
        print(log.to_string(index=False))
    else:
        print("No cleaning actions required.")

    print("-" * 70)

    print("Category values after standardization:")

    print(
        sorted(
            cleaned["Category"]
            .dropna()
            .unique()
            .tolist()
        )
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
    