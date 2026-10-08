from src.data_loader import load_workbook
from src.data_validator import DataValidator


def main():

    data = load_workbook()

    actuals = data["Actuals_Raw"]

    departments = set(
        data["Departments"]["Department"]
        .dropna()
        .astype(str)
    )

    categories = set(
        data["Categories"]["Category"]
        .dropna()
        .astype(str)
    )

    business_units = set(
        data["Business_Units"]["Business_Unit"]
        .dropna()
        .astype(str)
    )

    regions = set(
        data["Regions"]["Region"]
        .dropna()
        .astype(str)
    )

    validator = DataValidator(
        actuals=actuals,
        valid_departments=departments,
        valid_categories=categories,
        valid_business_units=business_units,
        valid_regions=regions,
    )

    results = validator.run_all_checks()

    print("\n")
    print("=" * 70)
    print("VARIA DATA QUALITY REPORT")
    print("=" * 70)

    for result in results:

        status_icon = {
            "PASS": "OK",
            "WARNING": "!!",
            "FAIL": "XX",
        }.get(result.status, "??")

        print(
            f"[{status_icon}] "
            f"{result.check_name:<30} "
            f"Issues: {result.issue_count}"
        )

        print(
            f"     {result.message}"
        )

    print("-" * 70)

    print(
        f"OVERALL DATA STATUS: "
        f"{validator.overall_status()}"
    )

    print("=" * 70)

    print("\nDetailed results:")
    print(validator.results_dataframe().to_string(index=False))


if __name__ == "__main__":
    main()