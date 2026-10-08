from src.data_loader import load_workbook
from src.data_cleaner import DataCleaner
from src.variance_engine import VarianceEngine
from src.forecast_engine import ForecastEngine
from src.management_action_engine import ManagementActionEngine


def main():

    print("\n")
    print("=" * 80)
    print("VARIA MANAGEMENT ACTION ENGINE")
    print("=" * 80)

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    data = load_workbook()

    budget = data["Budget"]
    actuals = data["Actuals_Raw"]

    # ---------------------------------------------------------
    # CLEAN ACTUALS
    # ---------------------------------------------------------

    cleaner = DataCleaner(
        actuals=actuals,
        category_mapping={
            "Digital Ads": "Digital Advertising"
        },
    )

    cleaned_actuals = cleaner.cleaned

    # ---------------------------------------------------------
    # VARIANCE ENGINE
    # ---------------------------------------------------------

    variance_engine = VarianceEngine(
        budget=budget,
        actuals=cleaned_actuals,
    )

    variance_data = variance_engine.run()

    # ---------------------------------------------------------
    # FORECAST ENGINE
    # ---------------------------------------------------------

    forecast_engine = ForecastEngine(
        variance_data=variance_data
    )

    # ---------------------------------------------------------
    # FORECAST
    # ---------------------------------------------------------

    forecast_data = (
        forecast_engine
        .ytd_forecast()
    )

    # ---------------------------------------------------------
    # EARLY WARNING
    # ---------------------------------------------------------

    warning_data = (
        forecast_engine
        .classify_early_warning(
            forecast_data
        )
    )

    # ---------------------------------------------------------
    # MANAGEMENT ACTION ENGINE
    # ---------------------------------------------------------

    management_engine = ManagementActionEngine(
        variance_data=variance_data,
        materiality_data=variance_data,
        root_cause_data=variance_data,
        forecast_data=forecast_data,
        warning_data=warning_data,
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print("\nMANAGEMENT SUMMARY")
    print("-" * 80)

    summary = management_engine.summary()

    for key, value in summary.items():
        print(f"{key}: {value}")

    # ---------------------------------------------------------
    # IMMEDIATE ATTENTION
    # ---------------------------------------------------------

    print("\nIMMEDIATE ATTENTION")
    print("-" * 80)

    immediate = (
        management_engine
        .priority_queue(
            "Immediate Attention"
        )
    )

    print(
        immediate[
            [
                "Business_Unit",
                "Department",
                "Category",
                "Variance_Type",
                "Projected_Variance",
                "Projected_Variance_Pct",
                "Early_Warning",
                "Management_Priority",
                "Management_Action",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # INVESTIGATION REQUIRED
    # ---------------------------------------------------------

    print("\nINVESTIGATION REQUIRED")
    print("-" * 80)

    investigation = (
        management_engine
        .priority_queue(
            "Investigation Required"
        )
    )

    if len(investigation) == 0:
        print("No investigation records found.")
    else:
        print(
            investigation[
                [
                    "Business_Unit",
                    "Department",
                    "Category",
                    "Actual_Amount",
                    "Variance",
                    "Variance_Type",
                    "Management_Priority",
                    "Management_Action",
                ]
            ]
            .to_string(index=False)
        )

    print("\n")
    print("=" * 80)


if __name__ == "__main__":
    main()