from src.data_loader import load_workbook
from src.data_cleaner import DataCleaner
from src.variance_engine import VarianceEngine
from src.forecast_engine import ForecastEngine


def main():

    print("\n")
    print("=" * 80)
    print("VARIA FORECAST ENGINE")
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
    # BUILD VARIANCE DATA
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
    # FISCAL YEAR SUMMARY
    # ---------------------------------------------------------

    print("\nFISCAL YEAR SUMMARY")
    print("-" * 80)

    fiscal_summary = (
        forecast_engine
        .fiscal_year_summary()
    )

    print(
        fiscal_summary.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # ANNUAL SUMMARY
    # ---------------------------------------------------------

    print("\nANNUAL SUMMARY")
    print("-" * 80)

    annual_summary = (
        forecast_engine
        .annual_summary()
    )

    print(
        annual_summary.head(10).to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # ENGINE SUMMARY
    # ---------------------------------------------------------

    print("\nENGINE SUMMARY")
    print("-" * 80)

    summary = forecast_engine.summary()

    for key, value in summary.items():
        print(f"{key}: {value}")

    # ---------------------------------------------------------
    # YTD FORECAST
    # ---------------------------------------------------------

    print("\nYTD FORECAST")
    print("-" * 80)

    ytd_forecast = (
        forecast_engine
        .ytd_forecast()
    )

    print(
        ytd_forecast.head(15).to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # EARLY WARNING
    # ---------------------------------------------------------

    print("\nEARLY WARNING")
    print("-" * 80)

    warning_data = (
        forecast_engine
        .classify_early_warning(
            ytd_forecast
        )
    )

    warning_summary = (
        forecast_engine
        .early_warning_summary(
            warning_data
        )
    )

    for key, value in warning_summary.items():
        print(f"{key}: {value}")

    # ---------------------------------------------------------
    # TOP RED WARNINGS
    # ---------------------------------------------------------

    print("\nTOP RED WARNINGS")
    print("-" * 80)

    red_warnings = (
        warning_data[
            warning_data["Early_Warning"] == "RED"
        ]
        .sort_values(
            "Projected_Variance",
            key=lambda x: x.abs(),
            ascending=False,
        )
    )

    print(
        red_warnings[
            [
                "Business_Unit",
                "Department",
                "Category",
                "YTD_Actual",
                "Estimated_Full_Year_Budget",
                "Forecast_At_Completion",
                "Projected_Variance",
                "Projected_Variance_Pct",
                "Early_Warning",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print("\n")
    print("=" * 80)


if __name__ == "__main__":
    main()