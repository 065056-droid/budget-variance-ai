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
    # YTD FORECAST
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
        materiality_data=pd.DataFrame(),
        root_cause_data=pd.DataFrame(),
        forecast_data=forecast_data,
        warning_data=warning_data,
    )

    # ---------------------------------------------------------
    # MANAGEMENT SUMMARY
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
    # PROJECTED OVERSPEND
    # ---------------------------------------------------------

    print("\nTOP PROJECTED OVERSPEND")
    print("-" * 80)

    overspend = (
        management_engine
        .overspend_queue()
    )

    print(
        overspend[
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
                "Management_Action",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # PROJECTED UNDERSPEND
    # ---------------------------------------------------------

    print("\nTOP PROJECTED UNDERSPEND")
    print("-" * 80)

    underspend = (
        management_engine
        .underspend_queue()
    )

    print(
        underspend[
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
                "Management_Action",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\n")
    print("=" * 80)


if __name__ == "__main__":
    main()