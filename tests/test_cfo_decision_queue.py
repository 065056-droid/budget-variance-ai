import pandas as pd

from src.data_loader import load_workbook
from src.data_cleaner import DataCleaner
from src.variance_engine import VarianceEngine
from src.materiality_engine import MaterialityEngine
from src.forecast_engine import ForecastEngine
from src.management_action_engine import ManagementActionEngine
from src.cfo_decision_queue import CFODecisionQueue


def main():

    print("\n")
    print("=" * 80)
    print("VARIA CFO DECISION QUEUE")
    print("=" * 80)

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    data = load_workbook()

    budget = data["Budget"]
    actuals = data["Actuals_Raw"]

    # ---------------------------------------------------------
    # CLEAN
    # ---------------------------------------------------------

    cleaner = DataCleaner(
        actuals=actuals,
        category_mapping={
            "Digital Ads": "Digital Advertising"
        },
    )

    cleaned_actuals = cleaner.cleaned

    # ---------------------------------------------------------
    # VARIANCE
    # ---------------------------------------------------------

    variance_engine = VarianceEngine(
        budget=budget,
        actuals=cleaned_actuals,
    )

    variance_data = variance_engine.run()

    # ---------------------------------------------------------
    # MATERIALITY
    # ---------------------------------------------------------

    materiality_engine = MaterialityEngine(
        variance=variance_data
    )

    materiality_data = (
        materiality_engine.run()
    )

    # ---------------------------------------------------------
    # FORECAST
    # ---------------------------------------------------------

    forecast_engine = ForecastEngine(
        variance_data=variance_data
    )

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
    # MANAGEMENT ACTION
    # ---------------------------------------------------------

    management_engine = ManagementActionEngine(
        variance_data=variance_data,
        materiality_data=materiality_data,
        root_cause_data=pd.DataFrame(),
        forecast_data=forecast_data,
        warning_data=warning_data,
    )

    management_data = (
        management_engine
        .prepare_data()
    )

    # ---------------------------------------------------------
    # CFO DECISION QUEUE
    # ---------------------------------------------------------

    decision_engine = CFODecisionQueue(
        variance_data=variance_data,
        materiality_data=materiality_data,
        management_data=management_data,
    )

    decision_queue = (
        decision_engine.build()
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print("\nCFO DECISION SUMMARY")
    print("-" * 80)

    summary = (
        decision_engine.summary()
    )

    for key, value in summary.items():
        print(f"{key}: {value}")

    # ---------------------------------------------------------
    # TOP DECISIONS
    # ---------------------------------------------------------

    print("\nTOP CFO DECISIONS")
    print("-" * 80)

    top_decisions = (
        decision_engine
        .top_decisions(15)
    )

    print(
    top_decisions[
        [
            "Business_Unit",
            "Department",
            "Category",
            "Current_Variance",
            "Projected_Variance",
            "Projected_Variance_Pct",
            "Material_Variance_Count",
            "High_Critical_Count",
            "Early_Warning",
            "Decision_Score",
            "Decision_Priority",
            "Management_Action",
        ]
    ]
    .to_string(index=False)
)

    # ---------------------------------------------------------
    # HIGH PRIORITY
    # ---------------------------------------------------------

    print("\nHIGH PRIORITY DECISIONS")
    print("-" * 80)

    high_priority = (
        decision_engine
        .high_priority_queue()
    )

    print(
        high_priority[
            [
                "Business_Unit",
                "Department",
                "Category",
                "Current_Variance",
                "Projected_Variance",
                "Projected_Variance_Pct",
                "Early_Warning",
                "Decision_Score",
                "Decision_Priority",
            ]
        ]
        .to_string(index=False)
    )

    print("\n")
    print("=" * 80)


if __name__ == "__main__":
    main()