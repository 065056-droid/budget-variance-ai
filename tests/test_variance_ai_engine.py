from src.data_loader import load_workbook
from src.data_cleaner import DataCleaner
from src.variance_engine import VarianceEngine
from src.materiality_engine import MaterialityEngine
from src.forecast_engine import ForecastEngine
from src.root_cause_analysis import RootCauseAnalysis
from src.management_action_engine import ManagementActionEngine
from src.cfo_decision_queue import CFODecisionQueue
from src.variance_ai_engine import VarianceAIEngine


def main():

    print("\n" + "=" * 80)
    print("VARIA AI VARIANCE EXPLANATION ENGINE")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------

    data = load_workbook()

    budget = data["Budget"]
    actuals = data["Actuals_Raw"]

    # ---------------------------------------------------------
    # 2. CLEAN ACTUALS
    # ---------------------------------------------------------

    cleaner = DataCleaner(
        actuals=actuals,
        category_mapping={
            "Digital Ads": "Digital Advertising"
        }
    )

    cleaned_actuals = cleaner.cleaned

    # ---------------------------------------------------------
    # 3. VARIANCE ENGINE
    # ---------------------------------------------------------

    variance_engine = VarianceEngine(
        budget=budget,
        actuals=cleaned_actuals,
    )

    variance = variance_engine.run()

    # ---------------------------------------------------------
    # 4. MATERIALITY
    # ---------------------------------------------------------

    materiality_engine = MaterialityEngine(variance)

    materiality = materiality_engine.run()

    # ---------------------------------------------------------
    # 5. ROOT CAUSE ANALYSIS
    # ---------------------------------------------------------

    root_cause_engine = RootCauseAnalysis(variance)

    root_cause = root_cause_engine.recurring_drivers(
        min_occurrences=3
    )

    # ---------------------------------------------------------
    # 6. FORECAST
    # ---------------------------------------------------------

    forecast_engine = ForecastEngine(variance)

    forecast = forecast_engine.ytd_forecast()

    warning = forecast_engine.classify_early_warning(
        forecast
    )

    # ---------------------------------------------------------
    # 7. MANAGEMENT ACTION
    # ---------------------------------------------------------

    management_engine = ManagementActionEngine(
        variance,
        materiality,
        root_cause,
        forecast,
        warning,
    )

    management = management_engine.prepare_data()

    # ---------------------------------------------------------
    # 8. CFO DECISION QUEUE
    # ---------------------------------------------------------

    cfo_engine = CFODecisionQueue(
        variance_data=variance,
        materiality_data=materiality,
        management_data=management,
    )

    cfo_queue = cfo_engine.build()

    # ---------------------------------------------------------
    # 9. AI VARIANCE EXPLANATION
    # ---------------------------------------------------------

    ai_engine = VarianceAIEngine(
        decision_queue=cfo_queue,
        root_cause_data=root_cause,
        forecast_data=forecast,
    )

    ai_output = ai_engine.run()

    # ---------------------------------------------------------
    # 10. SUMMARY
    # ---------------------------------------------------------

    print("\nAI ENGINE SUMMARY")
    print("-" * 80)

    summary = ai_engine.summary()

    for key, value in summary.items():
        print(f"{key}: {value}")

    # ---------------------------------------------------------
    # 11. TOP AI EXPLANATIONS
    # ---------------------------------------------------------

    print("\nTOP AI EXPLANATIONS")
    print("-" * 80)

    top = ai_engine.top_explanations(10)

    columns = [
        "Business_Unit",
        "Department",
        "Category",
        "Projected_Variance",
        "Projected_Variance_Pct",
        "Early_Warning",
        "Decision_Priority",
        "AI_Issue_Type",
        "AI_Explanation",
        "AI_Recommended_Action",
    ]

    print(
        top[columns].to_string(index=False)
    )

    # ---------------------------------------------------------
    # 12. EXECUTIVE SUMMARIES
    # ---------------------------------------------------------

    print("\nEXECUTIVE SUMMARIES")
    print("-" * 80)

    for _, row in top.head(5).iterrows():

        print(
            f"\n[{row['Business_Unit']} | "
            f"{row['Department']} | "
            f"{row['Category']}]"
        )

        print(row["AI_Executive_Summary"])


if __name__ == "__main__":
    main()