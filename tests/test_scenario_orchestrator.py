import pandas as pd

from src.scenario_orchestrator import ScenarioOrchestrator, ScenarioRunConfig


def build_forecast():
    return pd.DataFrame(
        [
            {
                "Business_Unit": "A",
                "Department": "M",
                "Category": "Ads",
                "YTD_Actual": 900000,
                "Forecast_Remaining_Spend": 400000,
                "Forecast_At_Completion": 1300000,
                "Estimated_Full_Year_Budget": 1100000,
                "Projected_Variance": 200000,
                "Projected_Variance_Pct": 18.1818,
            },
            {
                "Business_Unit": "B",
                "Department": "IT",
                "Category": "Cloud",
                "YTD_Actual": 700000,
                "Forecast_Remaining_Spend": 300000,
                "Forecast_At_Completion": 1000000,
                "Estimated_Full_Year_Budget": 1050000,
                "Projected_Variance": -50000,
                "Projected_Variance_Pct": -4.7619,
            },
            {
                "Business_Unit": "A",
                "Department": "Ops",
                "Category": "Raw Materials",
                "YTD_Actual": 800000,
                "Forecast_Remaining_Spend": 500000,
                "Forecast_At_Completion": 1300000,
                "Estimated_Full_Year_Budget": 1250000,
                "Projected_Variance": 50000,
                "Projected_Variance_Pct": 4.0,
            },
        ]
    )


def main():
    print("=" * 80)
    print("VARIA SCENARIO ORCHESTRATOR TEST")
    print("=" * 80)

    forecast = build_forecast()
    orchestrator = ScenarioOrchestrator(forecast)

    scenarios = [
        ScenarioRunConfig("Base", 0),
        ScenarioRunConfig(
            "10% Ads Spend Reduction",
            future_spend_change_pct=-10,
            scope={"Category": "Ads"},
        ),
        {
            "name": "20% Ads Spend Reduction",
            "future_spend_change_pct": -20,
            "scope": {"Category": "Ads"},
        },
    ]

    comparison, best, ranking = orchestrator.run_scenario_library(scenarios)
    assert len(comparison) == 3
    assert ranking.iloc[0]["Outcome_Rank"] == 1
    assert best["Variance_Improvement"] > 0

    sensitivity = orchestrator.run_sensitivity(
        dimension="Category",
        spend_reduction_pct=10,
        top_n=3,
    )
    assert len(sensitivity) == 3
    assert "Impact_Rank" in sensitivity.columns

    reallocation, realloc_summary = orchestrator.run_reallocation(
        source_scope={"Category": "Cloud"},
        target_scope={"Category": "Ads"},
        amount=100000,
        scenario_name="Move Funding",
    )
    assert abs(reallocation["Reallocation_Net"].sum()) < 1e-8
    assert abs(realloc_summary["Total_Budget_Change"]) < 1e-8
    assert realloc_summary["Source_Budget_Change"] < 0
    assert realloc_summary["Target_Budget_Change"] > 0

    action_data, action_summary = orchestrator.action_sizing()
    assert len(action_data) == len(forecast)
    assert action_summary["Total_Cost_Control_Required"] == 250000.0

    pack = orchestrator.decision_pack(
        scenarios,
        sensitivity_dimension="Category",
        sensitivity_reduction_pct=10,
        sensitivity_top_n=3,
    )
    required = {
        "base",
        "scenario_comparison",
        "scenario_ranking",
        "recommended_scenario",
        "sensitivity",
        "action_sizing",
        "action_summary",
    }
    assert required.issubset(pack.keys())

    single = orchestrator.compare_scenario_to_base(
        {
            "name": "Targeted Control",
            "future_spend_change_pct": -10,
            "scope": {"Category": "Ads"},
        }
    )
    assert single["Variance_Improvement"] > 0
    assert single["Scenario_Status"] == "Improved"

    print("Scenario library: PASS")
    print("Sensitivity ranking: PASS")
    print("Budget reallocation: PASS")
    print("Action sizing: PASS")
    print("Decision pack: PASS")
    print("Scenario vs base comparison: PASS")
    print("\n" + "=" * 80)
    print("VARIA SCENARIO ORCHESTRATOR TEST: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
